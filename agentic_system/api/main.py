"""
main.py
=======
FastAPI REST API for the Agentic Decision System.

Endpoints:
    POST /analyze          — Run the full pipeline on a text input
    POST /analyze/file     — Run the pipeline on an uploaded file
    GET  /health           — Health check
    GET  /kb/stats         — Knowledge base statistics

The Pipeline is initialized once at startup and reused across requests.

Install: pip install fastapi uvicorn python-multipart

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Author: Mohammed Mudassirullah Sheriff
Project: Intelligent Optimization of Data Pipelines using RAG-Based
         Knowledge Retrieval and Agentic Decision Systems
Institution: BITS Pilani WILP — M.Tech Data Science & Engineering
"""

import logging
import sys
import os
import tempfile
from pathlib import Path
from typing import Optional
import numpy as np
# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Form
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError(
        "FastAPI not installed. Run: pip install fastapi uvicorn python-multipart"
    )

from core.pipeline import Pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Pipeline Optimization API",
    description=(
        "Intelligent Data Pipeline Optimization using RAG-Based Knowledge "
        "Retrieval and Agentic Decision Systems. "
        "Dissertation project — BITS Pilani WILP M.Tech Data Science & Engineering."
    ),
    version="1.0.0",
)

# Allow CORS for local demo use
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance — loaded once at startup
_pipeline: Optional[Pipeline] = None


@app.on_event("startup")
async def startup_event():
    """Load the pipeline on API startup."""
    global _pipeline
    logger.info("API startup — initializing pipeline...")
    _pipeline = Pipeline()
    _pipeline.load()
    logger.info("Pipeline ready.")


def get_pipeline() -> Pipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")
    return _pipeline


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """Request body for the /analyze endpoint."""
    input_text: str = Field(
        ...,
        description="Raw input to analyze — SQL query, ETL YAML, pipeline log, or metadata JSON.",
        example="SELECT * FROM orders o WHERE o.amount > (SELECT AVG(amount) FROM orders WHERE customer_id = o.customer_id);"
    )
    input_type_hint: Optional[str] = Field(
        None,
        description="Optional hint for input type detection. One of: SQL Query | ETL Workflow | Pipeline Log | Metadata",
        example="SQL Query"
    )
    include_trace: bool = Field(
        True,
        description="Include full agent trace in response (useful for debugging and dissertation evidence)."
    )


class RecommendationItem(BaseModel):
    rank:             int
    entry_id:         str
    title:            str
    severity:         str
    problem:          str
    solution:         str
    before_example:   str
    after_example:    str
    explanation:      str
    expected_impact:  dict
    source:           str
    confidence:       float


class AnalyzeResponse(BaseModel):
    """Response body for the /analyze endpoint."""
    report_summary:         str
    total_issues_detected:  int
    recommendations:        list
    additional_notes:       str
    pipeline_metadata:      dict
    agent_trace:            Optional[dict] = None


def to_python_types(obj):
    """
    Recursively convert NumPy scalar types into native Python types
    so FastAPI/Pydantic can serialize them.
    """
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [to_python_types(v) for v in obj]

    elif isinstance(obj, tuple):
        return tuple(to_python_types(v) for v in obj)

    elif isinstance(obj, np.integer):
        return int(obj)

    elif isinstance(obj, np.floating):
        return float(obj)

    elif isinstance(obj, np.bool_):
        return bool(obj)

    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint. Confirms API and pipeline are running."""
    pipeline = get_pipeline()
    return {
        "status":   "healthy",
        "pipeline": "loaded" if pipeline._loaded else "not loaded",
        "version":  "1.0.0",
    }

@app.get("/kb/catalog", tags=["Knowledge Base"])
async def kb_catalog():

    pipeline = get_pipeline()
    kb = pipeline.knowledge_base

    if kb is None:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base not loaded."
        )

    entries = []

    for meta in kb.metadata.values():

        if not isinstance(meta, dict):
            continue

        entries.append({
            "id": meta.get("entry_id"),
            "title": meta.get("title"),
            "category": meta.get("category"),
            "subcategory": meta.get("subcategory"),
            "input_type": meta.get("input_type"),
            "severity": meta.get("severity"),
            "platforms": meta.get("platform", []),
            "summary": meta.get("problem_description"),
            "optimization": meta.get("optimization_strategy"),
            "tags": meta.get("tags", []),
            "confidence": meta.get("confidence_score"),
            "layer": meta.get("layer"),
            "version": meta.get("version")
        })

    entries.sort(key=lambda x: (x["category"] or "", x["title"] or ""))

    return {
        "total_entries": len(entries),
        "entries": entries
    }

@app.get("/kb/stats", tags=["Knowledge Base"])
async def kb_stats():
    """Return statistics about the loaded knowledge base."""
    pipeline = get_pipeline()
    kb       = pipeline.knowledge_base

    if kb is None:
        raise HTTPException(status_code=503, detail="Knowledge base not loaded.")

    total = kb.index.ntotal if kb.index else 0

    # Count by layer
    l1_count = sum(
        1 for v in kb.metadata.values()
        if isinstance(v, dict) and v.get("layer") == 1
    )
    l2_count = total - l1_count

    # Count by category
    category_counts = {}
    for v in kb.metadata.values():
        if isinstance(v, dict):
            cat = v.get("category", "Unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

    return {
        "total_vectors":    total,
        "layer1_entries":   l1_count,
        "layer2_chunks":    l2_count,
        "embedding_dim":    kb.index.d if kb.index else 0,
        "embedding_model":  kb.model_name,
        "category_breakdown": category_counts,
    }


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_text(request: AnalyzeRequest):
    """
    Run the full four-agent pipeline on a text input.

    Accepts SQL queries, ETL workflow definitions, pipeline execution logs,
    or schema/metadata JSON. Returns ranked optimization recommendations
    with explanations and expected impact estimates.
    """
    pipeline = get_pipeline()

    if not request.input_text.strip():
        raise HTTPException(
            status_code=400,
            detail="input_text is empty. Provide a SQL query, ETL YAML, log, or metadata."
        )

    try:
        result = pipeline.run(raw_input=request.input_text)
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution failed: {str(e)}"
        )
    logger.info(f"[Pipeline result]: {result}")

    payload = {
    "report_summary": result.report_summary,
    "total_issues_detected": result.total_issues_detected,
    "recommendations": result.recommendations,
    "additional_notes": result.additional_notes,
    "pipeline_metadata": result.pipeline_metadata,
    "agent_trace": result.agent_trace if request.include_trace else None,
}

    return AnalyzeResponse(**to_python_types(payload))
    # return AnalyzeResponse(
    #     report_summary=result.report_summary,
    #     total_issues_detected=result.total_issues_detected,
    #     recommendations=result.recommendations,
    #     additional_notes=result.additional_notes,
    #     pipeline_metadata=result.pipeline_metadata,
    #     agent_trace=result.agent_trace if request.include_trace else None,
    # )

@app.post("/analyze_code", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_code(input_text: str = Form(..., description="Paste your code here"),
                       input_type_hint: Optional[str] = Form(None),
                       include_trace: bool = Form(True),):
    """
    Run the full four-agent pipeline on a text input.

    Accepts SQL queries, ETL workflow definitions, pipeline execution logs,
    or schema/metadata JSON. Returns ranked optimization recommendations
    with explanations and expected impact estimates.
    """
    pipeline = get_pipeline()

    if not input_text.strip():
        raise HTTPException(
            status_code=400,
            detail="input_text is empty. Provide a SQL query, ETL workflow, log, or metadata."
        )
    try:
        result = pipeline.run(raw_input=input_text)
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution failed: {str(e)}"
        )
    logger.info(f"[Pipeline result]: {result}")

    payload = {
    "report_summary": result.report_summary,
    "total_issues_detected": result.total_issues_detected,
    "recommendations": result.recommendations,
    "additional_notes": result.additional_notes,
    "pipeline_metadata": result.pipeline_metadata,
    "agent_trace": result.agent_trace if include_trace else None,
}

    return AnalyzeResponse(**to_python_types(payload))

@app.post("/analyze/file", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_file(
    file: UploadFile = File(..., description="Upload a .sql, .yaml, .log, or .json file"),
    include_trace: bool = Query(True, description="Include agent trace in response"),
):
    """
    Run the full pipeline on an uploaded file.

    Supported formats: .sql, .yaml, .yml, .log, .txt, .json
    """
    pipeline = get_pipeline()

    # Validate file extension
    allowed_extensions = {".sql", ".yaml", ".yml", ".log", ".txt", ".json"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed_extensions}"
        )

    # Read file content
    content = await file.read()
    try:
        raw_text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="File encoding must be UTF-8."
        )

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = pipeline.run(raw_input=raw_text, file_path=file.filename)
    except Exception as e:
        logger.error(f"Pipeline error on file {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution failed: {str(e)}"
        )

    return AnalyzeResponse(
        report_summary=result.report_summary,
        total_issues_detected=result.total_issues_detected,
        recommendations=result.recommendations,
        additional_notes=result.additional_notes,
        pipeline_metadata=result.pipeline_metadata,
        agent_trace=result.agent_trace if include_trace else None,
    )
