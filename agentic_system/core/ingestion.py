"""
ingestion.py
============
Input ingestion and preprocessing module for Stage 2.1 of the pipeline.

Responsibilities:
    - Accept raw input from CLI or FastAPI
    - Detect input type (SQL Query / ETL Workflow / Pipeline Log / Metadata)
    - Normalize and clean the input
    - Extract structural features for the Analyzer Agent

Supported input types (matching KB taxonomy):
    - SQL Query     : .sql files or raw SQL strings
    - ETL Workflow  : .yaml / .yml / .json DAG definitions
    - Pipeline Log  : .log / .txt execution logs
    - Metadata      : .json schema/stats/lineage files

Author: Mohammed Mudassirullah Sheriff
Project: Intelligent Optimization of Data Pipelines using RAG-Based
         Knowledge Retrieval and Agentic Decision Systems
Institution: BITS Pilani WILP — M.Tech Data Science & Engineering
"""

import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

REGISTRY_PATH = "../agentic_system/table_registry/"
# ---------------------------------------------------------------------------
# Data model for preprocessed input
# ---------------------------------------------------------------------------

@dataclass
class PipelineInput:
    """
    Standardized input object passed to all downstream agents.
    Created by the Ingestion module; consumed by the Analyzer Agent.
    """
    raw_text:       str                        # Original input as-is
    input_type:     str                        # SQL Query | ETL Workflow | Pipeline Log | Metadata
    normalized_text: str                       # Cleaned version used for embedding + analysis
    source_file:    Optional[str] = None       # File path if loaded from disk
    features:       dict          = field(default_factory=dict)  # Extracted structural features
    char_count:     int           = 0
    line_count:     int           = 0

    def __post_init__(self):
        self.char_count = len(self.normalized_text)
        self.line_count = self.normalized_text.count("\n") + 1


# ---------------------------------------------------------------------------
# Input type detection
# ---------------------------------------------------------------------------

# SQL keywords that strongly indicate a SQL query
SQL_KEYWORDS = {
    "select", "insert", "update", "delete", "create", "drop", "alter",
    "with", "from", "where", "join", "group by", "order by", "having",
    "union", "intersect", "except", "explain", "analyze",
}

# ETL/DAG markers — common in Airflow, dbt, and pipeline YAML files
ETL_KEYWORDS = {
    "dag", "task", "operator", "pipeline", "workflow", "airflow",
    "spark", "glue", "dbt", "dependencies", "schedule_interval",
    "default_args", "catchup", "retries", "on_failure_callback",
    "transform", "extract", "load", "etl",
}

# Log markers — timestamps, log levels, execution markers
LOG_PATTERNS = [
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}",   # Timestamp
    r"\b(INFO|DEBUG|WARNING|ERROR|CRITICAL|WARN)\b", # Log level
    r"\b(Task|Stage|Job|Executor|Attempt)\s+\w+",    # Execution unit references
    r"(succeeded|failed|running|queued|skipped)",    # Task state words
    r"Duration:\s*\d+",                              # Duration metrics
]

# Metadata markers — JSON schema or stats objects
METADATA_MARKERS = {
    "schema", "columns", "statistics", "row_count", "null_count",
    "data_type", "lineage", "partitions", "table_name", "database",
    "num_rows", "num_files", "avg_file_size",
}


def detect_input_type(text: str, file_extension: str = "") -> str:
    """
    Detect the input type from text content and optional file extension.

    Detection priority:
        1. File extension (most reliable)
        2. Content pattern matching
        3. Keyword frequency scoring

    Returns:
        One of: "SQL Query" | "ETL Workflow" | "Pipeline Log" | "Metadata"
    """
    ext = file_extension.lower().lstrip(".")

    # Extension-based detection (highest confidence)
    if ext in ("sql",):
        return "SQL Query"
    if ext in ("yaml", "yml"):
        return "ETL Workflow"
    if ext in ("log",):
        return "Pipeline Log"
    if ext in ("json",) and _looks_like_metadata(text):
        return "Metadata"

    # Content-based detection
    text_lower = text.lower()

    # Check for log patterns first — timestamps are very distinctive
    log_hits = sum(
        1 for pattern in LOG_PATTERNS
        if re.search(pattern, text, re.IGNORECASE)
    )
    if log_hits >= 2:
        return "Pipeline Log"

    # Score SQL vs ETL vs Metadata by keyword frequency
    sql_score  = sum(1 for kw in SQL_KEYWORDS  if re.search(r'\b' + kw + r'\b', text_lower))
    etl_score  = sum(1 for kw in ETL_KEYWORDS  if kw in text_lower)
    meta_score = sum(1 for kw in METADATA_MARKERS if kw in text_lower)

    scores = {
        "SQL Query":     sql_score,
        "ETL Workflow":  etl_score,
        "Metadata":      meta_score,
        "Pipeline Log":  log_hits,
    }

    detected = max(scores, key=scores.get)

    # Fallback: if all scores are 0 or tied, default to SQL Query
    if scores[detected] == 0:
        logger.warning(
            "Could not confidently detect input type. Defaulting to 'SQL Query'."
        )
        return "SQL Query"

    logger.info(f"Input type detected: {detected} (scores: {scores})")
    return detected


def _looks_like_metadata(text: str) -> bool:
    """Check if JSON text looks like schema/stats metadata rather than a config."""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            keys = {k.lower() for k in obj.keys()}
            return bool(keys & METADATA_MARKERS)
    except (json.JSONDecodeError, TypeError):
        pass
    return False

# --------------------------------------------------------------------------
# Load the table_registry 
# --------------------------------------------------------------------------
def load_table_registry() -> dict:
    """
    Loads the table registry at startup.
    Returns a dict keyed by table_name
    Returns Empty dict if refistry file does not exist
    """
    path = Path(REGISTRY_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    registry = {}

    for file in path.glob("*.json"):

        with file.open("r", encoding="utf-8") as f:
            tables = json.load(f)

        # Allow either a single object or a list of objects
        if isinstance(tables, dict):
            tables = [tables]

        if not isinstance(tables, list):
            raise ValueError(f"{file} must contain an object or an array of objects.")

        for table in tables:
            table_name = table.get("table_name")
            if not table_name:
                continue
            key = table_name.upper()
            if key in registry:
                print(f"Warning: Duplicate table '{table_name}' found in {file.name}. Overwriting previous definition.")
            registry[key] = table

    return registry

def extract_table_names(query_text: str) -> list:
    pattern = re.compile(
        r"""
        (?:
            FROM
            |JOIN
            |UPDATE
            |INTO
        )
        \s+
        ([A-Za-z0-9_."]+)
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    tables = pattern.findall(query_text)
    logger.info(f"Extracted tables from query text: {tables}")
    # Remove quotes and duplicates while preserving order
    seen = set()
    result = []

    for table in tables:
        table = table.strip('"')
        if table not in seen:
            seen.add(table)
            result.append(table)
    return result

def enrich_with_registry(query_text: str, registry: dict) -> str:
    """
    Checks if nay table in the query exists in the registry
    If matches found, prepends the metadata context to querytext
    if not matches , returns the query text unchanged
    """
    table_names = extract_table_names(query_text)

    matched_tables = [
        registry[name.upper()] for name in table_names if name.upper() in registry
    ]
    if not matched_tables:
        return query_text

    context_lines = ["-- TABLE METADATA CONTEXT (auto-enriched)"]
    context_lines.append("--" + "="*50)

    for table in matched_tables:
        if table.get("description"):
            context_lines.append(
                f"-- Description        : {table['description']}"
            )

        if table.get("table_type"):
            context_lines.append(
                f"-- Table Type         : {table['table_type']}"
            )
        if table.get("primary_keys"):
            context_lines.append(
                "-- Primary Keys      : "
                + ", ".join(table["primary_keys"])
            )

        if table.get("partition_keys"):
            context_lines.append(
                "-- Partition Keys    : "
                + ", ".join(table["partition_keys"])
            )

        if table.get("clustering_keys"):
            context_lines.append(
                "-- Clustering Keys   : "
                + ", ".join(table["clustering_keys"])
            )

        if table.get("estimated_row_count"):
            context_lines.append(
                f"-- Estimated Rows    : {table['estimated_row_count']:,}"
            )

        if table.get("refresh_frequency"):
            context_lines.append(
                f"-- Refresh Frequency : {table['refresh_frequency']}"
            )

        context_lines.append("--")

    context_lines.append("--" + "="*50)
    contenxt_block = "\n".join(context_lines)

    return contenxt_block + "\n\n" + query_text

# ---------------------------------------------------------------------------
# Feature extraction per input type
# ---------------------------------------------------------------------------

def extract_sql_features(text: str) -> dict:
    """Extract structural features from a SQL query."""
    text_upper = text.upper()
    return {
        "has_select_star":      bool(re.search(r"SELECT\s+\*", text_upper)),
        "has_join":             "JOIN" in text_upper,
        "has_subquery":         text_upper.count("SELECT") > 1,
        "has_where":            "WHERE" in text_upper,
        "has_group_by":         "GROUP BY" in text_upper,
        "has_order_by":         "ORDER BY" in text_upper,
        "has_having":           "HAVING" in text_upper,
        "has_union":            "UNION" in text_upper,
        "has_distinct":         "DISTINCT" in text_upper,
        "has_correlated_subquery": bool(
            re.search(r"SELECT.*SELECT.*WHERE.*\.", text_upper, re.DOTALL)
        ),
        "join_count":           len(re.findall(r'\bJOIN\b', text_upper)),
        "subquery_depth":       text.count("(SELECT"),
        "estimated_complexity": _sql_complexity(text_upper),
    }


def extract_etl_features(text: str) -> dict:
    """Extract structural features from an ETL workflow definition."""
    text_lower = text.lower()
    return {
        "is_airflow_dag":       "dag" in text_lower and "operator" in text_lower,
        "is_yaml":              text.strip().startswith("---") or ":" in text[:100],
        "task_count":           len(re.findall(r"task_id|task:", text_lower)),
        "has_dependencies":     "depends_on" in text_lower or ">>" in text,
        "has_retries":          "retries" in text_lower,
        "has_full_reload":      any(kw in text_lower for kw in
                                    ["overwrite", "full_load", "truncate"]),
        "has_incremental":      any(kw in text_lower for kw in
                                    ["incremental", "watermark", "updated_at",
                                     "delta", "append"]),
        "has_spark":            "spark" in text_lower or "pyspark" in text_lower,
        "operator_types":       re.findall(r"(\w+Operator)", text),
    }


def extract_log_features(text: str) -> dict:
    """Extract structural features from a pipeline execution log."""
    lines = text.split("\n")
    error_lines   = [l for l in lines if re.search(r"\b(ERROR|FAILED|Exception)\b", l, re.I)]
    warning_lines = [l for l in lines if re.search(r"\b(WARNING|WARN)\b", l, re.I)]

    # Extract task durations if present
    durations = re.findall(r"Duration[:\s]+(\d+(?:\.\d+)?)\s*(s|m|ms|min|sec)?", text, re.I)

    return {
        "total_lines":      len(lines),
        "error_count":      len(error_lines),
        "warning_count":    len(warning_lines),
        "has_errors":       len(error_lines) > 0,
        "error_samples":    [l.strip() for l in error_lines[:3]],
        "duration_values":  durations[:5],
        "has_oom":          bool(re.search(r"OutOfMemory|OOM|GC overhead", text, re.I)),
        "has_timeout":      bool(re.search(r"timeout|timed out", text, re.I)),
        "has_skew":         bool(re.search(r"skew|uneven|imbalanced", text, re.I)),
        "task_names":       re.findall(r"Task[:\s]+(['\"]?)(\w+)\1", text)[:10],
    }


def extract_metadata_features(text: str) -> dict:
    """Extract structural features from schema/stats metadata."""
    features = {
        "is_valid_json": False,
        "has_schema":    False,
        "has_stats":     False,
        "table_count":   0,
        "column_count":  0,
    }
    try:
        obj = json.loads(text)
        features["is_valid_json"] = True
        flat = json.dumps(obj).lower()
        features["has_schema"]   = any(k in flat for k in ["columns", "schema", "dtype"])
        features["has_stats"]    = any(k in flat for k in ["row_count", "null_count", "statistics"])
        features["table_count"]  = flat.count("table_name")
        features["column_count"] = flat.count("column_name") + flat.count('"name"')
    except (json.JSONDecodeError, TypeError):
        pass
    return features


def _sql_complexity(text_upper: str) -> str:
    """Rough complexity classification for SQL queries."""
    score = 0
    score += text_upper.count("JOIN") * 2
    score += text_upper.count("SELECT") - 1   # Subqueries
    score += 1 if "GROUP BY" in text_upper else 0
    score += 1 if "HAVING"   in text_upper else 0
    score += 1 if "UNION"    in text_upper else 0
    if score <= 1:
        return "Simple"
    if score <= 4:
        return "Moderate"
    return "Complex"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_input(text: str, input_type: str, registry: dict = None) -> str:
    """
    Clean and normalize input text for consistent downstream processing.
    Preserves semantic content while removing noise.
    """
    # Remove null bytes and non-printable characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace
    text = text.strip()

    if input_type == "SQL Query":
        # Normalize SQL whitespace while preserving structure
        text = re.sub(r"[ \t]+", " ", text)
        # Remove inline comments
        text = re.sub(r"--[^\n]*", "", text)
        # Remove block comments
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        text = text.strip()
        if registry:
            logger.info("Enriching the query with table metadata")
            text = enrich_with_registry(text, registry)
        
    return text


# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------

def ingest(raw_input: str, file_path: str = None, registry=None) -> PipelineInput:
    """
    Main entry point for the ingestion module.

    Accepts raw input text (from CLI) or a file path.
    Returns a PipelineInput object ready for the Analyzer Agent.

    Args:
        raw_input  : Raw text content (SQL, YAML, log, metadata)
        file_path  : Optional source file path (used for extension-based detection)

    Returns:
        PipelineInput — standardized input object
    """
    # Load from file if path provided and raw_input is empty
    if file_path and not raw_input:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        raw_input = path.read_text(encoding="utf-8")
        logger.info(f"Loaded input from file: {file_path}")

    if not raw_input or not raw_input.strip():
        raise ValueError("Input is empty. Provide a SQL query, ETL workflow, log, or metadata.")

    
    # Detect type
    ext = Path(file_path).suffix if file_path else ""
    input_type = detect_input_type(raw_input, ext)

    # Normalize
    normalized = normalize_input(raw_input, input_type, registry)
    logger.info(f"Normalized text: {normalized}")
    # Extract features
    feature_extractors = {
        "SQL Query":     extract_sql_features,
        "ETL Workflow":  extract_etl_features,
        "Pipeline Log":  extract_log_features,
        "Metadata":      extract_metadata_features,
    }
    features = feature_extractors[input_type](normalized)

    pipeline_input = PipelineInput(
        raw_text=raw_input,
        input_type=input_type,
        normalized_text=normalized,
        source_file=file_path,
        features=features,
    )

    logger.info(
        f"Ingestion complete — type={input_type}, "
        f"chars={pipeline_input.char_count}, "
        f"lines={pipeline_input.line_count}"
    )

    return pipeline_input
