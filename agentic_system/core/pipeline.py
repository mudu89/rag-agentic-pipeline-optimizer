"""
pipeline.py
===========
Pipeline Orchestrator — connects all four agents sequentially.

This is the single entry point for the complete agentic system.
Both the CLI and FastAPI layer call Pipeline.run() — they never
instantiate individual agents directly.

Execution sequence:
    Ingestion → Agent 1 (Analyzer) → Agent 2 (Retrieval)
              → Agent 3 (Reasoning) → Agent 4 (Recommendation)

Author: Mohammed Mudassirullah Sheriff
Project: Intelligent Optimization of Data Pipelines using RAG-Based
         Knowledge Retrieval and Agentic Decision Systems
Institution: BITS Pilani WILP — M.Tech Data Science & Engineering
"""

import logging
import time
import os
import yaml
from pathlib import Path

from core.llm_client import LLMClient, load_config
from core.ingestion import ingest, PipelineInput, load_table_registry
from agents.analyzer_agent import AnalyzerAgent
from agents.retrieval_agent import RetrievalAgent
from agents.reasoning_agent import ReasoningAgent
from agents.recommendation_agent import RecommendationAgent, RecommendationOutput

logger = logging.getLogger(__name__)


def setup_logging(config: dict):
    """Configure logging from config settings."""
    log_config  = config.get("logging", {})
    level       = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    log_to_file = log_config.get("log_to_file", False)
    log_path    = log_config.get("log_path", "../logs/pipeline.log")

    handlers = [logging.StreamHandler()]

    if log_to_file:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class Pipeline:
    """
    Main pipeline orchestrator.

    Initializes all agents once and runs them sequentially for each input.
    The KnowledgeBase is loaded once at pipeline startup and reused across
    multiple queries — important for performance.

    Usage:
        pipeline = Pipeline()
        pipeline.load()
        result = pipeline.run(raw_input="SELECT * FROM orders WHERE ...")
        print(result.to_text_report())
    """

    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config      = load_config(config_path)
        #self.registry    = load_table_registry()
        self._loaded     = False

        setup_logging(self.config)

        # These are initialized in load()
        self.llm_client          = None
        self.knowledge_base      = None
        self.analyzer_agent      = None
        self.retrieval_agent     = None
        self.reasoning_agent     = None
        self.recommendation_agent = None
        self.registry            = None

    def load(self):
        """
        Initialize all agents and load the knowledge base.
        Call this once before running queries.
        Heavy resources (KB, embedding model, LLM client) are loaded here.
        """
        if self._loaded:
            return

        logger.info("=" * 60)
        logger.info("Initializing Pipeline...")
        logger.info("=" * 60)

        # Initialize LLM client
        logger.info("Loading LLM client...")
        self.llm_client = LLMClient(self.config_path)

        # Load knowledge base once
        logger.info("Loading knowledge base...")
        from agents.retrieval_agent import load_knowledge_base
        kb_config            = self.config.get("knowledge_base", {})
        self.knowledge_base  = load_knowledge_base(kb_config)

        # Initialize agents — pass the shared KB to retrieval agent
        llm_cfg = self.config.get("llm", {})
        self.llm_model = llm_cfg.get("model", "unknown")

        self.analyzer_agent       = AnalyzerAgent(self.llm_client, self.config)
        self.retrieval_agent      = RetrievalAgent(
            self.llm_client, self.config, knowledge_base=self.knowledge_base
        )
        self.reasoning_agent      = ReasoningAgent(self.llm_client, self.config)
        self.recommendation_agent = RecommendationAgent(self.llm_client, self.config)

        self._loaded = True

        #Load the registry
        self.registry = load_table_registry()

        logger.info("Pipeline ready.")

    def run(
        self,
        raw_input:  str,
        file_path:  str = None,
    ) -> RecommendationOutput:
        """
        Run the complete four-agent pipeline on a single input.

        Args:
            raw_input  : Raw text input (SQL, ETL YAML, log, metadata)
            file_path  : Optional file path (used for extension-based type detection)

        Returns:
            RecommendationOutput — the final pipeline result
        """
        if not self._loaded:
            self.load()

        pipeline_start = time.time()

        logger.info("=" * 60)
        logger.info("Pipeline run started")
        logger.info("=" * 60)

        # ----------------------------------------------------------------
        # Stage 2.1 — Ingestion & Preprocessing
        # ----------------------------------------------------------------
        logger.info("[Stage 2.1] Ingestion & Preprocessing")
        t0 = time.time()
        pipeline_input = ingest(raw_input=raw_input, file_path=file_path, registry=self.registry)
        logger.info(
            f"[Stage 2.1] Complete — type={pipeline_input.input_type}, "
            f"elapsed={time.time()-t0:.2f}s"
        )

        # ----------------------------------------------------------------
        # Stage 2.5 — Agent 1: Analyzer
        # ----------------------------------------------------------------
        logger.info("[Agent 1] Analyzer Agent")
        t0 = time.time()
        analyzer_output = self.analyzer_agent.run(pipeline_input)
        logger.info(
            f"[Agent 1] Complete — "
            f"patterns={len(analyzer_output.detected_patterns)}, "
            f"elapsed={time.time()-t0:.2f}s"
        )

        # ----------------------------------------------------------------
        # Stage 2.4+2.5 — Agent 2: Retrieval
        # ----------------------------------------------------------------
        logger.info("[Agent 2] Retrieval Agent")
        t0 = time.time()
        retrieval_output = self.retrieval_agent.run(analyzer_output)
        logger.info(
            f"[Agent 2] Complete — "
            f"selected={len(retrieval_output.selected_entries)}, "
            f"elapsed={time.time()-t0:.2f}s"
        )

        # ----------------------------------------------------------------
        # Stage 2.5 — Agent 3: Reasoning
        # ----------------------------------------------------------------
        logger.info("[Agent 3] Reasoning Agent")
        t0 = time.time()
        reasoning_output = self.reasoning_agent.run(analyzer_output, retrieval_output)
        logger.info(
            f"[Agent 3] Complete — "
            f"recommendations={len(reasoning_output.ranked_recommendations)}, "
            f"elapsed={time.time()-t0:.2f}s"
        )

        # ----------------------------------------------------------------
        # Stage 2.5 — Agent 4: Recommendation
        # ----------------------------------------------------------------
        logger.info("[Agent 4] Recommendation Agent")
        t0 = time.time()
        total_elapsed = time.time() - pipeline_start

        recommendation_output = self.recommendation_agent.run(
            analyzer_output=analyzer_output,
            retrieval_output=retrieval_output,
            reasoning_output=reasoning_output,
            llm_model=self.llm_model,
            elapsed_seconds=total_elapsed,
        )
        logger.info(
            f"[Agent 4] Complete — "
            f"elapsed={time.time()-t0:.2f}s"
        )

        total_elapsed = time.time() - pipeline_start
        logger.info(
            f"Pipeline run complete — "
            f"total_elapsed={total_elapsed:.2f}s, "
            f"recommendations={len(recommendation_output.recommendations)}"
        )

        return recommendation_output
