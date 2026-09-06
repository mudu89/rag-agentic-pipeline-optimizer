"""
retrieval_agent.py
==================
Agent 2 — Retrieval Agent

Responsibilities:
    - Receive AnalyzerOutput from Agent 1
    - Call the KnowledgeBase to retrieve Top-K similar entries
    - Use the LLM to select and justify the most relevant entries
    - Return a RetrievalOutput with selected entries and context block

Design note:
    This agent is split into two layers:
        Code layer  → KnowledgeBase.query() handles the actual retrieval
        LLM layer   → Selects and justifies top entries from retrieved results

    The LLM component is intentionally lightweight here — the heavy
    semantic work is done by FAISS + embeddings, not the LLM.

Author: Mohammed Mudassirullah Sheriff
Project: Intelligent Optimization of Data Pipelines using RAG-Based
         Knowledge Retrieval and Agentic Decision Systems
Institution: BITS Pilani WILP — M.Tech Data Science & Engineering
"""

import json
import logging
import re
import sys
import os
from dataclasses import dataclass, field
from typing import List, Optional

# Allow imports from knowledge_base/scripts
sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "..", "knowledge_base", "scripts")
)

from core.llm_client import LLMClient
from agents.analyzer_agent import AnalyzerOutput, extract_json_from_response
from prompts.prompts import retrieval_system_prompt, retrieval_user_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output data model
# ---------------------------------------------------------------------------

@dataclass
class RetrievalOutput:
    """
    Structured output from the Retrieval Agent.
    Passed to the Reasoning Agent.
    """
    selected_entries:    List[dict]          # Full metadata for selected KB entries
    rejected_entries:    List[dict]          # Entries retrieved but filtered out
    retrieval_summary:   str                 # Brief summary of what was found
    raw_retrieved:       List[dict]          # All raw results from FAISS (for trace)
    raw_llm_response:    str       = ""
    fallback_used:       bool      = False   # True if LLM selection failed

    def to_dict(self) -> dict:
        return {
            "selected_entries":  self.selected_entries,
            "rejected_entries":  self.rejected_entries,
            "retrieval_summary": self.retrieval_summary,
            "total_retrieved":   len(self.raw_retrieved),
            "total_selected":    len(self.selected_entries),
            "fallback_used":     self.fallback_used,
        }


# ---------------------------------------------------------------------------
# Lazy KnowledgeBase loader
# ---------------------------------------------------------------------------

def load_knowledge_base(kb_config: dict):
    """
    Dynamically import and load the KnowledgeBase from query_kb.py.
    Uses lazy loading so the KB is only loaded once per pipeline run.
    """
    try:
        from query_kb import KnowledgeBase
    except ImportError:
        # Fallback import path
        scripts_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "knowledge_base", "scripts"
        )
        sys.path.insert(0, scripts_path)
        from query_kb import KnowledgeBase

    kb = KnowledgeBase(
        index_path=kb_config.get("index_path",    "../knowledge_base/index/faiss_index.bin"),
        meta_path=kb_config.get("metadata_path",  "../knowledge_base/index/metadata_store.json"),
        model_name=kb_config.get("embedding_model", "all-MiniLM-L6-v2"),
    )
    kb.load()
    return kb


# ---------------------------------------------------------------------------
# Fallback selection (when LLM call fails)
# ---------------------------------------------------------------------------

def fallback_select_entries(
    retrieved: list,
    max_select: int = 3,
) -> tuple:
    """
    Select top entries by similarity score when LLM selection fails.
    Prefers Layer 1 entries over Layer 2 chunks.
    """
    # Sort: Layer 1 first, then by similarity score
    sorted_entries = sorted(
        retrieved,
        key=lambda x: (-(x.get("layer", 2) == 1), -x.get("similarity_score", 0)),
    )

    selected = []
    for entry in sorted_entries[:max_select]:
        eid = entry.get("entry_id") or entry.get("chunk_id", "UNKNOWN")
        selected.append({
            "entry_id":        eid,
            "title":           entry.get("title", "Untitled"),
            "relevance_reason":"Selected by similarity score (fallback mode)",
            "applicable":      True,
            "layer":           entry.get("layer", 1),
        })

    return selected, []


def validate_retrieval_output(data: dict, retrieved_ids: set) -> dict:
    """Validate and repair LLM retrieval selection output."""
    if "selected_entries" not in data or not isinstance(data["selected_entries"], list):
        data["selected_entries"] = []

    if "rejected_entries" not in data or not isinstance(data["rejected_entries"], list):
        data["rejected_entries"] = []

    if "retrieval_summary" not in data or not isinstance(data["retrieval_summary"], str):
        data["retrieval_summary"] = "Relevant optimization entries retrieved."

    # Validate that selected entry IDs exist in retrieved set
    valid_selected = []
    for entry in data["selected_entries"]:
        eid = entry.get("entry_id", "")
        if eid in retrieved_ids or not retrieved_ids:
            valid_selected.append(entry)
        else:
            logger.warning(
                f"[Retrieval Agent] LLM selected entry '{eid}' "
                f"which was not in retrieved results — discarding."
            )
    data["selected_entries"] = valid_selected

    return data


# ---------------------------------------------------------------------------
# Retrieval Agent
# ---------------------------------------------------------------------------

class RetrievalAgent:
    """
    Agent 2 — Retrieval Agent.

    Queries the knowledge base and uses the LLM to select
    the most relevant entries for the detected patterns.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: dict = None,
        knowledge_base=None,   # Pre-loaded KB instance (optional — avoids reloading)
    ):
        self.llm     = llm_client
        self.config  = config or {}
        self.kb      = knowledge_base
        self.retries = self.config.get("agents", {}).get("max_retries", 3)
        self.delay   = self.config.get("agents", {}).get("retry_delay_seconds", 2)

        kb_config    = self.config.get("knowledge_base", {})
        self.top_k   = kb_config.get("top_k", 5)
        self.min_sim = kb_config.get("min_similarity", 0.40)
        self.prefer_l1 = kb_config.get("layer1_preference", True)

    def _ensure_kb_loaded(self):
        """Load the KB if not already loaded."""
        if self.kb is None:
            kb_config = self.config.get("knowledge_base", {})
            logger.info("[Retrieval Agent] Loading knowledge base...")
            self.kb = load_knowledge_base(kb_config)

    def run(
        self,
        analyzer_output: AnalyzerOutput,
    ) -> RetrievalOutput:
        """
        Main entry point. Retrieve and select relevant KB entries.

        Args:
            analyzer_output: Output from the Analyzer Agent

        Returns:
            RetrievalOutput with selected entries and full retrieved context
        """
        logger.info(
            f"[Retrieval Agent] Running — "
            f"query='{analyzer_output.retrieval_query[:60]}...'"
        )

        # Step 1 — Ensure KB is loaded
        self._ensure_kb_loaded()

        # Step 2 — Query the knowledge base
        # Use suggested_categories for pre-filtering if available
        category_filter = (
            analyzer_output.suggested_categories[0]
            if len(analyzer_output.suggested_categories) == 1
            else None
        )

        retrieved = self.kb.query(
            query_text=analyzer_output.retrieval_query,
            top_k=self.top_k,
            input_type_filter=analyzer_output.input_type,
            category_filter=category_filter,
            min_similarity=self.min_sim,
            prefer_layer1=self.prefer_l1,
        )

        # If category filter returned too few results, retry without it
        if len(retrieved) < 2 and category_filter:
            logger.info(
                "[Retrieval Agent] Few results with category filter — "
                "retrying without filter."
            )
            retrieved = self.kb.query(
                query_text=analyzer_output.retrieval_query,
                top_k=self.top_k,
                input_type_filter=analyzer_output.input_type,
                min_similarity=self.min_sim,
                prefer_layer1=self.prefer_l1,
            )

        logger.info(f"[Retrieval Agent] Retrieved {len(retrieved)} entries from KB.")

        if not retrieved:
            logger.warning("[Retrieval Agent] No entries retrieved — returning empty output.")
            return RetrievalOutput(
                selected_entries=[],
                rejected_entries=[],
                retrieval_summary="No relevant entries found in the knowledge base.",
                raw_retrieved=[],
                fallback_used=True,
            )

        # Step 3 — LLM selection: choose best entries from retrieved set
        raw_response  = ""
        fallback_used = False

        retrieved_ids = {
            e.get("entry_id") or e.get("chunk_id", "") for e in retrieved
        }

        system_prompt = retrieval_system_prompt()
        user_prompt   = retrieval_user_prompt(
            analyzer_output=analyzer_output.to_dict(),
            retrieved_entries=retrieved,
        )
        logger.info(f"[Retrieval Agent] System Prompt: {system_prompt}")
        logger.info(f"[Retrieval Agent] User Prompt: {user_prompt}")
        try:
            raw_response = self.llm.complete_with_retry(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_retries=self.retries,
                retry_delay=float(self.delay),
            )
            parsed = extract_json_from_response(raw_response)
            parsed = validate_retrieval_output(parsed, retrieved_ids)

            selected  = parsed.get("selected_entries", [])
            rejected  = parsed.get("rejected_entries", [])
            summary   = parsed.get("retrieval_summary", "")

            # Fallback if LLM selected nothing
            if not selected:
                logger.warning(
                    "[Retrieval Agent] LLM selected no entries — using fallback."
                )
                selected, rejected = fallback_select_entries(retrieved)
                fallback_used = True

        except Exception as e:
            logger.warning(
                f"[Retrieval Agent] LLM selection failed: {e}. "
                f"Using fallback selection."
            )
            selected, rejected = fallback_select_entries(retrieved)
            summary       = "Entries selected by similarity score (fallback mode)."
            fallback_used = True

        # Step 4 — Enrich selected entries with full metadata from retrieved results
        retrieved_lookup = {
            (e.get("entry_id") or e.get("chunk_id", "")): e
            for e in retrieved
        }
        enriched_selected = []
        for sel in selected:
            eid      = sel.get("entry_id", "")
            full_entry = retrieved_lookup.get(eid, {})
            enriched_selected.append({
                **full_entry,       # Full KB metadata
                **sel,              # LLM relevance_reason + applicable flag
            })

        output = RetrievalOutput(
            selected_entries=enriched_selected,
            rejected_entries=rejected,
            retrieval_summary=summary if summary else (
                f"Retrieved {len(retrieved)} entries; selected {len(selected)} "
                f"most relevant."
            ),
            raw_retrieved=retrieved,
            raw_llm_response=raw_response,
            fallback_used=fallback_used,
        )

        logger.info(
            f"[Retrieval Agent] Complete — "
            f"selected={len(output.selected_entries)}, "
            f"fallback={fallback_used}"
        )

        return output
