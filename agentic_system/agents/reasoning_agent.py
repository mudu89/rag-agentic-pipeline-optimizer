"""
reasoning_agent.py
==================
Agent 3 — Reasoning Agent

Responsibilities:
    - Receive AnalyzerOutput + RetrievalOutput from Agents 1 and 2
    - Apply applicability conditions from KB entries to the specific input
    - Rank recommendations by expected impact
    - Identify trade-offs and reject inapplicable entries
    - Return a ReasoningOutput with ranked recommendations

This is the intellectual core of the agentic system.
It does the heaviest LLM reasoning work.

Author: Mohammed Mudassirullah Sheriff
Project: Intelligent Optimization of Data Pipelines using RAG-Based
         Knowledge Retrieval and Agentic Decision Systems
Institution: BITS Pilani WILP — M.Tech Data Science & Engineering
"""

import logging
from dataclasses import dataclass, field
from typing import List

from core.llm_client import LLMClient
from agents.analyzer_agent import AnalyzerOutput, extract_json_from_response
from agents.retrieval_agent import RetrievalOutput
from prompts.prompts import reasoning_system_prompt, reasoning_user_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output data model
# ---------------------------------------------------------------------------

@dataclass
class ReasoningOutput:
    """
    Structured output from the Reasoning Agent.
    Passed to the Recommendation Agent.
    """
    reasoning_steps:          List[str]
    ranked_recommendations:   List[dict]   # Sorted by rank ascending
    inapplicable_entries:     List[dict]
    overall_assessment:       str
    raw_llm_response:         str  = ""
    fallback_used:            bool = False

    def to_dict(self) -> dict:
        return {
            "reasoning_steps":        self.reasoning_steps,
            "ranked_recommendations": self.ranked_recommendations,
            "inapplicable_entries":   self.inapplicable_entries,
            "overall_assessment":     self.overall_assessment,
            "fallback_used":          self.fallback_used,
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


def validate_reasoning_output(data: dict) -> dict:
    """Validate and repair the Reasoning Agent LLM output."""
    if not isinstance(data.get("reasoning_steps"), list):
        data["reasoning_steps"] = ["Reasoning steps not available."]

    if not isinstance(data.get("ranked_recommendations"), list):
        data["ranked_recommendations"] = []

    if not isinstance(data.get("inapplicable_entries"), list):
        data["inapplicable_entries"] = []

    if not isinstance(data.get("overall_assessment"), str):
        data["overall_assessment"] = ""

    # Ensure rank field exists and is integer
    for i, rec in enumerate(data["ranked_recommendations"]):
        if "rank" not in rec or not isinstance(rec["rank"], int):
            rec["rank"] = i + 1
        if "applicable" not in rec:
            rec["applicable"] = True
        if "severity" not in rec:
            rec["severity"] = "Medium"
        if "trade_offs" not in rec:
            rec["trade_offs"] = "None identified."
        if "rationale" not in rec:
            rec["rationale"] = ""

    # Sort by rank ascending to ensure correct order
    data["ranked_recommendations"].sort(key=lambda x: x.get("rank", 99))

    return data


# ---------------------------------------------------------------------------
# Rule-based fallback reasoning
# ---------------------------------------------------------------------------

def fallback_reasoning(
    analyzer_output: AnalyzerOutput,
    retrieval_output: RetrievalOutput,
) -> dict:
    """
    Simple rule-based fallback when LLM reasoning fails.
    Ranks retrieved entries by severity then similarity score.
    """
    selected = retrieval_output.selected_entries

    # Sort by severity then similarity
    def sort_key(entry):
        sev   = SEVERITY_ORDER.get(entry.get("severity", "Medium"), 2)
        score = entry.get("similarity_score", 0.0)
        return (sev, -score)

    sorted_entries = sorted(selected, key=sort_key)

    ranked = []
    for i, entry in enumerate(sorted_entries):
        eid = entry.get("entry_id") or entry.get("chunk_id", f"ENTRY-{i}")
        ranked.append({
            "rank":       i + 1,
            "entry_id":   eid,
            "title":      entry.get("title", "Untitled"),
            "severity":   entry.get("severity", "Medium"),
            "rationale":  (
                f"Ranked by severity ({entry.get('severity', 'Medium')}) "
                f"and similarity score ({entry.get('similarity_score', 0):.3f}). "
                f"Fallback reasoning mode."
            ),
            "applicable": True,
            "trade_offs": "Review applicability conditions manually in fallback mode.",
        })

    patterns = analyzer_output.detected_patterns
    assessment = (
        f"Fallback reasoning applied. {len(patterns)} inefficiency pattern(s) detected. "
        f"{len(ranked)} optimization(s) identified from knowledge base. "
        f"Manual review recommended."
    )

    return {
        "reasoning_steps": [
            f"Step 1: Received {len(patterns)} detected patterns from Analyzer Agent.",
            f"Step 2: Retrieved {len(selected)} KB entries from Retrieval Agent.",
            f"Step 3: Ranked by severity and similarity score (fallback — LLM unavailable).",
        ],
        "ranked_recommendations": ranked,
        "inapplicable_entries":   [],
        "overall_assessment":     assessment,
    }


# ---------------------------------------------------------------------------
# Reasoning Agent
# ---------------------------------------------------------------------------

class ReasoningAgent:
    """
    Agent 3 — Reasoning Agent.

    The intellectual core of the pipeline. Evaluates retrieved KB entries
    against detected patterns, applies applicability conditions, ranks
    recommendations by impact, and identifies trade-offs.
    """

    def __init__(self, llm_client: LLMClient, config: dict = None):
        self.llm     = llm_client
        self.config  = config or {}
        self.retries = self.config.get("agents", {}).get("max_retries", 3)
        self.delay   = self.config.get("agents", {}).get("retry_delay_seconds", 2)
        self.max_rec = self.config.get("agents", {}).get("max_recommendations", 3)

    def run(
        self,
        analyzer_output:  AnalyzerOutput,
        retrieval_output: RetrievalOutput,
    ) -> ReasoningOutput:
        """
        Main entry point. Reason about retrieved entries and rank recommendations.

        Args:
            analyzer_output  : Output from Agent 1
            retrieval_output : Output from Agent 2

        Returns:
            ReasoningOutput with ranked recommendations and reasoning steps
        """
        logger.info(
            f"[Reasoning Agent] Running — "
            f"selected_entries={len(retrieval_output.selected_entries)}, "
            f"patterns={len(analyzer_output.detected_patterns)}"
        )

        # If no entries were retrieved, return empty reasoning
        if not retrieval_output.selected_entries:
            logger.warning(
                "[Reasoning Agent] No selected entries to reason about. "
                "Returning empty output."
            )
            return ReasoningOutput(
                reasoning_steps=[
                    "No relevant KB entries were retrieved. "
                    "Unable to generate recommendations."
                ],
                ranked_recommendations=[],
                inapplicable_entries=[],
                overall_assessment=(
                    "No optimization recommendations could be generated. "
                    "The knowledge base may not contain relevant entries for "
                    "this input type. Consider expanding the KB or rephrasing the input."
                ),
                fallback_used=True,
            )

        # Build the reasoning prompt
        system_prompt = reasoning_system_prompt()
        user_prompt   = reasoning_user_prompt(
            analyzer_output=analyzer_output.to_dict(),
            retrieval_output=retrieval_output.to_dict(),
            kb_entries_full=retrieval_output.selected_entries,
        )

        raw_response  = ""
        fallback_used = False
        logger.info(f"[Reasoning Agent] System Prompt: {system_prompt}")
        logger.info(f"[Reasoning Agent] User Prompt: {user_prompt}")
        try:
            raw_response = self.llm.complete_with_retry(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_retries=self.retries,
                retry_delay=float(self.delay),
            )
            logger.debug(f"[Reasoning Agent] Raw LLM response:\n{raw_response}")

            parsed = extract_json_from_response(raw_response)
            parsed = validate_reasoning_output(parsed)

            # Trim to max_recommendations
            parsed["ranked_recommendations"] = (
                parsed["ranked_recommendations"][:self.max_rec]
            )

        except Exception as e:
            logger.warning(
                f"[Reasoning Agent] LLM call failed: {e}. "
                f"Using fallback reasoning."
            )
            parsed        = fallback_reasoning(analyzer_output, retrieval_output)
            parsed        = validate_reasoning_output(parsed)
            fallback_used = True

        output = ReasoningOutput(
            reasoning_steps=parsed.get("reasoning_steps", []),
            ranked_recommendations=parsed.get("ranked_recommendations", []),
            inapplicable_entries=parsed.get("inapplicable_entries", []),
            overall_assessment=parsed.get("overall_assessment", ""),
            raw_llm_response=raw_response,
            fallback_used=fallback_used,
        )

        logger.info(
            f"[Reasoning Agent] Complete — "
            f"recommendations={len(output.ranked_recommendations)}, "
            f"inapplicable={len(output.inapplicable_entries)}, "
            f"fallback={fallback_used}"
        )

        return output
