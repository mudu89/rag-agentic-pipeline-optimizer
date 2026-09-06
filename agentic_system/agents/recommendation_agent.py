"""
recommendation_agent.py
=======================
Agent 4 — Recommendation Agent

Responsibilities:
    - Receive outputs from all three preceding agents
    - Generate the final human-readable, explainable recommendation report
    - Produce both a structured JSON output and a formatted text report
    - Include the full agent trace for explainability and dissertation evidence

This is the final stage of the agentic pipeline.
Its output is what the user sees — via CLI or FastAPI.

Author: Mohammed Mudassirullah Sheriff
Project: Intelligent Optimization of Data Pipelines using RAG-Based
         Knowledge Retrieval and Agentic Decision Systems
Institution: BITS Pilani WILP — M.Tech Data Science & Engineering
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from core.llm_client import LLMClient
from agents.analyzer_agent import AnalyzerOutput, extract_json_from_response
from agents.retrieval_agent import RetrievalOutput
from agents.reasoning_agent import ReasoningOutput
from prompts.prompts import recommendation_system_prompt, recommendation_user_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output data model
# ---------------------------------------------------------------------------

@dataclass
class RecommendationOutput:
    """
    Final structured output from the complete agentic pipeline.

    This is the object returned to the CLI and FastAPI layer.
    Contains:
        - The user-facing recommendation report
        - Full agent trace for explainability
        - Pipeline metadata
    """
    report_summary:         str
    total_issues_detected:  int
    recommendations:        List[dict]
    additional_notes:       str
    agent_trace:            dict          # Full trace of all agent outputs
    pipeline_metadata:      dict          # Timing, model used, KB stats
    raw_llm_response:       str  = ""
    fallback_used:          bool = False

    def to_dict(self, include_trace: bool = True) -> dict:
        output = {
            "report_summary":        self.report_summary,
            "total_issues_detected": self.total_issues_detected,
            "recommendations":       self.recommendations,
            "additional_notes":      self.additional_notes,
            "pipeline_metadata":     self.pipeline_metadata,
        }
        if include_trace:
            output["agent_trace"] = self.agent_trace
        return output

    def to_text_report(self) -> str:
        """
        Format the recommendation output as a readable text report
        for CLI display.
        """
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("  PIPELINE OPTIMIZATION REPORT")
        lines.append("=" * 70)
        lines.append(f"\n  {self.report_summary}\n")
        lines.append(f"  Issues detected : {self.total_issues_detected}")
        lines.append(
            f"  Recommendations : {len(self.recommendations)}"
        )
        lines.append(
            f"  Generated at    : "
            f"{self.pipeline_metadata.get('timestamp', 'N/A')}"
        )
        lines.append(
            f"  LLM model       : "
            f"{self.pipeline_metadata.get('llm_model', 'N/A')}"
        )

        if not self.recommendations:
            lines.append("\n  No recommendations generated.")
            lines.append("=" * 70 + "\n")
            return "\n".join(lines)

        lines.append("\n" + "-" * 70)
        lines.append("  RECOMMENDATIONS")
        lines.append("-" * 70)

        for rec in self.recommendations:
            sev_icon = {
                "Critical": "🔴",
                "High":     "🟠",
                "Medium":   "🟡",
                "Low":      "🟢",
            }.get(rec.get("severity", "Medium"), "⚪")

            lines.append(
                f"\n  [{rec.get('rank', '?')}] {sev_icon}  "
                f"{rec.get('title', 'Untitled')}  "
                f"[{rec.get('severity', 'N/A')}]"
            )
            lines.append(f"  Entry ID   : {rec.get('entry_id', 'N/A')}")
            lines.append(f"  Source     : {rec.get('source', 'N/A')}")
            lines.append(f"  Confidence : {rec.get('confidence', 'N/A')}")
            lines.append(f"\n  PROBLEM\n  {rec.get('problem', 'N/A')}")
            lines.append(f"\n  SOLUTION\n  {rec.get('solution', 'N/A')}")

            if rec.get("before_example") and rec.get("before_example") != "N/A":
                lines.append(f"\n  BEFORE:\n    {rec['before_example']}")
                lines.append(f"\n  AFTER:\n    {rec['after_example']}")

            lines.append(f"\n  WHY IT WORKS\n  {rec.get('explanation', 'N/A')}")

            impact = rec.get("expected_impact", {})
            if impact:
                lines.append("\n  EXPECTED IMPACT")
                for k, v in impact.items():
                    lines.append(f"    {k:<20} : {v}")

            lines.append("\n" + "-" * 70)

        if self.additional_notes:
            lines.append(f"\n  NOTES\n  {self.additional_notes}")

        lines.append("=" * 70 + "\n")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_recommendation_output(data: dict) -> dict:
    """Validate and repair the Recommendation Agent LLM output."""
    if not isinstance(data.get("report_summary"), str):
        data["report_summary"] = "Optimization analysis complete."

    if not isinstance(data.get("recommendations"), list):
        data["recommendations"] = []

    if not isinstance(data.get("additional_notes"), str):
        data["additional_notes"] = ""

    if not isinstance(data.get("total_issues_detected"), int):
        data["total_issues_detected"] = 0

    # Validate each recommendation has required fields
    for i, rec in enumerate(data["recommendations"]):
        if "rank" not in rec:
            rec["rank"] = i + 1
        for field_name in ["title", "severity", "problem", "solution",
                           "explanation", "source"]:
            if field_name not in rec or not rec[field_name]:
                rec[field_name] = "N/A"
        if "expected_impact" not in rec or not isinstance(rec["expected_impact"], dict):
            rec["expected_impact"] = {}
        if "confidence" not in rec:
            rec["confidence"] = 0.85

    return data


# ---------------------------------------------------------------------------
# Fallback report generation
# ---------------------------------------------------------------------------

def fallback_recommendation(
    analyzer_output:  AnalyzerOutput,
    retrieval_output: RetrievalOutput,
    reasoning_output: ReasoningOutput,
) -> dict:
    """
    Generate a basic recommendation report from reasoning output
    when the LLM call fails.
    """
    recommendations = []
    for rec in reasoning_output.ranked_recommendations:
        eid = rec.get("entry_id", "")

        # Find the full entry from retrieval output
        entry = {}
        for e in retrieval_output.selected_entries:
            if (e.get("entry_id") or e.get("chunk_id", "")) == eid:
                entry = e
                break

        recommendations.append({
            "rank":           rec.get("rank", 1),
            "entry_id":       eid,
            "title":          rec.get("title") or entry.get("title", "Optimization"),
            "severity":       rec.get("severity", "Medium"),
            "problem":        entry.get("problem_description", "See KB entry for details."),
            "solution":       entry.get("optimization_strategy", "See KB entry for details."),
            "before_example": entry.get("before_example", "N/A"),
            "after_example":  entry.get("after_example", "N/A"),
            "explanation":    entry.get("explanation", "N/A"),
            "expected_impact": entry.get("expected_impact", {}),
            "source":         entry.get("source", "N/A"),
            "confidence":     entry.get("confidence_score", 0.85),
        })

    return {
        "report_summary":        (
            f"{len(analyzer_output.detected_patterns)} inefficiency pattern(s) detected. "
            f"{len(recommendations)} optimization recommendation(s) generated. "
            f"Report generated in fallback mode."
        ),
        "total_issues_detected": len(analyzer_output.detected_patterns),
        "recommendations":       recommendations,
        "additional_notes":      (
            "This report was generated using fallback mode. "
            "LLM narrative generation was unavailable. "
            "Recommendations are based on KB entries selected by the Retrieval Agent."
        ),
    }


# ---------------------------------------------------------------------------
# Recommendation Agent
# ---------------------------------------------------------------------------

class RecommendationAgent:
    """
    Agent 4 — Recommendation Agent.

    Converts structured reasoning output into a human-readable,
    explainable recommendation report with full agent trace.
    """

    def __init__(self, llm_client: LLMClient, config: dict = None):
        self.llm          = llm_client
        self.config       = config or {}
        self.retries      = self.config.get("agents", {}).get("max_retries", 3)
        self.delay        = self.config.get("agents", {}).get("retry_delay_seconds", 2)
        self.max_rec      = self.config.get("agents", {}).get("max_recommendations", 3)
        self.include_trace = self.config.get("agents", {}).get("include_agent_trace", True)

    def run(
        self,
        analyzer_output:   AnalyzerOutput,
        retrieval_output:  RetrievalOutput,
        reasoning_output:  ReasoningOutput,
        llm_model:         str = "unknown",
        elapsed_seconds:   float = 0.0,
    ) -> RecommendationOutput:
        """
        Main entry point. Generate the final recommendation report.

        Args:
            analyzer_output  : Output from Agent 1
            retrieval_output : Output from Agent 2
            reasoning_output : Output from Agent 3
            llm_model        : Model name for pipeline metadata
            elapsed_seconds  : Time taken so far (for metadata)

        Returns:
            RecommendationOutput — the final pipeline result
        """
        logger.info(
            f"[Recommendation Agent] Running — "
            f"ranked_recommendations={len(reasoning_output.ranked_recommendations)}"
        )

        # Build pipeline metadata
        pipeline_metadata = {
            "timestamp":           datetime.now().isoformat(),
            "llm_model":           llm_model,
            "input_type":          analyzer_output.input_type,
            "patterns_detected":   len(analyzer_output.detected_patterns),
            "kb_entries_retrieved":len(retrieval_output.raw_retrieved),
            "kb_entries_selected": len(retrieval_output.selected_entries),
            "recommendations_generated": len(reasoning_output.ranked_recommendations),
            "elapsed_seconds":     round(elapsed_seconds, 2),
            "fallback_agents": {
                "analyzer":   analyzer_output.fallback_used,
                "retrieval":  retrieval_output.fallback_used,
                "reasoning":  reasoning_output.fallback_used,
            },
        }

        # Build agent trace (for explainability)
        agent_trace = {
            "analyzer_output":  analyzer_output.to_dict(),
            "retrieval_output": retrieval_output.to_dict(),
            "reasoning_output": reasoning_output.to_dict(),
        }

        # Build recommendation prompt
        system_prompt = recommendation_system_prompt()
        user_prompt   = recommendation_user_prompt(
            analyzer_output=analyzer_output.to_dict(),
            retrieval_output=retrieval_output.to_dict(),
            reasoning_output=reasoning_output.to_dict(),
            kb_entries_full=retrieval_output.selected_entries,
            max_recommendations=self.max_rec,
        )

        raw_response  = ""
        fallback_used = False
        logger.info(f"[Recommendation Agent] System Prompt: {system_prompt}")
        logger.info(f"[Recommendation Agent] User Prompt: {user_prompt}")
        try:
            raw_response = self.llm.complete_with_retry(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_retries=self.retries,
                retry_delay=float(self.delay),
            )
            logger.debug(
                f"[Recommendation Agent] Raw LLM response:\n{raw_response}"
            )

            parsed = extract_json_from_response(raw_response)
            parsed = validate_recommendation_output(parsed)

        except Exception as e:
            logger.warning(
                f"[Recommendation Agent] LLM call failed: {e}. "
                f"Using fallback report generation."
            )
            parsed        = fallback_recommendation(
                analyzer_output, retrieval_output, reasoning_output
            )
            parsed        = validate_recommendation_output(parsed)
            fallback_used = True

        output = RecommendationOutput(
            report_summary=parsed.get("report_summary", ""),
            total_issues_detected=parsed.get(
                "total_issues_detected",
                len(analyzer_output.detected_patterns)
            ),
            recommendations=parsed.get("recommendations", []),
            additional_notes=parsed.get("additional_notes", ""),
            agent_trace=agent_trace,
            pipeline_metadata=pipeline_metadata,
            raw_llm_response=raw_response,
            fallback_used=fallback_used,
        )

        logger.info(
            f"[Recommendation Agent] Complete — "
            f"recommendations={len(output.recommendations)}, "
            f"fallback={fallback_used}"
        )

        return output
