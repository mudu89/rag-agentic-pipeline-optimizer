"""
analyzer_agent.py
=================
Agent 1 — Analyzer Agent

Responsibilities:
    - Receive a PipelineInput object from the Ingestion module
    - Send it to the LLM with the Analyzer prompt
    - Parse and validate the structured JSON response
    - Return an AnalyzerOutput object for the Retrieval Agent

This agent does NOT recommend solutions.
Its only job is to detect and classify inefficiency patterns.

Author: Mohammed Mudassirullah Sheriff
Project: Intelligent Optimization of Data Pipelines using RAG-Based
         Knowledge Retrieval and Agentic Decision Systems
Institution: BITS Pilani WILP — M.Tech Data Science & Engineering
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List

from core.llm_client import LLMClient
from core.ingestion import PipelineInput
from prompts.prompts import analyzer_system_prompt, analyzer_user_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output data model
# ---------------------------------------------------------------------------

@dataclass
class AnalyzerOutput:
    """
    Structured output from the Analyzer Agent.
    Passed directly to the Retrieval Agent.
    """
    input_type:           str
    detected_patterns:    List[str]
    severity_signals:     List[str]
    suggested_categories: List[str]
    summary:              str
    retrieval_query:      str
    raw_llm_response:     str        = ""   # Preserved for agent trace
    fallback_used:        bool       = False # True if LLM failed and rule-based fallback ran

    def to_dict(self) -> dict:
        return {
            "input_type":           self.input_type,
            "detected_patterns":    self.detected_patterns,
            "severity_signals":     self.severity_signals,
            "suggested_categories": self.suggested_categories,
            "summary":              self.summary,
            "retrieval_query":      self.retrieval_query,
            "fallback_used":        self.fallback_used,
        }


# ---------------------------------------------------------------------------
# JSON parsing utilities
# ---------------------------------------------------------------------------

def extract_json_from_response(text: str) -> dict:
    """
    Robustly extract a JSON object from LLM response text.

    Handles cases where the model wraps the JSON in markdown fences
    or adds preamble text despite instructions not to.
    """
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Find first JSON object
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found:\n{text}")

    candidate = text[start:].strip()

    # ---- Temporary repair ----
    open_braces = candidate.count("{")
    close_braces = candidate.count("}")

    if open_braces == close_braces + 1:
        logger.warning(
            "LLM response appears to be missing the final '}'. "
            "Repairing automatically."
        )
        candidate += "}"

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Try extracting the first {...} block
    match = re.search(r"\{[\s\S]+\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from LLM response:\n{text}")


def validate_analyzer_output(data: dict) -> dict:
    """
    Validate and repair the parsed Analyzer output.
    Ensures all required fields exist with correct types.
    """
    required_lists = ["detected_patterns", "severity_signals", "suggested_categories"]
    required_strs  = ["input_type", "summary", "retrieval_query"]

    for field_name in required_lists:
        if field_name not in data or not isinstance(data[field_name], list):
            data[field_name] = []

    for field_name in required_strs:
        if field_name not in data or not isinstance(data[field_name], str):
            data[field_name] = ""

    # Ensure severity_signals matches detected_patterns length
    patterns  = data["detected_patterns"]
    severities = data["severity_signals"]
    if len(severities) < len(patterns):
        data["severity_signals"] = severities + ["Medium"] * (len(patterns) - len(severities))

    return data


# ---------------------------------------------------------------------------
# Rule-based fallback (when LLM call fails)
# ---------------------------------------------------------------------------

def rule_based_analysis(pipeline_input: PipelineInput) -> dict:
    """
    Simple rule-based fallback analyzer.
    Used when the LLM call fails after all retries.
    Produces a basic analysis from extracted features alone.
    """
    features   = pipeline_input.features
    input_type = pipeline_input.input_type
    text       = pipeline_input.normalized_text.upper()

    patterns    = []
    severities  = []
    categories  = []

    if input_type == "SQL Query":
        if features.get("has_select_star"):
            patterns.append("SELECT * used — all columns retrieved unnecessarily")
            severities.append("High")
            categories.append("SQL Optimization")
        if features.get("has_correlated_subquery") or features.get("subquery_depth", 0) > 0:
            patterns.append("Subquery detected — possible correlated subquery performance issue")
            severities.append("Critical")
            categories.append("SQL Optimization")
        if not features.get("has_where") and features.get("has_join"):
            patterns.append("JOIN without early WHERE filter — predicate pushdown opportunity")
            severities.append("High")
            categories.append("SQL Optimization")
        if features.get("join_count", 0) > 2:
            patterns.append(f"Multiple JOINs detected ({features['join_count']}) — join order optimization may help")
            severities.append("Medium")
            categories.append("SQL Optimization")

    elif input_type == "ETL Workflow":
        if features.get("has_full_reload") and not features.get("has_incremental"):
            patterns.append("Full reload pattern detected — incremental processing not used")
            severities.append("Critical")
            categories.append("ETL & Pipeline Optimization")
        if features.get("task_count", 0) > 5:
            patterns.append("Large number of tasks — dependency optimization may reduce critical path")
            severities.append("Medium")
            categories.append("ETL & Pipeline Optimization")

    elif input_type == "Pipeline Log":
        if features.get("has_errors"):
            patterns.append(f"Errors detected in log ({features.get('error_count', 0)} error lines)")
            severities.append("High")
            categories.append("Workflow Execution Optimization")
        if features.get("has_oom"):
            patterns.append("Out-of-memory error detected — resource allocation issue")
            severities.append("Critical")
            categories.append("Workflow Execution Optimization")
        if features.get("has_skew"):
            patterns.append("Data skew signals found in log")
            severities.append("High")
            categories.append("ETL & Pipeline Optimization")

    if not patterns:
        patterns   = ["No specific patterns detected by rule-based fallback"]
        severities = ["Low"]
        categories = [input_type.replace(" Query", " Optimization")
                      .replace(" Workflow", " & Pipeline Optimization")
                      .replace(" Log", " Execution Optimization")]

    retrieval_query = (
        f"{input_type} optimization. "
        + " ".join(patterns[:3])
        + " " + " ".join(categories)
    )

    return {
        "input_type":           input_type,
        "detected_patterns":    patterns,
        "severity_signals":     severities,
        "suggested_categories": list(set(categories)),
        "summary":              (
            f"Rule-based analysis of {input_type}. "
            f"Detected {len(patterns)} potential inefficiency pattern(s). "
            f"LLM analysis was unavailable — results may be less precise."
        ),
        "retrieval_query":      retrieval_query,
    }


# ---------------------------------------------------------------------------
# Analyzer Agent
# ---------------------------------------------------------------------------

class AnalyzerAgent:
    """
    Agent 1 — Analyzer Agent.

    Analyzes a PipelineInput and returns a structured AnalyzerOutput
    describing detected inefficiency patterns.
    """

    def __init__(self, llm_client: LLMClient, config: dict = None):
        self.llm     = llm_client
        self.config  = config or {}
        self.retries = self.config.get("agents", {}).get("max_retries", 3)
        self.delay   = self.config.get("agents", {}).get("retry_delay_seconds", 2)

    def run(self, pipeline_input: PipelineInput) -> AnalyzerOutput:
        """
        Main entry point. Analyze the pipeline input and return structured output.

        Args:
            pipeline_input: PipelineInput from the Ingestion module

        Returns:
            AnalyzerOutput with detected patterns and retrieval query
        """
        logger.info(f"[Analyzer Agent] Running on input_type={pipeline_input.input_type}")

        system_prompt = analyzer_system_prompt()
        user_prompt   = analyzer_user_prompt(
            input_type=pipeline_input.input_type,
            normalized_text=pipeline_input.normalized_text,
            features=pipeline_input.features,
        )

        # Attempt LLM call with retry
        raw_response  = ""
        fallback_used = False
        logger.info(f"[Analyzer Agent] System Prompt: {system_prompt}")
        logger.info(f"[Analyzer Agent] User Prompt: {user_prompt}")
        try:
            raw_response = self.llm.complete_with_retry(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_retries=self.retries,
                retry_delay=float(self.delay),
            )
            logger.info(f"[Analyzer Agent] Raw LLM response:\n{raw_response}")

            parsed = extract_json_from_response(raw_response)
            parsed = validate_analyzer_output(parsed)

        except Exception as e:
            logger.warning(
                f"[Analyzer Agent] LLM call failed: {e}. "
                f"Falling back to rule-based analysis."
            )
            parsed        = rule_based_analysis(pipeline_input)
            fallback_used = True

        output = AnalyzerOutput(
            input_type=parsed.get("input_type", pipeline_input.input_type),
            detected_patterns=parsed.get("detected_patterns", []),
            severity_signals=parsed.get("severity_signals", []),
            suggested_categories=parsed.get("suggested_categories", []),
            summary=parsed.get("summary", ""),
            retrieval_query=parsed.get("retrieval_query", pipeline_input.normalized_text),
            raw_llm_response=raw_response,
            fallback_used=fallback_used,
        )
        logger.info(f"[Analyzer Agent] Detected Patterns: {output.detected_patterns}")
        logger.info(
            f"[Analyzer Agent] Complete — "
            f"{len(output.detected_patterns)} patterns detected, "
            f"fallback={fallback_used}"
        )

        return output
