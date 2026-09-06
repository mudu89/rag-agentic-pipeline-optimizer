"""
prompts.py
==========
All LLM prompt templates for the four agents.

Each prompt is a function that accepts runtime values and returns
(system_prompt, user_prompt) tuples ready to pass to LLMClient.

Design principles:
    - Every agent prompt instructs the LLM to return ONLY valid JSON
    - Schema is embedded in the prompt so the model knows exactly
      what structure to produce
    - Low temperature (0.2) + explicit schema = consistent, parseable output

Author: Mohammed Mudassirullah Sheriff
Project: Intelligent Optimization of Data Pipelines using RAG-Based
         Knowledge Retrieval and Agentic Decision Systems
Institution: BITS Pilani WILP — M.Tech Data Science & Engineering
"""


# ---------------------------------------------------------------------------
# Agent 1 — Analyzer Agent prompts
# ---------------------------------------------------------------------------

def analyzer_system_prompt() -> str:
    return """You are a senior data engineering analyst specializing in SQL query optimization, ETL pipeline efficiency, and execution log analysis.

Your job is to analyze the provided input and identify data pipeline inefficiency patterns.

Rules:
- Do NOT recommend solutions. Only identify problems.
- Respond ONLY with a valid JSON object. No preamble, no explanation, no markdown fences.
- Be specific and technical. Vague observations are not useful.
- Base your analysis strictly on what is visible in the input."""


def analyzer_user_prompt(input_type: str, normalized_text: str, features: dict) -> str:
    features_str = "\n".join(f"  {k}: {v}" for k, v in features.items())

    return f"""Analyze this {input_type} for data pipeline inefficiencies.

INPUT TYPE: {input_type}

EXTRACTED FEATURES:
{features_str}

INPUT CONTENT:
{normalized_text}

Respond with this exact JSON structure:
{{
  "input_type": "{input_type}",
  "detected_patterns": [
    "List each specific inefficiency pattern detected — be concrete and technical"
  ],
  "severity_signals": [
    "Severity for each pattern above — one of: Low | Medium | High | Critical"
  ],
  "suggested_categories": [
    "Taxonomy categories that apply — choose from: SQL Optimization | ETL & Pipeline Optimization | Indexing Strategy | Partitioning Strategy | Caching Strategy | Workflow Execution Optimization | Data Storage Optimization"
  ],
  "summary": "2-3 sentence technical summary of the overall inefficiency profile",
  "retrieval_query": "A single search query string (50-100 words) that captures all detected issues — used to retrieve relevant knowledge base entries"
}}"""


# ---------------------------------------------------------------------------
# Agent 2 — Retrieval Agent prompts
# ---------------------------------------------------------------------------

def retrieval_system_prompt() -> str:
    return """You are a knowledge retrieval specialist for a data pipeline optimization system.

You are given a set of retrieved knowledge base entries and must select the most relevant ones for the detected inefficiency patterns.

Rules:
- Select a maximum of 3 entries from the provided list.
- Explain why each selected entry is relevant to the specific input.
- Flag any entry that may NOT apply to this case.
- Respond ONLY with a valid JSON object. No preamble, no markdown fences."""


def retrieval_user_prompt(
    analyzer_output: dict,
    retrieved_entries: list,
) -> str:
    # Format retrieved entries for the prompt
    entries_text = ""
    for i, entry in enumerate(retrieved_entries):
        entry_id    = entry.get("entry_id") or entry.get("chunk_id", f"ENTRY-{i}")
        title       = entry.get("title", "Untitled")
        category    = entry.get("category", "Unknown")
        subcategory = entry.get("subcategory", "Unknown")
        problem     = entry.get("problem_description") or entry.get("text", "")[:200]
        strategy    = entry.get("optimization_strategy", "N/A")
        score       = entry.get("similarity_score", 0.0)
        layer       = entry.get("layer", "?")
        conditions  = entry.get("applicability_conditions", [])

        entries_text += f"""
Entry {i+1}:
  ID          : {entry_id}
  Layer       : {layer}
  Category    : {category} → {subcategory}
  Similarity  : {score}
  Title       : {title}
  Problem     : {problem[:300]}
  Strategy    : {strategy[:300]}
  Conditions  : {conditions}
---"""

    patterns_str = "\n".join(
        f"  - {p}" for p in analyzer_output.get("detected_patterns", [])
    )

    return f"""The Analyzer Agent detected these inefficiency patterns:
{patterns_str}

Here are the retrieved knowledge base entries (ranked by similarity):
{entries_text}

Select the most relevant entries and explain your selection.

Respond with this exact JSON structure:
{{
  "selected_entries": [
    {{
      "entry_id": "ID of the selected entry",
      "title": "Title of the entry",
      "relevance_reason": "Why this entry directly applies to the detected patterns",
      "applicable": true,
      "layer": 1
    }}
  ],
  "rejected_entries": [
    {{
      "entry_id": "ID of the rejected entry",
      "rejection_reason": "Why this entry does not apply to this specific input"
    }}
  ],
  "retrieval_summary": "1-2 sentences summarizing what was found and why it is relevant"
}}"""


# ---------------------------------------------------------------------------
# Agent 3 — Reasoning Agent prompts
# ---------------------------------------------------------------------------

def reasoning_system_prompt() -> str:
    return """You are a senior data engineering optimization expert and AI reasoning specialist.

You are given a detected inefficiency analysis and a set of curated knowledge base entries. Your job is to reason about which optimizations apply, why, and in what priority order.

Rules:
- Show your reasoning step by step.
- Consider applicability conditions — do not recommend an optimization if the conditions are not met by the input.
- Rank recommendations by expected impact (Critical issues first, then High, Medium, Low).
- Be specific — reference the actual patterns detected and the actual KB entries.
- Respond ONLY with a valid JSON object. No preamble, no markdown fences."""


def reasoning_user_prompt(
    analyzer_output: dict,
    retrieval_output: dict,
    kb_entries_full: list,
) -> str:
    # Build full entry context for selected entries
    selected_ids = {
        e["entry_id"] for e in retrieval_output.get("selected_entries", [])
    }
    context_blocks = []
    for entry in kb_entries_full:
        eid = entry.get("entry_id") or entry.get("chunk_id", "")
        if eid not in selected_ids:
            continue

        context_blocks.append(f"""
[{eid}]
Title              : {entry.get('title', 'N/A')}
Category           : {entry.get('category', 'N/A')} → {entry.get('subcategory', 'N/A')}
Severity           : {entry.get('severity', 'N/A')}
Problem            : {entry.get('problem_description', 'N/A')}
Optimization       : {entry.get('optimization_strategy', 'N/A')}
Conditions         : {entry.get('applicability_conditions', [])}
Expected Impact    : {entry.get('expected_impact', {})}
Confidence Score   : {entry.get('confidence_score', 'N/A')}
Before             : {entry.get('before_example', 'N/A')}
After              : {entry.get('after_example', 'N/A')}
""")

    context_str   = "\n---".join(context_blocks) if context_blocks else "No entries selected."
    patterns_str  = "\n".join(f"  [{s}] {p}" for p, s in zip(
        analyzer_output.get("detected_patterns", []),
        analyzer_output.get("severity_signals", []),
    ))
    summary = analyzer_output.get("summary", "")

    return f"""INPUT ANALYSIS SUMMARY:
{summary}

DETECTED PATTERNS (with severity):
{patterns_str}

SELECTED KNOWLEDGE BASE ENTRIES:
{context_str}

RETRIEVAL SUMMARY:
{retrieval_output.get('retrieval_summary', 'N/A')}

Now reason about which optimizations to recommend and in what order.

Respond with this exact JSON structure:
{{
  "reasoning_steps": [
    "Step 1: [Explain your first reasoning step — reference specific patterns and KB entries]",
    "Step 2: [Continue reasoning...] ",
    "Step 3: [Final ranking decision and any trade-off considerations]"
  ],
  "ranked_recommendations": [
    {{
      "rank": 1,
      "entry_id": "KB entry ID",
      "title": "Optimization title",
      "severity": "Critical | High | Medium | Low",
      "rationale": "Why this is ranked first — which detected pattern does it address",
      "applicable": true,
      "trade_offs": "Any downsides or caveats to applying this optimization"
    }}
  ],
  "inapplicable_entries": [
    {{
      "entry_id": "KB entry ID",
      "reason": "Why this does not apply — which applicability condition is not met"
    }}
  ],
  "overall_assessment": "2-3 sentence overall evaluation of the input's optimization potential"
}}"""


# ---------------------------------------------------------------------------
# Agent 4 — Recommendation Agent prompts
# ---------------------------------------------------------------------------

def recommendation_system_prompt() -> str:
    return """You are a technical documentation expert and data engineering optimization advisor.

Your job is to convert a structured optimization reasoning output into a clear, actionable, and well-explained recommendation report.

The report must be:
- Professional and technically precise
- Readable by a data engineer without deep AI knowledge
- Specific to the actual input that was analyzed
- Grounded in the retrieved knowledge base evidence

Rules:
- Respond ONLY with a valid JSON object. No preamble, no markdown fences.
- Do not invent optimizations not supported by the reasoning output.
- Include concrete before/after code examples where available."""


def recommendation_user_prompt(
    analyzer_output: dict,
    retrieval_output: dict,
    reasoning_output: dict,
    kb_entries_full: list,
    max_recommendations: int = 3,
) -> str:
    # Build a lookup for KB entry details
    entry_lookup = {}
    for entry in kb_entries_full:
        eid = entry.get("entry_id") or entry.get("chunk_id", "")
        if eid:
            entry_lookup[eid] = entry

    # Collect ranked recommendations with full KB detail
    recs_text = ""
    ranked = reasoning_output.get("ranked_recommendations", [])[:max_recommendations]
    for rec in ranked:
        eid   = rec.get("entry_id", "")
        entry = entry_lookup.get(eid, {})
        recs_text += f"""
Rank {rec.get('rank')}: {rec.get('title', 'N/A')}
  Entry ID     : {eid}
  Severity     : {rec.get('severity', 'N/A')}
  Rationale    : {rec.get('rationale', 'N/A')}
  Trade-offs   : {rec.get('trade_offs', 'None')}
  Problem      : {entry.get('problem_description', 'N/A')}
  Strategy     : {entry.get('optimization_strategy', 'N/A')}
  Explanation  : {entry.get('explanation', 'N/A')}
  Before       : {entry.get('before_example', 'N/A')}
  After        : {entry.get('after_example', 'N/A')}
  Impact       : {entry.get('expected_impact', {})}
  Source       : {entry.get('source', 'N/A')}
  Confidence   : {entry.get('confidence_score', 'N/A')}
---"""

    return f"""Generate the final optimization recommendation report.

ANALYSIS SUMMARY:
{analyzer_output.get('summary', 'N/A')}

OVERALL ASSESSMENT:
{reasoning_output.get('overall_assessment', 'N/A')}

RANKED OPTIMIZATIONS TO REPORT ON:
{recs_text}

Respond with this exact JSON structure:
{{
  "report_summary": "2-3 sentence executive summary of what was found and what to do",
  "total_issues_detected": {len(analyzer_output.get('detected_patterns', []))},
  "recommendations": [
    {{
      "rank": 1,
      "entry_id": "KB entry ID",
      "title": "Optimization title — clear and action-oriented",
      "severity": "Critical | High | Medium | Low",
      "problem": "Plain English description of the problem detected",
      "solution": "Step-by-step or narrative description of what to do",
      "before_example": "Code snippet showing the current inefficient pattern",
      "after_example": "Code snippet showing the optimized pattern",
      "explanation": "Why this optimization works — technical justification",
      "expected_impact": {{
        "execution_time": "Estimated improvement",
        "cost": "Estimated cost impact",
        "memory": "Estimated memory impact"
      }},
      "source": "Source of this optimization knowledge",
      "confidence": 0.95
    }}
  ],
  "additional_notes": "Any caveats, prerequisites, or follow-up recommendations"
}}"""
