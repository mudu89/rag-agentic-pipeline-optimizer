"""
validate_entries.py
===================
Validates all Layer 1 knowledge base entries against kb_schema_v1.json rules
before embedding and indexing into FAISS.

Run this script after completing Layer 1 curation and before running
build_layer1_index.py. Fix all reported errors before proceeding.

Usage:
    python validate_entries.py
    python validate_entries.py --entries path/to/kb_entries_full_v1.json
    python validate_entries.py --entries kb_entries_full_v1.json --verbose

Author: Mohammed Mudassirullah Sheriff
Project: Intelligent Optimization of Data Pipelines using RAG-Based Knowledge
         Retrieval and Agentic Decision Systems
Institution: BITS Pilani WILP — M.Tech Data Science & Engineering
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Taxonomy: valid category → subcategory mappings
# Must stay in sync with kb_taxonomy_v1.json
# ---------------------------------------------------------------------------
VALID_TAXONOMY = {
    "SQL Optimization": [
        "Column Projection",
        "Predicate Pushdown",
        "Join Optimization",
        "Subquery Optimization",
        "Aggregation Optimization",
        "Query Rewrite Patterns",
        "Filter Optimization",
        "Scan Reduction",
    ],
    "ETL & Pipeline Optimization": [
        "Redundant Transformation Elimination",
        "Task Dependency Optimization",
        "Data Skew Handling",
        "Shuffle Minimization",
        "Pipeline Modularization",
        "Incremental Processing",
    ],
    "Indexing Strategy": [
        "Index Selection",
        "Missing Index Detection",
        "Index Overuse",
        "Columnar & Bloom Filter Indexing",
    ],
    "Partitioning Strategy": [
        "Partition Key Selection",
        "Partition Pruning",
        "Over-Partitioning",
        "Dynamic Partitioning",
    ],
    "Caching Strategy": [
        "Intermediate Result Caching",
        "Materialized View Usage",
        "Broadcast Variable Caching",
        "Query Result Caching",
    ],
    "Workflow Execution Optimization": [
        "Resource Allocation",
        "Retry & Failure Handling",
        "Scheduling Optimization",
        "Bottleneck Detection",
    ],
    "Data Storage Optimization": [
        "File Format Selection",
        "Compression Strategy",
        "Small File Problem",
        "Data Layout Optimization",
    ],
}

VALID_INPUT_TYPES = {"SQL Query", "ETL Workflow", "Pipeline Log", "Metadata"}
VALID_SEVERITIES  = {"Low", "Medium", "High", "Critical"}

# Mandatory fields and their expected Python types
MANDATORY_FIELDS = {
    "entry_id":               str,
    "title":                  str,
    "category":               str,
    "subcategory":            str,
    "problem_description":    str,
    "root_cause":             str,
    "optimization_strategy":  str,
    "before_example":         str,
    "after_example":          str,
    "explanation":            str,
    "expected_impact":        dict,
    "applicability_conditions": list,
    "tags":                   list,
    "platform":               list,
    "input_type":             str,
    "severity":               str,
    "source":                 str,
    "embedding_text":         str,
    "confidence_score":       float,
    "created_date":           str,
}

OPTIONAL_FIELDS = {"related_entries", "notes", "version"}

MIN_CONFIDENCE = 0.70   # Entries below this are flagged as warnings
MIN_STRING_LEN = 10     # Minimum character length for meaningful string fields


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------

def validate_entry(entry: dict, all_entry_ids: set) -> tuple[list, list]:
    """
    Validate a single knowledge base entry.

    Returns:
        errors   — list of strings describing hard failures (entry will be rejected)
        warnings — list of strings describing soft issues (entry is kept but flagged)
    """
    errors   = []
    warnings = []
    eid      = entry.get("entry_id", "<unknown>")

    # 1. Mandatory field presence and type check
    for field, expected_type in MANDATORY_FIELDS.items():
        if field not in entry:
            errors.append(f"Missing mandatory field: '{field}'")
            continue
        value = entry[field]
        # confidence_score may be stored as int in JSON — accept both
        if field == "confidence_score":
            if not isinstance(value, (int, float)):
                errors.append(f"Field '{field}' must be a number, got {type(value).__name__}")
        elif not isinstance(value, expected_type):
            errors.append(
                f"Field '{field}' must be {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )

    # 2. Category must be in taxonomy
    category = entry.get("category", "")
    if category and category not in VALID_TAXONOMY:
        errors.append(
            f"Invalid category '{category}'. "
            f"Valid values: {sorted(VALID_TAXONOMY.keys())}"
        )

    # 3. Subcategory must belong to declared category
    subcategory = entry.get("subcategory", "")
    if category in VALID_TAXONOMY and subcategory:
        if subcategory not in VALID_TAXONOMY[category]:
            errors.append(
                f"Subcategory '{subcategory}' not valid under category '{category}'. "
                f"Valid values: {VALID_TAXONOMY[category]}"
            )

    # 4. input_type enum check
    input_type = entry.get("input_type", "")
    if input_type and input_type not in VALID_INPUT_TYPES:
        errors.append(
            f"Invalid input_type '{input_type}'. "
            f"Valid values: {VALID_INPUT_TYPES}"
        )

    # 5. severity enum check
    severity = entry.get("severity", "")
    if severity and severity not in VALID_SEVERITIES:
        errors.append(
            f"Invalid severity '{severity}'. "
            f"Valid values: {VALID_SEVERITIES}"
        )

    # 6. confidence_score range
    score = entry.get("confidence_score")
    if score is not None:
        score = float(score)
        if not (0.0 <= score <= 1.0):
            errors.append(
                f"confidence_score {score} is out of range. Must be 0.0–1.0"
            )
        elif score < MIN_CONFIDENCE:
            warnings.append(
                f"confidence_score {score} is below recommended minimum "
                f"of {MIN_CONFIDENCE}. Consider revising this entry."
            )

    # 7. Non-empty string checks for key narrative fields
    for field in ["problem_description", "optimization_strategy",
                  "explanation", "embedding_text"]:
        value = entry.get(field, "")
        if isinstance(value, str) and len(value.strip()) < MIN_STRING_LEN:
            errors.append(
                f"Field '{field}' is too short ({len(value.strip())} chars). "
                f"Minimum {MIN_STRING_LEN} characters required."
            )

    # 8. before_example and after_example must not be identical
    before = entry.get("before_example", "")
    after  = entry.get("after_example", "")
    if before and after and before.strip() == after.strip():
        errors.append(
            "before_example and after_example are identical. "
            "The after_example must show the optimized state."
        )

    # 9. Lists must not be empty
    for field in ["tags", "platform", "applicability_conditions"]:
        value = entry.get(field, [])
        if isinstance(value, list) and len(value) == 0:
            errors.append(f"Field '{field}' must not be an empty list.")

    # 10. related_entries (optional) — if present, must reference valid IDs
    related = entry.get("related_entries", [])
    if related:
        for ref_id in related:
            if ref_id == eid:
                errors.append(
                    f"related_entries contains self-reference: '{ref_id}'"
                )
            elif ref_id not in all_entry_ids:
                warnings.append(
                    f"related_entries references '{ref_id}' which was not "
                    f"found in this file. Verify the ID is correct."
                )

    # 11. created_date format check (ISO 8601 YYYY-MM-DD)
    created = entry.get("created_date", "")
    if created:
        try:
            datetime.strptime(created, "%Y-%m-%d")
        except ValueError:
            errors.append(
                f"created_date '{created}' is not valid ISO 8601 format. "
                f"Expected YYYY-MM-DD."
            )

    # 12. Warn if no related_entries cross-reference
    if not related:
        warnings.append(
            "No related_entries defined. "
            "Adding 1–2 cross-references improves recommendation quality."
        )

    return errors, warnings


def validate_all(entries: list, verbose: bool = False) -> dict:
    """
    Run validation across all entries.

    Returns a summary dict with counts and per-entry results.
    """
    # Build a set of all entry_ids for cross-reference checks
    all_entry_ids = {e.get("entry_id") for e in entries if "entry_id" in e}

    # Check for duplicate entry_ids
    seen_ids   = {}
    duplicates = []
    for i, entry in enumerate(entries):
        eid = entry.get("entry_id")
        if eid in seen_ids:
            duplicates.append((eid, seen_ids[eid], i))
        else:
            seen_ids[eid] = i

    results = {
        "total":      len(entries),
        "passed":     0,
        "failed":     0,
        "warned":     0,
        "duplicates": duplicates,
        "entries":    [],
    }

    for entry in entries:
        eid = entry.get("entry_id", "<unknown>")
        entry_errors, entry_warnings = validate_entry(entry, all_entry_ids)

        status = "PASS"
        if entry_errors:
            status = "FAIL"
            results["failed"] += 1
        elif entry_warnings:
            status = "WARN"
            results["warned"] += 1
            results["passed"] += 1
        else:
            results["passed"] += 1

        entry_result = {
            "entry_id": eid,
            "status":   status,
            "errors":   entry_errors,
            "warnings": entry_warnings,
        }
        results["entries"].append(entry_result)

        if verbose or entry_errors:
            _print_entry_result(entry_result)

    return results


def _print_entry_result(result: dict):
    """Print a single entry's validation result."""
    icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}.get(result["status"], "?")
    print(f"\n  {icon} {result['entry_id']}  [{result['status']}]")
    for err in result["errors"]:
        print(f"       ERROR   → {err}")
    for warn in result["warnings"]:
        print(f"       WARNING → {warn}")


def print_summary(results: dict):
    """Print the final validation summary report."""
    total  = results["total"]
    passed = results["passed"]
    failed = results["failed"]
    warned = results["warned"]
    dups   = results["duplicates"]

    print("\n" + "=" * 60)
    print("  VALIDATION REPORT")
    print("=" * 60)
    print(f"  Total entries : {total}")
    print(f"  Passed        : {passed}  (includes {warned} with warnings)")
    print(f"  Failed        : {failed}")

    if dups:
        print(f"\n  DUPLICATE entry_ids detected ({len(dups)}):")
        for eid, first_idx, second_idx in dups:
            print(f"    '{eid}' appears at index {first_idx} and {second_idx}")

    if failed == 0 and not dups:
        print("\n  ✅ All entries passed validation.")
        print("     Safe to proceed with build_layer1_index.py")
    else:
        print(f"\n  ❌ {failed} entries failed. Fix all errors before indexing.")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate Layer 1 knowledge base entries against schema rules."
    )
    parser.add_argument(
        "--entries",
        type=str,
        default="../entries/kb_entries_full_v1.json",
        help="Path to the entries JSON file (default: ../entries/kb_entries_full_v1.json)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print result for every entry, not just failures"
    )
    args = parser.parse_args()

    entries_path = Path(args.entries)
    if not entries_path.exists():
        print(f"\n❌ File not found: {entries_path}")
        print("   Check the --entries path and try again.\n")
        sys.exit(1)

    print(f"\nLoading entries from: {entries_path}")
    with open(entries_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both a plain list and the {"entries": [...]} wrapper format
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict) and "entries" in data:
        entries = data["entries"]
    else:
        print("❌ Unexpected JSON structure. Expected a list or {'entries': [...]}.")
        sys.exit(1)

    print(f"Found {len(entries)} entries. Running validation...\n")

    results = validate_all(entries, verbose=args.verbose)
    print_summary(results)

    # Exit with non-zero code if any entries failed — useful for CI pipelines
    sys.exit(1 if results["failed"] > 0 or results["duplicates"] else 0)


if __name__ == "__main__":
    main()
