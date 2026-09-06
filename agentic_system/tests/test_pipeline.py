"""
test_pipeline.py
================
End-to-end test suite for the Agentic Decision System.

Tests the complete pipeline with four sample inputs — one per input type.
Each test validates that the pipeline produces a structurally correct
output without errors, even when the LLM is unavailable (fallback mode).

Run:
    python tests/test_pipeline.py
    python tests/test_pipeline.py --verbose
    python tests/test_pipeline.py --test sql    # Run only SQL test

Author: Mohammed Mudassirullah Sheriff
Project: Intelligent Optimization of Data Pipelines using RAG-Based
         Knowledge Retrieval and Agentic Decision Systems
Institution: BITS Pilani WILP — M.Tech Data Science & Engineering
"""

import sys
import os
import json
import argparse
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Sample test inputs — one per input type
# ---------------------------------------------------------------------------

TEST_INPUTS = {

    "sql": {
        "label": "SQL Query — Correlated Subquery + SELECT *",
        "input_type_expected": "SQL Query",
        "raw_input": """
SELECT *
FROM orders o
WHERE o.amount > (
    SELECT AVG(amount)
    FROM orders
    WHERE customer_id = o.customer_id
)
AND YEAR(o.created_at) = 2025;
        """.strip(),
    },

    "etl": {
        "label": "ETL Workflow — Full Reload DAG",
        "input_type_expected": "ETL Workflow",
        "raw_input": """
dag_id: customer_data_pipeline
schedule_interval: "0 0 * * *"
default_args:
  retries: 1
  retry_delay_minutes: 5

tasks:
  - task_id: extract_customers
    operator: PythonOperator
    description: Extract all customer records from source

  - task_id: transform_customers
    operator: PythonOperator
    depends_on: extract_customers
    description: Apply transformations and derive features

  - task_id: filter_active
    operator: PythonOperator
    depends_on: transform_customers
    description: Filter only active customers

  - task_id: filter_active_again
    operator: PythonOperator
    depends_on: filter_active
    description: Re-filter active customers for downstream safety

  - task_id: load_customers
    operator: PythonOperator
    depends_on: filter_active_again
    mode: overwrite
    description: Full overwrite of target table every run
        """.strip(),
    },

    "log": {
        "label": "Pipeline Log — Bottleneck + OOM",
        "input_type_expected": "Pipeline Log",
        "raw_input": """
2026-05-14 02:00:01 INFO  Task ingest_raw_events STARTED
2026-05-14 02:04:13 INFO  Task ingest_raw_events SUCCEEDED Duration: 4m 12s
2026-05-14 02:04:14 INFO  Task join_customer_dimension STARTED
2026-05-14 03:59:47 ERROR Task join_customer_dimension FAILED Duration: 1h 55m 33s
2026-05-14 03:59:47 ERROR java.lang.OutOfMemoryError: GC overhead limit exceeded
2026-05-14 03:59:48 WARNING Executor memory usage exceeded threshold (95%)
2026-05-14 04:00:01 INFO  Task join_customer_dimension RETRY attempt 2
2026-05-14 04:03:22 INFO  Task join_customer_dimension SUCCEEDED Duration: 3m 21s
2026-05-14 04:03:23 INFO  Task aggregate_daily_metrics STARTED
2026-05-14 04:21:45 INFO  Task aggregate_daily_metrics SUCCEEDED Duration: 18m 22s
2026-05-14 04:21:46 INFO  Task write_output_table STARTED
2026-05-14 04:47:58 INFO  Task write_output_table SUCCEEDED Duration: 26m 12s
2026-05-14 04:47:59 INFO  Pipeline completed Total Duration: 2h 47m 58s
        """.strip(),
    },

    "metadata": {
        "label": "Metadata — Missing Index on High-Cardinality Column",
        "input_type_expected": "Metadata",
        "raw_input": json.dumps({
            "table_name":   "transactions",
            "database":     "analytics_db",
            "row_count":    48_500_000,
            "num_files":    1240,
            "avg_file_size_mb": 85,
            "storage_format": "CSV",
            "columns": [
                {"column_name": "transaction_id",  "data_type": "STRING",  "null_count": 0,         "cardinality": "HIGH",   "indexed": True},
                {"column_name": "customer_id",     "data_type": "STRING",  "null_count": 0,         "cardinality": "HIGH",   "indexed": False},
                {"column_name": "amount",           "data_type": "DOUBLE",  "null_count": 1200,      "cardinality": "HIGH",   "indexed": False},
                {"column_name": "status",           "data_type": "STRING",  "null_count": 0,         "cardinality": "LOW",    "indexed": False},
                {"column_name": "created_at",       "data_type": "TIMESTAMP","null_count": 0,        "cardinality": "HIGH",   "indexed": False},
                {"column_name": "region",           "data_type": "STRING",  "null_count": 450,       "cardinality": "LOW",    "indexed": False},
            ],
            "statistics": {
                "most_frequent_queries": [
                    "SELECT * FROM transactions WHERE customer_id = ?",
                    "SELECT * FROM transactions WHERE status = 'pending'",
                    "SELECT SUM(amount) FROM transactions WHERE region = ? AND created_at > ?",
                ],
                "avg_query_duration_ms": 12400,
                "full_scan_queries_pct": 78,
            },
            "partitioned": False,
            "partition_column": None,
        }, indent=2),
    },
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_output_structure(result) -> list:
    """
    Check that the pipeline output has all required fields.
    Returns a list of validation errors (empty = pass).
    """
    errors = []

    if not isinstance(result.report_summary, str) or not result.report_summary:
        errors.append("report_summary is empty or not a string")

    if not isinstance(result.total_issues_detected, int):
        errors.append("total_issues_detected is not an integer")

    if not isinstance(result.recommendations, list):
        errors.append("recommendations is not a list")

    if not isinstance(result.pipeline_metadata, dict):
        errors.append("pipeline_metadata is not a dict")

    if not isinstance(result.agent_trace, dict):
        errors.append("agent_trace is not a dict")

    for key in ["analyzer_output", "retrieval_output", "reasoning_output"]:
        if key not in result.agent_trace:
            errors.append(f"agent_trace missing key: {key}")

    for i, rec in enumerate(result.recommendations):
        for field in ["rank", "title", "severity", "problem", "solution"]:
            if field not in rec:
                errors.append(f"recommendation[{i}] missing field: {field}")

    return errors


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_test(
    pipeline: Pipeline,
    test_key: str,
    test_case: dict,
    verbose: bool = False,
) -> dict:
    """Run a single test case and return a result dict."""
    label = test_case["label"]
    print(f"\n{'─' * 60}")
    print(f"  TEST: {label}")
    print(f"{'─' * 60}")

    result_dict = {
        "test":    test_key,
        "label":   label,
        "passed":  False,
        "errors":  [],
        "warnings": [],
    }

    try:
        result = pipeline.run(raw_input=test_case["raw_input"])

        # Validate output structure
        struct_errors = validate_output_structure(result)
        result_dict["errors"].extend(struct_errors)

        # Check input type detection
        detected_type = result.agent_trace.get(
            "analyzer_output", {}
        ).get("input_type", "Unknown")
        expected_type = test_case["input_type_expected"]

        if detected_type != expected_type:
            result_dict["warnings"].append(
                f"Input type: expected '{expected_type}', got '{detected_type}'"
            )

        # Check at least one recommendation was generated
        if len(result.recommendations) == 0:
            result_dict["warnings"].append(
                "No recommendations generated — KB may need more entries"
            )

        # Check fallback usage
        meta = result.pipeline_metadata
        fallbacks = meta.get("fallback_agents", {})
        active_fallbacks = [k for k, v in fallbacks.items() if v]
        if active_fallbacks:
            result_dict["warnings"].append(
                f"Fallback mode used by: {active_fallbacks}"
            )

        result_dict["passed"] = len(result_dict["errors"]) == 0
        result_dict["recommendations_count"] = len(result.recommendations)
        result_dict["input_type_detected"] = detected_type
        result_dict["elapsed_seconds"] = meta.get("elapsed_seconds", 0)

        # Print report
        if verbose:
            print(result.to_text_report())
        else:
            print(f"\n  Summary   : {result.report_summary[:120]}...")
            print(f"  Issues    : {result.total_issues_detected}")
            print(f"  Recs      : {len(result.recommendations)}")
            print(f"  Detected  : {detected_type} (expected: {expected_type})")
            print(f"  Elapsed   : {meta.get('elapsed_seconds', 0)}s")
            for rec in result.recommendations:
                print(f"  [{rec.get('rank')}] [{rec.get('severity')}] {rec.get('title', 'N/A')[:60]}")

    except Exception as e:
        result_dict["errors"].append(f"Pipeline raised exception: {str(e)}")
        if verbose:
            traceback.print_exc()

    # Print pass/fail
    if result_dict["passed"]:
        status = "✅ PASS"
    else:
        status = "❌ FAIL"

    print(f"\n  Status    : {status}")
    for err in result_dict["errors"]:
        print(f"  ERROR     : {err}")
    for warn in result_dict["warnings"]:
        print(f"  WARNING   : {warn}")

    return result_dict


def print_summary(results: list):
    """Print the overall test summary."""
    total  = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    print(f"  Total  : {total}")
    print(f"  Passed : {passed}")
    print(f"  Failed : {failed}")
    print()

    for r in results:
        icon = "✅" if r["passed"] else "❌"
        print(f"  {icon} {r['label']}")

    if failed == 0:
        print("\n  ✅ All tests passed.")
        print("     System is ready for evaluation.")
    else:
        print(f"\n  ❌ {failed} test(s) failed. Review errors above.")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end test suite for the Agentic Decision System."
    )
    parser.add_argument(
        "--test",
        choices=list(TEST_INPUTS.keys()) + ["all"],
        default="all",
        help="Which test to run (default: all)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print full text report for each test"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save test results to JSON file"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Agentic System — End-to-End Test Suite")
    print("=" * 60)

    # Load pipeline once for all tests
    print("\nInitializing pipeline...")
    pipeline = Pipeline(config_path=args.config)
    pipeline.load()

    # Select tests to run
    if args.test == "all":
        tests_to_run = TEST_INPUTS
    else:
        tests_to_run = {args.test: TEST_INPUTS[args.test]}

    # Run tests
    results = []
    for key, case in tests_to_run.items():
        result = run_test(pipeline, key, case, verbose=args.verbose)
        results.append(result)

    # Summary
    print_summary(results)

    # Save results if requested
    if args.output:
        out_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"  Test results saved to: {out_path}\n")

    # Exit code: 0 if all passed, 1 if any failed
    sys.exit(0 if all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    main()
