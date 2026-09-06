"""
run_pipeline.py
===============
CLI entry point for the Agentic Decision System.

Usage examples:
    # Analyze a raw SQL query
    python run_pipeline.py --query "SELECT * FROM orders WHERE status = 'active'"

    # Analyze a file
    python run_pipeline.py --file path/to/query.sql

    # Show full agent trace
    python run_pipeline.py --file query.sql --trace

    # Save output to JSON file
    python run_pipeline.py --file query.sql --output results/output.json

    # Use a custom config file
    python run_pipeline.py --query "..." --config path/to/config.yaml

Author: Mohammed Mudassirullah Sheriff
Project: Intelligent Optimization of Data Pipelines using RAG-Based
         Knowledge Retrieval and Agentic Decision Systems
Institution: BITS Pilani WILP — M.Tech Data Science & Engineering
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.pipeline import Pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Intelligent Data Pipeline Optimizer — "
            "RAG-Based Knowledge Retrieval and Agentic Decision Systems"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --query "SELECT * FROM large_table WHERE id = 1"
  python run_pipeline.py --file ../tests/test_inputs/sample_query.sql
  python run_pipeline.py --file ../tests/test_inputs/sample_dag.yaml --trace
  python run_pipeline.py --query "..." --output results/output.json
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--query", "-q",
        type=str,
        help="Raw input text (SQL query, ETL YAML, pipeline log)"
    )
    input_group.add_argument(
        "--file", "-f",
        type=str,
        help="Path to input file (.sql, .yaml, .log, .json)"
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config.yaml (default: agentic_system/config/config.yaml)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Save full JSON output to this file path"
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print full agent trace alongside the report"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON output instead of formatted text report"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = Pipeline(config_path=args.config)
    pipeline.load()

    # Get input
    raw_input = ""
    file_path = None

    if args.query:
        raw_input = args.query
    elif args.file:
        file_path = args.file
        path = Path(file_path)
        if not path.exists():
            print(f"\n❌ File not found: {file_path}")
            sys.exit(1)
        raw_input = path.read_text(encoding="utf-8")
        print(f"Loaded input from: {file_path}")

    # Run pipeline
    print("Running pipeline...\n")
    result = pipeline.run(raw_input=raw_input, file_path=file_path)

    # Display output
    if args.json:
        print(json.dumps(result.to_dict(include_trace=args.trace), indent=2))
    else:
        print(result.to_text_report())

        if args.trace:
            print("\n" + "=" * 70)
            print("  AGENT TRACE")
            print("=" * 70)
            print(json.dumps(result.agent_trace, indent=2))

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(include_trace=True), f, indent=2, ensure_ascii=False)
        print(f"\n✅ Full output saved to: {output_path}")


if __name__ == "__main__":
    main()
