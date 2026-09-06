"""
query_kb.py
===========
Queries the unified FAISS knowledge base (Layer 1 + Layer 2) and returns
ranked Top-K results with full metadata, similarity scores, and layer source.

This script serves three purposes:
    1. Interactive testing during development
    2. Retrieval quality evaluation (20-query validation suite)
    3. The retrieval function called by the Agentic Decision System at runtime

Usage:
    # Interactive query
    python query_kb.py --query "SELECT * causing slow performance on large table"

    # Run full evaluation suite
    python query_kb.py --evaluate

    # Filter by input type
    python query_kb.py --query "full table reload ETL" --input-type "ETL Workflow"

    # Filter by category
    python query_kb.py --query "join optimization" --category "SQL Optimization"

    # Adjust number of results
    python query_kb.py --query "broadcast join" --top-k 5

Author: Mohammed Mudassirullah Sheriff
Project: Intelligent Optimization of Data Pipelines using RAG-Based Knowledge
         Retrieval and Agentic Decision Systems
Institution: BITS Pilani WILP — M.Tech Data Science & Engineering
"""

import json
import argparse
import time
import csv
from pathlib import Path
from datetime import datetime

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers not installed. "
        "Run: pip install sentence-transformers"
    )
try:
    import faiss
except ImportError:
    raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")




# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_INDEX_PATH = "../index/faiss_index.bin"
DEFAULT_META_PATH  = "../index/metadata_store.json"
DEFAULT_MODEL      = "all-MiniLM-L6-v2"
DEFAULT_TOP_K      = 3
MIN_SIMILARITY     = 0.40    # Results below this threshold are excluded

# Layer preference: when Layer 1 and Layer 2 results are close in score,
# Layer 1 is preferred (higher weight multiplier)
LAYER1_SCORE_BONUS = 0.05    # Added to Layer 1 scores during reranking


# ---------------------------------------------------------------------------
# Evaluation suite — 20 queries covering all 7 taxonomy categories
# ---------------------------------------------------------------------------
EVALUATION_QUERIES = [
    # SQL Optimization (6 queries)
    {
        "query": "SELECT * causing unnecessary I/O on large table queries",
        "expected_id": "KB-SQL-001",
        "expected_category": "SQL Optimization",
        "input_type": "SQL Query",
    },
    {
        "query": "broadcast join small dimension table avoid shuffle Spark",
        "expected_id": "KB-SQL-002",
        "expected_category": "SQL Optimization",
        "input_type": "SQL Query",
    },
    {
        "query": "WHERE filter applied after join slow query performance",
        "expected_id": "KB-SQL-003",
        "expected_category": "SQL Optimization",
        "input_type": "SQL Query",
    },
    {
        "query": "correlated subquery executes once per row repeated scan",
        "expected_id": "KB-SQL-004",
        "expected_category": "SQL Optimization",
        "input_type": "SQL Query",
    },
    {
        "query": "query scanning all partitions despite date filter",
        "expected_id": "KB-PAR-001",
        "expected_category": "Partitioning Strategy",
        "input_type": "SQL Query",
    },
    {
        "query": "UNION ALL vs UNION query rewrite performance difference",
        "expected_id": None,   # No specific entry — tests category retrieval
        "expected_category": "SQL Optimization",
        "input_type": "SQL Query",
    },
    # ETL & Pipeline Optimization (4 queries)
    {
        "query": "same filter condition repeated multiple times in ETL pipeline",
        "expected_id": "KB-ETL-001",
        "expected_category": "ETL & Pipeline Optimization",
        "input_type": "ETL Workflow",
    },
    {
        "query": "full reload entire dataset every pipeline run too expensive",
        "expected_id": "KB-ETL-002",
        "expected_category": "ETL & Pipeline Optimization",
        "input_type": "ETL Workflow",
    },
    {
        "query": "delta processing watermark incremental load pattern",
        "expected_id": "KB-ETL-002",
        "expected_category": "ETL & Pipeline Optimization",
        "input_type": "ETL Workflow",
    },
    {
        "query": "data skew uneven partition sizes slow Spark tasks",
        "expected_id": None,
        "expected_category": "ETL & Pipeline Optimization",
        "input_type": "ETL Workflow",
    },
    # Indexing Strategy (2 queries)
    {
        "query": "query doing full table scan on frequently filtered column",
        "expected_id": "KB-IDX-001",
        "expected_category": "Indexing Strategy",
        "input_type": "Metadata",
    },
    {
        "query": "too many indexes slowing down INSERT and UPDATE operations",
        "expected_id": None,
        "expected_category": "Indexing Strategy",
        "input_type": "Metadata",
    },
    # Partitioning Strategy (2 queries)
    {
        "query": "partition pruning not working query scanning all data",
        "expected_id": "KB-PAR-001",
        "expected_category": "Partitioning Strategy",
        "input_type": "SQL Query",
    },
    {
        "query": "too many small files in distributed storage small partition problem",
        "expected_id": None,
        "expected_category": "Partitioning Strategy",
        "input_type": "Metadata",
    },
    # Caching Strategy (2 queries)
    {
        "query": "small lookup table joined multiple times in Spark pipeline",
        "expected_id": "KB-CAC-001",
        "expected_category": "Caching Strategy",
        "input_type": "ETL Workflow",
    },
    {
        "query": "materialized view repeated aggregation query performance",
        "expected_id": None,
        "expected_category": "Caching Strategy",
        "input_type": "SQL Query",
    },
    # Workflow Execution Optimization (2 queries)
    {
        "query": "one pipeline task taking much longer than others bottleneck",
        "expected_id": "KB-WRK-001",
        "expected_category": "Workflow Execution Optimization",
        "input_type": "Pipeline Log",
    },
    {
        "query": "Airflow DAG all tasks scheduled at midnight resource contention",
        "expected_id": None,
        "expected_category": "Workflow Execution Optimization",
        "input_type": "ETL Workflow",
    },
    # Data Storage Optimization (2 queries)
    {
        "query": "CSV files slow query performance should use Parquet or ORC",
        "expected_id": None,
        "expected_category": "Data Storage Optimization",
        "input_type": "Metadata",
    },
    {
        "query": "many small output files coalesce repartition compaction strategy",
        "expected_id": None,
        "expected_category": "Data Storage Optimization",
        "input_type": "Metadata",
    },
]


# ---------------------------------------------------------------------------
# Knowledge Base class
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """
    Unified knowledge base query interface.

    Wraps FAISS index + metadata store + embedding model into a single
    queryable object. This class is designed to be imported directly
    by the Agentic Decision System pipeline.
    """

    def __init__(
        self,
        index_path: str = DEFAULT_INDEX_PATH,
        meta_path: str  = DEFAULT_META_PATH,
        model_name: str = DEFAULT_MODEL,
    ):
        self.index_path = Path(index_path)
        self.meta_path  = Path(meta_path)
        self.model_name = model_name
        self.index      = None
        self.metadata   = {}
        self.model      = None
        self._loaded    = False

    def load(self):
        """Load FAISS index, metadata store, and embedding model into memory."""
        print(f"\nLoading knowledge base...")

        # Load FAISS index
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {self.index_path}\n"
                f"Run merge_indexes.py first (or build_layer1_index.py for Layer 1 only)."
            )
        self.index = faiss.read_index(str(self.index_path))
        print(f"  ✅ FAISS index loaded     — {self.index.ntotal} vectors, dim={self.index.d}")

        # Load metadata store
        if not self.meta_path.exists():
            raise FileNotFoundError(
                f"Metadata store not found: {self.meta_path}"
            )
        with open(self.meta_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.metadata = raw.get("entries", raw)
        print(f"  ✅ Metadata store loaded  — {len(self.metadata)} entries")

        # Load embedding model
        print(f"  Loading embedding model: {self.model_name} ...")
        self.model  = SentenceTransformer(self.model_name)
        self._loaded = True
        print(f"  ✅ Model loaded           — dim={self.model.get_sentence_embedding_dimension()}")
        print(f"  Knowledge base ready.\n")

    def query(
        self,
        query_text: str,
        top_k: int = DEFAULT_TOP_K,
        input_type_filter: str = None,
        category_filter: str = None,
        min_similarity: float = MIN_SIMILARITY,
        prefer_layer1: bool = True,
    ) -> list:
        """
        Query the knowledge base and return ranked results.

        Args:
            query_text         : Natural language or code query
            top_k              : Number of results to return
            input_type_filter  : Optional — filter by input_type enum value
            category_filter    : Optional — filter by category name
            min_similarity     : Exclude results below this cosine similarity
            prefer_layer1      : Apply score bonus to Layer 1 entries

        Returns:
            List of result dicts sorted by adjusted similarity score (descending)
        """
        if not self._loaded:
            self.load()

        # Encode and normalize query
        query_vec = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # Search FAISS — retrieve more than top_k to allow for post-filtering
        search_k  = min(top_k * 5, self.index.ntotal)
        scores, positions = self.index.search(query_vec, search_k)

        # Build result list with metadata
        results = []
        for score, pos in zip(scores[0], positions[0]):
            if pos < 0:
                continue   # FAISS returns -1 for empty slots

            meta = self.metadata.get(str(pos), {})
            if not meta:
                continue

            similarity = float(score)

            # Apply minimum similarity threshold
            if similarity < min_similarity:
                continue

            # Apply metadata filters
            if input_type_filter:
                if meta.get("input_type", "") != input_type_filter:
                    continue
            if category_filter:
                if meta.get("category", "") != category_filter:
                    continue

            # Apply Layer 1 preference bonus for reranking
            adjusted_score = similarity
            if prefer_layer1 and meta.get("layer") == 1:
                adjusted_score += LAYER1_SCORE_BONUS

            # Apply confidence score weighting for Layer 1 entries
            if meta.get("layer") == 1:
                confidence = float(meta.get("confidence_score", 1.0))
                adjusted_score *= confidence

            results.append({
                "rank":             len(results) + 1,
                "entry_id":         meta.get("entry_id") or meta.get("chunk_id", "?"),
                "title":            meta.get("title") or meta.get("chunk_id", "Untitled"),
                "category":         meta.get("category", "Unknown"),
                "subcategory":      meta.get("subcategory", "Unknown"),
                "input_type":       meta.get("input_type", "Unknown"),
                "layer":            meta.get("layer", "?"),
                "similarity_score": round(similarity, 4),
                "adjusted_score":   round(adjusted_score, 4),
                "severity":         meta.get("severity", "N/A"),
                "confidence_score": meta.get("confidence_score", "N/A"),
                # Display fields
                "problem_description":   meta.get("problem_description") or meta.get("text", "")[:200],
                "optimization_strategy": meta.get("optimization_strategy", "N/A"),
                "explanation":           meta.get("explanation", "N/A"),
                "before_example":        meta.get("before_example", "N/A"),
                "after_example":         meta.get("after_example", "N/A"),
                "expected_impact":       meta.get("expected_impact", {}),
                "source":                meta.get("source") or meta.get("source_document", "N/A"),
                "related_entries":       meta.get("related_entries", []),
                "faiss_position":        pos,
            })

        # Sort by adjusted score descending
        results.sort(key=lambda x: x["adjusted_score"], reverse=True)

        # Assign final ranks and trim to top_k
        for i, r in enumerate(results[:top_k]):
            r["rank"] = i + 1

        return results[:top_k]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_query_results(query_text: str, results: list, verbose: bool = False):
    """Print query results in a readable format."""
    print("\n" + "=" * 70)
    print(f"  QUERY: {query_text}")
    print("=" * 70)

    if not results:
        print("  No results found above similarity threshold.")
        print("  Try lowering --min-similarity or removing filters.\n")
        return

    for r in results:
        layer_tag = f"[L{r['layer']}]" if r["layer"] in (1, 2) else "[L?]"
        sev_tag   = f"[{r['severity']}]" if r["severity"] != "N/A" else ""
        print(f"\n  Rank {r['rank']} {layer_tag} {sev_tag}")
        print(f"  ID       : {r['entry_id']}")
        print(f"  Title    : {r['title']}")
        print(f"  Category : {r['category']} → {r['subcategory']}")
        print(f"  Scores   : similarity={r['similarity_score']}  "
              f"adjusted={r['adjusted_score']}")

        if verbose:
            print(f"\n  Problem  : {r['problem_description'][:300]}")
            print(f"\n  Strategy : {r['optimization_strategy'][:300]}")
            if r["before_example"] != "N/A":
                print(f"\n  Before   :\n    {r['before_example'][:200]}")
                print(f"\n  After    :\n    {r['after_example'][:200]}")
            print(f"\n  Source   : {r['source']}")

        print(f"  {'-' * 66}")

    print()


# ---------------------------------------------------------------------------
# Evaluation suite
# ---------------------------------------------------------------------------

def run_evaluation(kb: KnowledgeBase, top_k: int = 3, output_csv: str = None):
    """
    Run the 20-query evaluation suite and compute retrieval quality metrics.

    Metrics computed:
        Top-1 Accuracy    : Correct entry at rank 1
        Top-3 Accuracy    : Correct entry in Top 3
        Category Accuracy : Correct category at rank 1 (for queries with no expected_id)
        Mean Similarity   : Average cosine similarity of Top-1 results
        Layer 1 Rate      : Fraction of Top-1 results from Layer 1
    """
    print("\n" + "=" * 70)
    print("  RETRIEVAL EVALUATION — 20 Query Validation Suite")
    print("=" * 70)

    eval_rows = []
    top1_correct       = 0
    topk_correct       = 0
    category_correct   = 0
    similarity_sum     = 0.0
    layer1_top1_count  = 0
    queries_with_id    = 0

    for i, q in enumerate(EVALUATION_QUERIES):
        results = kb.query(
            query_text=q["query"],
            top_k=top_k,
            input_type_filter=q.get("input_type"),
        )

        top1_id       = results[0]["entry_id"] if results else None
        top1_category = results[0]["category"] if results else None
        top1_score    = results[0]["similarity_score"] if results else 0.0
        top1_layer    = results[0]["layer"] if results else None
        topk_ids      = [r["entry_id"] for r in results]

        expected_id  = q.get("expected_id")
        expected_cat = q.get("expected_category")

        # Top-1 and Top-K accuracy (only for queries with a specific expected_id)
        top1_hit = False
        topk_hit = False
        if expected_id:
            queries_with_id += 1
            top1_hit = (top1_id == expected_id)
            topk_hit = (expected_id in topk_ids)
            if top1_hit:
                top1_correct += 1
            if topk_hit:
                topk_correct += 1

        # Category accuracy (all queries)
        cat_hit = (top1_category == expected_cat)
        if cat_hit:
            category_correct += 1

        similarity_sum += top1_score
        if top1_layer == 1:
            layer1_top1_count += 1

        status = "✅" if (top1_hit or (not expected_id and cat_hit)) else "❌"
        print(f"\n  Q{i+1:02d} {status}  {q['query'][:60]}...")
        print(f"       Expected  : {expected_id or f'[{expected_cat}]'}")
        print(f"       Got Top-1 : {top1_id}  [{top1_category}]  "
              f"score={top1_score:.4f}  layer={top1_layer}")
        if len(results) > 1:
            print(f"       Top-{top_k}     : {topk_ids}")

        eval_rows.append({
            "query_num":        i + 1,
            "query":            q["query"],
            "expected_id":      expected_id or "",
            "expected_category":expected_cat,
            "top1_id":          top1_id or "",
            "top1_category":    top1_category or "",
            "top1_score":       top1_score,
            "top1_layer":       top1_layer or "",
            "topk_ids":         "|".join(topk_ids),
            "top1_correct":     top1_hit,
            "topk_correct":     topk_hit,
            "category_correct": cat_hit,
        })

    # Compute metrics
    n_total = len(EVALUATION_QUERIES)
    top1_acc   = top1_correct / queries_with_id if queries_with_id > 0 else 0.0
    topk_acc   = topk_correct / queries_with_id if queries_with_id > 0 else 0.0
    cat_acc    = category_correct / n_total
    mean_sim   = similarity_sum / n_total
    layer1_rate = layer1_top1_count / n_total

    print("\n" + "=" * 70)
    print("  EVALUATION METRICS")
    print("=" * 70)
    print(f"  Queries evaluated          : {n_total}")
    print(f"  Queries with expected_id   : {queries_with_id}")
    print(f"")
    print(f"  Top-1 Accuracy             : {top1_acc:.1%}  "
          f"(target ≥ 70%)  {'✅' if top1_acc >= 0.70 else '❌'}")
    print(f"  Top-{top_k} Accuracy             : {topk_acc:.1%}  "
          f"(target ≥ 85%)  {'✅' if topk_acc >= 0.85 else '❌'}")
    print(f"  Category Accuracy (Top-1)  : {cat_acc:.1%}")
    print(f"  Mean Similarity (Top-1)    : {mean_sim:.4f}  "
          f"(target ≥ 0.65)  {'✅' if mean_sim >= 0.65 else '❌'}")
    print(f"  Layer 1 Top-1 Rate         : {layer1_rate:.1%}  "
          f"(target ≥ 60%)  {'✅' if layer1_rate >= 0.60 else '❌'}")
    print("=" * 70)

    # Save CSV if requested
    if output_csv:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=eval_rows[0].keys())
            writer.writeheader()
            writer.writerows(eval_rows)
        print(f"\n  📄 Evaluation results saved → {csv_path}")
        print("     Include this CSV as Appendix in your dissertation.\n")

    return {
        "top1_accuracy":    top1_acc,
        "topk_accuracy":    topk_acc,
        "category_accuracy":cat_acc,
        "mean_similarity":  mean_sim,
        "layer1_rate":      layer1_rate,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Query the unified knowledge base and retrieve Top-K results."
    )
    parser.add_argument(
        "--query",       type=str, default=None,
        help="Query string for single retrieval"
    )
    parser.add_argument(
        "--top-k",       type=int, default=DEFAULT_TOP_K,
        help=f"Number of results to return (default: {DEFAULT_TOP_K})"
    )
    parser.add_argument(
        "--input-type",  type=str, default=None,
        help="Filter results by input_type (SQL Query / ETL Workflow / Pipeline Log / Metadata)"
    )
    parser.add_argument(
        "--category",    type=str, default=None,
        help="Filter results by category name"
    )
    parser.add_argument(
        "--min-similarity", type=float, default=MIN_SIMILARITY,
        help=f"Minimum cosine similarity threshold (default: {MIN_SIMILARITY})"
    )
    parser.add_argument(
        "--index",       type=str, default=DEFAULT_INDEX_PATH,
        help=f"Path to FAISS index (default: {DEFAULT_INDEX_PATH})"
    )
    parser.add_argument(
        "--metadata",    type=str, default=DEFAULT_META_PATH,
        help=f"Path to metadata store (default: {DEFAULT_META_PATH})"
    )
    parser.add_argument(
        "--model",       type=str, default=DEFAULT_MODEL,
        help=f"Embedding model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--evaluate",    action="store_true",
        help="Run the full 20-query evaluation suite"
    )
    parser.add_argument(
        "--eval-csv",    type=str, default=None,
        help="Path to save evaluation results CSV (used with --evaluate)"
    )
    parser.add_argument(
        "--verbose",     action="store_true",
        help="Show full entry details in results"
    )
    args = parser.parse_args()

    # Initialize and load knowledge base
    kb = KnowledgeBase(
        index_path=args.index,
        meta_path=args.metadata,
        model_name=args.model,
    )
    kb.load()

    if args.evaluate:
        # Run full evaluation suite
        run_evaluation(kb, top_k=args.top_k, output_csv=args.eval_csv)

    elif args.query:
        # Single query mode
        start   = time.time()
        results = kb.query(
            query_text=args.query,
            top_k=args.top_k,
            input_type_filter=args.input_type,
            category_filter=args.category,
            min_similarity=args.min_similarity,
        )
        elapsed = time.time() - start
        print_query_results(args.query, results, verbose=args.verbose)
        print(f"  Retrieval time: {elapsed*1000:.1f}ms\n")

    else:
        # Interactive mode
        print("\nEntering interactive query mode. Type 'quit' to exit.\n")
        while True:
            try:
                query = input("Query > ").strip()
                if query.lower() in ("quit", "exit", "q"):
                    break
                if not query:
                    continue
                start   = time.time()
                results = kb.query(query, top_k=args.top_k)
                elapsed = time.time() - start
                print_query_results(query, results, verbose=args.verbose)
                print(f"  Retrieval time: {elapsed*1000:.1f}ms\n")
            except KeyboardInterrupt:
                break
        print("\nExiting.\n")


if __name__ == "__main__":
    main()
