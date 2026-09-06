"""
build_layer1_index.py
=====================
Loads validated Layer 1 knowledge base entries, generates embeddings using
a HuggingFace sentence-transformer model, and builds a FAISS index.

Outputs:
    faiss_index_layer1.bin      — Serialized FAISS index
    metadata_store_layer1.json  — Maps FAISS vector position → full entry

Prerequisites:
    pip install faiss-cpu sentence-transformers torch

Run validate_entries.py and confirm zero failures before running this script.

Usage:
    python build_layer1_index.py
    python build_layer1_index.py --entries path/to/kb_entries_full_v1.json
    python build_layer1_index.py --model all-mpnet-base-v2

Author: Mohammed Mudassirullah Sheriff
Project: Intelligent Optimization of Data Pipelines using RAG-Based Knowledge
         Retrieval and Agentic Decision Systems
Institution: BITS Pilani WILP — M.Tech Data Science & Engineering
"""

import json
import argparse
import time
import numpy as np
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers is not installed. "
        "Run: pip install sentence-transformers"
    )

try:
    import faiss
except ImportError:
    raise ImportError(
        "faiss-cpu is not installed. Run: pip install faiss-cpu"
    )


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_ENTRIES_PATH  = "../entries/kb_entries_full_v1.json"
DEFAULT_OUTPUT_DIR    = "../index"
DEFAULT_MODEL         = "all-MiniLM-L6-v2"   # 384-dim; fast and accurate
BATCH_SIZE            = 32                    # Entries per embedding batch


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_entries(entries_path: Path) -> list:
    """Load and return the list of knowledge base entries from JSON."""
    print(f"\n[1/5] Loading entries from: {entries_path}")
    with open(entries_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict) and "entries" in data:
        entries = data["entries"]
    else:
        raise ValueError(
            "Unexpected JSON structure. Expected a list or {'entries': [...]}."
        )

    print(f"      Loaded {len(entries)} entries.")
    return entries


def load_model(model_name: str) -> SentenceTransformer:
    """Load the HuggingFace sentence-transformer embedding model."""
    print(f"\n[2/5] Loading embedding model: {model_name}")
    print("      (First run will download the model — may take 1–2 minutes)")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    print(f"      Model loaded. Embedding dimension: {dim}")
    return model


def generate_embeddings(
    entries: list,
    model: SentenceTransformer,
    batch_size: int = BATCH_SIZE
) -> np.ndarray:
    """
    Extract embedding_text from each entry and generate embeddings in batches.

    Returns:
        np.ndarray of shape (num_entries, embedding_dim) — float32
    """
    print(f"\n[3/5] Generating embeddings for {len(entries)} entries...")

    texts = []
    for i, entry in enumerate(entries):
        embedding_text = entry.get("embedding_text", "").strip()
        if not embedding_text:
            # Fallback: build from title + problem + strategy if field missing
            embedding_text = (
                f"{entry.get('title', '')}. "
                f"{entry.get('problem_description', '')}. "
                f"{entry.get('optimization_strategy', '')}."
            )
            print(
                f"      ⚠️  Entry {entry.get('entry_id', i)} has no "
                f"embedding_text — using fallback concatenation."
            )
        texts.append(embedding_text)

    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2 normalization — cosine similarity via dot product
    )
    elapsed = time.time() - start

    print(f"      Embeddings generated in {elapsed:.1f}s.")
    print(f"      Shape: {embeddings.shape}  |  dtype: {embeddings.dtype}")
    print(type(embeddings))
    print(embeddings.shape)
    print(embeddings.dtype)

    print("NaN:", np.isnan(embeddings).sum())
    print("Inf:", np.isinf(embeddings).sum())

    print("Contiguous:", embeddings.flags["C_CONTIGUOUS"])

    # FAISS requires float32
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS IndexFlatIP (inner product) index.

    Because embeddings are L2-normalized, inner product == cosine similarity.
    IndexFlatIP is exact (no approximation) — correct for dissertation scale
    of < 500 vectors. Switch to IndexIVFFlat for 10,000+ vectors.
    """
    print(f"\n[4/5] Building FAISS index...")
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # IP = inner product (cosine on normalized vectors)
    index.add(embeddings)
    print(f"      Index built. Type: IndexFlatIP | Dimension: {dim} | Vectors: {index.ntotal}")
    return index


def build_metadata_store(entries: list) -> dict:
    """
    Build a metadata store that maps FAISS vector position (int) to full entry.

    The FAISS index stores vectors at positions 0, 1, 2, ...
    This store allows retrieval of the full entry dict by position.
    """
    metadata_store = {}
    for i, entry in enumerate(entries):
        metadata_store[str(i)] = {
            **entry,
            "faiss_index_position": i,
            "layer": 1,
        }
    return metadata_store


def save_outputs(
    index: faiss.Index,
    metadata_store: dict,
    output_dir: Path,
    model_name: str
):
    """Serialize the FAISS index and metadata store to disk."""
    print(f"\n[5/5] Saving outputs to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    index_path = output_dir / "faiss_index_layer1.bin"
    faiss.write_index(index, str(index_path))
    print(f"      ✅ FAISS index saved      → {index_path}")

    # Save metadata store
    meta_path = output_dir / "metadata_store_layer1.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "_meta": {
                    "total_entries": len(metadata_store),
                    "layer": 1,
                    "embedding_model": model_name,
                    "index_type": "IndexFlatIP",
                    "embedding_dim": index.d,
                    "normalized": True,
                },
                "entries": metadata_store,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"      ✅ Metadata store saved   → {meta_path}")


def run_spot_check(
    index: faiss.Index,
    metadata_store: dict,
    model: SentenceTransformer,
    top_k: int = 3
):
    """
    Run 5 spot-check queries to verify retrieval quality after indexing.
    Results are printed to console — review to confirm expected entries surface.
    """
    print("\n" + "=" * 60)
    print("  SPOT CHECK — Retrieval Quality Verification")
    print("=" * 60)

    test_queries = [
        ("SELECT * from large table causing slow query performance", "KB-SQL-001"),
        ("broadcast join small dimension table Spark shuffle", "KB-SQL-002"),
        ("correlated subquery running slow on every row repeated scan", "KB-SQL-004"),
        ("full table reload every pipeline run too slow incremental", "KB-ETL-002"),
        ("pipeline bottleneck one task taking too long execution log", "KB-WRK-001"),
    ]

    passed = 0
    for query_text, expected_id in test_queries:
        # Encode and normalize the query
        query_vec = model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # Search FAISS
        scores, positions = index.search(query_vec, top_k)

        # Retrieve metadata for top results
        top_ids = [
            metadata_store.get(str(pos), {}).get("entry_id", "?")
            for pos in positions[0]
        ]
        top_scores = [f"{s:.4f}" for s in scores[0]]

        hit = expected_id in top_ids
        if hit:
            passed += 1

        status = "✅ PASS" if hit else "❌ MISS"
        print(f"\n  Query   : {query_text[:65]}...")
        print(f"  Expected: {expected_id}")
        print(f"  Top-{top_k}   : {list(zip(top_ids, top_scores))}")
        print(f"  Result  : {status}")

    print(f"\n  Spot check complete: {passed}/{len(test_queries)} queries passed.")
    if passed < len(test_queries):
        print(
            "  ⚠️  Some queries missed expected entries. "
            "Review embedding_text quality for those entries."
        )
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build Layer 1 FAISS index from knowledge base entries."
    )
    parser.add_argument(
        "--entries",
        type=str,
        default=DEFAULT_ENTRIES_PATH,
        help=f"Path to entries JSON file (default: {DEFAULT_ENTRIES_PATH})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write FAISS index and metadata (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace sentence-transformer model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--skip-spot-check",
        action="store_true",
        help="Skip retrieval spot check after indexing"
    )
    args = parser.parse_args()

    entries_path = Path(args.entries)
    output_dir   = Path(args.output_dir)

    print("\n" + "=" * 60)
    print("  build_layer1_index.py — Layer 1 KB Indexing")
    print("=" * 60)

    # Pipeline
    entries          = load_entries(entries_path)
    print("✓ entries loaded")
    model            = load_model(args.model)
    print("✓ model loaded")

    embeddings       = generate_embeddings(entries, model)
    print("✓ embeddings generated")
    index            = build_faiss_index(embeddings)
    print("✓ index built")
    metadata_store   = build_metadata_store(entries)
    print("✓ metadata built")
    save_outputs(index, metadata_store, output_dir, args.model)
    print("✓ outputs saved")

    if not args.skip_spot_check:
        run_spot_check(index, metadata_store, model)

    print("✅ Layer 1 indexing complete.\n")
    print("   Next step: run build_layer2_index.py to process Layer 2 documents.")
    print("   Or run query_kb.py to test retrieval now.\n")


if __name__ == "__main__":
    main()
