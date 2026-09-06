"""
build_layer2_index.py

Embeds and indexes Layer 2 chunks into FAISS.

--------------------------------------------------------------------------
CHANGED: chunking responsibility moved out of this script
--------------------------------------------------------------------------
This script previously did naive token-count sliding-window chunking of raw
documents internally. That has been replaced by sentence_chunk_enrich.py,
which performs sentence-boundary-aware chunking and tag enrichment as a
separate ingestion step. This script's job is now strictly:
    load enriched chunks -> generate embeddings -> build FAISS index -> save

Run sentence_chunk_enrich.py first:
    python sentence_chunk_enrich.py

Then run this script:
    python build_layer2_index.py

If knowledge_base/chunks/kb_chunks_layer2_v1.json does not exist yet, this
script will tell you to run sentence_chunk_enrich.py first rather than
silently generating placeholder data — chunk generation is no longer this
script's responsibility.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import faiss
except ImportError:
    faiss = None


DEFAULT_CHUNKS_PATH = "../chunks/kb_chunks_layer2_v1.json"
DEFAULT_INDEX_OUTPUT = "../index/faiss_index_layer2.bin"
DEFAULT_METADATA_OUTPUT = "../index/metadata_store_layer2.json"
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def check_dependencies():
    missing = []
    if np is None:
        missing.append("numpy")
    if SentenceTransformer is None:
        missing.append("sentence-transformers")
    if faiss is None:
        missing.append("faiss-cpu")
    if missing:
        print(f"ERROR: missing required package(s): {', '.join(missing)}")
        print(f"       Install with: pip install {' '.join(missing)}")
        sys.exit(1)


def load_chunks(path: Path):
    if not path.exists():
        print(f"ERROR: chunks file not found at {path}")
        print(f"       Run sentence_chunk_enrich.py first to generate it:")
        print(f"       python sentence_chunk_enrich.py")
        sys.exit(1)

    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = data.get("chunks", [])
    if not chunks:
        print(f"ERROR: {path} contains no chunks.")
        sys.exit(1)
    return chunks


def build_index(chunks, model_name: str):
    print(f"[2/5] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"[3/5] Generating embeddings for {len(chunks)} chunk(s)...")
    texts = [c["embedding_text"] for c in chunks]
    embeddings = model.encode(
        texts, show_progress_bar=False, convert_to_numpy=True
    ).astype("float32")

    # Normalize for cosine similarity via inner product (IndexFlatIP),
    # consistent with the Layer 1 index.
    faiss.normalize_L2(embeddings)

    print(f"[4/5] Building FAISS index... Vectors: {embeddings.shape[0]}, "
          f"Dim: {embeddings.shape[1]}")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings


def save_outputs(index, chunks, index_output: Path, metadata_output: Path):
    print(f"[5/5] Saving outputs...")
    index_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_output))

    # Metadata store: position in FAISS index -> chunk metadata (minus the
    # duplicate embedding_text field, since it's identical to text here)
    metadata_store = []
    for i, chunk in enumerate(chunks):
        entry = {k: v for k, v in chunk.items() if k != "embedding_text"}
        entry["faiss_index_position"] = i
        metadata_store.append(entry)

    metadata_output.write_text(json.dumps(metadata_store, indent=2), encoding="utf-8")

    print(f"      Index saved to: {index_output}")
    print(f"      Metadata saved to: {metadata_output}")


def spot_check(model, index, chunks, queries=None):
    """Optional sanity check: run a few known queries and confirm top result
    comes from a plausible source document."""
    if queries is None:
        queries = [
            ("How do I reduce shuffle cost in Spark?", "spark"),
            ("How does BigQuery partition pruning work?", "bigquery"),
            ("How should I stagger Airflow DAG start times?", "airflow"),
        ]

    passed = 0
    for query_text, expected_keyword in queries:
        q_emb = model.encode([query_text], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        _, indices = index.search(q_emb, 1)
        top_idx = int(indices[0][0])
        if top_idx < 0 or top_idx >= len(chunks):
            continue
        top_source = chunks[top_idx]["source_document"].lower()
        if expected_keyword in top_source:
            passed += 1

    print(f"      SPOT CHECK — {passed}/{len(queries)} queries matched expected source.")
    return passed, len(queries)


def main():
    parser = argparse.ArgumentParser(
        description="Embed and index Layer 2 chunks (produced by sentence_chunk_enrich.py) into FAISS."
    )
    parser.add_argument("--chunks", type=str, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--index-output", type=str, default=DEFAULT_INDEX_OUTPUT)
    parser.add_argument("--metadata-output", type=str, default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--skip-spot-check", action="store_true")
    args = parser.parse_args()

    check_dependencies()

    chunks_path = Path(args.chunks)
    index_output = Path(args.index_output)
    metadata_output = Path(args.metadata_output)

    print(f"[1/5] Loading chunks from {chunks_path} ...")
    chunks = load_chunks(chunks_path)
    print(f"      Loaded {len(chunks)} chunk(s).")

    index, embeddings = build_index(chunks, args.model)
    save_outputs(index, chunks, index_output, metadata_output)

    if not args.skip_spot_check:
        model = SentenceTransformer(args.model)
        spot_check(model, index, chunks)

    print("Layer 2 indexing complete.")


if __name__ == "__main__":
    main()
