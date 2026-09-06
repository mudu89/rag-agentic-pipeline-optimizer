"""
merge_indexes.py
================
Merges the Layer 1 (structured entries) and Layer 2 (document chunks) FAISS
indexes into a single unified index with a combined metadata store.

The merged index is what the query_kb.py script uses at runtime.

Inputs:
    faiss_index_layer1.bin      — Built by build_layer1_index.py
    metadata_store_layer1.json  — Built by build_layer1_index.py
    faiss_index_layer2.bin      — Built by build_layer2_index.py
    metadata_store_layer2.json  — Built by build_layer2_index.py

Outputs:
    faiss_index.bin             — Unified FAISS index (Layer 1 + Layer 2)
    metadata_store.json         — Combined metadata store

Design notes:
    - Layer 1 vectors are inserted first (positions 0 to N1-1)
    - Layer 2 vectors are inserted after (positions N1 to N1+N2-1)
    - The 'layer' field in metadata distinguishes entry types at query time
    - Both layers must use the same embedding model and dimension

Usage:
    python merge_indexes.py
    python merge_indexes.py --index-dir path/to/index/

Author: Mohammed Mudassirullah Sheriff
Project: Intelligent Optimization of Data Pipelines using RAG-Based Knowledge
         Retrieval and Agentic Decision Systems
Institution: BITS Pilani WILP — M.Tech Data Science & Engineering
"""

import json
import argparse
import numpy as np
from pathlib import Path

try:
    import faiss
except ImportError:
    raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_INDEX_DIR = "../index"


# ---------------------------------------------------------------------------
# Core merge logic
# ---------------------------------------------------------------------------

def load_faiss_index(path: Path) -> faiss.Index:
    """Load a serialized FAISS index from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {path}\n"
            f"Run the corresponding build script first."
        )
    index = faiss.read_index(str(path))
    print(f"      Loaded: {path.name}  |  Vectors: {index.ntotal}  |  Dim: {index.d}")
    return index


def load_metadata_store(path: Path) -> dict:
    """Load a metadata store JSON file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata store not found: {path}\n"
            f"Run the corresponding build script first."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("entries", data)   # Support both wrapped and flat formats
    print(f"      Loaded: {path.name}  |  Entries: {len(entries)}")
    return entries


def validate_compatibility(index1: faiss.Index, index2: faiss.Index):
    """
    Confirm both indexes have the same embedding dimension.
    Mismatched dimensions indicate different embedding models were used —
    a fatal error that would silently corrupt retrieval results.
    """
    if index1.d != index2.d:
        raise ValueError(
            f"Embedding dimension mismatch: "
            f"Layer 1 dim={index1.d}, Layer 2 dim={index2.d}.\n"
            f"Both layers must use the same embedding model. "
            f"Rebuild one or both indexes with a consistent model."
        )
    print(f"      ✅ Dimension compatibility confirmed: {index1.d}d")


def extract_vectors(index: faiss.Index) -> np.ndarray:
    """
    Extract all vectors from a FAISS IndexFlatIP index.

    Note: Vector extraction is only supported for Flat index types.
    IVF or HNSW indexes do not support reconstruct_n.
    """
    n   = index.ntotal
    dim = index.d
    vectors = np.zeros((n, dim), dtype=np.float32)
    index.reconstruct_n(0, n, vectors)
    return vectors


def merge_indexes(
    index1: faiss.Index,
    meta1: dict,
    index2: faiss.Index,
    meta2: dict,
) -> tuple:
    """
    Merge two FAISS indexes and their metadata stores.

    Layer 1 entries occupy positions 0 to (n1-1).
    Layer 2 entries occupy positions n1 to (n1+n2-1).

    Returns:
        merged_index     — Combined FAISS IndexFlatIP
        merged_metadata  — Combined metadata dict keyed by string position
    """
    print("\n[3/5] Extracting vectors from both indexes...")
    vectors1 = extract_vectors(index1)
    vectors2 = extract_vectors(index2)
    print(f"      Layer 1 vectors: {vectors1.shape}")
    print(f"      Layer 2 vectors: {vectors2.shape}")

    print("\n[4/5] Building merged FAISS index...")
    dim            = index1.d
    merged_index   = faiss.IndexFlatIP(dim)
    merged_vectors = np.vstack([vectors1, vectors2])
    merged_index.add(merged_vectors)
    print(f"      Merged index built. Total vectors: {merged_index.ntotal}")

    print("      Building merged metadata store...")
    merged_metadata = {}
    n1 = index1.ntotal

    # Layer 1 entries — positions 0 to n1-1
    for old_pos_str, entry in meta1.items():
        if old_pos_str.startswith("_"):
            continue
        new_pos = int(old_pos_str)   # Layer 1 keeps same positions
        merged_metadata[str(new_pos)] = {
            **entry,
            "faiss_index_position": new_pos,
            "layer": 1,
        }

    # Layer 2 entries — positions n1 to n1+n2-1
    for old_pos_str, chunk in meta2.items():
        if old_pos_str.startswith("_"):
            continue
        new_pos = n1 + int(old_pos_str)   # Offset by Layer 1 count
        merged_metadata[str(new_pos)] = {
            **chunk,
            "faiss_index_position": new_pos,
            "layer": chunk.get("layer", 2),
        }

    print(f"      Metadata entries: {len(merged_metadata)} "
          f"(L1: {n1}, L2: {index2.ntotal})")

    return merged_index, merged_metadata


def save_merged_outputs(
    merged_index: faiss.Index,
    merged_metadata: dict,
    meta1_raw: dict,
    meta2_raw: dict,
    output_dir: Path,
):
    """Save the merged FAISS index and metadata store."""
    print("\n[5/5] Saving merged outputs...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract model info from individual metadata files
    model1 = meta1_raw.get("_meta", {}).get("embedding_model", "unknown")
    model2 = meta2_raw.get("_meta", {}).get("embedding_model", "unknown")

    # Save merged FAISS index
    index_path = output_dir / "faiss_index.bin"
    faiss.write_index(merged_index, str(index_path))
    print(f"      ✅ Merged FAISS index saved  → {index_path}")

    # Save merged metadata store
    meta_path = output_dir / "metadata_store.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "_meta": {
                    "description": "Unified knowledge base — Layer 1 + Layer 2",
                    "total_vectors": merged_index.ntotal,
                    "layer1_count": meta1_raw.get("_meta", {}).get("total_entries",
                                    len([k for k in merged_metadata
                                         if merged_metadata[k].get("layer") == 1])),
                    "layer2_count": meta2_raw.get("_meta", {}).get("total_chunks",
                                    len([k for k in merged_metadata
                                         if merged_metadata[k].get("layer") == 2])),
                    "embedding_model_layer1": model1,
                    "embedding_model_layer2": model2,
                    "index_type": "IndexFlatIP",
                    "embedding_dim": merged_index.d,
                    "normalized": True,
                    "layer1_positions": f"0 to {meta1_raw.get('_meta', {}).get('total_entries', '?') - 1 if isinstance(meta1_raw.get('_meta', {}).get('total_entries'), int) else '?'}",
                    "layer2_positions": f"{meta1_raw.get('_meta', {}).get('total_entries', '?')} onward",
                },
                "entries": merged_metadata,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"      ✅ Merged metadata saved     → {meta_path}")


def print_merge_summary(merged_index: faiss.Index, merged_metadata: dict):
    """Print a summary of the merged index composition."""
    l1_count = sum(1 for v in merged_metadata.values() if v.get("layer") == 1)
    l2_count = sum(1 for v in merged_metadata.values() if v.get("layer") == 2)

    # Category breakdown
    category_counts = {}
    for v in merged_metadata.values():
        cat = v.get("category", "Unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print("\n" + "=" * 60)
    print("  MERGE SUMMARY")
    print("=" * 60)
    print(f"  Total vectors in unified index : {merged_index.ntotal}")
    print(f"  Layer 1 entries                : {l1_count}")
    print(f"  Layer 2 chunks                 : {l2_count}")
    print(f"  Embedding dimension            : {merged_index.d}")
    print(f"\n  Category breakdown:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat:<40} {count:>4} entries/chunks")
    print("=" * 60)
    print("\n  ✅ Merge complete.")
    print("     Run query_kb.py to test the unified knowledge base.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Merge Layer 1 and Layer 2 FAISS indexes into a unified index."
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=DEFAULT_INDEX_DIR,
        help=f"Directory containing layer indexes (default: {DEFAULT_INDEX_DIR})"
    )
    args = parser.parse_args()

    index_dir = Path(args.index_dir)

    print("\n" + "=" * 60)
    print("  merge_indexes.py — Unified KB Index Builder")
    print("=" * 60)

    # Step 1 — Load Layer 1
    print("\n[1/5] Loading Layer 1 index and metadata...")
    index1    = load_faiss_index(index_dir / "faiss_index_layer1.bin")
    meta1_raw_path = index_dir / "metadata_store_layer1.json"
    with open(meta1_raw_path, "r", encoding="utf-8") as f:
        meta1_raw = json.load(f)
    meta1 = meta1_raw.get("entries", meta1_raw)

    # Step 2 — Load Layer 2
    print("\n[2/5] Loading Layer 2 index and metadata...")
    index2    = load_faiss_index(index_dir / "faiss_index_layer2.bin")
    meta2_raw_path = index_dir / "metadata_store_layer2.json"
    with open(meta2_raw_path, "r", encoding="utf-8") as f:
        meta2_raw = json.load(f)
    meta2 = meta2_raw.get("entries", meta2_raw)

    # Validate compatibility before merging
    validate_compatibility(index1, index2)

    # Steps 3–4 — Merge
    merged_index, merged_metadata = merge_indexes(index1, meta1, index2, meta2)

    # Step 5 — Save
    save_merged_outputs(merged_index, merged_metadata, meta1_raw, meta2_raw, index_dir)

    # Summary
    print_merge_summary(merged_index, merged_metadata)


if __name__ == "__main__":
    main()
