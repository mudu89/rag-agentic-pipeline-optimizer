"""
build_layer2_index.py
=====================
Processes unstructured documents (plain text / Markdown), chunks them using
a paragraph-aware sliding window strategy, generates embeddings, and builds
a FAISS index for Layer 2 of the knowledge base.

Outputs:
    faiss_index_layer2.bin      — Serialized FAISS index for Layer 2 chunks
    metadata_store_layer2.json  — Maps FAISS vector position → chunk metadata
    kb_chunks_layer2_v1.json    — All generated chunks (for inspection/audit)

Prerequisites:
    pip install faiss-cpu sentence-transformers torch

Usage:
    python build_layer2_index.py
    python build_layer2_index.py --docs-dir path/to/documents/processed/
    python build_layer2_index.py --chunk-size 450 --overlap 60

Author: Mohammed Mudassirullah Sheriff
Project: Intelligent Optimization of Data Pipelines using RAG-Based Knowledge
         Retrieval and Agentic Decision Systems
Institution: BITS Pilani WILP — M.Tech Data Science & Engineering
"""

import json
import argparse
import re
import time
import numpy as np
from pathlib import Path
from datetime import date

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
DEFAULT_DOCS_DIR   = "../documents/processed"
DEFAULT_OUTPUT_DIR = "../index"
DEFAULT_CHUNKS_OUT = "../chunks/kb_chunks_layer2_v1.json"
DEFAULT_MODEL      = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 450    # Target tokens per chunk
DEFAULT_OVERLAP    = 60     # Overlap tokens between consecutive chunks
MIN_CHUNK_TOKENS   = 100    # Discard chunks below this size
MAX_CHUNK_TOKENS   = 600    # Hard cap — split further if exceeded
BATCH_SIZE         = 32

# Category hints: if a document filename contains these keywords,
# auto-assign a default category. Override manually in the chunk JSON if wrong.
FILENAME_CATEGORY_HINTS = {
    "airflow":      ("Workflow Execution Optimization", "Scheduling Optimization"),
    "spark":        ("ETL & Pipeline Optimization",     "Shuffle Minimization"),
    "bigquery":     ("SQL Optimization",                "Query Rewrite Patterns"),
    "postgresql":   ("Indexing Strategy",               "Index Selection"),
    "delta":        ("ETL & Pipeline Optimization",     "Incremental Processing"),
    "glue":         ("ETL & Pipeline Optimization",     "Redundant Transformation Elimination"),
    "redshift":     ("SQL Optimization",                "Aggregation Optimization"),
    "partition":    ("Partitioning Strategy",           "Partition Key Selection"),
    "index":        ("Indexing Strategy",               "Missing Index Detection"),
    "cache":        ("Caching Strategy",                "Intermediate Result Caching"),
    "storage":      ("Data Storage Optimization",       "File Format Selection"),
}


# ---------------------------------------------------------------------------
# Chunking utilities
# ---------------------------------------------------------------------------

def simple_token_count(text: str) -> int:
    """
    Approximate token count using whitespace splitting.
    Accurate enough for chunking decisions without requiring a full tokenizer.
    One word ≈ 1.3 tokens on average for technical English text.
    """
    words = len(text.split())
    return int(words * 1.3)


def split_into_paragraphs(text: str) -> list:
    """
    Split document text into paragraph units.

    Paragraph boundaries are detected by:
    - Two or more consecutive newlines
    - Markdown heading markers (## or ###)
    - Horizontal rules (---, ***)

    Code blocks (``` ... ```) are kept intact as single paragraphs.
    """
    # Protect code blocks from being split
    code_block_pattern = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    code_blocks = {}
    placeholder_template = "CODEBLOCK_{}"

    def replace_code_block(match):
        key = placeholder_template.format(len(code_blocks))
        code_blocks[key] = match.group(0)
        return f"\n\n{key}\n\n"

    text_protected = code_block_pattern.sub(replace_code_block, text)

    # Split on paragraph/section boundaries
    raw_paragraphs = re.split(
        r"\n{2,}|(?=^#{1,3} )",
        text_protected,
        flags=re.MULTILINE
    )

    paragraphs = []
    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue
        # Restore code blocks
        for key, block in code_blocks.items():
            para = para.replace(key, block)
        paragraphs.append(para)

    return paragraphs


def chunk_document(
    text: str,
    target_tokens: int = DEFAULT_CHUNK_SIZE,
    overlap_tokens: int = DEFAULT_OVERLAP,
    min_tokens: int = MIN_CHUNK_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list:
    """
    Paragraph-aware sliding window chunker.

    Strategy:
    1. Split document into paragraphs
    2. Merge consecutive paragraphs until target token count is reached
    3. When target is reached, emit chunk and start next chunk with
       the last overlap_tokens worth of text from the previous chunk
    4. Never split in the middle of a paragraph

    Returns:
        List of chunk text strings
    """
    paragraphs = split_into_paragraphs(text)
    chunks     = []
    buffer     = []
    buffer_tokens = 0

    for para in paragraphs:
        para_tokens = simple_token_count(para)

        # If a single paragraph exceeds max_tokens, split it by sentence
        if para_tokens > max_tokens:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                s_tokens = simple_token_count(sentence)
                if buffer_tokens + s_tokens >= target_tokens:
                    if buffer:
                        chunks.append(" ".join(buffer))
                    # Carry overlap: keep last overlap_tokens worth
                    buffer       = _trim_to_tokens(buffer, overlap_tokens)
                    buffer_tokens = sum(simple_token_count(b) for b in buffer)
                buffer.append(sentence)
                buffer_tokens += s_tokens
            continue

        # Normal paragraph: add to buffer
        if buffer_tokens + para_tokens >= target_tokens:
            if buffer:
                chunks.append("\n\n".join(buffer))
            # Carry overlap
            buffer        = _trim_to_tokens(buffer, overlap_tokens)
            buffer_tokens = sum(simple_token_count(b) for b in buffer)

        buffer.append(para)
        buffer_tokens += para_tokens

    # Flush remaining buffer
    if buffer:
        chunk_text = "\n\n".join(buffer)
        if simple_token_count(chunk_text) >= min_tokens:
            chunks.append(chunk_text)

    # Filter out any chunks below minimum size
    chunks = [c for c in chunks if simple_token_count(c) >= min_tokens]

    return chunks


def _trim_to_tokens(paragraphs: list, target_tokens: int) -> list:
    """
    Keep only the trailing paragraphs that sum to approximately target_tokens.
    Used to create the overlap between consecutive chunks.
    """
    result = []
    total  = 0
    for para in reversed(paragraphs):
        tokens = simple_token_count(para)
        if total + tokens <= target_tokens:
            result.insert(0, para)
            total += tokens
        else:
            break
    return result


def guess_category(filename: str) -> tuple:
    """
    Guess category and subcategory from document filename using keyword hints.
    Returns ("Unknown", "Unknown") if no hint matches.
    """
    name_lower = filename.lower()
    for keyword, (category, subcategory) in FILENAME_CATEGORY_HINTS.items():
        if keyword in name_lower:
            return category, subcategory
    return "Unknown", "Unknown"


def guess_source_type(filename: str) -> str:
    """
    Guess source type from filename conventions.
    Internal documents should be prefixed with 'internal_'.
    """
    if filename.startswith("internal_"):
        return "internal_confluence"
    if filename.startswith("team_"):
        return "team_doc"
    if filename.startswith("paper_"):
        return "academic_paper"
    return "public_guide"


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def process_documents(
    docs_dir: Path,
    target_tokens: int,
    overlap_tokens: int,
) -> list:
    """
    Process all .txt and .md files in docs_dir.

    Returns:
        List of chunk metadata dicts conforming to the Layer 2 schema.
    """
    print(f"\n[1/5] Scanning documents in: {docs_dir}")

    supported = list(docs_dir.glob("*.txt")) + list(docs_dir.glob("*.md"))
    if not supported:
        print(
            f"      ⚠️  No .txt or .md files found in {docs_dir}.\n"
            f"      Place preprocessed plain-text documents there and re-run.\n"
            f"      Creating a demo chunk from placeholder text for testing...\n"
        )
        return _generate_demo_chunks()

    print(f"      Found {len(supported)} document(s): "
          f"{[f.name for f in supported]}")

    all_chunks = []
    chunk_counter = 1

    for doc_path in supported:
        print(f"\n      Processing: {doc_path.name}")
        text = doc_path.read_text(encoding="utf-8")
        text = preprocess_text(text)

        raw_chunks = chunk_document(text, target_tokens, overlap_tokens)
        print(f"        → {len(raw_chunks)} chunks generated")

        category, subcategory = guess_category(doc_path.name)
        source_type           = guess_source_type(doc_path.name)

        # Guess input_type from category
        input_type_map = {
            "SQL Optimization":               "SQL Query",
            "ETL & Pipeline Optimization":    "ETL Workflow",
            "Workflow Execution Optimization":"Pipeline Log",
        }
        input_type = input_type_map.get(category, "Metadata")

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_id = f"L2-CHUNK-{chunk_counter:04d}"
            chunk_counter += 1

            chunk_meta = {
                "chunk_id":         chunk_id,
                "source_document":  doc_path.name,
                "source_type":      source_type,
                "category":         category,
                "subcategory":      subcategory,
                "input_type":       input_type,
                "chunk_index":      idx,
                "total_chunks_in_doc": len(raw_chunks),
                "token_count":      simple_token_count(chunk_text),
                "text":             chunk_text,
                "embedding_text":   chunk_text,   # For Layer 2, text == embedding_text
                "layer":            2,
                "source_url":       "",            # Fill in manually for public docs
                "created_date":     date.today().isoformat(),
            }
            all_chunks.append(chunk_meta)

    print(f"\n      Total chunks generated: {len(all_chunks)}")
    return all_chunks


def preprocess_text(text: str) -> str:
    """
    Clean raw document text before chunking.
    Removes navigation noise, normalizes whitespace.
    """
    # Remove excessive blank lines (keep max 2 consecutive)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalize Windows line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing whitespace from each line
    lines = [line.rstrip() for line in text.split("\n")]
    text  = "\n".join(lines)
    return text.strip()


def _generate_demo_chunks() -> list:
    """
    Generate 3 demo chunks for testing when no documents are available.
    These allow the full pipeline to be validated end-to-end.
    """
    demo_texts = [
        (
            "Apache Airflow Scheduling Best Practices\n\n"
            "When scheduling DAGs in Apache Airflow, avoid setting all pipelines "
            "to trigger at the same start time such as midnight. This creates a "
            "thundering herd problem where all DAGs compete for the same scheduler "
            "and worker resources simultaneously. Stagger start times across 30-60 "
            "minute windows to distribute load evenly. For pipelines with SLA "
            "dependencies, use the depends_on_past flag to ensure sequential "
            "execution where necessary. Monitor the scheduler heartbeat metric to "
            "detect backlog buildup early.",
            "airflow_best_practices.md",
            "Workflow Execution Optimization",
            "Scheduling Optimization",
        ),
        (
            "Spark Shuffle Optimization\n\n"
            "Shuffle operations in Apache Spark are one of the most expensive "
            "operations in distributed computation. A shuffle occurs whenever data "
            "needs to be redistributed across partitions — for example, during "
            "groupBy, join, or repartition operations. To minimize shuffle cost: "
            "first, reduce the number of shuffle operations by combining multiple "
            "groupBy steps. Second, use broadcast joins for small tables to "
            "eliminate the shuffle entirely. Third, tune spark.sql.shuffle.partitions "
            "to match your cluster size — the default of 200 is often too high for "
            "small datasets and too low for very large ones.",
            "spark_performance_tuning.md",
            "ETL & Pipeline Optimization",
            "Shuffle Minimization",
        ),
        (
            "BigQuery Partition and Clustering Best Practices\n\n"
            "Partitioning and clustering are the two primary mechanisms for "
            "controlling scan cost in BigQuery. Partitioning divides a table into "
            "segments based on a date or integer column. Clustering further sorts "
            "data within each partition by one or more columns. When both are "
            "applied, BigQuery can eliminate entire partitions and then prune "
            "blocks within the remaining partitions, achieving very high scan "
            "reduction. Always filter on the partition column in WHERE clauses "
            "to activate partition pruning. Wrap the partition column in a function "
            "such as DATE() and pruning is disabled.",
            "bigquery_best_practices.md",
            "Partitioning Strategy",
            "Partition Pruning",
        ),
    ]

    chunks = []
    for i, (text, source, category, subcategory) in enumerate(demo_texts):
        chunks.append({
            "chunk_id":              f"L2-CHUNK-{i+1:04d}",
            "source_document":       source,
            "source_type":           "public_guide",
            "category":              category,
            "subcategory":           subcategory,
            "input_type":            "ETL Workflow" if "ETL" in category else "SQL Query",
            "chunk_index":           0,
            "total_chunks_in_doc":   1,
            "token_count":           simple_token_count(text),
            "text":                  text,
            "embedding_text":        text,
            "layer":                 2,
            "source_url":            "",
            "created_date":          date.today().isoformat(),
            "_note":                 "Demo chunk — replace with real document processing",
        })

    print(f"      Generated {len(chunks)} demo chunks for pipeline testing.")
    return chunks


def embed_chunks(chunks: list, model: SentenceTransformer) -> np.ndarray:
    """Generate embeddings for all Layer 2 chunks."""
    print(f"\n[3/5] Generating embeddings for {len(chunks)} chunks...")
    texts = [c["embedding_text"] for c in chunks]

    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    elapsed = time.time() - start
    print(f"      Embeddings generated in {elapsed:.1f}s. Shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


def save_outputs(
    chunks: list,
    index: faiss.Index,
    output_dir: Path,
    chunks_out_path: Path,
    model_name: str
):
    """Save chunks JSON, FAISS index, and metadata store."""
    print(f"\n[5/5] Saving outputs...")

    # Save chunk list (audit trail)
    chunks_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"_chunks_meta": {"total": len(chunks), "layer": 2}, "chunks": chunks},
            f, indent=2, ensure_ascii=False,
        )
    print(f"      ✅ Chunks saved           → {chunks_out_path}")

    # Save FAISS index
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "faiss_index_layer2.bin"
    faiss.write_index(index, str(index_path))
    print(f"      ✅ FAISS index saved      → {index_path}")

    # Save metadata store
    metadata_store = {
        "_meta": {
            "total_chunks": len(chunks),
            "layer": 2,
            "embedding_model": model_name,
            "index_type": "IndexFlatIP",
            "normalized": True,
        },
        "entries": {str(i): {**c, "faiss_index_position": i}
                    for i, c in enumerate(chunks)},
    }
    meta_path = output_dir / "metadata_store_layer2.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, indent=2, ensure_ascii=False)
    print(f"      ✅ Metadata store saved   → {meta_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build Layer 2 FAISS index from unstructured documents."
    )
    parser.add_argument("--docs-dir",    type=str, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--output-dir",  type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunks-out",  type=str, default=DEFAULT_CHUNKS_OUT)
    parser.add_argument("--model",       type=str, default=DEFAULT_MODEL)
    parser.add_argument("--chunk-size",  type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap",     type=int, default=DEFAULT_OVERLAP)
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  build_layer2_index.py — Layer 2 Document Indexing")
    print("=" * 60)
    print(f"  Chunk size : {args.chunk_size} tokens")
    print(f"  Overlap    : {args.overlap} tokens")
    print(f"  Model      : {args.model}")

    docs_dir        = Path(args.docs_dir)
    output_dir      = Path(args.output_dir)
    chunks_out_path = Path(args.chunks_out)

    # Step 1 — Process documents into chunks
    chunks = process_documents(docs_dir, args.chunk_size, args.overlap)
    if not chunks:
        print("No chunks generated. Exiting.")
        return

    # Step 2 — Load model
    print(f"\n[2/5] Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)
    print(f"      Dimension: {model.get_sentence_embedding_dimension()}")

    # Step 3 — Embed
    embeddings = embed_chunks(chunks, model)

    # Step 4 — Build FAISS index
    print(f"\n[4/5] Building FAISS index...")
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"      Index built. Vectors: {index.ntotal}")

    # Step 5 — Save
    save_outputs(chunks, index, output_dir, chunks_out_path, args.model)

    print("\n✅ Layer 2 indexing complete.")
    print("   Next step: run merge_indexes.py to combine Layer 1 and Layer 2.\n")


if __name__ == "__main__":
    main()
