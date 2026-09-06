"""
sentence_chunk_enrich.py

Replaces the naive token-count sliding-window chunking previously done inside
build_layer2_index.py with sentence-boundary-aware chunking, plus hybrid tag
extraction (schema-driven vocabulary matching + unsupervised keyphrase
extraction).

Input : knowledge_base/documents/processed/*.txt   (output of preprocess_documents.py)
Output: knowledge_base/chunks/kb_chunks_layer2_v1.json

--------------------------------------------------------------------------
NLP components used, and why (for viva defensibility)
--------------------------------------------------------------------------
- Sentence segmentation: spaCy blank English pipeline + the rule-based
  `sentencizer` component only. Deliberately NOT the full statistical
  `en_core_web_sm` model -- sentence-boundary detection does not require
  dependency parsing or NER, so the rule-based sentencizer keeps ingestion
  lightweight, deterministic, and free of an external model download.

- Keyphrase extraction: YAKE (unsupervised, statistics-based; requires no
  training corpus). Chosen over topic modeling (LDA) because the source
  corpus (a handful of vendor documents) is far too small for LDA to
  produce stable, interpretable topics.

- Platform tagging: deterministic, word-boundary vocabulary match against
  a controlled platform list -- NOT statistical NER. General-purpose NER
  models are not trained to recognize "Spark" or "BigQuery" as named
  entities, so controlled-vocabulary matching is the more reliable and
  more explainable choice here.

--------------------------------------------------------------------------
Deliberately NOT automated here
--------------------------------------------------------------------------
`category`, `subcategory`, and `input_type` per document are taxonomy
assignments -- curation decisions, the same as Layer 1 KB entries -- and
are read from a manual `document_metadata_map.json` file. Auto-classifying
documents into the 7-category taxonomy would require a trained classifier,
which is out of scope for a small, fixed source corpus, and would also
blur the line between what is genuinely NLP-derived (tags) and what is a
curated label (category).

Usage:
    python sentence_chunk_enrich.py
    python sentence_chunk_enrich.py --target-tokens 450 --overlap-sentences 2
    python sentence_chunk_enrich.py --metadata-map ../documents/document_metadata_map.json
"""

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

try:
    import spacy
except ImportError:
    spacy = None

try:
    import yake
except ImportError:
    yake = None


# --------------------------------------------------------------------------
# Controlled vocabulary for platform tagging
# --------------------------------------------------------------------------

PLATFORM_VOCAB = [
    "Spark", "Hive", "BigQuery", "PostgreSQL", "MySQL", "Trino", "Presto",
    "Redshift", "Snowflake", "Airflow", "Delta Lake", "Databricks", "Kafka",
    "Athena", "Oracle", "SQL Server", "DynamoDB", "MongoDB",
]

DEFAULT_TARGET_TOKENS = 450
DEFAULT_OVERLAP_SENTENCES = 2
DEFAULT_MIN_CHUNK_TOKENS = 50
DEFAULT_MAX_KEYPHRASES = 5


# --------------------------------------------------------------------------
# NLP setup
# --------------------------------------------------------------------------

def load_sentencizer():
    """Blank spaCy pipeline + rule-based sentencizer only. No model download."""
    if spacy is None:
        print("ERROR: spaCy is required. Install with: pip install spacy")
        sys.exit(1)
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


def load_keyword_extractor(max_keywords: int):
    if yake is None:
        print("WARNING: yake not installed — keyphrase_tags will be empty. "
              "Install with: pip install yake")
        return None
    return yake.KeywordExtractor(lan="en", n=3, top=max_keywords, dedupLim=0.7)


def split_sentences(nlp, text: str):
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def word_count(text: str) -> int:
    return len(text.split())


# --------------------------------------------------------------------------
# Sentence-boundary-aware chunking with sentence-level overlap
# --------------------------------------------------------------------------

def build_chunks_for_document(sentences, target_tokens, overlap_sentences, min_tokens):
    """
    Accumulate sentences into chunks up to ~target_tokens, then start the next
    chunk with the last `overlap_sentences` sentences of the previous chunk
    (sentence-aligned overlap, never a mid-sentence cut).

    A single sentence longer than target_tokens is placed in its own chunk
    rather than being split mid-sentence or causing an infinite loop.

    Returns: list of sentence-groups (list of list of str) — NOT joined text,
    so callers can compute sentence_count without re-running the segmenter.
    """
    chunks = []
    current = []
    current_tokens = 0

    def flush():
        nonlocal current, current_tokens
        if current:
            chunks.append(list(current))
            overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current = list(overlap)
            current_tokens = sum(word_count(s) for s in current)

    for sent in sentences:
        sent_tokens = word_count(sent)
        if sent_tokens > target_tokens:
            flush()
            chunks.append([sent])
            continue
        if current and current_tokens + sent_tokens > target_tokens:
            flush()
        current.append(sent)
        current_tokens += sent_tokens

    if current:
        # Avoid a tiny trailing chunk: merge into the previous one if too small
        if chunks and word_count(" ".join(current)) < min_tokens:
            chunks[-1] = chunks[-1] + current
        else:
            chunks.append(current)

    return chunks


# --------------------------------------------------------------------------
# Tag extraction
# --------------------------------------------------------------------------

def extract_platform_tags(text: str):
    found = []
    for platform in PLATFORM_VOCAB:
        pattern = r"\b" + re.escape(platform) + r"\b"
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(platform)
    return found


def extract_keyphrases(text: str, extractor, max_keywords: int):
    if extractor is None:
        return []
    try:
        keywords = extractor.extract_keywords(text)
    except Exception:
        return []
    # YAKE scores are lower = more relevant
    keywords_sorted = sorted(keywords, key=lambda kv: kv[1])
    return [kw for kw, _score in keywords_sorted[:max_keywords]]


# --------------------------------------------------------------------------
# Document metadata (curated, not NLP-derived)
# --------------------------------------------------------------------------

def load_metadata_map(path: Path):
    if not path.exists():
        print(f"      WARNING: metadata map not found at {path}")
        print(f"               category/subcategory/input_type will default to "
              f"'Unclassified' / 'Metadata'.")
        print(f"               Create this file to assign curated taxonomy labels "
              f"per source document — see sample format in the docstring.")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Per-document processing
# --------------------------------------------------------------------------

def process_document(path: Path, nlp, keyword_extractor, metadata_map, args, id_start):
    text = path.read_text(encoding="utf-8", errors="ignore")
    sentences = split_sentences(nlp, text)

    if not sentences:
        print(f"    - {path.name}: WARNING — no sentences extracted, skipping.")
        return [], id_start

    sentence_groups = build_chunks_for_document(
        sentences, args.target_tokens, args.overlap_sentences, args.min_chunk_tokens
    )

    doc_meta = metadata_map.get(path.name, {})
    category = doc_meta.get("category", "Unclassified")
    subcategory = doc_meta.get("subcategory", "Unclassified")
    input_type = doc_meta.get("input_type", "Metadata")
    source_type = doc_meta.get("source_type", "public_guide")
    source_url = doc_meta.get("source_url", "")

    chunks = []
    chunk_id = id_start
    all_platform_tags = set()

    for idx, group in enumerate(sentence_groups):
        chunk_text = " ".join(group)
        platform_tags = extract_platform_tags(chunk_text)
        keyphrase_tags = extract_keyphrases(chunk_text, keyword_extractor, args.max_keyphrases)
        all_platform_tags.update(platform_tags)

        chunks.append({
            "chunk_id": f"L2-CHUNK-{chunk_id:04d}",
            "source_document": path.name,
            "source_type": source_type,
            "category": category,
            "subcategory": subcategory,
            "input_type": input_type,
            "chunk_index": idx,
            "total_chunks_in_doc": len(sentence_groups),
            "token_count": word_count(chunk_text),
            "sentence_count": len(group),
            "text": chunk_text,
            "embedding_text": chunk_text,
            "platform_tags": platform_tags,
            "keyphrase_tags": keyphrase_tags,
            "layer": 2,
            "source_url": source_url,
            "created_date": str(date.today()),
        })
        chunk_id += 1

    print(f"    - {path.name}: {len(sentence_groups)} chunk(s), "
          f"{len(sentences)} sentence(s), "
          f"platform tags seen: {sorted(all_platform_tags) or 'none'}")
    if category == "Unclassified":
        print(f"      NOTE: no entry in document_metadata_map.json for "
              f"'{path.name}' — assign one for accurate taxonomy labels.")

    return chunks, chunk_id


# --------------------------------------------------------------------------
# Fallback demo chunks (used only if no processed documents exist yet —
# mirrors the previous behavior of build_layer2_index.py so early pipeline
# testing still works without a document corpus)
# --------------------------------------------------------------------------

_DEMO_DOCS = [
    {
        "source_document": "airflow_best_practices.md",
        "category": "Workflow Execution Optimization",
        "subcategory": "Scheduling Optimization",
        "input_type": "SQL Query",
        "text": (
            "Apache Airflow Scheduling Best Practices. When scheduling DAGs in "
            "Apache Airflow, avoid setting all pipelines to trigger at the same "
            "start time such as midnight. This creates a thundering herd problem "
            "where all DAGs compete for the same scheduler and worker resources "
            "simultaneously. Stagger start times across 30-60 minute windows to "
            "distribute load evenly. For pipelines with SLA dependencies, use the "
            "depends_on_past flag to ensure sequential execution where necessary. "
            "Monitor the scheduler heartbeat metric to detect backlog buildup early."
        ),
    },
    {
        "source_document": "spark_performance_tuning.md",
        "category": "ETL & Pipeline Optimization",
        "subcategory": "Shuffle Minimization",
        "input_type": "ETL Workflow",
        "text": (
            "Spark Shuffle Optimization. Shuffle operations in Apache Spark are "
            "one of the most expensive operations in distributed computation. A "
            "shuffle occurs whenever data needs to be redistributed across "
            "partitions, for example during groupBy, join, or repartition "
            "operations. To minimize shuffle cost, first reduce the number of "
            "shuffle operations by combining multiple groupBy steps. Second, use "
            "broadcast joins for small tables to eliminate the shuffle entirely. "
            "Third, tune spark.sql.shuffle.partitions to match your cluster size, "
            "the default of 200 is often too high for small datasets and too low "
            "for very large ones."
        ),
    },
    {
        "source_document": "bigquery_best_practices.md",
        "category": "Partitioning Strategy",
        "subcategory": "Partition Pruning",
        "input_type": "SQL Query",
        "text": (
            "BigQuery Partition and Clustering Best Practices. Partitioning and "
            "clustering are the two primary mechanisms for controlling scan cost "
            "in BigQuery. Partitioning divides a table into segments based on a "
            "date or integer column. Clustering further sorts data within each "
            "partition by one or more columns. When both are applied, BigQuery "
            "can eliminate entire partitions and then prune blocks within the "
            "remaining partitions, achieving very high scan reduction. Always "
            "filter on the partition column in WHERE clauses to activate "
            "partition pruning. Wrap the partition column in a function such as "
            "DATE() and pruning is disabled."
        ),
    },
]


def generate_demo_chunks(nlp, keyword_extractor, args):
    print("      No processed .txt documents found.")
    print("      Falling back to 3 demo chunks for pipeline testing purposes.")
    chunks = []
    for i, demo in enumerate(_DEMO_DOCS, start=1):
        sentences = split_sentences(nlp, demo["text"])
        platform_tags = extract_platform_tags(demo["text"])
        keyphrase_tags = extract_keyphrases(demo["text"], keyword_extractor, args.max_keyphrases)
        chunks.append({
            "chunk_id": f"L2-CHUNK-{i:04d}",
            "source_document": demo["source_document"],
            "source_type": "public_guide",
            "category": demo["category"],
            "subcategory": demo["subcategory"],
            "input_type": demo["input_type"],
            "chunk_index": 0,
            "total_chunks_in_doc": 1,
            "token_count": word_count(demo["text"]),
            "sentence_count": len(sentences),
            "text": demo["text"],
            "embedding_text": demo["text"],
            "platform_tags": platform_tags,
            "keyphrase_tags": keyphrase_tags,
            "layer": 2,
            "source_url": "",
            "created_date": str(date.today()),
            "_note": "Demo chunk — replace with real document processing",
        })
    return chunks


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sentence-aware chunking + hybrid tag enrichment for Layer 2 KB."
    )
    parser.add_argument("--input-dir", type=str, default="../documents/processed")
    parser.add_argument("--output", type=str, default="../chunks/kb_chunks_layer2_v1.json")
    parser.add_argument("--metadata-map", type=str,
                         default="../schema/document_metadata_map.json")
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--overlap-sentences", type=int, default=DEFAULT_OVERLAP_SENTENCES)
    parser.add_argument("--min-chunk-tokens", type=int, default=DEFAULT_MIN_CHUNK_TOKENS)
    parser.add_argument("--max-keyphrases", type=int, default=DEFAULT_MAX_KEYPHRASES)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    metadata_map_path = Path(args.metadata_map)

    print(f"[1/3] Loading NLP components ...")
    nlp = load_sentencizer()
    keyword_extractor = load_keyword_extractor(args.max_keyphrases)
    metadata_map = load_metadata_map(metadata_map_path)

    txt_files = sorted(input_dir.glob("*.txt")) if input_dir.exists() else []
    print(f"[2/3] Found {len(txt_files)} processed document(s) in {input_dir}")

    if not txt_files:
        all_chunks = generate_demo_chunks(nlp, keyword_extractor, args)
    else:
        all_chunks = []
        next_id = 1
        for f in txt_files:
            doc_chunks, next_id = process_document(
                f, nlp, keyword_extractor, metadata_map, args, next_id
            )
            all_chunks.extend(doc_chunks)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "_chunks_meta": {"total": len(all_chunks), "layer": 2},
        "chunks": all_chunks,
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"[3/3] Wrote {len(all_chunks)} chunk(s) to {output_path}")


if __name__ == "__main__":
    main()
