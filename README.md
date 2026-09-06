# Execution Guide
# Intelligent Optimization of Data Pipelines using RAG-Based Knowledge
# Retrieval and Agentic Decision Systems
#
# Author: Mohammed Mudassirullah Sheriff
# BITS Pilani WILP — M.Tech Data Science & Engineering
# =============================================================================

## Prerequisites

- Python 3.10 or higher
- pip package manager
- At least one LLM API key (OpenAI recommended for development)
- 4GB RAM minimum (8GB recommended for local HuggingFace models)

---

## Step 1 — Set Up the Project Folder

Create the complete folder structure from your project root:

    mkdir -p knowledge_base/schema
    mkdir -p knowledge_base/taxonomy
    mkdir -p knowledge_base/entries
    mkdir -p knowledge_base/documents/raw
    mkdir -p knowledge_base/documents/processed
    mkdir -p knowledge_base/chunks
    mkdir -p knowledge_base/index
    mkdir -p knowledge_base/scripts
    mkdir -p agentic_system/config
    mkdir -p agentic_system/core
    mkdir -p agentic_system/agents
    mkdir -p agentic_system/prompts
    mkdir -p agentic_system/api
    mkdir -p agentic_system/cli
    mkdir -p agentic_system/tests/test_inputs
    mkdir -p logs
    mkdir -p results
    mkdir -p docs

Place all downloaded files into their respective folders as shown in the
folder structure document.

---

## Step 2 — Create a Virtual Environment

    python -m venv venv

    # Activate — Mac/Linux
    source venv/bin/activate

    # Activate — Windows
    venv\Scripts\activate

---

## Step 3 — Install Dependencies

    pip install -r requirements.txt

If you only want to use one LLM provider, you can skip the others:

    # OpenAI only
    pip install openai

    # Anthropic only
    pip install anthropic

    # HuggingFace only
    pip install huggingface-hub

    # Google only
    pip install google-generativeai

---

## Step 4 — Set Your API Key

Do not hardcode API keys in any file. Set them as environment variables.

    # Mac/Linux
    export OPENAI_API_KEY="your-openai-api-key-here"
    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
    # Windows
    set OPENAI_API_KEY=your-openai-api-key-here

    

Alternatively, create a .env file in the project root:

    OPENAI_API_KEY=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

And load it by adding this to the top of run_pipeline.py or pipeline.py:

    from dotenv import load_dotenv
    load_dotenv()

To switch to a different LLM provider, edit config.yaml:

    llm:
      provider: "anthropic"
      model: "claude-sonnet-4-6"
      api_key_env: "ANTHROPIC_API_KEY"

---

## Step 5 — Build the Knowledge Base

All scripts are run from inside knowledge_base/scripts/.

    cd knowledge_base/scripts

### 5a. Validate the sample entries

    python validate_entries.py --entries ../entries/kb_entries_sample_v1.json

Expected output:
    Total entries: 10
    Passed: 10
    Failed: 0
    All entries passed validation.

### 5b. Build the Layer 1 FAISS index

    python build_layer1_index.py --entries ../entries/kb_entries_sample_v1.json

Expected output:
    [1/5] Loading entries... Loaded 10 entries.
    [2/5] Loading embedding model: all-MiniLM-L6-v2
    [3/5] Generating embeddings for 10 entries...
    [4/5] Building FAISS index... Vectors: 10
    [5/5] Saving outputs...
    SPOT CHECK — 5/5 queries passed.
    Layer 1 indexing complete.

Output files created:
    knowledge_base/index/faiss_index_layer1.bin
    knowledge_base/index/metadata_store_layer1.json

### 5c. Build the Layer 2 index (optional for initial testing)

Layer 2 ingestion is a three-stage process. Each stage is a separate
script with a distinct responsibility:

    raw documents  --preprocess_documents.py-->  cleaned .txt
    cleaned .txt   --sentence_chunk_enrich.py-->  enriched chunks (JSON)
    enriched chunks --build_layer2_index.py-->    FAISS index

#### 5c-i. Convert raw source documents to cleaned plain text

Place raw source documents (PDF, Markdown, or HTML) in
knowledge_base/documents/raw/, then run:

    python preprocess_documents.py

Output files created:
    knowledge_base/documents/processed/*.txt

#### 5c-ii. Assign taxonomy labels to each source document (one-time, manual)

Open knowledge_base/documents/document_metadata_map.json and add an entry
for each processed document, specifying its category, subcategory, and
input_type (values must match knowledge_base/taxonomy/kb_taxonomy_v1.json
and knowledge_base/schema/kb_schema_v1.json exactly). This is a curation
step, not an automated one — the same reasoning as assigning taxonomy
labels to Layer 1 KB entries. Documents without an entry in this file will
default to category "Unclassified", subcategory "Unclassified", and
input_type "Metadata", with a warning printed at chunking time.

#### 5c-iii. Sentence-aware chunking and tag enrichment

    python sentence_chunk_enrich.py

This performs sentence-boundary-aware chunking (spaCy sentencizer) instead
of a naive token-count window, and enriches each chunk with:
    - platform_tags     (schema-driven vocabulary match against known
                          data platforms, e.g. Spark, BigQuery, Airflow)
    - keyphrase_tags    (unsupervised keyphrase extraction via YAKE)

If knowledge_base/documents/processed/ is empty, this script falls back to
3 demo chunks for pipeline testing purposes, matching prior behavior.

Requires additionally:
    pip install spacy yake

Output files created:
    knowledge_base/chunks/kb_chunks_layer2_v1.json

#### 5c-iv. Embed and index the enriched chunks

    python build_layer2_index.py

This step no longer performs any chunking — it strictly loads the chunks
produced by sentence_chunk_enrich.py, generates embeddings, and builds the
FAISS index. If kb_chunks_layer2_v1.json does not exist yet, this script
will stop and tell you to run sentence_chunk_enrich.py first.

Output files created:
    knowledge_base/index/faiss_index_layer2.bin
    knowledge_base/index/metadata_store_layer2.json

### 5d. Merge Layer 1 and Layer 2 into unified index

    python merge_indexes.py

Output files created:
    knowledge_base/index/faiss_index.bin       <- This is the production index
    knowledge_base/index/metadata_store.json

---

## Step 6 — Configure the Agentic System

Open agentic_system/config/config.yaml and verify these paths are correct
relative to where you run the CLI from:

    knowledge_base:
      index_path: "../../knowledge_base/index/faiss_index.bin"
      metadata_path: "../../knowledge_base/index/metadata_store.json"

Adjust the relative paths if you run scripts from a different directory.

---

## Step 7 — Run the End-to-End Tests

    cd agentic_system/tests
    python test_pipeline.py

This runs 4 test cases — one per input type:
    - SQL Query with correlated subquery and SELECT *
    - ETL DAG with full reload and redundant transformation
    - Pipeline log with bottleneck and OOM error
    - Metadata with missing index and unpartitioned table

Expected output:
    TEST SUMMARY
    Total  : 4
    Passed : 4
    Failed : 0
    All tests passed. System is ready for evaluation.

To see the full recommendation report for each test:

    python test_pipeline.py --verbose

To run a single test:

    python test_pipeline.py --test sql
    python test_pipeline.py --test etl
    python test_pipeline.py --test log
    python test_pipeline.py --test metadata

---

## Step 8 — Use the CLI

Run from inside agentic_system/cli/.

    cd agentic_system/cli

### Analyze a raw query string

    python run_pipeline.py --query "SELECT * FROM orders WHERE status = 'active'"

### Analyze a file

    python run_pipeline.py --file ../tests/test_inputs/sample_query.sql
    python run_pipeline.py --file ../tests/test_inputs/sample_dag.yaml

### Show full agent trace

    python run_pipeline.py --file ../tests/test_inputs/sample_query.sql --trace

### Save output to JSON

    python run_pipeline.py --query "SELECT * FROM orders" --output ../../results/output.json

### Print raw JSON instead of formatted report

    python run_pipeline.py --query "SELECT * FROM orders" --json

---

## Step 9 — Start the FastAPI Server

    cd agentic_system
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

The API will be available at:
    http://localhost:8000

Interactive API documentation (Swagger UI):
    http://localhost:8000/docs

### Available endpoints

    GET  /health          Health check — confirms pipeline is loaded
    GET  /kb/stats        Knowledge base statistics
    POST /analyze         Analyze a text input
    POST /analyze/file    Analyze an uploaded file

### Example API call using curl

    curl -X POST http://localhost:8000/analyze \
      -H "Content-Type: application/json" \
      -d '{
        "input_text": "SELECT * FROM orders WHERE YEAR(created_at) = 2025",
        "include_trace": true
      }'

### Example API call using Python requests

    import requests

    response = requests.post(
        "http://localhost:8000/analyze",
        json={
            "input_text": "SELECT * FROM orders WHERE status = 'active'",
            "include_trace": False
        }
    )
    print(response.json())

---

## Step 10 — Run the Retrieval Evaluation Suite

From inside knowledge_base/scripts/:

    python query_kb.py --evaluate --eval-csv ../../results/eval_results.csv

This runs 20 pre-defined queries and computes:
    - Top-1 Accuracy     (target >= 70%)
    - Top-3 Accuracy     (target >= 85%)
    - Mean Similarity    (target >= 0.65)
    - Layer 1 Rate       (target >= 60%)

The CSV output at results/eval_results.csv is used directly in your
dissertation evaluation chapter.

---

## Switching LLM Providers

No code changes are needed. Edit config.yaml only:

### Switch to Anthropic Claude

    llm:
      provider: "anthropic"
      model: "claude-sonnet-4-6"
      api_key_env: "ANTHROPIC_API_KEY"

    export ANTHROPIC_API_KEY="your-key"

### Switch to Google Gemini

    llm:
      provider: "google"
      model: "gemini-1.5-flash"
      api_key_env: "GOOGLE_API_KEY"

    export GOOGLE_API_KEY="your-key"

### Switch to HuggingFace (Mistral 7B)

    llm:
      provider: "huggingface"
      model: "mistralai/Mistral-7B-Instruct-v0.2"
      api_key_env: "HF_API_KEY"

    export HF_API_KEY="your-key"

---

## Common Issues and Fixes

### FAISS index not found
    Error: FileNotFoundError: FAISS index not found
    Fix:   Run Steps 5b and 5d before running the agentic system.

### API key not set
    Error: API key environment variable 'OPENAI_API_KEY' is not set
    Fix:   Run: export OPENAI_API_KEY="your-key"

### Module not found errors
    Error: ModuleNotFoundError: No module named 'faiss'
    Fix:   Run: pip install faiss-cpu

### Config path errors
    Error: FileNotFoundError: config.yaml not found
    Fix:   Verify you are running scripts from the correct directory.
           Adjust relative paths in config.yaml if needed.

### Low retrieval accuracy (Top-1 < 70%)
    Cause: Only 10 sample entries in KB — too few for full evaluation
    Fix:   Expand KB to 68-88 entries before running formal evaluation.
           Review embedding_text quality in existing entries.

---

## Execution Order Summary

    Step 1  Create folder structure
    Step 2  Create virtual environment
    Step 3  Install requirements.txt
    Step 4  Set API key environment variable
    Step 5  Build knowledge base (validate → L1 index → L2 index → merge)
    Step 6  Verify config.yaml paths
    Step 7  Run end-to-end tests
    Step 8  Use CLI for interactive queries
    Step 9  Start FastAPI for demo
    Step 10 Run evaluation suite for dissertation metrics
