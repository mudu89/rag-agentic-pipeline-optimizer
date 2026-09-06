Step 5 — Knowledge Base Build Plan
Overview
This plan takes you from an empty folder to a fully populated, queryable FAISS knowledge base. It is structured as 6 milestones with clear deliverables, effort estimates, and validation checkpoints at each stage.

Recommended Folder Structure
dissertation_project/
├── knowledge_base/
│   ├── schema/
│   │   └── kb_schema_v1.json              ← Already done ✅
│   ├── taxonomy/
│   │   └── kb_taxonomy_v1.json            ← Already done ✅
│   ├── entries/
│   │   ├── kb_entries_sample_v1.json      ← Already done ✅ (10 entries)
│   │   └── kb_entries_full_v1.json        ← To be built (68–88 entries)
│   ├── documents/
│   │   ├── raw/                           ← Raw source documents (PDF, MD, TXT)
│   │   └── processed/                     ← Cleaned plain text versions
│   ├── chunks/
│   │   └── kb_chunks_layer2_v1.json       ← Layer 2 chunks after processing
│   ├── index/
│   │   ├── faiss_index.bin                ← Serialized FAISS index
│   │   └── metadata_store.json            ← Maps FAISS vector ID → full entry/chunk
│   └── scripts/
│       ├── validate_entries.py            ← Schema validation script
│       ├── build_layer1_index.py          ← Embed + index Layer 1 entries
│       ├── build_layer2_index.py          ← Chunk + embed + index Layer 2 docs
│       ├── merge_indexes.py               ← Merge L1 and L2 into single FAISS index
│       └── query_kb.py                    ← Query the KB and retrieve Top-K results

Milestone Breakdown

Milestone 1 — Complete Layer 1 Entry Curation
Effort: 4–6 days | Output: kb_entries_full_v1.json
What to do
Expand the 10 sample entries to the full 68–88 target, distributed across all 7 categories using the taxonomy as your guide.
Curation Approach
Work category by category in this order — highest value first:
OrderCategoryTargetWhy This Order1SQL Optimization20–25Largest category; most testable2ETL & Pipeline Optimization15–20Core to your Comcast context3Partitioning Strategy8–10High impact; directly observable4Indexing Strategy8–10High impact; well-documented patterns5Caching Strategy6–8Overlaps SQL and ETL — fills naturally6Workflow Execution Optimization6–8Log-driven; good evaluation variety7Data Storage Optimization5–7Quickest to document; fills last
Curation Sources Per Category
CategoryRecommended SourcesSQL OptimizationBigQuery SQL Best Practices, PostgreSQL docs, Das 2020 paperETL & Pipeline OptimizationApache Airflow docs, Databricks Delta Lake guide, Kumar & Patel 2021Partitioning & IndexingSpark tuning guide, AWS Athena docs, Hive partitioning docsCachingSpark documentation, BigQuery caching guideWorkflow ExecutionAirflow task monitoring docs, AWS Glue developer guideData StorageParquet/ORC format docs, Databricks file format guide
Validation Rule Before Moving On
Every entry must pass this checklist before being added:
☐ entry_id follows naming convention (KB-SQL-XXX, KB-ETL-XXX, etc.)
☐ category and subcategory match taxonomy exactly
☐ before_example and after_example are concrete — not abstract
☐ embedding_text follows the template exactly
☐ confidence_score is set (0.7 minimum for inclusion)
☐ at least one related_entry cross-reference populated
☐ source is a real, citable document or guide

Milestone 2 — Layer 1 Schema Validation Script
Effort: 2–3 hours | Output: validate_entries.py
What it does
Reads kb_entries_full_v1.json and checks every entry against the schema rules before embedding. Catches errors early — far cheaper to fix JSON than to rebuild a FAISS index.
Validations to implement
1. All mandatory fields present
2. category value exists in taxonomy enum
3. subcategory value exists under correct parent category
4. input_type value in allowed enum
5. severity value in allowed enum
6. confidence_score is float between 0.0 and 1.0
7. embedding_text is non-empty string
8. before_example and after_example are non-empty
9. No duplicate entry_id values across the file
10. related_entries references point to existing entry_ids
Output format
Validation Report — kb_entries_full_v1.json
Total entries: 78
Passed: 76
Failed: 2

FAIL — KB-ETL-009: subcategory 'ETL Optimization' not found under 'ETL & Pipeline Optimization'
FAIL — KB-SQL-017: confidence_score 1.2 out of valid range (0.0–1.0)

Milestone 3 — Build Layer 1 FAISS Index
Effort: 3–4 hours | Output: faiss_index_layer1.bin + metadata_store_layer1.json
Process
Load kb_entries_full_v1.json
        │
        ▼
For each entry:
  extract embedding_text field
        │
        ▼
Batch encode with sentence-transformer
  model: all-MiniLM-L6-v2
  output: 384-dim vector per entry
        │
        ▼
Add all vectors to FAISS IndexFlatL2
        │
        ▼
Serialize FAISS index → faiss_index_layer1.bin
        │
        ▼
Build metadata_store_layer1.json
  { "0": {full entry KB-SQL-001}, "1": {full entry KB-SQL-002}, ... }
Validation Test After Build
Run 5 test queries and confirm results are semantically correct:
Test QueryExpected Top Result"SELECT * from large table causing slow query"KB-SQL-001 (Column Projection)"broadcast join small dimension table Spark"KB-SQL-002 (Join Optimization)"correlated subquery running slow on every row"KB-SQL-004 (Subquery Optimization)"full table reload every pipeline run too slow"KB-ETL-002 (Incremental Processing)"pipeline bottleneck one task taking hours"KB-WRK-001 (Bottleneck Detection)
All 5 must return the expected entry in Top-3 results. If any fail, the embedding text for that entry needs revision.

Milestone 4 — Process and Chunk Layer 2 Documents
Effort: 2–3 days | Output: kb_chunks_layer2_v1.json
Step-by-step
Step 1 — Collect source documents
  Download and save to knowledge_base/documents/raw/
  Recommended starting set (5–6 public documents):
    - apache_airflow_best_practices.md
    - spark_performance_tuning_guide.md
    - bigquery_sql_best_practices.md
    - postgresql_indexing_guide.md
    - databricks_delta_lake_guide.md

Step 2 — Preprocess each document
  Convert to plain text if PDF
  Remove headers, footers, navigation
  Normalize whitespace
  Save cleaned version to documents/processed/

Step 3 — Chunk each document
  Target: 400–500 tokens per chunk
  Overlap: 50–75 tokens
  Never split mid-code-block
  Assign chunk_id, chunk_index, token_count

Step 4 — Manually assign category + subcategory to each chunk
  Takes 2–3 minutes per document
  Use taxonomy as reference

Step 5 — Assemble into kb_chunks_layer2_v1.json
  Format matches Layer 2 metadata schema from Step 4
Expected output volume
5–6 source documents × 10–20 chunks each = 60–100 chunks

Milestone 5 — Build Layer 2 FAISS Index and Merge
Effort: 2–3 hours | Output: faiss_index.bin + metadata_store.json (combined)
Process
1. Load kb_chunks_layer2_v1.json
2. Extract text field from each chunk
3. Embed using same model — all-MiniLM-L6-v2
4. Build faiss_index_layer2.bin

5. Merge Layer 1 and Layer 2:
   - Load both FAISS indexes
   - Concatenate vectors into single IndexFlatL2
   - Rebuild unified metadata_store.json
     { "0": {layer1 entry}, ..., "87": {layer1 entry},
       "88": {layer2 chunk}, ..., "187": {layer2 chunk} }
   - Save as faiss_index.bin (the final production index)
Critical rule
The layer field in every metadata entry must be set correctly (1 or 2) — this is how the Retrieval Agent distinguishes entry types at query time.

Milestone 6 — End-to-End Retrieval Validation
Effort: 1 day | Output: Retrieval quality report
What to test
Run a structured set of 20 test queries across all 7 categories. For each query record:
Query text
Expected category
Returned Top-1 entry/chunk
Returned Top-3 entries/chunks
Layer of each result (1 or 2)
Similarity score
Correct? (Yes / No)
Metrics to compute from this test
MetricHow to computeTargetTop-1 Accuracy% of queries where correct entry is rank 1≥ 70%Top-3 Accuracy% of queries where correct entry is in Top 3≥ 85%Layer 1 preference rate% of Top-1 results from Layer 1≥ 60%Mean similarity scoreAverage cosine similarity of Top-1 results≥ 0.65
This validation report becomes Section 4.x of your dissertation — Retrieval Evaluation. Keep the raw results in a CSV file for reproducibility.

Complete Timeline
MilestoneTaskEffortCumulativeM1Complete Layer 1 curation (68–88 entries)4–6 daysDay 6M2Build and run validation script2–3 hoursDay 6M3Build Layer 1 FAISS index + spot test3–4 hoursDay 7M4Process and chunk Layer 2 documents2–3 daysDay 10M5Build Layer 2 index and merge2–3 hoursDay 10M6End-to-end retrieval validation (20 queries)1 dayDay 11
Total: 11 working days — well within your design and development phase window.

Scripts Summary
ScriptInputOutputWhen to runvalidate_entries.pykb_entries_full_v1.jsonValidation reportAfter M1, before M3build_layer1_index.pykb_entries_full_v1.jsonfaiss_index_layer1.bin + metadata_store_layer1.jsonM3build_layer2_index.pykb_chunks_layer2_v1.jsonfaiss_index_layer2.bin + metadata_store_layer2.jsonM5merge_indexes.pyBoth layer indexesfaiss_index.bin + metadata_store.jsonM5query_kb.pyQuery string + FAISS indexTop-K results with metadataM6 and ongoing

⚠️ Three Risks to Manage
RiskLikelihoodMitigationLayer 1 curation takes longer than expectedHigh — most time-intensive stepStart immediately; do 10 entries per day; use sample entries as templatesInternal Confluence documents not approved for useMediumStart with public documents only; add internal docs only if permissions confirmedRetrieval quality below 70% Top-1 accuracyMediumFix embedding_text quality first; try all-mpnet-base-v2 (768-dim) if MiniLM underperforms