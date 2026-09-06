Use the same embedding model for both Layer 1 and Layer 2. This is non-negotiable — mixing embedding models breaks similarity comparisons across layers.
DecisionValueModelall-MiniLM-L6-v2 (HuggingFace sentence-transformers)
Dimension384
What gets embeddedtext field for Layer 2 chunks (same as embedding_text in Layer 1)
Batch size32–64 chunks per batch
For Layer 2, the full chunk text is embedded directly — no template needed, since the text is already narrative prose.

Stage 5 — How Layer 1 and Layer 2 Coexist in FAISS
This is an important architectural point. Both layers share one FAISS index but are distinguished by their metadata.
FAISS Index (flat or IVF)
├── Vector 0001  →  Layer 1 entry  KB-SQL-001
├── Vector 0002  →  Layer 1 entry  KB-SQL-002
├── ...
├── Vector 0088  →  Layer 1 entry  KB-WRK-014
├── Vector 0089  →  Layer 2 chunk  L2-CHUNK-0001
├── Vector 0090  →  Layer 2 chunk  L2-CHUNK-0002
├── ...
└── Vector 0300  →  Layer 2 chunk  L2-CHUNK-0212
A separate metadata store (a JSON file or SQLite table) maps each FAISS vector index position to its full metadata object. At retrieval time:

FAISS returns Top-K vector indices and similarity scores
The metadata store is looked up by index position to retrieve full entry/chunk details
The Retrieval Agent filters and ranks results using metadata fields (layer, category, severity, confidence_score)
Layer 1 entries are preferred over Layer 2 chunks when similarity scores are comparable — Layer 1 is more reliable

Retrieval Preference Logic (for Reasoning Agent)
If Layer 1 entry similarity >= 0.75  →  Use Layer 1 entry as primary
If Layer 1 entry similarity < 0.75 and Layer 2 chunk similarity >= 0.65  →  Use Layer 2 as supporting context
If both layers return results  →  Present Layer 1 as recommendation, Layer 2 as additional context

Recommended Layer 2 Source Documents (Starting Set)
DocumentSource TypeEstimated ChunksCategory CoverageApache Airflow Best PracticesPublic guide15–20ETL, Workflow ExecutionSpark Performance Tuning GuidePublic guide20–25SQL, ETL, CachingBigQuery SQL Best PracticesPublic guide10–15SQL, Partitioning, IndexingAWS Glue ETL Developer GuidePublic guide10–15ETL, StoragePostgreSQL Index DocumentationPublic guide8–10Indexing StrategyDatabricks Delta Lake GuidePublic guide10–12ETL, Storage, PartitioningInternal Confluence pages*InternalTBDAll categories
*Pending organizational permission verification.
Estimated total Layer 2 corpus: 73–97 chunks — a healthy ratio relative to 68–88 Layer 1 entries.

⚠️ One Scope Warning
Automatically classifying chunks into taxonomy categories using an LLM at index time would be academically interesting but adds significant complexity. For this dissertation, assign category and subcategory manually or semi-manually during chunking — it takes 2–3 minutes per document and is far more reliable. This is the correct M.Tech scope decision.