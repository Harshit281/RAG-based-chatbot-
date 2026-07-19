# Scripts — Core RAG Pipeline Modules

This directory contains the **core pipeline modules** that implement all six stages of the RAG (Retrieval-Augmented Generation) workflow. Each module is a standalone Python file (`.py` only — no notebooks) responsible for a single stage.

> Each module's detailed README section is also available in the [root README](../README.md#rag-pipeline-steps).

---

## Pipeline Flow

```
 ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌──────────────┐    ┌────────────┐    ┌────────────┐
 │ load_data  │───►│ chunk_data │───►│ embed_data │───►│ vector_store │───►│ retriever  │───►│ generator  │
 │   .py      │    │   .py      │    │   .py      │    │   .py        │    │   .py      │    │   .py      │
 │            │    │            │    │            │    │              │    │            │    │            │
 │ Step 1:    │    │ Step 2:    │    │ Step 3:    │    │ Step 4:      │    │ Step 5:    │    │ Step 6:    │
 │ Data       │    │ Chunking   │    │ Embedding  │    │ Vector       │    │ Retrieval  │    │ Generation │
 │ Loading    │    │            │    │            │    │ Storage      │    │            │    │            │
 └────────────┘    └────────────┘    └────────────┘    └──────────────┘    └────────────┘    └────────────┘
```

---

## Module Reference

### 0. `prepare_opp115.py` — OPP-115 Data Preparation

Fetches the OPP-115 privacy policy dataset from Hugging Face and populates the SQLite database.

**Usage:**

```bash
pip install datasets          # one-time install
python Scripts/prepare_opp115.py
```

**What it does:**

1. Drops and recreates the `policy_data` table in `Data/policy_data.db`.
2. Loads the `alzoubi36/opp_115` dataset (train split) from Hugging Face.
3. Maps integer labels to human-readable privacy category names.
4. Inserts all rows in a single atomic transaction with duplicate-ID handling.
5. Verifies the final table and prints a topic distribution summary.

**Dependencies:** `datasets` (Hugging Face), `sqlite3`, `os`, `re`

---

### 1. `load_data.py` — Data Loading & Validation

Reads the policy data from the SQLite database and returns a lightweight `PolicyData` object.

**Classes & Functions:**

| Name | Type | Description |
|---|---|---|
| `PolicyData(rows)` | Class | Lightweight wrapper around a list of row dicts. Supports `len()` and `iterrows()` (yields `(index, row_dict)` pairs), matching the API surface that `chunk_data.py` expects. |
| `load_db` | `(filepath: str) -> PolicyData` | Loads the `policy_data` table from a SQLite database. Validates that the table exists, is non-empty, and has the expected columns. Raises `FileNotFoundError` if the file is missing or `ValueError` if the table is absent or empty. |

**Dependencies:** `sqlite3`, `os` (stdlib only)

---

### 2. `chunk_data.py` — Row → Chunk Conversion

Transforms each `PolicyData` row into a dictionary suitable for embedding and retrieval.

**Functions:**

| Function | Signature | Description |
|---|---|---|
| `create_chunks` | `(df: PolicyData) → List[Dict]` | Iterates over rows and produces chunk dictionaries using `Utils.text_cleaning` for normalisation. |

**Chunk dictionary schema:**

```python
{
    "id":       "OPP-0001",                                    # Row ID (string)
    "topic":    "Data Retention",                              # Privacy policy category
    "content":  "We will retain your information for as...",   # Original content text
    "ref_code": "",                                            # Extracted reference code (may be "")
    "text":     "Data Retention. We will retain your...",      # Normalised text for embedding
}
```

**Dependencies:** `Utils.text_cleaning`

---

### 3. `embed_data.py` — Embedding Model & Text Encoding

Loads a sentence-transformers model and converts text into dense vector representations.

**Functions:**

| Function | Signature | Description |
|---|---|---|
| `load_embedding_model` | `(model_name: str = "all-MiniLM-L6-v2") → SentenceTransformer` | Downloads (on first use) and loads the specified embedding model. |
| `generate_embeddings` | `(model: SentenceTransformer, texts: List[str]) → np.ndarray` | Encodes a list of strings into a 2-D NumPy array. Non-string inputs are handled gracefully. Single embeddings are expanded to 2-D. |

**Dependencies:** `sentence-transformers`, `numpy`

---

### 4. `vector_store.py` — Vector Index (FAISS / scikit-learn Fallback)

Provides a unified `VectorStore` class that abstracts the underlying vector search backend.

**Classes & Functions:**

| Name | Type | Description |
|---|---|---|
| `VectorStore(embeddings)` | Class | Builds an L2 nearest-neighbour index over the provided embeddings. Uses FAISS `IndexFlatL2` if available; otherwise falls back to `sklearn.neighbors.NearestNeighbors`. |
| `VectorStore.search(query_embedding, top_k=5)` | Method | Returns `(distances, indices)` NumPy arrays for the `top_k` nearest vectors. |
| `save_index(store, path)` | Function | Persists the index to disk — native FAISS format or Python pickle for the sklearn fallback. |
| `load_index(path, embeddings)` | Function | Restores a `VectorStore` from a saved index file. |

**Dependencies:** `numpy`, `faiss-cpu` (optional), `scikit-learn` (fallback)

---

### 5. `retriever.py` — Multi-Query Retrieval & Ranking

Expands the user query into multiple variants, searches the vector store for each, and returns merged, deduplicated, ranked results.

**Functions:**

| Function | Signature | Description |
|---|---|---|
| `retrieve` | `(query, embedding_model, vector_store, chunks, top_k=5, multi_query=True) → List[Dict]` | Main entry point. Returns the top-k most relevant chunk dictionaries. |

**Multi-query variants generated (when `multi_query=True`):**

| # | Variant Template | Example (query = `"data retention"`) |
|---|---|---|
| 1 | `<query>` | `data retention` |
| 2 | `<query> policy` | `data retention policy` |
| 3 | `company policy for <query>` | `company policy for data retention` |
| 4 | `guidelines for <query>` | `guidelines for data retention` |

**Ranking algorithm:**
1. Results from all variants are merged; duplicate chunk indices are combined.
2. Each chunk is scored by **average L2 distance** across all variants that matched it.
3. Ties are broken by **hit count** (chunks appearing in more variants rank higher).
4. The top-k chunks by this combined score are returned.

**Dependencies:** `Scripts.embed_data`, `Scripts.vector_store`

---

### 6. `generator.py` — Groq LLM Answer Generation

Builds a grounded prompt from retrieved context and calls the Groq chat completions API.

**Functions:**

| Function | Signature | Description |
|---|---|---|
| `setup_groq_api` | `(api_key: str, api_url: str = None) → dict` | Reads environment variables and returns an API configuration dictionary (`api_key`, `api_url`, `model`, `temperature`, `max_output_tokens`). |
| `generate_answer` | `(query: str, chunks: List[Dict], api_config: dict) → str` | Builds context from chunks (truncating to 900 chars each), constructs a system-grounded prompt, calls the Groq `/chat/completions` endpoint, and returns the parsed answer string. |

**System prompt:**
> *"You are a policy assistant. Answer the user's question using only the context provided. If the answer is not contained in the context, say you are not able to answer from the available policy information."*

**Internal helpers:**

| Function | Description |
|---|---|
| `_build_prompt(query, context)` | Constructs the full prompt string. |
| `_shorten_text(text, max_chars)` | Truncates text at a word boundary. |
| `_build_context(chunks, max_chars_per_chunk)` | Formats chunks into a context string with headers. |
| `_extract_groq_response(response_json)` | Robust parser that handles multiple Groq/OpenAI response envelope formats. |

**Dependencies:** `requests`, `os`
