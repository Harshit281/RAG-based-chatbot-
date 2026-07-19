# Privacy Policy Q&A Assistant (RAG Chatbot)

A complete **Retrieval-Augmented Generation (RAG)** chatbot built in Python that answers questions about how companies collect, use, share, retain, and secure personal data. It uses the **OPP-115** (Online Privacy Policies) dataset — a well-known NLP research dataset built from real, published privacy policies of actual companies — as its knowledge base, stored in a SQLite database.

The chatbot loads 2,185 annotated privacy policy segments, chunks and embeds them into a vector store, retrieves the most relevant passages for any user question, and generates a grounded natural-language answer using a Groq-hosted LLM — all from an interactive terminal interface.

> **Why OPP-115?** This project was originally built with synthetic/hand-written HR policy data. Per our internship coordinator's guidance to use a standard, real-world dataset for production-readiness, we migrated to OPP-115 — a research-grade dataset of real company privacy policies annotated into 10+ standard categories (data collection, third-party sharing, user choice, data retention, security, etc.).

---

## Table of Contents

1. [Overview](#overview)
2. [Sample Questions](#sample-questions)
3. [RAG Pipeline Steps](#rag-pipeline-steps)
4. [Repository Structure](#repository-structure)
5. [Directory Details](#directory-details)
6. [Prerequisites](#prerequisites)
7. [Installation & Setup](#installation--setup)
8. [Running the Chatbot](#running-the-chatbot)
9. [Configuration Reference](#configuration-reference)
10. [Architecture & Data Flow](#architecture--data-flow)
11. [Key Design Decisions](#key-design-decisions)
12. [Extending the Knowledge Base](#extending-the-knowledge-base)
13. [Troubleshooting](#troubleshooting)
14. [License](#license)

---

## Overview

This project demonstrates a **standard, end-to-end RAG implementation** using the OPP-115 privacy policy dataset as the knowledge base. It covers every core stage of the RAG pipeline:

| # | RAG Step | Module | Description |
|---|---|---|---|
| 1 | **Data Loading** | `Scripts/load_data.py` | Reads the policy SQLite database and validates the required schema (`ID`, `Topic`, `Content`). |
| 2 | **Chunking** | `Scripts/chunk_data.py` | Converts each database row into a retrieval-ready chunk with normalised text, topic metadata, and extracted reference codes. |
| 3 | **Embedding** | `Scripts/embed_data.py` | Encodes chunk text into dense vector representations using a `sentence-transformers` model (`all-MiniLM-L6-v2` by default). |
| 4 | **Vector Storage** | `Scripts/vector_store.py` | Stores embeddings in a FAISS `IndexFlatL2` index for fast similarity search (falls back to scikit-learn `NearestNeighbors` if FAISS is unavailable). |
| 5 | **Retrieval** | `Scripts/retriever.py` | Embeds the user query (with optional multi-query expansion), searches the vector store, deduplicates and ranks results by average distance. |
| 6 | **Generation** | `Scripts/generator.py` | Constructs a grounded prompt from retrieved context, sends it to the Groq LLM via the OpenAI-compatible chat completions API, and parses the response. |

The interactive REPL loop in `main.py` orchestrates these steps and presents results through a rich, ANSI-coloured terminal UI.

---

## Sample Questions

Here are some example questions you can ask the chatbot, grouped by OPP-115 privacy topic:

| Privacy Topic | Example Questions |
|---|---|
| **Data Retention** | "How long does the company retain my personal information?" · "Does the company store my data indefinitely?" |
| **Data Security** | "What security measures are used to protect user data?" · "How is my personal information encrypted?" |
| **Third Party Sharing** | "Does the company share my data with third-party advertisers?" · "What information is shared with third-party providers?" |
| **User Choice & Control** | "Can I opt out of targeted advertising?" · "How do cookies and tracking technologies affect my privacy?" |
| **User Access & Deletion** | "How can I request to delete my personal data?" · "What happens to my data if I cancel my account?" |
| **Do Not Track** | "How does the company handle Do Not Track browser signals?" |
| **Policy Changes** | "How will I be notified if the privacy policy changes?" |
| **International Audiences** | "Is my data transferred or stored internationally?" · "What privacy protections exist for children under 13?" |

---

## RAG Pipeline Steps

Below is a detailed breakdown of how each RAG stage is implemented.

### Step 1 — Data Loading (`Scripts/load_data.py`)

- Reads `Data/policy_data.db` (SQLite) using the `sqlite3` standard library.
- Validates that the `policy_data` table exists and contains the required columns: `ID`, `Topic`, `Content`.
- Returns a lightweight `PolicyData` object (supports `len()` and `iterrows()`) so the downstream pipeline works unchanged.
- Raises `FileNotFoundError` if the database is missing, or `ValueError` if the table is absent or empty.

### Step 2 — Chunking (`Scripts/chunk_data.py`)

- Iterates over each row from the `PolicyData` object.
- For each row, creates a **chunk dictionary** with the following keys:

  | Key | Source | Purpose |
  |---|---|---|
  | `id` | `row['ID']` (as string) | Unique identifier |
  | `topic` | `row['Topic']` | Privacy policy category |
  | `content` | `row['Content']` | Original privacy policy text |
  | `ref_code` | Extracted via regex from `Content` | Citation code (if present) |
  | `text` | Normalised `Topic + Content` | Clean text used for embedding |

- Uses `Utils/text_cleaning.py` for whitespace normalisation and ref-code extraction.

### Step 3 — Embedding (`Scripts/embed_data.py`)

- Loads a `sentence-transformers` model (default: `all-MiniLM-L6-v2`, configurable via `.env`).
- Encodes all chunk texts into a 2-D NumPy array of dense vectors.
- Handles edge cases: non-string inputs are replaced with empty strings; single embeddings are expanded to 2-D.
- **Caching**: After the first run, embeddings are saved to `Data/cache/embeddings.npy` so subsequent launches skip this step entirely.

### Step 4 — Vector Storage (`Scripts/vector_store.py`)

- Wraps the vector index behind a `VectorStore` class with a unified `.search()` API.
- **Primary backend**: FAISS `IndexFlatL2` (exact L2 nearest-neighbour search).
- **Fallback backend**: scikit-learn `NearestNeighbors` (used automatically when `faiss-cpu` is not installed).
- Provides `save_index()` / `load_index()` for persisting and restoring the index to `Data/cache/faiss.index`.
- **Caching**: The built index is saved after the first run and loaded from disk on subsequent launches.

### Step 5 — Retrieval (`Scripts/retriever.py`)

- **Multi-query expansion** (enabled by default, configurable): generates four variants of the user query:
  1. `<query>` (original)
  2. `<query> policy`
  3. `company policy for <query>`
  4. `guidelines for <query>`
- Each variant is embedded and searched against the vector store independently.
- Results are **merged and deduplicated** by chunk index; chunks that appear across multiple variants are boosted.
- Final ranking is by **average L2 distance** (lower is better), with tie-breaking by **hit count** (higher is better).
- Returns the top-k ranked chunks (default `k=5`).

### Step 6 — Generation (`Scripts/generator.py`)

- Selects the top `GENERATION_TOP_K` chunks (default 3) from the retrieval results.
- Builds a context string from chunk text, truncating individual chunks to 900 characters if needed.
- Constructs a **system-grounded prompt**:
  > *"You are a policy assistant. Answer the user's question using only the context provided. If the answer is not contained in the context, say you are not able to answer from the available policy information."*
- Sends the prompt to the Groq API via the OpenAI-compatible `/chat/completions` endpoint.
- Parses the response using a robust extractor that handles multiple response envelope formats.
- The final answer is displayed in the terminal with source citations.

---

## Repository Structure

The project is organised into **three core directories** plus root-level configuration files:

```
rag-policy-chatbot/
│
├── main.py                      # Application entry point — orchestrates the full RAG pipeline
├── requirements.txt             # Python package dependencies
├── .env                         # API keys and runtime configuration (git-ignored)
├── .env.example                 # Template for .env with default values
├── .gitignore                   # Excludes .env, cache/, __pycache__/, *.pyc
├── README.md                    # This file — repository-level documentation
│
├── Data/                        # Knowledge base and cached artifacts
│   ├── README.md                # Documentation for the Data directory
│   ├── policy_data.db           # OPP-115 privacy policy database (2,185 entries)
│   └── cache/                   # Auto-generated on first run (git-ignored)
│       ├── embeddings.npy       #   Cached sentence-transformer embeddings
│       └── faiss.index          #   Cached FAISS vector index
│
├── Scripts/                     # Core RAG pipeline modules (Python only)
│   ├── README.md                # Documentation for the Scripts directory
│   ├── __init__.py              # Package initialiser
│   ├── prepare_opp115.py        # Data preparation: fetches OPP-115 from Hugging Face → policy_data.db
│   ├── load_data.py             # Step 1: Data loading & SQLite validation
│   ├── chunk_data.py            # Step 2: Row-to-chunk conversion
│   ├── embed_data.py            # Step 3: Embedding model & text encoding
│   ├── vector_store.py          # Step 4: FAISS / scikit-learn vector index
│   ├── retriever.py             # Step 5: Multi-query retrieval & ranking
│   └── generator.py             # Step 6: Groq LLM prompt building & answer generation
│
└── Utils/                       # Shared utility & helper scripts
    ├── README.md                # Documentation for the Utils directory
    ├── __init__.py              # Package initialiser
    ├── text_cleaning.py         # Text normalisation, whitespace cleanup, ref-code extraction
    └── formatting.py            # ANSI terminal colours, banners, status messages, output formatting
```

> **Note:** All scripts are `.py` files only — no Jupyter notebooks (`.ipynb`) are included in this repository.

---

## Directory Details

### `Data/` — Knowledge Base

Contains the OPP-115 privacy policy database and auto-generated cache files. See [`Data/README.md`](Data/README.md) for full details.

| File | Description |
|---|---|
| `policy_data.db` | The primary knowledge base — **2,185 OPP-115 privacy policy segments** stored in a SQLite database with `ID`, `Topic`, and `Content` columns. Topics span 12 privacy categories (First Party Collection, User Choice/Control, Data Security, Data Retention, Third Party Sharing, Do Not Track, and more). Sourced from [`alzoubi36/opp_115`](https://huggingface.co/datasets/alzoubi36/opp_115) on Hugging Face. |
| `cache/` | Auto-created directory storing `embeddings.npy` and `faiss.index` after the first run. Listed in `.gitignore`. Delete to force a full rebuild. |

### `Scripts/` — RAG Pipeline Modules

Contains one Python module per RAG pipeline stage plus the data preparation script. See [`Scripts/README.md`](Scripts/README.md) for function signatures and detailed API docs.

| Module | RAG Step | Key Functions |
|---|---|---|
| `prepare_opp115.py` | Data Preparation | Fetches OPP-115 from Hugging Face → `policy_data.db` |
| `load_data.py` | Data Loading | `load_db(filepath) → PolicyData` |
| `chunk_data.py` | Chunking | `create_chunks(df) → List[Dict]` |
| `embed_data.py` | Embedding | `load_embedding_model(name)`, `generate_embeddings(model, texts)` |
| `vector_store.py` | Vector Storage | `VectorStore(embeddings)`, `.search()`, `save_index()`, `load_index()` |
| `retriever.py` | Retrieval | `retrieve(query, model, store, chunks, top_k, multi_query)` |
| `generator.py` | Generation | `setup_groq_api(key)`, `generate_answer(query, chunks, config)` |

### `Utils/` — Utility Helpers

Contains formatting and text-cleaning helpers used across the pipeline. See [`Utils/README.md`](Utils/README.md) for the full API reference.

| Module | Purpose | Key Functions |
|---|---|---|
| `text_cleaning.py` | Text normalisation | `clean_text()`, `extract_ref_code()`, `normalize_text()` |
| `formatting.py` | Terminal UI output | `print_banner()`, `print_status()`, `format_context()`, `format_response()`, and more |

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.9 or later |
| **Groq API key** | Sign up at [console.groq.com](https://console.groq.com/) to obtain a free API key |
| **Internet access** | Required on first run to download the `sentence-transformers` model (~80 MB) and for Groq API calls |

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Harshit281/RAG-based-chatbot-.git
cd RAG-based-chatbot-
```

### 2. Create a Virtual Environment (recommended)

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | Purpose |
|---|---|
| `datasets` | Fetching the OPP-115 dataset from Hugging Face (used by `prepare_opp115.py`) |
| `sentence-transformers` | Local embedding model for semantic search |
| `faiss-cpu` | Fast approximate nearest-neighbour vector index |
| `scikit-learn` | Fallback nearest-neighbour search (if FAISS is unavailable) |
| `python-dotenv` | Loads environment variables from `.env` |
| `requests` | HTTP calls to the Groq API |

### 4. Configure the `.env` File

Copy the template and fill in your Groq API key:

```bash
cp .env.example .env
```

Then edit `.env`:

```dotenv
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional overrides (defaults shown)
GROQ_API_URL=https://api.groq.com/openai/v1
GROQ_MODEL=ALLaM-2-7b
GROQ_TEMPERATURE=0.2
GROQ_MAX_OUTPUT_TOKENS=512
EMBEDDING_MODEL=all-MiniLM-L6-v2
MULTI_QUERY_ENABLED=true
TOP_K=5
GENERATION_TOP_K=3
```

### 5. Prepare the Database (if not already present)

The repository ships with a pre-built `Data/policy_data.db`. To rebuild it from scratch:

```bash
pip install datasets          # one-time install
python Scripts/prepare_opp115.py
```

---

## Running the Chatbot

```bash
python main.py
```

**First-run sequence:**

1. Loads and validates `Data/policy_data.db` (2,185 OPP-115 privacy policy segments).
2. Creates chunks from each database row.
3. Generates embeddings using `sentence-transformers` (saved to `Data/cache/embeddings.npy`).
4. Builds a FAISS vector index (saved to `Data/cache/faiss.index`).
5. Loads the embedding model for query-time encoding.
6. Enters the interactive REPL — type a privacy policy question and press Enter.

**Subsequent runs** load cached embeddings and the index from disk, making startup near-instant.

**Example session:**

```
? Ask a policy question: How long does the company retain my data?

──────────────── 📖 RETRIEVED SNIPPETS ────────────────
  1. Data Retention
     Data retention policy We will retain your information for
     as long as your account is active or as needed to provide
     you with services...

══════════════════ 🤖 ANSWER ══════════════════════════
  The company keeps your data for as long as your account
  remains active or as long as it's needed to provide you
  with its services...
```

**To exit:** Type `quit` or `exit` at the prompt.

---

## Configuration Reference

All settings are controlled via environment variables in the `.env` file. Only `GROQ_API_KEY` is required.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Your Groq API key for LLM generation |
| `GROQ_API_URL` | `https://api.groq.com/openai/v1` | Base URL for the Groq OpenAI-compatible API |
| `GROQ_MODEL` | `ALLaM-2-7b` | Model identifier to use for answer generation |
| `GROQ_TEMPERATURE` | `0.2` | Sampling temperature (0 = deterministic, 2 = max randomness) |
| `GROQ_MAX_OUTPUT_TOKENS` | `512` | Maximum number of tokens in the generated answer |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model for embedding queries and chunks |
| `MULTI_QUERY_ENABLED` | `true` | Whether to expand the user query into multiple search variants |
| `TOP_K` | `5` | Number of chunks to retrieve from the vector store |
| `GENERATION_TOP_K` | `3` | Number of retrieved chunks to include in the LLM prompt context |

---

## Architecture & Data Flow

```
                            ┌──────────────────────────────────────────────┐
                            │              main.py (Orchestrator)          │
                            └──────────────────────┬───────────────────────┘
                                                   │
          ┌────────────────────────────────────────┼────────────────────────────────────────┐
          │                                        │                                        │
          ▼                                        ▼                                        ▼
  ┌───────────────┐                      ┌─────────────────┐                      ┌─────────────────┐
  │   Data/       │                      │   Scripts/      │                      │   Utils/        │
  │               │                      │                 │                      │                 │
  │ policy_data   │◄────── loaded by ────│ load_data.py    │                      │ text_cleaning   │
  │   .db         │                      │                 │                      │   .py           │
  │               │                      │ chunk_data.py ──┼── uses ─────────────►│                 │
  │ cache/        │◄── saved/loaded ─────│ embed_data.py   │                      │ formatting.py   │
  │  embeddings   │                      │ vector_store.py │                      │   (terminal UI) │
  │  faiss.index  │                      │ retriever.py    │                      └─────────────────┘
  └───────────────┘                      │ generator.py ───┼── calls ──► Groq API
                                         └─────────────────┘
```

**Query-time flow (per user question):**

```
  User question
       │
       ▼
  ┌─────────────────────────────────┐
  │  1. Multi-query expansion       │  retriever.py
  │     "data retention"            │  → "data retention"
  │                                 │  → "data retention policy"
  │                                 │  → "company policy for data retention"
  │                                 │  → "guidelines for data retention"
  └──────────────┬──────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────┐
  │  2. Embed each query variant    │  embed_data.py
  └──────────────┬──────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────┐
  │  3. Search vector store         │  vector_store.py
  │     (FAISS IndexFlatL2)         │
  └──────────────┬──────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────┐
  │  4. Merge, deduplicate & rank   │  retriever.py
  │     by avg distance + hit count │
  └──────────────┬──────────────────┘
                 │  top-k chunks
                 ▼
  ┌─────────────────────────────────┐
  │  5. Build context & prompt      │  generator.py
  │     "Answer using ONLY this     │
  │      context..."                │
  └──────────────┬──────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────┐
  │  6. Call Groq API               │  generator.py → Groq /chat/completions
  └──────────────┬──────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────┐
  │  7. Format & display answer     │  formatting.py
  │     with source citations       │
  └─────────────────────────────────┘
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **OPP-115 as data source** | A standard, research-grade NLP dataset of real company privacy policies from Hugging Face ([`alzoubi36/opp_115`](https://huggingface.co/datasets/alzoubi36/opp_115)). Provides 2,185 annotated segments across 12 privacy categories — far more representative than hand-written synthetic data. Chosen per internship coordinator guidance to use a real-world dataset. |
| **SQLite as knowledge base** | Schema enforcement, transactional writes, and built-in queryability — no CSV-escaping issues. The `prepare_opp115.py` script populates it from Hugging Face, ensuring reproducible builds. `sqlite3` is a Python stdlib module, so no extra dependency is needed at runtime. |
| **`sentence-transformers` for local embeddings** | Runs offline after initial model download. No API cost for embedding. `all-MiniLM-L6-v2` provides a good balance of quality and speed. |
| **FAISS with scikit-learn fallback** | FAISS offers high-performance vector search but may not install cleanly on all platforms. The automatic fallback ensures the chatbot works everywhere. |
| **Multi-query expansion** | Privacy policy questions can be phrased in many ways. Expanding to multiple query variants improves recall without requiring the user to guess the "right" wording. |
| **Embedding & index caching** | Avoids recomputing embeddings (~10–30 seconds) and the FAISS index on every launch. Cache lives in `Data/cache/` and is git-ignored. |
| **Groq OpenAI-compatible API** | Groq provides fast inference. Using the OpenAI-compatible endpoint means switching to another provider is a one-line `.env` change. |
| **Grounded generation prompt** | The system prompt explicitly constrains the LLM to answer only from provided context, reducing hallucination. |
| **ANSI terminal UI with ASCII fallback** | Provides a rich experience on modern terminals while remaining functional on legacy Windows `cmd.exe`. Respects the `NO_COLOR` standard. |
| **Python-only scripts (no `.ipynb`)** | All source code is in `.py` files for clean version control, easy importing, and straightforward command-line execution. |

---

## Extending the Knowledge Base

### Refreshing from OPP-115

Re-run the data preparation script to fetch the latest dataset and rebuild the database:

```bash
pip install datasets          # one-time install
python Scripts/prepare_opp115.py
```

Then delete the cache and restart:

```bash
# Mac/Linux
rm -rf Data/cache

# Windows
rmdir /s /q Data\cache
```

```bash
python main.py
```

### Adding custom entries

You can add rows directly to the `policy_data` table in `Data/policy_data.db` using any SQLite client (e.g., [DB Browser for SQLite](https://sqlitebrowser.org/), the `sqlite3` CLI, or Python):

```sql
INSERT INTO policy_data (ID, Topic, Content)
VALUES ('CUSTOM-001', 'Data Security', 'Full policy text here.');
```

After adding or modifying data:

1. **Delete the cache**: remove `Data/cache/` so embeddings and the index are rebuilt.
2. **Re-run**: `python main.py` — the chatbot will automatically reprocess and re-index all entries.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `GROQ_API_KEY is missing or still set to a placeholder` | Edit `.env` and set `GROQ_API_KEY` to your real Groq API key. |
| `faiss-cpu` fails to install | The chatbot will automatically fall back to scikit-learn. You can safely remove `faiss-cpu` from `requirements.txt` if needed. |
| Unicode / box-drawing characters display as `?` | Your terminal may not support UTF-8. The chatbot auto-detects this and falls back to ASCII characters. Set `chcp 65001` on Windows CMD for UTF-8 support. |
| `ModuleNotFoundError: No module named 'Scripts'` | Make sure you are running `python main.py` from the project root directory (`rag-policy-chatbot/`). |
| `policy_data.db` is missing or empty | Run `python Scripts/prepare_opp115.py` to fetch the OPP-115 dataset and create the database. |
| Slow first launch | On the first run, embeddings are generated for all 2,185 policy segments (~30–90 seconds depending on hardware). Subsequent runs load from cache and start in under 2 seconds. |
| Stale answers after updating the database | Delete `Data/cache/` and restart to force re-embedding and re-indexing. |

---

## License

This project is provided as-is for educational and research use.
