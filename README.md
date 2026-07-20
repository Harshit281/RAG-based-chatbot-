<div align="center">

# 🛡️ RAG Policy Chatbot

### Retrieval-Augmented Generation for Privacy Policy Q&A

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Groq API](https://img.shields.io/badge/Groq-LLM%20API-F55036?style=for-the-badge&logo=openai&logoColor=white)](https://console.groq.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-0467DF?style=for-the-badge&logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Dataset](https://img.shields.io/badge/HuggingFace-OPP--115-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/alzoubi36/opp_115)
[![License](https://img.shields.io/badge/License-Educational%20%26%20Research-green?style=for-the-badge)](./README.md)

---

**An end-to-end RAG chatbot** that answers natural-language questions about how companies collect, use, share, retain, and secure personal data — powered by the **OPP-115** research dataset, **FAISS** vector search, and **Groq** LLM inference.

[Getting Started](#-getting-started) · [How It Works](#-how-it-works) · [Configuration](#%EF%B8%8F-configuration) · [Troubleshooting](#-troubleshooting)

</div>

---

## ✨ Highlights

| | Feature | Details |
|---|---|---|
| 📚 | **Real-World Data** | **2,185 annotated segments** from the [OPP-115](https://huggingface.co/datasets/alzoubi36/opp_115) privacy-policy dataset (12 categories) |
| 🔍 | **Multi-Query Retrieval** | Expands every question into 4 search variants for better recall |
| ⚡ | **Fast Vector Search** | FAISS `IndexFlatL2` with automatic scikit-learn fallback |
| 🤖 | **Grounded Generation** | Groq-hosted LLM answers *only* from retrieved context — no hallucination |
| 💾 | **Smart Caching** | Embeddings + index cached after first run — subsequent starts take < 2 s |
| 🎨 | **Rich Terminal UI** | ANSI colours, Unicode box-drawing, with full ASCII fallback |

---

## 🖥️ Demo

```
╭──────────────────────────────────────────────────────────────────────╮
│                                                                      │
│                       RAG Policy Chatbot                             │
│         Retrieval-augmented answers from company policy data         │
│                                                                      │
│                     Type quit or exit to stop                        │
│                                                                      │
╰──────────────────────────────────────────────────────────────────────╯

  ▸ Loading policy data...
  ✓ Loaded 2185 policy entries.
  ✓ Ready!

  ? Ask a policy question: How long does the company retain my data?

──────────────── 🔍 YOUR QUESTION ────────────────
  How long does the company retain my data?

──────────────── 📖 RETRIEVED SNIPPETS ────────────────
  1. Data Retention
     We will retain your information for as long as your
     account is active or as needed to provide you services...

══════════════════ 🤖 ANSWER ══════════════════════════
  The company keeps your data for as long as your account
  remains active or as long as it's needed to provide you
  with its services...

  Sources: DR-001
```

---

## 💬 Sample Questions

> Try these privacy-related questions to explore the chatbot's capabilities:

| Topic | Try asking… |
|---|---|
| **Data Retention** | *"How long does the company retain my personal information?"* |
| **Data Security** | *"What security measures are used to protect user data?"* |
| **Third Party Sharing** | *"Does the company share my data with advertisers?"* |
| **User Choice & Control** | *"Can I opt out of targeted advertising?"* |
| **User Access & Deletion** | *"How can I request to delete my personal data?"* |
| **Do Not Track** | *"How does the company handle Do Not Track browser signals?"* |
| **Policy Changes** | *"How will I be notified if the privacy policy changes?"* |
| **International** | *"Is my data transferred or stored internationally?"* |

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.9 or later |
| **Groq API Key** | Free at [console.groq.com](https://console.groq.com/) |
| **Internet** | First run only — downloads the embedding model (~80 MB) + Groq API calls |

### 1 — Clone & Setup

```bash
git clone https://github.com/Harshit281/RAG-based-chatbot-.git
cd RAG-based-chatbot-
```

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# macOS / Linux:
source .venv/bin/activate
```

### 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>📦 What gets installed</summary>

| Package | Purpose |
|---|---|
| `sentence-transformers` | Local embedding model for semantic search |
| `faiss-cpu` | High-performance vector nearest-neighbour index |
| `scikit-learn` | Fallback nearest-neighbour search |
| `datasets` | Fetching the OPP-115 dataset from Hugging Face |
| `python-dotenv` | Loads configuration from `.env` |
| `requests` | HTTP calls to the Groq API |

</details>

### 3 — Configure API Key

```bash
cp .env.example .env
```

Edit `.env` and replace the placeholder with your real Groq API key:

```dotenv
GROQ_API_KEY=gsk_your_actual_key_here
```

### 4 — Run

```bash
python main.py
```

> **First launch** generates embeddings and builds the vector index (~30–90 s). All subsequent launches load from cache and start in < 2 seconds.

Type `quit` or `exit` to leave the interactive session.

---

## 🔬 How It Works

### Architecture Overview

```
                         ┌──────────────────────────────────────────┐
                         │           main.py  (Orchestrator)        │
                         └────────────────────┬─────────────────────┘
                                              │
          ┌───────────────────────────────────┼───────────────────────────────────┐
          │                                   │                                   │
          ▼                                   ▼                                   ▼
  ┌───────────────┐                 ┌─────────────────┐                 ┌─────────────────┐
  │    Data/      │                 │    Scripts/      │                 │    Utils/        │
  │               │                 │                  │                 │                  │
  │ policy_data   │◄── loaded by ──│ load_data.py     │                 │ text_cleaning.py │
  │   .db         │                │                  │                 │                  │
  │               │                │ chunk_data.py ───┼── uses ────────►│                  │
  │ cache/        │◄── cached ─────│ embed_data.py    │                 │ formatting.py    │
  │  embeddings   │                │ vector_store.py  │                 │   (terminal UI)  │
  │  faiss.index  │                │ retriever.py     │                 └─────────────────┘
  └───────────────┘                │ generator.py ────┼── calls ──► Groq API
                                   └─────────────────┘
```

### The 6-Stage RAG Pipeline

```
 ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌──────────────┐   ┌────────────┐   ┌────────────┐
 │  Step 1    │──►│  Step 2    │──►│  Step 3    │──►│   Step 4     │──►│  Step 5    │──►│  Step 6    │
 │            │   │            │   │            │   │              │   │            │   │            │
 │  Load      │   │  Chunk     │   │  Embed     │   │  Index       │   │  Retrieve  │   │  Generate  │
 │  Data      │   │  Data      │   │  Text      │   │  Vectors     │   │  Context   │   │  Answer    │
 └────────────┘   └────────────┘   └────────────┘   └──────────────┘   └────────────┘   └────────────┘
  load_data.py     chunk_data.py    embed_data.py    vector_store.py    retriever.py     generator.py
```

| Step | Module | What Happens |
|---|---|---|
| **1. Load** | `load_data.py` | Reads `Data/policy_data.db` (SQLite), validates the `policy_data` table schema (`ID`, `Topic`, `Content`), returns a lightweight `PolicyData` object |
| **2. Chunk** | `chunk_data.py` | Converts each DB row into a chunk dict with normalised text, topic metadata, and extracted reference codes via `Utils/text_cleaning.py` |
| **3. Embed** | `embed_data.py` | Encodes all chunk texts into dense vectors using `sentence-transformers` (`all-MiniLM-L6-v2`) — cached to `Data/cache/embeddings.npy` |
| **4. Index** | `vector_store.py` | Stores embeddings in a FAISS `IndexFlatL2` index (auto-falls back to scikit-learn `NearestNeighbors`) — cached to `Data/cache/faiss.index` |
| **5. Retrieve** | `retriever.py` | Expands the user query into 4 variants, searches the index, deduplicates, and ranks results by average L2 distance × hit count |
| **6. Generate** | `generator.py` | Builds a grounded prompt from retrieved context, sends it to Groq `/chat/completions`, and parses the response |

### Query-Time Flow

```
  User: "How long is my data retained?"
       │
       ▼
  ┌─────────────────────────────────────┐
  │  Multi-query expansion              │
  │  → "How long is my data retained?"  │
  │  → "... retained? policy"           │
  │  → "company policy for ..."         │
  │  → "guidelines for ..."             │
  └──────────────┬──────────────────────┘
                 ▼
  ┌─────────────────────────────────────┐
  │  Embed each variant                 │
  │  (sentence-transformers)            │
  └──────────────┬──────────────────────┘
                 ▼
  ┌─────────────────────────────────────┐
  │  Search FAISS index (top-k per      │
  │  variant)                           │
  └──────────────┬──────────────────────┘
                 ▼
  ┌─────────────────────────────────────┐
  │  Merge + Deduplicate + Rank         │
  │  (avg L2 distance, hit count)       │
  └──────────────┬──────────────────────┘
                 ▼  top-k chunks
  ┌─────────────────────────────────────┐
  │  Build grounded prompt + call Groq  │
  │  "Answer ONLY from this context..." │
  └──────────────┬──────────────────────┘
                 ▼
  ┌─────────────────────────────────────┐
  │  Format answer with source          │
  │  citations → display in terminal    │
  └─────────────────────────────────────┘
```

---

## 📁 Project Structure

```
rag-policy-chatbot/
│
├── main.py                      # Entry point — orchestrates the full RAG pipeline
├── requirements.txt             # Python dependencies
├── .env.example                 # Template for API keys & config
├── .env                         # Your local config (git-ignored)
├── .gitignore
├── README.md                    # ← You are here
│
├── Data/                        # Knowledge base & cache
│   ├── README.md                # Data directory documentation
│   ├── policy_data.db           # OPP-115 SQLite database (2,185 entries)
│   └── cache/                   # Auto-generated, git-ignored
│       ├── embeddings.npy       #   Cached dense vector embeddings
│       └── faiss.index          #   Cached FAISS vector index
│
├── Scripts/                     # Core RAG pipeline modules
│   ├── README.md                # Scripts API documentation
│   ├── __init__.py
│   ├── prepare_opp115.py        # Data prep: Hugging Face → SQLite
│   ├── load_data.py             # Step 1: Data loading & validation
│   ├── chunk_data.py            # Step 2: Row → chunk conversion
│   ├── embed_data.py            # Step 3: Text → vector embedding
│   ├── vector_store.py          # Step 4: FAISS / sklearn index
│   ├── retriever.py             # Step 5: Multi-query retrieval
│   └── generator.py             # Step 6: Groq LLM generation
│
└── Utils/                       # Shared utilities
    ├── README.md                # Utils API documentation
    ├── __init__.py
    ├── text_cleaning.py         # Text normalisation & ref-code extraction
    └── formatting.py            # ANSI terminal UI, banners, colours
```

> 📌 All source code is in `.py` files — no Jupyter notebooks. Each subdirectory has its own `README.md` with detailed API documentation.

---

## ⚙️ Configuration

All settings live in `.env`. Only **`GROQ_API_KEY`** is required — everything else has sensible defaults.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `GROQ_API_URL` | `https://api.groq.com/openai/v1` | Groq API base URL |
| `GROQ_MODEL` | `ALLaM-2-7b` | LLM model for answer generation |
| `GROQ_TEMPERATURE` | `0.2` | Sampling temperature (`0` = deterministic) |
| `GROQ_MAX_OUTPUT_TOKENS` | `512` | Max tokens in the generated answer |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model name |
| `MULTI_QUERY_ENABLED` | `true` | Enable multi-query expansion |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `GENERATION_TOP_K` | `3` | Number of chunks to include in LLM context |

---

## 📊 Knowledge Base

### OPP-115 Dataset

The chatbot uses the **OPP-115** (Online Privacy Policies, 115 websites) dataset — a research-grade NLP corpus of real, published company privacy policies from Hugging Face ([`alzoubi36/opp_115`](https://huggingface.co/datasets/alzoubi36/opp_115)).

**Database schema** (`Data/policy_data.db → policy_data` table):

| Column | Type | Description |
|---|---|---|
| `ID` | TEXT (PK) | Unique segment identifier (e.g., `OPP-0001`) |
| `Topic` | TEXT | Privacy category | 
| `Content` | TEXT | Full policy text segment |

**Topic distribution across 2,185 entries:**

| Topic | Count | |
|---|---:|---|
| First Party Collection and Use | 705 | ████████████████████ |
| User Choice and Control | 368 | ██████████ |
| Practice Not Covered | 231 | ██████ |
| Other | 212 | ██████ |
| International and Specific Audiences | 189 | █████ |
| Data Security | 128 | ████ |
| User Access Edit and Deletion | 113 | ███ |
| Third Party Sharing and Collection | 71 | ██ |
| Policy Change | 65 | ██ |
| Introductory/Generic | 61 | ██ |
| Data Retention | 29 | █ |
| Do Not Track | 13 | ▏ |

### Rebuilding / Extending the Database

**Re-import from OPP-115:**

```bash
pip install datasets
python Scripts/prepare_opp115.py
```

**Add custom entries:**

```sql
INSERT INTO policy_data (ID, Topic, Content)
VALUES ('CUSTOM-001', 'Data Security', 'Your custom policy text here.');
```

After any data changes, delete the cache and restart:

```bash
# Windows
rmdir /s /q Data\cache

# macOS / Linux
rm -rf Data/cache
```

```bash
python main.py
```

---

## 🏗️ Design Decisions

| Decision | Why |
|---|---|
| **OPP-115 dataset** | Research-grade, real-world privacy policies with 2,185 annotated segments across 12 categories — far better than synthetic data |
| **SQLite storage** | Schema enforcement, atomic writes, no CSV issues. `sqlite3` is Python stdlib — zero extra runtime dependencies |
| **Local embeddings** | `sentence-transformers` runs offline after initial download. No API cost. `all-MiniLM-L6-v2` balances quality & speed |
| **FAISS + sklearn fallback** | FAISS for performance; automatic sklearn fallback ensures the chatbot runs everywhere |
| **Multi-query expansion** | Privacy questions can be phrased many ways — 4 query variants improve recall without user effort |
| **Embedding & index caching** | Avoids recomputing embeddings (~30–90 s) on every launch. Cache lives in `Data/cache/`, git-ignored |
| **Groq OpenAI-compatible API** | Fast inference. Standard `/chat/completions` endpoint — switch providers with one `.env` change |
| **Grounded generation** | System prompt constrains the LLM to answer only from provided context, minimising hallucination |
| **ANSI + ASCII fallback** | Rich UI on modern terminals; functional on legacy Windows `cmd.exe`. Respects `NO_COLOR` |

---

## 🔧 Troubleshooting

<details>
<summary><strong>❌ "GROQ_API_KEY is missing or still set to a placeholder"</strong></summary>

Edit `.env` and set `GROQ_API_KEY` to your real Groq API key from [console.groq.com](https://console.groq.com/).
</details>

<details>
<summary><strong>❌ <code>faiss-cpu</code> fails to install</strong></summary>

The chatbot automatically falls back to scikit-learn. You can safely remove `faiss-cpu` from `requirements.txt` if needed.
</details>

<details>
<summary><strong>❌ Unicode / box-drawing characters display as <code>?</code></strong></summary>

Your terminal may not support UTF-8. The chatbot auto-detects this and falls back to ASCII. On Windows CMD, run `chcp 65001` for UTF-8 support.
</details>

<details>
<summary><strong>❌ <code>ModuleNotFoundError: No module named 'Scripts'</code></strong></summary>

Make sure you're running `python main.py` from the project root directory (`rag-policy-chatbot/`).
</details>

<details>
<summary><strong>❌ <code>policy_data.db</code> is missing or empty</strong></summary>

Run `python Scripts/prepare_opp115.py` to fetch the OPP-115 dataset and create the database.
</details>

<details>
<summary><strong>⏳ Slow first launch</strong></summary>

On the first run, embeddings are generated for all 2,185 policy segments (~30–90 s depending on hardware). Subsequent runs load from cache and start in under 2 seconds.
</details>

<details>
<summary><strong>🔄 Stale answers after updating the database</strong></summary>

Delete `Data/cache/` and restart to force re-embedding and re-indexing.
</details>

---

## 🗂️ Further Reading

Each subdirectory contains its own detailed documentation:

| Directory | Documentation | Contents |
|---|---|---|
| `Data/` | [`Data/README.md`](Data/README.md) | Database schema, topic distribution, data flow diagram |
| `Scripts/` | [`Scripts/README.md`](Scripts/README.md) | Function signatures, API docs for all 7 pipeline modules |
| `Utils/` | [`Utils/README.md`](Utils/README.md) | ANSI colour constants, symbol tables, formatting API reference |

---

## 📄 License

This project is provided as-is for **educational and research use**.

---

<div align="center">

**Built with** 🐍 Python · 🤗 Hugging Face · ⚡ FAISS · 🚀 Groq

</div>
