# Build a Multi-Tool RAG Chatbot With LangChain, VectLite, and Gemini

*A step-by-step guide to building an AI-powered chatbot that answers questions about hospital reviews using hybrid vector search, structured data queries, and a multi-tool LangChain agent.*

---

In this tutorial, you'll build a complete Retrieve-Augment-Generate (RAG) chatbot that can answer both qualitative and quantitative questions about patient reviews from Montreal hospitals. Unlike traditional RAG systems that rely on a single retrieval mechanism, your chatbot will use a **multi-tool LangChain agent** that automatically decides whether to search through patient reviews using semantic similarity or query structured data for statistics — just like a real analyst would.

At the heart of this project is **[VectLite](https://vectlite.mcsedition.org/)**, an embedded vector database written in Rust with Python and Node.js bindings. VectLite stores everything in a single portable `.vdb` file — no server, no Docker, no infrastructure. It supports **hybrid search** combining dense vector similarity (HNSW) with sparse BM25 keyword retrieval, making it ideal for real-world RAG applications where pure semantic search isn't enough.

By the end of this tutorial, you'll have a working system that looks like this:

```
User Question
      |
      v
  LangChain ReAct Agent (Gemini 2.5 Flash)
      |
      |--- "reviews" tool ---> VectLite Hybrid Search ---> RAG Chain
      |                          (dense + BM25 sparse)
      |
      |--- "structured_data" tool ---> Pandas Query ---> Stats Chain
      |
      v
  Final Answer (with sources)
```

**Here's an example of what the chatbot can do:**

*"What are the top 3 hospitals by patient volume and how do patients rate the emergency services at each?"*

The agent first calls the `structured_data` tool to rank hospitals by patient count, then calls the `reviews` tool to search through thousands of patient reviews about emergency services — and synthesizes both into a single, coherent answer with source citations.

## What You'll Learn

- How to use **VectLite** as an embedded vector database with hybrid search
- How to build an **ETL pipeline** that ingests CSV data, generates embeddings, and stores them with sparse BM25 terms
- How to create a **LangChain retriever** backed by VectLite's hybrid dense+sparse search
- How to build a **multi-tool ReAct agent** using LangGraph that decides which tool to call
- How to serve the agent as a **FastAPI** backend and build a **Streamlit + Plotly** frontend

## What You'll Build

The project follows a clean, modular architecture inspired by production patterns:

```
hospital_rag/
|
+-- etl/                        # ETL Pipeline
|   +-- settings.py             # Pydantic settings
|   +-- extract.py              # Load and merge CSVs
|   +-- transform.py            # Clean text, build documents
|   +-- load.py                 # OpenAI embeddings -> VectLite bulk_ingest
|   +-- pipeline.py             # Orchestrator: extract -> transform -> load
|
+-- chatbot_api/                # FastAPI Backend
|   +-- config/
|   |   +-- settings.py         # Pydantic settings
|   +-- agents/
|   |   +-- hospital_agent.py   # Multi-tool ReAct agent (LangGraph)
|   |   +-- retriever.py        # VectLite hybrid retriever (LangChain)
|   +-- chains/
|   |   +-- reviews_chain.py    # RAG chain for patient reviews
|   |   +-- structured_chain.py # Pandas-based structured data chain
|   |   +-- rag_chain.py        # Agent orchestration entry point
|   +-- models/
|   |   +-- schemas.py          # Pydantic request/response schemas
|   +-- utils/
|   |   +-- embedding.py        # OpenAI embedding service
|   |   +-- vectordb.py         # VectLite utilities
|   +-- main.py                 # FastAPI app + routes
|   +-- entrypoint.sh           # Startup script
|
+-- chatbot_front/              # Streamlit + Plotly Frontend
|   +-- config/
|   |   +-- settings.py         # API base URL config
|   +-- components/
|   |   +-- charts.py           # Plotly chart components
|   |   +-- chat_view.py        # Chat interface
|   |   +-- sidebar.py          # Dashboard sidebar
|   +-- services/
|   |   +-- api_client.py       # HTTP client to chatbot_api
|   +-- app.py                  # Streamlit entry point
|
+-- data/
|   +-- raw/                    # Source CSV files
|   +-- vectordb/               # VectLite .vdb file
|
+-- .env                        # API keys
+-- requirements.txt
+-- pyproject.toml
```

## VectLite: The SQLite Moment for Vector Databases

### The Problem With Today's Vector Databases

If you've built RAG applications before, you know the pain. Before you can even store your first embedding, you need to:

- **Pinecone**: Create an account, provision a cloud index, manage API keys, pay per query, hope the service doesn't go down during your demo
- **Qdrant**: Pull a Docker image, run `docker-compose up`, configure volumes, manage ports, hope Docker Desktop doesn't eat all your RAM
- **Weaviate**: Same Docker story, plus a more complex schema configuration
- **ChromaDB**: Better — it can run in-process — but still requires a client-server architecture for anything serious, and doesn't support sparse/BM25 search natively
- **pgvector**: Install PostgreSQL, enable the extension, manage a database server, learn SQL vector syntax

Every single one of these requires you to manage **infrastructure** before you write a single line of RAG logic. For a technology that's supposed to make AI accessible, that's a lot of DevOps.

### What if vector search worked like SQLite?

Think about what made SQLite revolutionary for relational data. Before SQLite, if you wanted a database in your application, you needed MySQL or PostgreSQL running somewhere — a server process, network configuration, authentication, backups. SQLite changed that equation entirely: `import sqlite3`, point it at a file, and you have a full relational database. No server. No network. No Docker. Just a file.

**VectLite does the same thing for vector search.**

```python
pip install vectlite
```

```python
import vectlite

# That's it. Your entire vector database is this one file.
db = vectlite.open("knowledge.vdb", dimension=1536)

# Store vectors with metadata and sparse terms for BM25
db.upsert("doc1", embedding, {"source": "blog"}, sparse=vectlite.sparse_terms("your text"))

# Hybrid search: dense similarity + keyword matching
results = db.search(query_embedding, k=10, sparse=vectlite.sparse_terms("search query"))
```

No server process. No Docker container. No network calls. No cloud account. No API keys for the database itself. Your entire vector database — embeddings, metadata, BM25 index, HNSW graph — lives in a **single `.vdb` file** that you can copy, email, put in a Git LFS repo, or drop into an S3 bucket.

### What Makes VectLite Different

**1. True Zero-Infrastructure Deployment**

The core is written in **Rust** and compiled into a native binary that ships with the Python package. When you `pip install vectlite`, you get a self-contained library with no external dependencies. There is no background process, no port to configure, no service to monitor. It works on macOS, Linux, and Windows — anywhere Python runs.

This means your RAG application is as portable as a Python script. Deploy it on a Raspberry Pi, in a Lambda function, on an air-gapped server, or in a Jupyter notebook — it works the same way everywhere.

**2. Hybrid Search Out of the Box**

Most embedded vector databases only support dense vector search. But pure semantic search has a well-known weakness: it can miss results that contain exact keywords the user is looking for. If a patient writes "Dr. Martin was terrible," a dense search for "Dr. Martin" might rank semantically similar but irrelevant reviews higher.

VectLite solves this with **built-in hybrid search** that combines:
- **Dense vectors** (HNSW index) for semantic similarity
- **Sparse BM25** (inverted index) for keyword precision
- **Fusion strategies**: linear combination or Reciprocal Rank Fusion (RRF)

You control the balance with `dense_weight` and `sparse_weight`:

```python
# 70% semantic, 30% keyword matching
results = db.search(
    query_embedding,
    k=10,
    sparse=vectlite.sparse_terms("Dr. Martin complaint"),
    dense_weight=0.7,
    sparse_weight=0.3,
    fusion="rrf",
)
```

**3. ACID Transactions and Crash Safety**

Unlike most embedded vector stores, VectLite uses **Write-Ahead Logging (WAL)** — the same technique SQLite and PostgreSQL use to guarantee data integrity. If your process crashes mid-write, the database recovers to a consistent state automatically. You also get:
- Atomic transactions with rollback
- File locking to prevent concurrent corruption
- Snapshots and backups
- Read-only mode for safe concurrent readers

**4. Blazing Fast Bulk Ingestion**

Loading data into VectLite is designed for real-world datasets. The `bulk_ingest()` method batches WAL writes and rebuilds indexes only once at completion:

```python
records = [
    {"id": doc_id, "vector": embedding, "metadata": meta, "sparse": vectlite.sparse_terms(text)}
    for doc_id, embedding, meta, text in your_data
]
db.bulk_ingest(records, batch_size=5000)
```

In this project, we ingested **11,058 documents** with 1536-dimensional vectors and full BM25 sparse terms in **22 seconds**. The embedding generation (OpenAI API) took longer than the database write.

### Head-to-Head Comparison

| Feature | VectLite | ChromaDB | Pinecone | Qdrant | pgvector |
|---------|----------|----------|----------|--------|----------|
| **Install** | `pip install` | `pip install` | Cloud signup | Docker | PostgreSQL + extension |
| **Server required** | No | Optional | Yes (cloud) | Yes (Docker) | Yes |
| **Storage** | Single `.vdb` file | Directory | Cloud | Docker volume | PostgreSQL tables |
| **Hybrid search** | Dense + BM25 native | Dense only | Dense + Sparse | Dense + Sparse | Dense only |
| **Offline capable** | Yes | Yes | No | Yes | Yes |
| **Language** | Rust (Python/Node bindings) | Python | Cloud API | Rust | C/SQL |
| **ACID transactions** | Yes (WAL) | No | N/A | No | Yes |
| **Bulk ingestion** | `bulk_ingest()` | `add()` | `upsert()` batch | `upload_points()` | `COPY` |
| **Portability** | Copy one file | Copy directory | N/A | Export/import | pg_dump |

### When to Use VectLite

VectLite is the right choice when:

- **You're prototyping** and don't want to spend 30 minutes on infrastructure before writing your first query
- **You need hybrid search** and don't want to manage separate dense and sparse indexes
- **Your application runs on edge** — mobile, IoT, desktop apps, air-gapped environments
- **You want portability** — ship your entire knowledge base as a single file alongside your application
- **You value simplicity** — `pip install`, open a file, search. That's the entire workflow

VectLite is **not** designed to replace a distributed vector database for billion-scale deployments. If you need horizontal scaling, multi-tenant isolation, or cloud-native managed infrastructure, look at Pinecone or Qdrant Cloud. But for the vast majority of RAG applications — from prototypes to production systems handling millions of documents — a single-file embedded database is not just simpler, it's faster.

> The best infrastructure is the infrastructure you don't need to manage.

---

## Prerequisites

Before you begin, make sure you have:

- **Python 3.11+** installed on your system
- **An OpenAI API key** — used for generating text embeddings (`text-embedding-3-small`). You'll need this for both the ETL pipeline and the chatbot API. [Get one here](https://platform.openai.com/api-keys)
- **A Google Gemini API key** — used for the LLM that generates the chatbot's answers (`gemini-2.5-flash`). [Get one here](https://aistudio.google.com/apikey)
- **Basic familiarity with LangChain** — you should know what chains, retrievers, and prompts are. If not, start with the [LangChain quickstart](https://python.langchain.com/docs/get_started/quickstart)

> **Why two API keys?** This project uses a best-of-both-worlds approach: OpenAI's `text-embedding-3-small` for embeddings (fast, cheap, high-quality 1536-dimensional vectors) and Google's `gemini-2.5-flash` for generation (fast, large context window, excellent reasoning). You could use a single provider for both, but this combination gives the best price-to-performance ratio at the time of writing.

### Clone the Repository and Install Dependencies

```bash
git clone https://github.com/Stefen-Taime/hospital-rag.git
cd hospital_rag
```

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Here's what gets installed:

| Package | Purpose |
|---------|---------|
| `vectlite` | Embedded vector database — the star of the show |
| `langchain`, `langchain-core` | Chain orchestration, retrievers, prompt templates |
| `langchain-google-genai` | Gemini LLM integration for LangChain |
| `openai` | OpenAI embedding API client |
| `pydantic-settings` | Type-safe configuration from `.env` files |
| `pandas` | Data loading and structured queries |
| `fastapi`, `uvicorn` | REST API backend |
| `streamlit`, `plotly` | Interactive frontend with charts |
| `httpx` | Async HTTP client for frontend → API communication |

### Configure Your API Keys

Create a `.env` file at the project root:

```bash
# .env
OPENAI_API_KEY=sk-your-openai-key-here
GEMINI_API_KEY=your-gemini-key-here
```

Both the ETL pipeline and the chatbot API will read from this file automatically using `pydantic-settings`. No environment variable exports needed — just drop the file in and go.

---

## Step 1: Understand the Data

The chatbot answers questions about **real Google reviews** from six Montreal hospitals. The dataset consists of four CSV files that follow a normalized relational schema:

```
hospitals.csv  ──┐
                  ├──>  visits.csv  ──>  reviews.csv
patients.csv  ──┘
```

### `hospitals.csv` — 6 hospitals

Each row is a Montreal hospital with rich metadata:

| Column | Example |
|--------|---------|
| `hospital_id` | `1` |
| `hospital_name` | `Hôpital du Sacré-Cœur-de-Montréal` |
| `description` | Historical description of the hospital |
| `Address` | `5400 Boul Gouin O, Montréal, QC H4J 1C5` |
| `Phone` | `+1 514-338-2222` |
| `Established` | `1898` |
| `Number_of_Beds` | `554` |
| `Emergencies` | `Ouvert 24h/24` |

This is the data the `structured_data` tool queries — factual, quantitative information that doesn't need semantic search.

### `patients.csv` — 7,095 patients

A simple table linking patients to their admission dates:

| Column | Type |
|--------|------|
| `patient_id` | Unique identifier |
| `date_of_admission` | Date (may be null) |

### `visits.csv` — 7,095 visits

The junction table connecting patients to hospitals:

| Column | Type |
|--------|------|
| `patient_id` | FK → patients |
| `visit_id` | Unique identifier |
| `hospital_id` | FK → hospitals |
| `date_of_admission` | Date |

### `reviews.csv` — 7,107 reviews

The heart of the dataset. Each review is a patient's written feedback about their hospital experience:

| Column | Example |
|--------|---------|
| `review_id` | `1` |
| `visit_id` | `3732` |
| `review` | `"Excellente prise en main lors d'un situation d'urgence..."` |

These reviews are what gets embedded into VectLite and searched by the `reviews` tool. They cover everything from emergency room wait times to staff bedside manner, surgical outcomes, food quality, and administrative frustrations.

### The Merged View

When you join all four tables, each review gains full context:

```
review_text + hospital_name + hospital_id + patient_id + visit_id + date_of_admission
```

This merged view is what the ETL pipeline produces — each document stored in VectLite contains both the review text (for embedding and search) and the structured metadata (for source citations in the chatbot's answers).

### Why This Data Structure Matters

The split between **unstructured reviews** and **structured hospital data** is exactly why the chatbot needs a multi-tool agent. Consider these two questions:

1. *"How do patients describe the emergency room at Sacré-Cœur?"* → This requires **semantic search** through thousands of review texts. The `reviews` tool handles this.

2. *"Which hospital has the most reviews?"* → This requires a **quantitative query** on structured data. The `structured_data` tool handles this.

3. *"What are the top 3 hospitals by patient volume and how do patients rate the emergency services at each?"* → This requires **both tools**. The agent calls `structured_data` first to rank hospitals by count, then calls `reviews` to find emergency-related feedback for each.

A single-tool RAG system would struggle with questions 2 and 3. The multi-tool agent handles all three naturally.

---

## Step 2: Build the ETL Pipeline

The ETL (Extract, Transform, Load) pipeline takes raw CSV files and turns them into a searchable VectLite database with dense embeddings and sparse BM25 terms. The pipeline follows a classic three-stage pattern, with each stage in its own module:

```
extract.py          transform.py           load.py
CSV files  ──>  Merged DataFrame  ──>  Documents  ──>  VectLite .vdb
(4 files)       (7,107 rows)          (cleaned)       (embeddings + BM25)
```

### Project Configuration With Pydantic Settings

Before writing any pipeline code, you need a clean way to manage configuration. Instead of scattering `os.environ` calls throughout the codebase, this project uses **Pydantic Settings** — a type-safe configuration layer that reads from `.env` files and validates values at startup:

```python
# etl/settings.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class EtlSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = base_dir / "data" / "raw"
    vectordb_path: Path = base_dir / "data" / "vectordb" / "hospital_reviews.vdb"

    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    batch_size: int = 500


settings = EtlSettings()
```

This gives you several advantages over raw environment variables:

- **Type validation** — `embedding_dimension` is guaranteed to be an `int`, not a string you forgot to cast
- **Default values** — sensible defaults for model names, paths, and batch sizes
- **Single source of truth** — every module imports `settings` from the same place
- **Fail-fast** — if `OPENAI_API_KEY` is missing from your `.env`, the application crashes immediately with a clear error instead of failing halfway through ingestion

### Stage 1: Extract — Load and Merge CSVs

The extract stage loads all four CSV files and joins them into a single enriched DataFrame:

```python
# etl/extract.py
import logging
import pandas as pd
from etl.settings import settings

logger = logging.getLogger(__name__)


def _load_csv(name: str) -> pd.DataFrame:
    path = settings.data_dir / name
    df = pd.read_csv(path)
    logger.info("Loaded %s — %d rows", name, len(df))
    return df


def extract() -> pd.DataFrame:
    """Load the 4 CSV files and merge them into an enriched DataFrame."""
    hospitals = _load_csv("hospitals.csv")
    patients = _load_csv("patients.csv")
    reviews = _load_csv("reviews.csv")
    visits = _load_csv("visits.csv")

    merged = (
        visits
        .merge(hospitals, on="hospital_id", how="left")
        .merge(patients, on=["patient_id", "date_of_admission"], how="left")
        .merge(reviews, on="visit_id", how="inner")
    )

    merged["date_of_admission"] = merged["date_of_admission"].fillna("Non spécifié")

    logger.info("Merge complete — %d documents", len(merged))
    return merged
```

The join strategy matters here:

- **`visits ← hospitals`** (`LEFT JOIN` on `hospital_id`): every visit gets its hospital metadata
- **`visits ← patients`** (`LEFT JOIN` on `patient_id` + `date_of_admission`): enrich with patient info
- **`visits → reviews`** (`INNER JOIN` on `visit_id`): only keep visits that have reviews — this is our dataset filter

The result is a DataFrame where each row is a review with full context: which hospital, which patient, which visit, and the review text itself. From 4 normalized tables to one flat, ready-to-embed dataset.

### Stage 2: Transform — Clean Text and Build Documents

The transform stage takes the merged DataFrame, cleans the text, and builds structured `Document` objects ready for embedding:

```python
# etl/transform.py
import logging
import re
from typing import TypedDict
import pandas as pd

logger = logging.getLogger(__name__)


class Document(TypedDict):
    id: str
    text: str
    metadata: dict


def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def transform(df: pd.DataFrame) -> list[Document]:
    """Clean reviews and build documents for ingestion."""
    df = df.copy()
    df["review"] = df["review"].apply(_clean_text)
    df = df[df["review"].str.len() > 0]

    documents: list[Document] = []
    for _, row in df.iterrows():
        text = (
            f"Hôpital: {row['hospital_name']}\n"
            f"Adresse: {row.get('Address', 'N/A')}\n"
            f"Date d'admission: {row['date_of_admission']}\n"
            f"Avis: {row['review']}"
        )
        documents.append(Document(
            id=str(row["review_id"]),
            text=text,
            metadata={
                "hospital_id": int(row["hospital_id"]),
                "hospital_name": str(row["hospital_name"]),
                "visit_id": int(row["visit_id"]),
                "patient_id": int(row["patient_id"]),
                "date_of_admission": str(row["date_of_admission"]),
                "review_raw": str(row["review"]),
            },
        ))

    logger.info("Transform complete — %d valid documents", len(documents))
    return documents
```

Two key design decisions here:

**1. The `text` field is a formatted string, not just the raw review.** By prepending the hospital name, address, and admission date to the review text, you're giving the embedding model more context. When someone searches for "Sacré-Cœur emergency room," the dense vector will match not just on "emergency room" semantics but also on the hospital name in the text. This is a simple but effective technique for enriching embeddings without adding complexity.

**2. The `metadata` dict carries structured fields separately.** While `text` goes into the embedding, `metadata` is stored alongside the vector in VectLite and returned with search results. This lets the chatbot cite sources with specific hospital names, visit IDs, and dates without having to parse them out of the text.

### Stage 3: Load — Embed and Ingest Into VectLite

This is where the magic happens. The load stage takes the cleaned documents, generates embeddings via OpenAI, and bulk-ingests everything into VectLite with both dense vectors and sparse BM25 terms:

```python
# etl/load.py
import logging
from concurrent.futures import ThreadPoolExecutor

import vectlite
from openai import OpenAI
from etl.settings import settings

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=settings.openai_api_key)
EMBED_BATCH = 2000


def _embed_all(texts: list[str]) -> list[list[float]]:
    """Embed all texts in parallel using large batches."""
    all_embeddings: list[list[float]] = [[] for _ in texts]

    def _embed_chunk(start: int, chunk: list[str]):
        resp = _client.embeddings.create(
            model=settings.embedding_model, input=chunk
        )
        return start, [item.embedding for item in resp.data]

    futures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i in range(0, len(texts), EMBED_BATCH):
            chunk = texts[i : i + EMBED_BATCH]
            futures.append(pool.submit(_embed_chunk, i, chunk))

        for future in futures:
            start, embs = future.result()
            for j, emb in enumerate(embs):
                all_embeddings[start + j] = emb

    return all_embeddings
```

**The embedding strategy is optimized for throughput.** Instead of embedding one document at a time (which would take minutes due to API round-trip latency), we batch 2,000 texts per API call and run 4 concurrent requests. For 7,107 documents, this means only 4 API calls, all running in parallel via `ThreadPoolExecutor`. The entire embedding step completes in about **10 seconds**.

> **Why not use OpenAI's async client?** For CPU-bound work you'd want async, but embedding API calls are I/O-bound — the bottleneck is network latency, not computation. `ThreadPoolExecutor` with 4 workers saturates the API rate limit just as effectively as async, and the code is simpler to read and debug.

Now, the VectLite ingestion:

```python
def load(documents: list) -> None:
    """Embed and store documents in VectLite via bulk_ingest."""
    total = len(documents)
    texts = [doc["text"] for doc in documents]

    logger.info("Generating embeddings for %d documents...", total)
    embeddings = _embed_all(texts)
    logger.info("Embeddings generated.")

    logger.info("Building records for bulk_ingest...")
    records = [
        {
            "id": doc["id"],
            "vector": emb,
            "metadata": doc["metadata"] | {"text": doc["text"]},
            "sparse": vectlite.sparse_terms(doc["text"]),
        }
        for doc, emb in zip(documents, embeddings)
    ]

    logger.info("Writing to VectLite (bulk_ingest, %d records)...", total)
    db = vectlite.open(
        str(settings.vectordb_path),
        dimension=settings.embedding_dimension
    )
    db.bulk_ingest(records)

    logger.info("Ingestion complete — %d documents -> %s", total, settings.vectordb_path)
```

Let's break down the record structure that VectLite expects:

```python
{
    "id": "1",                           # Unique document ID
    "vector": [0.012, -0.034, ...],      # 1536-dim dense embedding
    "metadata": {                         # Arbitrary JSON metadata
        "hospital_name": "Sacré-Cœur",
        "visit_id": 3732,
        "text": "Hôpital: Sacré-Cœur\n..."
    },
    "sparse": {"emergency": 1.2, ...}    # BM25 term weights
}
```

The `sparse` field is the key to hybrid search. `vectlite.sparse_terms(text)` takes a string and returns a dictionary of `{term: weight}` pairs using BM25 tokenization. These sparse terms are stored in VectLite's inverted index alongside the dense vector in the HNSW graph. At query time, both indexes are searched and the results are fused.

**Why `bulk_ingest()` and not `upsert()`?** This is a critical performance detail. VectLite's `upsert()` method commits each record individually and rebuilds indexes incrementally. For one-by-one inserts during serving, that's fine. But for initial data loading, the overhead compounds: 7,000 individual `upsert()` calls with sparse terms would take **over an hour** due to repeated index rebuilds.

`bulk_ingest()` solves this by batching WAL (Write-Ahead Log) writes and rebuilding the HNSW index only once at the end. The result: **11,058 documents with 1536-dimensional vectors and full BM25 sparse terms ingested in 22 seconds.** That's a 300x speedup over individual upserts.

### Orchestrate the Pipeline

The pipeline module ties all three stages together with timing:

```python
# etl/pipeline.py
import logging
import time

from etl.extract import extract
from etl.load import load
from etl.transform import transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run() -> None:
    start = time.perf_counter()

    logger.info("=== ETL Pipeline START ===")

    logger.info("[1/3] Extract")
    df = extract()

    logger.info("[2/3] Transform")
    documents = transform(df)

    logger.info("[3/3] Load")
    load(documents)

    elapsed = time.perf_counter() - start
    logger.info("=== ETL Pipeline END — %.1fs ===", elapsed)


if __name__ == "__main__":
    run()
```

### Run the Pipeline

From the project root:

```bash
python -m etl.pipeline
```

You should see output like this:

```
14:23:01 | INFO    | etl.extract | Loaded hospitals.csv — 6 rows
14:23:01 | INFO    | etl.extract | Loaded patients.csv — 7095 rows
14:23:01 | INFO    | etl.extract | Loaded reviews.csv — 7107 rows
14:23:01 | INFO    | etl.extract | Loaded visits.csv — 7095 rows
14:23:01 | INFO    | etl.extract | Merge complete — 7107 documents
14:23:01 | INFO    | etl.transform | Transform complete — 7107 valid documents
14:23:01 | INFO    | etl.load | Generating embeddings for 7107 documents...
14:23:11 | INFO    | etl.load | Embeddings generated.
14:23:11 | INFO    | etl.load | Building records for bulk_ingest...
14:23:12 | INFO    | etl.load | Writing to VectLite (bulk_ingest, 7107 records)...
14:23:34 | INFO    | etl.load | Ingestion complete — 7107 documents -> data/vectordb/hospital_reviews.vdb
14:23:34 | INFO    | etl.pipeline | === ETL Pipeline END — 33.2s ===
```

**The entire pipeline — loading CSVs, merging tables, cleaning text, generating 7,107 embeddings via OpenAI, computing BM25 sparse terms, and writing everything to VectLite — completes in under 40 seconds.** The embedding API calls take about 10 seconds, and VectLite's `bulk_ingest()` handles the rest in about 22 seconds. Your entire searchable knowledge base is now a single `hospital_reviews.vdb` file sitting in `data/vectordb/`.

You can verify the ingestion:

```python
import vectlite

db = vectlite.open("data/vectordb/hospital_reviews.vdb", dimension=1536, read_only=True)
print(f"Documents stored: {db.count()}")
# Documents stored: 7107
```

---

## Step 3: Build the Multi-Tool RAG Agent

This is the core of the chatbot — the part that takes a user question, decides *how* to answer it, retrieves the right information, and generates a response. You'll build four components that work together:

1. **Embedding service** — generates query embeddings via OpenAI
2. **VectLite retriever** — a LangChain-compatible retriever that performs hybrid search
3. **Two LCEL chains** — one for review analysis (RAG), one for structured data queries
4. **A ReAct agent** — powered by LangGraph, decides which chain to call based on the question

```
User Question
      │
      ▼
  ReAct Agent (Gemini 2.5 Flash)
      │
      ├─── "What do patients think about..."
      │         │
      │         ▼
      │    reviews tool
      │         │
      │         ▼
      │    VectLiteRetriever ──> embed_query() ──> VectLite hybrid search
      │         │
      │         ▼
      │    Reviews RAG Chain ──> Gemini ──> Answer with citations
      │
      ├─── "How many patients visited..."
      │         │
      │         ▼
      │    structured_data tool
      │         │
      │         ▼
      │    Pandas stats summary ──> Gemini ──> Answer with figures
      │
      └─── "Compare hospitals by volume AND patient satisfaction..."
                │
                ▼
           Calls BOTH tools, synthesizes results
```

### The Embedding Service

Before the retriever can search VectLite, it needs to convert the user's question into a 1536-dimensional vector. This thin wrapper around OpenAI's API handles that:

```python
# chatbot_api/utils/embedding.py
from openai import OpenAI
from chatbot_api.config.settings import get_settings

_settings = get_settings()
_client = OpenAI(api_key=_settings.openai_api_key)


def embed_query(text: str) -> list[float]:
    """Generate the embedding for a user query."""
    response = _client.embeddings.create(
        model=_settings.embedding_model,
        input=text,
    )
    return response.data[0].embedding
```

This is intentionally simple — a single function that takes text and returns a vector. The OpenAI client is initialized once at module load (not per-request), and the model name comes from settings. Since user queries are short (one sentence), there's no need for batching here.

### The VectLite Hybrid Retriever

LangChain's retrieval system is built around the `BaseRetriever` interface. Any class that implements `_get_relevant_documents(query: str) -> list[Document]` can be plugged into any LangChain chain. Here's the VectLite implementation:

```python
# chatbot_api/agents/retriever.py
import vectlite
from vectlite import sparse_terms
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from chatbot_api.config.settings import get_settings
from chatbot_api.utils.embedding import embed_query

_settings = get_settings()


class VectLiteRetriever(BaseRetriever):
    """Hybrid dense + sparse retriever via VectLite."""

    db_path: str = str(_settings.vectordb_path)
    dimension: int = _settings.embedding_dimension
    top_k: int = _settings.top_k
    dense_weight: float = _settings.dense_weight
    sparse_weight: float = _settings.sparse_weight

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        db = vectlite.open(self.db_path, dimension=self.dimension, read_only=True)
        query_embedding = embed_query(query)

        results = db.search(
            query_embedding,
            k=self.top_k,
            sparse=sparse_terms(query),
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
        )

        return [
            Document(
                page_content=r["metadata"].get("text", ""),
                metadata={
                    "hospital_name": r["metadata"].get("hospital_name", ""),
                    "hospital_id": r["metadata"].get("hospital_id"),
                    "visit_id": r["metadata"].get("visit_id"),
                    "patient_id": r["metadata"].get("patient_id"),
                    "date_of_admission": r["metadata"].get("date_of_admission", ""),
                    "review_raw": r["metadata"].get("review_raw", ""),
                    "score": r["score"],
                },
            )
            for r in results
        ]
```

Let's break down the critical design decisions:

**`read_only=True` is essential.** The chatbot API serves multiple concurrent requests. Without `read_only=True`, VectLite acquires an exclusive file lock on every `open()` call — fine for a single-user ETL script, but it would serialize all API requests into a single queue. Read-only mode uses shared locks, allowing multiple concurrent readers. This one flag is the difference between a responsive API and a bottleneck.

**Hybrid search happens in a single call.** The `db.search()` method takes both a dense vector (`query_embedding`) and sparse terms (`sparse_terms(query)`). VectLite internally:
1. Searches the **HNSW graph** for the nearest dense vectors
2. Searches the **BM25 inverted index** for keyword matches
3. **Fuses** the two result sets using the configured weights (70% dense, 30% sparse by default)

This means a query like *"Dr. Martin emergency"* gets the best of both worlds: semantic understanding of "emergency" (dense) AND exact keyword matching on "Dr. Martin" (sparse). Without hybrid search, the dense-only retriever might return reviews about emergency rooms that never mention Dr. Martin.

**VectLite returns dictionaries, not objects.** In v0.1.10, search results are plain Python dicts: `r["metadata"]`, `r["score"]` — not `r.metadata`, `r.score`. This caught us during development and is worth noting if you're used to other vector database clients.

### The Reviews RAG Chain

The reviews chain is a classic RAG pipeline: retrieve → format → prompt → generate. It uses LangChain Expression Language (LCEL) to wire the components together:

```python
# chatbot_api/chains/reviews_chain.py
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from chatbot_api.agents.retriever import VectLiteRetriever
from chatbot_api.config.settings import get_settings

_settings = get_settings()

REVIEWS_PROMPT = """\
You are an expert assistant for analyzing hospital reviews in Montreal.
Always respond in English, in a clear, structured and factual manner.
Base your answers **only** on the reviews provided in the context below.
If the requested information is not present in the context, state it clearly.
Mention the relevant hospitals and summarize trends when appropriate.

Context:
{context}
"""


def _format_docs(docs: list[Document]) -> str:
    parts = []
    for d in docs:
        header = (
            f"[Hospital: {d.metadata.get('hospital_name', 'N/A')} | "
            f"Score: {d.metadata.get('score', 0):.3f}]"
        )
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def build_reviews_chain():
    """Build the RAG chain for patient reviews."""
    retriever = VectLiteRetriever()

    llm = ChatGoogleGenerativeAI(
        model=_settings.llm_model,
        google_api_key=_settings.gemini_api_key,
        temperature=_settings.llm_temperature,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", REVIEWS_PROMPT),
        ("human", "{question}"),
    ])

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
```

The LCEL pipeline reads left-to-right:

1. **`retriever | _format_docs`** — the query goes into VectLiteRetriever, which returns `list[Document]`, then `_format_docs` converts that into a readable string with hospital names and relevance scores as headers
2. **`RunnablePassthrough()`** — passes the original question through unchanged
3. **`prompt`** — slots the context and question into the system/human message template
4. **`llm`** — sends the prompt to Gemini 2.5 Flash
5. **`StrOutputParser()`** — extracts the text content from Gemini's response

The `_format_docs` function is worth noting — it prepends each review with a header showing the hospital name and relevance score. This gives the LLM clear attribution context, so it can say things like "According to reviews from Sacré-Cœur..." rather than generic "some patients say..."

### The Structured Data Chain

Not every question needs vector search. *"Which hospital has the most patients?"* is a data query, not a retrieval task. The structured chain handles these with Pandas:

```python
# chatbot_api/chains/structured_chain.py
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from chatbot_api.config.settings import get_settings

_settings = get_settings()
_DATA_DIR = _settings.base_dir / "data" / "raw"


def _load_data() -> dict[str, pd.DataFrame]:
    """Load the CSV files into memory."""
    hospitals = pd.read_csv(_DATA_DIR / "hospitals.csv")
    patients = pd.read_csv(_DATA_DIR / "patients.csv")
    reviews = pd.read_csv(_DATA_DIR / "reviews.csv")
    visits = pd.read_csv(_DATA_DIR / "visits.csv")

    merged = (
        visits
        .merge(hospitals, on="hospital_id", how="left")
        .merge(patients, on=["patient_id", "date_of_admission"], how="left")
        .merge(reviews, on="visit_id", how="inner")
    )

    return {
        "hospitals": hospitals,
        "patients": patients,
        "reviews": reviews,
        "visits": visits,
        "merged": merged,
    }


def _compute_stats(data: dict[str, pd.DataFrame]) -> str:
    """Generate a complete statistical summary of the data."""
    merged = data["merged"]
    hospitals = data["hospitals"]

    lines = ["=== HOSPITAL DATA STATISTICS ===\n"]

    # Number of reviews per hospital
    review_counts = merged.groupby("hospital_name").size().sort_values(ascending=False)
    lines.append("## Number of reviews per hospital")
    for name, count in review_counts.items():
        lines.append(f"- {name}: {count} reviews")
    lines.append(f"- TOTAL: {review_counts.sum()} reviews\n")

    # Number of unique patients per hospital
    patient_counts = (
        merged.groupby("hospital_name")["patient_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    lines.append("## Number of unique patients per hospital")
    for name, count in patient_counts.items():
        lines.append(f"- {name}: {count} patients")
    lines.append("")

    # Number of visits per hospital
    visit_counts = (
        merged.groupby("hospital_name")["visit_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    lines.append("## Number of visits per hospital")
    for name, count in visit_counts.items():
        lines.append(f"- {name}: {count} visits")
    lines.append("")

    # Hospital information
    lines.append("## Hospital information")
    for _, h in hospitals.iterrows():
        lines.append(
            f"- {h['hospital_name']}: {h.get('Address', 'N/A')} | "
            f"Founded: {h.get('Established', 'N/A')} | "
            f"Beds: {h.get('Number_of_Beds', 'N/A')} | "
            f"Phone: {h.get('Phone', 'N/A')}"
        )

    return "\n".join(lines)
```

The approach here is **pre-compute, then prompt**. Instead of asking the LLM to write Pandas code (fragile, prone to hallucination), we:

1. Load all CSVs once at startup
2. Compute every useful statistic upfront — review counts, patient counts, visit counts, hospital metadata
3. Format it as a plain-text summary
4. Feed that summary into the LLM's context alongside the user's question

```python
STRUCTURED_PROMPT = """\
You are a hospital data analyst for Montreal.
Always respond in English with precise figures drawn from the data below.
Use tables or lists when appropriate.
Do not make assumptions — base your answers only on the provided data.

Data:
{data_summary}
"""


def build_structured_chain():
    """Build the chain for structured queries."""
    data = _load_data()
    stats_summary = _compute_stats(data)

    llm = ChatGoogleGenerativeAI(
        model=_settings.llm_model,
        google_api_key=_settings.gemini_api_key,
        temperature=0.1,  # Low temperature for factual precision
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", STRUCTURED_PROMPT),
        ("human", "{question}"),
    ])

    chain = (
        {"data_summary": lambda _: stats_summary, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
```

Note the **low temperature (0.1)** for the structured chain — when the user asks "how many reviews does Sacré-Cœur have?", you want the exact number from the data, not a creative paraphrase. The reviews chain uses 0.3 for slightly more natural language synthesis.

### The ReAct Agent — Putting It All Together

Now for the brain of the operation. The agent doesn't execute chains directly — it *decides which chain to call* based on the user's question. This is a **ReAct (Reason + Act)** agent built with LangGraph:

```python
# chatbot_api/agents/hospital_agent.py
from functools import lru_cache

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from chatbot_api.chains.reviews_chain import build_reviews_chain
from chatbot_api.chains.structured_chain import build_structured_chain
from chatbot_api.config.settings import get_settings

_settings = get_settings()

AGENT_SYSTEM_PROMPT = """\
You are an expert assistant on Montreal hospitals.
You have access to two tools to answer questions:

1. **reviews** — Semantic search in patient reviews. Use this tool for:
   - Questions about quality of care, patient experience
   - Reviews, opinions, feedback, patient comments
   - Satisfaction, complaints, praise
   - Staff communication, reception, cleanliness
   - Comparing experiences between hospitals

2. **structured_data** — Queries on structured data. Use this tool for:
   - Statistics: number of reviews, patients, visits
   - Quantitative comparisons between hospitals
   - Factual information: addresses, phone numbers, founding year, number of beds
   - Rankings by count, averages, totals

Choose the right tool based on the nature of the question.
If the question requires both types of information, call both tools.
Always respond in English.
"""


@lru_cache(maxsize=1)
def _get_chains():
    return build_reviews_chain(), build_structured_chain()


@tool
def reviews(query: str) -> str:
    """Semantic search in Montreal hospital patient reviews.
    Use this tool for questions about quality of care,
    patient experience, satisfaction, complaints, praise,
    staff communication, reception, cleanliness,
    or anything related to patient sentiment and opinions."""
    reviews_chain, _ = _get_chains()
    return reviews_chain.invoke(query)


@tool
def structured_data(query: str) -> str:
    """Queries on Montreal hospital structured data.
    Use this tool for statistics (number of reviews, patients,
    visits), quantitative comparisons, rankings,
    factual information (addresses, phone numbers, founding year, beds),
    averages and totals."""
    _, structured_chain = _get_chains()
    return structured_chain.invoke(query)


@lru_cache(maxsize=1)
def build_agent():
    """Build the multi-tool ReAct agent."""
    llm = ChatGoogleGenerativeAI(
        model=_settings.llm_model,
        google_api_key=_settings.gemini_api_key,
        temperature=_settings.llm_temperature,
    )

    agent = create_react_agent(
        model=llm,
        tools=[reviews, structured_data],
        prompt=AGENT_SYSTEM_PROMPT,
    )

    return agent
```

Let's unpack the key design patterns:

**Why `langgraph.prebuilt.create_react_agent` instead of `AgentExecutor`?** LangChain's original `AgentExecutor` was deprecated in v0.3. LangGraph's `create_react_agent` is the modern replacement — it builds a stateful graph where the LLM node can loop through tool calls, observe results, and decide whether to call another tool or return a final answer. Same ReAct pattern, better architecture.

**The tools are thin wrappers around LCEL chains.** Each `@tool`-decorated function simply delegates to the corresponding chain. The tool *docstrings* are critical — they're what the LLM reads to decide which tool to call. Notice how the docstrings explicitly list the types of questions each tool handles. Vague docstrings lead to wrong tool selection; specific docstrings lead to accurate routing.

**`@lru_cache(maxsize=1)` on `_get_chains()` and `build_agent()`.** Both chains load data and initialize models at construction time. Caching ensures this happens once, not per-request. The first call takes a few seconds (loading CSVs, initializing VectLite); subsequent calls return instantly.

**The system prompt enables multi-tool calls.** The line *"If the question requires both types of information, call both tools"* is critical. Without it, the agent tends to pick one tool and stop. With it, a question like *"What are the busiest hospitals and what do patients say about them?"* triggers two tool calls in sequence — first `structured_data` for rankings, then `reviews` for sentiment — and the agent synthesizes both results into a single answer.

### Agent Orchestration Entry Point

The final piece is the orchestration layer that runs the agent and formats the response:

```python
# chatbot_api/chains/rag_chain.py
import logging
from functools import lru_cache

from chatbot_api.agents.hospital_agent import build_agent
from chatbot_api.agents.retriever import VectLiteRetriever
from chatbot_api.models.schemas import ChatResponse, SourceDocument

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_retriever():
    return VectLiteRetriever()


def _extract_tools_used(messages: list) -> str:
    """Extract the names of tools used from agent messages."""
    tools = []
    for msg in messages:
        if hasattr(msg, 'type') and msg.type == 'tool':
            tools.append(msg.name)
    return ", ".join(dict.fromkeys(tools)) if tools else "agent"


async def ask(question: str) -> ChatResponse:
    """Run the multi-tool agent and return the response."""
    agent = build_agent()

    result = agent.invoke({"messages": [("human", question)]})
    messages = result.get("messages", [])

    # Last message = agent response (content can be str or list of blocks)
    raw_content = messages[-1].content if messages else ""
    if isinstance(raw_content, list):
        answer = "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in raw_content
        )
    else:
        answer = str(raw_content)

    tool_used = _extract_tools_used(messages)
    logger.info("Tools used: %s", tool_used)

    # Retrieve sources if the reviews tool was used
    sources = []
    if "reviews" in tool_used:
        retriever = _get_retriever()
        docs = retriever.invoke(question)
        sources = [
            SourceDocument(
                hospital_name=d.metadata.get("hospital_name", ""),
                score=round(d.metadata.get("score", 0), 4),
                visit_id=d.metadata.get("visit_id", 0),
                date_of_admission=d.metadata.get("date_of_admission", ""),
                review_excerpt=d.metadata.get("review_raw", "")[:200],
            )
            for d in docs
        ]

    return ChatResponse(answer=answer, tool_used=tool_used, sources=sources)
```

There's a subtle but important detail in the content parsing:

```python
if isinstance(raw_content, list):
    answer = "\n".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in raw_content
    )
else:
    answer = str(raw_content)
```

**Gemini returns different content formats depending on the agent's execution path.** When the agent calls a single tool, the final response is a plain string. But when it calls multiple tools, Gemini returns a **list of content blocks** — each block is a dict with a `"text"` key. Without this check, your API would crash with a 500 error on complex multi-tool questions. This is one of those production gotchas that only surfaces when you test with real user questions, not simple demos.

The `sources` list is also populated here — if the agent used the `reviews` tool, we re-run the retriever to get the source documents with their relevance scores. These are returned to the frontend for displaying source citations alongside the answer.

---

## Step 4: Deploy With FastAPI

The agent is built — now you need to serve it. The FastAPI backend exposes three endpoints that the frontend (or any HTTP client) can call:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/health` | GET | Health check — verifies VectLite is accessible |
| `/api/v1/chat` | POST | Send a question, get the agent's answer + sources |
| `/api/v1/stats` | GET | Dashboard stats — review counts per hospital |

### Pydantic Schemas

Every request and response has a strict schema. No loose dicts, no guessing what the API returns:

```python
# chatbot_api/models/schemas.py
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        examples=["Which hospital has the best reviews?"],
    )


class SourceDocument(BaseModel):
    hospital_name: str
    score: float
    visit_id: int
    date_of_admission: str
    review_excerpt: str


class ChatResponse(BaseModel):
    answer: str
    tool_used: str = ""
    sources: list[SourceDocument] = []


class HospitalStat(BaseModel):
    name: str
    review_count: int


class StatsResponse(BaseModel):
    total_reviews: int
    hospitals: list[HospitalStat]


class HealthResponse(BaseModel):
    status: str = "ok"
    vectordb_size: int
```

The `ChatResponse` schema deserves attention. It returns three things:

- **`answer`** — the agent's natural language response
- **`tool_used`** — which tool(s) the agent called (`"reviews"`, `"structured_data"`, or `"reviews, structured_data"` for multi-tool questions). The frontend uses this to show a badge indicating how the question was answered
- **`sources`** — when the reviews tool was used, a list of source documents with hospital names, relevance scores, and review excerpts. This gives the user transparency into *which specific reviews* informed the answer

### The FastAPI Application

```python
# chatbot_api/main.py
import logging

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from chatbot_api.chains.rag_chain import ask
from chatbot_api.config.settings import get_settings
from chatbot_api.models.schemas import (
    ChatRequest, ChatResponse, HealthResponse,
    HospitalStat, StatsResponse,
)
from chatbot_api.utils.vectordb import get_document_count, get_hospital_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title="Montreal Hospital RAG",
    description="Conversational API for Montreal hospital reviews",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=HealthResponse)
async def health():
    try:
        size = get_document_count()
        return HealthResponse(status="ok", vectordb_size=size)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.info("Question received: %s", request.question[:100])
    try:
        return await ask(request.question)
    except Exception as exc:
        logger.exception("Error during processing")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/stats", response_model=StatsResponse)
async def stats():
    try:
        total, hospitals = get_hospital_stats()
        return StatsResponse(
            total_reviews=total,
            hospitals=[HospitalStat(**h) for h in hospitals],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


app.include_router(router)
```

A few things to note:

**CORS middleware is wide open (`allow_origins=["*"]`).** For a local development project, this is fine — the Streamlit frontend runs on a different port (8501) than the API (8100), so cross-origin requests are expected. In production, you'd lock this down to specific domains.

**The `/health` endpoint checks VectLite directly.** It calls `get_document_count()` which opens the `.vdb` file in read-only mode and returns the document count. If the file is missing or corrupted, the endpoint returns 503. The frontend uses this to show a green "API connected" badge or a red error message.

**The `/chat` endpoint is `async` but the agent is synchronous.** FastAPI handles this correctly — it runs the synchronous `agent.invoke()` in a threadpool. This means multiple concurrent chat requests won't block each other, even though the agent itself is not async.

### Start the API

```bash
uvicorn chatbot_api.main:app --host 0.0.0.0 --port 8100 --reload
```

Test it with curl:

```bash
# Health check
curl http://localhost:8100/api/v1/health
# {"status":"ok","vectordb_size":7107}

# Ask a question
curl -X POST http://localhost:8100/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Which hospital has the most reviews?"}'

# Dashboard stats
curl http://localhost:8100/api/v1/stats
```

FastAPI also auto-generates interactive documentation at `http://localhost:8100/docs` — you can test all three endpoints directly from the browser.

---

## Step 5: Build the Frontend With Streamlit & Plotly

The frontend is a Streamlit application with two main areas: a **sidebar dashboard** showing hospital statistics with Plotly charts, and a **chat interface** where users interact with the agent.

### Frontend Configuration

Like the API, the frontend uses Pydantic Settings for configuration:

```python
# chatbot_front/config/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class FrontSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_base_url: str = "http://localhost:8100/api/v1"


settings = FrontSettings()
```

The only config needed is the API base URL. If you deploy the API elsewhere, just set `API_BASE_URL` in your `.env` or environment.

### The HTTP Client

The frontend communicates with the API through a thin `httpx` client:

```python
# chatbot_front/services/api_client.py
import httpx
from chatbot_front.config.settings import settings

_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


def chat(question: str) -> dict:
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post(
            f"{settings.api_base_url}/chat",
            json={"question": question},
        )
        resp.raise_for_status()
        return resp.json()


def get_stats() -> dict:
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.get(f"{settings.api_base_url}/stats")
        resp.raise_for_status()
        return resp.json()


def health_check() -> dict:
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.get(f"{settings.api_base_url}/health")
        resp.raise_for_status()
        return resp.json()
```

**The 120-second timeout is intentional.** Complex multi-tool questions can take 15-30 seconds — the agent calls Gemini multiple times (once per tool, plus the final synthesis). A default 5-second timeout would kill most real queries before the agent finishes reasoning.

### Plotly Chart Components

The charts are reusable Plotly components that render in both the sidebar and the chat:

```python
# chatbot_front/components/charts.py
import plotly.express as px
import plotly.graph_objects as go


def hospital_review_bar_chart(hospitals: list[dict]) -> go.Figure:
    """Horizontal bar chart: number of reviews per hospital."""
    names = [h["name"] for h in hospitals]
    counts = [h["review_count"] for h in hospitals]

    fig = px.bar(
        x=counts,
        y=names,
        orientation="h",
        labels={"x": "Number of reviews", "y": ""},
        color=counts,
        color_continuous_scale="Teal",
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=10),
        height=max(250, len(names) * 50),
        showlegend=False,
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed"),
    )
    return fig


def source_relevance_chart(sources: list[dict]) -> go.Figure:
    """Bar chart: relevance scores of returned sources."""
    labels = [
        f"{s['hospital_name'][:30]} (v:{s['visit_id']})"
        for s in sources
    ]
    scores = [s["score"] for s in sources]

    fig = px.bar(
        x=scores,
        y=labels,
        orientation="h",
        labels={"x": "Relevance score", "y": ""},
        color=scores,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=10),
        height=max(200, len(labels) * 38),
        showlegend=False,
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed"),
    )
    return fig
```

Two charts serve different purposes:

- **`hospital_review_bar_chart`** — appears in the sidebar dashboard, shows the distribution of reviews across hospitals. Uses a Teal color scale so higher counts stand out visually
- **`source_relevance_chart`** — appears inside the expandable "Sources and relevance" section under each chat answer. Uses Viridis to visualize relevance scores, showing which source documents the retriever ranked highest

Both charts dynamically adjust their height based on the number of items (`max(250, len(names) * 50)`) — so they don't look cramped with 3 hospitals or overflow with 20.

### The Sidebar Dashboard

The sidebar gives users an at-a-glance view of the system's state:

```python
# chatbot_front/components/sidebar.py
import streamlit as st
from chatbot_front.components.charts import hospital_review_bar_chart
from chatbot_front.services.api_client import get_stats, health_check


def render_sidebar():
    """Display the sidebar with the stats dashboard."""
    with st.sidebar:
        st.header("Dashboard")

        try:
            health = health_check()
            st.success(
                f"API connected — {health['vectordb_size']} indexed documents"
            )
        except Exception:
            st.error("API unavailable. Start `chatbot_api` first.")
            st.stop()

        st.divider()

        try:
            stats = get_stats()
            st.metric("Total indexed reviews", stats["total_reviews"])
            st.subheader("Reviews per hospital")
            fig = hospital_review_bar_chart(stats["hospitals"])
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.warning("Unable to load statistics")
```

The sidebar does three things in order:

1. **Health check** — calls `/api/v1/health`. If the API is down, `st.stop()` halts the entire app and shows a clear error. This prevents users from typing questions into a dead chat
2. **Metric card** — shows the total number of indexed reviews as a big number
3. **Plotly bar chart** — visualizes the distribution of reviews per hospital

### The Chat Interface

The main area is a conversational interface with session-based message history:

```python
# chatbot_front/components/chat_view.py
import streamlit as st
from chatbot_front.components.charts import source_relevance_chart
from chatbot_front.services.api_client import chat

TOOL_LABELS = {
    "reviews": ":mag: Search in patient reviews",
    "structured_data": ":bar_chart: Query on structured data",
}


def _tool_badge(tool_used: str) -> str:
    parts = [t.strip() for t in tool_used.split(",")]
    labels = [TOOL_LABELS.get(t, t) for t in parts]
    return " + ".join(labels)


def render_chat():
    """Display the RAG chat interface."""
    st.title("RAG — Montreal Hospital Reviews")
    st.caption("Multi-tool agent: semantic search in reviews + structured queries")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Your question about hospitals..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("The agent is analyzing your question..."):
                try:
                    result = chat(question)
                    answer = result["answer"]
                    sources = result.get("sources", [])
                    tool_used = result.get("tool_used", "")

                    if tool_used:
                        st.caption(_tool_badge(tool_used))

                    st.markdown(answer)

                    if sources:
                        with st.expander("Sources and relevance", expanded=False):
                            fig = source_relevance_chart(sources)
                            st.plotly_chart(fig, use_container_width=True)

                            for i, s in enumerate(sources, 1):
                                st.markdown(
                                    f"**{i}. {s['hospital_name']}** "
                                    f"(score: {s['score']:.4f})\n\n"
                                    f"> {s['review_excerpt']}..."
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                    })

                except Exception as exc:
                    st.error(f"Error: {exc}")
```

Several UX details make this chat feel polished:

**Tool badges show *how* the agent answered.** After each response, a caption like ":mag: Search in patient reviews" or ":bar_chart: Query on structured data" (or both joined with " + ") tells the user which tool was used. This builds trust — users can see that statistical questions use data queries, not fuzzy text search.

**Sources are expandable, not inline.** The "Sources and relevance" expander keeps the chat clean by default. Users who want to verify the answer can expand it to see:
- A Plotly chart of relevance scores for each source document
- A numbered list of source reviews with hospital name, score, and a 200-character excerpt

**Session state preserves history.** `st.session_state.messages` persists the conversation across Streamlit reruns. Users can scroll up through their entire conversation without losing context.

### The Streamlit Entry Point

```python
# chatbot_front/app.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from chatbot_front.components.chat_view import render_chat
from chatbot_front.components.sidebar import render_sidebar

st.set_page_config(
    page_title="Montreal Hospital RAG",
    page_icon=":hospital:",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_sidebar()
render_chat()
```

The `sys.path.insert` at the top is a necessary workaround. Streamlit runs `app.py` as a standalone script, not as part of a package — so `from chatbot_front.components import ...` would fail with `ModuleNotFoundError` unless the project root is on `sys.path`. This one line fixes that without requiring users to install the project as a package.

### Launch Everything

You need two terminal windows:

**Terminal 1 — Start the API:**

```bash
uvicorn chatbot_api.main:app --host 0.0.0.0 --port 8100 --reload
```

**Terminal 2 — Start the frontend:**

```bash
streamlit run chatbot_front/app.py
```

Open `http://localhost:8501` in your browser. You should see:

- A **sidebar** with a green "API connected — 7107 indexed documents" badge, a metric card, and a horizontal bar chart showing reviews per hospital
- A **chat area** with an input field at the bottom

Try these questions to test both tools:

| Question | Expected Tool |
|----------|---------------|
| *"How do patients describe the emergency room at Sacré-Cœur?"* | `reviews` |
| *"Which hospital has the most reviews?"* | `structured_data` |
| *"What is the phone number of CHUM?"* | `structured_data` |
| *"What do patients think about staff communication at Jewish General?"* | `reviews` |
| *"What are the top 3 hospitals by patient volume and how do patients rate the emergency services at each?"* | Both tools |

---

## Conclusion

You've built a complete, production-structured RAG chatbot from scratch. Let's recap what the system does and the key architectural decisions that make it work:

### What You Built

```
4 CSV files (hospitals, patients, visits, reviews)
        │
        ▼
   ETL Pipeline (extract → transform → load)
        │
        ├── OpenAI text-embedding-3-small (1536-dim, batched, ~10s)
        ├── vectlite.sparse_terms() for BM25 tokenization
        └── VectLite bulk_ingest() → single .vdb file (~22s)
        │
        ▼
   FastAPI Backend (port 8100)
        │
        ├── /health  → VectLite read-only connection check
        ├── /stats   → Pandas aggregation from source CSVs
        └── /chat    → LangGraph ReAct Agent (Gemini 2.5 Flash)
                          │
                          ├── reviews tool → VectLite hybrid search → RAG chain
                          └── structured_data tool → Pandas stats → LLM chain
        │
        ▼
   Streamlit + Plotly Frontend (port 8501)
        │
        ├── Sidebar: health badge, metric card, Plotly bar chart
        └── Chat: tool badges, expandable sources, relevance chart
```

### Key Takeaways

**1. VectLite eliminates the infrastructure barrier.** The entire vector database — 7,107 documents with 1536-dimensional HNSW vectors and a full BM25 inverted index — is a single `.vdb` file. No Docker, no server, no cloud account. `pip install vectlite`, open a file, search. If you've ever spent an afternoon debugging Docker volumes or Pinecone API keys before writing your first query, you'll appreciate how much friction this removes.

**2. Hybrid search is not optional for production RAG.** Dense-only retrieval misses keyword-specific queries ("Dr. Martin", "room 302", "January 15th"). Sparse-only retrieval misses semantic intent ("patients who felt unsafe" doesn't contain the word "dangerous"). VectLite's built-in dense + BM25 fusion — tunable with `dense_weight` and `sparse_weight` — gives you both in a single API call.

**3. Multi-tool agents beat single-retrieval pipelines.** A traditional RAG system would try to answer *"Which hospital has the most reviews and what do patients say about their emergency services?"* with a single vector search — and fail. The ReAct agent naturally decomposes this into two tool calls, uses the right data source for each, and synthesizes the results. LangGraph's `create_react_agent` makes this pattern surprisingly simple to implement.

**4. The details matter in production.** This project hit every common RAG pitfall:
- Gemini returns content as a list of blocks on multi-tool calls → parse both formats or crash with 500 errors
- VectLite acquires exclusive file locks by default → use `read_only=True` for concurrent API reads
- `bulk_ingest()` is 300x faster than individual `upsert()` calls → use it for initial data loading
- Streamlit runs scripts directly, not as packages → `sys.path.insert` fixes module imports

### Where to Go From Here

This project is a solid foundation. Here are ideas for extending it:

- **Add a conversation memory tool** — let the agent reference previous questions in the same session using LangGraph's built-in state management
- **Implement streaming responses** — use FastAPI's `StreamingResponse` with LangChain's streaming callbacks to show the agent's reasoning in real-time
- **Add more data sources** — physician directories, wait time APIs, hospital websites. Each new data source is just a new tool for the agent
- **Deploy with Docker** — ironically, VectLite's portability makes this trivial: mount the `.vdb` file as a volume and you're done
- **Fine-tune the hybrid weights** — experiment with different `dense_weight` / `sparse_weight` ratios for your specific query patterns. Try RRF fusion (`fusion="rrf"`) for queries where both dense and sparse results are equally important
- **Add evaluation** — build a test set of question/expected-answer pairs and measure retrieval precision, answer accuracy, and tool selection correctness

### Source Code

The complete source code for this project is available on GitHub:

**[https://github.com/Stefen-Taime/hospital-rag-vectlite-V2](https://github.com/Stefen-Taime/hospital-rag-vectlite-V2)**

Clone it, drop in your API keys, run the ETL pipeline, and you'll have a working chatbot in under 5 minutes.

---

*If you found this tutorial useful, check out [VectLite](https://vectlite.mcsedition.org/) for more on the embedded vector database that powers the project, and the [LangChain documentation](https://python.langchain.com/) for deeper dives into chains, retrievers, and agents.*
