"""Load: generate embeddings via OpenAI and ingest via VectLite bulk_ingest."""

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
        resp = _client.embeddings.create(model=settings.embedding_model, input=chunk)
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
    db = vectlite.open(str(settings.vectordb_path), dimension=settings.embedding_dimension)
    db.bulk_ingest(records)

    logger.info("Ingestion complete — %d documents -> %s", total, settings.vectordb_path)
