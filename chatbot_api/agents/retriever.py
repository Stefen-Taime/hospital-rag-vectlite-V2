"""Retrieval agent — hybrid dense + sparse search via VectLite."""

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
