"""Transform: text cleaning and document construction."""

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
