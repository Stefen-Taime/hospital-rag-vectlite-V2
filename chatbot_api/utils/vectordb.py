"""Utilities for accessing the VectLite database and source data."""

from collections import Counter
from pathlib import Path

import pandas as pd
import vectlite

from chatbot_api.config.settings import get_settings

_settings = get_settings()
_DATA_DIR = _settings.base_dir / "data" / "raw"


def get_db():
    """Return a connection to the VectLite database."""
    return vectlite.open(str(_settings.vectordb_path), dimension=_settings.embedding_dimension, read_only=True)


def get_document_count() -> int:
    return get_db().count()


def get_hospital_stats() -> tuple[int, list[dict]]:
    """Return the total number of reviews and the distribution per hospital from source CSVs."""
    visits = pd.read_csv(_DATA_DIR / "visits.csv")
    hospitals = pd.read_csv(_DATA_DIR / "hospitals.csv")
    reviews = pd.read_csv(_DATA_DIR / "reviews.csv")

    merged = visits.merge(hospitals, on="hospital_id", how="left").merge(
        reviews, on="visit_id", how="inner"
    )

    counts = merged.groupby("hospital_name").size().sort_values(ascending=False)
    total = int(counts.sum())

    hospital_list = [
        {"name": name, "review_count": int(count)}
        for name, count in counts.items()
    ]

    return total, hospital_list
