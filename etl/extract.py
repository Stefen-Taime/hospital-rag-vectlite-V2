"""Extract: load and merge CSV data sources."""

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
