"""Chain for structured queries on hospital data via Pandas."""

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
    patient_counts = merged.groupby("hospital_name")["patient_id"].nunique().sort_values(ascending=False)
    lines.append("## Number of unique patients per hospital")
    for name, count in patient_counts.items():
        lines.append(f"- {name}: {count} patients")
    lines.append("")

    # Number of visits per hospital
    visit_counts = merged.groupby("hospital_name")["visit_id"].nunique().sort_values(ascending=False)
    lines.append("## Number of visits per hospital")
    for name, count in visit_counts.items():
        lines.append(f"- {name}: {count} visits")
    lines.append("")

    # Average reviews per visit
    avg_reviews = (review_counts / visit_counts).sort_values(ascending=False)
    lines.append("## Average reviews per visit")
    for name, avg in avg_reviews.items():
        lines.append(f"- {name}: {avg:.2f} reviews/visit")
    lines.append("")

    # Distribution by admission date
    dated = merged[merged["date_of_admission"] != "Non spécifié"]
    if not dated.empty:
        date_counts = dated.groupby("date_of_admission").size().sort_index()
        lines.append("## Distribution by admission date")
        for date, count in date_counts.items():
            lines.append(f"- {date}: {count} reviews")
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
        temperature=0.1,
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
