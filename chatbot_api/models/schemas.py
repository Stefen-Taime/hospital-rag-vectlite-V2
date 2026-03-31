"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel, Field


# -- Chat -----------------------------------------------------------------

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


# -- Stats ----------------------------------------------------------------

class HospitalStat(BaseModel):
    name: str
    review_count: int


class StatsResponse(BaseModel):
    total_reviews: int
    hospitals: list[HospitalStat]


# -- Health ---------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    vectordb_size: int
