"""FastAPI application entry point."""

import logging

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from chatbot_api.chains.rag_chain import ask
from chatbot_api.config.settings import get_settings
from chatbot_api.models.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    HospitalStat,
    StatsResponse,
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

# -- Routes ---------------------------------------------------------------

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
