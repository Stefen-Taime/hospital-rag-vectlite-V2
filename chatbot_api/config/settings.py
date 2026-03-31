from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    vectordb_path: Path = base_dir / "data" / "vectordb" / "hospital_reviews.vdb"

    # OpenAI (embeddings)
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # Gemini (LLM)
    gemini_api_key: str
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.3

    # Retrieval
    top_k: int = 10
    dense_weight: float = 0.7
    sparse_weight: float = 0.3

    # Server
    host: str = "0.0.0.0"
    port: int = 8100
    cors_origins: list[str] = ["*"]


@lru_cache(maxsize=1)
def get_settings() -> ApiSettings:
    return ApiSettings()
