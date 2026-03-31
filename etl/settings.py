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
