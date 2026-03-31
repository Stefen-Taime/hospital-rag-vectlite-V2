from pydantic_settings import BaseSettings, SettingsConfigDict


class FrontSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_base_url: str = "http://localhost:8100/api/v1"


settings = FrontSettings()
