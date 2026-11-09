from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> parents[2] = project root (jahan .env hai)
ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Zaroori (Phase 1-3 mein chahiye)
    groq_api_key: SecretStr
    qdrant_url: str
    qdrant_api_key: SecretStr = SecretStr("")
    site_url: str
    resume_url: str
    qdrant_collection: str = "portfolio"
    groq_model: str = "openai/gpt-oss-120b"
    groq_reasoning_effort: str = "medium"   #/low, medium aur high
    postgres_url: str = ""

    # Aage ke phases mein chahiye, abhi khali chalega
    github_token: SecretStr = SecretStr("")
    langfuse_public_key: str = ""
    langfuse_secret_key: SecretStr = SecretStr("")
    langfuse_host: str = ""
    


@lru_cache
def get_settings() -> Settings:
    return Settings()