"""
app/config.py - Central configuration management
"""
from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    APP_NAME: str = "Semantic Code Fusion"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    SECRET_KEY: str = "change-me-in-production"
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:8000"

    # OpenAI
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API Key")
    OPENAI_MODEL: str = "gpt-4o"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://scf_user:scf_pass@localhost:5432/semantic_code_fusion"
    SYNC_DATABASE_URL: str = "postgresql://scf_user:scf_pass@localhost:5432/semantic_code_fusion"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # Vector Store
    FAISS_INDEX_PATH: str = "./data/faiss_index"
    FAISS_DIMENSION: int = 1536

    # Fusion
    MAX_CODE_LENGTH: int = 50000
    DEFAULT_FUSION_TEMPERATURE: float = 0.2
    MAX_RETRIES: int = 3
    FUSION_TIMEOUT_SECONDS: int = 120

    # Security
    ENABLE_SECURITY_SCAN: bool = True
    SECURITY_SCAN_LEVEL: str = "medium"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"

    @property
    def allowed_origins_list(self) -> List[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
