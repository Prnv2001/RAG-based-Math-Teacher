"""Application settings loaded from environment variables via pydantic-settings."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to the project root (4 levels up from this file:
# config/ -> math_teacher/ -> src/ -> project_root/)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """All application configuration. Values come from environment or .env file."""

    model_config = SettingsConfigDict(env_file=str(_ENV_FILE), extra="ignore")

    # ---- Database ----
    database_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/math_teacher"

    # ---- Ollama (kept for easy revert — not used when Groq is active) ----
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "deepseek-r1:latest"

    # ---- Groq Cloud LLM ----
    groq_api_key: str = ""           # Primary account key
    groq_api_key_fallback: str = ""  # Secondary account key (auto-used on rate-limit)
    groq_model: str = "openai/gpt-oss-20b"

    # ---- Embeddings ----
    embedding_model: str = "nomic-embed-text"
    embedding_dimensions: int = 768  # nomic-embed-text native dimension

    # ---- Retrieval ----
    retrieval_top_k: int = 20
    rerank_top_k: int = 8

    # ---- RAG Quality ----
    grounding_threshold: float = 0.25
    max_agent_retries: int = 2

    # ---- Context Builder ----
    max_context_chars: int = 12_000

    # ---- Chunking ----
    max_chunk_chars: int = 1_800

    # ---- Admin / Security ----
    admin_api_key: str = "change-me-before-deploy"


# Module-level singleton — import this everywhere
settings = Settings()
