"""App-wide settings: model IDs, paths and tuning knobs, loaded from the
environment and an optional .env file. This is the ONLY place model IDs
should ever be written — call sites should import them from here rather
than hard-coding a model name.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for NoteCast, read from the environment and `.env`.

    Every field is read as `NOTECAST_<FIELD_NAME>`, except `anthropic_api_key`,
    which is read as the plain `ANTHROPIC_API_KEY` (matching the common
    convention used by the Anthropic SDK and most other tools).
    """

    model_config = SettingsConfigDict(
        env_prefix="NOTECAST_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "NOTECAST_ANTHROPIC_API_KEY"),
    )
    notebooks_dir: Path = Path("notebooks")
    chat_model: str = "claude-sonnet-5"
    helper_model: str = "claude-haiku-4-5"
    script_model: str = "claude-sonnet-5"
    deep_model: str = "claude-sonnet-5"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # "fastembed" = real local ONNX model; "hashing" = tiny offline stand-in (tests only).
    embedding_backend: Literal["fastembed", "hashing"] = "fastembed"
    model_cache_dir: Path = Path(".cache/models")
    device: Literal["auto", "cuda", "cpu"] = "auto"
    retrieval_top_k: int = 10
    # Optional cross-encoder reranker, applied to the top `rerank_candidates` hits.
    rerank_enabled: bool = False
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 30
    audio_max_minutes: int = 45


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings, loaded once and cached.

    Tests that need different settings should call `get_settings.cache_clear()`
    after setting environment variables (see tests/conftest.py).
    """
    return Settings()
