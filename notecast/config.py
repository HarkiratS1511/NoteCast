"""App-wide settings: model IDs, paths and tuning knobs, loaded from the
environment and an optional .env file. This is the ONLY place model IDs
should ever be written — call sites should import them from here rather
than hard-coding a model name.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for NoteCast, read from the environment and `.env`.

    Every field is read as `NOTECAST_<FIELD_NAME>`, except the API keys and
    AgentAUS base URL, which are also read as plain `ANTHROPIC_API_KEY`,
    `AGENTAUS_API_KEY` and `AGENTAUS_BASE_URL`.

    `provider` picks the LLM backend. With `provider="agentaus"`, any of
    `chat_model` / `helper_model` / `script_model` / `deep_model` that was not
    set explicitly is replaced by `agentaus_model` (or `agentaus_helper_model`
    for the helper), so the Claude defaults are never sent to AgentAUS.
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
    # LLM backend: "agentaus" (Trellis Data's sovereign, OpenAI-compatible API)
    # or "anthropic" (Claude). See docs/AGENTAUS.md.
    provider: Literal["anthropic", "agentaus"] = "agentaus"
    agentaus_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AGENTAUS_API_KEY", "NOTECAST_AGENTAUS_API_KEY"),
    )
    # OpenAI-compatible base URL, including the /v1 suffix.
    agentaus_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AGENTAUS_BASE_URL", "NOTECAST_AGENTAUS_BASE_URL"),
    )
    # Model ID used for every role unless a role's model is set explicitly.
    agentaus_model: str | None = None
    # Optional cheaper/faster model for query rewriting and helper calls.
    agentaus_helper_model: str | None = None
    # Model context window; deep mode refuses material that won't fit.
    agentaus_context_tokens: int = 128_000
    # Cap applied to every request's max_tokens (servers reject larger values).
    agentaus_max_output_tokens: int = 8192
    # Send response_format={"type": "json_object"} when JSON is required.
    agentaus_json_mode: bool = True
    agentaus_timeout_seconds: float = 600.0
    # Optional prices (USD per million tokens) for cost estimates.
    agentaus_price_input_per_mtok: float | None = None
    agentaus_price_output_per_mtok: float | None = None
    notebooks_dir: Path = Path("notebooks")
    chat_model: str = "claude-sonnet-5"
    helper_model: str = "claude-haiku-4-5"
    script_model: str = "claude-sonnet-5"
    deep_model: str = "claude-sonnet-5"
    # Effort for chat answers (low | medium | high); lower = cheaper and faster.
    chat_effort: Literal["low", "medium", "high"] = "medium"
    chat_max_tokens: int = 8000
    # Max web searches Claude may run per question in "open" mode.
    web_search_max_uses: int = 3
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
    audio_min_minutes: int = 5
    # Spoken pace used to turn a time budget into a word budget.
    audio_words_per_minute: int = 150
    # Local text-to-speech (Kokoro via ONNX). Voices: see docs/LEARN.md.
    tts_model: Literal["fp32", "int8"] = "fp32"
    host_a_voice: str = "af_heart"
    host_b_voice: str = "bm_george"
    tts_speed: float = 1.0
    tts_lang: str = "en-us"
    audio_mp3_bitrate: int = 96

    @model_validator(mode="after")
    def _apply_agentaus_models(self) -> Settings:
        if self.provider != "agentaus" or not self.agentaus_model:
            return self
        explicit = self.model_fields_set
        for field in ("chat_model", "script_model", "deep_model"):
            if field not in explicit:
                setattr(self, field, self.agentaus_model)
        if "helper_model" not in explicit:
            self.helper_model = self.agentaus_helper_model or self.agentaus_model
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings, loaded once and cached.

    Tests that need different settings should call `get_settings.cache_clear()`
    after setting environment variables (see tests/conftest.py).
    """
    return Settings()
