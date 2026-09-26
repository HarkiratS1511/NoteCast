"""Tests for notecast.config: defaults, env-var overrides, and the special
plain ANTHROPIC_API_KEY alias.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from notecast.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = Settings(_env_file=None)
    assert settings.anthropic_api_key is None
    assert settings.notebooks_dir == Path("notebooks")
    assert settings.chat_model == "claude-sonnet-5"
    assert settings.helper_model == "claude-haiku-4-5"
    assert settings.script_model == "claude-sonnet-5"
    assert settings.deep_model == "claude-sonnet-5"
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.device == "auto"
    assert settings.retrieval_top_k == 10
    assert settings.audio_max_minutes == 45


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOTECAST_CHAT_MODEL", "claude-opus-9")
    monkeypatch.setenv("NOTECAST_RETRIEVAL_TOP_K", "5")
    settings = Settings(_env_file=None)
    assert settings.chat_model == "claude-opus-9"
    assert settings.retrieval_top_k == 5


def test_anthropic_api_key_read_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    settings = Settings(_env_file=None)
    assert settings.anthropic_api_key == "sk-test-123"


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOTECAST_CHAT_MODEL", "claude-cached")
    first = get_settings()
    monkeypatch.setenv("NOTECAST_CHAT_MODEL", "claude-changed")
    second = get_settings()
    assert first is second
    assert second.chat_model == "claude-cached"
