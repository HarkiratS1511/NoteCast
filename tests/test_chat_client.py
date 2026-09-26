"""Tests for notecast.chat.client: API key resolution and error mapping."""

from __future__ import annotations

import anthropic
import httpx2
import pytest

from notecast.chat.client import (
    ChatError,
    MissingApiKeyError,
    ProviderConfigError,
    _cached_client,
    get_client,
    map_anthropic_error,
)
from notecast.config import Settings, get_settings
from notecast.providers.openai_compat import OpenAICompatClient


@pytest.fixture(autouse=True)
def _clear_client_cache():
    _cached_client.cache_clear()
    yield
    _cached_client.cache_clear()


def _make_settings(api_key: str | None) -> Settings:
    get_settings.cache_clear()
    return Settings(anthropic_api_key=api_key, notebooks_dir="notebooks")


def test_missing_api_key_raises_friendly_error() -> None:
    settings = _make_settings(None)
    with pytest.raises(MissingApiKeyError) as exc_info:
        get_client(settings)
    message = str(exc_info.value)
    assert "console.anthropic.com" in message
    assert "ANTHROPIC_API_KEY" in message


def test_get_client_returns_client_and_is_cached() -> None:
    settings = _make_settings("sk-ant-test-key")
    client1 = get_client(settings)
    client2 = get_client(settings)
    assert isinstance(client1, anthropic.Anthropic)
    assert client1 is client2


def _response(status_code: int, headers: dict | None = None) -> httpx2.Response:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return httpx2.Response(status_code, request=request, headers=headers or {})


def test_map_authentication_error() -> None:
    exc = anthropic.AuthenticationError("bad key", response=_response(401), body=None)
    mapped = map_anthropic_error(exc)
    assert isinstance(mapped, ChatError)
    assert "ANTHROPIC_API_KEY" in str(mapped)


def test_map_rate_limit_error_includes_retry_after() -> None:
    exc = anthropic.RateLimitError(
        "too many requests", response=_response(429, {"retry-after": "12"}), body=None
    )
    mapped = map_anthropic_error(exc)
    assert "12" in str(mapped)


def test_map_overloaded_error() -> None:
    exc = anthropic.OverloadedError("overloaded", response=_response(529), body=None)
    mapped = map_anthropic_error(exc)
    assert "overloaded" in str(mapped).lower()


def test_map_server_error() -> None:
    exc = anthropic.InternalServerError("boom", response=_response(500), body=None)
    mapped = map_anthropic_error(exc)
    assert "server error" in str(mapped).lower()


def test_map_connection_error() -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    exc = anthropic.APIConnectionError(request=request)
    mapped = map_anthropic_error(exc)
    assert "connect" in str(mapped).lower()


def test_map_not_found_error() -> None:
    exc = anthropic.NotFoundError("model not found", response=_response(404), body=None)
    mapped = map_anthropic_error(exc)
    assert "model" in str(mapped).lower()


def test_map_unknown_error_still_returns_chat_error() -> None:
    mapped = map_anthropic_error(ValueError("something else"))
    assert isinstance(mapped, ChatError)


# --- get_client dispatch on provider ----------------------------------------


@pytest.fixture(autouse=True)
def _clear_agentaus_client_cache():
    from notecast.chat import client as client_module

    client_module._agentaus_clients.clear()
    yield
    client_module._agentaus_clients.clear()


def _agentaus_settings(**overrides) -> Settings:
    defaults = dict(
        _env_file=None,
        provider="agentaus",
        agentaus_api_key="test-key",
        agentaus_base_url="https://agentaus.example.com/v1",
        agentaus_model="agentaus-model",
        notebooks_dir="notebooks",
    )
    defaults.update(overrides)
    get_settings.cache_clear()
    return Settings(**defaults)


def test_get_client_dispatches_to_anthropic_by_default() -> None:
    settings = _make_settings("sk-ant-test-key")
    client = get_client(settings)
    assert isinstance(client, anthropic.Anthropic)


def test_get_client_dispatches_to_openai_compat_for_agentaus() -> None:
    settings = _agentaus_settings()
    client = get_client(settings)
    assert isinstance(client, OpenAICompatClient)


def test_get_client_agentaus_is_cached() -> None:
    settings = _agentaus_settings()
    client1 = get_client(settings)
    client2 = get_client(settings)
    assert client1 is client2


def test_get_client_agentaus_missing_api_key_raises_missing_api_key_error() -> None:
    settings = _agentaus_settings(agentaus_api_key=None)
    with pytest.raises(MissingApiKeyError) as exc_info:
        get_client(settings)
    assert "AGENTAUS_API_KEY" in str(exc_info.value)


def test_get_client_agentaus_missing_base_url_raises_provider_config_error() -> None:
    settings = _agentaus_settings(agentaus_base_url=None)
    with pytest.raises(ProviderConfigError) as exc_info:
        get_client(settings)
    message = str(exc_info.value)
    assert "AGENTAUS_BASE_URL" in message
    assert "docs/AGENTAUS.md" in message


def test_get_client_agentaus_missing_model_raises_provider_config_error() -> None:
    settings = _agentaus_settings(agentaus_model=None)
    with pytest.raises(ProviderConfigError) as exc_info:
        get_client(settings)
    assert "NOTECAST_AGENTAUS_MODEL" in str(exc_info.value)


def test_missing_api_key_error_default_message_unchanged() -> None:
    # Existing callers (CLI, other tests) rely on the zero-arg message
    # mentioning ANTHROPIC_API_KEY.
    message = str(MissingApiKeyError())
    assert "ANTHROPIC_API_KEY" in message
    assert "console.anthropic.com" in message
