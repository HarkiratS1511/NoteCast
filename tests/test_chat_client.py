"""Tests for notecast.chat.client: API key resolution and error mapping."""

from __future__ import annotations

import anthropic
import httpx2
import pytest

from notecast.chat.client import (
    ChatError,
    MissingApiKeyError,
    _cached_client,
    get_client,
    map_anthropic_error,
)
from notecast.config import Settings, get_settings


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
