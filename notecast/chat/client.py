"""LLM client construction and error mapping for grounded chat.

`get_client` dispatches on `settings.provider`: "anthropic" returns a real
`anthropic.Anthropic`; "agentaus" returns `OpenAICompatClient`
(`notecast/providers/openai_compat.py`), which duck-types the same surface
but talks to AgentAUS's OpenAI-compatible API. Call sites should type the
client as `Any` (or a Protocol) rather than `anthropic.Anthropic`.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import anthropic

from notecast.config import Settings, get_settings


class MissingApiKeyError(RuntimeError):
    """Raised when no Anthropic (or AgentAUS) API key is configured."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or (
                "No Anthropic API key found. Get one at https://console.anthropic.com/ "
                "and add it to your .env file as:\n\n    ANTHROPIC_API_KEY=sk-ant-...\n"
            )
        )


class ProviderConfigError(MissingApiKeyError):
    """Raised when `settings.provider == "agentaus"` but AgentAUS isn't
    fully configured (missing base URL or model).
    """


class ChatError(RuntimeError):
    """A friendly, user-facing error raised when an LLM API call fails."""


@cache
def _cached_client(api_key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key)


# Cache of AgentAUS adapters, keyed by the settings that affect how the
# underlying `openai.OpenAI` client is built (or how requests are shaped).
# `Settings` isn't hashable, so this is a plain dict rather than `@cache`.
_agentaus_clients: dict[tuple[Any, ...], Any] = {}


def _agentaus_cache_key(settings: Settings) -> tuple[Any, ...]:
    return (
        settings.agentaus_api_key,
        settings.agentaus_base_url,
        settings.agentaus_timeout_seconds,
        settings.agentaus_max_output_tokens,
        settings.agentaus_json_mode,
    )


def _cached_agentaus_client(settings: Settings) -> Any:
    from notecast.providers.openai_compat import OpenAICompatClient

    key = _agentaus_cache_key(settings)
    client = _agentaus_clients.get(key)
    if client is None:
        client = OpenAICompatClient(settings)
        _agentaus_clients[key] = client
    return client


_cached_agentaus_client.cache_clear = _agentaus_clients.clear  # type: ignore[attr-defined]


def _get_agentaus_client(settings: Settings) -> Any:
    if not settings.agentaus_api_key:
        raise MissingApiKeyError(
            "No AgentAUS API key found. Add it to your .env file as:\n\n"
            "    AGENTAUS_API_KEY=...\n\nSee docs/AGENTAUS.md."
        )
    missing_lines = []
    if not settings.agentaus_base_url:
        missing_lines.append("AGENTAUS_BASE_URL=https://.../v1")
    if not settings.agentaus_model:
        missing_lines.append("NOTECAST_AGENTAUS_MODEL=<model id>")
    if missing_lines:
        lines = "\n".join(f"    {line}" for line in missing_lines)
        raise ProviderConfigError(
            "AgentAUS provider is missing required configuration. Add to your .env file:\n\n"
            f"{lines}\n\nSee docs/AGENTAUS.md for details."
        )
    return _cached_agentaus_client(settings)


def get_client(settings: Settings | None = None) -> Any:
    """Return a cached LLM client for `settings.provider`.

    Raises `MissingApiKeyError` (or its subclass `ProviderConfigError`) if
    the selected provider isn't fully configured.
    """
    settings = settings or get_settings()
    if settings.provider == "agentaus":
        return _get_agentaus_client(settings)
    if not settings.anthropic_api_key:
        raise MissingApiKeyError()
    return _cached_client(settings.anthropic_api_key)


def map_anthropic_error(exc: Exception) -> ChatError:
    """Map an `anthropic` SDK exception to a friendly `ChatError`, checking
    the most specific exception types first.
    """
    if isinstance(exc, anthropic.AuthenticationError):
        return ChatError(
            "Anthropic rejected the API key. Check ANTHROPIC_API_KEY in your .env file."
        )
    if isinstance(exc, anthropic.RateLimitError):
        retry_after = None
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            retry_after = headers.get("retry-after")
        message = "You've hit the Anthropic API rate limit."
        if retry_after:
            message += f" Retry after {retry_after}s."
        return ChatError(message)
    if isinstance(exc, anthropic.OverloadedError):
        return ChatError("Anthropic's API is temporarily overloaded. Please try again shortly.")
    if isinstance(exc, anthropic.NotFoundError):
        return ChatError(
            "Anthropic API returned 'not found' — check the model IDs in notecast/config.py."
        )
    if isinstance(exc, anthropic.APIStatusError):
        if exc.status_code >= 500:
            return ChatError("Anthropic's API had a server error. Please try again shortly.")
        return ChatError(f"Anthropic API request failed: {exc.message}")
    if isinstance(exc, anthropic.APIConnectionError):
        return ChatError("Could not connect to the Anthropic API. Check your network connection.")
    if isinstance(exc, anthropic.AnthropicError):
        return ChatError(f"Anthropic API error: {exc}")
    return ChatError(f"Unexpected error calling the Anthropic API: {exc}")
