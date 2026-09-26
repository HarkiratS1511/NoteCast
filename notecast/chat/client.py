"""Anthropic client construction and error mapping for grounded chat."""

from __future__ import annotations

from functools import cache

import anthropic

from notecast.config import Settings, get_settings


class MissingApiKeyError(RuntimeError):
    """Raised when no Anthropic API key is configured."""

    def __init__(self) -> None:
        super().__init__(
            "No Anthropic API key found. Get one at https://console.anthropic.com/ "
            "and add it to your .env file as:\n\n    ANTHROPIC_API_KEY=sk-ant-...\n"
        )


class ChatError(RuntimeError):
    """A friendly, user-facing error raised when a Claude API call fails."""


@cache
def _cached_client(api_key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key)


def get_client(settings: Settings | None = None) -> anthropic.Anthropic:
    """Return a cached `anthropic.Anthropic` client, or raise
    `MissingApiKeyError` if no key is configured.
    """
    settings = settings or get_settings()
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
