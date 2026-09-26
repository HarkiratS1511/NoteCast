"""Cost estimation for Claude API usage.

Prices are per-million-tokens (MTok), as of 2026-09 — verify against
https://claude.com/pricing before trusting these for real billing decisions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from notecast.config import Settings

# {model_id: {"input": $/MTok, "output": $/MTok}}
PRICES_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00},
}

CACHE_WRITE_MULTIPLIER = 1.25
# Default cache-read multiplier (of the model's input price); some models
# price cache reads differently — see _CACHE_READ_MULTIPLIER_OVERRIDES.
CACHE_READ_MULTIPLIER = 0.10
# claude-opus-5-5 cache reads are $0.20/MTok, i.e. 0.05x its $4 input price
# (not the usual 0.10x) — see notecast/chat/pricing.py's module docstring.
_CACHE_READ_MULTIPLIER_OVERRIDES: dict[str, float] = {
    "claude-opus-5-5": 0.05,
}
WEB_SEARCH_PRICE_PER_1000 = 10.00


def estimate_cost(model: str, usage: dict, *, settings: Settings | None = None) -> float | None:
    """Estimate the USD cost of one API call's usage, or None if `model`
    isn't in `PRICES_PER_MTOK` and can't be priced another way.

    `usage` is a plain dict with keys `input_tokens`, `output_tokens`,
    `cache_read_tokens`, `cache_write_tokens`, `web_searches` (all optional,
    default 0).

    If `model` isn't a known Claude model and `settings.provider ==
    "agentaus"` with both `agentaus_price_input_per_mtok` and
    `agentaus_price_output_per_mtok` set, the cost is estimated from those
    flat per-MTok prices instead (AgentAUS has no prompt caching, so cache
    read/write tokens are priced as ordinary input, and there's no web
    search cost). Otherwise this still returns None.
    """
    prices = PRICES_PER_MTOK.get(model)
    if prices is None:
        return _estimate_agentaus_cost(usage, settings)

    input_tokens = usage.get("input_tokens", 0) or 0
    output_tokens = usage.get("output_tokens", 0) or 0
    cache_read_tokens = usage.get("cache_read_tokens", 0) or 0
    cache_write_tokens = usage.get("cache_write_tokens", 0) or 0
    web_searches = usage.get("web_searches", 0) or 0
    cache_read_multiplier = _CACHE_READ_MULTIPLIER_OVERRIDES.get(model, CACHE_READ_MULTIPLIER)

    cost = (
        input_tokens / 1_000_000 * prices["input"]
        + output_tokens / 1_000_000 * prices["output"]
        + cache_write_tokens / 1_000_000 * prices["input"] * CACHE_WRITE_MULTIPLIER
        + cache_read_tokens / 1_000_000 * prices["input"] * cache_read_multiplier
        + web_searches / 1000 * WEB_SEARCH_PRICE_PER_1000
    )
    return cost


def _estimate_agentaus_cost(usage: dict, settings: Settings | None) -> float | None:
    if settings is None or settings.provider != "agentaus":
        return None
    price_in = settings.agentaus_price_input_per_mtok
    price_out = settings.agentaus_price_output_per_mtok
    if price_in is None or price_out is None:
        return None

    input_tokens = usage.get("input_tokens", 0) or 0
    output_tokens = usage.get("output_tokens", 0) or 0
    # No prompt caching on AgentAUS: cache read/write tokens (if any linger
    # in `usage` from shared call sites) are priced as ordinary input.
    cache_read_tokens = usage.get("cache_read_tokens", 0) or 0
    cache_write_tokens = usage.get("cache_write_tokens", 0) or 0
    total_input_tokens = input_tokens + cache_read_tokens + cache_write_tokens

    return total_input_tokens / 1_000_000 * price_in + output_tokens / 1_000_000 * price_out


def provider_label(settings: Settings) -> str:
    """A short display label for `settings.provider` ("AgentAUS" or
    "Claude"), for use in UI/CLI text.
    """
    return "AgentAUS" if settings.provider == "agentaus" else "Claude"
