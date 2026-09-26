"""Cost estimation for Claude API usage.

Prices are per-million-tokens (MTok), as of 2026-09 — verify against
https://claude.com/pricing before trusting these for real billing decisions.
"""

from __future__ import annotations

# {model_id: {"input": $/MTok, "output": $/MTok}}
PRICES_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00},
}

CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10
WEB_SEARCH_PRICE_PER_1000 = 10.00


def estimate_cost(model: str, usage: dict) -> float | None:
    """Estimate the USD cost of one API call's usage, or None if `model`
    isn't in `PRICES_PER_MTOK`.

    `usage` is a plain dict with keys `input_tokens`, `output_tokens`,
    `cache_read_tokens`, `cache_write_tokens`, `web_searches` (all optional,
    default 0).
    """
    prices = PRICES_PER_MTOK.get(model)
    if prices is None:
        return None

    input_tokens = usage.get("input_tokens", 0) or 0
    output_tokens = usage.get("output_tokens", 0) or 0
    cache_read_tokens = usage.get("cache_read_tokens", 0) or 0
    cache_write_tokens = usage.get("cache_write_tokens", 0) or 0
    web_searches = usage.get("web_searches", 0) or 0

    cost = (
        input_tokens / 1_000_000 * prices["input"]
        + output_tokens / 1_000_000 * prices["output"]
        + cache_write_tokens / 1_000_000 * prices["input"] * CACHE_WRITE_MULTIPLIER
        + cache_read_tokens / 1_000_000 * prices["input"] * CACHE_READ_MULTIPLIER
        + web_searches / 1000 * WEB_SEARCH_PRICE_PER_1000
    )
    return cost
