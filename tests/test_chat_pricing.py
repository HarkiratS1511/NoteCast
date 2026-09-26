"""Tests for notecast.chat.pricing."""

from __future__ import annotations

import pytest

from notecast.chat.pricing import (
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER,
    PRICES_PER_MTOK,
    WEB_SEARCH_PRICE_PER_1000,
    estimate_cost,
)


def test_unknown_model_returns_none() -> None:
    assert estimate_cost("some-unknown-model", {"input_tokens": 100}) is None


def test_input_and_output_tokens_priced() -> None:
    cost = estimate_cost("claude-sonnet-5", {"input_tokens": 1_000_000, "output_tokens": 1_000_000})
    prices = PRICES_PER_MTOK["claude-sonnet-5"]
    assert cost == prices["input"] + prices["output"]


def test_cache_write_and_read_multipliers() -> None:
    prices = PRICES_PER_MTOK["claude-sonnet-5"]
    cost = estimate_cost(
        "claude-sonnet-5",
        {"cache_write_tokens": 1_000_000, "cache_read_tokens": 1_000_000},
    )
    expected = prices["input"] * CACHE_WRITE_MULTIPLIER + prices["input"] * CACHE_READ_MULTIPLIER
    assert cost == expected


def test_web_search_cost() -> None:
    cost = estimate_cost("claude-sonnet-5", {"web_searches": 1000})
    assert cost == WEB_SEARCH_PRICE_PER_1000


def test_missing_usage_fields_default_to_zero() -> None:
    assert estimate_cost("claude-sonnet-5", {}) == 0.0


def test_opus_5_5_has_lower_cache_read_multiplier() -> None:
    prices = PRICES_PER_MTOK["claude-opus-5-5"]
    cost = estimate_cost("claude-opus-5-5", {"cache_read_tokens": 1_000_000})
    assert cost == pytest.approx(prices["input"] * 0.05)
    assert cost == pytest.approx(0.20)


def test_other_models_keep_default_cache_read_multiplier() -> None:
    prices = PRICES_PER_MTOK["claude-opus-5"]
    cost = estimate_cost("claude-opus-5", {"cache_read_tokens": 1_000_000})
    assert cost == pytest.approx(prices["input"] * CACHE_READ_MULTIPLIER)


def test_haiku_and_opus_prices_present() -> None:
    assert PRICES_PER_MTOK["claude-haiku-4-5"] == {"input": 1.00, "output": 5.00}
    assert PRICES_PER_MTOK["claude-opus-5"] == {"input": 5.00, "output": 25.00}
    assert PRICES_PER_MTOK["claude-opus-5-5"] == {"input": 4.00, "output": 20.00}
