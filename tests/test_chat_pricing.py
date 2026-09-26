"""Tests for notecast.chat.pricing."""

from __future__ import annotations

import pytest

from notecast.chat.pricing import (
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER,
    PRICES_PER_MTOK,
    WEB_SEARCH_PRICE_PER_1000,
    estimate_cost,
    provider_label,
)
from notecast.config import Settings


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


# --- AgentAUS -----------------------------------------------------------


def _agentaus_settings(**overrides: object) -> Settings:
    defaults = dict(
        _env_file=None,
        provider="agentaus",
        agentaus_api_key="key",
        agentaus_base_url="https://example.test/v1",
        agentaus_model="trellis-large",
        agentaus_price_input_per_mtok=3.0,
        agentaus_price_output_per_mtok=15.0,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def test_unknown_model_still_none_without_settings() -> None:
    assert estimate_cost("trellis-large", {"input_tokens": 100}) is None


def test_unknown_model_still_none_for_anthropic_provider() -> None:
    settings = Settings(_env_file=None, provider="anthropic")
    assert estimate_cost("trellis-large", {"input_tokens": 1_000_000}, settings=settings) is None


def test_unknown_model_none_when_agentaus_prices_unset() -> None:
    settings = _agentaus_settings(
        agentaus_price_input_per_mtok=None, agentaus_price_output_per_mtok=None
    )
    assert estimate_cost("trellis-large", {"input_tokens": 1_000_000}, settings=settings) is None


def test_agentaus_prices_input_and_output() -> None:
    settings = _agentaus_settings()
    cost = estimate_cost(
        "trellis-large",
        {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
        settings=settings,
    )
    assert cost == pytest.approx(3.0 + 15.0)


def test_agentaus_prices_cache_tokens_as_input_no_discount() -> None:
    settings = _agentaus_settings()
    cost = estimate_cost(
        "trellis-large",
        {"cache_read_tokens": 1_000_000, "cache_write_tokens": 1_000_000},
        settings=settings,
    )
    assert cost == pytest.approx(3.0 * 2)


def test_agentaus_ignores_web_searches() -> None:
    settings = _agentaus_settings()
    cost = estimate_cost("trellis-large", {"web_searches": 1000}, settings=settings)
    assert cost == pytest.approx(0.0)


def test_provider_label_anthropic() -> None:
    settings = Settings(_env_file=None, provider="anthropic")
    assert provider_label(settings) == "Claude"


def test_provider_label_agentaus() -> None:
    settings = _agentaus_settings()
    assert provider_label(settings) == "AgentAUS"
