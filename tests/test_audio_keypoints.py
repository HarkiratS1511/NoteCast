"""Tests for notecast.audio.keypoints -- no network, fake Anthropic client."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from notecast.audio.keypoints import (
    EXTRACTION_MAX_TOKENS,
    MERGE_MAX_TOKENS,
    STREAMING_THRESHOLD,
    KeyPointRun,
    _CompactMergeResult,
    _fallback_rank,
    _MergeResult,
    extract_key_points,
    group_by_source,
    merge_and_rank,
    select_chunks,
)
from notecast.audio.models import AudioScope, KeyPoint
from notecast.chat.client import ChatError
from notecast.config import Settings
from notecast.models import Chunk, Location, SourceType


def make_chunk(
    source_path: str,
    ordinal: int,
    *,
    text: str = "some lecture text",
    week: int | None = 1,
    slide: int | None = None,
    page: int | None = None,
    t_start: float | None = None,
    approx_minute: float | None = None,
    speaker: str | None = None,
    source_type: SourceType = SourceType.PDF,
    header: str = "",
) -> Chunk:
    location = Location(
        slide=slide,
        page=page,
        t_start=t_start,
        approx_minute=approx_minute,
        speaker=speaker,
    )
    chunk_id = Chunk.make_id("comp4650", source_path, ordinal, text)
    return Chunk(
        chunk_id=chunk_id,
        course="comp4650",
        source_path=source_path,
        source_type=source_type,
        ordinal=ordinal,
        text=text,
        header=header,
        location=location,
        week=week,
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, anthropic_api_key="sk-test")


def _agentaus_settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = dict(
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


# ---------------------------------------------------------------------------
# Fake Anthropic client
# ---------------------------------------------------------------------------


@dataclass
class FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 50
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeCreateResponse:
    content: list[FakeTextBlock]
    usage: FakeUsage = field(default_factory=FakeUsage)
    stop_reason: str = "end_turn"


@dataclass
class FakeParseResponse:
    parsed_output: Any
    usage: FakeUsage = field(default_factory=FakeUsage)
    stop_reason: str = "end_turn"
    content: list[FakeTextBlock] = field(default_factory=list)


class FakeStreamContext:
    """Mimics `with client.messages.stream(...) as stream: stream.get_final_message()`."""

    def __init__(self, response: Any) -> None:
        self._response = response

    def __enter__(self) -> FakeStreamContext:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None

    def get_final_message(self) -> Any:
        return self._response


class FakeMessages:
    """Scripted `.create` / `.parse` / `.stream` responses, in call order."""

    def __init__(
        self,
        create_responses: list[Any] | None = None,
        parse_responses: list[Any] | None = None,
        stream_responses: list[Any] | None = None,
    ) -> None:
        self.create_responses = list(create_responses or [])
        self.parse_responses = list(parse_responses or [])
        self.stream_responses = list(stream_responses or [])
        self.create_calls: list[dict[str, Any]] = []
        self.parse_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        return self.create_responses.pop(0)

    def parse(self, **kwargs: Any) -> Any:
        self.parse_calls.append(kwargs)
        return self.parse_responses.pop(0)

    def stream(self, **kwargs: Any) -> FakeStreamContext:
        self.stream_calls.append(kwargs)
        return FakeStreamContext(self.stream_responses.pop(0))


class FakeClient:
    def __init__(
        self,
        create_responses: list[Any] | None = None,
        parse_responses: list[Any] | None = None,
        stream_responses: list[Any] | None = None,
    ) -> None:
        self.messages = FakeMessages(create_responses, parse_responses, stream_responses)


def extraction_response(points: list[dict[str, Any]]) -> FakeCreateResponse:
    body = json.dumps({"key_points": points})
    return FakeCreateResponse(content=[FakeTextBlock(text=body)])


def merge_response(points: list[dict[str, Any]]) -> FakeParseResponse:
    parsed = _MergeResult.model_validate({"ranked_points": points})
    return FakeParseResponse(parsed_output=parsed)


def compact_merge_response(groups: list[dict[str, Any]]) -> FakeParseResponse:
    parsed = _CompactMergeResult.model_validate({"groups": groups})
    return FakeParseResponse(parsed_output=parsed)


# ---------------------------------------------------------------------------
# select_chunks / group_by_source
# ---------------------------------------------------------------------------


def test_select_chunks_filters_by_week_and_source_and_is_stable() -> None:
    c1 = make_chunk("b.pdf", 2, week=1)
    c2 = make_chunk("a.pdf", 1, week=1)
    c3 = make_chunk("a.pdf", 0, week=2)
    chunks = [c1, c2, c3]

    result = select_chunks(chunks, AudioScope(weeks=[1]))
    assert result == [c2, c1]  # sorted by (source_path, ordinal)

    result = select_chunks(chunks, AudioScope(source_paths=["a.pdf"]))
    assert result == [c3, c2]  # sorted by ordinal within a.pdf (0 before 1)

    result = select_chunks(chunks, AudioScope(weeks=[1], source_paths=["a.pdf"]))
    assert result == [c2]

    result = select_chunks(chunks, AudioScope())
    assert result == [c3, c2, c1]


def test_group_by_source_preserves_ordinal_order() -> None:
    c1 = make_chunk("a.pdf", 2)
    c2 = make_chunk("a.pdf", 0)
    c3 = make_chunk("b.pdf", 0)
    grouped = group_by_source([c1, c2, c3])
    assert set(grouped) == {"a.pdf", "b.pdf"}
    assert grouped["a.pdf"] == [c2, c1]
    assert grouped["b.pdf"] == [c3]


# ---------------------------------------------------------------------------
# extract_key_points
# ---------------------------------------------------------------------------


def test_extraction_prompt_contains_all_chunk_ids_and_focus(settings: Settings) -> None:
    chunks = [make_chunk("a.pdf", 0, text="alpha"), make_chunk("a.pdf", 1, text="beta")]
    client = FakeClient(create_responses=[extraction_response([])])

    extract_key_points("a.pdf", chunks, client=client, settings=settings, focus="exam prep")

    [call] = client.messages.create_calls
    prompt = call["messages"][0]["content"]
    for chunk in chunks:
        assert chunk.chunk_id in prompt
    assert "exam prep" in prompt


def test_extraction_drops_unknown_chunk_ids(settings: Settings) -> None:
    chunks = [make_chunk("a.pdf", 0, text="alpha")]
    response = extraction_response(
        [
            {
                "title": "T",
                "summary": "S",
                "kind": "concept",
                "importance": 4,
                "evidence": ["e"],
                "source_chunk_ids": [chunks[0].chunk_id, "not-a-real-id"],
            }
        ]
    )
    client = FakeClient(create_responses=[response])

    points = extract_key_points("a.pdf", chunks, client=client, settings=settings)

    assert len(points) == 1
    assert points[0].source_chunk_ids == [chunks[0].chunk_id]


def test_extraction_sets_in_slides_and_in_transcript_flags(settings: Settings) -> None:
    slide_chunk = make_chunk("a.pdf", 0, text="slide text", slide=3)
    transcript_chunk = make_chunk(
        "a.vtt", 0, text="spoken text", t_start=12.0, source_type=SourceType.VTT
    )
    chunks = [slide_chunk, transcript_chunk]

    response = extraction_response(
        [
            {
                "title": "Slide only",
                "summary": "S",
                "source_chunk_ids": [slide_chunk.chunk_id],
            },
            {
                "title": "Transcript only",
                "summary": "S",
                "source_chunk_ids": [transcript_chunk.chunk_id],
            },
            {
                "title": "Both",
                "summary": "S",
                "source_chunk_ids": [slide_chunk.chunk_id, transcript_chunk.chunk_id],
            },
        ]
    )
    client = FakeClient(create_responses=[response])

    points = extract_key_points("a.pdf", chunks, client=client, settings=settings)

    by_title = {p.title: p for p in points}
    assert by_title["Slide only"].in_slides is True
    assert by_title["Slide only"].in_transcript is False
    assert by_title["Transcript only"].in_slides is False
    assert by_title["Transcript only"].in_transcript is True
    assert by_title["Both"].in_slides is True
    assert by_title["Both"].in_transcript is True


def test_extraction_retries_once_on_invalid_json_then_succeeds(settings: Settings) -> None:
    chunks = [make_chunk("a.pdf", 0, text="alpha")]
    bad = FakeCreateResponse(content=[FakeTextBlock(text="not json")])
    good = extraction_response(
        [{"title": "T", "summary": "S", "source_chunk_ids": [chunks[0].chunk_id]}]
    )
    client = FakeClient(create_responses=[bad, good])

    points = extract_key_points("a.pdf", chunks, client=client, settings=settings)

    assert len(client.messages.create_calls) == 2
    assert len(points) == 1


def test_extraction_raises_after_two_invalid_responses(settings: Settings) -> None:
    chunks = [make_chunk("a.pdf", 0, text="alpha")]
    bad = FakeCreateResponse(content=[FakeTextBlock(text="not json")])
    client = FakeClient(create_responses=[bad, bad])

    with pytest.raises(ValueError):
        extract_key_points("a.pdf", chunks, client=client, settings=settings)


def test_extraction_splits_large_sources_into_multiple_calls(settings: Settings) -> None:
    # Each chunk's text is ~70k tokens' worth of characters, so 2 chunks
    # already exceed MAX_SOURCE_TOKENS (60k) and must split into 2 parts.
    big_text = "word " * 70_000  # ~70k tokens at ~4 chars/token estimate... use char count instead
    big_text = "x" * (61_000 * 4)  # ~61k tokens each chunk
    chunks = [make_chunk("a.pdf", 0, text=big_text), make_chunk("a.pdf", 1, text=big_text)]
    responses = [extraction_response([]) for _ in chunks]
    client = FakeClient(create_responses=responses)

    extract_key_points("a.pdf", chunks, client=client, settings=settings)

    assert len(client.messages.create_calls) == 2


def test_extraction_empty_chunks_makes_no_calls(settings: Settings) -> None:
    client = FakeClient(create_responses=[])
    assert extract_key_points("a.pdf", [], client=client, settings=settings) == []
    assert client.messages.create_calls == []


def test_extraction_tracks_usage_and_cost(settings: Settings) -> None:
    chunks = [make_chunk("a.pdf", 0, text="alpha")]
    response = extraction_response(
        [{"title": "T", "summary": "S", "source_chunk_ids": [chunks[0].chunk_id]}]
    )
    response.usage = FakeUsage(input_tokens=1000, output_tokens=200)
    client = FakeClient(create_responses=[response])

    run = KeyPointRun()
    extract_key_points("a.pdf", chunks, client=client, settings=settings, usage=run)

    assert run.calls == 1
    assert run.input_tokens == 1000
    assert run.output_tokens == 200
    assert run.est_cost_usd > 0


def test_extraction_tracks_agentaus_cost_when_priced() -> None:
    settings = _agentaus_settings()
    chunks = [make_chunk("a.pdf", 0, text="alpha")]
    response = extraction_response(
        [{"title": "T", "summary": "S", "source_chunk_ids": [chunks[0].chunk_id]}]
    )
    response.usage = FakeUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    client = FakeClient(create_responses=[response])

    run = KeyPointRun()
    extract_key_points("a.pdf", chunks, client=client, settings=settings, usage=run)

    # 1M input tokens @ $3/MTok + 1M output tokens @ $15/MTok.
    assert run.est_cost_usd == pytest.approx(18.0)


def test_extraction_agentaus_cost_stays_zero_when_unpriced() -> None:
    settings = _agentaus_settings(
        agentaus_price_input_per_mtok=None, agentaus_price_output_per_mtok=None
    )
    chunks = [make_chunk("a.pdf", 0, text="alpha")]
    response = extraction_response(
        [{"title": "T", "summary": "S", "source_chunk_ids": [chunks[0].chunk_id]}]
    )
    response.usage = FakeUsage(input_tokens=1000, output_tokens=200)
    client = FakeClient(create_responses=[response])

    run = KeyPointRun()
    extract_key_points("a.pdf", chunks, client=client, settings=settings, usage=run)

    # No AgentAUS prices configured -> estimate_cost returns None for every
    # call, so the accumulator's cost never advances past its 0.0 default
    # (KeyPointRun.est_cost_usd is a plain float, not Optional -- see
    # notecast/audio/keypoints.py::KeyPointRun.add).
    assert run.calls == 1
    assert run.est_cost_usd == 0.0


# ---------------------------------------------------------------------------
# merge_and_rank
# ---------------------------------------------------------------------------


def _key_point(title: str, **overrides: Any) -> KeyPoint:
    defaults: dict[str, Any] = dict(
        id="kp-x",
        title=title,
        summary="summary",
        importance=3,
        source_chunk_ids=["c1"],
    )
    defaults.update(overrides)
    return KeyPoint(**defaults)


def test_merge_and_rank_uses_parse_and_builds_ranked_points(settings: Settings) -> None:
    points_by_source = {"a.pdf": [_key_point("Alpha", source_chunk_ids=["c1"])]}
    response = merge_response(
        [
            {
                "title": "Alpha",
                "summary": "summary",
                "importance": 5,
                "source_chunk_ids": ["c1"],
                "in_slides": True,
                "in_transcript": False,
                "tier": "A",
            }
        ]
    )
    client = FakeClient(parse_responses=[response])

    ranked = merge_and_rank(points_by_source, client=client, settings=settings, focus="exam prep")

    assert len(ranked) == 1
    assert ranked[0].id == "kp-1"
    assert ranked[0].rank == 1
    assert ranked[0].tier == "A"
    [call] = client.messages.parse_calls
    assert "exam prep" in call["messages"][0]["content"]
    assert call["output_format"].__name__ == "_MergeResult"


def test_merge_and_rank_drops_unknown_chunk_ids(settings: Settings) -> None:
    points_by_source = {"a.pdf": [_key_point("Alpha", source_chunk_ids=["c1"])]}
    response = merge_response(
        [
            {
                "title": "Alpha",
                "summary": "summary",
                "source_chunk_ids": ["c1", "bogus"],
                "tier": "A",
            }
        ]
    )
    client = FakeClient(parse_responses=[response])

    ranked = merge_and_rank(points_by_source, client=client, settings=settings)
    assert ranked[0].source_chunk_ids == ["c1"]


def test_merge_and_rank_falls_back_deterministically_on_invalid_output(settings: Settings) -> None:
    points_by_source = {
        "a.pdf": [
            _key_point(
                "High both",
                importance=5,
                in_slides=True,
                in_transcript=True,
                source_chunk_ids=["c1"],
            ),
            _key_point("Low", importance=1, source_chunk_ids=["c2"]),
        ]
    }
    # Both attempts return a response with no usable parsed_output.
    bad1 = FakeParseResponse(parsed_output=None)
    bad2 = FakeParseResponse(parsed_output=None)
    client = FakeClient(parse_responses=[bad1, bad2])

    ranked = merge_and_rank(points_by_source, client=client, settings=settings)

    assert len(client.messages.parse_calls) == 2
    assert len(ranked) == 2
    # importance desc, both-sources first -> "High both" ranks first
    assert ranked[0].title == "High both"
    assert ranked[0].tier == "A"


def test_merge_and_rank_empty_input_returns_empty_without_calling_api(settings: Settings) -> None:
    client = FakeClient(parse_responses=[])
    assert merge_and_rank({}, client=client, settings=settings) == []
    assert client.messages.parse_calls == []


def test_merge_and_rank_tracks_usage_and_cost(settings: Settings) -> None:
    points_by_source = {"a.pdf": [_key_point("Alpha")]}
    response = merge_response(
        [{"title": "Alpha", "summary": "s", "source_chunk_ids": ["c1"], "tier": "A"}]
    )
    response.usage = FakeUsage(input_tokens=2000, output_tokens=500)
    client = FakeClient(parse_responses=[response])

    run = KeyPointRun()
    merge_and_rank(points_by_source, client=client, settings=settings, usage=run)

    assert run.calls == 1
    assert run.input_tokens == 2000
    assert run.output_tokens == 500
    assert run.est_cost_usd > 0


def test_merge_and_rank_tracks_agentaus_cost_when_priced() -> None:
    # AgentAUS routes through the compact merge path (see the
    # test_compact_merge_* suite below), which expects a {"groups": [...]}
    # response, not the full {"ranked_points": [...]} shape.
    settings = _agentaus_settings()
    points_by_source = {"a.pdf": [_key_point("Alpha")]}
    response = compact_merge_response([{"ids": ["p1"], "tier": "A"}])
    response.usage = FakeUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    client = FakeClient(parse_responses=[response])

    run = KeyPointRun()
    merge_and_rank(points_by_source, client=client, settings=settings, usage=run)

    assert run.est_cost_usd == pytest.approx(18.0)


def test_combined_usage_accumulates_across_both_calls(settings: Settings) -> None:
    chunks = [make_chunk("a.pdf", 0, text="alpha")]
    extraction = extraction_response(
        [{"title": "T", "summary": "S", "source_chunk_ids": [chunks[0].chunk_id]}]
    )
    extraction.usage = FakeUsage(input_tokens=500, output_tokens=100)
    merge_resp = merge_response(
        [{"title": "T", "summary": "S", "source_chunk_ids": [chunks[0].chunk_id], "tier": "A"}]
    )
    merge_resp.usage = FakeUsage(input_tokens=800, output_tokens=200)
    client = FakeClient(create_responses=[extraction], parse_responses=[merge_resp])

    run = KeyPointRun()
    points = extract_key_points("a.pdf", chunks, client=client, settings=settings, usage=run)
    merge_and_rank({"a.pdf": points}, client=client, settings=settings, usage=run)

    assert run.calls == 2
    assert run.input_tokens == 1300
    assert run.output_tokens == 300


# ---------------------------------------------------------------------------
# max_tokens headroom, stop_reason handling, id collisions (follow-up fixes)
# ---------------------------------------------------------------------------


def test_extraction_uses_16k_max_tokens(settings: Settings) -> None:
    chunks = [make_chunk("a.pdf", 0, text="alpha")]
    client = FakeClient(create_responses=[extraction_response([])])

    extract_key_points("a.pdf", chunks, client=client, settings=settings)

    assert EXTRACTION_MAX_TOKENS == 16_000
    assert client.messages.create_calls[0]["max_tokens"] == 16_000


def test_merge_uses_16k_max_tokens(settings: Settings) -> None:
    points_by_source = {"a.pdf": [_key_point("Alpha")]}
    response = merge_response([{"title": "Alpha", "summary": "s", "tier": "A"}])
    client = FakeClient(parse_responses=[response])

    merge_and_rank(points_by_source, client=client, settings=settings)

    assert MERGE_MAX_TOKENS == 16_000
    assert client.messages.parse_calls[0]["max_tokens"] == 16_000


def test_extraction_retries_once_with_doubled_max_tokens_on_max_tokens_stop(
    settings: Settings,
) -> None:
    chunks = [make_chunk("a.pdf", 0, text="alpha")]
    truncated = FakeCreateResponse(content=[FakeTextBlock(text="")], stop_reason="max_tokens")
    good = extraction_response(
        [{"title": "T", "summary": "S", "source_chunk_ids": [chunks[0].chunk_id]}]
    )
    # Doubled max_tokens (32000) exceeds STREAMING_THRESHOLD, so the retry goes via .stream().
    client = FakeClient(create_responses=[truncated], stream_responses=[good])

    points = extract_key_points("a.pdf", chunks, client=client, settings=settings)

    assert len(points) == 1
    [create_call] = client.messages.create_calls
    assert create_call["max_tokens"] == EXTRACTION_MAX_TOKENS
    [stream_call] = client.messages.stream_calls
    assert stream_call["max_tokens"] == EXTRACTION_MAX_TOKENS * 2
    assert stream_call["max_tokens"] > STREAMING_THRESHOLD


def test_extraction_raises_chat_error_on_refusal(settings: Settings) -> None:
    chunks = [make_chunk("a.pdf", 0, text="alpha")]
    refusal = FakeCreateResponse(content=[FakeTextBlock(text="")], stop_reason="refusal")
    client = FakeClient(create_responses=[refusal])

    with pytest.raises(ChatError):
        extract_key_points("a.pdf", chunks, client=client, settings=settings)


def test_merge_retries_once_with_doubled_max_tokens_on_max_tokens_stop(settings: Settings) -> None:
    points_by_source = {"a.pdf": [_key_point("Alpha", source_chunk_ids=["c1"])]}
    truncated = FakeParseResponse(parsed_output=None, stop_reason="max_tokens")
    # anthropic 1.8.0's client.messages.stream(..., output_format=...) populates
    # parsed_output on the final message the same way .parse() does.
    good_via_stream = merge_response(
        [{"title": "Alpha", "summary": "s", "source_chunk_ids": ["c1"], "tier": "A"}]
    )
    client = FakeClient(parse_responses=[truncated], stream_responses=[good_via_stream])

    ranked = merge_and_rank(points_by_source, client=client, settings=settings)

    assert len(ranked) == 1
    assert ranked[0].title == "Alpha"
    [parse_call] = client.messages.parse_calls
    assert parse_call["max_tokens"] == MERGE_MAX_TOKENS
    assert parse_call["output_format"] is _MergeResult
    [stream_call] = client.messages.stream_calls
    assert stream_call["max_tokens"] == MERGE_MAX_TOKENS * 2
    # output_format is passed straight through to .stream(); no hand-built schema.
    assert stream_call["output_format"] is _MergeResult
    assert "output_config" not in stream_call or "format" not in stream_call["output_config"]


def test_merge_raises_chat_error_on_refusal(settings: Settings) -> None:
    points_by_source = {"a.pdf": [_key_point("Alpha")]}
    refusal = FakeParseResponse(parsed_output=None, stop_reason="refusal")
    client = FakeClient(parse_responses=[refusal])

    with pytest.raises(ChatError):
        merge_and_rank(points_by_source, client=client, settings=settings)


def test_extraction_ids_do_not_collide_across_same_named_sources(settings: Settings) -> None:
    chunk_a = make_chunk("week-01/slides.pdf", 0, text="alpha")
    chunk_b = make_chunk("week-02/slides.pdf", 0, text="beta")
    response_a = extraction_response(
        [{"title": "A", "summary": "s", "source_chunk_ids": [chunk_a.chunk_id]}]
    )
    response_b = extraction_response(
        [{"title": "B", "summary": "s", "source_chunk_ids": [chunk_b.chunk_id]}]
    )
    client = FakeClient(create_responses=[response_a, response_b])

    points_a = extract_key_points("week-01/slides.pdf", [chunk_a], client=client, settings=settings)
    points_b = extract_key_points("week-02/slides.pdf", [chunk_b], client=client, settings=settings)

    assert points_a[0].id != points_b[0].id


# ---------------------------------------------------------------------------
# merge_and_rank -- AgentAUS compact path
# ---------------------------------------------------------------------------


def test_compact_merge_used_for_agentaus_full_prompt_for_anthropic() -> None:
    """AgentAUS dispatches through the compact `{"groups": [...]}` shape;
    Anthropic keeps using the full `{"ranked_points": [...]}` shape --
    same points, different `output_format` class and prompt.
    """
    points_by_source = {"a.pdf": [_key_point("Alpha", id="kp-1", source_chunk_ids=["c1"])]}

    agentaus_settings = _agentaus_settings()
    agentaus_client = FakeClient(
        parse_responses=[compact_merge_response([{"ids": ["p1"], "tier": "A"}])]
    )
    merge_and_rank(points_by_source, client=agentaus_client, settings=agentaus_settings)
    [agentaus_call] = agentaus_client.messages.parse_calls
    assert agentaus_call["output_format"] is _CompactMergeResult
    assert '"p1"' in agentaus_call["messages"][0]["content"]

    anthropic_settings = Settings(_env_file=None, anthropic_api_key="sk-test")
    anthropic_client = FakeClient(
        parse_responses=[
            merge_response([{"title": "Alpha", "summary": "s", "source_chunk_ids": ["c1"]}])
        ]
    )
    merge_and_rank(points_by_source, client=anthropic_client, settings=anthropic_settings)
    [anthropic_call] = anthropic_client.messages.parse_calls
    assert anthropic_call["output_format"] is _MergeResult
    assert '"p1"' not in anthropic_call["messages"][0]["content"]


def test_compact_merge_groups_duplicates_into_one_point() -> None:
    points_by_source = {
        "slides.pdf": [
            _key_point(
                "Gradient descent",
                id="kp-1",
                importance=3,
                source_chunk_ids=["c1"],
                in_slides=True,
                in_transcript=False,
            )
        ],
        "transcript.vtt": [
            _key_point(
                "Gradient descent (lecture)",
                id="kp-2",
                importance=5,
                summary="The lecturer's fuller explanation.",
                source_chunk_ids=["c2"],
                in_slides=False,
                in_transcript=True,
            )
        ],
    }
    response = compact_merge_response(
        [{"ids": ["p1", "p2"], "tier": "A", "title": "Gradient descent (merged)"}]
    )
    client = FakeClient(parse_responses=[response])

    ranked = merge_and_rank(points_by_source, client=client, settings=_agentaus_settings())

    assert len(ranked) == 1
    merged = ranked[0]
    assert merged.id == "kp-1"
    assert merged.rank == 1
    assert merged.tier == "A"
    assert merged.title == "Gradient descent (merged)"
    # Highest-importance member ("kp-2") is the base for summary/kind/evidence.
    assert merged.summary == "The lecturer's fuller explanation."
    assert merged.importance == 5
    assert merged.in_slides is True
    assert merged.in_transcript is True
    assert merged.source_chunk_ids == ["c1", "c2"]


def test_compact_merge_keeps_base_title_when_model_omits_one() -> None:
    points_by_source = {"a.pdf": [_key_point("Alpha", id="kp-1", source_chunk_ids=["c1"])]}
    response = compact_merge_response([{"ids": ["p1"], "tier": "B"}])  # no "title"
    client = FakeClient(parse_responses=[response])

    ranked = merge_and_rank(points_by_source, client=client, settings=_agentaus_settings())

    assert ranked[0].title == "Alpha"
    assert ranked[0].tier == "B"


def test_compact_merge_appends_ids_the_model_omitted() -> None:
    """An id the model never mentions is appended afterwards, ordered and
    tiered by `_fallback_rank`, so nothing extracted is silently dropped.
    """
    points_by_source = {
        "a.pdf": [
            _key_point("Alpha", id="kp-1", importance=5, source_chunk_ids=["c1"]),
            _key_point("Beta", id="kp-2", importance=1, source_chunk_ids=["c2"]),
        ]
    }
    # The model only groups p1 ("Alpha"); p2 ("Beta") is never mentioned.
    response = compact_merge_response([{"ids": ["p1"], "tier": "A"}])
    client = FakeClient(parse_responses=[response])

    ranked = merge_and_rank(points_by_source, client=client, settings=_agentaus_settings())

    assert [p.title for p in ranked] == ["Alpha", "Beta"]
    assert ranked[0].rank == 1
    assert ranked[1].rank == 2
    # Appended via _fallback_rank on the single remaining point -> tier A
    # (round(1 * 0.3) == 0, so index 0 of 1 point falls in the B slice)...
    # what matters here is just that it's present, tiered deterministically,
    # and not lost.
    fallback = _fallback_rank([points_by_source["a.pdf"][1]])
    assert ranked[1].tier == fallback[0].tier


def test_compact_merge_duplicate_id_first_group_wins() -> None:
    points_by_source = {
        "a.pdf": [
            _key_point("Alpha", id="kp-1", importance=5, source_chunk_ids=["c1"]),
            _key_point("Beta", id="kp-2", importance=1, source_chunk_ids=["c2"]),
        ]
    }
    # "p1" appears in both groups; the second group's claim on it is ignored.
    response = compact_merge_response(
        [
            {"ids": ["p1"], "tier": "A"},
            {"ids": ["p1", "p2"], "tier": "C"},
        ]
    )
    client = FakeClient(parse_responses=[response])

    ranked = merge_and_rank(points_by_source, client=client, settings=_agentaus_settings())

    assert len(ranked) == 2
    assert ranked[0].title == "Alpha"
    assert ranked[0].tier == "A"
    # Second group kept only "p2" once "p1" was already claimed.
    assert ranked[1].title == "Beta"
    assert ranked[1].tier == "C"
    assert ranked[1].source_chunk_ids == ["c2"]


def test_compact_merge_ignores_unknown_ids() -> None:
    points_by_source = {"a.pdf": [_key_point("Alpha", id="kp-1", source_chunk_ids=["c1"])]}
    response = compact_merge_response([{"ids": ["p1", "p99", "bogus"], "tier": "A"}])
    client = FakeClient(parse_responses=[response])

    ranked = merge_and_rank(points_by_source, client=client, settings=_agentaus_settings())

    assert len(ranked) == 1
    assert ranked[0].title == "Alpha"


def test_compact_merge_group_of_only_unknown_ids_is_dropped_not_empty() -> None:
    points_by_source = {"a.pdf": [_key_point("Alpha", id="kp-1", source_chunk_ids=["c1"])]}
    response = compact_merge_response(
        [
            {"ids": ["bogus"], "tier": "A"},
            {"ids": ["p1"], "tier": "B"},
        ]
    )
    client = FakeClient(parse_responses=[response])

    ranked = merge_and_rank(points_by_source, client=client, settings=_agentaus_settings())

    # The all-unknown group contributes no ranked point at all.
    assert len(ranked) == 1
    assert ranked[0].title == "Alpha"
    assert ranked[0].tier == "B"


def test_compact_merge_invalid_json_twice_falls_back_deterministically() -> None:
    points_by_source = {
        "a.pdf": [
            _key_point("Alpha", id="kp-1", importance=5, source_chunk_ids=["c1"]),
            _key_point("Beta", id="kp-2", importance=1, source_chunk_ids=["c2"]),
        ]
    }
    bad1 = FakeParseResponse(parsed_output=None)
    bad2 = FakeParseResponse(parsed_output=None)
    client = FakeClient(parse_responses=[bad1, bad2])

    ranked = merge_and_rank(points_by_source, client=client, settings=_agentaus_settings())

    assert len(client.messages.parse_calls) == 2
    assert len(ranked) == 2
    # Deterministic fallback: importance desc.
    assert ranked[0].title == "Alpha"
    assert ranked[1].title == "Beta"


def test_compact_merge_unions_chunk_ids_without_duplicates() -> None:
    """Chunk ids are derived locally from the grouped points (never asked
    of the model), unioned in first-seen order with no duplicates even
    when two members share a chunk id.
    """
    points_by_source = {
        "a.pdf": [
            _key_point("Alpha", id="kp-1", importance=3, source_chunk_ids=["c1", "c2"]),
            _key_point("Alpha (again)", id="kp-2", importance=5, source_chunk_ids=["c2", "c3"]),
        ]
    }
    response = compact_merge_response([{"ids": ["p1", "p2"], "tier": "A"}])
    client = FakeClient(parse_responses=[response])

    ranked = merge_and_rank(points_by_source, client=client, settings=_agentaus_settings())

    assert ranked[0].source_chunk_ids == ["c1", "c2", "c3"]


def test_compact_merge_tracks_usage_and_cost() -> None:
    settings = _agentaus_settings()
    points_by_source = {"a.pdf": [_key_point("Alpha", id="kp-1", source_chunk_ids=["c1"])]}
    response = compact_merge_response([{"ids": ["p1"], "tier": "A"}])
    response.usage = FakeUsage(input_tokens=2000, output_tokens=500)
    client = FakeClient(parse_responses=[response])

    run = KeyPointRun()
    merge_and_rank(points_by_source, client=client, settings=settings, usage=run)

    assert run.calls == 1
    assert run.input_tokens == 2000
    assert run.output_tokens == 500
    assert run.est_cost_usd > 0
