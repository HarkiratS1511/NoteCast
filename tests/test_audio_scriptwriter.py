"""Tests for notecast.audio.scriptwriter, with a fully fake Claude client —
no network access, no reads under notebooks/.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from notecast.audio.models import (
    AudioPlan,
    AudioScope,
    ChapterPlan,
    RankedKeyPoint,
)
from notecast.audio.prompts import SCRIPTWRITER_SYSTEM_PROMPT, build_material_block
from notecast.audio.scriptwriter import (
    ChatError,
    script_to_transcript_markdown,
    write_script,
)
from notecast.config import Settings
from notecast.models import Chunk, SourceType

# --- Fakes -------------------------------------------------------------


class FakeStreamContext:
    def __init__(self, message: Any) -> None:
        self._message = message

    def __enter__(self) -> FakeStreamContext:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def get_final_message(self) -> Any:
        return self._message


class FakeMessages:
    def __init__(
        self,
        stream_responses: list[Any] | None = None,
        create_responses: list[Any] | None = None,
    ) -> None:
        self._stream_responses = list(stream_responses or [])
        self._create_responses = list(create_responses or [])
        self.stream_calls: list[dict] = []
        self.create_calls: list[dict] = []

    def stream(self, **kwargs: Any) -> FakeStreamContext:
        self.stream_calls.append(kwargs)
        message = self._stream_responses.pop(0)
        if isinstance(message, Exception):
            raise message
        return FakeStreamContext(message)

    def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        message = self._create_responses.pop(0)
        if isinstance(message, Exception):
            raise message
        return message


class FakeClient:
    def __init__(
        self, stream_responses: list[Any] | None = None, create_responses: list[Any] | None = None
    ) -> None:
        self.messages = FakeMessages(stream_responses, create_responses)


# --- Fixtures ------------------------------------------------------------


def _settings() -> Settings:
    return Settings(
        anthropic_api_key="sk-ant-test",
        script_model="claude-sonnet-5",
        helper_model="claude-haiku-4-5",
    )


def _chunk(chunk_id: str, source_path: str, ordinal: int, text: str, header: str = "") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        course="comp4650",
        source_path=source_path,
        source_type=SourceType.PPTX,
        ordinal=ordinal,
        text=text,
        header=header,
    )


def _chunks() -> list[Chunk]:
    return [
        _chunk("c2", "week-01.pptx", 1, "Second slide text.", header="Week 1 · slide 2"),
        _chunk("c1", "week-01.pptx", 0, "First slide text.", header="Week 1 · slide 1"),
        _chunk("c3", "week-02.pptx", 0, "Week two slide text.", header="Week 2 · slide 1"),
    ]


def _plan() -> AudioPlan:
    points = [
        RankedKeyPoint(
            id="kp-1",
            title="Point one",
            summary="Explain point one.",
            tier="A",
            rank=1,
            evidence=["lecturer: this will be on the exam"],
            source_chunk_ids=["c1"],
        ),
        RankedKeyPoint(
            id="kp-2",
            title="Point two",
            summary="Explain point two briefly.",
            tier="B",
            rank=2,
            source_chunk_ids=["c2"],
        ),
        RankedKeyPoint(
            id="kp-3",
            title="Point three",
            summary="Explain point three.",
            tier="A",
            rank=3,
            source_chunk_ids=["c3"],
        ),
    ]
    chapters = [
        ChapterPlan(index=1, title="Getting started", point_ids=["kp-1", "kp-2"], target_words=200),
        ChapterPlan(index=2, title="Going deeper", point_ids=["kp-3"], target_words=150),
    ]
    return AudioPlan(
        scope=AudioScope(weeks=[1, 2]),
        points=points,
        target_minutes=5,
        target_words=350,
        chapters=chapters,
    )


def _usage(**overrides: Any) -> dict:
    usage = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    usage.update(overrides)
    return usage


def _chapter_message(
    lines: list[dict],
    recap: str = "Recap text.",
    *,
    stop_reason: str = "end_turn",
    stop_details: dict | None = None,
    **usage_overrides: Any,
) -> dict:
    message = {
        "content": [{"type": "text", "text": json.dumps({"lines": lines, "recap": recap})}],
        "stop_reason": stop_reason,
        "usage": _usage(**usage_overrides),
    }
    if stop_details is not None:
        message["stop_details"] = stop_details
    return message


def _raw_text_message(text: str, *, stop_reason: str = "end_turn", **usage_overrides: Any) -> dict:
    return {
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "usage": _usage(**usage_overrides),
    }


def _coverage_message(
    covered: list[str],
    missing: list[str],
    *,
    stop_reason: str = "end_turn",
    stop_details: dict | None = None,
    **usage_overrides: Any,
) -> dict:
    message = {
        "content": [
            {
                "type": "text",
                "text": json.dumps({"covered_ids": covered, "missing_ids": missing}),
            }
        ],
        "stop_reason": stop_reason,
        "usage": _usage(**usage_overrides),
    }
    if stop_details is not None:
        message["stop_details"] = stop_details
    return message


_LINES_CH1 = [
    {"speaker": "A", "text": "Point one is important.", "source_chunk_ids": ["c1"]},
    {"speaker": "B", "text": "Why does that matter?", "source_chunk_ids": []},
]
_LINES_CH2 = [
    {"speaker": "A", "text": "Point three explained.", "source_chunk_ids": ["c3"]},
]


def _full_coverage_run(client: FakeClient) -> Any:
    settings = _settings()
    return write_script(
        _plan(), _chunks(), client=client, settings=settings, course_name="COMP4650"
    )


# --- Tests -----------------------------------------------------------------


def test_material_block_identical_and_cache_controlled_first_block() -> None:
    client = FakeClient(
        stream_responses=[
            _chapter_message(_LINES_CH1),
            _chapter_message(_LINES_CH2),
        ],
        create_responses=[_coverage_message(["kp-1", "kp-3"], [])],
    )

    _full_coverage_run(client)

    assert len(client.messages.stream_calls) == 2
    materials = []
    for call in client.messages.stream_calls:
        content = call["messages"][0]["content"]
        first_block = content[0]
        assert first_block["type"] == "text"
        assert first_block["cache_control"] == {"type": "ephemeral"}
        materials.append(first_block["text"])
    assert materials[0] == materials[1]
    # Deterministic course order: week-01 chunks (by ordinal) before week-02.
    assert materials[0].index("[c1]") < materials[0].index("[c2]") < materials[0].index("[c3]")


def test_chapter_request_includes_points_and_running_recap() -> None:
    client = FakeClient(
        stream_responses=[
            _chapter_message(_LINES_CH1, recap="Chapter one covered points one and two."),
            _chapter_message(_LINES_CH2),
        ],
        create_responses=[_coverage_message(["kp-1", "kp-3"], [])],
    )

    _full_coverage_run(client)

    first_tail = client.messages.stream_calls[0]["messages"][0]["content"][1]["text"]
    assert "kp-1" in first_tail
    assert "kp-2" in first_tail
    assert "no prior recap" in first_tail
    assert "opening chapter" in first_tail

    second_tail = client.messages.stream_calls[1]["messages"][0]["content"][1]["text"]
    assert "kp-3" in second_tail
    assert "Chapter one covered points one and two." in second_tail
    assert "final chapter" in second_tail


def test_no_temperature_or_prefill_and_settings_used() -> None:
    client = FakeClient(
        stream_responses=[
            _chapter_message(_LINES_CH1),
            _chapter_message(_LINES_CH2),
        ],
        create_responses=[_coverage_message(["kp-1", "kp-3"], [])],
    )

    _full_coverage_run(client)

    chapter_kwargs = client.messages.stream_calls[0]
    assert chapter_kwargs["model"] == "claude-sonnet-5"
    assert chapter_kwargs["output_config"]["effort"] == "medium"
    assert "temperature" not in chapter_kwargs
    assert "top_p" not in chapter_kwargs
    assert "top_k" not in chapter_kwargs
    assert chapter_kwargs["messages"][-1]["role"] == "user"

    coverage_kwargs = client.messages.create_calls[0]
    assert coverage_kwargs["model"] == "claude-haiku-4-5"
    assert "effort" not in coverage_kwargs["output_config"]
    assert "temperature" not in coverage_kwargs


def test_coverage_triggers_regeneration_of_right_chapter_only_once() -> None:
    client = FakeClient(
        stream_responses=[
            _chapter_message(_LINES_CH1),
            _chapter_message(_LINES_CH2),
            # Regeneration of chapter 2 only.
            _chapter_message(
                [
                    {
                        "speaker": "A",
                        "text": "Point three explained properly.",
                        "source_chunk_ids": ["c3"],
                    }
                ],
                recap="Chapter two now covers point three in depth.",
            ),
        ],
        create_responses=[
            _coverage_message(["kp-1"], ["kp-3"]),  # first check: kp-3 missing
            _coverage_message(["kp-1", "kp-3"], []),  # re-check: now covered
        ],
    )

    script = _full_coverage_run(client)

    assert len(client.messages.stream_calls) == 3
    # Only chapter 2 was regenerated — its request carries the "missed" instruction.
    regen_tail = client.messages.stream_calls[2]["messages"][0]["content"][1]["text"]
    assert "IMPORTANT" in regen_tail
    assert "did not adequately explain" in regen_tail
    assert "kp-3" in regen_tail
    # Chapter 1 was not touched again — no fourth stream call, and coverage was
    # checked exactly twice (initial + one re-check).
    assert len(client.messages.create_calls) == 2

    assert script.coverage.patched == ["kp-3"]
    assert script.coverage.missing == []
    assert sorted(script.coverage.covered) == ["kp-1", "kp-3"]
    assert script.chapters[1].lines[0].text == "Point three explained properly."


def test_coverage_report_fields_when_nothing_missing() -> None:
    client = FakeClient(
        stream_responses=[
            _chapter_message(_LINES_CH1),
            _chapter_message(_LINES_CH2),
        ],
        create_responses=[_coverage_message(["kp-1", "kp-3"], [])],
    )

    script = _full_coverage_run(client)

    assert sorted(script.coverage.covered) == ["kp-1", "kp-3"]
    assert script.coverage.missing == []
    assert script.coverage.patched == []
    # Only one coverage call — no regeneration needed.
    assert len(client.messages.create_calls) == 1


def test_markdown_symbols_are_stripped_from_lines() -> None:
    lines = [
        {
            "speaker": "A",
            "text": "- **This** is `code` and # a heading with _emphasis_.",
            "source_chunk_ids": ["c1"],
        },
    ]
    client = FakeClient(
        stream_responses=[
            _chapter_message(lines),
            _chapter_message(_LINES_CH2),
        ],
        create_responses=[_coverage_message(["kp-1", "kp-3"], [])],
    )

    script = _full_coverage_run(client)

    text = script.chapters[0].lines[0].text
    for symbol in ("*", "#", "`", "_", "-"):
        assert symbol not in text
    assert "This is code and a heading with emphasis." in text


def test_invalid_speaker_and_empty_text_are_dropped_and_unknown_chunk_ids_filtered() -> None:
    lines = [
        {"speaker": "A", "text": "Valid line.", "source_chunk_ids": ["c1", "does-not-exist"]},
        {"speaker": "C", "text": "Invalid speaker.", "source_chunk_ids": []},
        {"speaker": "B", "text": "   ", "source_chunk_ids": []},
    ]
    client = FakeClient(
        stream_responses=[
            _chapter_message(lines),
            _chapter_message(_LINES_CH2),
        ],
        create_responses=[_coverage_message(["kp-1", "kp-3"], [])],
    )

    script = _full_coverage_run(client)

    chapter1_lines = script.chapters[0].lines
    assert len(chapter1_lines) == 1
    assert chapter1_lines[0].speaker == "A"
    assert chapter1_lines[0].source_chunk_ids == ["c1"]


def test_cost_accumulates_across_calls_including_cache_tokens() -> None:
    client = FakeClient(
        stream_responses=[
            _chapter_message(
                _LINES_CH1,
                input_tokens=1000,
                output_tokens=200,
                cache_read_input_tokens=500,
                cache_creation_input_tokens=100,
            ),
            _chapter_message(_LINES_CH2, input_tokens=1000, output_tokens=200),
        ],
        create_responses=[
            _coverage_message(["kp-1", "kp-3"], [], input_tokens=300, output_tokens=50)
        ],
    )

    script = _full_coverage_run(client)

    assert script.est_cost_usd is not None
    assert script.est_cost_usd > 0

    from notecast.chat.pricing import estimate_cost

    script_usage = {
        "input_tokens": 2000,
        "output_tokens": 400,
        "cache_read_tokens": 500,
        "cache_write_tokens": 100,
    }
    helper_usage = {
        "input_tokens": 300,
        "output_tokens": 50,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }
    expected = estimate_cost("claude-sonnet-5", script_usage) + estimate_cost(
        "claude-haiku-4-5", helper_usage
    )
    assert script.est_cost_usd == pytest.approx(expected)


def test_title_uses_course_name_and_scope_label() -> None:
    client = FakeClient(
        stream_responses=[
            _chapter_message(_LINES_CH1),
            _chapter_message(_LINES_CH2),
        ],
        create_responses=[_coverage_message(["kp-1", "kp-3"], [])],
    )

    script = _full_coverage_run(client)

    assert script.title == "COMP4650 — Weeks 1–2"


def test_transcript_markdown_format() -> None:
    client = FakeClient(
        stream_responses=[
            _chapter_message(_LINES_CH1),
            _chapter_message(_LINES_CH2),
        ],
        create_responses=[_coverage_message(["kp-1", "kp-3"], [])],
    )
    script = _full_coverage_run(client)
    chunk_lookup = {chunk.chunk_id: chunk for chunk in _chunks()}

    markdown = script_to_transcript_markdown(script, chunk_lookup)

    assert markdown.startswith(f"# {script.title}\n")
    assert "## Chapter 1: Getting started" in markdown
    assert "## Chapter 2: Going deeper" in markdown
    assert "**Host A:** Point one is important." in markdown
    assert "**Host B:** Why does that matter?" in markdown
    assert "Sources: Week 1 · slide 1" in markdown
    assert "Sources: Week 2 · slide 1" in markdown
    # No timing markers — the renderer adds those.
    assert "start_seconds" not in markdown


# --- Stop-reason guard: max_tokens retry / refusal / exhausted --------------


def test_chapter_max_tokens_retries_once_with_doubled_tokens_and_concise_instruction() -> None:
    client = FakeClient(
        stream_responses=[
            _chapter_message(_LINES_CH1, stop_reason="max_tokens"),
            _chapter_message(_LINES_CH1, recap="Retried recap."),
            _chapter_message(_LINES_CH2),
        ],
        create_responses=[_coverage_message(["kp-1", "kp-3"], [])],
    )

    script = _full_coverage_run(client)

    # Chapter 1 was called twice (initial truncation + retry), chapter 2 once.
    assert len(client.messages.stream_calls) == 3
    first_call, retry_call = client.messages.stream_calls[0], client.messages.stream_calls[1]
    # target_words=200 -> max(4000, 600) = 4000, doubled on retry -> 8000.
    assert first_call["max_tokens"] == 4000
    assert retry_call["max_tokens"] == 8000
    retry_tail = retry_call["messages"][0]["content"][1]["text"]
    assert "cut off for exceeding the length limit" in retry_tail
    assert "Be noticeably more concise" in retry_tail
    # The successful retry's content is what ends up in the script.
    assert script.chapters[0].lines[0].text == "Point one is important."


def test_chapter_refusal_raises_chat_error_naming_the_chapter() -> None:
    client = FakeClient(
        stream_responses=[
            _chapter_message(
                _LINES_CH1,
                stop_reason="refusal",
                stop_details={"category": "cyber"},
            ),
        ],
        create_responses=[],
    )

    with pytest.raises(ChatError) as exc_info:
        _full_coverage_run(client)

    message = str(exc_info.value)
    assert "Getting started" in message
    assert "cyber" in message


def test_chapter_max_tokens_exhausted_after_retry_raises_chat_error() -> None:
    client = FakeClient(
        stream_responses=[
            _chapter_message(_LINES_CH1, stop_reason="max_tokens"),
            _chapter_message(_LINES_CH1, stop_reason="max_tokens"),
        ],
        create_responses=[],
    )

    with pytest.raises(ChatError) as exc_info:
        _full_coverage_run(client)

    message = str(exc_info.value)
    assert "Getting started" in message
    assert "length limit" in message
    # No third attempt — capped at two attempts total.
    assert len(client.messages.stream_calls) == 2


def test_chapter_invalid_json_raises_chat_error_naming_the_chapter() -> None:
    client = FakeClient(
        stream_responses=[_raw_text_message("this is not valid json")],
        create_responses=[],
    )

    with pytest.raises(ChatError) as exc_info:
        _full_coverage_run(client)

    assert "Getting started" in str(exc_info.value)


def test_coverage_refusal_raises_chat_error() -> None:
    client = FakeClient(
        stream_responses=[
            _chapter_message(_LINES_CH1),
            _chapter_message(_LINES_CH2),
        ],
        create_responses=[
            _coverage_message([], [], stop_reason="refusal", stop_details={"category": "cyber"})
        ],
    )

    with pytest.raises(ChatError):
        _full_coverage_run(client)


def test_coverage_max_tokens_retries_once_with_doubled_tokens() -> None:
    client = FakeClient(
        stream_responses=[
            _chapter_message(_LINES_CH1),
            _chapter_message(_LINES_CH2),
        ],
        create_responses=[
            _coverage_message([], [], stop_reason="max_tokens"),
            _coverage_message(["kp-1", "kp-3"], []),
        ],
    )

    script = _full_coverage_run(client)

    assert len(client.messages.create_calls) == 2
    assert client.messages.create_calls[0]["max_tokens"] == 2000
    assert client.messages.create_calls[1]["max_tokens"] == 4000
    assert script.coverage.missing == []


def test_coverage_max_tokens_exhausted_raises_chat_error() -> None:
    client = FakeClient(
        stream_responses=[
            _chapter_message(_LINES_CH1),
            _chapter_message(_LINES_CH2),
        ],
        create_responses=[
            _coverage_message([], [], stop_reason="max_tokens"),
            _coverage_message([], [], stop_reason="max_tokens"),
        ],
    )

    with pytest.raises(ChatError, match="length limit"):
        _full_coverage_run(client)


# --- Missing derivation ignores the model's own missing_ids list -----------


def test_missing_is_derived_from_covered_ids_not_the_models_missing_list() -> None:
    # The model claims nothing is missing, but only confirms kp-1 as covered —
    # kp-3 (tier A) must still be treated as missing.
    client = FakeClient(
        stream_responses=[
            _chapter_message(_LINES_CH1),
            _chapter_message(_LINES_CH2),
            _chapter_message(_LINES_CH2, recap="Patched recap."),
        ],
        create_responses=[
            _coverage_message(["kp-1"], []),  # missing_ids=[] is inconsistent — ignored
            _coverage_message(["kp-1", "kp-3"], []),
        ],
    )

    script = _full_coverage_run(client)

    assert script.coverage.patched == ["kp-3"]
    assert len(client.messages.stream_calls) == 3


# --- Skip the re-check when no chapter owns the missing point --------------


def test_skips_second_coverage_check_when_no_chapter_owns_the_missing_point() -> None:
    orphan_point = RankedKeyPoint(
        id="kp-orphan",
        title="Orphan point",
        summary="Never assigned to a chapter.",
        tier="A",
        rank=4,
    )
    plan = _plan()
    plan.points.append(orphan_point)

    client = FakeClient(
        stream_responses=[
            _chapter_message(_LINES_CH1),
            _chapter_message(_LINES_CH2),
        ],
        # Only kp-1 and kp-3 confirmed covered -> kp-orphan is derived missing,
        # but no chapter's point_ids include it.
        create_responses=[_coverage_message(["kp-1", "kp-3"], [])],
    )

    script = write_script(plan, _chunks(), client=client, settings=_settings())

    assert len(client.messages.stream_calls) == 2  # no regeneration attempted
    assert len(client.messages.create_calls) == 1  # no re-check
    assert script.coverage.missing == ["kp-orphan"]
    assert script.coverage.patched == []
    assert any("kp-orphan" in note for note in script.coverage.notes)


# --- Recap capping -----------------------------------------------------


def test_running_recap_keeps_last_three_full_and_summarizes_earlier_titles() -> None:
    points = [
        RankedKeyPoint(id=f"kp-{i}", title=f"Point {i}", summary=f"About point {i}.", tier="B")
        for i in range(1, 6)
    ]
    chapters_plan = [
        ChapterPlan(index=i, title=f"Chapter {i}", point_ids=[f"kp-{i}"], target_words=100)
        for i in range(1, 6)
    ]
    plan = AudioPlan(
        scope=AudioScope(),
        points=points,
        target_minutes=10,
        target_words=500,
        chapters=chapters_plan,
    )
    stream_responses = [
        _chapter_message(
            [{"speaker": "A", "text": f"Line {i}.", "source_chunk_ids": []}],
            recap=f"Recap of chapter {i}.",
        )
        for i in range(1, 6)
    ]
    client = FakeClient(stream_responses=stream_responses, create_responses=[])

    write_script(plan, _chunks(), client=client, settings=_settings())

    fifth_tail = client.messages.stream_calls[4]["messages"][0]["content"][1]["text"]
    assert "Earlier chapters already covered (titles only): Chapter 1." in fifth_tail
    assert "Recap of chapter 2." in fifth_tail
    assert "Recap of chapter 3." in fifth_tail
    assert "Recap of chapter 4." in fifth_tail
    # Chapter 1's full recap text was dropped in favor of just its title.
    assert "Recap of chapter 1." not in fifth_tail


# --- Guard rails ---------------------------------------------------------


def test_empty_chunks_raises_value_error_before_any_api_call() -> None:
    client = FakeClient(stream_responses=[], create_responses=[])

    with pytest.raises(ValueError, match="Nothing in scope to talk about"):
        write_script(_plan(), [], client=client, settings=_settings())

    assert client.messages.stream_calls == []
    assert client.messages.create_calls == []


# --- Course order: week (numeric) beats lexicographic source path ----------


def test_course_order_sorts_by_week_numerically_then_source_path_then_ordinal() -> None:
    week_10 = _chunk("w10", "week-10/lecture.pptx", 0, "Week ten text.")
    week_10.week = 10
    week_2 = _chunk("w2", "week-2/lecture.pptx", 0, "Week two text.")
    week_2.week = 2
    no_week = _chunk("nw", "misc/appendix.pptx", 0, "No week text.")
    no_week.week = None

    material = build_material_block([week_10, week_2, no_week])

    # Numeric week order (2 before 10) beats what plain lexicographic string
    # sorting of "week-10" vs "week-2" would give, and unset weeks sort last.
    assert material.index("[w2]") < material.index("[w10]") < material.index("[nw]")


# --- Prompt content --------------------------------------------------------


def test_system_prompt_covers_stage_directions_speaker_labels_and_examples() -> None:
    assert "No stage directions" in SCRIPTWRITER_SYSTEM_PROMPT
    assert "Never say the speaker labels aloud" in SCRIPTWRITER_SYSTEM_PROMPT
    assert "invent an example that isn't in the material" in SCRIPTWRITER_SYSTEM_PROMPT
