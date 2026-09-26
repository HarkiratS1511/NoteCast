"""Tests for notecast.audio.planner -- pure functions, no API calls."""

from __future__ import annotations

from notecast.audio.models import AudioScope, RankedKeyPoint
from notecast.audio.planner import budget_minutes, plan_audio
from notecast.config import Settings
from notecast.models import Chunk, Location


def make_ranked_point(
    idx: int,
    *,
    tier: str = "A",
    importance: int = 4,
    chunk_id: str | None = None,
) -> RankedKeyPoint:
    return RankedKeyPoint(
        id=f"kp-{idx}",
        title=f"Point {idx}",
        summary="summary",
        importance=importance,
        source_chunk_ids=[chunk_id] if chunk_id else [],
        tier=tier,
        rank=idx,
    )


def make_chunk(course: str, source_path: str, ordinal: int, *, week: int | None = 1) -> Chunk:
    text = f"text-{source_path}-{ordinal}"
    return Chunk(
        chunk_id=Chunk.make_id(course, source_path, ordinal, text),
        course=course,
        source_path=source_path,
        source_type="pdf",
        ordinal=ordinal,
        text=text,
        location=Location(),
        week=week,
    )


def settings_with(**overrides) -> Settings:
    defaults = dict(
        _env_file=None,
        anthropic_api_key="sk-test",
        audio_min_minutes=5,
        audio_max_minutes=45,
        audio_words_per_minute=150,
    )
    defaults.update(overrides)
    return Settings(**defaults)


# ---------------------------------------------------------------------------
# budget_minutes
# ---------------------------------------------------------------------------


def test_budget_10a_10b_is_21_minutes() -> None:
    points = [make_ranked_point(i, tier="A") for i in range(10)] + [
        make_ranked_point(10 + i, tier="B") for i in range(10)
    ]
    minutes, notes = budget_minutes(points, settings_with())
    assert minutes == 21.0
    assert notes == []


def test_budget_compresses_when_over_max_and_notes_each_step() -> None:
    points = [make_ranked_point(i, tier="A") for i in range(40)] + [
        make_ranked_point(40 + i, tier="B") for i in range(40)
    ]
    minutes, notes = budget_minutes(points, settings_with())
    assert minutes <= 45.0
    assert notes  # some compression note recorded
    joined = " ".join(notes)
    assert "tier A" in joined or "tier B" in joined


def test_budget_compression_never_implied_to_drop_tier_a() -> None:
    # Way over budget: many A points alone already exceed the max.
    points = [make_ranked_point(i, tier="A") for i in range(60)]
    minutes, notes = budget_minutes(points, settings_with())
    assert minutes == 45.0
    assert any("tier A" in n for n in notes)


def test_budget_clamped_up_to_minimum_for_tiny_scope() -> None:
    points = [make_ranked_point(0, tier="B")]
    minutes, notes = budget_minutes(points, settings_with())
    assert minutes == 5.0
    assert notes


def test_budget_tier_c_never_adds_minutes() -> None:
    points = [make_ranked_point(0, tier="C") for _ in range(5)]
    minutes, _ = budget_minutes(points, settings_with())
    assert minutes == 5.0  # clamped to min, since C contributes 0 and total is 1.0


# ---------------------------------------------------------------------------
# plan_audio
# ---------------------------------------------------------------------------


def test_plan_audio_orders_chapters_in_course_order() -> None:
    course = "comp4650"
    c_week2 = make_chunk(course, "week2.pdf", 0, week=2)
    c_week1_late = make_chunk(course, "week1.pdf", 5, week=1)
    c_week1_early = make_chunk(course, "week1.pdf", 0, week=1)
    chunks = [c_week2, c_week1_late, c_week1_early]

    points = [
        make_ranked_point(1, tier="A", chunk_id=c_week2.chunk_id),
        make_ranked_point(2, tier="A", chunk_id=c_week1_late.chunk_id),
        make_ranked_point(3, tier="A", chunk_id=c_week1_early.chunk_id),
    ]
    plan = plan_audio(points, AudioScope(), settings_with(), chunks)

    all_point_ids_in_order = [pid for chapter in plan.chapters for pid in chapter.point_ids]
    assert all_point_ids_in_order == ["kp-3", "kp-2", "kp-1"]


def test_plan_audio_produces_3_to_8_chapters_and_words_sum_close_to_target() -> None:
    course = "comp4650"
    chunks = [make_chunk(course, "week1.pdf", i) for i in range(20)]
    points = [
        make_ranked_point(i, tier="A" if i % 2 == 0 else "B", chunk_id=chunks[i].chunk_id)
        for i in range(20)
    ]
    plan = plan_audio(points, AudioScope(), settings_with(), chunks)

    assert 3 <= len(plan.chapters) <= 8
    total_words = sum(c.target_words for c in plan.chapters)
    # intro/outro words are folded into the first/last chapter, so the sum
    # should be close to target_words (+/- rounding).
    assert abs(total_words - plan.target_words) <= len(plan.chapters)


def test_plan_audio_drops_tier_c_when_no_slack() -> None:
    course = "comp4650"
    chunks = [make_chunk(course, "week1.pdf", i) for i in range(30)]
    points = [make_ranked_point(i, tier="A", chunk_id=chunks[i].chunk_id) for i in range(30)] + [
        make_ranked_point(30, tier="C", chunk_id=chunks[0].chunk_id)
    ]
    plan = plan_audio(points, AudioScope(), settings_with(), chunks)

    all_point_ids = {pid for chapter in plan.chapters for pid in chapter.point_ids}
    assert "kp-30" not in all_point_ids  # the tier-C point


def test_plan_audio_includes_tier_c_when_slack_available() -> None:
    course = "comp4650"
    chunks = [make_chunk(course, "week1.pdf", i) for i in range(4)]
    points = [make_ranked_point(i, tier="A", chunk_id=chunks[i].chunk_id) for i in range(3)] + [
        make_ranked_point(3, tier="C", chunk_id=chunks[3].chunk_id)
    ]
    plan = plan_audio(points, AudioScope(), settings_with(), chunks)

    all_point_ids = {pid for chapter in plan.chapters for pid in chapter.point_ids}
    assert "kp-3" in all_point_ids
    assert any("tier C" in n for n in plan.notes)


def test_plan_audio_keeps_all_tier_a_points() -> None:
    course = "comp4650"
    chunks = [make_chunk(course, "week1.pdf", i) for i in range(15)]
    points = [make_ranked_point(i, tier="A", chunk_id=chunks[i].chunk_id) for i in range(15)]
    plan = plan_audio(points, AudioScope(), settings_with(), chunks)

    all_point_ids = {pid for chapter in plan.chapters for pid in chapter.point_ids}
    assert all_point_ids == {f"kp-{i}" for i in range(15)}
