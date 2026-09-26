"""Turns ranked key points into a time/word budget and a chapter outline.

Pure functions, no API calls (see docs/PLAN.md §5).
"""

from __future__ import annotations

from notecast.audio.models import AudioPlan, AudioScope, ChapterPlan, RankedKeyPoint
from notecast.config import Settings
from notecast.models import Chunk

INTRO_OUTRO_MINUTES = 1.0
TIER_A_MINUTES = 1.5
TIER_B_MINUTES = 0.5
TIER_B_COMPRESSED_MINUTES = 0.25
TIER_C_MINUTES = 0.25
MIN_CHAPTERS = 3
MAX_CHAPTERS = 8
# If there's at least this much slack under the max, tier C points are
# included briefly instead of dropped.
TIER_C_SLACK_MINUTES = 2.0


def budget_minutes(points: list[RankedKeyPoint], settings: Settings) -> tuple[float, list[str]]:
    """Compute the target length in minutes, and notes describing any
    compression applied to fit `settings.audio_max_minutes`.

    1.5 min/tier-A point, 0.5 min/tier-B point, 0 for tier C, +1.0 intro/outro.
    Clamped to [audio_min_minutes, audio_max_minutes]. If the raw total is
    over the max: first compress tier B to 0.25 min each, then drop tier B
    entirely, then shrink tier A proportionally -- tier A points are never
    dropped.
    """
    notes: list[str] = []
    a_count = sum(1 for p in points if p.tier == "A")
    b_count = sum(1 for p in points if p.tier == "B")

    total = a_count * TIER_A_MINUTES + b_count * TIER_B_MINUTES + INTRO_OUTRO_MINUTES
    max_minutes = float(settings.audio_max_minutes)
    min_minutes = float(settings.audio_min_minutes)

    if total > max_minutes:
        compressed = (
            a_count * TIER_A_MINUTES + b_count * TIER_B_COMPRESSED_MINUTES + INTRO_OUTRO_MINUTES
        )
        if compressed <= max_minutes:
            total = compressed
            notes.append(
                "compressed tier B coverage from "
                f"{TIER_B_MINUTES} to {TIER_B_COMPRESSED_MINUTES} min each to fit the budget"
            )
        else:
            dropped = a_count * TIER_A_MINUTES + INTRO_OUTRO_MINUTES
            if dropped <= max_minutes:
                total = dropped
                notes.append("dropped tier B coverage entirely to fit the budget")
            else:
                total = max_minutes
                notes.append(
                    "shrank tier A coverage proportionally to fit the budget "
                    "(tier A points were never dropped)"
                )

    clamped = max(min_minutes, min(max_minutes, total))
    if clamped != total:
        if clamped == min_minutes and total < min_minutes:
            notes.append(f"clamped up to the minimum {min_minutes} min")
        elif clamped == max_minutes and total > max_minutes:
            notes.append(f"clamped down to the maximum {max_minutes} min")
    return clamped, notes


def _point_weight(point: RankedKeyPoint, include_c: bool) -> float:
    if point.tier == "A":
        return TIER_A_MINUTES
    if point.tier == "B":
        return TIER_B_MINUTES
    return TIER_C_MINUTES if include_c else 0.0


def _is_slide_chunk(chunk: Chunk) -> bool:
    """True if this chunk comes from a slide/page source rather than a
    transcript -- slides define the lecture's structure, so they sort
    before transcript chunks within the same week.
    """
    return chunk.location.slide is not None or chunk.location.page is not None


def _chunk_order_key(chunks_by_id: dict[str, Chunk]) -> dict[str, tuple[int, int, str, int]]:
    """For every chunk id, a (week, kind_rank, source_path, ordinal) sort
    key, with an absent week sorted last (a big sentinel) and kind_rank 0
    for slide/page chunks, 1 for transcript chunks -- so course order is
    week -> slides-before-transcript -> source_path -> ordinal.
    """
    keys: dict[str, tuple[int, int, str, int]] = {}
    for chunk_id, chunk in chunks_by_id.items():
        week = chunk.week if chunk.week is not None else 10**9
        kind_rank = 0 if _is_slide_chunk(chunk) else 1
        keys[chunk_id] = (week, kind_rank, chunk.source_path, chunk.ordinal)
    return keys


def _course_order(points: list[RankedKeyPoint], chunks: list[Chunk]) -> list[RankedKeyPoint]:
    """Order points by course order of first appearance: earliest
    (week, kind_rank, source_path, ordinal) among a point's own
    source_chunk_ids, preferring its earliest SLIDE chunk when it has one
    (a point present in both slides and a transcript is ordered by where
    it first appears on slides, not where the lecturer happens to mention
    it). Points with no known chunks sort last, in their given (ranked)
    order.
    """
    chunks_by_id = {c.chunk_id: c for c in chunks}
    order_keys = _chunk_order_key(chunks_by_id)
    sentinel = (10**9, 1, "￿", 10**9)

    def key(point: RankedKeyPoint) -> tuple[tuple[int, int, str, int], int]:
        candidates = [order_keys[cid] for cid in point.source_chunk_ids if cid in order_keys]
        slide_candidates = [c for c in candidates if c[1] == 0]
        best = (
            min(slide_candidates)
            if slide_candidates
            else (min(candidates) if candidates else sentinel)
        )
        return best, point.rank

    return sorted(points, key=key)


def _make_chapters(
    ordered: list[RankedKeyPoint],
    *,
    target_words: int,
    intro_words: int,
    outro_words: int,
) -> list[ChapterPlan]:
    """Group `ordered` points (already course-ordered, tier C already
    dropped where appropriate) into 3-8 chapters, weighting each chapter's
    target_words by the time weight of its points. `target_words` here is
    the content-only word budget (intro/outro words are added on top, to
    the first/last chapter respectively).
    """
    if not ordered:
        return []

    n_chapters = max(MIN_CHAPTERS, min(MAX_CHAPTERS, len(ordered)))
    n_chapters = min(n_chapters, len(ordered)) or 1

    # Split `ordered` into n_chapters contiguous groups of roughly equal size.
    groups: list[list[RankedKeyPoint]] = []
    base_size, remainder = divmod(len(ordered), n_chapters)
    start = 0
    for i in range(n_chapters):
        size = base_size + (1 if i < remainder else 0)
        if size == 0:
            continue
        groups.append(ordered[start : start + size])
        start += size

    weights = [
        sum(
            TIER_A_MINUTES if p.tier == "A" else TIER_B_MINUTES if p.tier == "B" else TIER_C_MINUTES
            for p in group
        )
        for group in groups
    ]
    total_weight = sum(weights) or 1.0

    chapters: list[ChapterPlan] = []
    for i, (group, weight) in enumerate(zip(groups, weights, strict=True)):
        words = round(target_words * weight / total_weight)
        if i == 0:
            words += intro_words
        if i == len(groups) - 1:
            words += outro_words
        title_point = max(group, key=lambda p: p.importance)
        chapters.append(
            ChapterPlan(
                index=i + 1,
                title=title_point.title,
                point_ids=[p.id for p in group],
                target_words=max(words, 1),
            )
        )
    return chapters


def plan_audio(
    points: list[RankedKeyPoint], scope: AudioScope, settings: Settings, chunks: list[Chunk]
) -> AudioPlan:
    """Build the full audio plan: length budget + chapter outline.

    `chunks` is used only to look up each point's earliest chunk for course
    ordering (week, then slides-before-transcript, then source_path, then
    chunk ordinal).
    """
    if not points:
        raise ValueError("No key points to plan")
    minutes, notes = budget_minutes(points, settings)
    target_words = round(minutes * settings.audio_words_per_minute)

    a_minutes = sum(TIER_A_MINUTES for p in points if p.tier == "A")
    b_minutes = sum(TIER_B_MINUTES for p in points if p.tier == "B")
    used_minutes = a_minutes + b_minutes + INTRO_OUTRO_MINUTES
    slack = settings.audio_max_minutes - used_minutes
    include_c = slack >= TIER_C_SLACK_MINUTES and any(p.tier == "C" for p in points)
    if include_c:
        notes.append(f"included tier C points briefly ({slack:.1f} min of slack under the max)")

    coverable = [p for p in points if p.tier in ("A", "B") or (p.tier == "C" and include_c)]
    ordered = _course_order(coverable, chunks)

    intro_words = round(INTRO_OUTRO_MINUTES / 2 * settings.audio_words_per_minute)
    outro_words = round(INTRO_OUTRO_MINUTES / 2 * settings.audio_words_per_minute)
    content_words = max(target_words - intro_words - outro_words, 0)
    chapters = _make_chapters(
        ordered,
        target_words=content_words,
        intro_words=intro_words,
        outro_words=outro_words,
    )

    return AudioPlan(
        scope=scope,
        points=points,
        target_minutes=minutes,
        target_words=target_words,
        chapters=chapters,
        notes=notes,
    )
