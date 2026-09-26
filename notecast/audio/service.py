"""Wires the audio-overview pipeline (key points -> plan -> script -> speech)
into one entry point per notebook: `generate_overview` runs the whole thing
from a scope, and `render_saved_script` re-renders a previously-saved script
without paying for the Claude calls again.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field

from notecast.audio.keypoints import (
    MAX_SOURCE_TOKENS,
    KeyPointRun,
    extract_key_points,
    group_by_source,
    merge_and_rank,
    select_chunks,
)
from notecast.audio.models import AudioScope, AudioScript, RenderResult
from notecast.audio.planner import MAX_CHAPTERS, MIN_CHAPTERS, plan_audio
from notecast.audio.render import render_script
from notecast.audio.scriptwriter import script_to_transcript_markdown, write_script
from notecast.audio.tts import KokoroTTS
from notecast.chat.client import ChatError, get_client
from notecast.chat.pricing import estimate_cost
from notecast.config import Settings, get_settings
from notecast.index.service import get_retriever
from notecast.ingest.course import display_name, load_course_config
from notecast.ingest.textutils import estimate_tokens
from notecast.models import Chunk
from notecast.notebook import Notebook

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_ProgressFn = Callable[[str, float], None]

# --- Pre-flight cost estimate: fixed/approximate token counts for calls we
# haven't made yet (see `estimate_overview_cost`'s docstring for the model).
_EXTRACTION_OUTPUT_TOKENS = 3_000
_MERGE_INPUT_TOKENS = 10_000
_MERGE_OUTPUT_TOKENS = 6_000
_CHAPTER_TAIL_TOKENS = 2_000
_CHAPTER_WORDS_TO_OUTPUT_TOKENS = 1.6
_COVERAGE_INPUT_TOKENS = 3_000
_COVERAGE_OUTPUT_TOKENS = 500
_CHAPTERS_PER_MINUTES = 5


def _slugify(text: str) -> str:
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")
    return slug or "notecast-episode"


def _report(progress: _ProgressFn | None, stage: str, fraction: float) -> None:
    if progress is not None:
        progress(stage, fraction)


class OverviewFailed(RuntimeError):
    """Raised when the audio pipeline fails after it has already spent some
    API budget, so the caller can tell the user how much was spent and at
    which stage it stopped.
    """

    def __init__(self, stage: str, est_cost_usd: float, message: str) -> None:
        self.stage = stage
        self.est_cost_usd = est_cost_usd
        super().__init__(message)


class OverviewEstimate(BaseModel):
    """A pre-flight cost estimate for `generate_overview`, computed without
    calling Claude. See `estimate_overview_cost` for the model behind it.
    """

    total_material_tokens: int
    n_sources: int
    n_chapters_low: int
    n_chapters_high: int
    est_cost_usd: float
    est_cost_low_usd: float
    est_cost_high_usd: float


class RenderSavedResult(BaseModel):
    """The result of `render_saved_script`: the rendered audio, plus any
    chunk ids the script cites that are no longer in the current index
    (e.g. the notebook was re-ingested and those chunks changed or vanished).
    """

    model_config = {"arbitrary_types_allowed": True}

    render: RenderResult
    stale_chunk_ids: list[str] = Field(default_factory=list)


class OverviewResult(BaseModel):
    """The result of one `generate_overview` run."""

    model_config = {"arbitrary_types_allowed": True}

    script: AudioScript
    render: RenderResult | None = None
    script_json_path: Path
    est_cost_usd: float = 0.0
    timings: dict[str, float] = Field(default_factory=dict)


def _script_json_path(nb: Notebook, script: AudioScript) -> Path:
    slug = _slugify(script.title)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    return nb.audio_dir / f"{slug}-{timestamp}.script.json"


def _save_script(nb: Notebook, script: AudioScript) -> Path:
    nb.audio_dir.mkdir(parents=True, exist_ok=True)
    path = _script_json_path(nb, script)
    path.write_text(script.model_dump_json(indent=2), encoding="utf-8")
    return path


def _load_script(path: Path) -> AudioScript:
    return AudioScript.model_validate_json(path.read_text(encoding="utf-8"))


def _chapter_count_for_minutes(minutes: float) -> int:
    """Quick heuristic for how many chapter calls a run of this length would
    make: about one chapter per `_CHAPTERS_PER_MINUTES` minutes, clamped to
    the planner's own [MIN_CHAPTERS, MAX_CHAPTERS] range.
    """
    return max(MIN_CHAPTERS, min(MAX_CHAPTERS, round(minutes / _CHAPTERS_PER_MINUTES) or 1))


def _chapter_calls_cost(minutes: float, material_tokens: int, settings: Settings) -> float:
    """Estimated cost of the chapter-writing calls for a `minutes`-long
    episode: the first chapter call writes `material_tokens` to the prompt
    cache (1.25x the input price), later chapters read it back (0.10x),
    each call also carries a small uncached tail and an output proportional
    to that chapter's target word count.
    """
    n_chapters = _chapter_count_for_minutes(minutes)
    target_words = minutes * settings.audio_words_per_minute
    words_per_chapter = target_words / n_chapters
    output_tokens = words_per_chapter * _CHAPTER_WORDS_TO_OUTPUT_TOKENS

    cost = 0.0
    for i in range(n_chapters):
        usage = {"input_tokens": _CHAPTER_TAIL_TOKENS, "output_tokens": output_tokens}
        if i == 0:
            usage["cache_write_tokens"] = material_tokens
        else:
            usage["cache_read_tokens"] = material_tokens
        cost += estimate_cost(settings.script_model, usage) or 0.0
    return cost


def estimate_overview_cost(
    nb: Notebook, scope: AudioScope, settings: Settings | None = None
) -> OverviewEstimate:
    """A pre-flight, no-API-calls estimate of what `generate_overview(scope)`
    would cost, so the CLI can show it and ask for confirmation before
    spending anything.

    The episode length isn't known until after key-point extraction and
    ranking, so this estimates the cheapest end of the pipeline (the
    notebook's `audio_min_minutes`) and the priciest end
    (`audio_max_minutes`) and reports both as a range; `est_cost_usd` is
    their midpoint. Everything else (extraction, merge, coverage calls) is
    priced with fixed/approximate token counts -- see the module-level
    `_EXTRACTION_OUTPUT_TOKENS` etc. constants -- since the real amounts
    depend on the model's own output, which we can't know in advance.

    Raises `ValueError` if `scope` selects no material, and
    `IndexNotBuiltError` if the notebook hasn't been indexed yet.
    """
    settings = settings or get_settings()
    retriever = get_retriever(nb, settings=settings)
    all_chunks: list[Chunk] = retriever.store.all_chunks()
    selected = select_chunks(all_chunks, scope)
    if not selected:
        raise ValueError(f"No material in {scope.label()} — check the week numbers or run ingest")

    grouped = group_by_source(selected)
    total_tokens = 0
    extraction_cost = 0.0
    for chunks in grouped.values():
        source_tokens = sum(estimate_tokens(c.embed_text) for c in chunks)
        total_tokens += source_tokens
        n_parts = max(1, math.ceil(source_tokens / MAX_SOURCE_TOKENS))
        for _ in range(n_parts):
            extraction_cost += (
                estimate_cost(
                    settings.helper_model,
                    {
                        "input_tokens": source_tokens / n_parts,
                        "output_tokens": _EXTRACTION_OUTPUT_TOKENS,
                    },
                )
                or 0.0
            )

    merge_cost = (
        estimate_cost(
            settings.script_model,
            {"input_tokens": _MERGE_INPUT_TOKENS, "output_tokens": _MERGE_OUTPUT_TOKENS},
        )
        or 0.0
    )

    coverage_cost_one = (
        estimate_cost(
            settings.helper_model,
            {"input_tokens": _COVERAGE_INPUT_TOKENS, "output_tokens": _COVERAGE_OUTPUT_TOKENS},
        )
        or 0.0
    )

    n_chapters_low = _chapter_count_for_minutes(float(settings.audio_min_minutes))
    n_chapters_high = _chapter_count_for_minutes(float(settings.audio_max_minutes))
    chapters_cost_low = _chapter_calls_cost(
        float(settings.audio_min_minutes), total_tokens, settings
    )
    chapters_cost_high = _chapter_calls_cost(
        float(settings.audio_max_minutes), total_tokens, settings
    )

    cost_low = extraction_cost + merge_cost + chapters_cost_low + coverage_cost_one
    cost_high = extraction_cost + merge_cost + chapters_cost_high + 2 * coverage_cost_one

    return OverviewEstimate(
        total_material_tokens=total_tokens,
        n_sources=len(grouped),
        n_chapters_low=n_chapters_low,
        n_chapters_high=n_chapters_high,
        est_cost_usd=(cost_low + cost_high) / 2,
        est_cost_low_usd=cost_low,
        est_cost_high_usd=cost_high,
    )


def generate_overview(
    nb: Notebook,
    scope: AudioScope,
    *,
    settings: Settings | None = None,
    client: anthropic.Anthropic | None = None,
    tts: KokoroTTS | None = None,
    render: bool = True,
    progress: _ProgressFn | None = None,
) -> OverviewResult:
    """Run the full audio-overview pipeline for `scope` and (by default)
    render it to an MP3. Raises `ValueError` if `scope` selects no material,
    and `IndexNotBuiltError` (from `notecast.index.service`) if the
    notebook hasn't been indexed yet.
    """
    settings = settings or get_settings()
    client = client or get_client(settings)

    timings: dict[str, float] = {}

    def _timed(stage: str, fraction: float, start: float) -> None:
        timings[stage] = time.monotonic() - start
        _report(progress, stage, fraction)

    t = time.monotonic()
    _report(progress, "loading", 0.0)
    retriever = get_retriever(nb, settings=settings)
    all_chunks: list[Chunk] = retriever.store.all_chunks()
    _timed("loading", 0.05, t)

    t = time.monotonic()
    selected = select_chunks(all_chunks, scope)
    if not selected:
        raise ValueError(f"No material in {scope.label()} — check the week numbers or run ingest")
    _timed("selecting", 0.1, t)

    t = time.monotonic()
    kp_usage = KeyPointRun()
    grouped = group_by_source(selected)
    points_by_source = {}
    sources = list(grouped.items())
    try:
        for i, (source_path, chunks) in enumerate(sources):
            points_by_source[source_path] = extract_key_points(
                source_path,
                chunks,
                client=client,
                settings=settings,
                focus=scope.focus,
                usage=kp_usage,
            )
            _report(progress, "key_points", 0.1 + 0.3 * (i + 1) / max(len(sources), 1))
    except (ChatError, ValueError) as exc:
        raise OverviewFailed("key_points", kp_usage.est_cost_usd, str(exc)) from exc
    timings["key_points"] = time.monotonic() - t

    t = time.monotonic()
    try:
        ranked = merge_and_rank(
            points_by_source, client=client, settings=settings, focus=scope.focus, usage=kp_usage
        )
    except (ChatError, ValueError) as exc:
        raise OverviewFailed("merging", kp_usage.est_cost_usd, str(exc)) from exc
    _timed("merging", 0.45, t)

    t = time.monotonic()
    try:
        plan = plan_audio(ranked, scope, settings, selected)
    except (ChatError, ValueError) as exc:
        raise OverviewFailed("planning", kp_usage.est_cost_usd, str(exc)) from exc
    _timed("planning", 0.5, t)

    t = time.monotonic()
    course_name = display_name(nb, load_course_config(nb))
    try:
        script = write_script(
            plan, selected, client=client, settings=settings, course_name=course_name
        )
    except (ChatError, ValueError) as exc:
        spent = kp_usage.est_cost_usd + getattr(exc, "partial_est_cost_usd", 0.0)
        raise OverviewFailed("scripting", spent, str(exc)) from exc
    _timed("scripting", 0.8, t)

    t = time.monotonic()
    script_json_path = _save_script(nb, script)
    _timed("saving_script", 0.82, t)

    est_cost_usd = kp_usage.est_cost_usd + (script.est_cost_usd or 0.0)

    render_result: RenderResult | None = None
    if render:
        t = time.monotonic()
        render_result = _render(nb, script, selected, settings=settings, tts=tts, progress=progress)
        timings["rendering"] = time.monotonic() - t

    _report(progress, "done", 1.0)

    return OverviewResult(
        script=script,
        render=render_result,
        script_json_path=script_json_path,
        est_cost_usd=est_cost_usd,
        timings=timings,
    )


def _render(
    nb: Notebook,
    script: AudioScript,
    chunks: list[Chunk],
    *,
    settings: Settings,
    tts: KokoroTTS | None,
    progress: _ProgressFn | None,
) -> RenderResult:
    chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
    transcript_markdown = script_to_transcript_markdown(script, chunk_lookup)

    def _render_progress(done: int, total: int) -> None:
        fraction = 0.82 + 0.18 * (done / total if total else 1.0)
        _report(progress, "rendering", fraction)

    return render_script(
        script,
        nb.audio_dir,
        settings=settings,
        tts=tts,
        transcript_markdown=transcript_markdown,
        progress=_render_progress,
    )


def _cited_chunk_ids(script: AudioScript) -> set[str]:
    return {
        chunk_id
        for chapter in script.chapters
        for line in chapter.lines
        for chunk_id in line.source_chunk_ids
    }


def render_saved_script(
    nb: Notebook,
    script_json_path: Path,
    *,
    settings: Settings | None = None,
    tts: KokoroTTS | None = None,
    progress: _ProgressFn | None = None,
) -> RenderSavedResult:
    """Re-render a script that was already saved to disk by an earlier
    `generate_overview` run, without calling Claude again.

    If the notebook was re-ingested since the script was written, some of
    the chunk ids it cites may no longer be in the index; those are
    reported back as `stale_chunk_ids` rather than failing the render (the
    script text itself still renders fine -- only its "Sources:" lines lose
    those citations).
    """
    settings = settings or get_settings()
    script = _load_script(script_json_path)

    retriever = get_retriever(nb, settings=settings)
    all_chunks: list[Chunk] = retriever.store.all_chunks()
    selected = select_chunks(all_chunks, script.plan.scope)

    available_ids = {chunk.chunk_id for chunk in selected}
    stale_chunk_ids = sorted(_cited_chunk_ids(script) - available_ids)

    render_result = _render(nb, script, selected, settings=settings, tts=tts, progress=progress)
    return RenderSavedResult(render=render_result, stale_chunk_ids=stale_chunk_ids)


__all__ = [
    "OverviewEstimate",
    "OverviewFailed",
    "OverviewResult",
    "RenderSavedResult",
    "estimate_overview_cost",
    "generate_overview",
    "render_saved_script",
]
