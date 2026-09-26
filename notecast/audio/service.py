"""Wires the audio-overview pipeline (key points -> plan -> script -> speech)
into one entry point per notebook: `generate_overview` runs the whole thing
from a scope, and `render_saved_script` re-renders a previously-saved script
without paying for the Claude calls again.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field

from notecast.audio.keypoints import (
    KeyPointRun,
    extract_key_points,
    group_by_source,
    merge_and_rank,
    select_chunks,
)
from notecast.audio.models import AudioScope, AudioScript, RenderResult
from notecast.audio.planner import plan_audio
from notecast.audio.render import render_script
from notecast.audio.scriptwriter import script_to_transcript_markdown, write_script
from notecast.audio.tts import KokoroTTS
from notecast.chat.client import get_client
from notecast.config import Settings, get_settings
from notecast.index.service import get_retriever
from notecast.ingest.course import display_name, load_course_config
from notecast.models import Chunk
from notecast.notebook import Notebook

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_ProgressFn = Callable[[str, float], None]


def _slugify(text: str) -> str:
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")
    return slug or "notecast-episode"


def _report(progress: _ProgressFn | None, stage: str, fraction: float) -> None:
    if progress is not None:
        progress(stage, fraction)


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
    return nb.audio_dir / f"{slug}.script.json"


def _save_script(nb: Notebook, script: AudioScript) -> Path:
    nb.audio_dir.mkdir(parents=True, exist_ok=True)
    path = _script_json_path(nb, script)
    path.write_text(script.model_dump_json(indent=2), encoding="utf-8")
    return path


def _load_script(path: Path) -> AudioScript:
    return AudioScript.model_validate_json(path.read_text(encoding="utf-8"))


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
    timings["key_points"] = time.monotonic() - t

    t = time.monotonic()
    ranked = merge_and_rank(
        points_by_source, client=client, settings=settings, focus=scope.focus, usage=kp_usage
    )
    _timed("merging", 0.45, t)

    t = time.monotonic()
    plan = plan_audio(ranked, scope, settings, selected)
    _timed("planning", 0.5, t)

    t = time.monotonic()
    course_name = display_name(nb, load_course_config(nb))
    script = write_script(plan, selected, client=client, settings=settings, course_name=course_name)
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


def render_saved_script(
    nb: Notebook,
    script_json_path: Path,
    *,
    settings: Settings | None = None,
    tts: KokoroTTS | None = None,
    progress: _ProgressFn | None = None,
) -> RenderResult:
    """Re-render a script that was already saved to disk by an earlier
    `generate_overview` run, without calling Claude again.
    """
    settings = settings or get_settings()
    script = _load_script(script_json_path)

    retriever = get_retriever(nb, settings=settings)
    all_chunks: list[Chunk] = retriever.store.all_chunks()
    selected = select_chunks(all_chunks, script.plan.scope)

    return _render(nb, script, selected, settings=settings, tts=tts, progress=progress)


__all__ = ["OverviewResult", "generate_overview", "render_saved_script"]
