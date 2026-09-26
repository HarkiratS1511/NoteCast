"""The two-host audio script writer, chapter by chapter, with a
coverage-first check: verify every tier A key point actually gets explained,
and regenerate the specific chapter(s) that missed one.

The source material (all in-scope chunks, rendered once, in a deterministic
course order) is sent as the first user content block on every chapter call,
with a `cache_control` breakpoint on it, so the ~16-80k-token material is
written to the prompt cache once and read cheaply on every later chapter
call — same model, same bytes, every time (see `prompts.build_material_block`).
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import anthropic

from notecast.audio.models import (
    AudioPlan,
    AudioScript,
    ChapterPlan,
    ChapterScript,
    CoverageReport,
    RankedKeyPoint,
    ScriptLine,
)
from notecast.audio.prompts import (
    CHAPTER_SCRIPT_SCHEMA,
    COVERAGE_CHECK_SYSTEM_PROMPT,
    COVERAGE_SCHEMA,
    SCRIPTWRITER_SYSTEM_PROMPT,
    build_material_block,
)
from notecast.chat.client import ChatError, map_anthropic_error
from notecast.chat.pricing import estimate_cost
from notecast.config import Settings
from notecast.models import Chunk

_CHAPTER_MIN_MAX_TOKENS = 4000
_CHAPTER_MAX_TOKENS_CAP = 32000
_COVERAGE_MAX_TOKENS = 2000
_COVERAGE_MAX_TOKENS_CAP = 8000
_SPEAKERS = {"A", "B"}
_RECAP_KEEP_FULL = 3
_MAX_ATTEMPTS = 2


def write_script(
    plan: AudioPlan,
    chunks: list[Chunk],
    *,
    client: anthropic.Anthropic,
    settings: Settings,
    course_name: str = "",
) -> AudioScript:
    """Write a two-host `AudioScript` for `plan`, chapter by chapter, then
    run a coverage check and patch any chapter that missed a tier A point.
    """
    if not chunks:
        raise ValueError("Nothing in scope to talk about")

    chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
    valid_chunk_ids = set(chunk_lookup)
    material_content = {
        "type": "text",
        "text": build_material_block(chunks),
        "cache_control": {"type": "ephemeral"},
    }
    system_blocks = [
        {
            "type": "text",
            "text": SCRIPTWRITER_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    points_by_id = {point.id: point for point in plan.points}
    sorted_chapter_plans = sorted(plan.chapters, key=lambda chapter: chapter.index)

    usage_by_model: dict[str, dict[str, int]] = defaultdict(_new_usage_totals)

    chapter_titles = [chapter_plan.title for chapter_plan in sorted_chapter_plans]

    chapters: list[ChapterScript] = []
    recaps: list[str] = []
    for i, chapter_plan in enumerate(sorted_chapter_plans):
        chapter_points = _points_for(chapter_plan, points_by_id)
        running_recap = _build_running_recap(recaps, chapter_titles[:i])
        chapter_script, recap, usage = _generate_chapter(
            client=client,
            settings=settings,
            system_blocks=system_blocks,
            material_content=material_content,
            chapter_plan=chapter_plan,
            points=chapter_points,
            running_recap=running_recap,
            is_first=(i == 0),
            is_last=(i == len(sorted_chapter_plans) - 1),
            valid_chunk_ids=valid_chunk_ids,
        )
        _accumulate_usage(usage_by_model[settings.script_model], usage)
        chapters.append(chapter_script)
        recaps.append(recap)

    tier_a_points = [point for point in plan.points if point.tier == "A"]
    tier_a_ids = {point.id for point in tier_a_points}

    coverage_data, cov_usage = _check_coverage(client, settings, tier_a_points, chapters)
    _accumulate_usage(usage_by_model[settings.helper_model], cov_usage)
    covered = set(coverage_data.get("covered_ids", [])) & tier_a_ids
    # Anything not explicitly confirmed covered counts as missing — don't
    # trust the model's own `missing_ids` list, which can be inconsistent
    # with what it actually put in `covered_ids`.
    missing = tier_a_ids - covered

    patched: list[str] = []
    notes: list[str] = []

    if missing:
        chapters_to_fix = _chapters_owning(sorted_chapter_plans, missing)
        if not chapters_to_fix:
            notes.append(
                "Coverage check flagged points that no chapter plan owns, so they "
                f"could not be patched by regenerating a chapter: {', '.join(sorted(missing))}"
            )
        else:
            for idx, missing_ids in chapters_to_fix.items():
                chapter_plan = sorted_chapter_plans[idx]
                chapter_points = _points_for(chapter_plan, points_by_id)
                missed_points = [points_by_id[pid] for pid in missing_ids if pid in points_by_id]
                running_recap = _build_running_recap(recaps[:idx], chapter_titles[:idx])
                chapter_script, recap, usage = _generate_chapter(
                    client=client,
                    settings=settings,
                    system_blocks=system_blocks,
                    material_content=material_content,
                    chapter_plan=chapter_plan,
                    points=chapter_points,
                    running_recap=running_recap,
                    is_first=(idx == 0),
                    is_last=(idx == len(sorted_chapter_plans) - 1),
                    valid_chunk_ids=valid_chunk_ids,
                    missed_points=missed_points,
                )
                _accumulate_usage(usage_by_model[settings.script_model], usage)
                chapters[idx] = chapter_script
                recaps[idx] = recap
                notes.append(
                    f"Regenerated chapter {chapter_plan.index} ({chapter_plan.title}) "
                    f"to cover: {', '.join(sorted(missing_ids))}"
                )

            coverage_data2, cov_usage2 = _check_coverage(client, settings, tier_a_points, chapters)
            _accumulate_usage(usage_by_model[settings.helper_model], cov_usage2)
            covered2 = set(coverage_data2.get("covered_ids", [])) & tier_a_ids
            missing2 = tier_a_ids - covered2
            patched = sorted(missing - missing2)
            covered = covered2
            missing = missing2

    coverage = CoverageReport(
        covered=sorted(covered),
        missing=sorted(missing),
        patched=patched,
        notes=notes,
    )

    est_cost_usd = _estimate_total_cost(usage_by_model)

    title = f"{course_name} — {plan.scope.label()}" if course_name else plan.scope.label()

    return AudioScript(
        title=title,
        plan=plan,
        chapters=chapters,
        coverage=coverage,
        est_cost_usd=est_cost_usd,
    )


def script_to_transcript_markdown(script: AudioScript, chunk_lookup: dict[str, Chunk]) -> str:
    """Render `script` as a markdown transcript: title, then per chapter a
    heading, "**Host A:** ..." lines, and a compact "Sources:" list of the
    unique chunk headers that chapter drew on. No timings — the renderer
    (Phase 6 track: tts/render) adds those once the audio exists.
    """
    lines: list[str] = [f"# {script.title}", ""]
    for chapter in script.chapters:
        lines.append(f"## Chapter {chapter.index}: {chapter.title}")
        lines.append("")
        seen_chunk_ids: list[str] = []
        for line in chapter.lines:
            speaker_label = "Host A" if line.speaker == "A" else "Host B"
            lines.append(f"**{speaker_label}:** {line.text}")
            for chunk_id in line.source_chunk_ids:
                if chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.append(chunk_id)
        lines.append("")
        headers: list[str] = []
        for chunk_id in seen_chunk_ids:
            chunk = chunk_lookup.get(chunk_id)
            if chunk is None:
                continue
            header = chunk.header or chunk.source_path
            if header not in headers:
                headers.append(header)
        if headers:
            lines.append("Sources: " + ", ".join(headers))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# --- Chapter generation ----------------------------------------------------


def _points_for(
    chapter_plan: ChapterPlan, points_by_id: dict[str, RankedKeyPoint]
) -> list[RankedKeyPoint]:
    return [points_by_id[pid] for pid in chapter_plan.point_ids if pid in points_by_id]


def _build_running_recap(recaps: list[str], titles: list[str]) -> str:
    """The "running recap" sent to a chapter call: the last `_RECAP_KEEP_FULL`
    chapter recaps in full, plus — once there are more than that — a single
    line naming the earlier chapters by title, so the recap doesn't grow
    without bound over a long episode.
    """
    if not recaps:
        return ""
    if len(recaps) <= _RECAP_KEEP_FULL:
        return "\n".join(recaps)
    earlier_titles = titles[: len(recaps) - _RECAP_KEEP_FULL]
    recent = recaps[-_RECAP_KEEP_FULL:]
    parts = [f"Earlier chapters already covered (titles only): {', '.join(earlier_titles)}."]
    parts.extend(recent)
    return "\n\n".join(parts)


def _chapters_owning(
    sorted_chapter_plans: list[ChapterPlan], missing: set[str]
) -> dict[int, list[str]]:
    """Map chapter index (into `sorted_chapter_plans`) -> the missing point
    ids that chapter was planned to hold.
    """
    owned: dict[int, list[str]] = {}
    for idx, chapter_plan in enumerate(sorted_chapter_plans):
        ids = [pid for pid in chapter_plan.point_ids if pid in missing]
        if ids:
            owned[idx] = ids
    return owned


def _chapter_max_tokens(target_words: int) -> int:
    return min(max(_CHAPTER_MIN_MAX_TOKENS, target_words * 3), _CHAPTER_MAX_TOKENS_CAP)


def _stream_chapter(
    client: anthropic.Anthropic,
    settings: Settings,
    system_blocks: list[dict],
    material_content: dict,
    tail_text: str,
    max_tokens: int,
) -> Any:
    messages = [
        {
            "role": "user",
            "content": [material_content, {"type": "text", "text": tail_text}],
        }
    ]
    kwargs: dict[str, Any] = {
        "model": settings.script_model,
        "max_tokens": max_tokens,
        "system": system_blocks,
        "messages": messages,
        "output_config": {
            "effort": "medium",
            "format": {"type": "json_schema", "schema": CHAPTER_SCRIPT_SCHEMA},
        },
    }
    try:
        with client.messages.stream(**kwargs) as stream:
            return stream.get_final_message()
    except anthropic.AnthropicError as exc:
        raise map_anthropic_error(exc) from exc


def _generate_chapter(
    *,
    client: anthropic.Anthropic,
    settings: Settings,
    system_blocks: list[dict],
    material_content: dict,
    chapter_plan: ChapterPlan,
    points: list[RankedKeyPoint],
    running_recap: str,
    is_first: bool,
    is_last: bool,
    valid_chunk_ids: set[str],
    missed_points: list[RankedKeyPoint] | None = None,
) -> tuple[ChapterScript, str, dict]:
    tail_text = _chapter_tail_text(
        chapter_plan, points, running_recap, is_first, is_last, missed_points
    )
    max_tokens = _chapter_max_tokens(chapter_plan.target_words)
    usage_totals = _new_usage_totals()
    attempt_tail = tail_text

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        message = _stream_chapter(
            client, settings, system_blocks, material_content, attempt_tail, max_tokens
        )
        _accumulate_usage(usage_totals, _extract_usage(message))
        stop_reason = _get(message, "stop_reason")

        if stop_reason == "refusal":
            raise ChatError(_chapter_refusal_message(message, chapter_plan))

        if stop_reason == "max_tokens":
            if attempt < _MAX_ATTEMPTS:
                max_tokens = min(max_tokens * 2, _CHAPTER_MAX_TOKENS_CAP)
                attempt_tail = tail_text + "\n\n" + _CONCISE_RETRY_INSTRUCTION
                continue
            raise ChatError(
                f"Chapter {chapter_plan.index} ('{chapter_plan.title}') kept exceeding the "
                "length limit even after retrying with more room."
            )

        try:
            data = _first_json_object(message)
            lines = _validate_lines(data.get("lines", []) or [], valid_chunk_ids)
            chapter_script = ChapterScript(
                index=chapter_plan.index, title=chapter_plan.title, lines=lines
            )
            recap = str(data.get("recap", "") or "").strip()
            return chapter_script, recap, usage_totals
        except (json.JSONDecodeError, ChatError) as exc:
            raise ChatError(
                f"Could not get a usable script from Claude for chapter {chapter_plan.index} "
                f"('{chapter_plan.title}'): {exc}"
            ) from exc

    # Ran out of attempts because the model kept hitting the length limit.
    raise ChatError(
        f"Chapter {chapter_plan.index} ('{chapter_plan.title}') kept exceeding the length "
        "limit even after retrying with more room."
    )


_CONCISE_RETRY_INSTRUCTION = (
    "IMPORTANT: your previous attempt at this chapter was cut off for exceeding the length "
    "limit. Be noticeably more concise while still covering every point above."
)


def _chapter_refusal_message(message: Any, chapter_plan: ChapterPlan) -> str:
    stop_details = _get(message, "stop_details")
    category = _get(stop_details, "category") if stop_details else None
    label = f"chapter {chapter_plan.index} ('{chapter_plan.title}')"
    if category:
        return f"Claude declined to write {label} ({category})."
    return f"Claude declined to write {label}."


def _format_points(points: list[RankedKeyPoint]) -> str:
    if not points:
        return "(none)"
    rendered = []
    for point in points:
        evidence = "; ".join(point.evidence) if point.evidence else "none"
        chunk_ids = ", ".join(point.source_chunk_ids) if point.source_chunk_ids else "none"
        rendered.append(
            f"- [{point.id}] ({point.tier}) {point.title}: {point.summary} "
            f"(evidence: {evidence}; chunks: {chunk_ids})"
        )
    return "\n".join(rendered)


def _chapter_tail_text(
    chapter_plan: ChapterPlan,
    points: list[RankedKeyPoint],
    running_recap: str,
    is_first: bool,
    is_last: bool,
    missed_points: list[RankedKeyPoint] | None,
) -> str:
    parts = [
        f"Chapter {chapter_plan.index}: {chapter_plan.title}",
        f"Target length: about {chapter_plan.target_words} words.",
        "Key points to cover in this chapter:",
        _format_points(points),
    ]
    if running_recap:
        parts.append(
            "Recap of chapters covered so far, for continuity — don't repeat this material, "
            "refer back to it naturally where relevant:"
        )
        parts.append(running_recap)
    else:
        parts.append("This is the first chapter — there is no prior recap yet.")
    if is_first:
        parts.append(
            "This is the opening chapter of the episode: start with a short, natural "
            "introduction where the hosts welcome listeners and preview what the episode "
            "covers, before getting into the material."
        )
    if is_last:
        parts.append(
            "This is the final chapter of the episode: after covering this chapter's points, "
            "end with a short wrap-up/summary of the whole episode."
        )
    if missed_points:
        parts.append(
            "IMPORTANT: a coverage check found that the previous version of this chapter did "
            "not adequately explain the following points. Make sure each one is now explained "
            "in real depth, not just mentioned in passing:"
        )
        parts.append(_format_points(missed_points))
    return "\n\n".join(parts)


# --- Coverage check ----------------------------------------------------


def _script_to_plain_text(chapters: list[ChapterScript]) -> str:
    lines: list[str] = []
    for chapter in chapters:
        lines.append(f"Chapter {chapter.index}: {chapter.title}")
        for line in chapter.lines:
            lines.append(f"Host {line.speaker}: {line.text}")
        lines.append("")
    return "\n".join(lines)


def _check_coverage(
    client: anthropic.Anthropic,
    settings: Settings,
    tier_a_points: list[RankedKeyPoint],
    chapters: list[ChapterScript],
) -> tuple[dict, dict]:
    if not tier_a_points:
        return {"covered_ids": []}, _new_usage_totals()

    points_text = "\n".join(
        f"- [{point.id}] {point.title}: {point.summary}" for point in tier_a_points
    )
    script_text = _script_to_plain_text(chapters)
    user_text = (
        "Tier A key points that must be adequately explained (not merely mentioned):\n"
        f"{points_text}\n\n"
        "Full script:\n"
        f"{script_text}"
    )
    usage_totals = _new_usage_totals()
    max_tokens = _COVERAGE_MAX_TOKENS

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        kwargs: dict[str, Any] = {
            "model": settings.helper_model,
            "max_tokens": max_tokens,
            "system": COVERAGE_CHECK_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_text}],
            "output_config": {"format": {"type": "json_schema", "schema": COVERAGE_SCHEMA}},
        }
        try:
            message = client.messages.create(**kwargs)
        except anthropic.AnthropicError as exc:
            raise map_anthropic_error(exc) from exc

        _accumulate_usage(usage_totals, _extract_usage(message))
        stop_reason = _get(message, "stop_reason")

        if stop_reason == "refusal":
            raise ChatError("Claude declined to run the coverage check on the script.")

        if stop_reason == "max_tokens":
            if attempt < _MAX_ATTEMPTS:
                max_tokens = min(max_tokens * 2, _COVERAGE_MAX_TOKENS_CAP)
                continue
            raise ChatError(
                "The coverage check kept exceeding the length limit even after retrying."
            )

        try:
            data = _first_json_object(message)
            return data, usage_totals
        except (json.JSONDecodeError, ChatError) as exc:
            raise ChatError(
                f"Could not parse the coverage check result from Claude: {exc}"
            ) from exc

    raise ChatError("The coverage check kept exceeding the length limit even after retrying.")


# --- Line validation ----------------------------------------------------

_MD_CHARS = "*#`_"


def _clean_text(text: str) -> str:
    """Strip markdown emphasis/heading/bullet symbols that can't be spoken,
    and collapse whitespace to a single line.
    """
    cleaned_lines = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        while stripped[:1] in ("-", "•", "*", "•"):
            stripped = stripped[1:].strip()
        cleaned_lines.append(stripped)
    joined = " ".join(part for part in cleaned_lines if part)
    for char in _MD_CHARS:
        joined = joined.replace(char, "")
    return " ".join(joined.split())


def _validate_lines(raw_lines: list[dict], valid_chunk_ids: set[str]) -> list[ScriptLine]:
    lines: list[ScriptLine] = []
    for raw in raw_lines:
        speaker = raw.get("speaker")
        if speaker not in _SPEAKERS:
            continue
        text = _clean_text(str(raw.get("text", "") or ""))
        if not text:
            continue
        source_ids = [
            chunk_id
            for chunk_id in raw.get("source_chunk_ids", []) or []
            if chunk_id in valid_chunk_ids
        ]
        lines.append(ScriptLine(speaker=speaker, text=text, source_chunk_ids=source_ids))
    return lines


# --- Usage / cost / JSON helpers ----------------------------------------


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _new_usage_totals() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }


def _extract_usage(message: Any) -> dict[str, int]:
    usage = _get(message, "usage")
    totals = _new_usage_totals()
    if usage is None:
        return totals
    totals["input_tokens"] = _get(usage, "input_tokens", 0) or 0
    totals["output_tokens"] = _get(usage, "output_tokens", 0) or 0
    totals["cache_read_tokens"] = _get(usage, "cache_read_input_tokens", 0) or 0
    totals["cache_write_tokens"] = _get(usage, "cache_creation_input_tokens", 0) or 0
    return totals


def _accumulate_usage(totals: dict[str, int], addition: dict[str, int]) -> None:
    for key in totals:
        totals[key] += addition.get(key, 0)


def _estimate_total_cost(usage_by_model: dict[str, dict[str, int]]) -> float | None:
    total = 0.0
    have_cost = False
    for model, usage in usage_by_model.items():
        cost = estimate_cost(model, usage)
        if cost is not None:
            have_cost = True
            total += cost
    return total if have_cost else None


def _first_json_object(message: Any) -> dict:
    for block in _get(message, "content", []) or []:
        if _get(block, "type") == "text":
            text = _get(block, "text", "") or ""
            if text:
                return json.loads(text)
    raise ChatError("Claude did not return the expected structured JSON output.")
