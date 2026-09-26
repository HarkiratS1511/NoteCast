"""Key-point extraction and ranking for audio overviews.

Pipeline (see docs/PLAN.md §5):

1. `select_chunks` / `group_by_source` narrow a course's chunks down to the
   requested `AudioScope` and group them by source file.
2. `extract_key_points` makes ONE `helper_model` (Haiku 4.5) call per source
   (or one call per *part*, for oversized sources) asking for every concept,
   definition, method, formula, worked example, caveat, misconception and
   admin item a student must know, each with an importance score and
   evidence. `in_slides` / `in_transcript` are computed locally from the
   chunk metadata, not asked of the model.
3. `merge_and_rank` makes ONE `script_model` (Sonnet 5, effort "medium")
   call across *all* extracted points (never the raw source text) to merge
   duplicates across sources and assign a tier (A/B/C) and rank.

Structured outputs: per the claude-api skill (`python/claude-api/tool-use.md`
-> Structured Outputs), `client.messages.parse()` with a pydantic
`output_format` is the recommended way to get validated JSON back, and the
skill's model catalogue does not single out Haiku 4.5 as unsupported. Being
unable to confirm per-model structured-output support against a live Models
API from here, we follow the brief's conservative fallback: `merge_and_rank`
(Sonnet 5, a model the skill explicitly documents `effort` and structured
outputs for) uses `messages.parse`, while `extract_key_points` (Haiku 4.5)
asks for JSON in the prompt and validates the response with pydantic,
retrying once on a parse/validation failure before raising.

Cost tracking: `extract_key_points` and `merge_and_rank` both take an
optional `usage: KeyPointRun` accumulator. Callers that want run-wide token
counts and estimated cost create one `KeyPointRun` and pass it to every
call; each call mutates it in place. This keeps the two functions' return
types exactly as specified (`list[KeyPoint]` / `list[RankedKeyPoint]`)
while still giving the planner/CLI a way to report spend.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import anthropic
from pydantic import BaseModel, Field, ValidationError

from notecast.audio.models import AudioScope, KeyPoint, KeyPointKind, RankedKeyPoint, Tier
from notecast.chat.client import ChatError, map_anthropic_error
from notecast.chat.pricing import estimate_cost
from notecast.config import Settings
from notecast.ingest.textutils import estimate_tokens
from notecast.models import Chunk, SourceType

logger = logging.getLogger(__name__)

# Sources whose text is roughly larger than this many tokens are split into
# parts before extraction, so a single Haiku call never has to swallow an
# entire textbook chapter in one prompt.
MAX_SOURCE_TOKENS = 60_000

# Non-streaming requests risk an SDK HTTP timeout above ~16K max_tokens (see
# the claude-api skill's model-migration notes) -- anything above this uses
# `client.messages.stream(...).get_final_message()` instead of `.create()`.
STREAMING_THRESHOLD = 16_000
# Headroom for a single-source extraction call and for the merge/rank call.
EXTRACTION_MAX_TOKENS = 16_000
MERGE_MAX_TOKENS = 16_000


class KeyPointRun(BaseModel):
    """Accumulates token usage and estimated cost across `extract_key_points`
    and `merge_and_rank` calls. See module docstring for how it's used.
    """

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    est_cost_usd: float = 0.0

    def add(self, model: str, usage: dict[str, Any]) -> None:
        self.calls += 1
        self.input_tokens += usage.get("input_tokens", 0) or 0
        self.output_tokens += usage.get("output_tokens", 0) or 0
        cost = estimate_cost(model, usage)
        if cost is not None:
            self.est_cost_usd += cost


def _usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_write_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
    }


def _invoke(
    *,
    client: anthropic.Anthropic,
    model: str,
    max_tokens: int,
    system: str,
    messages: list[dict[str, Any]],
    parse_model: type[BaseModel] | None = None,
    effort: str | None = None,
) -> Any:
    """Call Claude, streaming automatically when `max_tokens` exceeds
    `STREAMING_THRESHOLD`. When `parse_model` is given, `output_format` is
    passed through to either `messages.parse` (non-streaming) or
    `messages.stream` (streaming; `anthropic` 1.8.0's `.stream()` accepts
    `output_format=` directly and populates `parsed_output` on the final
    message the same way `.parse()` does) -- so the response's
    `parsed_output` is set either way, with no hand-built JSON schema and
    no manual JSON parsing needed.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }
    if parse_model is not None:
        kwargs["output_format"] = parse_model
    if effort:
        kwargs["output_config"] = {"effort": effort}

    if max_tokens > STREAMING_THRESHOLD:
        with client.messages.stream(**kwargs) as stream:
            return stream.get_final_message()
    return (
        client.messages.create(**kwargs) if parse_model is None else client.messages.parse(**kwargs)
    )


def _response_text(response: Any) -> str | None:
    content = getattr(response, "content", None) or []
    return next((b.text for b in content if getattr(b, "type", None) == "text"), None)


def _invoke_with_retries(
    *,
    client: anthropic.Anthropic,
    model: str,
    max_tokens: int,
    system: str,
    messages: list[dict[str, Any]],
    parse_model: type[BaseModel] | None = None,
    effort: str | None = None,
    usage: KeyPointRun | None = None,
) -> Any:
    """`_invoke`, plus explicit `stop_reason` handling: "refusal" raises a
    `ChatError` immediately; "max_tokens" logs a warning and retries once
    with doubled max_tokens (streaming if that now exceeds
    `STREAMING_THRESHOLD`); anything else is returned as-is. Every call
    (including the retry) is added to `usage` if given.
    """

    def call(tokens: int) -> Any:
        response = _invoke(
            client=client,
            model=model,
            max_tokens=tokens,
            system=system,
            messages=messages,
            parse_model=parse_model,
            effort=effort,
        )
        if usage is not None:
            usage.add(model, _usage_dict(response))
        return response

    response = call(max_tokens)
    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason == "refusal":
        logger.error("%s refused the request (stop_reason=refusal)", model)
        raise ChatError(f"Claude ({model}) refused this request.")
    if stop_reason == "max_tokens":
        doubled = max_tokens * 2
        logger.warning(
            "%s hit max_tokens=%d (stop_reason=max_tokens); retrying once with max_tokens=%d%s",
            model,
            max_tokens,
            doubled,
            " (streaming)" if doubled > STREAMING_THRESHOLD else "",
        )
        response = call(doubled)
        retried_stop = getattr(response, "stop_reason", None)
        if retried_stop == "refusal":
            logger.error("%s refused the retried request (stop_reason=refusal)", model)
            raise ChatError(f"Claude ({model}) refused this request.")
        if retried_stop == "max_tokens":
            logger.warning(
                "%s hit max_tokens again at %d after doubling; giving up on this attempt",
                model,
                doubled,
            )
        else:
            logger.debug(
                "%s recovered after doubling max_tokens (stop_reason=%s)", model, retried_stop
            )
    else:
        logger.debug("%s stop_reason=%s", model, stop_reason)
    return response


def select_chunks(chunks: list[Chunk], scope: AudioScope) -> list[Chunk]:
    """Filter `chunks` to the given scope (weeks AND source_paths, whichever
    are set), in a stable order of (source_path, ordinal).
    """
    selected = list(chunks)
    if scope.weeks:
        weeks = set(scope.weeks)
        selected = [c for c in selected if c.week in weeks]
    if scope.source_paths:
        paths = set(scope.source_paths)
        selected = [c for c in selected if c.source_path in paths]
    return sorted(selected, key=lambda c: (c.source_path, c.ordinal))


def group_by_source(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    """Group chunks by source_path, each group sorted by ordinal."""
    grouped: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.source_path, []).append(chunk)
    for group in grouped.values():
        group.sort(key=lambda c: c.ordinal)
    return grouped


def _chunk_flags(chunk: Chunk) -> tuple[bool, bool]:
    """(in_slides, in_transcript) for one chunk, from its location/type."""
    loc = chunk.location
    in_slides = loc.slide is not None or loc.page is not None
    in_transcript = (
        loc.approx_minute is not None
        or loc.t_start is not None
        or (
            chunk.source_type in (SourceType.VTT, SourceType.SRT, SourceType.TXT)
            and loc.speaker is not None
        )
    )
    return in_slides, in_transcript


def _passage_block(chunks: list[Chunk]) -> str:
    lines = []
    for chunk in chunks:
        header = chunk.header or chunk.location.label() or chunk.source_path
        lines.append(f"[{chunk.chunk_id}] {header}\n{chunk.text}")
    return "\n\n".join(lines)


def _split_into_parts(chunks: list[Chunk]) -> list[list[Chunk]]:
    """Split a source's chunks into parts, each under ~MAX_SOURCE_TOKENS."""
    parts: list[list[Chunk]] = []
    current: list[Chunk] = []
    current_tokens = 0
    for chunk in chunks:
        tokens = estimate_tokens(chunk.embed_text)
        if current and current_tokens + tokens > MAX_SOURCE_TOKENS:
            parts.append(current)
            current = []
            current_tokens = 0
        current.append(chunk)
        current_tokens += tokens
    if current:
        parts.append(current)
    return parts or [[]]


class _ExtractedPoint(BaseModel):
    title: str
    summary: str
    kind: KeyPointKind = "concept"
    importance: int = Field(3, ge=1, le=5)
    evidence: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)


class _ExtractionResult(BaseModel):
    key_points: list[_ExtractedPoint]


_EXTRACTION_SYSTEM = """You extract key points from lecture material (slides and/or \
transcripts) so a student can be taught them in an audio overview.

List every concept, definition, method, formula, worked example, caveat, \
misconception, and admin item (deadlines, assessment logistics) a student must \
know from the passages given. Each passage is numbered with a chunk id in \
square brackets, e.g. "[abc123] Slide 4\\n...text...".

For each key point, score importance 1-5 using these signals:
- Explicit emphasis: the lecturer says things like "this will be on the exam", \
"the key idea here is", "remember this".
- Time spent: transcript passages the lecturer dwells on for a while.
- Assessment links: mentions in the context of a quiz or assignment.
- Corrections: a lecturer correcting an error on a slide is a caveat worth \
including at high importance.
- Whether the point is a building block other later topics depend on.

Respond with ONLY a JSON object (no markdown fences, no commentary) matching \
this shape:
{"key_points": [{"title": str, "summary": str (1-3 sentences), \
"kind": one of "concept"|"definition"|"method"|"formula"|"example"|"caveat"|\
"misconception"|"admin", "importance": int 1-5, "evidence": [str, ...], \
"source_chunk_ids": [str, ...]}]}

`source_chunk_ids` MUST only contain chunk ids that appear in the passages \
given to you, in square brackets. Do not invent ids."""


def _extraction_user_prompt(chunks: list[Chunk], focus: str | None) -> str:
    focus_line = f"\nFocus for this overview: {focus}\n" if focus else ""
    return f"{focus_line}\nPassages:\n\n{_passage_block(chunks)}\n\nReturn the JSON object now."


def _parse_extraction_json(text: str) -> _ExtractionResult:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    data = json.loads(text)
    return _ExtractionResult.model_validate(data)


def _extract_part(
    chunks: list[Chunk],
    *,
    client: anthropic.Anthropic,
    settings: Settings,
    focus: str | None,
    usage: KeyPointRun | None,
) -> list[_ExtractedPoint]:
    valid_ids = {c.chunk_id for c in chunks}
    user_prompt = _extraction_user_prompt(chunks, focus)
    last_error: Exception | None = None
    for attempt in range(2):
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
        if attempt == 1 and last_error is not None:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That response was not valid JSON matching the required "
                        f"shape ({last_error}). Reply with ONLY the JSON object, "
                        "nothing else."
                    ),
                }
            )
        try:
            response = _invoke_with_retries(
                client=client,
                model=settings.helper_model,
                max_tokens=EXTRACTION_MAX_TOKENS,
                system=_EXTRACTION_SYSTEM,
                messages=messages,
                usage=usage,
            )
        except anthropic.AnthropicError as exc:
            raise map_anthropic_error(exc) from exc
        text = _response_text(response) or ""
        try:
            result = _parse_extraction_json(text)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            logger.warning("extraction JSON validation failed (attempt %d): %s", attempt + 1, exc)
            continue
        logger.debug("extraction succeeded on attempt %d", attempt + 1)
        for point in result.key_points:
            point.source_chunk_ids = [cid for cid in point.source_chunk_ids if cid in valid_ids]
        return result.key_points
    raise ValueError(f"Key point extraction returned invalid JSON twice: {last_error}")


def extract_key_points(
    source_path: str,
    chunks: list[Chunk],
    *,
    client: anthropic.Anthropic,
    settings: Settings,
    focus: str | None = None,
    usage: KeyPointRun | None = None,
) -> list[KeyPoint]:
    """Extract key points for one source. One helper_model call per source,
    or one call per part for sources over ~MAX_SOURCE_TOKENS.
    """
    if not chunks:
        return []
    chunks_by_id = {c.chunk_id: c for c in chunks}
    parts = _split_into_parts(chunks)

    # Derived from the full source_path (not just the basename) so that
    # e.g. week-01/slides.pdf and week-02/slides.pdf never collide.
    source_slug = hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:8]

    points: list[KeyPoint] = []
    counter = 0
    for part in parts:
        if not part:
            continue
        extracted = _extract_part(part, client=client, settings=settings, focus=focus, usage=usage)
        for item in extracted:
            counter += 1
            source_chunks = [chunks_by_id[cid] for cid in item.source_chunk_ids]
            in_slides = any(_chunk_flags(c)[0] for c in source_chunks)
            in_transcript = any(_chunk_flags(c)[1] for c in source_chunks)
            points.append(
                KeyPoint(
                    id=f"kp-{source_slug}-{counter}",
                    title=item.title,
                    summary=item.summary,
                    kind=item.kind,
                    importance=item.importance,
                    evidence=item.evidence,
                    source_chunk_ids=item.source_chunk_ids,
                    in_slides=in_slides,
                    in_transcript=in_transcript,
                )
            )
    return points


class _MergedPoint(BaseModel):
    title: str
    summary: str
    kind: KeyPointKind = "concept"
    importance: int = Field(3, ge=1, le=5)
    evidence: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    in_slides: bool = False
    in_transcript: bool = False
    tier: Tier = "B"


class _MergeResult(BaseModel):
    ranked_points: list[_MergedPoint]


_MERGE_SYSTEM = """You merge and rank key points extracted from a course's \
slides and transcripts into one ranked list for an audio overview.

Merge duplicates: when a slide point and a transcript point describe the same \
idea, combine them into ONE point. The merged point's source_chunk_ids is the \
union of both, in_slides and in_transcript are both true if either input point \
had them true, and importance is the max of the inputs, +1 (capped at 5) when \
the merged point is both in_slides AND in_transcript.

Assign each point a tier:
- A: core ideas, anything emphasised or assessed, and prerequisites for later \
topics. Must be covered in depth.
- B: worth a brief mention.
- C: minor or admin items. Admin items about assessment deadlines are tier B, \
not C.

Order the whole list by importance, tier A first, most important first \
overall (this is the rank order).

Respond with ONLY a JSON object (no markdown fences, no commentary) matching:
{"ranked_points": [{"title": str, "summary": str, "kind": str, \
"importance": int 1-5, "evidence": [str,...], "source_chunk_ids": [str,...], \
"in_slides": bool, "in_transcript": bool, "tier": "A"|"B"|"C"}]}"""


def _merge_user_prompt(points_by_source: dict[str, list[KeyPoint]], focus: str | None) -> str:
    focus_line = f"\nFocus for this overview: {focus}\n" if focus else ""
    payload = {
        source: [p.model_dump(mode="json") for p in points]
        for source, points in points_by_source.items()
    }
    return f"{focus_line}\nExtracted key points by source:\n\n{json.dumps(payload, indent=2)}"


def _fallback_rank(all_points: list[KeyPoint]) -> list[RankedKeyPoint]:
    """Deterministic fallback ranking: importance desc, both-sources first;
    top 30% -> A, next 40% -> B, rest -> C.
    """
    ordered = sorted(
        all_points,
        key=lambda p: (p.importance, p.in_slides and p.in_transcript),
        reverse=True,
    )
    n = len(ordered)
    a_cut = round(n * 0.3)
    b_cut = a_cut + round(n * 0.4)
    ranked: list[RankedKeyPoint] = []
    for i, point in enumerate(ordered):
        tier: Tier = "A" if i < a_cut else ("B" if i < b_cut else "C")
        ranked.append(
            RankedKeyPoint(
                **point.model_dump(),
                tier=tier,
                rank=i + 1,
            )
        )
    return ranked


def merge_and_rank(
    points_by_source: dict[str, list[KeyPoint]],
    *,
    client: anthropic.Anthropic,
    settings: Settings,
    focus: str | None = None,
    usage: KeyPointRun | None = None,
) -> list[RankedKeyPoint]:
    """Merge duplicate points across sources and rank them into tiers.
    One script_model (Sonnet 5, effort medium) call via `messages.parse`,
    with a deterministic fallback if the model's output fails validation
    twice.
    """
    all_points = [p for points in points_by_source.values() for p in points]
    if not all_points:
        return []

    user_prompt = _merge_user_prompt(points_by_source, focus)
    all_chunk_ids = {cid for p in all_points for cid in p.source_chunk_ids}

    last_error: Exception | None = None
    for attempt in range(2):
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
        if attempt == 1 and last_error is not None:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That response was invalid "
                        f"({last_error}). Reply with ONLY the JSON object matching the schema."
                    ),
                }
            )
        try:
            response = _invoke_with_retries(
                client=client,
                model=settings.script_model,
                max_tokens=MERGE_MAX_TOKENS,
                system=_MERGE_SYSTEM,
                messages=messages,
                parse_model=_MergeResult,
                effort="medium",
                usage=usage,
            )
        except anthropic.AnthropicError as exc:
            raise map_anthropic_error(exc) from exc
        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            last_error = ValueError("no parsed_output on response")
            logger.warning("merge attempt %d returned no usable output", attempt + 1)
            continue
        try:
            result = (
                parsed if isinstance(parsed, _MergeResult) else _MergeResult.model_validate(parsed)
            )
        except ValidationError as exc:
            last_error = exc
            continue

        ranked: list[RankedKeyPoint] = []
        for i, item in enumerate(result.ranked_points):
            chunk_ids = [cid for cid in item.source_chunk_ids if cid in all_chunk_ids]
            ranked.append(
                RankedKeyPoint(
                    id=f"kp-{i + 1}",
                    title=item.title,
                    summary=item.summary,
                    kind=item.kind,
                    importance=item.importance,
                    evidence=item.evidence,
                    source_chunk_ids=chunk_ids,
                    in_slides=item.in_slides,
                    in_transcript=item.in_transcript,
                    tier=item.tier,
                    rank=i + 1,
                )
            )
        if ranked:
            logger.debug("merge_and_rank succeeded on attempt %d", attempt + 1)
            return ranked
        last_error = ValueError("model returned zero ranked points")

    logger.warning("merge_and_rank falling back to deterministic ranking after: %s", last_error)
    return _fallback_rank(all_points)
