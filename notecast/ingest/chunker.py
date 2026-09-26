"""Structure-aware chunking: one chunk per slide/page (splitting oversized
ones), and merge-or-window chunking for flowing text (transcripts, prose,
heading sections), each chunk carrying a contextual header for embedding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from notecast.ingest.textutils import SPOKEN_WORDS_PER_MINUTE, estimate_tokens
from notecast.models import Chunk, Location, ParsedDocument, Section

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TRANSCRIPT_KINDS = {"speaker_transcript", "timed_transcript"}

# Slides/pages with less text than this (title-only slides, section
# dividers, "Questions?" slides) get folded into a neighbouring chunk
# instead of standing alone as their own near-empty, near-duplicate chunk.
_TINY_SLIDE_TOKENS = 15
# Never fold more than this many consecutive tiny slides into one chunk.
_MAX_TINY_SLIDE_RUN = 3


@dataclass
class _RawChunk:
    """One chunk-to-be, before ids/headers are assigned. `is_slide` and the
    `*_num` fields track slide/page numbers so tiny slides can be folded
    into a neighbour and the merged range recorded for the header.
    """

    text: str
    location: Location
    title: str | None
    is_slide: bool = False
    start_num: int | None = None
    end_num: int | None = None
    # How many original slide/page sections this chunk already represents,
    # so tiny-slide folding can respect `_MAX_TINY_SLIDE_RUN`.
    component_count: int = 1


def _is_slide_like(section: Section) -> bool:
    return section.location.page is not None or section.location.slide is not None


def _pack(units: list[str], max_tokens: int, sep: str) -> list[str]:
    """Greedily pack `units` into groups, each roughly under `max_tokens`."""
    groups: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for unit in units:
        tokens = estimate_tokens(unit)
        if current and current_tokens + tokens > max_tokens:
            groups.append(sep.join(current))
            current = [unit]
            current_tokens = tokens
        else:
            current.append(unit)
            current_tokens += tokens
    if current:
        groups.append(sep.join(current))
    return groups


def _split_to_max(text: str, max_tokens: int) -> list[str]:
    """Split `text` into pieces each under `max_tokens`, preferring line
    boundaries, then sentence boundaries, then word boundaries.
    """
    if estimate_tokens(text) <= max_tokens:
        return [text]

    lines = [line for line in text.split("\n") if line.strip()]
    groups = _pack(lines, max_tokens, "\n") if len(lines) > 1 else [text]

    result: list[str] = []
    for group in groups:
        if estimate_tokens(group) <= max_tokens:
            result.append(group)
            continue
        sentences = [s for s in _SENTENCE_SPLIT.split(group) if s.strip()]
        subgroups = _pack(sentences, max_tokens, " ") if len(sentences) > 1 else [group]
        for subgroup in subgroups:
            if estimate_tokens(subgroup) <= max_tokens:
                result.append(subgroup)
                continue
            words = subgroup.split()
            result.extend(_pack(words, max_tokens, " "))
    return result


def _merge_flowing(buffer: list[Section]) -> tuple[str, Location, str | None]:
    """Combine consecutive small flowing sections into one chunk."""
    first, last = buffer[0], buffer[-1]
    pieces: list[str] = []
    for section in buffer:
        text = section.text.strip()
        if not text:
            continue
        speaker = section.location.speaker
        pieces.append(f"{speaker}: {text}" if speaker else text)
    text = "\n\n".join(pieces)

    loc = first.location.model_copy()
    if last.location.t_end is not None:
        loc.t_end = last.location.t_end
    if last.location.char_end is not None:
        loc.char_end = last.location.char_end
    return text, loc, first.title


def _flowing_units(text: str, max_tokens: int) -> list[tuple[str, int, int]]:
    """Split flowing text into sentence-ish units, each under `max_tokens`,
    returning (unit_text, char_start, char_end) tuples relative to `text`.
    """
    raw_sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not raw_sentences and text.strip():
        raw_sentences = [text]

    units: list[str] = []
    for sentence in raw_sentences:
        if estimate_tokens(sentence) <= max_tokens:
            units.append(sentence)
        else:
            units.extend(_pack(sentence.split(), max_tokens, " "))

    spans: list[tuple[str, int, int]] = []
    pos = 0
    for unit in units:
        idx = text.find(unit, pos)
        if idx == -1:
            idx = pos
        spans.append((unit, idx, idx + len(unit)))
        pos = idx + len(unit)
    return spans


def _window_indices(
    unit_tokens: list[int], target_tokens: int, max_tokens: int, overlap_ratio: float
) -> list[list[int]]:
    """Group unit indices into overlapping windows targeting `target_tokens`."""
    n = len(unit_tokens)
    windows: list[list[int]] = []
    i = 0
    while i < n:
        current: list[int] = []
        tokens = 0
        j = i
        while j < n:
            t = unit_tokens[j]
            if current and tokens + t > max_tokens:
                break
            current.append(j)
            tokens += t
            j += 1
            if tokens >= target_tokens:
                break
        windows.append(current)
        if j >= n:
            break
        overlap_budget = tokens * overlap_ratio
        overlap_count = 0
        acc = 0
        while overlap_count < len(current) - 1:
            idx = current[-(overlap_count + 1)]
            acc += unit_tokens[idx]
            if acc > overlap_budget:
                break
            overlap_count += 1
        i = j - overlap_count if overlap_count > 0 else j
    return windows


def _window_location(section: Section, start_char: int, end_char: int) -> Location:
    loc = section.location.model_copy()
    if section.location.char_start is not None:
        loc.char_start = section.location.char_start + start_char
        loc.char_end = section.location.char_start + end_char
    if section.location.approx_minute is not None:
        words_before = len(section.text[:start_char].split())
        loc.approx_minute = section.location.approx_minute + words_before / SPOKEN_WORDS_PER_MINUTE
    if section.location.t_start is not None and section.location.t_end is not None:
        total_chars = max(1, len(section.text))
        span = section.location.t_end - section.location.t_start
        loc.t_start = section.location.t_start + span * (start_char / total_chars)
        loc.t_end = section.location.t_start + span * (end_char / total_chars)
    return loc


def _split_flowing_section(
    section: Section, target_tokens: int, max_tokens: int, overlap_ratio: float
) -> list[tuple[str, Location]]:
    text = section.text
    units = _flowing_units(text, max_tokens)
    if not units:
        return []
    unit_tokens = [estimate_tokens(u[0]) for u in units]
    windows = _window_indices(unit_tokens, target_tokens, max_tokens, overlap_ratio)

    result: list[tuple[str, Location]] = []
    for idx_list in windows:
        if not idx_list:
            continue
        start_char = units[idx_list[0]][1]
        end_char = units[idx_list[-1]][2]
        chunk_text = text[start_char:end_char]
        result.append((chunk_text, _window_location(section, start_char, end_char)))
    return result


def _is_tiny_slide(chunk: _RawChunk) -> bool:
    return estimate_tokens(chunk.text) < _TINY_SLIDE_TOKENS


def _combine_slide_chunks(chunks: list[_RawChunk]) -> _RawChunk:
    first, last = chunks[0], chunks[-1]
    text = "\n\n".join(c.text for c in chunks if c.text.strip())
    title = last.title or first.title
    return _RawChunk(
        text=text,
        location=first.location,
        title=title,
        is_slide=True,
        start_num=first.start_num,
        end_num=last.end_num,
        component_count=sum(c.component_count for c in chunks),
    )


def _merge_tiny_slide_run(entries: list[_RawChunk]) -> list[_RawChunk]:
    """Fold tiny slide/page chunks into a neighbour: forward into the next
    slide (leading and interior tiny slides), or backward into the previous
    chunk when a tiny slide is the last one in the document. Never folds
    more than `_MAX_TINY_SLIDE_RUN` consecutive tiny slides together.
    """
    n = len(entries)
    merged: list[_RawChunk] = []
    i = 0
    while i < n:
        entry = entries[i]
        if _is_tiny_slide(entry) and i < n - 1:
            group = [entry]
            j = i
            while len(group) < _MAX_TINY_SLIDE_RUN and _is_tiny_slide(entries[j]) and j + 1 < n:
                j += 1
                group.append(entries[j])
            merged.append(_combine_slide_chunks(group))
            i = j + 1
        elif _is_tiny_slide(entry) and i == n - 1:
            # Last slide in the document and tiny: nothing to merge forward
            # into, so fold it backward into the previous chunk instead —
            # unless that would push the run past the cap, in which case it
            # stands alone rather than joining an already-full group.
            if merged and merged[-1].component_count < _MAX_TINY_SLIDE_RUN:
                merged[-1] = _combine_slide_chunks([merged[-1], entry])
            else:
                merged.append(entry)
            i += 1
        else:
            merged.append(entry)
            i += 1
    return merged


def _apply_tiny_slide_merging(raw: list[_RawChunk]) -> list[_RawChunk]:
    """Run tiny-slide folding over each contiguous run of slide/page
    chunks, leaving any flowing chunks between them untouched.
    """
    result: list[_RawChunk] = []
    n = len(raw)
    i = 0
    while i < n:
        if raw[i].is_slide:
            j = i
            while j < n and raw[j].is_slide:
                j += 1
            result.extend(_merge_tiny_slide_run(raw[i:j]))
            i = j
        else:
            result.append(raw[i])
            i += 1
    return result


def _build_header(
    *,
    course_name: str,
    week: int | None,
    doc: ParsedDocument,
    section_title: str | None,
    location: Location,
    location_label: str | None = None,
) -> str:
    parts: list[str] = []
    if course_name:
        parts.append(course_name)
    if week is not None:
        parts.append(f"Week {week}")
    if doc.title:
        parts.append(doc.title)
    if section_title and section_title != doc.title:
        parts.append(section_title)
    if doc.metadata.get("kind") in _TRANSCRIPT_KINDS:
        parts.append("Lecture transcript")
    label = location_label if location_label is not None else location.label()
    if label:
        parts.append(label)
    return " · ".join(p for p in parts if p)


def chunk_document(
    doc: ParsedDocument,
    *,
    course: str,
    course_name: str,
    week: int | None,
    topic: str | None,
    target_tokens: int = 350,
    max_tokens: int = 500,
    overlap_ratio: float = 0.15,
) -> list[Chunk]:
    """Turn a parsed document's sections into retrievable chunks."""
    raw: list[_RawChunk] = []
    buffer: list[Section] = []

    def flush_buffer() -> None:
        if not buffer:
            return
        text, loc, title = _merge_flowing(buffer)
        if text.strip():
            raw.append(_RawChunk(text=text, location=loc, title=title))
        buffer.clear()

    for section in doc.sections:
        text = section.text.strip()
        if not text:
            continue

        if _is_slide_like(section):
            flush_buffer()
            slide_num = section.location.slide
            page_num = section.location.page
            num = slide_num if slide_num is not None else page_num
            if estimate_tokens(text) <= max_tokens:
                raw.append(
                    _RawChunk(
                        text=text,
                        location=section.location,
                        title=section.title,
                        is_slide=True,
                        start_num=num,
                        end_num=num,
                    )
                )
            else:
                for part in _split_to_max(text, max_tokens):
                    part = part.strip()
                    if part:
                        raw.append(
                            _RawChunk(
                                text=part,
                                location=section.location,
                                title=section.title,
                                is_slide=True,
                                start_num=num,
                                end_num=num,
                            )
                        )
            continue

        tokens = estimate_tokens(text)
        if tokens > max_tokens:
            flush_buffer()
            for window_text, window_loc in _split_flowing_section(
                section, target_tokens, max_tokens, overlap_ratio
            ):
                window_text = window_text.strip()
                if window_text:
                    raw.append(
                        _RawChunk(text=window_text, location=window_loc, title=section.title)
                    )
            continue

        if buffer:
            combined_tokens = sum(estimate_tokens(s.text) for s in buffer) + tokens
            same_heading = buffer[-1].location.heading_path == section.location.heading_path
            if same_heading and combined_tokens <= target_tokens:
                buffer.append(section)
                continue
            flush_buffer()
        buffer.append(section)

    flush_buffer()

    raw = _apply_tiny_slide_merging(raw)

    chunks: list[Chunk] = []
    ordinal = 0
    for item in raw:
        if not item.text.strip():
            continue
        location_label: str | None = None
        if (
            item.is_slide
            and item.start_num is not None
            and item.end_num is not None
            and item.start_num != item.end_num
        ):
            unit = "pages" if item.location.page is not None else "slides"
            location_label = f"{unit} {item.start_num}–{item.end_num}"
        header = _build_header(
            course_name=course_name,
            week=week,
            doc=doc,
            section_title=item.title,
            location=item.location,
            location_label=location_label,
        )
        chunk_id = Chunk.make_id(course, doc.source_path, ordinal, item.text)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                course=course,
                source_path=doc.source_path,
                source_type=doc.source_type,
                ordinal=ordinal,
                text=item.text,
                header=header,
                location=item.location,
                week=week,
                topic=topic,
            )
        )
        ordinal += 1
    return chunks
