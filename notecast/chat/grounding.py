"""Turn retrieved chunks into Claude `search_result` content blocks, and
parse Claude's response back into answer segments + citations mapped to
exact course locations (or web results).
"""

from __future__ import annotations

import math
import re
from typing import Any

from notecast.chat.models import AnswerSegment, Citation
from notecast.chat.prompts import NOT_IN_SOURCES_TOKEN
from notecast.models import Chunk, SearchHit

_MAX_BLOCKS_PER_RESULT = 12
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Deep mode ordering: slide/page-style sources (PPTX, PDF, DOCX, TXT, MD) read
# before flowing transcripts (VTT, SRT) within the same week.
_DEEP_TYPE_RANK = {
    "pptx": 0,
    "pdf": 0,
    "docx": 1,
    "txt": 1,
    "md": 1,
    "vtt": 2,
    "srt": 2,
}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from an SDK object or a plain dict, defensively."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _group_units(
    units: list[str], joiner: str, max_blocks: int = _MAX_BLOCKS_PER_RESULT
) -> list[str]:
    """Group small text units (lines or sentences) into blocks of ~1-3
    units each, capped at `max_blocks` blocks total.
    """
    units = [u.strip() for u in units if u.strip()]
    if not units:
        return []
    group_size = 3
    for candidate in (1, 2, 3):
        if math.ceil(len(units) / candidate) <= max_blocks:
            group_size = candidate
            break
    else:
        group_size = math.ceil(len(units) / max_blocks)
    blocks = []
    for i in range(0, len(units), group_size):
        blocks.append(joiner.join(units[i : i + group_size]))
    return blocks


def _is_slide_like(chunk: Any) -> bool:
    return getattr(chunk.location, "slide", None) is not None


def _split_chunk_text(chunk: Any) -> list[str]:
    """Split a chunk's text into a handful of small blocks for fine-grained
    citations: line groups for slides, sentence groups for flowing text.
    """
    text = chunk.text or ""
    if not text.strip():
        return []
    if _is_slide_like(chunk):
        lines = text.splitlines()
        return _group_units(lines, "\n")
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return _group_units(sentences, " ")


def _chunk_to_search_result(chunk: Chunk) -> dict:
    """Build one `search_result` content block for a single chunk."""
    blocks = _split_chunk_text(chunk)
    if not blocks:
        blocks = [(chunk.text or "").strip()]
    return {
        "type": "search_result",
        "source": chunk.source_path,
        "title": chunk.header or chunk.source_path,
        "content": [{"type": "text", "text": block} for block in blocks],
        "citations": {"enabled": True},
    }


def hits_to_search_results(hits: list[SearchHit]) -> list[dict]:
    """Build one `search_result` content block per hit, ready to send in a
    user message's content list.
    """
    return [_chunk_to_search_result(hit.chunk) for hit in hits]


def deep_sort_key(chunk: Chunk) -> tuple[int, int, str, int]:
    """Sort key for deep mode's "whole course" material: by week, then
    slide/page sources before transcripts, then source file, then ordinal.
    """
    week = chunk.week if chunk.week is not None else 10**9
    rank = _DEEP_TYPE_RANK.get(chunk.source_type.value, 1)
    return (week, rank, chunk.source_path, chunk.ordinal)


def sort_chunks_for_deep(chunks: list[Chunk]) -> list[Chunk]:
    """Order chunks the way deep mode presents them to Claude: by week,
    slide/page material before transcripts, then deterministically by
    source file and ordinal.
    """
    return sorted(chunks, key=deep_sort_key)


def chunks_to_search_results(chunks: list[Chunk]) -> list[dict]:
    """Build one `search_result` content block per chunk, in the order
    given (see `sort_chunks_for_deep` for deep mode's ordering).
    """
    return [_chunk_to_search_result(chunk) for chunk in chunks]


def parse_response(
    message: Any, hits: list[SearchHit]
) -> tuple[list[AnswerSegment], list[Citation], bool]:
    """Walk a Claude response's content blocks into answer segments plus a
    deduplicated citation list, and detect a leading NOT_IN_SOURCES token.
    """
    segments: list[AnswerSegment] = []
    citations: list[Citation] = []
    chunk_id_to_n: dict[str, int] = {}
    url_to_n: dict[str, int] = {}
    next_n = 1

    content = _get(message, "content", []) or []
    for block in content:
        block_type = _get(block, "type")
        if block_type != "text":
            # Skip thinking, server_tool_use, web_search_tool_result, etc.
            continue

        text = _get(block, "text", "") or ""
        citation_numbers: list[int] = []

        for cit in _get(block, "citations", None) or []:
            cit_type = _get(cit, "type")
            cited_text = _get(cit, "cited_text", "") or ""

            if cit_type == "search_result_location":
                index = _get(cit, "search_result_index")
                if index is None or not (0 <= index < len(hits)):
                    continue
                chunk = hits[index].chunk
                n = chunk_id_to_n.get(chunk.chunk_id)
                if n is None:
                    n = next_n
                    next_n += 1
                    chunk_id_to_n[chunk.chunk_id] = n
                    citations.append(
                        Citation(
                            n=n,
                            kind="course",
                            cited_text=cited_text,
                            chunk_id=chunk.chunk_id,
                            source_path=chunk.source_path,
                            location_label=chunk.location.label(),
                            header=chunk.header,
                            week=chunk.week,
                        )
                    )
                citation_numbers.append(n)

            elif cit_type == "web_search_result_location":
                url = _get(cit, "url", "") or ""
                title = _get(cit, "title", None)
                n = url_to_n.get(url)
                if n is None:
                    n = next_n
                    next_n += 1
                    url_to_n[url] = n
                    citations.append(
                        Citation(
                            n=n,
                            kind="web",
                            cited_text=cited_text,
                            url=url,
                            title=title,
                        )
                    )
                citation_numbers.append(n)

        segments.append(AnswerSegment(text=text, citation_numbers=citation_numbers))

    not_in_sources = False
    if segments and segments[0].text.startswith(NOT_IN_SOURCES_TOKEN):
        stripped = segments[0].text[len(NOT_IN_SOURCES_TOKEN) :].lstrip()
        segments[0] = AnswerSegment(text=stripped, citation_numbers=segments[0].citation_numbers)
        not_in_sources = True

    return segments, citations, not_in_sources
