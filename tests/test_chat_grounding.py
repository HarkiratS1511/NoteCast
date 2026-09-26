"""Tests for notecast.chat.grounding: search_result construction and
citation parsing.
"""

from __future__ import annotations

from types import SimpleNamespace

from notecast.chat.grounding import (
    chunks_to_search_results,
    hits_to_search_results,
    parse_response,
    sort_chunks_for_deep,
)
from notecast.chat.prompts import NOT_IN_SOURCES_TOKEN
from notecast.models import Chunk, Location, SearchHit, SourceType


def _slide_hit(chunk_id: str = "slide1", n_lines: int = 20) -> SearchHit:
    text = "\n".join(f"Bullet point number {i}" for i in range(n_lines))
    chunk = Chunk(
        chunk_id=chunk_id,
        course="comp4650",
        source_path="week-03/lecture-2.pptx",
        source_type=SourceType.PPTX,
        ordinal=0,
        text=text,
        header="COMP4650 · Week 3 · Lecture 2 · slide 4",
        location=Location(slide=4),
        week=3,
    )
    return SearchHit(chunk=chunk, score=0.9)


def _transcript_hit(chunk_id: str = "trans1", n_sentences: int = 20) -> SearchHit:
    text = " ".join(f"This is sentence number {i}." for i in range(n_sentences))
    chunk = Chunk(
        chunk_id=chunk_id,
        course="comp4650",
        source_path="week-03/lecture-2.vtt",
        source_type=SourceType.VTT,
        ordinal=1,
        text=text,
        header="COMP4650 · Week 3 · Lecture 2 · 12:00-13:00",
        location=Location(t_start=720, t_end=780),
        week=3,
    )
    return SearchHit(chunk=chunk, score=0.8)


def test_slide_chunk_split_by_lines_and_capped() -> None:
    hit = _slide_hit()
    results = hits_to_search_results([hit])
    assert len(results) == 1
    result = results[0]
    assert result["type"] == "search_result"
    assert result["source"] == "week-03/lecture-2.pptx"
    assert result["title"] == "COMP4650 · Week 3 · Lecture 2 · slide 4"
    assert result["citations"] == {"enabled": True}
    assert 1 <= len(result["content"]) <= 12
    for block in result["content"]:
        assert block["type"] == "text"
        assert block["text"].strip()


def test_transcript_chunk_split_by_sentences_and_capped() -> None:
    hit = _transcript_hit()
    results = hits_to_search_results([hit])
    result = results[0]
    assert 1 <= len(result["content"]) <= 12
    # Sentences within a block should be space-joined, not newline-joined.
    assert all("\n" not in block["text"] for block in result["content"])


def test_short_chunk_produces_at_least_one_block() -> None:
    chunk = Chunk(
        chunk_id="short1",
        course="comp4650",
        source_path="week-01/intro.pdf",
        source_type=SourceType.PDF,
        ordinal=0,
        text="Just one short slide.",
        header="Intro",
        location=Location(page=1),
    )
    results = hits_to_search_results([SearchHit(chunk=chunk, score=1.0)])
    assert len(results[0]["content"]) == 1


def test_empty_chunk_text_still_yields_one_block() -> None:
    chunk = Chunk(
        chunk_id="empty1",
        course="comp4650",
        source_path="week-01/blank.pdf",
        source_type=SourceType.PDF,
        ordinal=0,
        text="",
        header="Blank",
        location=Location(page=1),
    )
    results = hits_to_search_results([SearchHit(chunk=chunk, score=1.0)])
    assert len(results[0]["content"]) == 1
    assert results[0]["content"][0]["text"] == ""


def test_whitespace_only_chunk_text_yields_single_empty_block() -> None:
    chunk = Chunk(
        chunk_id="whitespace1",
        course="comp4650",
        source_path="week-01/blank.pdf",
        source_type=SourceType.PDF,
        ordinal=0,
        text="   \n\t  ",
        header="Blank",
        location=Location(page=1),
    )
    results = hits_to_search_results([SearchHit(chunk=chunk, score=1.0)])
    assert len(results[0]["content"]) == 1
    assert results[0]["content"][0]["text"] == ""


def test_prose_pdf_without_slide_location_split_by_sentences() -> None:
    """A PDF chunk with no slide number (a real prose document, not a
    slide export) should be split like flowing text, not like slides.
    """
    text = " ".join(f"This is sentence number {i}." for i in range(20))
    chunk = Chunk(
        chunk_id="prose1",
        course="comp4650",
        source_path="week-01/notes.pdf",
        source_type=SourceType.PDF,
        ordinal=0,
        text=text,
        header="Notes",
        location=Location(page=3),
    )
    results = hits_to_search_results([SearchHit(chunk=chunk, score=1.0)])
    assert all("\n" not in block["text"] for block in results[0]["content"])


def _course_chunk(
    chunk_id: str, *, week: int | None, source_type: SourceType, source_path: str, ordinal: int = 0
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        course="comp4650",
        source_path=source_path,
        source_type=source_type,
        ordinal=ordinal,
        text="Some material.",
        header=source_path,
        week=week,
    )


def test_sort_chunks_for_deep_orders_by_week_then_type_then_source_then_ordinal() -> None:
    chunks = [
        _course_chunk("a", week=5, source_type=SourceType.PPTX, source_path="w5/a.pptx"),
        _course_chunk("b", week=3, source_type=SourceType.VTT, source_path="w3/a.vtt"),
        _course_chunk("c", week=3, source_type=SourceType.PDF, source_path="w3/a.pdf"),
        _course_chunk("d", week=3, source_type=SourceType.PDF, source_path="w3/a.pdf", ordinal=1),
        _course_chunk("e", week=None, source_type=SourceType.TXT, source_path="misc/notes.txt"),
    ]
    ordered = sort_chunks_for_deep(chunks)
    assert [c.chunk_id for c in ordered] == ["c", "d", "b", "a", "e"]


def test_chunks_to_search_results_preserves_given_order() -> None:
    chunks = [
        _course_chunk("first", week=1, source_type=SourceType.PDF, source_path="w1/a.pdf"),
        _course_chunk("second", week=1, source_type=SourceType.VTT, source_path="w1/a.vtt"),
    ]
    results = chunks_to_search_results(chunks)
    assert [r["source"] for r in results] == ["w1/a.pdf", "w1/a.vtt"]
    assert all(r["citations"] == {"enabled": True} for r in results)


def _dict_message(content: list[dict]) -> dict:
    return {"content": content, "stop_reason": "end_turn"}


def test_parse_response_maps_search_result_index_to_hit_dict_form() -> None:
    hits = [_slide_hit("chunk-a"), _transcript_hit("chunk-b")]
    message = _dict_message(
        [
            {
                "type": "text",
                "text": "Bullet point number 0",
                "citations": [
                    {
                        "type": "search_result_location",
                        "cited_text": "Bullet point number 0",
                        "source": hits[0].chunk.source_path,
                        "title": hits[0].chunk.header,
                        "search_result_index": 0,
                        "start_block_index": 0,
                        "end_block_index": 1,
                    }
                ],
            }
        ]
    )
    segments, citations, not_in_sources = parse_response(message, hits)
    assert not_in_sources is False
    assert len(segments) == 1
    assert segments[0].citation_numbers == [1]
    assert len(citations) == 1
    citation = citations[0]
    assert citation.n == 1
    assert citation.kind == "course"
    assert citation.chunk_id == "chunk-a"
    assert citation.source_path == hits[0].chunk.source_path
    assert citation.location_label == hits[0].chunk.location.label()
    assert citation.week == 3


def test_parse_response_dedupes_citations_by_chunk_id() -> None:
    hits = [_slide_hit("chunk-a")]
    citation_block = {
        "type": "search_result_location",
        "cited_text": "some text",
        "source": hits[0].chunk.source_path,
        "title": hits[0].chunk.header,
        "search_result_index": 0,
        "start_block_index": 0,
        "end_block_index": 1,
    }
    message = _dict_message(
        [
            {"type": "text", "text": "first mention", "citations": [citation_block]},
            {"type": "text", "text": "second mention", "citations": [citation_block]},
        ]
    )
    segments, citations, _ = parse_response(message, hits)
    assert len(citations) == 1
    assert segments[0].citation_numbers == [1]
    assert segments[1].citation_numbers == [1]


def test_parse_response_web_citation_deduped_by_url() -> None:
    message = _dict_message(
        [
            {
                "type": "text",
                "text": "web fact",
                "citations": [
                    {
                        "type": "web_search_result_location",
                        "cited_text": "web fact detail",
                        "url": "https://example.com/a",
                        "title": "Example A",
                        "encrypted_index": "abc",
                    }
                ],
            },
            {
                "type": "text",
                "text": "same web fact again",
                "citations": [
                    {
                        "type": "web_search_result_location",
                        "cited_text": "web fact detail",
                        "url": "https://example.com/a",
                        "title": "Example A",
                        "encrypted_index": "abc",
                    }
                ],
            },
        ]
    )
    segments, citations, _ = parse_response(message, [])
    assert len(citations) == 1
    assert citations[0].kind == "web"
    assert citations[0].url == "https://example.com/a"
    assert citations[0].title == "Example A"
    assert segments[0].citation_numbers == segments[1].citation_numbers == [1]


def test_parse_response_ignores_thinking_and_tool_blocks() -> None:
    message = _dict_message(
        [
            {"type": "thinking", "thinking": "reasoning..."},
            {"type": "server_tool_use", "name": "web_search"},
            {"type": "web_search_tool_result", "content": []},
            {"type": "text", "text": "final answer"},
        ]
    )
    segments, citations, _ = parse_response(message, [])
    assert len(segments) == 1
    assert segments[0].text == "final answer"
    assert citations == []


def test_parse_response_detects_and_strips_not_in_sources_token() -> None:
    message = _dict_message(
        [
            {
                "type": "text",
                "text": f"{NOT_IN_SOURCES_TOKEN} This isn't covered in the material.",
            }
        ]
    )
    segments, citations, not_in_sources = parse_response(message, [])
    assert not_in_sources is True
    assert segments[0].text == "This isn't covered in the material."


def test_parse_response_works_with_sdk_object_form() -> None:
    hits = [_slide_hit("chunk-a")]
    citation = SimpleNamespace(
        type="search_result_location",
        cited_text="Bullet point number 0",
        source=hits[0].chunk.source_path,
        title=hits[0].chunk.header,
        search_result_index=0,
        start_block_index=0,
        end_block_index=1,
    )
    text_block = SimpleNamespace(type="text", text="Bullet point number 0", citations=[citation])
    message = SimpleNamespace(content=[text_block], stop_reason="end_turn")

    segments, citations_out, not_in_sources = parse_response(message, hits)
    assert not_in_sources is False
    assert segments[0].citation_numbers == [1]
    assert citations_out[0].chunk_id == "chunk-a"


def test_parse_response_ignores_out_of_range_search_result_index() -> None:
    message = _dict_message(
        [
            {
                "type": "text",
                "text": "stray citation",
                "citations": [
                    {
                        "type": "search_result_location",
                        "cited_text": "x",
                        "source": "nowhere",
                        "title": None,
                        "search_result_index": 5,
                        "start_block_index": 0,
                        "end_block_index": 1,
                    }
                ],
            }
        ]
    )
    segments, citations, _ = parse_response(message, [])
    assert citations == []
    assert segments[0].citation_numbers == []
