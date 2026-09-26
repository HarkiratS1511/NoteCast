"""Tests for structure-aware chunking."""

from __future__ import annotations

from notecast.ingest.chunker import chunk_document
from notecast.ingest.textutils import estimate_tokens
from notecast.models import Location, ParsedDocument, Section, SourceType

COURSE = "comp4650"
COURSE_NAME = "COMP4650 Document Analysis"


def _doc(sections: list[Section], **kwargs) -> ParsedDocument:
    return ParsedDocument(
        source_path=kwargs.pop("source_path", "week-03/lecture.pdf"),
        source_type=kwargs.pop("source_type", SourceType.PDF),
        title=kwargs.pop("title", None),
        sections=sections,
        metadata=kwargs.pop("metadata", {}),
    )


def _chunk(doc: ParsedDocument, **kwargs):
    kwargs.setdefault("course", COURSE)
    kwargs.setdefault("course_name", COURSE_NAME)
    kwargs.setdefault("week", 3)
    kwargs.setdefault("topic", None)
    return chunk_document(doc, **kwargs)


def _paragraph(n_sentences: int, prefix: str = "Sentence") -> str:
    return " ".join(
        f"{prefix} number {i} covers some example lecture content." for i in range(n_sentences)
    )


# ---------------------------------------------------------------------------
# Slide/page sections: one chunk each, never merged, split if oversized.
# ---------------------------------------------------------------------------


def test_slide_sections_one_chunk_each_never_merged() -> None:
    doc = _doc(
        [
            Section(text="Slide one content.", location=Location(slide=1), title="Intro"),
            Section(text="Slide two content.", location=Location(slide=2), title="Details"),
        ]
    )
    chunks = _chunk(doc)
    assert len(chunks) == 2
    assert chunks[0].location.slide == 1
    assert chunks[0].text == "Slide one content."
    assert chunks[1].location.slide == 2
    assert chunks[1].text == "Slide two content."


def test_oversized_slide_is_split_but_keeps_same_location() -> None:
    lines = [f"Bullet point {i} about smoothing and language models here." for i in range(60)]
    text = "\n".join(lines)
    assert estimate_tokens(text) > 500

    doc = _doc([Section(text=text, location=Location(slide=14), title="Smoothing")])
    chunks = _chunk(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.location.slide == 14
        # Packing is approximate (separators aren't counted per-unit), so
        # allow a little slack over the nominal max.
        assert estimate_tokens(chunk.text) <= 550
    # No content is dropped: every bullet line shows up somewhere.
    joined = "\n".join(c.text for c in chunks)
    for line in lines:
        assert line in joined


def test_empty_slide_section_is_skipped() -> None:
    doc = _doc(
        [
            Section(text="   ", location=Location(slide=1)),
            Section(text="Real content.", location=Location(slide=2)),
        ]
    )
    chunks = _chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].location.slide == 2


# ---------------------------------------------------------------------------
# Flowing sections: merge small ones, window large ones.
# ---------------------------------------------------------------------------


def test_small_turns_are_merged() -> None:
    doc = _doc(
        [
            Section(
                text="What is smoothing?",
                location=Location(heading_path=["Q&A"], speaker="Student"),
            ),
            Section(
                text="It's a way to handle unseen n-grams.",
                location=Location(heading_path=["Q&A"], speaker="Lecturer"),
            ),
        ],
        source_type=SourceType.VTT,
    )
    chunks = _chunk(doc)
    assert len(chunks) == 1
    assert "Student: What is smoothing?" in chunks[0].text
    assert "Lecturer: It's a way to handle unseen n-grams." in chunks[0].text


def test_merge_stops_across_different_heading_paths() -> None:
    doc = _doc(
        [
            Section(text="Point one.", location=Location(heading_path=["A"])),
            Section(text="Point two.", location=Location(heading_path=["B"])),
        ],
        source_type=SourceType.DOCX,
    )
    chunks = _chunk(doc)
    assert len(chunks) == 2


def test_large_flowing_section_split_with_overlap_and_increasing_minute() -> None:
    text = _paragraph(120)
    assert estimate_tokens(text) > 500

    section = Section(
        text=text,
        location=Location(heading_path=["Lecture"], approx_minute=10.0),
    )
    doc = _doc([section], source_type=SourceType.VTT)
    chunks = _chunk(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        assert estimate_tokens(chunk.text) <= 500
        assert chunk.location.approx_minute is not None

    minutes = [c.location.approx_minute for c in chunks]
    assert minutes == sorted(minutes)
    assert minutes[0] >= 10.0
    assert minutes[-1] > minutes[0]

    # Overlap: some text from the end of one window reappears at the start
    # of the next.
    first_tail = chunks[0].text[-30:]
    assert first_tail in chunks[1].text or chunks[1].text.startswith(chunks[0].text[-10:])


def test_char_offsets_on_split_section() -> None:
    text = _paragraph(120)
    section = Section(
        text=text,
        location=Location(heading_path=["Notes"], char_start=1000, char_end=1000 + len(text)),
    )
    doc = _doc([section], source_type=SourceType.DOCX)
    chunks = _chunk(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.location.char_start is not None
        assert chunk.location.char_end is not None
        assert chunk.location.char_start >= 1000
        assert chunk.location.char_end <= 1000 + len(text)
        # The chunk's text matches the slice of the original doc text these
        # offsets claim to point at.
        start = chunk.location.char_start - 1000
        end = chunk.location.char_end - 1000
        assert chunk.text == text[start:end]


def test_timed_interpolation_on_split_section() -> None:
    text = _paragraph(120)
    section = Section(
        text=text,
        location=Location(heading_path=["Lecture"], t_start=0.0, t_end=1000.0),
    )
    doc = _doc([section], source_type=SourceType.VTT)
    chunks = _chunk(doc)

    assert len(chunks) > 1
    starts = [c.location.t_start for c in chunks]
    ends = [c.location.t_end for c in chunks]
    assert all(s is not None for s in starts)
    assert all(e is not None for e in ends)
    assert starts == sorted(starts)
    assert starts[0] == 0.0
    assert ends[-1] <= 1000.0
    for s, e in zip(starts, ends, strict=True):
        assert s <= e


# ---------------------------------------------------------------------------
# Headers, ids, and skipping empty text.
# ---------------------------------------------------------------------------


def test_header_format_for_slide() -> None:
    doc = _doc(
        [Section(text="Content.", location=Location(slide=14), title="Smoothing")],
        title="N-gram Language Models",
    )
    chunks = _chunk(doc)
    assert chunks[0].header == (
        "COMP4650 Document Analysis · Week 3 · N-gram Language Models · Smoothing · slide 14"
    )


def test_header_omits_section_title_when_same_as_doc_title() -> None:
    doc = _doc(
        [Section(text="Content.", location=Location(slide=1), title="Intro")],
        title="Intro",
    )
    chunks = _chunk(doc)
    assert chunks[0].header == "COMP4650 Document Analysis · Week 3 · Intro · slide 1"


def test_header_labels_transcript_kind() -> None:
    doc = _doc(
        [Section(text="Some spoken content.", location=Location(approx_minute=23.0))],
        source_type=SourceType.VTT,
        metadata={"kind": "speaker_transcript"},
    )
    chunks = _chunk(doc, week=None)
    assert chunks[0].header == "COMP4650 Document Analysis · Lecture transcript · ≈ 23 min"


def test_header_no_week_omits_week_part() -> None:
    doc = _doc([Section(text="Content.", location=Location(slide=1))])
    chunks = _chunk(doc, week=None)
    assert "Week" not in chunks[0].header


def test_chunk_ids_are_stable_and_ordinals_sequential() -> None:
    doc = _doc(
        [
            Section(text="Slide one.", location=Location(slide=1)),
            Section(text="Slide two.", location=Location(slide=2)),
        ]
    )
    chunks_a = _chunk(doc)
    chunks_b = _chunk(doc)
    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]
    assert [c.ordinal for c in chunks_a] == [0, 1]
    assert len({c.chunk_id for c in chunks_a}) == 2


def test_whitespace_only_sections_skipped_entirely() -> None:
    doc = _doc(
        [
            Section(text="", location=Location(heading_path=["A"])),
            Section(text="   \n  ", location=Location(heading_path=["A"])),
        ]
    )
    chunks = _chunk(doc)
    assert chunks == []
