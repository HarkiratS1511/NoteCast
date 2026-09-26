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
    # Long enough (>= 15 tokens) that the tiny-slide folding below doesn't
    # kick in — this test is about plain, unrelated slides staying separate.
    slide_one = "Slide one content, covering an overview of the course syllabus in detail."
    slide_two = "Slide two content, covering the assessment breakdown and due dates in detail."
    doc = _doc(
        [
            Section(text=slide_one, location=Location(slide=1), title="Intro"),
            Section(text=slide_two, location=Location(slide=2), title="Details"),
        ]
    )
    chunks = _chunk(doc)
    assert len(chunks) == 2
    assert chunks[0].location.slide == 1
    assert chunks[0].text == slide_one
    assert chunks[1].location.slide == 2
    assert chunks[1].text == slide_two


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
            Section(
                text="Slide one has a very decent amount of content on it for testing purposes.",
                location=Location(slide=1),
            ),
            Section(
                text="Slide two also has a decent amount of content on it for testing purposes.",
                location=Location(slide=2),
            ),
        ]
    )
    chunks_a = _chunk(doc)
    chunks_b = _chunk(doc)
    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]
    assert [c.ordinal for c in chunks_a] == [0, 1]
    assert len({c.chunk_id for c in chunks_a}) == 2


# ---------------------------------------------------------------------------
# Tiny title-only slides (< 15 tokens) get folded into a neighbouring chunk.
# ---------------------------------------------------------------------------


def _real_slide(n: int) -> str:
    return f"Slide {n} has enough real content on it to not be considered a tiny slide at all."


def test_tiny_slide_merges_forward_into_next_slide() -> None:
    doc = _doc(
        [
            Section(text="Overview", location=Location(slide=1), title="Overview"),
            Section(text=_real_slide(2), location=Location(slide=2), title="Details"),
        ]
    )
    chunks = _chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].location.slide == 1
    assert chunks[0].location.slide_end == 2
    assert "Overview" in chunks[0].text
    assert _real_slide(2) in chunks[0].text
    assert chunks[0].header.endswith("slides 1–2")


def test_tiny_slide_at_end_merges_backward() -> None:
    doc = _doc(
        [
            Section(text=_real_slide(1), location=Location(slide=1), title="Details"),
            Section(text="Questions?", location=Location(slide=2), title="Questions"),
        ]
    )
    chunks = _chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].location.slide == 1
    assert chunks[0].location.slide_end == 2
    assert _real_slide(1) in chunks[0].text
    assert "Questions?" in chunks[0].text
    assert chunks[0].header.endswith("slides 1–2")


def test_two_consecutive_tiny_slides_merge_into_next_real_slide() -> None:
    doc = _doc(
        [
            Section(text="Overview", location=Location(slide=1)),
            Section(text="Part Two", location=Location(slide=2)),
            Section(text=_real_slide(3), location=Location(slide=3)),
        ]
    )
    chunks = _chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].location.slide == 1
    assert chunks[0].location.slide_end == 3
    assert chunks[0].header.endswith("slides 1–3")
    for text in ("Overview", "Part Two", _real_slide(3)):
        assert text in chunks[0].text


def test_tiny_slide_run_is_capped_at_three() -> None:
    # Four consecutive tiny slides: at most 3 get folded together.
    doc = _doc(
        [
            Section(text="A", location=Location(slide=1)),
            Section(text="B", location=Location(slide=2)),
            Section(text="C", location=Location(slide=3)),
            Section(text="D", location=Location(slide=4)),
        ]
    )
    chunks = _chunk(doc)
    # Slides 1-3 fold together (cap of 3). Slide 4 is tiny, last, and
    # nothing is left to merge into — the cap means it does NOT fold
    # backward into the already-full group of 3, so it stands alone.
    assert len(chunks) == 2
    assert chunks[0].location.slide == 1
    assert chunks[0].location.slide_end == 3
    assert chunks[0].header.endswith("slides 1–3")
    for text in ("A", "B", "C"):
        assert text in chunks[0].text
    assert "D" not in chunks[0].text

    assert chunks[1].location.slide == 4
    assert chunks[1].location.slide_end is None
    assert chunks[1].text == "D"
    assert chunks[1].header.endswith("slide 4")


def test_non_tiny_slides_keep_normal_single_slide_label() -> None:
    doc = _doc([Section(text=_real_slide(1), location=Location(slide=1))])
    chunks = _chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].location.slide_end is None
    assert chunks[0].header.endswith("slide 1")
    assert "slides" not in chunks[0].header


def test_whitespace_only_sections_skipped_entirely() -> None:
    doc = _doc(
        [
            Section(text="", location=Location(heading_path=["A"])),
            Section(text="   \n  ", location=Location(heading_path=["A"])),
        ]
    )
    chunks = _chunk(doc)
    assert chunks == []
