"""Tests for notecast.models: Location labels, SourceType inference, and
Chunk id/embed_text behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from notecast.models import Chunk, Location, SourceType


class TestSourceTypeFromPath:
    def test_common_extensions(self) -> None:
        assert SourceType.from_path(Path("lecture.pdf")) is SourceType.PDF
        assert SourceType.from_path(Path("slides.pptx")) is SourceType.PPTX
        assert SourceType.from_path(Path("notes.docx")) is SourceType.DOCX
        assert SourceType.from_path(Path("captions.vtt")) is SourceType.VTT
        assert SourceType.from_path(Path("captions.srt")) is SourceType.SRT
        assert SourceType.from_path(Path("readme.txt")) is SourceType.TXT
        assert SourceType.from_path(Path("readme.md")) is SourceType.MD

    def test_uppercase_extension(self) -> None:
        assert SourceType.from_path(Path("SLIDES.PPTX")) is SourceType.PPTX

    def test_unsupported_extension_raises(self) -> None:
        with pytest.raises(ValueError):
            SourceType.from_path(Path("video.mp4"))


class TestLocationLabel:
    def test_page_only(self) -> None:
        assert Location(page=12).label() == "p. 12"

    def test_slide_only(self) -> None:
        assert Location(slide=4).label() == "slide 4"

    def test_heading_path(self) -> None:
        loc = Location(heading_path=["Introduction", "Motivation"])
        assert loc.label() == "Introduction › Motivation"

    def test_time_range_minutes_seconds(self) -> None:
        loc = Location(t_start=872, t_end=940)
        assert loc.label() == "14:32–15:40"

    def test_time_range_over_an_hour(self) -> None:
        loc = Location(t_start=3600, t_end=3661)
        assert loc.label() == "1:00:00–1:01:01"

    def test_time_start_only(self) -> None:
        loc = Location(t_start=65)
        assert loc.label() == "01:05"

    def test_combines_multiple_parts(self) -> None:
        loc = Location(page=3, heading_path=["Intro"])
        assert loc.label() == "p. 3, Intro"

    def test_empty_location(self) -> None:
        assert Location().label() == ""


class TestChunkMakeId:
    def test_stable(self) -> None:
        a = Chunk.make_id("comp4650", "week-01/lec.pdf", 0, "hello world")
        b = Chunk.make_id("comp4650", "week-01/lec.pdf", 0, "hello world")
        assert a == b
        assert len(a) == 16

    def test_sensitive_to_each_field(self) -> None:
        base = Chunk.make_id("comp4650", "week-01/lec.pdf", 0, "hello world")
        assert base != Chunk.make_id("comp4651", "week-01/lec.pdf", 0, "hello world")
        assert base != Chunk.make_id("comp4650", "week-02/lec.pdf", 0, "hello world")
        assert base != Chunk.make_id("comp4650", "week-01/lec.pdf", 1, "hello world")
        assert base != Chunk.make_id("comp4650", "week-01/lec.pdf", 0, "goodbye world")


class TestChunkEmbedText:
    def test_with_header(self) -> None:
        chunk = Chunk(
            chunk_id="abc",
            course="comp4650",
            source_path="week-01/lec.pdf",
            source_type=SourceType.PDF,
            ordinal=0,
            text="Some body text.",
            header="COMP4650 · Week 1 · slide 4",
        )
        assert chunk.embed_text == "COMP4650 · Week 1 · slide 4\n\nSome body text."

    def test_without_header(self) -> None:
        chunk = Chunk(
            chunk_id="abc",
            course="comp4650",
            source_path="week-01/lec.pdf",
            source_type=SourceType.PDF,
            ordinal=0,
            text="Some body text.",
        )
        assert chunk.embed_text == "Some body text."
