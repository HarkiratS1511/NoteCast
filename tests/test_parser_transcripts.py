"""Tests for notecast.ingest.parsers.transcripts: VTT, SRT, TXT (speaker-turn
and plain), and Markdown parsing. All fixtures are synthetic and written to
tmp_path; nothing under notebooks/ is ever read here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from notecast.ingest.parsers import get_parser, load_builtin_parsers
from notecast.ingest.parsers.transcripts import (
    MarkdownParser,
    SrtParser,
    TxtParser,
    VttParser,
    clean_disfluencies,
)
from notecast.models import SourceType

# --------------------------------------------------------------------------
# VTT
# --------------------------------------------------------------------------


def test_vtt_voice_tags_and_grouping(tmp_path: Path) -> None:
    vtt_text = """WEBVTT

00:00:00.000 --> 00:00:20.000
<v Alice>We begin with the introduction to the topic.

00:00:20.000 --> 00:00:45.000
<v Alice>Next we look at some examples in detail.

00:00:45.000 --> 00:01:05.000
<v Alice>And that wraps up the first idea nicely.

00:01:05.000 --> 00:01:30.000
<v Bob>Now here is a completely different section starting.
"""
    path = tmp_path / "lecture.vtt"
    path.write_text(vtt_text, encoding="utf-8")

    doc = VttParser().parse(path, "lecture.vtt")

    assert doc.metadata["kind"] == "timed_transcript"
    assert doc.metadata["duration_seconds"] == pytest.approx(90.0)
    # First group closes once duration >= 60s AND text ends a sentence
    # (cues 1-3 span 0-65s and end with a period), the 4th cue starts a new
    # group.
    assert len(doc.sections) == 2
    first = doc.sections[0]
    assert first.location.t_start == pytest.approx(0.0)
    assert first.location.t_end == pytest.approx(65.0)
    assert "introduction" in first.text
    assert "wraps up the first idea" in first.text
    assert first.location.speaker == "Alice"

    second = doc.sections[1]
    assert second.location.t_start == pytest.approx(65.0)
    assert second.location.t_end == pytest.approx(90.0)
    assert second.location.speaker == "Bob"


def test_vtt_rolling_caption_dedup(tmp_path: Path) -> None:
    # Auto-caption style: each cue repeats the previous cue's text and adds
    # more words.
    vtt_text = """WEBVTT

00:00:00.000 --> 00:00:05.000
hello there

00:00:05.000 --> 00:00:10.000
hello there this is

00:00:10.000 --> 00:01:35.000
hello there this is a rolling caption test that ends here.
"""
    path = tmp_path / "auto.vtt"
    path.write_text(vtt_text, encoding="utf-8")

    doc = VttParser().parse(path, "auto.vtt")

    full_text = " ".join(section.text for section in doc.sections)
    # The merged text should read like the final, longest cue -- not have
    # "hello there" repeated three times.
    assert full_text.count("hello there") == 1
    assert "rolling caption test that ends here." in full_text


def test_vtt_hard_cap_at_90_seconds(tmp_path: Path) -> None:
    # A single cue with no sentence-ending punctuation running past 90s must
    # still be cut off at the hard cap.
    vtt_text = """WEBVTT

00:00:00.000 --> 00:01:40.000
this cue just keeps going without ever ending a sentence

00:01:40.000 --> 00:01:50.000
and now a new cue continues after the cap
"""
    path = tmp_path / "long.vtt"
    path.write_text(vtt_text, encoding="utf-8")

    doc = VttParser().parse(path, "long.vtt")

    assert doc.sections[0].location.t_end == pytest.approx(100.0)
    assert doc.sections[0].location.t_start == pytest.approx(0.0)
    assert len(doc.sections) == 2


# --------------------------------------------------------------------------
# SRT
# --------------------------------------------------------------------------


def test_srt_grouping_and_times(tmp_path: Path) -> None:
    srt_text = (
        "1\n00:00:00,000 --> 00:00:30,000\nThis is the first part of the talk.\n\n"
        "2\n00:00:30,000 --> 00:01:05,000\nAnd this finishes the opening remarks.\n\n"
        "3\n00:01:05,000 --> 00:01:20,000\nA brand new section begins now.\n"
    )
    path = tmp_path / "clip.srt"
    path.write_text(srt_text, encoding="utf-8")

    doc = SrtParser().parse(path, "clip.srt")

    assert doc.source_type == SourceType.SRT
    assert doc.metadata["kind"] == "timed_transcript"
    assert len(doc.sections) == 2
    assert doc.sections[0].location.t_start == pytest.approx(0.0)
    assert doc.sections[0].location.t_end == pytest.approx(65.0)
    assert doc.sections[1].location.t_start == pytest.approx(65.0)
    assert doc.sections[1].location.t_end == pytest.approx(80.0)
    assert "brand new section" in doc.sections[1].text


def test_srt_malformed_content_raises_friendly_value_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.srt"
    path.write_text("this is not valid srt content at all, no timestamps here\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not parse SRT file"):
        SrtParser().parse(path, "broken.srt")


def test_srt_empty_file_returns_zero_sections(tmp_path: Path) -> None:
    path = tmp_path / "empty.srt"
    path.write_text("", encoding="utf-8")

    doc = SrtParser().parse(path, "empty.srt")

    assert doc.sections == []
    assert doc.metadata["kind"] == "timed_transcript"


def test_srt_whitespace_only_file_returns_zero_sections(tmp_path: Path) -> None:
    path = tmp_path / "blank.srt"
    path.write_text("   \n\n  \n", encoding="utf-8")

    doc = SrtParser().parse(path, "blank.srt")

    assert doc.sections == []


def test_vtt_malformed_content_raises_friendly_value_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.vtt"
    path.write_text("this is not a webvtt file at all\njust garbage\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not parse VTT file"):
        VttParser().parse(path, "broken.vtt")


def test_vtt_empty_file_returns_zero_sections(tmp_path: Path) -> None:
    path = tmp_path / "empty.vtt"
    path.write_text("", encoding="utf-8")

    doc = VttParser().parse(path, "empty.vtt")

    assert doc.sections == []
    assert doc.metadata["kind"] == "timed_transcript"


# --------------------------------------------------------------------------
# TXT: speaker-turn transcript
# --------------------------------------------------------------------------

_NOTICE = (
    "This material has been reproduced and communicated to you by or on behalf "
    "of the Australian National University in accordance with Section 113P of "
    "the Copyright Act 1968. Do not remove this notice."
)


def _speaker_turn_fixture(word_count_turn1: int = 300) -> str:
    filler_words = " ".join(["word"] * word_count_turn1)
    lines = [
        "SPEAKER 0",
        _NOTICE,
        "",
        "SPEAKER 1",
        filler_words,
        "",
        "SPEAKER 2",
        "attending the lab last week and the tutor told me, told me, told me a lot.",
        "",
        "SPEAKER 1",
        "Oh, OK. Yes, I will, yeah, I, I, I, I will work on that.",
    ]
    return "\r\n".join(lines)


def test_speaker_transcript_detection_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "week03_transcript.txt"
    path.write_text(_speaker_turn_fixture(), encoding="utf-8", newline="")

    doc = TxtParser().parse(path, "week03_transcript.txt")

    assert doc.metadata["kind"] == "speaker_transcript"
    assert doc.metadata["is_transcript"] is True
    assert doc.metadata["speakers"] == ["SPEAKER 1", "SPEAKER 2"]
    # SPEAKER 0's turn only contained the copyright notice, which is removed,
    # so it should produce no section at all.
    assert all(section.location.speaker != "SPEAKER 0" for section in doc.sections)
    assert doc.title == "week03 transcript"


def test_speaker_transcript_drops_notice_only_turn(tmp_path: Path) -> None:
    path = tmp_path / "t.txt"
    path.write_text(_speaker_turn_fixture(), encoding="utf-8", newline="")

    doc = TxtParser().parse(path, "t.txt")

    # 3 non-empty turns remain (SPEAKER 1, SPEAKER 2, SPEAKER 1) once the
    # copyright-only SPEAKER 0 turn is dropped.
    assert len(doc.sections) == 3


def test_speaker_transcript_approx_minute_from_raw_words(tmp_path: Path) -> None:
    path = tmp_path / "t.txt"
    path.write_text(_speaker_turn_fixture(word_count_turn1=300), encoding="utf-8", newline="")

    doc = TxtParser().parse(path, "t.txt")

    minutes = [s.location.approx_minute for s in doc.sections]
    # First remaining turn (SPEAKER 1) has 300 raw words -> starts at 0 min.
    assert minutes[0] == pytest.approx(0.0)
    # Second remaining turn (SPEAKER 2) follows those 300 words -> 2.0 min.
    assert minutes[1] == pytest.approx(2.0)
    # Monotonically non-decreasing.
    assert minutes == sorted(minutes)


def test_speaker_transcript_char_offsets_slice_cleaned_text(tmp_path: Path) -> None:
    path = tmp_path / "t.txt"
    path.write_text(_speaker_turn_fixture(), encoding="utf-8", newline="")

    doc = TxtParser().parse(path, "t.txt")
    full_text = "\n\n".join(s.text for s in doc.sections)

    for section in doc.sections:
        start, end = section.location.char_start, section.location.char_end
        assert full_text[start:end] == section.text


def test_speaker_transcript_main_speaker_and_duration(tmp_path: Path) -> None:
    path = tmp_path / "t.txt"
    path.write_text(_speaker_turn_fixture(word_count_turn1=300), encoding="utf-8", newline="")

    doc = TxtParser().parse(path, "t.txt")

    # SPEAKER 0 was dropped (notice only), so SPEAKER 1 (two turns) has the
    # most words among the remaining speakers.
    assert doc.metadata["main_speaker"] == "SPEAKER 1"
    assert doc.metadata["approx_duration_min"] > 0


def test_notes_file_with_label_colons_is_not_a_speaker_transcript(tmp_path: Path) -> None:
    text = (
        "Definition:\n"
        "A stutter is an involuntary repetition of a word or sound in speech.\n"
        "\n"
        "Example:\n"
        'Saying "the the cat" is a simple example of a stutter.\n'
        "\n"
        "Note:\n"
        "Not every repeated word is actually a stutter in casual speech.\n"
        "\n"
        "Definition:\n"
        "A filler word is a sound used by a speaker to fill a pause.\n"
    )
    path = tmp_path / "study_notes.txt"
    path.write_text(text, encoding="utf-8")

    doc = TxtParser().parse(path, "study_notes.txt")

    assert doc.metadata["kind"] == "plain_text"


def test_genuine_interview_transcript_is_detected_as_speaker_transcript(tmp_path: Path) -> None:
    alice_turn = " ".join(["alpha"] * 25) + "."
    bob_turn = " ".join(["beta"] * 25) + "."
    text = f"Alice:\n{alice_turn}\n\nBob:\n{bob_turn}\n\nAlice:\n{alice_turn}\n\nBob:\n{bob_turn}\n"
    path = tmp_path / "interview.txt"
    path.write_text(text, encoding="utf-8")

    doc = TxtParser().parse(path, "interview.txt")

    assert doc.metadata["kind"] == "speaker_transcript"
    assert doc.metadata["speakers"] == ["Alice", "Bob"]


# --------------------------------------------------------------------------
# Disfluency cleanup: positive + negative cases
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Uh, OK.", "OK."),
        ("the, the cat sat down.", "The cat sat down."),
        ("I, I, I, I will work on that.", "I will work on that."),
        ("you, you know what I mean.", "You know what I mean."),
        ("there's some uh fundamentals.", "There's some fundamentals."),
    ],
)
def test_clean_disfluencies_positive(raw: str, expected: str) -> None:
    assert clean_disfluencies(raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        "I saw an umbrella today.",
        "The sum of the two numbers is correct.",
        "The list is 1, 1, 2, 3.",
        "The count reached 22 yesterday.",
    ],
)
def test_clean_disfluencies_negative_cases_unchanged_content(text: str) -> None:
    cleaned = clean_disfluencies(text)
    # These must not be mangled: key words/numbers survive untouched.
    for token in ("umbrella", "sum", "1, 1, 2, 3", "22"):
        if token in text:
            assert token in cleaned


def test_clean_disfluencies_keeps_deliberate_separated_repeats() -> None:
    text = "The cat sat there and the cat ran away."
    cleaned = clean_disfluencies(text)
    assert cleaned.count("cat") == 2


def test_clean_disfluencies_keeps_real_repeated_content_word_across_filler() -> None:
    # Real line from the owner's transcript: "all" is a quantifier, not a
    # stutter-prone function word, so it must survive twice even though
    # removing "uh" makes the two "all"s adjacent.
    text = "because after all, uh, all these are language models."
    cleaned = clean_disfluencies(text)
    assert cleaned.count("all") == 2
    assert cleaned == "Because after all, all these are language models."


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("We, uh, we will handle it.", "We will handle it."),
        ("Check your, uh, your notes.", "Check your notes."),
        ("You may, um, may want to check.", "You may want to check."),
        ("But, uh, but I think so.", "But I think so."),
    ],
)
def test_clean_disfluencies_collapses_function_words_across_filler(raw: str, expected: str) -> None:
    assert clean_disfluencies(raw) == expected


def test_clean_disfluencies_raw_adjacent_still_collapses() -> None:
    # Sanity check: raw-adjacent repeats (no filler between them) keep
    # collapsing regardless of the cross-filler restriction.
    assert clean_disfluencies("the, the cat sat down.") == "The cat sat down."


# --------------------------------------------------------------------------
# TXT: plain text
# --------------------------------------------------------------------------


def test_plain_text_paragraph_grouping(tmp_path: Path) -> None:
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three, a bit longer this time."
    path = tmp_path / "notes.txt"
    path.write_text(text, encoding="utf-8")

    doc = TxtParser().parse(path, "notes.txt")

    assert doc.metadata["kind"] == "plain_text"
    assert len(doc.sections) >= 1
    joined = "\n\n".join(s.text for s in doc.sections)
    assert "Paragraph one." in joined
    assert "Paragraph three" in joined


def test_plain_text_strips_copyright_notice(tmp_path: Path) -> None:
    text = f"{_NOTICE}\n\nSome real content here."
    path = tmp_path / "notes.txt"
    path.write_text(text, encoding="utf-8")

    doc = TxtParser().parse(path, "notes.txt")

    joined = "\n\n".join(s.text for s in doc.sections)
    assert "Copyright Act" not in joined
    assert "Some real content here." in joined


def test_txt_cp1252_encoding_with_smart_quotes(tmp_path: Path) -> None:
    text = "The café was “fully booked” that day.\n\nA second paragraph follows here."
    path = tmp_path / "cp1252.txt"
    path.write_bytes(text.encode("cp1252"))

    doc = TxtParser().parse(path, "cp1252.txt")

    joined = "\n\n".join(s.text for s in doc.sections)
    assert "café" in joined
    assert "fully booked" in joined


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------


def test_markdown_headings_and_code_fence(tmp_path: Path) -> None:
    text = """# Week 3 Notes

Intro paragraph before any subheading content.

## Background

Some background text.

### Details

More detail text, including a fenced code block below.

```
## this looks like a heading but is inside a code fence
still code
```

## Summary

Final summary text.
"""
    path = tmp_path / "notes.md"
    path.write_text(text, encoding="utf-8")

    doc = MarkdownParser().parse(path, "notes.md")

    assert doc.title == "Week 3 Notes"
    titles = [s.title for s in doc.sections]
    assert "Background" in titles
    assert "Details" in titles
    assert "Summary" in titles

    details_section = next(s for s in doc.sections if s.title == "Details")
    assert details_section.location.heading_path == ["Week 3 Notes", "Background", "Details"]
    # The fenced "heading" must not have split the Details section.
    assert "this looks like a heading" in details_section.text
    assert "still code" in details_section.text


def test_markdown_title_falls_back_to_stem_without_h1(tmp_path: Path) -> None:
    text = "## Just a subheading\n\nSome text.\n"
    path = tmp_path / "my-notes.md"
    path.write_text(text, encoding="utf-8")

    doc = MarkdownParser().parse(path, "my-notes.md")

    assert doc.title == "my notes"


# --------------------------------------------------------------------------
# Registry integration
# --------------------------------------------------------------------------


def test_registry_lookup_after_import() -> None:
    load_builtin_parsers()
    assert isinstance(get_parser(Path("a.vtt")), VttParser)
    assert isinstance(get_parser(Path("a.srt")), SrtParser)
    assert isinstance(get_parser(Path("a.txt")), TxtParser)
    assert isinstance(get_parser(Path("a.md")), MarkdownParser)
