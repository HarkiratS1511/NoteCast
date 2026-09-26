"""Parsers for timed transcripts (``.vtt``, ``.srt``), plain/speaker-turn
text transcripts (``.txt``), and Markdown documents (``.md``).

Each parser class is registered with the shared registry at the bottom of
this module, so importing it (e.g. via
``notecast.ingest.parsers.load_builtin_parsers``) is enough to make these
extensions resolvable through ``get_parser``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import srt
import webvtt

from notecast.ingest.parsers import register
from notecast.ingest.textutils import (
    SPOKEN_WORDS_PER_MINUTE,
    normalize_whitespace,
    strip_copyright_notice,
)
from notecast.models import Location, ParsedDocument, Section, SourceType

# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


def _decode_bytes(raw: bytes) -> str:
    """Decode raw file bytes, trying utf-8-sig, then cp1252, then latin-1.

    latin-1 can decode any byte sequence, so this always succeeds.
    """
    for encoding in _ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    # Unreachable in practice (latin-1 never raises), but keeps mypy happy.
    return raw.decode("latin-1")


def _read_text(path: Path) -> str:
    return _decode_bytes(path.read_bytes())


def _parse_timestamp(value: str) -> float:
    """Parse a "HH:MM:SS.mmm" (or "MM:SS.mmm") timestamp into float seconds.

    webvtt-py's own `start_in_seconds`/`end_in_seconds` truncate to whole
    seconds, so we parse the timestamp string ourselves for sub-second
    precision.
    """
    parts = value.strip().split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        hours = "0"
        minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


_VOICE_TAG = re.compile(r"<v(?:\.[\w-]+)*\s+([^>]+)>")
_CUE_TAG = re.compile(r"<[^>]*>")
_SENTENCE_END = (".", "!", "?")


def _split_voice(raw_text: str) -> tuple[str | None, str]:
    """Extract a `<v Speaker>` voice name (if present) and strip all cue tags."""
    match = _VOICE_TAG.search(raw_text)
    speaker = match.group(1).strip() if match else None
    text = _CUE_TAG.sub("", raw_text)
    text = normalize_whitespace(text.replace("\n", " "))
    return speaker, text


@dataclass
class _Cue:
    start: float
    end: float
    speaker: str | None
    text: str


def _dedup_rolling_captions(cues: list[_Cue]) -> list[_Cue]:
    """Collapse the rolling-caption pattern of auto-generated captions, where
    consecutive cues repeat the previous cue's text and just append new words.

    A cue that exactly repeats the previous one is emptied out; a cue whose
    text starts with the previous cue's text keeps only the new suffix.
    """
    cleaned: list[_Cue] = []
    previous_text = ""
    for cue in cues:
        text = cue.text
        if text and previous_text:
            if text == previous_text:
                text = ""
            elif text.startswith(previous_text):
                text = text[len(previous_text) :].strip()
        if cue.text:
            previous_text = cue.text
        cleaned.append(_Cue(cue.start, cue.end, cue.speaker, text))
    return cleaned


@dataclass
class _Group:
    start: float
    end: float
    speaker: str | None
    texts: list[str] = field(default_factory=list)


def _group_cues(cues: list[_Cue]) -> list[_Group]:
    """Group consecutive cues into sections of ~60-90s.

    A group closes once its accumulated duration reaches 60s and the last
    non-empty cue text ends a sentence, or once it hits the 90s hard cap.
    """
    groups: list[_Group] = []
    current: _Group | None = None
    for cue in cues:
        if current is None:
            current = _Group(start=cue.start, end=cue.end, speaker=cue.speaker)
        else:
            current.end = cue.end
        if cue.text:
            current.texts.append(cue.text)
        duration = current.end - current.start
        last_text = next((t for t in reversed(current.texts) if t), "")
        ends_sentence = last_text.rstrip().endswith(_SENTENCE_END)
        if duration >= 90 or (duration >= 60 and ends_sentence):
            groups.append(current)
            current = None
    if current is not None:
        groups.append(current)
    return groups


def _groups_to_sections(groups: list[_Group]) -> list[Section]:
    sections = []
    for group in groups:
        text = " ".join(t for t in group.texts if t)
        sections.append(
            Section(
                text=text,
                location=Location(
                    t_start=group.start,
                    t_end=group.end,
                    speaker=group.speaker,
                ),
            )
        )
    return sections


# --------------------------------------------------------------------------
# VTT
# --------------------------------------------------------------------------


class VttParser:
    """Parses WebVTT caption files into ~60-90s sections."""

    suffixes = (".vtt",)

    def parse(self, path: Path, source_path: str) -> ParsedDocument:
        text = _read_text(path)
        title = path.stem.replace("_", " ").replace("-", " ")
        if not text.strip():
            return ParsedDocument(
                source_path=source_path,
                source_type=SourceType.VTT,
                title=title,
                sections=[],
                metadata={"kind": "timed_transcript", "duration_seconds": 0.0},
            )
        try:
            vtt = webvtt.WebVTT.from_string(text)
        except webvtt.errors.MalformedFileError as exc:
            raise ValueError(f"Could not parse VTT file {path}: {exc}") from exc
        cues: list[_Cue] = []
        for caption in vtt.captions:
            speaker, clean_text = _split_voice(caption.raw_text)
            cues.append(
                _Cue(
                    start=_parse_timestamp(caption.start),
                    end=_parse_timestamp(caption.end),
                    speaker=speaker,
                    text=clean_text,
                )
            )
        cues = _dedup_rolling_captions(cues)
        groups = _group_cues(cues)
        sections = _groups_to_sections(groups)
        duration = cues[-1].end if cues else 0.0
        return ParsedDocument(
            source_path=source_path,
            source_type=SourceType.VTT,
            title=title,
            sections=sections,
            metadata={"kind": "timed_transcript", "duration_seconds": duration},
        )


# --------------------------------------------------------------------------
# SRT
# --------------------------------------------------------------------------


class SrtParser:
    """Parses SubRip subtitle files into ~60-90s sections."""

    suffixes = (".srt",)

    def parse(self, path: Path, source_path: str) -> ParsedDocument:
        text = _read_text(path)
        title = path.stem.replace("_", " ").replace("-", " ")
        if not text.strip():
            return ParsedDocument(
                source_path=source_path,
                source_type=SourceType.SRT,
                title=title,
                sections=[],
                metadata={"kind": "timed_transcript", "duration_seconds": 0.0},
            )
        try:
            subtitles = list(srt.parse(text))
        except srt.SRTParseError as exc:
            raise ValueError(f"Could not parse SRT file {path}: {exc}") from exc
        cues: list[_Cue] = []
        for sub in subtitles:
            speaker, clean_text = _split_voice(sub.content or "")
            cues.append(
                _Cue(
                    start=sub.start.total_seconds(),
                    end=sub.end.total_seconds(),
                    speaker=speaker,
                    text=clean_text,
                )
            )
        cues = _dedup_rolling_captions(cues)
        groups = _group_cues(cues)
        sections = _groups_to_sections(groups)
        duration = cues[-1].end if cues else 0.0
        return ParsedDocument(
            source_path=source_path,
            source_type=SourceType.SRT,
            title=title,
            sections=sections,
            metadata={"kind": "timed_transcript", "duration_seconds": duration},
        )


# --------------------------------------------------------------------------
# TXT: speaker-turn transcript detection + disfluency cleanup
# --------------------------------------------------------------------------

_SPEAKER_STYLE_PATTERNS = (
    re.compile(r"^\s*SPEAKER\s+\d+\s*:?\s*$"),
    re.compile(r"^\s*Speaker\s+\d+\s*:"),
)
_GENERIC_HEADING_PATTERN = re.compile(r"^\s*([A-Z][\w .'-]{0,40}):\s*$")
_HEADING_PATTERNS = (*_SPEAKER_STYLE_PATTERNS, _GENERIC_HEADING_PATTERN)

# Note-style labels that must NOT be treated as speaker names, even though
# they match the generic "Name:" heading shape (e.g. study notes with
# "Definition:" / "Example:" / "Note:" lines).
_NOTES_LABEL_DENYLIST = {
    "definition", "example", "examples", "note", "notes", "q", "a",
    "question", "answer", "summary", "theorem", "lemma", "proof",
    "exercise", "solution", "hint", "remark", "tip", "warning",
    "important", "key point", "key points", "recap", "outline",
    "agenda", "objectives", "references", "reading",
}  # fmt: skip


def _is_heading_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return any(p.match(line) for p in _HEADING_PATTERNS)


def _is_speaker_style_heading(line: str) -> bool:
    return any(p.match(line) for p in _SPEAKER_STYLE_PATTERNS)


def _generic_heading_label(line: str) -> str | None:
    match = _GENERIC_HEADING_PATTERN.match(line)
    if not match:
        return None
    return match.group(1).strip()


_FILLER_WORDS = ("uh", "um", "er", "erm", "ah", "mhm", "mm", "mmm")
_FILLER_RE = re.compile(r"\b(?:" + "|".join(_FILLER_WORDS) + r")\b,?", re.IGNORECASE)

_WORD = r"[A-Za-z']+"

# Words a stutter repeat is allowed to collapse across a *removed filler*
# (e.g. "we, uh, we will" -> "we will"). Deliberately narrow: quantifiers
# ("all", "any", "some", "each", "no", "every") and other content words must
# NOT collapse this way, since "after all, uh, all these" is two distinct
# uses of "all", not a stutter.
_CROSS_FILLER_WORDS = {
    # pronouns / possessives
    "i", "we", "you", "they", "he", "she", "it",
    "my", "your", "our", "their",
    # articles
    "a", "an", "the",
    # conjunctions
    "and", "but", "so", "or", "because",
    # auxiliaries / modals
    "is", "are", "was", "were", "will", "can", "may", "might",
    "would", "should", "could", "do", "does", "have", "has",
    # prepositions
    "to", "of", "in", "on", "for", "with", "at", "by",
    # demonstratives (only ever trigger the "same word repeated" rule anyway)
    "this", "that",
}  # fmt: skip


def _stutter_pattern(n: int) -> re.Pattern[str]:
    if n == 1:
        phrase = f"({_WORD})"
    else:
        phrase = "(" + r"\s+".join([_WORD] * n) + ")"
    return re.compile(rf"\b{phrase}\b(?:,?\s+\1\b)+", re.IGNORECASE)


# Pass 1: repeats that are already adjacent in the raw text (only a comma
# and/or whitespace between the repeated words) -- any 1-3 word phrase.
_RAW_STUTTER_PATTERNS = [_stutter_pattern(n) for n in (3, 2, 1)]

# Pass 2: after fillers are removed, repeats that only became adjacent
# because a filler used to sit between them -- restricted to short,
# stutter-prone function words so real repeated content words (like "all")
# are never merged.
_CROSS_FILLER_ALTERNATION = "|".join(
    re.escape(word) for word in sorted(_CROSS_FILLER_WORDS, key=len, reverse=True)
)
_CROSS_FILLER_PATTERN = re.compile(
    rf"\b({_CROSS_FILLER_ALTERNATION})\b(?:,?\s+\1\b)+", re.IGNORECASE
)


def _collapse_repeats(text: str, patterns: list[re.Pattern[str]]) -> str:
    previous = None
    while previous != text:
        previous = text
        for pattern in patterns:
            text = pattern.sub(lambda m: m.group(1), text)
    return text


def _tidy_whitespace_and_punctuation(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*\.", ".", text)
    text = re.sub(r"^[,\s]+", "", text)
    return text


def _capitalize_sentences(text: str) -> str:
    def _cap(match: re.Match[str]) -> str:
        return match.group(1) + match.group(2).upper()

    text = re.sub(r"^(\s*)([a-z])", _cap, text)
    text = re.sub(r"([.!?]\s+)([a-z])", _cap, text)
    return text


def clean_disfluencies(text: str) -> str:
    """Remove standalone filler words and collapse immediate stutter repeats.

    Leaves fillers embedded in other words alone ("umbrella", "sum"), never
    touches purely numeric repeats ("1, 1, 2, 3"), and never merges a real
    repeated content word ("after all, uh, all these") just because removing
    a filler made it adjacent -- only a narrow set of short function words
    ("we, uh, we will", "your, uh, your", "may, um, may") collapse that way.
    """
    # Pass 1: collapse stutters that were already adjacent in the raw text,
    # before any filler is touched (any word/phrase, e.g. "the, the",
    # "I, I, I, I", "told me, told me, told me").
    text = _collapse_repeats(text, _RAW_STUTTER_PATTERNS)
    # Pass 2: remove standalone filler words.
    text = _FILLER_RE.sub("", text)
    text = _tidy_whitespace_and_punctuation(text)
    # Pass 3: collapse repeats that only became adjacent once a filler was
    # removed, restricted to stutter-prone function words.
    text = _collapse_repeats(text, [_CROSS_FILLER_PATTERN])
    text = _tidy_whitespace_and_punctuation(text)
    text = _capitalize_sentences(text.strip())
    return text.strip()


def _split_speaker_turns(text: str) -> list[tuple[str, str]]:
    """Split normalized text into (speaker, raw_turn_text) pairs."""
    lines = text.split("\n")
    turns: list[tuple[str, str]] = []
    current_speaker: str | None = None
    current_lines: list[str] = []
    for line in lines:
        if _is_heading_line(line):
            if current_speaker is not None:
                turns.append(
                    (current_speaker, " ".join(line for line in current_lines if line).strip())
                )
            current_speaker = line.strip().rstrip(":").strip()
            current_lines = []
        else:
            current_lines.append(line.strip())
    if current_speaker is not None:
        turns.append((current_speaker, " ".join(line for line in current_lines if line).strip()))
    return turns


_MIN_GENERIC_TURN_WORDS = 20
_MIN_GENERIC_SWITCH_RATE = 0.6


def _looks_like_speaker_transcript(text: str) -> bool:
    """Detect the Echo360-style "SPEAKER N" transcript shape, or a genuine
    generic "Name:" interview transcript -- while rejecting note-style files
    that happen to use bare "Label:" lines (e.g. "Definition:", "Example:").
    """
    lines = text.split("\n")
    speaker_style_count = sum(1 for line in lines if _is_speaker_style_heading(line))
    if speaker_style_count >= 2:
        return True

    labels = [
        label
        for line in lines
        if (label := _generic_heading_label(line)) is not None
        and label.lower() not in _NOTES_LABEL_DENYLIST
    ]
    if len(labels) < 2:
        return False

    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    distinct_repeated = [label for label, count in counts.items() if count >= 2]
    if len(distinct_repeated) < 2:
        return False

    transitions = len(labels) - 1
    switches = sum(1 for a, b in zip(labels, labels[1:], strict=False) if a != b)
    if transitions <= 0 or switches / transitions < _MIN_GENERIC_SWITCH_RATE:
        return False

    label_set = set(labels)
    turns = _split_speaker_turns(text)
    relevant_turn_texts = [turn_text for speaker, turn_text in turns if speaker in label_set]
    if not relevant_turn_texts:
        return False
    mean_words = sum(len(t.split()) for t in relevant_turn_texts) / len(relevant_turn_texts)
    return mean_words >= _MIN_GENERIC_TURN_WORDS


# --------------------------------------------------------------------------
# TXT: plain text grouping
# --------------------------------------------------------------------------

_MAX_PLAIN_SECTION_CHARS = 1500


def _group_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sections: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
        if buffer and len(candidate) > _MAX_PLAIN_SECTION_CHARS:
            sections.append(buffer)
            buffer = paragraph
        else:
            buffer = candidate
    if buffer:
        sections.append(buffer)
    return sections


class TxtParser:
    """Parses ``.txt`` files as either a speaker-turn transcript (Echo360-style)
    or plain paragraph text, auto-detected from the file's shape.
    """

    suffixes = (".txt",)

    def __init__(self, clean_disfluencies: bool = True) -> None:
        self.clean_disfluencies = clean_disfluencies

    def parse(self, path: Path, source_path: str) -> ParsedDocument:
        raw_text = normalize_whitespace(_read_text(path))
        stem = path.stem
        title = stem.replace("_", " ").replace("-", " ")
        if _looks_like_speaker_transcript(raw_text):
            return self._parse_speaker_transcript(raw_text, path, source_path, title)
        return self._parse_plain_text(raw_text, path, source_path, title)

    def _parse_speaker_transcript(
        self, raw_text: str, path: Path, source_path: str, title: str
    ) -> ParsedDocument:
        raw_turns = _split_speaker_turns(raw_text)

        cumulative_words = 0
        speaker_word_counts: dict[str, int] = {}
        texts: list[str] = []
        speakers: list[str] = []
        approx_minutes: list[float] = []

        for speaker, turn_text in raw_turns:
            notice_stripped = strip_copyright_notice(turn_text).strip()
            if not notice_stripped:
                continue
            raw_word_count = len(notice_stripped.split())
            approx_minutes.append(cumulative_words / SPOKEN_WORDS_PER_MINUTE)
            cumulative_words += raw_word_count
            speaker_word_counts[speaker] = speaker_word_counts.get(speaker, 0) + raw_word_count

            final_text = notice_stripped
            if self.clean_disfluencies:
                final_text = clean_disfluencies(final_text)
            final_text = normalize_whitespace(final_text)

            texts.append(final_text)
            speakers.append(speaker)

        full_text = "\n\n".join(texts)
        sections: list[Section] = []
        offset = 0
        for speaker, text, approx_minute in zip(speakers, texts, approx_minutes, strict=True):
            char_start = offset
            char_end = offset + len(text)
            offset = char_end + 2  # account for the "\n\n" joiner
            sections.append(
                Section(
                    text=text,
                    location=Location(
                        speaker=speaker,
                        approx_minute=approx_minute,
                        char_start=char_start,
                        char_end=char_end,
                    ),
                )
            )

        main_speaker = (
            max(speaker_word_counts, key=lambda s: speaker_word_counts[s])
            if speaker_word_counts
            else None
        )
        metadata: dict[str, object] = {
            "kind": "speaker_transcript",
            "speakers": sorted(speaker_word_counts),
            "main_speaker": main_speaker,
            "approx_duration_min": cumulative_words / SPOKEN_WORDS_PER_MINUTE,
            "full_text_length": len(full_text),
        }
        if "transcript" in path.stem.lower():
            metadata["is_transcript"] = True

        return ParsedDocument(
            source_path=source_path,
            source_type=SourceType.TXT,
            title=title,
            sections=sections,
            metadata=metadata,
        )

    def _parse_plain_text(
        self, raw_text: str, path: Path, source_path: str, title: str
    ) -> ParsedDocument:
        stripped = strip_copyright_notice(raw_text)
        section_texts = _group_paragraphs(stripped)
        full_text = "\n\n".join(section_texts)
        sections = []
        offset = 0
        for text in section_texts:
            char_start = offset
            char_end = offset + len(text)
            offset = char_end + 2
            sections.append(
                Section(text=text, location=Location(char_start=char_start, char_end=char_end))
            )
        return ParsedDocument(
            source_path=source_path,
            source_type=SourceType.TXT,
            title=title,
            sections=sections,
            metadata={"kind": "plain_text", "full_text_length": len(full_text)},
        )


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------

_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


class MarkdownParser:
    """Parses Markdown files into sections split on ATX headings."""

    suffixes = (".md",)

    def parse(self, path: Path, source_path: str) -> ParsedDocument:
        text = _read_text(path)
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

        sections: list[Section] = []
        heading_stack: list[tuple[int, str]] = []  # (level, title)
        current_title: str | None = None
        current_path: list[str] = []
        current_lines: list[str] = []
        in_fence = False
        first_h1: str | None = None

        def flush() -> None:
            content = "\n".join(current_lines).strip()
            if content or current_path:
                sections.append(
                    Section(
                        text=content,
                        title=current_title,
                        location=Location(heading_path=list(current_path)),
                    )
                )

        for line in lines:
            if _FENCE.match(line):
                in_fence = not in_fence
                current_lines.append(line)
                continue
            match = _ATX_HEADING.match(line) if not in_fence else None
            if match:
                flush()
                level = len(match.group(1))
                heading_text = match.group(2).strip()
                if level == 1 and first_h1 is None:
                    first_h1 = heading_text
                heading_stack = [h for h in heading_stack if h[0] < level]
                heading_stack.append((level, heading_text))
                current_title = heading_text
                current_path = [h[1] for h in heading_stack]
                current_lines = []
            else:
                current_lines.append(line)
        flush()

        title = first_h1 or path.stem.replace("_", " ").replace("-", " ")
        return ParsedDocument(
            source_path=source_path,
            source_type=SourceType.MD,
            title=title,
            sections=sections,
            metadata={"kind": "markdown"},
        )


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

register(VttParser())
register(SrtParser())
register(TxtParser())
register(MarkdownParser())
