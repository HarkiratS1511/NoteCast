"""Core data shapes shared across NoteCast: what kind of file something is,
where a piece of text came from in that file, and the chunks we search over.
These are plain pydantic models with no external dependencies, so every
later phase (parsers, embedder, store, chat) can import them safely.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SourceType(str, Enum):  # noqa: UP042 -- kept as str+Enum for JSON/CLI-friendly values
    """The kind of file a source document is, inferred from its extension."""

    PDF = "pdf"
    PPTX = "pptx"
    DOCX = "docx"
    VTT = "vtt"
    SRT = "srt"
    TXT = "txt"
    MD = "md"

    @classmethod
    def from_path(cls, path: Path) -> SourceType:
        """Infer the source type from a file's extension (case-insensitive)."""
        suffix = path.suffix.lower().lstrip(".")
        for member in cls:
            if member.value == suffix:
                return member
        raise ValueError(f"Unsupported file extension: {path.suffix!r} (from {path})")


def _format_timestamp(seconds: float) -> str:
    """Format seconds as mm:ss, or h:mm:ss once the duration reaches an hour."""
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class Location(BaseModel):
    """Where a section or chunk sits within its source: a page, a slide, a
    place in a document's heading structure, or a time range in a transcript.
    """

    page: int | None = None
    slide: int | None = None
    heading_path: list[str] = Field(default_factory=list)
    t_start: float | None = None
    t_end: float | None = None

    def label(self) -> str:
        """A short human-readable label, e.g. "p. 12", "slide 4",
        "14:32-15:40", or a heading path — whichever parts are set.
        """
        parts: list[str] = []
        if self.page is not None:
            parts.append(f"p. {self.page}")
        if self.slide is not None:
            parts.append(f"slide {self.slide}")
        if self.heading_path:
            parts.append(" › ".join(self.heading_path))
        if self.t_start is not None:
            start = _format_timestamp(self.t_start)
            if self.t_end is not None:
                parts.append(f"{start}–{_format_timestamp(self.t_end)}")
            else:
                parts.append(start)
        return ", ".join(parts)


class Section(BaseModel):
    """One natural unit of a parsed document: a page, a slide, a group of
    transcript cues, or a heading section. Parsers emit a list of these.
    """

    text: str
    location: Location = Field(default_factory=Location)
    title: str | None = None


class ParsedDocument(BaseModel):
    """The output of parsing one source file: its sections, ready to chunk."""

    source_path: str
    source_type: SourceType
    title: str | None = None
    sections: list[Section]
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable piece of text, with enough metadata to cite it back to
    an exact place in a source file.
    """

    chunk_id: str
    course: str
    source_path: str
    source_type: SourceType
    ordinal: int
    text: str
    header: str = ""
    location: Location = Field(default_factory=Location)
    week: int | None = None
    topic: str | None = None

    @property
    def embed_text(self) -> str:
        """The text actually sent to the embedding model: the contextual
        header (if any) followed by a blank line, then the chunk's text.
        """
        if self.header:
            return f"{self.header}\n\n{self.text}"
        return self.text

    @staticmethod
    def make_id(course: str, source_path: str, ordinal: int, text: str) -> str:
        """A stable, content-sensitive chunk id: the first 16 hex characters
        of a sha256 hash over (course, source_path, ordinal, text).
        """
        joined = "\x1f".join([course, source_path, str(ordinal), text])
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


class SearchHit(BaseModel):
    """One retrieval result: a chunk and its similarity/relevance score."""

    chunk: Chunk
    score: float


class SearchFilters(BaseModel):
    """Optional constraints narrowing a search to specific weeks, source
    types, or specific files.
    """

    weeks: list[int] | None = None
    source_types: list[SourceType] | None = None
    source_paths: list[str] | None = None
