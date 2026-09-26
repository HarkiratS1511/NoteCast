"""Data shapes for grounded chat: what a question/answer round trip looks
like, with citations mapped back to exact source locations.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from notecast.models import SearchHit

ChatMode = Literal["sources", "open"]


class Citation(BaseModel):
    """One citation attached to part of an answer: either a course chunk
    (with an exact file/slide/timestamp location) or a web search result.
    """

    n: int
    kind: Literal["course", "web"]
    cited_text: str = ""

    # Course citations
    chunk_id: str | None = None
    source_path: str | None = None
    location_label: str | None = None
    header: str | None = None
    week: int | None = None

    # Web citations
    url: str | None = None
    title: str | None = None


class AnswerSegment(BaseModel):
    """One piece of the answer's text, plus which citation numbers (if any)
    apply to it.
    """

    text: str
    citation_numbers: list[int] = Field(default_factory=list)


class Usage(BaseModel):
    """Token/cost accounting for one chat turn."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    web_searches: int = 0
    est_cost_usd: float | None = None


class ChatAnswer(BaseModel):
    """The full result of one `ChatSession.ask()` call."""

    question: str
    standalone_query: str
    mode: ChatMode
    segments: list[AnswerSegment] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    not_in_sources: bool = False
    hits: list[SearchHit] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    stop_reason: str | None = None

    @property
    def text(self) -> str:
        """The answer text with citation markers (e.g. "[1][2]") appended
        after each segment that cites something.
        """
        parts: list[str] = []
        for segment in self.segments:
            piece = segment.text
            if segment.citation_numbers:
                markers = "".join(f"[{n}]" for n in segment.citation_numbers)
                piece = f"{piece}{markers}"
            parts.append(piece)
        return "".join(parts)

    def citations_markdown(self) -> str:
        """A numbered source list suitable for rendering under the answer."""
        lines: list[str] = []
        for citation in sorted(self.citations, key=lambda c: c.n):
            if citation.kind == "course":
                location = f" — {citation.location_label}" if citation.location_label else ""
                lines.append(f"[{citation.n}] {citation.source_path}{location}")
            else:
                title = citation.title or citation.url or ""
                url = f" — {citation.url}" if citation.url else ""
                lines.append(f"[{citation.n}] {title}{url}")
        return "\n".join(lines)
