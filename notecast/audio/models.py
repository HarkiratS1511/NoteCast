"""Data shapes for audio overviews: what's in scope, the key points we must
cover, the time/word budget, the two-host script, and the rendered result.

The pipeline is: scope -> key points (per source) -> ranked key points ->
plan (length + chapters) -> script (chapter by chapter) -> coverage check ->
speech -> MP3 + transcript.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Speaker = Literal["A", "B"]
Tier = Literal["A", "B", "C"]
KeyPointKind = Literal[
    "concept",
    "definition",
    "method",
    "formula",
    "example",
    "caveat",
    "misconception",
    "admin",
]


class AudioScope(BaseModel):
    """Which material an overview covers. None/empty means "no constraint"."""

    weeks: list[int] | None = None
    source_paths: list[str] | None = None
    focus: str | None = None  # optional steer, e.g. "exam prep" or "just the maths"

    def label(self) -> str:
        """A short human label, e.g. "Week 3" or "Weeks 1–3" or "Whole course"."""
        if self.source_paths:
            base = ", ".join(self.source_paths)
        elif self.weeks:
            weeks = sorted(set(self.weeks))
            if len(weeks) == 1:
                base = f"Week {weeks[0]}"
            elif weeks == list(range(weeks[0], weeks[-1] + 1)):
                base = f"Weeks {weeks[0]}–{weeks[-1]}"
            else:
                base = "Weeks " + ", ".join(str(w) for w in weeks)
        else:
            base = "Whole course"
        return f"{base} ({self.focus})" if self.focus else base


class KeyPoint(BaseModel):
    """One thing a student should come away understanding."""

    id: str  # stable within one plan, e.g. "kp-7"
    title: str
    summary: str  # 1-3 sentences: what must be explained
    kind: KeyPointKind = "concept"
    importance: int = Field(3, ge=1, le=5)
    # Why it matters, e.g. "lecturer: 'this will be on the exam'", "slides 12–14", "≈ 45 min".
    evidence: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    in_slides: bool = False
    in_transcript: bool = False


class RankedKeyPoint(KeyPoint):
    """A key point after merging across sources and ranking.

    Tier A = must cover in depth, B = cover briefly, C = mention or skip.
    """

    tier: Tier = "B"
    rank: int = 0  # 1 = most important


class ChapterPlan(BaseModel):
    index: int  # 1-based
    title: str
    point_ids: list[str]
    target_words: int


class AudioPlan(BaseModel):
    scope: AudioScope
    points: list[RankedKeyPoint]
    target_minutes: float
    target_words: int
    chapters: list[ChapterPlan]
    notes: list[str] = Field(default_factory=list)  # e.g. "compressed tier B to fit 45 min"


class ScriptLine(BaseModel):
    speaker: Speaker
    text: str
    source_chunk_ids: list[str] = Field(default_factory=list)


class ChapterScript(BaseModel):
    index: int
    title: str
    lines: list[ScriptLine]

    @property
    def word_count(self) -> int:
        return sum(len(line.text.split()) for line in self.lines)


class CoverageReport(BaseModel):
    covered: list[str] = Field(default_factory=list)  # key point ids
    missing: list[str] = Field(default_factory=list)  # tier A ids still not explained
    patched: list[str] = Field(default_factory=list)  # ids fixed by regenerating a chapter
    notes: list[str] = Field(default_factory=list)


class AudioScript(BaseModel):
    title: str
    plan: AudioPlan
    chapters: list[ChapterScript]
    coverage: CoverageReport = Field(default_factory=CoverageReport)
    est_cost_usd: float | None = None

    @property
    def word_count(self) -> int:
        return sum(chapter.word_count for chapter in self.chapters)

    def est_minutes(self, words_per_minute: int = 150) -> float:
        return self.word_count / words_per_minute


class RenderedChapter(BaseModel):
    index: int
    title: str
    start_seconds: float


class RenderResult(BaseModel):
    mp3_path: Path
    transcript_path: Path
    duration_seconds: float
    chapters: list[RenderedChapter]
