"""Retrieval evaluation: load a YAML eval set of question/expected-source
cases and score any retriever's `search()` against it.

This module depends only on a duck-typed retriever: any object with

    search(query: str, k: int = 10, filters: SearchFilters | None = None) -> list[SearchHit]

It does not know or care whether that retriever is hybrid search, a vector
store, or a fake used in tests.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any, Protocol

import yaml
from pydantic import BaseModel, Field, model_validator

from notecast.models import Chunk, SearchFilters, SearchHit


class Retriever(Protocol):
    """Structural type for anything retrieval eval can run against."""

    def search(
        self, query: str, k: int = 10, filters: SearchFilters | None = None
    ) -> list[SearchHit]: ...


class ExpectedSource(BaseModel):
    """One acceptable place the answer to a case's question can live.

    A chunk matches this expectation when its `source_path` matches the
    `source` glob AND every constraint provided (slides/pages/contains/
    minutes) holds for that chunk.
    """

    source: str
    slides: list[int] | None = None
    pages: list[int] | None = None
    contains: list[str] | None = None
    minutes: tuple[float, float] | None = None

    def matches(self, chunk: Chunk) -> bool:
        if not fnmatch.fnmatch(chunk.source_path, self.source):
            return False
        if self.slides is not None and chunk.location.slide not in self.slides:
            return False
        if self.pages is not None and chunk.location.page not in self.pages:
            return False
        if self.contains is not None:
            text_lower = chunk.text.lower()
            if not any(phrase.lower() in text_lower for phrase in self.contains):
                return False
        if self.minutes is not None:
            approx = chunk.location.approx_minute
            if approx is None:
                return False
            lo, hi = self.minutes
            if not (lo <= approx <= hi):
                return False
        return True


class EvalCase(BaseModel):
    """One question in an eval set, with the sources that should answer it."""

    id: str
    question: str
    answerable: bool = True
    weeks: list[int] | None = None
    expect: list[ExpectedSource] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_expect(self) -> EvalCase:
        if self.answerable and not self.expect:
            raise ValueError(f"eval case {self.id!r} is answerable but has no `expect` entries")
        return self


def _friendly_error(index: int, raw: Any, exc: Exception) -> ValueError:
    case_id = raw.get("id") if isinstance(raw, dict) else None
    label = f"case {case_id!r} (index {index})" if case_id else f"case at index {index}"
    return ValueError(f"invalid eval {label}: {exc}")


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    """Load and validate an eval YAML file into a list of `EvalCase`.

    Raises a friendly `ValueError` naming the offending case's id (or index,
    if it has none) on any bad entry, and rejects duplicate ids.
    """
    path = Path(path)
    raw_text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw_text)
    if data is None:
        data = []
    if not isinstance(data, list):
        raise ValueError(f"eval file {path} must contain a YAML list of cases, got {type(data)}")

    cases: list[EvalCase] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(data):
        try:
            if not isinstance(raw, dict):
                raise TypeError(f"expected a mapping, got {type(raw)}")
            case = EvalCase.model_validate(raw)
        except Exception as exc:  # noqa: BLE001 -- re-raised as a friendly ValueError
            raise _friendly_error(index, raw, exc) from exc
        if case.id in seen_ids:
            raise ValueError(f"duplicate eval case id {case.id!r} (index {index})")
        seen_ids.add(case.id)
        cases.append(case)
    return cases


class CaseResult(BaseModel):
    """The outcome of running one eval case against a retriever."""

    case_id: str
    question: str
    rank: int | None = None
    matched_expect_index: int | None = None
    top_hits: list[str] = Field(default_factory=list)

    @property
    def hit(self) -> bool:
        return self.rank is not None


class RetrievalEvalReport(BaseModel):
    """Aggregate metrics plus per-case results for one eval run."""

    k: int
    count: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    hit_at_k: float
    mrr_at_k: float
    results: list[CaseResult] = Field(default_factory=list)


def _hit_label(hit: SearchHit) -> str:
    label = hit.chunk.location.label()
    if label:
        return f"{hit.chunk.source_path} ({label})"
    return hit.chunk.source_path


def _first_match_rank(
    hits: list[SearchHit], expect: list[ExpectedSource]
) -> tuple[int | None, int | None]:
    """Return (1-based rank, expect-index) of the first hit matching any
    expectation, or (None, None) if none match.
    """
    for rank, hit in enumerate(hits, start=1):
        for expect_index, expectation in enumerate(expect):
            if expectation.matches(hit.chunk):
                return rank, expect_index
    return None, None


def run_retrieval_eval(
    retriever: Retriever, cases: list[EvalCase], *, k: int = 10
) -> RetrievalEvalReport:
    """Run every answerable case's question through `retriever.search()` and
    score where (if at all) a matching chunk shows up in the top k.
    """
    results: list[CaseResult] = []
    for case in cases:
        if not case.answerable:
            continue
        filters = SearchFilters(weeks=case.weeks) if case.weeks else None
        hits = retriever.search(case.question, k=k, filters=filters)
        rank, expect_index = _first_match_rank(hits, case.expect)
        results.append(
            CaseResult(
                case_id=case.id,
                question=case.question,
                rank=rank,
                matched_expect_index=expect_index,
                top_hits=[_hit_label(hit) for hit in hits[:3]],
            )
        )

    count = len(results)

    def _hit_rate(cutoff: int) -> float:
        if count == 0:
            return 0.0
        hits_within = sum(1 for r in results if r.rank is not None and r.rank <= cutoff)
        return hits_within / count

    def _mrr() -> float:
        if count == 0:
            return 0.0
        total = sum(1.0 / r.rank for r in results if r.rank is not None)
        return total / count

    return RetrievalEvalReport(
        k=k,
        count=count,
        hit_at_1=_hit_rate(1),
        hit_at_3=_hit_rate(3),
        hit_at_5=_hit_rate(5),
        hit_at_k=_hit_rate(k),
        mrr_at_k=_mrr(),
        results=results,
    )


def format_report(report: RetrievalEvalReport) -> str:
    """A readable plain-text table: one row per case, then a summary line."""
    lines: list[str] = []
    id_width = max((len(r.case_id) for r in report.results), default=2)
    header = f"{'id':<{id_width}}  {'rank':>4}  top hit"
    lines.append(header)
    lines.append("-" * len(header))
    for r in report.results:
        rank_str = str(r.rank) if r.rank is not None else "miss"
        top_hit = r.top_hits[0] if r.top_hits else "-"
        lines.append(f"{r.case_id:<{id_width}}  {rank_str:>4}  {top_hit}")
    lines.append("")
    lines.append(
        "summary: "
        f"n={report.count} "
        f"hit@1={report.hit_at_1:.2f} "
        f"hit@3={report.hit_at_3:.2f} "
        f"hit@5={report.hit_at_5:.2f} "
        f"hit@{report.k}={report.hit_at_k:.2f} "
        f"mrr@{report.k}={report.mrr_at_k:.2f}"
    )
    return "\n".join(lines)
