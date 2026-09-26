"""Tests for the retrieval evaluation harness. Everything here is synthetic:
a fake retriever with scripted hits, and small YAML fixtures written to a
tmp_path. No network, no real course material.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from notecast.eval.retrieval import (
    EvalCase,
    ExpectedSource,
    format_report,
    load_eval_cases,
    run_retrieval_eval,
)
from notecast.models import Chunk, Location, SearchFilters, SearchHit


def make_chunk(
    source_path: str,
    text: str,
    *,
    slide: int | None = None,
    page: int | None = None,
    approx_minute: float | None = None,
    week: int | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=Chunk.make_id("course", source_path, 0, text),
        course="course",
        source_path=source_path,
        source_type="pdf" if source_path.endswith(".pdf") else "txt",
        ordinal=0,
        text=text,
        location=Location(slide=slide, page=page, approx_minute=approx_minute),
        week=week,
    )


class FakeRetriever:
    """A retriever with a scripted answer per query, ignoring the actual
    text passed and just returning whatever the test configured.
    """

    def __init__(self, script: dict[str, list[SearchHit]]) -> None:
        self.script = script
        self.calls: list[tuple[str, int, SearchFilters | None]] = []

    def search(
        self, query: str, k: int = 10, filters: SearchFilters | None = None
    ) -> list[SearchHit]:
        self.calls.append((query, k, filters))
        return self.script.get(query, [])[:k]


# --- loading + validation -------------------------------------------------


def test_load_eval_cases_basic(tmp_path: Path) -> None:
    yaml_text = """
- id: case-a
  question: "What is an inverted index?"
  weeks: [2]
  expect:
    - source: "week-02/*.pdf"
      slides: [20, 21]
- id: case-b
  question: "Not in the material"
  answerable: false
  expect: []
"""
    path = tmp_path / "eval.yaml"
    path.write_text(yaml_text)
    cases = load_eval_cases(path)
    assert len(cases) == 2
    assert cases[0].id == "case-a"
    assert cases[0].answerable is True
    assert cases[0].weeks == [2]
    assert cases[1].answerable is False
    assert cases[1].expect == []


def test_load_eval_cases_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert load_eval_cases(path) == []


def test_load_eval_cases_not_a_list(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("id: not-a-list\n")
    with pytest.raises(ValueError, match="YAML list"):
        load_eval_cases(path)


def test_load_eval_cases_missing_required_field(tmp_path: Path) -> None:
    yaml_text = """
- id: broken
  expect: []
"""
    path = tmp_path / "eval.yaml"
    path.write_text(yaml_text)
    with pytest.raises(ValueError, match="broken"):
        load_eval_cases(path)


def test_load_eval_cases_answerable_needs_expect(tmp_path: Path) -> None:
    yaml_text = """
- id: no-expect
  question: "What?"
"""
    path = tmp_path / "eval.yaml"
    path.write_text(yaml_text)
    with pytest.raises(ValueError, match="no-expect"):
        load_eval_cases(path)


def test_load_eval_cases_duplicate_ids(tmp_path: Path) -> None:
    yaml_text = """
- id: dup
  question: "One"
  expect:
    - source: "*.pdf"
- id: dup
  question: "Two"
  expect:
    - source: "*.pdf"
"""
    path = tmp_path / "eval.yaml"
    path.write_text(yaml_text)
    with pytest.raises(ValueError, match="duplicate"):
        load_eval_cases(path)


def test_load_eval_cases_bad_entry_uses_index_when_no_id(tmp_path: Path) -> None:
    yaml_text = """
- question: "No id here"
  expect:
    - source: "*.pdf"
"""
    path = tmp_path / "eval.yaml"
    path.write_text(yaml_text)
    with pytest.raises(ValueError, match="index 0"):
        load_eval_cases(path)


# --- matching rules --------------------------------------------------------


def test_expected_source_matches_slides() -> None:
    expect = ExpectedSource(source="week-02/*.pdf", slides=[20, 21])
    hit = make_chunk("week-02/slides.pdf", "Inverted index text", slide=20)
    miss = make_chunk("week-02/slides.pdf", "Inverted index text", slide=5)
    other_source = make_chunk("week-03/slides.pdf", "Inverted index text", slide=20)
    assert expect.matches(hit)
    assert not expect.matches(miss)
    assert not expect.matches(other_source)


def test_expected_source_matches_merged_slide_range() -> None:
    expect = ExpectedSource(source="week-02/*.pdf", slides=[6])
    chunk = Chunk(
        chunk_id=Chunk.make_id("course", "week-02/x.pdf", 0, "text"),
        course="course",
        source_path="week-02/x.pdf",
        source_type="pdf",
        ordinal=0,
        text="text",
        location=Location(slide=4, slide_end=6),
    )
    assert expect.matches(chunk)


def test_expected_source_rejects_reversed_minutes() -> None:
    with pytest.raises(ValueError, match="minutes range"):
        ExpectedSource(source="*", minutes=(20.0, 10.0))


def test_expected_source_matches_pages() -> None:
    expect = ExpectedSource(source="*.pdf", pages=[3, 4])
    assert expect.matches(make_chunk("x.pdf", "text", page=3))
    assert not expect.matches(make_chunk("x.pdf", "text", page=5))


def test_expected_source_matches_contains_any_of() -> None:
    expect = ExpectedSource(source="*", contains=["add one", "Laplace"])
    assert expect.matches(make_chunk("t.txt", "This covers Laplace smoothing"))
    assert expect.matches(make_chunk("t.txt", "We add one to every count"))
    assert not expect.matches(make_chunk("t.txt", "Backoff and interpolation"))


def test_expected_source_matches_minutes_range() -> None:
    expect = ExpectedSource(source="*", minutes=(10.0, 20.0))
    assert expect.matches(make_chunk("t.txt", "text", approx_minute=15.0))
    assert not expect.matches(make_chunk("t.txt", "text", approx_minute=25.0))
    assert not expect.matches(make_chunk("t.txt", "text", approx_minute=None))


def test_expected_source_requires_all_constraints() -> None:
    expect = ExpectedSource(source="*.pdf", slides=[5], contains=["smoothing"])
    ok = make_chunk("x.pdf", "About smoothing", slide=5)
    wrong_slide = make_chunk("x.pdf", "About smoothing", slide=6)
    wrong_text = make_chunk("x.pdf", "About backoff", slide=5)
    assert expect.matches(ok)
    assert not expect.matches(wrong_slide)
    assert not expect.matches(wrong_text)


def test_expected_source_glob_source() -> None:
    expect = ExpectedSource(source="week-03/*transcript*")
    assert expect.matches(make_chunk("week-03/lecture-transcript.txt", "hi"))
    assert not expect.matches(make_chunk("week-03/slides.pdf", "hi"))


# --- rank computation + metrics --------------------------------------------


def _case(case_id: str, expect: list[ExpectedSource], weeks: list[int] | None = None) -> EvalCase:
    return EvalCase(id=case_id, question=f"question for {case_id}", weeks=weeks, expect=expect)


def test_run_retrieval_eval_rank_and_metrics() -> None:
    hit_chunk = make_chunk("week-02/slides.pdf", "Inverted index", slide=20)
    other_chunk = make_chunk("week-02/slides.pdf", "Something else", slide=1)
    miss_only_chunk = make_chunk("week-02/slides.pdf", "Totally unrelated", slide=99)

    cases = [
        _case("first-place", [ExpectedSource(source="*.pdf", slides=[20])]),
        _case("third-place", [ExpectedSource(source="*.pdf", slides=[20])]),
        _case("no-hit", [ExpectedSource(source="*.pdf", slides=[20])]),
    ]

    retriever = FakeRetriever(
        {
            "question for first-place": [SearchHit(chunk=hit_chunk, score=0.9)],
            "question for third-place": [
                SearchHit(chunk=other_chunk, score=0.9),
                SearchHit(chunk=other_chunk, score=0.8),
                SearchHit(chunk=hit_chunk, score=0.7),
            ],
            "question for no-hit": [SearchHit(chunk=miss_only_chunk, score=0.5)],
        }
    )

    report = run_retrieval_eval(retriever, cases, k=10)

    assert report.count == 3
    by_id = {r.case_id: r for r in report.results}
    assert by_id["first-place"].rank == 1
    assert by_id["third-place"].rank == 3
    assert by_id["no-hit"].rank is None

    assert report.hit_at_1 == pytest.approx(1 / 3)
    assert report.hit_at_3 == pytest.approx(2 / 3)
    assert report.hit_at_5 == pytest.approx(2 / 3)
    assert report.hit_at_k == pytest.approx(2 / 3)
    assert report.mrr_at_k == pytest.approx((1.0 + (1 / 3) + 0.0) / 3)


def test_run_retrieval_eval_respects_k_truncation() -> None:
    hit_chunk = make_chunk("x.pdf", "Match", slide=1)
    filler = make_chunk("x.pdf", "Filler", slide=2)
    cases = [_case("late-hit", [ExpectedSource(source="*.pdf", slides=[1])])]
    retriever = FakeRetriever(
        {
            "question for late-hit": [
                SearchHit(chunk=filler, score=0.5),
                SearchHit(chunk=filler, score=0.4),
                SearchHit(chunk=filler, score=0.3),
                SearchHit(chunk=hit_chunk, score=0.1),
            ]
        }
    )
    report = run_retrieval_eval(retriever, cases, k=3)
    assert report.results[0].rank is None  # hit is at position 4, beyond k=3


def test_run_retrieval_eval_skips_unanswerable_cases() -> None:
    cases = [
        EvalCase(id="answerable", question="q1", expect=[ExpectedSource(source="*.pdf")]),
        EvalCase(id="unanswerable", question="q2", answerable=False, expect=[]),
    ]
    retriever = FakeRetriever({"q1": [SearchHit(chunk=make_chunk("a.pdf", "x"), score=1.0)]})
    report = run_retrieval_eval(retriever, cases, k=5)
    assert report.count == 1
    assert report.results[0].case_id == "answerable"
    # The unanswerable case's question must never have been searched.
    assert all(query != "q2" for query, _, _ in retriever.calls)


def test_run_retrieval_eval_passes_weeks_filter_through() -> None:
    cases = [_case("with-weeks", [ExpectedSource(source="*.pdf")], weeks=[2, 3])]
    no_weeks_case = _case("no-weeks", [ExpectedSource(source="*.pdf")])
    retriever = FakeRetriever(
        {
            "question for with-weeks": [SearchHit(chunk=make_chunk("a.pdf", "x"), score=1.0)],
            "question for no-weeks": [SearchHit(chunk=make_chunk("a.pdf", "x"), score=1.0)],
        }
    )
    run_retrieval_eval(retriever, [cases[0], no_weeks_case], k=5)
    calls_by_query = {query: filters for query, _, filters in retriever.calls}
    assert calls_by_query["question for with-weeks"] == SearchFilters(weeks=[2, 3])
    assert calls_by_query["question for no-weeks"] is None


def test_run_retrieval_eval_records_matched_expect_index_and_top_hits() -> None:
    hit_chunk = make_chunk("week-03/transcript.txt", "Laplace smoothing add one", slide=None)
    cases = [
        _case(
            "matched-second-expect",
            [
                ExpectedSource(source="*.pdf"),
                ExpectedSource(source="*transcript*", contains=["Laplace"]),
            ],
        )
    ]
    retriever = FakeRetriever(
        {"question for matched-second-expect": [SearchHit(chunk=hit_chunk, score=0.9)]}
    )
    report = run_retrieval_eval(retriever, cases, k=5)
    result = report.results[0]
    assert result.rank == 1
    assert result.matched_expect_index == 1
    assert result.top_hits == ["week-03/transcript.txt"]


def test_run_retrieval_eval_empty_cases_returns_zeroed_report() -> None:
    report = run_retrieval_eval(FakeRetriever({}), [], k=5)
    assert report.count == 0
    assert report.hit_at_1 == 0.0
    assert report.mrr_at_k == 0.0
    assert report.results == []


# --- formatting --------------------------------------------------------


def test_format_report_smoke() -> None:
    cases = [_case("a", [ExpectedSource(source="*.pdf", slides=[1])])]
    hit_chunk = make_chunk("x.pdf", "text", slide=1)
    retriever = FakeRetriever({"question for a": [SearchHit(chunk=hit_chunk, score=0.9)]})
    report = run_retrieval_eval(retriever, cases, k=5)
    text = format_report(report)
    assert "a" in text
    assert "1" in text
    assert "summary:" in text
    assert "hit@1=" in text
    assert "mrr@5=" in text


def test_format_report_shows_miss() -> None:
    cases = [_case("nothing-found", [ExpectedSource(source="*.pdf", slides=[1])])]
    retriever = FakeRetriever({})
    report = run_retrieval_eval(retriever, cases, k=5)
    text = format_report(report)
    assert "miss" in text
