"""Tests for notecast.audio.service, with the Claude-calling pipeline steps
(key points, merge, plan, script, render) replaced by fakes -- no network
access, no reads under notebooks/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import notecast.audio.service as service
from notecast.audio.keypoints import KeyPointRun
from notecast.audio.models import (
    AudioPlan,
    AudioScope,
    AudioScript,
    ChapterPlan,
    ChapterScript,
    CoverageReport,
    KeyPoint,
    RankedKeyPoint,
    RenderedChapter,
    RenderResult,
    ScriptLine,
)
from notecast.audio.service import OverviewResult, generate_overview, render_saved_script
from notecast.chat.pricing import estimate_cost
from notecast.models import Chunk, Location, SourceType
from notecast.notebook import Notebook

# --- Fixtures / fakes ---------------------------------------------------


def _chunk(source_path: str, ordinal: int, week: int, chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        course="test-course",
        source_path=source_path,
        source_type=SourceType.TXT,
        ordinal=ordinal,
        text=f"Some material from {source_path} #{ordinal}.",
        header=f"Week {week} - {source_path}",
        location=Location(),
        week=week,
    )


def _sample_chunks() -> list[Chunk]:
    return [
        _chunk("week-01/a.txt", 0, 1, "c1"),
        _chunk("week-01/a.txt", 1, 1, "c2"),
        _chunk("week-02/b.txt", 0, 2, "c3"),
    ]


class FakeStore:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def all_chunks(self) -> list[Chunk]:
        return self._chunks


class FakeRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.store = FakeStore(chunks)


def _fake_key_point(chunk: Chunk, suffix: str) -> KeyPoint:
    return KeyPoint(
        id=f"kp-{chunk.chunk_id}-{suffix}",
        title=f"Point about {chunk.source_path}",
        summary="A thing a student should know.",
        source_chunk_ids=[chunk.chunk_id],
        in_slides=True,
    )


def _fake_ranked(points_by_source: dict[str, list[KeyPoint]]) -> list[RankedKeyPoint]:
    all_points = [p for points in points_by_source.values() for p in points]
    ranked = []
    for i, point in enumerate(all_points):
        ranked.append(RankedKeyPoint(**point.model_dump(), tier="A" if i == 0 else "B", rank=i + 1))
    return ranked


def _fake_plan(points: list[RankedKeyPoint], scope: AudioScope, chunks: list[Chunk]) -> AudioPlan:
    return AudioPlan(
        scope=scope,
        points=points,
        target_minutes=6.0,
        target_words=900,
        chapters=[
            ChapterPlan(index=1, title="Intro", point_ids=[p.id for p in points], target_words=900)
        ],
    )


def _fake_script(plan: AudioPlan, chunks: list[Chunk], course_name: str) -> AudioScript:
    chunk_ids = [c.chunk_id for c in chunks]
    lines = [
        ScriptLine(speaker="A", text="Welcome to the show.", source_chunk_ids=chunk_ids[:1]),
        ScriptLine(speaker="B", text="Let's get into it.", source_chunk_ids=chunk_ids[1:2]),
    ]
    return AudioScript(
        title=f"{course_name} — {scope_label(plan)}",
        plan=plan,
        chapters=[ChapterScript(index=1, title="Intro", lines=lines)],
        coverage=CoverageReport(covered=[p.id for p in plan.points]),
        est_cost_usd=0.02,
    )


def scope_label(plan: AudioPlan) -> str:
    return plan.scope.label()


def _fake_render_result(out_dir: Path) -> RenderResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    mp3_path = out_dir / "fake.mp3"
    transcript_path = out_dir / "fake.md"
    mp3_path.write_bytes(b"fake-mp3-bytes")
    transcript_path.write_text("# fake transcript\n", encoding="utf-8")
    return RenderResult(
        mp3_path=mp3_path,
        transcript_path=transcript_path,
        duration_seconds=42.0,
        chapters=[RenderedChapter(index=1, title="Intro", start_seconds=0.0)],
    )


class FakePipeline:
    """Bundles patchable fakes for the pipeline steps `generate_overview`
    calls, with call counters so tests can assert on how often each ran.
    """

    def __init__(self) -> None:
        self.extract_calls = 0
        self.merge_calls = 0
        self.plan_calls = 0
        self.write_calls = 0
        self.render_calls = 0

    def extract_key_points(
        self,
        source_path: str,
        chunks: list[Chunk],
        *,
        client: Any,
        settings: Any,
        focus: str | None = None,
        usage: KeyPointRun | None = None,
    ) -> list[KeyPoint]:
        self.extract_calls += 1
        if usage is not None:
            usage.add(
                settings.helper_model,
                {"input_tokens": 100, "output_tokens": 50},
                settings=settings,
            )
        return [_fake_key_point(chunks[0], "x")] if chunks else []

    def merge_and_rank(
        self,
        points_by_source: dict[str, list[KeyPoint]],
        *,
        client: Any,
        settings: Any,
        focus: str | None = None,
        usage: KeyPointRun | None = None,
    ) -> list[RankedKeyPoint]:
        self.merge_calls += 1
        if usage is not None:
            usage.add(
                settings.script_model,
                {"input_tokens": 200, "output_tokens": 100},
                settings=settings,
            )
        return _fake_ranked(points_by_source)

    def plan_audio(
        self, points: list[RankedKeyPoint], scope: AudioScope, settings: Any, chunks: list[Chunk]
    ) -> AudioPlan:
        self.plan_calls += 1
        return _fake_plan(points, scope, chunks)

    def write_script(
        self,
        plan: AudioPlan,
        chunks: list[Chunk],
        *,
        client: Any,
        settings: Any,
        course_name: str = "",
    ) -> AudioScript:
        self.write_calls += 1
        return _fake_script(plan, chunks, course_name)

    def render_script(
        self,
        script: AudioScript,
        out_dir: Path,
        *,
        settings: Any,
        tts: Any = None,
        transcript_markdown: str | None = None,
        progress: Any = None,
    ) -> RenderResult:
        self.render_calls += 1
        if progress is not None:
            progress(1, 1)
        return _fake_render_result(out_dir)


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch) -> FakePipeline:
    fake = FakePipeline()
    monkeypatch.setattr(service, "extract_key_points", fake.extract_key_points)
    monkeypatch.setattr(service, "merge_and_rank", fake.merge_and_rank)
    monkeypatch.setattr(service, "plan_audio", fake.plan_audio)
    monkeypatch.setattr(service, "write_script", fake.write_script)
    monkeypatch.setattr(service, "render_script", fake.render_script)
    return fake


@pytest.fixture
def notebook(tmp_notebooks_root: Path) -> Notebook:
    return Notebook.create("my-course", root=tmp_notebooks_root)


def _patch_retriever(monkeypatch: pytest.MonkeyPatch, chunks: list[Chunk]) -> None:
    monkeypatch.setattr(service, "get_retriever", lambda nb, settings=None: FakeRetriever(chunks))


# --- Tests ---------------------------------------------------------------


def test_stage_progress_reported_in_order(
    notebook: Notebook, pipeline: FakePipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_retriever(monkeypatch, _sample_chunks())
    scope = AudioScope(weeks=[1, 2])

    stages: list[str] = []

    def progress(stage: str, fraction: float) -> None:
        stages.append(stage)
        assert 0.0 <= fraction <= 1.0

    result = generate_overview(notebook, scope, client=object(), progress=progress)

    assert isinstance(result, OverviewResult)
    assert stages[0] == "loading"
    assert stages[-1] == "done"
    assert "selecting" in stages
    assert "key_points" in stages
    assert "merging" in stages
    assert "planning" in stages
    assert "scripting" in stages
    assert "rendering" in stages
    assert pipeline.extract_calls == 2  # two source groups (week-01, week-02)
    assert pipeline.merge_calls == 1
    assert pipeline.plan_calls == 1
    assert pipeline.write_calls == 1
    assert pipeline.render_calls == 1
    assert result.render is not None
    assert result.render.mp3_path.exists()


def test_empty_scope_raises_friendly_error(
    notebook: Notebook, pipeline: FakePipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_retriever(monkeypatch, _sample_chunks())
    scope = AudioScope(weeks=[99])

    with pytest.raises(ValueError, match="No material in"):
        generate_overview(notebook, scope, client=object())

    assert pipeline.extract_calls == 0
    assert pipeline.merge_calls == 0


def test_script_json_saved_and_render_saved_script_skips_pipeline(
    notebook: Notebook, pipeline: FakePipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_retriever(monkeypatch, _sample_chunks())
    scope = AudioScope(weeks=[1])

    result = generate_overview(notebook, scope, client=object(), render=False)

    assert result.render is None
    assert result.script_json_path.exists()
    assert pipeline.render_calls == 0

    calls_before = (
        pipeline.extract_calls,
        pipeline.merge_calls,
        pipeline.plan_calls,
        pipeline.write_calls,
    )

    saved = render_saved_script(notebook, result.script_json_path)

    assert saved.render.mp3_path.exists()
    assert saved.stale_chunk_ids == []
    assert pipeline.render_calls == 1
    assert (
        pipeline.extract_calls,
        pipeline.merge_calls,
        pipeline.plan_calls,
        pipeline.write_calls,
    ) == calls_before


def test_cost_is_summed_from_key_points_and_script(
    notebook: Notebook, pipeline: FakePipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_retriever(monkeypatch, _sample_chunks())
    scope = AudioScope(weeks=[1])

    result = generate_overview(notebook, scope, client=object(), render=False)

    # One source group ("week-01/a.txt") -> one extract_key_points call, plus
    # one merge_and_rank call, plus the fake script's own est_cost_usd (0.02).
    from notecast.chat.pricing import estimate_cost

    expected_kp_cost = (
        estimate_cost("claude-haiku-4-5", {"input_tokens": 100, "output_tokens": 50}) or 0.0
    ) + (estimate_cost("claude-sonnet-5", {"input_tokens": 200, "output_tokens": 100}) or 0.0)
    assert result.est_cost_usd == pytest.approx(expected_kp_cost + 0.02)


def test_cost_is_none_when_agentaus_key_point_calls_are_unpriced(
    notebook: Notebook, pipeline: FakePipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When AgentAUS has no configured prices, `kp_usage` can't be priced
    (`KeyPointRun.all_priced` is False), so the overall result is `None`,
    not just the fake script's own (still non-None) 0.02 -- an unknown
    part of the spend makes the whole total unknown.
    """
    _patch_retriever(monkeypatch, _sample_chunks())
    scope = AudioScope(weeks=[1])
    settings = _agentaus_settings(
        agentaus_price_input_per_mtok=None, agentaus_price_output_per_mtok=None
    )

    result = generate_overview(notebook, scope, client=object(), settings=settings, render=False)

    assert result.est_cost_usd is None


# --- estimate_overview_cost ---------------------------------------------


def test_estimate_overview_cost_returns_a_range(
    notebook: Notebook, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_retriever(monkeypatch, _sample_chunks())
    scope = AudioScope(weeks=[1, 2])

    estimate = service.estimate_overview_cost(notebook, scope)

    assert estimate.n_sources == 2
    assert estimate.total_material_tokens > 0
    assert 0 < estimate.est_cost_low_usd <= estimate.est_cost_usd <= estimate.est_cost_high_usd
    assert estimate.n_chapters_low >= 3
    assert estimate.n_chapters_high <= 8
    assert estimate.n_chapters_low <= estimate.n_chapters_high


def _agentaus_settings(**overrides: Any) -> Any:
    from notecast.config import Settings

    defaults: dict[str, Any] = dict(
        _env_file=None,
        provider="agentaus",
        agentaus_api_key="key",
        agentaus_base_url="https://example.test/v1",
        agentaus_model="trellis-large",
        agentaus_price_input_per_mtok=3.0,
        agentaus_price_output_per_mtok=15.0,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def test_estimate_overview_cost_agentaus_priced_is_positive(
    notebook: Notebook, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_retriever(monkeypatch, _sample_chunks())
    scope = AudioScope(weeks=[1, 2])
    settings = _agentaus_settings()

    estimate = service.estimate_overview_cost(notebook, scope, settings)

    assert 0 < estimate.est_cost_low_usd <= estimate.est_cost_usd <= estimate.est_cost_high_usd


def test_estimate_overview_cost_agentaus_unpriced_is_none(
    notebook: Notebook, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With AgentAUS prices unset, every underlying `estimate_cost(...)`
    call returns `None`, so the whole estimate is `None` -- "n/a" once
    the CLI/UI render it -- rather than a misleading $0.00.
    """
    _patch_retriever(monkeypatch, _sample_chunks())
    scope = AudioScope(weeks=[1, 2])
    settings = _agentaus_settings(
        agentaus_price_input_per_mtok=None, agentaus_price_output_per_mtok=None
    )

    estimate = service.estimate_overview_cost(notebook, scope, settings)

    assert estimate.est_cost_low_usd is None
    assert estimate.est_cost_usd is None
    assert estimate.est_cost_high_usd is None


def test_estimate_overview_cost_empty_scope_raises(
    notebook: Notebook, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_retriever(monkeypatch, _sample_chunks())
    scope = AudioScope(weeks=[99])

    with pytest.raises(ValueError, match="No material in"):
        service.estimate_overview_cost(notebook, scope)


def test_estimate_overview_cost_makes_no_api_calls(
    notebook: Notebook, pipeline: FakePipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_retriever(monkeypatch, _sample_chunks())
    scope = AudioScope(weeks=[1])

    service.estimate_overview_cost(notebook, scope)

    assert pipeline.extract_calls == 0
    assert pipeline.merge_calls == 0
    assert pipeline.plan_calls == 0
    assert pipeline.write_calls == 0
    assert pipeline.render_calls == 0


# --- OverviewFailed (cost-on-failure) ------------------------------------


def test_generate_overview_wraps_failure_with_spend_so_far(
    notebook: Notebook, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_retriever(monkeypatch, _sample_chunks())
    scope = AudioScope(weeks=[1])

    def _spendy_extract(source_path, chunks, *, client, settings, focus=None, usage=None):  # noqa: ANN001
        if usage is not None:
            usage.add("claude-haiku-4-5", {"input_tokens": 1000, "output_tokens": 200})
        raise service.ChatError("boom during extraction")

    monkeypatch.setattr(service, "extract_key_points", _spendy_extract)

    with pytest.raises(service.OverviewFailed) as exc_info:
        generate_overview(notebook, scope, client=object(), render=False)

    err = exc_info.value
    assert err.stage == "key_points"
    assert err.est_cost_usd > 0
    assert "boom during extraction" in str(err)


def test_generate_overview_scripting_failure_includes_partial_script_spend(
    notebook: Notebook, pipeline: FakePipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `write_script` failure that carries `partial_est_cost_usd` (e.g. a
    refusal partway through a multi-chapter episode) should have that spend
    folded into the reported cost, on top of the key-point/merge spend.
    """
    _patch_retriever(monkeypatch, _sample_chunks())
    scope = AudioScope(weeks=[1])

    def _spendy_write_script(plan, chunks, *, client, settings, course_name=""):  # noqa: ANN001
        err = service.ChatError("refused on chapter 3")
        err.partial_est_cost_usd = 0.4321
        raise err

    monkeypatch.setattr(service, "write_script", _spendy_write_script)

    with pytest.raises(service.OverviewFailed) as exc_info:
        generate_overview(notebook, scope, client=object(), render=False)

    err = exc_info.value
    assert err.stage == "scripting"
    # kp_usage spend (extraction + merge) plus the script's partial spend.
    expected_kp_cost = (
        estimate_cost("claude-haiku-4-5", {"input_tokens": 100, "output_tokens": 50}) or 0.0
    ) + (estimate_cost("claude-sonnet-5", {"input_tokens": 200, "output_tokens": 100}) or 0.0)
    assert err.est_cost_usd == pytest.approx(expected_kp_cost + 0.4321)


def test_generate_overview_scripting_refusal_on_chapter_1_keeps_kp_cost_for_anthropic(
    notebook: Notebook, pipeline: FakePipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a *real* `write_script` failing on its very first
    chapter call records no usage at all, so its `partial_est_cost_usd`
    must be 0.0 (see notecast.audio.scriptwriter._estimate_total_cost) --
    not `None`, which would wrongly wipe out the already-known,
    already-priced key-point/merge spend for an ordinary Anthropic run.
    """
    import notecast.audio.scriptwriter as scriptwriter_module

    # Let the real write_script run (instead of the pipeline fixture's
    # fake), against a fake Anthropic client that refuses chapter 1.
    monkeypatch.setattr(service, "write_script", scriptwriter_module.write_script)

    class _FakeStreamContext:
        def __init__(self, message: dict) -> None:
            self._message = message

        def __enter__(self) -> _FakeStreamContext:
            return self

        def __exit__(self, *exc: Any) -> bool:
            return False

        def get_final_message(self) -> dict:
            return self._message

    class _FakeMessages:
        def stream(self, **kwargs: Any) -> _FakeStreamContext:
            return _FakeStreamContext(
                {
                    "content": [],
                    "stop_reason": "refusal",
                    "stop_details": {"category": "policy"},
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                }
            )

        def create(self, **kwargs: Any) -> dict:  # pragma: no cover - coverage check unreachable
            raise AssertionError("coverage check should not run: chapter 1 already failed")

    class _FakeClient:
        def __init__(self) -> None:
            self.messages = _FakeMessages()

    _patch_retriever(monkeypatch, _sample_chunks())
    scope = AudioScope(weeks=[1])

    with pytest.raises(service.OverviewFailed) as exc_info:
        generate_overview(notebook, scope, client=_FakeClient(), render=False)

    err = exc_info.value
    assert err.stage == "scripting"
    # Only key-point/merge spend -- write_script itself spent nothing
    # (partial_est_cost_usd == 0.0, not None) since chapter 1 refused
    # before any usage was recorded.
    expected_kp_cost = (
        estimate_cost("claude-haiku-4-5", {"input_tokens": 100, "output_tokens": 50}) or 0.0
    ) + (estimate_cost("claude-sonnet-5", {"input_tokens": 200, "output_tokens": 100}) or 0.0)
    assert err.est_cost_usd == pytest.approx(expected_kp_cost)


def test_generate_overview_failure_est_cost_is_none_when_agentaus_unpriced(
    notebook: Notebook, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure's `est_cost_usd` should be `None`, not 0.0, when the spend
    so far genuinely can't be priced (AgentAUS with no configured prices) --
    0.0 would wrongly claim the run so far was free.
    """
    _patch_retriever(monkeypatch, _sample_chunks())
    scope = AudioScope(weeks=[1])
    settings = _agentaus_settings(
        agentaus_price_input_per_mtok=None, agentaus_price_output_per_mtok=None
    )

    def _spendy_extract(source_path, chunks, *, client, settings, focus=None, usage=None):  # noqa: ANN001
        if usage is not None:
            usage.add(
                settings.helper_model,
                {"input_tokens": 1000, "output_tokens": 200},
                settings=settings,
            )
        raise service.ChatError("boom during extraction")

    monkeypatch.setattr(service, "extract_key_points", _spendy_extract)

    with pytest.raises(service.OverviewFailed) as exc_info:
        generate_overview(notebook, scope, client=object(), settings=settings, render=False)

    err = exc_info.value
    assert err.stage == "key_points"
    assert err.est_cost_usd is None


def test_generate_overview_scripting_failure_without_partial_cost_defaults_to_zero(
    notebook: Notebook, pipeline: FakePipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plain ChatError without `partial_est_cost_usd` (e.g. from code that
    hasn't been updated to attach it) shouldn't blow up -- it just falls
    back to no extra spend on top of the key-point/merge cost.
    """
    _patch_retriever(monkeypatch, _sample_chunks())
    scope = AudioScope(weeks=[1])

    def _plain_write_script(plan, chunks, *, client, settings, course_name=""):  # noqa: ANN001
        raise service.ChatError("no partial cost attached")

    monkeypatch.setattr(service, "write_script", _plain_write_script)

    with pytest.raises(service.OverviewFailed) as exc_info:
        generate_overview(notebook, scope, client=object(), render=False)

    err = exc_info.value
    assert err.stage == "scripting"
    expected_kp_cost = (
        estimate_cost("claude-haiku-4-5", {"input_tokens": 100, "output_tokens": 50}) or 0.0
    ) + (estimate_cost("claude-sonnet-5", {"input_tokens": 200, "output_tokens": 100}) or 0.0)
    assert err.est_cost_usd == pytest.approx(expected_kp_cost)


def test_generate_overview_failure_before_any_spend_is_not_wrapped(
    notebook: Notebook, pipeline: FakePipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An empty scope fails before any API call is made, so it's a plain
    # ValueError, not an OverviewFailed.
    _patch_retriever(monkeypatch, _sample_chunks())
    scope = AudioScope(weeks=[99])

    with pytest.raises(ValueError) as exc_info:
        generate_overview(notebook, scope, client=object(), render=False)

    assert not isinstance(exc_info.value, service.OverviewFailed)


# --- Timestamped script filenames ----------------------------------------


def test_script_json_filename_includes_timestamp(
    notebook: Notebook, pipeline: FakePipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    import re

    _patch_retriever(monkeypatch, _sample_chunks())
    scope = AudioScope(weeks=[1])

    result = generate_overview(notebook, scope, client=object(), render=False)

    assert re.search(r"-\d{8}-\d{4}\.script\.json$", result.script_json_path.name)


# --- Stale chunk detection in render_saved_script -------------------------


def test_render_saved_script_reports_stale_chunk_ids(
    notebook: Notebook, pipeline: FakePipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = _sample_chunks()
    _patch_retriever(monkeypatch, chunks)
    scope = AudioScope(weeks=[1])

    result = generate_overview(notebook, scope, client=object(), render=False)

    # Re-ingest: the chunk the fake script cited ("c1") is now gone.
    _patch_retriever(monkeypatch, [c for c in chunks if c.chunk_id != "c1"])

    saved = render_saved_script(notebook, result.script_json_path)

    assert saved.stale_chunk_ids == ["c1"]
