"""Tests for the notecast CLI, using Typer's CliRunner against a temporary
notebooks root (never the real notebooks/ directory).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from notecast import __version__
from notecast.cli import app
from notecast.config import get_settings
from notecast.notebook import Notebook

runner = CliRunner()


@pytest.fixture(autouse=True)
def _hashing_embedder(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every CLI test uses the offline hashing embedder backend, never a
    real (network/GPU-touching) fastembed model.
    """
    monkeypatch.setenv("NOTECAST_EMBEDDING_BACKEND", "hashing")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_notebooks_list_empty(tmp_notebooks_root: Path) -> None:
    result = runner.invoke(app, ["notebooks", "list"])
    assert result.exit_code == 0
    assert "No notebooks yet" in result.output


def test_notebooks_create_and_list(tmp_notebooks_root: Path) -> None:
    create_result = runner.invoke(app, ["notebooks", "create", "my-course"])
    assert create_result.exit_code == 0
    assert "my-course" in create_result.output
    assert str(tmp_notebooks_root / "my-course" / "sources") in create_result.output

    (tmp_notebooks_root / "my-course" / "sources" / "a.txt").write_text("hello")

    list_result = runner.invoke(app, ["notebooks", "list"])
    assert list_result.exit_code == 0
    assert "my-course" in list_result.output
    assert "1 source file" in list_result.output


def test_notebooks_create_invalid_slug(tmp_notebooks_root: Path) -> None:
    result = runner.invoke(app, ["notebooks", "create", "Bad Slug"])
    assert result.exit_code != 0
    assert "traceback" not in result.output.lower()


def test_notebooks_create_duplicate(tmp_notebooks_root: Path) -> None:
    runner.invoke(app, ["notebooks", "create", "my-course"])
    result = runner.invoke(app, ["notebooks", "create", "my-course"])
    assert result.exit_code != 0
    assert "already exists" in result.output


def test_ingest_missing_notebook(tmp_notebooks_root: Path) -> None:
    result = runner.invoke(app, ["ingest", "my-course"])
    assert result.exit_code == 1
    assert "No notebook named" in result.output


def test_ingest_empty_notebook(tmp_notebooks_root: Path) -> None:
    runner.invoke(app, ["notebooks", "create", "my-course"])
    result = runner.invoke(app, ["ingest", "my-course"])
    assert result.exit_code == 0
    assert "Added 0, updated 0, unchanged 0, removed 0, failed 0." in result.output
    assert "indexed 0 chunks (total 0 in index)" in result.output


def test_ingest_summary_indexes_chunks(tmp_notebooks_root: Path) -> None:
    notebook = Notebook.create("my-course", root=tmp_notebooks_root)
    (notebook.sources_dir / "notes.txt").write_text(
        "Add-one smoothing avoids zero probabilities in language models.\n" * 5
    )
    result = runner.invoke(app, ["ingest", "my-course"])
    assert result.exit_code == 0
    assert "Added 1, updated 0, unchanged 0, removed 0, failed 0." in result.output
    assert "indexed" in result.output
    assert "total" in result.output

    # Re-running with unchanged content shouldn't re-add or re-index.
    result2 = runner.invoke(app, ["ingest", "my-course"])
    assert result2.exit_code == 0
    assert "unchanged 1" in result2.output
    assert "indexed 0 chunks" in result2.output


def test_ingest_reports_repaired_sources(tmp_notebooks_root: Path) -> None:
    notebook = Notebook.create("my-course", root=tmp_notebooks_root)
    (notebook.sources_dir / "notes.txt").write_text(
        "Add-one smoothing avoids zero probabilities in language models.\n" * 5
    )
    # A second source keeps the store non-empty after we delete "notes.txt"'s
    # rows below, so the notebook takes the partial self-heal path rather
    # than the "store empty, manifest not" full-reindex path.
    (notebook.sources_dir / "other.txt").write_text(
        "PageRank models a random walk over the web graph.\n" * 5
    )
    runner.invoke(app, ["ingest", "my-course"])

    from notecast.index.embedder import get_embedder
    from notecast.index.service import open_store

    store = open_store(notebook, embedder=get_embedder())
    store.delete_source("notes.txt")
    store.rebuild_text_index()

    result = runner.invoke(app, ["ingest", "my-course"])
    assert result.exit_code == 0
    assert "repaired 1 source(s): notes.txt" in result.output


def test_search_finds_indexed_chunk(tmp_notebooks_root: Path) -> None:
    notebook = Notebook.create("my-course", root=tmp_notebooks_root)
    (notebook.sources_dir / "smoothing.txt").write_text(
        "Add-one smoothing assigns nonzero probability to unseen n-grams.\n" * 5
    )
    (notebook.sources_dir / "pagerank.txt").write_text(
        "PageRank models a random walk over the web graph to rank pages.\n" * 5
    )
    runner.invoke(app, ["ingest", "my-course"])

    result = runner.invoke(app, ["search", "my-course", "add-one smoothing n-grams", "-k", "2"])
    assert result.exit_code == 0
    assert "smoothing.txt" in result.output
    assert "1." in result.output


def test_search_not_indexed(tmp_notebooks_root: Path) -> None:
    Notebook.create("my-course", root=tmp_notebooks_root)
    result = runner.invoke(app, ["search", "my-course", "anything"])
    assert result.exit_code == 1
    assert "no search index" in result.output.lower() or "notecast ingest" in result.output


def test_eval_missing_file(tmp_notebooks_root: Path) -> None:
    Notebook.create("my-course", root=tmp_notebooks_root)
    result = runner.invoke(app, ["eval", "my-course"])
    assert result.exit_code == 1
    assert "No eval file found" in result.output


def test_eval_runs_against_indexed_notebook(tmp_notebooks_root: Path) -> None:
    notebook = Notebook.create("my-course", root=tmp_notebooks_root)
    (notebook.sources_dir / "week-01" / "smoothing.txt").parent.mkdir(parents=True, exist_ok=True)
    (notebook.sources_dir / "week-01" / "smoothing.txt").write_text(
        "Add-one smoothing assigns nonzero probability to unseen n-grams.\n" * 5
    )
    runner.invoke(app, ["ingest", "my-course"])

    eval_path = notebook.path / "eval.yaml"
    eval_path.write_text(
        "- id: case-1\n"
        '  question: "add-one smoothing unseen n-grams"\n'
        "  expect:\n"
        '    - source: "week-01/smoothing.txt"\n',
        encoding="utf-8",
    )

    result = runner.invoke(app, ["eval", "my-course"])
    assert result.exit_code == 0
    assert "summary:" in result.output
    assert "case-1" in result.output


def test_ingest_invalid_slug_no_traceback(tmp_notebooks_root: Path) -> None:
    result = runner.invoke(app, ["ingest", "Bad Slug"])
    assert result.exit_code != 0
    assert "traceback" not in result.output.lower()


@pytest.fixture
def _restore_parser_registry() -> Iterator[None]:
    from notecast.ingest.parsers import _PARSERS

    snapshot = dict(_PARSERS)
    yield
    _PARSERS.clear()
    _PARSERS.update(snapshot)


def test_ingest_with_failure_exits_nonzero(
    tmp_notebooks_root: Path, _restore_parser_registry: None
) -> None:
    from notecast.ingest.parsers import register

    # Pre-import the built-in parser modules (tolerating any that don't
    # exist yet) so registering our broken parser below, after they've
    # already registered themselves, is the one that wins for ".txt".
    from notecast.ingest.pipeline import load_builtin_parsers

    load_builtin_parsers()

    class BrokenParser:
        suffixes = (".txt",)

        def parse(self, path, source_path):  # noqa: ANN001
            raise RuntimeError("boom")

    register(BrokenParser())
    notebook = Notebook.create("my-course", root=tmp_notebooks_root)
    (notebook.sources_dir / "notes.txt").write_text("hello")

    result = runner.invoke(app, ["ingest", "my-course"])
    assert result.exit_code == 1
    assert "boom" in result.output
    assert "failed 1" in result.output


def test_ingest_generic_failure_prints_friendly_message(
    tmp_notebooks_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import notecast.cli as cli_mod

    Notebook.create("my-course", root=tmp_notebooks_root)

    def _boom(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise ValueError("kaboom")

    monkeypatch.setattr(cli_mod, "index_notebook", _boom)

    result = runner.invoke(app, ["ingest", "my-course"])
    assert result.exit_code == 1
    assert "kaboom" in result.output
    assert "ValueError" in result.output
    assert "traceback" not in result.output.lower()


def test_invalid_slug_message_no_traceback(tmp_notebooks_root: Path) -> None:
    result = runner.invoke(app, ["chat", "Bad Slug"])
    assert result.exit_code != 0
    assert "traceback" not in result.output.lower()


# --- chat / ask ------------------------------------------------------


def _indexed_notebook(root: Path, slug: str = "my-course") -> Notebook:
    notebook = Notebook.create(slug, root=root)
    (notebook.sources_dir / "a.txt").write_text(
        "Add-one smoothing assigns nonzero probability to unseen n-grams.\n" * 5
    )
    runner.invoke(app, ["ingest", slug])
    return notebook


def _fake_answer(
    text_segments: list[tuple[str, list[int]]],
    *,
    citations: list | None = None,
    not_in_sources: bool = False,
    cost: float = 0.0012,
):
    from notecast.chat.models import AnswerSegment, ChatAnswer, Usage

    return ChatAnswer(
        question="q",
        standalone_query="q",
        mode="sources",
        segments=[AnswerSegment(text=t, citation_numbers=n) for t, n in text_segments],
        citations=citations or [],
        not_in_sources=not_in_sources,
        usage=Usage(input_tokens=100, output_tokens=50, est_cost_usd=cost),
        model="claude-sonnet-5",
    )


class _FakeDeepSettings:
    deep_model = "claude-sonnet-5"


class FakeChatSession:
    """Replaces `notecast.chat.session.ChatSession` in CLI tests: returns a
    scripted sequence of `ChatAnswer`s and records how it was called.
    """

    def __init__(
        self,
        answers: list,
        *,
        deep_tokens: int | list[int] = 1000,
        deep_cost: float = 0.01,
    ) -> None:
        self._answers = list(answers)
        self.calls: list[dict] = []
        self.reset_calls = 0
        self.settings = _FakeDeepSettings()
        # Either a fixed token count for every call, or a list popped one
        # value at a time (so a test can simulate the scope growing after a
        # `/week` change between deep turns).
        self._deep_tokens = list(deep_tokens) if isinstance(deep_tokens, list) else deep_tokens
        self._deep_cost = deep_cost
        self.estimate_deep_cost_calls = 0

    def __call__(  # noqa: ANN001
        self, retriever, *, settings=None, client=None, course_name="", chunk_source=None
    ):
        return self

    def estimate_deep_cost(self, filters=None):  # noqa: ANN001
        self.estimate_deep_cost_calls += 1
        if isinstance(self._deep_tokens, list):
            tokens = self._deep_tokens.pop(0)
        else:
            tokens = self._deep_tokens
        return tokens, self._deep_cost

    def ask(self, question, *, mode="sources", filters=None, k=None):  # noqa: ANN001
        self.calls.append({"question": question, "mode": mode, "filters": filters, "k": k})
        return self._answers.pop(0)

    def reset(self) -> None:
        self.reset_calls += 1


def test_filter_chunks_applies_week_source_type_and_path_filters() -> None:
    from notecast.cli import _filter_chunks
    from notecast.models import Chunk, Location, SearchFilters, SourceType

    def _c(source_path: str, week: int, source_type: SourceType, chunk_id: str) -> Chunk:
        return Chunk(
            chunk_id=chunk_id,
            course="c",
            source_path=source_path,
            source_type=source_type,
            ordinal=0,
            text="x",
            location=Location(),
            week=week,
        )

    chunks = [
        _c("a.txt", 1, SourceType.TXT, "c1"),
        _c("b.pdf", 2, SourceType.PDF, "c2"),
        _c("c.txt", 1, SourceType.TXT, "c3"),
    ]

    assert _filter_chunks(chunks, None) == chunks
    assert [c.chunk_id for c in _filter_chunks(chunks, SearchFilters(weeks=[1]))] == ["c1", "c3"]
    assert [
        c.chunk_id for c in _filter_chunks(chunks, SearchFilters(source_types=[SourceType.PDF]))
    ] == ["c2"]
    assert [c.chunk_id for c in _filter_chunks(chunks, SearchFilters(source_paths=["a.txt"]))] == [
        "c1"
    ]


def test_make_chat_session_wires_chunk_source(
    tmp_notebooks_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import notecast.cli as cli_mod
    from notecast.models import Chunk, Location, SearchFilters, SourceType

    notebook = Notebook.create("my-course", root=tmp_notebooks_root)

    chunks = [
        Chunk(
            chunk_id="c1",
            course="c",
            source_path="a.txt",
            source_type=SourceType.TXT,
            ordinal=0,
            text="x",
            location=Location(),
            week=1,
        ),
        Chunk(
            chunk_id="c2",
            course="c",
            source_path="b.txt",
            source_type=SourceType.TXT,
            ordinal=0,
            text="y",
            location=Location(),
            week=2,
        ),
    ]

    class FakeStore:
        def all_chunks(self) -> list[Chunk]:
            return chunks

    class FakeRetriever:
        store = FakeStore()

    captured: dict = {}

    def _fake_chat_session_cls(retriever, *, course_name="", chunk_source=None):  # noqa: ANN001
        captured["chunk_source"] = chunk_source
        return object()

    monkeypatch.setattr(cli_mod, "ChatSession", _fake_chat_session_cls)

    cli_mod._make_chat_session(notebook, FakeRetriever())

    result = captured["chunk_source"](SearchFilters(weeks=[2]))
    assert [c.chunk_id for c in result] == ["c2"]


def test_ask_prints_markers_sources_and_cost_footer(
    tmp_notebooks_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import notecast.cli as cli_mod
    from notecast.chat.models import Citation

    _indexed_notebook(tmp_notebooks_root)
    citation = Citation(n=1, kind="course", source_path="a.txt", location_label="p. 1")
    answer = _fake_answer([("Smoothing avoids zero counts.", [1])], citations=[citation])
    monkeypatch.setattr(cli_mod, "ChatSession", FakeChatSession([answer]))

    result = runner.invoke(app, ["ask", "my-course", "what is smoothing?"])

    assert result.exit_code == 0
    assert "Smoothing avoids zero counts.[1]" in result.output
    assert "Sources:" in result.output
    assert "[1] a.txt" in result.output
    assert "tokens" in result.output
    assert "est cost" in result.output


def test_ask_not_in_sources_banner(
    tmp_notebooks_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import notecast.cli as cli_mod

    _indexed_notebook(tmp_notebooks_root)
    answer = _fake_answer([("The course material doesn't cover this.", [])], not_in_sources=True)
    monkeypatch.setattr(cli_mod, "ChatSession", FakeChatSession([answer]))

    result = runner.invoke(app, ["ask", "my-course", "what's the capital of France?"])

    assert result.exit_code == 0
    assert "Not in your course material" in result.output


def test_ask_missing_api_key_friendly_message_exit_1(
    tmp_notebooks_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import notecast.cli as cli_mod
    from notecast.chat.client import MissingApiKeyError

    _indexed_notebook(tmp_notebooks_root)

    def _raise_missing_key(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise MissingApiKeyError()

    monkeypatch.setattr(cli_mod, "ChatSession", _raise_missing_key)

    result = runner.invoke(app, ["ask", "my-course", "anything"])

    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.output


def test_ask_not_indexed(tmp_notebooks_root: Path) -> None:
    Notebook.create("my-course", root=tmp_notebooks_root)
    result = runner.invoke(app, ["ask", "my-course", "anything"])
    assert result.exit_code == 1
    assert "notecast ingest" in result.output.lower() or "no search index" in result.output.lower()


def test_chat_repl_mode_switch_and_quit(
    tmp_notebooks_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import notecast.cli as cli_mod

    _indexed_notebook(tmp_notebooks_root)
    answers = [
        _fake_answer([("Answer one.", [])], cost=0.001),
        _fake_answer([("Answer two.", [])], cost=0.002),
    ]
    fake_session = FakeChatSession(answers)
    monkeypatch.setattr(cli_mod, "ChatSession", fake_session)

    result = runner.invoke(app, ["chat", "my-course"], input="q1\n/mode open\nq2\n/quit\n")

    assert result.exit_code == 0
    assert "Answer one." in result.output
    assert "Answer two." in result.output
    assert "Mode set to open." in result.output
    assert "Total est cost this session" in result.output
    assert fake_session.calls[0]["mode"] == "sources"
    assert fake_session.calls[1]["mode"] == "open"


def test_ask_deep_mode_prints_estimate_and_confirms(
    tmp_notebooks_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import notecast.cli as cli_mod

    _indexed_notebook(tmp_notebooks_root)
    answer = _fake_answer([("Deep answer.", [])])
    fake_session = FakeChatSession([answer], deep_tokens=12345, deep_cost=0.03)
    monkeypatch.setattr(cli_mod, "ChatSession", fake_session)

    result = runner.invoke(
        app, ["ask", "my-course", "explain everything", "--mode", "deep"], input="y\n"
    )

    assert result.exit_code == 0
    assert "Deep mode will send ~12,345 tokens" in result.output
    assert "first question" in result.output
    assert "per follow-up" in result.output
    assert "Deep answer." in result.output
    assert fake_session.estimate_deep_cost_calls == 1


def test_ask_deep_mode_declined_exits_nonzero(
    tmp_notebooks_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import notecast.cli as cli_mod

    _indexed_notebook(tmp_notebooks_root)
    fake_session = FakeChatSession([])
    monkeypatch.setattr(cli_mod, "ChatSession", fake_session)

    result = runner.invoke(
        app, ["ask", "my-course", "explain everything", "--mode", "deep"], input="n\n"
    )

    assert result.exit_code == 1
    assert fake_session.calls == []


def test_ask_deep_mode_yes_skips_confirmation(
    tmp_notebooks_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import notecast.cli as cli_mod

    _indexed_notebook(tmp_notebooks_root)
    answer = _fake_answer([("Deep answer.", [])])
    fake_session = FakeChatSession([answer])
    monkeypatch.setattr(cli_mod, "ChatSession", fake_session)

    result = runner.invoke(
        app, ["ask", "my-course", "explain everything", "--mode", "deep", "--yes"]
    )

    assert result.exit_code == 0
    assert "Deep answer." in result.output


def test_chat_deep_mode_estimate_shown_once(
    tmp_notebooks_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import notecast.cli as cli_mod

    _indexed_notebook(tmp_notebooks_root)
    answers = [
        _fake_answer([("Deep one.", [])]),
        _fake_answer([("Deep two.", [])]),
    ]
    fake_session = FakeChatSession(answers, deep_tokens=5000, deep_cost=0.02)
    monkeypatch.setattr(cli_mod, "ChatSession", fake_session)

    result = runner.invoke(
        app,
        ["chat", "my-course", "--mode", "deep"],
        input="q1\ny\nq2\n/quit\n",
    )

    assert result.exit_code == 0
    assert "Deep one." in result.output
    assert "Deep two." in result.output
    # The estimate is re-checked every deep turn (so scope growth via /week
    # can be caught), but the same-size second turn doesn't re-print it or
    # ask for confirmation again.
    assert result.output.count("Deep mode will send") == 1
    assert fake_session.estimate_deep_cost_calls == 2


def test_chat_deep_mode_reconfirms_when_scope_grows_after_week_change(
    tmp_notebooks_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `/week` change that grows the deep scope must be confirmed again,
    even though deep mode was already confirmed once this session.
    """
    import notecast.cli as cli_mod

    _indexed_notebook(tmp_notebooks_root)
    answers = [
        _fake_answer([("Deep one.", [])]),
        _fake_answer([("Deep two.", [])]),
    ]
    # First deep turn: 2,000 tokens (week 1). After "/week all", the second
    # deep turn covers everything: 9,000 tokens -- bigger, so it must be
    # confirmed again.
    fake_session = FakeChatSession(answers, deep_tokens=[2000, 9000], deep_cost=0.02)
    monkeypatch.setattr(cli_mod, "ChatSession", fake_session)

    result = runner.invoke(
        app,
        ["chat", "my-course", "--mode", "deep", "--week", "1"],
        input="q1\ny\n/week all\nq2\ny\n/quit\n",
    )

    assert result.exit_code == 0
    assert "Deep one." in result.output
    assert "Deep two." in result.output
    assert result.output.count("Deep mode will send") == 2
    assert "~2,000 tokens" in result.output
    assert "~9,000 tokens" in result.output
    assert fake_session.estimate_deep_cost_calls == 2


def test_chat_deep_mode_no_reconfirm_when_scope_shrinks(
    tmp_notebooks_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `/week` change that *shrinks* the deep scope shouldn't trigger a
    second confirmation -- only growth past what's already been confirmed
    does.
    """
    import notecast.cli as cli_mod

    _indexed_notebook(tmp_notebooks_root)
    answers = [
        _fake_answer([("Deep one.", [])]),
        _fake_answer([("Deep two.", [])]),
    ]
    fake_session = FakeChatSession(answers, deep_tokens=[9000, 2000], deep_cost=0.02)
    monkeypatch.setattr(cli_mod, "ChatSession", fake_session)

    result = runner.invoke(
        app,
        ["chat", "my-course", "--mode", "deep"],
        input="q1\ny\n/week 1\nq2\n/quit\n",
    )

    assert result.exit_code == 0
    assert "Deep one." in result.output
    assert "Deep two." in result.output
    assert result.output.count("Deep mode will send") == 1


# --- audio -------------------------------------------------------------


def _fake_overview_result(notebook: Notebook, *, render: bool):
    from notecast.audio.models import (
        AudioPlan,
        AudioScope,
        AudioScript,
        ChapterPlan,
        ChapterScript,
        CoverageReport,
        RankedKeyPoint,
        RenderedChapter,
        RenderResult,
        ScriptLine,
    )
    from notecast.audio.service import OverviewResult

    point = RankedKeyPoint(id="kp-1", title="Smoothing", summary="What it is.", tier="A", rank=1)
    plan = AudioPlan(
        scope=AudioScope(weeks=[1]),
        points=[point],
        target_minutes=6.0,
        target_words=900,
        chapters=[ChapterPlan(index=1, title="Intro", point_ids=["kp-1"], target_words=900)],
    )
    script = AudioScript(
        title="My Course — Week 1",
        plan=plan,
        chapters=[
            ChapterScript(
                index=1,
                title="Intro",
                lines=[ScriptLine(speaker="A", text="Hello.", source_chunk_ids=[])],
            )
        ],
        coverage=CoverageReport(covered=["kp-1"]),
        est_cost_usd=0.05,
    )
    script_json_path = notebook.audio_dir / "week-1.script.json"
    script_json_path.parent.mkdir(parents=True, exist_ok=True)
    script_json_path.write_text(script.model_dump_json(), encoding="utf-8")

    render_result = None
    if render:
        mp3_path = notebook.audio_dir / "week-1.mp3"
        transcript_path = notebook.audio_dir / "week-1.md"
        mp3_path.write_bytes(b"fake")
        transcript_path.write_text("# transcript\n", encoding="utf-8")
        render_result = RenderResult(
            mp3_path=mp3_path,
            transcript_path=transcript_path,
            duration_seconds=10.0,
            chapters=[RenderedChapter(index=1, title="Intro", start_seconds=0.0)],
        )

    return OverviewResult(
        script=script,
        render=render_result,
        script_json_path=script_json_path,
        est_cost_usd=0.05,
        timings={},
    )


def _fake_estimate(**overrides):
    from notecast.audio.service import OverviewEstimate

    defaults = dict(
        total_material_tokens=5000,
        n_sources=1,
        n_chapters_low=3,
        n_chapters_high=5,
        est_cost_usd=0.1,
        est_cost_low_usd=0.05,
        est_cost_high_usd=0.15,
    )
    defaults.update(overrides)
    return OverviewEstimate(**defaults)


def test_audio_script_only_prints_plan_summary(
    tmp_notebooks_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import notecast.cli as cli_mod

    notebook = Notebook.create("my-course", root=tmp_notebooks_root)
    result_obj = _fake_overview_result(notebook, render=False)

    monkeypatch.setattr(
        cli_mod, "estimate_overview_cost", lambda nb, scope, settings: _fake_estimate()
    )

    def _fake_generate_overview(nb, scope, *, settings=None, render=True, progress=None, **kw):  # noqa: ANN001
        if progress is not None:
            progress("loading", 0.0)
            progress("done", 1.0)
        return result_obj

    monkeypatch.setattr(cli_mod, "generate_overview", _fake_generate_overview)

    def _boom_render(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise AssertionError("render_saved_script should not be called with --script-only")

    monkeypatch.setattr(cli_mod, "render_saved_script", _boom_render)

    result = runner.invoke(app, ["audio", "my-course", "--week", "1", "--script-only", "--yes"])

    assert result.exit_code == 0
    assert "Estimated cost: ~$0.1000 (range $0.0500" in result.output
    assert "My Course — Week 1" in result.output
    assert "Target length: 6.0 min, 1 chapters" in result.output
    assert "Tier A points: 1, Tier B points: 0" in result.output
    assert "Estimated API cost: $0.0500" in result.output
    assert str(result_obj.script_json_path) in result.output


def test_audio_requires_confirmation_without_yes(
    tmp_notebooks_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import notecast.cli as cli_mod

    Notebook.create("my-course", root=tmp_notebooks_root)
    monkeypatch.setattr(
        cli_mod, "estimate_overview_cost", lambda nb, scope, settings: _fake_estimate()
    )

    def _boom(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise AssertionError("generate_overview should not run if the user declines")

    monkeypatch.setattr(cli_mod, "generate_overview", _boom)

    result = runner.invoke(app, ["audio", "my-course", "--week", "1", "--script-only"], input="n\n")

    assert result.exit_code == 1
    assert "Cancelled." in result.output


def test_audio_empty_scope_friendly_error(
    tmp_notebooks_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import notecast.cli as cli_mod

    Notebook.create("my-course", root=tmp_notebooks_root)

    def _raise_empty(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise ValueError("No material in Week 9 — check the week numbers or run ingest")

    monkeypatch.setattr(cli_mod, "estimate_overview_cost", _raise_empty)

    result = runner.invoke(app, ["audio", "my-course", "--week", "9", "--script-only"])

    assert result.exit_code == 1
    assert "No material in Week 9" in result.output


def test_audio_failure_after_spend_reports_cost_and_stage(
    tmp_notebooks_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import notecast.cli as cli_mod
    from notecast.audio.service import OverviewFailed

    Notebook.create("my-course", root=tmp_notebooks_root)
    monkeypatch.setattr(
        cli_mod, "estimate_overview_cost", lambda nb, scope, settings: _fake_estimate()
    )

    def _raise_failed(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise OverviewFailed("scripting", 0.1234, "boom while scripting")

    monkeypatch.setattr(cli_mod, "generate_overview", _raise_failed)

    result = runner.invoke(app, ["audio", "my-course", "--week", "1", "--script-only", "--yes"])

    assert result.exit_code == 1
    assert "failed at scripting" in result.output
    assert "boom while scripting" in result.output
    assert "spent ~$0.1234" in result.output


def _write_minimal_script(path: Path) -> None:
    from notecast.audio.models import (
        AudioPlan,
        AudioScope,
        AudioScript,
        ChapterPlan,
        ChapterScript,
        ScriptLine,
    )

    plan = AudioPlan(
        scope=AudioScope(weeks=[1]),
        points=[],
        target_minutes=3.0,
        target_words=450,
        chapters=[ChapterPlan(index=1, title="Intro", point_ids=[], target_words=450)],
    )
    script = AudioScript(
        title="My Course — Week 1",
        plan=plan,
        chapters=[
            ChapterScript(
                index=1,
                title="Intro",
                lines=[ScriptLine(speaker="A", text="Hello.", source_chunk_ids=["c1"])],
            )
        ],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script.model_dump_json(), encoding="utf-8")


def test_audio_render_command(tmp_notebooks_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import notecast.cli as cli_mod
    from notecast.audio.models import RenderedChapter, RenderResult
    from notecast.audio.service import RenderSavedResult

    notebook = Notebook.create("my-course", root=tmp_notebooks_root)
    script_json_path = notebook.audio_dir / "week-1.script.json"
    _write_minimal_script(script_json_path)

    mp3_path = notebook.audio_dir / "week-1.mp3"
    transcript_path = notebook.audio_dir / "week-1.md"
    mp3_path.write_bytes(b"fake")
    transcript_path.write_text("# transcript\n", encoding="utf-8")

    def _fake_render_saved_script(nb, path, *, settings=None, tts=None, progress=None):  # noqa: ANN001
        assert path == script_json_path
        if progress is not None:
            progress("rendering", 1.0)
        render_result = RenderResult(
            mp3_path=mp3_path,
            transcript_path=transcript_path,
            duration_seconds=10.0,
            chapters=[RenderedChapter(index=1, title="Intro", start_seconds=0.0)],
        )
        return RenderSavedResult(render=render_result, stale_chunk_ids=["c1"])

    monkeypatch.setattr(cli_mod, "render_saved_script", _fake_render_saved_script)

    result = runner.invoke(app, ["audio-render", "my-course", str(script_json_path)])

    assert result.exit_code == 0
    assert "Estimated render time" in result.output
    assert str(mp3_path) in result.output
    assert str(transcript_path) in result.output
    assert "no longer in the current index" in result.output
    assert "1 cited chunk(s)" in result.output


def test_audio_render_missing_script_file(tmp_notebooks_root: Path) -> None:
    Notebook.create("my-course", root=tmp_notebooks_root)
    missing = tmp_notebooks_root / "my-course" / "audio" / "nope.script.json"
    result = runner.invoke(app, ["audio-render", "my-course", str(missing)])
    assert result.exit_code == 1
    assert "No script file" in result.output
