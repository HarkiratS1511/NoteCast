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


class FakeChatSession:
    """Replaces `notecast.chat.session.ChatSession` in CLI tests: returns a
    scripted sequence of `ChatAnswer`s and records how it was called.
    """

    def __init__(self, answers: list) -> None:
        self._answers = list(answers)
        self.calls: list[dict] = []
        self.reset_calls = 0

    def __call__(self, retriever, *, settings=None, client=None, course_name=""):  # noqa: ANN001
        return self

    def ask(self, question, *, mode="sources", filters=None, k=None):  # noqa: ANN001
        self.calls.append({"question": question, "mode": mode, "filters": filters, "k": k})
        return self._answers.pop(0)

    def reset(self) -> None:
        self.reset_calls += 1


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


def test_audio_script_only_prints_plan_summary(
    tmp_notebooks_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import notecast.cli as cli_mod

    notebook = Notebook.create("my-course", root=tmp_notebooks_root)
    result_obj = _fake_overview_result(notebook, render=False)

    def _fake_generate_overview(nb, scope, *, settings=None, render=True, progress=None, **kw):  # noqa: ANN001
        if progress is not None:
            progress("loading", 0.0)
            progress("done", 1.0)
        return result_obj

    monkeypatch.setattr(cli_mod, "generate_overview", _fake_generate_overview)

    def _boom_render(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise AssertionError("render_saved_script should not be called with --script-only")

    monkeypatch.setattr(cli_mod, "render_saved_script", _boom_render)

    result = runner.invoke(app, ["audio", "my-course", "--week", "1", "--script-only"])

    assert result.exit_code == 0
    assert "My Course — Week 1" in result.output
    assert "Target length: 6.0 min, 1 chapters" in result.output
    assert "Tier A points: 1, Tier B points: 0" in result.output
    assert "Estimated API cost: $0.0500" in result.output
    assert str(result_obj.script_json_path) in result.output


def test_audio_empty_scope_friendly_error(
    tmp_notebooks_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import notecast.cli as cli_mod

    Notebook.create("my-course", root=tmp_notebooks_root)

    def _raise_empty(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise ValueError("No material in Week 9 — check the week numbers or run ingest")

    monkeypatch.setattr(cli_mod, "generate_overview", _raise_empty)

    result = runner.invoke(app, ["audio", "my-course", "--week", "9", "--script-only"])

    assert result.exit_code == 1
    assert "No material in Week 9" in result.output


def test_audio_render_command(tmp_notebooks_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import notecast.cli as cli_mod
    from notecast.audio.models import RenderedChapter, RenderResult

    notebook = Notebook.create("my-course", root=tmp_notebooks_root)
    script_json_path = notebook.audio_dir / "week-1.script.json"
    script_json_path.parent.mkdir(parents=True, exist_ok=True)
    script_json_path.write_text("{}", encoding="utf-8")

    mp3_path = notebook.audio_dir / "week-1.mp3"
    transcript_path = notebook.audio_dir / "week-1.md"
    mp3_path.write_bytes(b"fake")
    transcript_path.write_text("# transcript\n", encoding="utf-8")

    def _fake_render_saved_script(nb, path, *, settings=None, tts=None, progress=None):  # noqa: ANN001
        assert path == script_json_path
        if progress is not None:
            progress("rendering", 1.0)
        return RenderResult(
            mp3_path=mp3_path,
            transcript_path=transcript_path,
            duration_seconds=10.0,
            chapters=[RenderedChapter(index=1, title="Intro", start_seconds=0.0)],
        )

    monkeypatch.setattr(cli_mod, "render_saved_script", _fake_render_saved_script)

    result = runner.invoke(app, ["audio-render", "my-course", str(script_json_path)])

    assert result.exit_code == 0
    assert str(mp3_path) in result.output
    assert str(transcript_path) in result.output


def test_audio_render_missing_script_file(tmp_notebooks_root: Path) -> None:
    Notebook.create("my-course", root=tmp_notebooks_root)
    missing = tmp_notebooks_root / "my-course" / "audio" / "nope.script.json"
    result = runner.invoke(app, ["audio-render", "my-course", str(missing)])
    assert result.exit_code == 1
    assert "No script file" in result.output
