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


def test_chat_placeholder(tmp_notebooks_root: Path) -> None:
    result = runner.invoke(app, ["chat", "my-course"])
    assert result.exit_code == 0
    assert "Phase 3" in result.output


def test_audio_placeholder(tmp_notebooks_root: Path) -> None:
    result = runner.invoke(app, ["audio", "my-course"])
    assert result.exit_code == 0
    assert "Phase 6" in result.output


def test_invalid_slug_message_no_traceback(tmp_notebooks_root: Path) -> None:
    result = runner.invoke(app, ["chat", "Bad Slug"])
    assert result.exit_code != 0
    assert "traceback" not in result.output.lower()
