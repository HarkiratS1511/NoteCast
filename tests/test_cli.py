"""Tests for the notecast CLI, using Typer's CliRunner against a temporary
notebooks root (never the real notebooks/ directory).
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from notecast import __version__
from notecast.cli import app

runner = CliRunner()


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


def test_ingest_placeholder(tmp_notebooks_root: Path) -> None:
    result = runner.invoke(app, ["ingest", "my-course"])
    assert result.exit_code == 0
    assert "Phase 1" in result.output


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
