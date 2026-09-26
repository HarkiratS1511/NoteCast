"""Tests for the ingest pipeline, using fake parsers registered on real
source suffixes (source_files() only returns supported SourceType suffixes).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from notecast.ingest import pipeline
from notecast.ingest.parsers import _PARSERS, register
from notecast.ingest.pipeline import ingest_notebook, load_chunks
from notecast.models import ParsedDocument, Section, SourceType
from notecast.notebook import Notebook


class FakeTextParser:
    suffixes = (".txt",)

    def parse(self, path: Path, source_path: str) -> ParsedDocument:
        return ParsedDocument(
            source_path=source_path,
            source_type=SourceType.TXT,
            title=path.stem,
            sections=[Section(text=path.read_text(encoding="utf-8"))],
        )


class FailingParser:
    suffixes = (".md",)

    def parse(self, path: Path, source_path: str) -> ParsedDocument:
        raise RuntimeError("simulated parse failure")


@pytest.fixture(autouse=True)
def _fake_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    snapshot = dict(_PARSERS)
    register(FakeTextParser())
    register(FailingParser())
    monkeypatch.setattr(pipeline, "load_builtin_parsers", lambda: None)
    yield
    _PARSERS.clear()
    _PARSERS.update(snapshot)


def test_ingest_adds_new_files(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("my-course", root=tmp_notebooks_root)
    (nb.sources_dir / "notes.txt").write_text("hello world")

    report = ingest_notebook(nb)

    assert report.added == ["notes.txt"]
    assert report.updated == []
    assert report.failed == {}
    assert report.chunk_count == 1

    processed = nb.processed_dir / "notes.txt.json"
    chunks_file = nb.processed_dir / "notes.txt.chunks.jsonl"
    assert processed.exists()
    assert chunks_file.exists()


def test_ingest_rerun_is_unchanged(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("my-course", root=tmp_notebooks_root)
    (nb.sources_dir / "notes.txt").write_text("hello world")
    ingest_notebook(nb)

    report = ingest_notebook(nb)
    assert report.added == []
    assert report.unchanged == ["notes.txt"]
    assert report.chunk_count == 0


def test_ingest_updates_on_content_change(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("my-course", root=tmp_notebooks_root)
    source = nb.sources_dir / "notes.txt"
    source.write_text("hello world")
    ingest_notebook(nb)

    source.write_text("hello world, updated")
    report = ingest_notebook(nb)
    assert report.updated == ["notes.txt"]
    assert report.added == []
    assert report.unchanged == []


def test_ingest_force_reprocesses_unchanged_file(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("my-course", root=tmp_notebooks_root)
    (nb.sources_dir / "notes.txt").write_text("hello world")
    ingest_notebook(nb)

    report = ingest_notebook(nb, force=True)
    assert report.unchanged == []
    assert report.updated == ["notes.txt"]


def test_ingest_removed_file_calls_on_removed_and_deletes_processed(
    tmp_notebooks_root: Path,
) -> None:
    nb = Notebook.create("my-course", root=tmp_notebooks_root)
    source = nb.sources_dir / "notes.txt"
    source.write_text("hello world")
    ingest_notebook(nb)
    assert (nb.processed_dir / "notes.txt.json").exists()

    source.unlink()
    removed_calls: list[str] = []
    report = ingest_notebook(nb, on_removed=removed_calls.append)

    assert report.removed == ["notes.txt"]
    assert removed_calls == ["notes.txt"]
    assert not (nb.processed_dir / "notes.txt.json").exists()
    assert not (nb.processed_dir / "notes.txt.chunks.jsonl").exists()


def test_ingest_parser_exception_recorded_others_succeed(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("my-course", root=tmp_notebooks_root)
    (nb.sources_dir / "good.txt").write_text("hello world")
    (nb.sources_dir / "bad.md").write_text("# heading\nsome text")

    report = ingest_notebook(nb)

    assert report.added == ["good.txt"]
    assert "bad.md" in report.failed
    assert "simulated parse failure" in report.failed["bad.md"]
    # The failed file leaves no processed output and no manifest entry.
    assert not (nb.processed_dir / "bad.md.json").exists()

    manifest = pipeline.manifest_mod.load_manifest(nb)
    assert "bad.md" not in manifest.entries
    assert "good.txt" in manifest.entries


def test_on_chunks_callback_invoked(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("my-course", root=tmp_notebooks_root)
    (nb.sources_dir / "notes.txt").write_text("hello world")

    calls = []
    ingest_notebook(nb, on_chunks=lambda source_path, chunks: calls.append((source_path, chunks)))

    assert len(calls) == 1
    assert calls[0][0] == "notes.txt"
    assert len(calls[0][1]) == 1


def test_progress_callback_invoked_per_file(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("my-course", root=tmp_notebooks_root)
    (nb.sources_dir / "a.txt").write_text("hello")
    (nb.sources_dir / "b.txt").write_text("world")

    calls = []
    ingest_notebook(
        nb,
        progress=lambda source_path, index, total: calls.append((source_path, index, total)),
    )

    assert len(calls) == 2
    assert all(total == 2 for _, _, total in calls)


def test_load_chunks_reads_manifested_sources(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("my-course", root=tmp_notebooks_root)
    (nb.sources_dir / "a.txt").write_text("hello world")
    (nb.sources_dir / "b.txt").write_text("goodbye world")
    ingest_notebook(nb)

    chunks = load_chunks(nb)
    assert {c.source_path for c in chunks} == {"a.txt", "b.txt"}


def test_on_chunks_failure_recorded_and_manifest_entry_dropped(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("my-course", root=tmp_notebooks_root)
    (nb.sources_dir / "good.txt").write_text("hello world")
    (nb.sources_dir / "bad.txt").write_text("goodbye world")

    def flaky_on_chunks(source_path: str, chunks: list) -> None:  # noqa: ANN001
        if source_path == "bad.txt":
            raise RuntimeError("embed boom")

    report = ingest_notebook(nb, on_chunks=flaky_on_chunks)

    assert report.added == ["good.txt"]
    assert report.failed == {"bad.txt": "embed boom"}

    manifest = pipeline.manifest_mod.load_manifest(nb)
    assert "good.txt" in manifest.entries
    assert "bad.txt" not in manifest.entries
    # Processed output was still written (parsing/chunking succeeded); only
    # the manifest entry (and thus "is this file up to date") is withheld.
    assert (nb.processed_dir / "bad.txt.chunks.jsonl").exists()


def test_on_chunks_failure_on_update_drops_existing_manifest_entry(
    tmp_notebooks_root: Path,
) -> None:
    nb = Notebook.create("my-course", root=tmp_notebooks_root)
    source = nb.sources_dir / "notes.txt"
    source.write_text("hello world")
    ingest_notebook(nb)
    manifest = pipeline.manifest_mod.load_manifest(nb)
    assert "notes.txt" in manifest.entries

    source.write_text("hello world, updated")

    def failing_on_chunks(source_path: str, chunks: list) -> None:  # noqa: ANN001
        raise RuntimeError("embed boom")

    report = ingest_notebook(nb, on_chunks=failing_on_chunks)

    assert report.updated == []
    assert report.failed == {"notes.txt": "embed boom"}
    manifest = pipeline.manifest_mod.load_manifest(nb)
    assert "notes.txt" not in manifest.entries

    # A later normal run (no failure) must treat it as new again, not as
    # already up to date.
    report2 = ingest_notebook(nb)
    assert report2.added == ["notes.txt"]
    assert report2.failed == {}


def test_on_removed_failure_is_retried_next_run(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("my-course", root=tmp_notebooks_root)
    source = nb.sources_dir / "notes.txt"
    source.write_text("hello world")
    ingest_notebook(nb)
    source.unlink()

    calls: list[str] = []

    def flaky_on_removed(source_path: str) -> None:
        calls.append(source_path)
        if len(calls) == 1:
            raise RuntimeError("delete boom")

    report1 = ingest_notebook(nb, on_removed=flaky_on_removed)
    assert report1.removed == []
    assert report1.failed == {"notes.txt": "delete boom"}
    manifest = pipeline.manifest_mod.load_manifest(nb)
    assert "notes.txt" in manifest.entries
    # Processed files are kept too, so a later successful removal call
    # doesn't need anything re-parsed.
    assert (nb.processed_dir / "notes.txt.chunks.jsonl").exists()

    report2 = ingest_notebook(nb, on_removed=flaky_on_removed)
    assert report2.removed == ["notes.txt"]
    assert report2.failed == {}
    manifest = pipeline.manifest_mod.load_manifest(nb)
    assert "notes.txt" not in manifest.entries
    assert not (nb.processed_dir / "notes.txt.chunks.jsonl").exists()


def test_load_source_chunks_reads_one_source(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("my-course", root=tmp_notebooks_root)
    (nb.sources_dir / "a.txt").write_text("hello world")
    (nb.sources_dir / "b.txt").write_text("goodbye world")
    ingest_notebook(nb)

    chunks = pipeline.load_source_chunks(nb, "a.txt")
    assert {c.source_path for c in chunks} == {"a.txt"}

    assert pipeline.load_source_chunks(nb, "missing.txt") == []


def test_manifest_survives_crash_after_each_file(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("my-course", root=tmp_notebooks_root)
    (nb.sources_dir / "notes.txt").write_text("hello world")
    ingest_notebook(nb)

    from notecast.ingest.manifest import load_manifest

    manifest = load_manifest(nb)
    assert "notes.txt" in manifest.entries
