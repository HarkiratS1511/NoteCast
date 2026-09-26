"""The end-to-end ingest pipeline: parse new/changed source files, chunk
them, write processed output, and keep the manifest up to date.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from notecast.ingest import chunker
from notecast.ingest import course as course_mod
from notecast.ingest import manifest as manifest_mod
from notecast.ingest.parsers import BUILTIN_PARSER_MODULES, UnsupportedFileError, get_parser
from notecast.models import Chunk, ManifestEntry
from notecast.notebook import Notebook

logger = logging.getLogger(__name__)


class IngestReport(BaseModel):
    """A summary of one `ingest_notebook` run."""

    added: list[str] = Field(default_factory=list)
    updated: list[str] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    failed: dict[str, str] = Field(default_factory=dict)
    chunk_count: int = 0


def load_builtin_parsers() -> None:
    """Import every built-in parser module, tolerating any one of them
    failing to import (e.g. a missing optional dependency mid-development).
    """
    for module_name in BUILTIN_PARSER_MODULES:
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            logger.warning("Could not load parser module %s: %s", module_name, exc)


def _processed_json_path(nb: Notebook, source_path: str) -> Path:
    return nb.processed_dir / f"{source_path}.json"


def _chunks_path(nb: Notebook, source_path: str) -> Path:
    return nb.processed_dir / f"{source_path}.chunks.jsonl"


def _write_processed(nb: Notebook, source_path: str, doc: object) -> None:
    path = _processed_json_path(nb, source_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")  # type: ignore[union-attr]


def _write_chunks(nb: Notebook, source_path: str, chunks: list[Chunk]) -> None:
    path = _chunks_path(nb, source_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json())
            f.write("\n")


def _delete_processed(nb: Notebook, source_path: str) -> None:
    for path in (_processed_json_path(nb, source_path), _chunks_path(nb, source_path)):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def ingest_notebook(
    nb: Notebook,
    *,
    force: bool = False,
    on_chunks: Callable[[str, list[Chunk]], None] | None = None,
    on_removed: Callable[[str], None] | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> IngestReport:
    """Parse/chunk new or changed source files, drop removed ones, and keep
    the manifest and processed/ directory in sync.
    """
    load_builtin_parsers()
    m = manifest_mod.load_manifest(nb)
    config = course_mod.load_course_config(nb)
    course_name = course_mod.display_name(nb, config)

    report = IngestReport()
    current_files = nb.source_files()
    current_paths = {nb.relative_source_path(p): p for p in current_files}
    total = len(current_paths)

    for index, source_path in enumerate(sorted(current_paths), start=1):
        path = current_paths[source_path]
        if progress:
            progress(source_path, index, total)

        try:
            sha = manifest_mod.file_sha256(path)
        except OSError as exc:
            report.failed[source_path] = f"Could not read file: {exc}"
            continue

        existing = m.entries.get(source_path)
        if existing is not None and existing.sha256 == sha and not force:
            report.unchanged.append(source_path)
            continue

        try:
            parser = get_parser(path)
            doc = parser.parse(path, source_path)
            week = course_mod.infer_week(source_path, config)
            topic = course_mod.infer_topic(source_path, config)
            chunks = chunker.chunk_document(
                doc,
                course=nb.slug,
                course_name=course_name,
                week=week,
                topic=topic,
            )
            _write_processed(nb, source_path, doc)
            _write_chunks(nb, source_path, chunks)
        except UnsupportedFileError as exc:
            report.failed[source_path] = str(exc)
            continue
        except Exception as exc:  # noqa: BLE001 -- one bad file must not stop the run
            report.failed[source_path] = str(exc)
            continue

        if on_chunks:
            try:
                on_chunks(source_path, chunks)
            except Exception as exc:  # noqa: BLE001 -- one bad file must not stop the run
                report.failed[source_path] = str(exc)
                # Drop any existing manifest entry so this file is retried
                # (not silently treated as "unchanged") next run, whatever
                # partial state `on_chunks` left the index in.
                m.entries.pop(source_path, None)
                manifest_mod.save_manifest(nb, m)
                continue

        m.entries[source_path] = ManifestEntry(
            source_path=source_path,
            sha256=sha,
            source_type=doc.source_type,
            title=doc.title,
            chunk_count=len(chunks),
            ingested_at=datetime.now(UTC),
        )
        if existing is None:
            report.added.append(source_path)
        else:
            report.updated.append(source_path)
        report.chunk_count += len(chunks)
        manifest_mod.save_manifest(nb, m)

    removed_paths = [sp for sp in m.entries if sp not in current_paths]
    for source_path in removed_paths:
        if on_removed:
            try:
                on_removed(source_path)
            except Exception as exc:  # noqa: BLE001 -- one bad removal must not stop the run
                report.failed[source_path] = str(exc)
                # Keep the manifest entry (and processed files) so removal
                # is retried next run instead of the source being silently
                # dropped from the manifest while still sitting in the index.
                continue
        _delete_processed(nb, source_path)
        del m.entries[source_path]
        report.removed.append(source_path)

    manifest_mod.save_manifest(nb, m)
    return report


def load_source_chunks(nb: Notebook, source_path: str) -> list[Chunk]:
    """Read one already-ingested source's chunks from its `.chunks.jsonl`
    file (without re-parsing or re-chunking). Empty list if it doesn't exist.
    """
    path = _chunks_path(nb, source_path)
    if not path.exists():
        return []
    chunks: list[Chunk] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            chunks.append(Chunk.model_validate_json(line))
    return chunks


def load_chunks(nb: Notebook) -> list[Chunk]:
    """Read all chunks for currently-manifested sources from their
    `.chunks.jsonl` files.
    """
    m = manifest_mod.load_manifest(nb)
    chunks: list[Chunk] = []
    for source_path in sorted(m.entries):
        chunks.extend(load_source_chunks(nb, source_path))
    return chunks
