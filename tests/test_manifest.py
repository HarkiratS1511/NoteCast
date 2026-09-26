"""Tests for manifest load/save and file hashing."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from notecast.ingest.manifest import file_sha256, load_manifest, save_manifest
from notecast.models import Manifest, ManifestEntry, SourceType
from notecast.notebook import Notebook


def test_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    f = tmp_path / "a.txt"
    f.write_bytes(b"hello world" * 1000)
    assert file_sha256(f) == hashlib.sha256(b"hello world" * 1000).hexdigest()


def test_load_manifest_missing_returns_empty(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("my-course", root=tmp_notebooks_root)
    manifest = load_manifest(nb)
    assert manifest == Manifest()


def test_save_and_load_round_trip(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("my-course", root=tmp_notebooks_root)
    manifest = Manifest()
    manifest.entries["a.txt"] = ManifestEntry(
        source_path="a.txt",
        sha256="abc123",
        source_type=SourceType.TXT,
        title="A",
        chunk_count=3,
        ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    save_manifest(nb, manifest)
    assert nb.manifest_path.exists()

    loaded = load_manifest(nb)
    assert loaded.entries["a.txt"].sha256 == "abc123"
    assert loaded.entries["a.txt"].chunk_count == 3


def test_save_manifest_is_atomic_no_leftover_temp_files(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("my-course", root=tmp_notebooks_root)
    save_manifest(nb, Manifest())
    leftovers = [p for p in nb.path.iterdir() if p.name.startswith(".manifest-")]
    assert leftovers == []


def test_load_manifest_corrupt_json_backs_up(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("my-course", root=tmp_notebooks_root)
    nb.manifest_path.write_text("{not valid json", encoding="utf-8")

    manifest = load_manifest(nb)
    assert manifest == Manifest()

    backup = nb.manifest_path.with_name("manifest.json.bak")
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == "{not valid json"
    assert not nb.manifest_path.exists()


def test_load_manifest_invalid_shape_backs_up(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("my-course", root=tmp_notebooks_root)
    nb.manifest_path.write_text('{"entries": "not-a-dict"}', encoding="utf-8")

    manifest = load_manifest(nb)
    assert manifest == Manifest()

    backup = nb.manifest_path.with_name("manifest.json.bak")
    assert backup.exists()
