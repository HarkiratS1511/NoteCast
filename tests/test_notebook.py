"""Tests for notecast.notebook: slug validation, create/list, and source
file discovery.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from notecast.notebook import Notebook


class TestSlugValidation:
    @pytest.mark.parametrize(
        "slug",
        ["comp4650-document-analysis", "a", "a1-b2", "week01"],
    )
    def test_valid_slugs(self, slug: str, tmp_path: Path) -> None:
        Notebook(slug, root=tmp_path)  # should not raise

    @pytest.mark.parametrize(
        "slug",
        ["-leading-hyphen", "Has-Upper", "has space", "has_underscore", "", "a" * 65],
    )
    def test_invalid_slugs(self, slug: str, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            Notebook(slug, root=tmp_path)


class TestCreateAndExists:
    def test_create_makes_dirs(self, tmp_path: Path) -> None:
        nb = Notebook.create("my-course", root=tmp_path)
        assert nb.exists()
        assert nb.sources_dir.exists()
        assert nb.processed_dir.exists()
        assert nb.audio_dir.exists()

    def test_not_exists_before_create(self, tmp_path: Path) -> None:
        nb = Notebook("my-course", root=tmp_path)
        assert not nb.exists()

    def test_duplicate_create_raises(self, tmp_path: Path) -> None:
        Notebook.create("my-course", root=tmp_path)
        with pytest.raises(ValueError):
            Notebook.create("my-course", root=tmp_path)

    def test_paths(self, tmp_path: Path) -> None:
        nb = Notebook("my-course", root=tmp_path)
        assert nb.path == tmp_path / "my-course"
        assert nb.index_dir == tmp_path / "my-course" / "index.lancedb"
        assert nb.manifest_path == tmp_path / "my-course" / "manifest.json"
        assert nb.config_path == tmp_path / "my-course" / "course.yaml"


class TestListAll:
    def test_empty_root(self, tmp_path: Path) -> None:
        assert Notebook.list_all(root=tmp_path) == []

    def test_missing_root(self, tmp_path: Path) -> None:
        assert Notebook.list_all(root=tmp_path / "does-not-exist") == []

    def test_lists_only_notebooks_with_sources(self, tmp_path: Path) -> None:
        Notebook.create("course-a", root=tmp_path)
        Notebook.create("course-b", root=tmp_path)
        (tmp_path / "not-a-notebook").mkdir()  # no sources/ subdir
        (tmp_path / "Invalid-Slug").mkdir()
        (tmp_path / "Invalid-Slug" / "sources").mkdir()

        slugs = [nb.slug for nb in Notebook.list_all(root=tmp_path)]
        assert slugs == ["course-a", "course-b"]

    def test_sorted_by_slug(self, tmp_path: Path) -> None:
        Notebook.create("zebra", root=tmp_path)
        Notebook.create("apple", root=tmp_path)
        slugs = [nb.slug for nb in Notebook.list_all(root=tmp_path)]
        assert slugs == ["apple", "zebra"]


class TestSourceFiles:
    def test_filters_hidden_and_lock_files(self, tmp_path: Path) -> None:
        nb = Notebook.create("course-a", root=tmp_path)
        (nb.sources_dir / "week-01").mkdir()
        (nb.sources_dir / "week-01" / "lec.pdf").write_text("pdf content")
        (nb.sources_dir / "week-01" / "~$lock.pptx").write_text("lock file")
        (nb.sources_dir / ".hidden").write_text("hidden")
        (nb.sources_dir / ".hidden.txt").write_text("hidden txt")
        (nb.sources_dir / "notes.md").write_text("markdown")
        (nb.sources_dir / "ignored.mp4").write_text("not supported")

        files = nb.source_files()
        names = sorted(f.name for f in files)
        assert names == ["lec.pdf", "notes.md"]

    def test_sorted(self, tmp_path: Path) -> None:
        nb = Notebook.create("course-a", root=tmp_path)
        (nb.sources_dir / "b.txt").write_text("b")
        (nb.sources_dir / "a.txt").write_text("a")
        files = nb.source_files()
        assert [f.name for f in files] == ["a.txt", "b.txt"]

    def test_no_sources_dir_returns_empty(self, tmp_path: Path) -> None:
        nb = Notebook("course-a", root=tmp_path)
        assert nb.source_files() == []


class TestRelativeSourcePath:
    def test_is_posix_style(self, tmp_path: Path) -> None:
        nb = Notebook.create("course-a", root=tmp_path)
        nested = nb.sources_dir / "week-01" / "lec.pdf"
        nested.parent.mkdir(parents=True, exist_ok=True)
        nested.write_text("content")
        assert nb.relative_source_path(nested) == "week-01/lec.pdf"
