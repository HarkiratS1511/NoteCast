"""A Notebook is one course's folder on disk: its raw source files, its
extracted text, its vector index and its generated audio. This module knows
how to find, create and list notebooks, and how to walk a notebook's
source files — it doesn't parse or index anything itself.
"""

from __future__ import annotations

import re
from pathlib import Path

from notecast.config import get_settings
from notecast.models import SourceType

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def _is_valid_slug(slug: str) -> bool:
    return bool(_SLUG_RE.match(slug))


class Notebook:
    """One course's data folder: `<root>/<slug>/`."""

    def __init__(self, slug: str, root: Path | None = None) -> None:
        if not _is_valid_slug(slug):
            raise ValueError(
                "Notebook slug must be 1-64 characters of lowercase letters, "
                f"digits and hyphens, and not start with a hyphen: {slug!r}"
            )
        self.slug = slug
        self.root = root if root is not None else get_settings().notebooks_dir

    @property
    def path(self) -> Path:
        return self.root / self.slug

    @property
    def sources_dir(self) -> Path:
        return self.path / "sources"

    @property
    def processed_dir(self) -> Path:
        return self.path / "processed"

    @property
    def index_dir(self) -> Path:
        return self.path / "index.lancedb"

    @property
    def audio_dir(self) -> Path:
        return self.path / "audio"

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    @property
    def config_path(self) -> Path:
        return self.path / "course.yaml"

    def exists(self) -> bool:
        """Whether this notebook has already been created."""
        return self.sources_dir.exists()

    def ensure_dirs(self) -> None:
        """Create the notebook's directories if they don't exist yet."""
        for directory in (self.sources_dir, self.processed_dir, self.audio_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(cls, slug: str, root: Path | None = None) -> Notebook:
        """Create a brand-new notebook. Raises ValueError if one already
        exists with this slug.
        """
        notebook = cls(slug, root=root)
        if notebook.exists():
            raise ValueError(f"Notebook {slug!r} already exists at {notebook.path}")
        notebook.ensure_dirs()
        return notebook

    @classmethod
    def list_all(cls, root: Path | None = None) -> list[Notebook]:
        """List every notebook under `root` (default: the configured
        notebooks directory), sorted by slug. Only directories with a
        `sources/` subdirectory count, and names that aren't valid slugs
        are silently skipped.
        """
        base = root if root is not None else get_settings().notebooks_dir
        if not base.exists():
            return []
        notebooks: list[Notebook] = []
        for entry in base.iterdir():
            if not entry.is_dir() or not _is_valid_slug(entry.name):
                continue
            if not (entry / "sources").exists():
                continue
            notebooks.append(cls(entry.name, root=base))
        return sorted(notebooks, key=lambda nb: nb.slug)

    def source_files(self) -> list[Path]:
        """All supported source files under sources/, recursively, sorted.
        Skips hidden files (dotfiles) and Office lock files ("~$...").
        """
        if not self.sources_dir.exists():
            return []
        suffixes = {f".{member.value}" for member in SourceType}
        files: list[Path] = []
        for p in self.sources_dir.rglob("*"):
            if not p.is_file():
                continue
            if p.name.startswith(".") or p.name.startswith("~$"):
                continue
            if p.suffix.lower() not in suffixes:
                continue
            files.append(p)
        return sorted(files)

    def relative_source_path(self, p: Path) -> str:
        """`p`'s path relative to this notebook's sources directory, as a
        POSIX-style string (forward slashes, even on Windows).
        """
        return p.relative_to(self.sources_dir).as_posix()
