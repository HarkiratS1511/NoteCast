"""Shared pytest fixtures: a temporary notebooks root so tests never touch
the real notebooks/ directory (which holds real, copyrighted course files).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from notecast.config import get_settings


@pytest.fixture
def tmp_notebooks_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A temporary directory to use as the notebooks root, with settings
    pointed at it via the environment (cleared afterwards).
    """
    root = tmp_path / "notebooks"
    root.mkdir()
    monkeypatch.setenv("NOTECAST_NOTEBOOKS_DIR", str(root))
    get_settings.cache_clear()
    yield root
    get_settings.cache_clear()
