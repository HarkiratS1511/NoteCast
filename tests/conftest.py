"""Shared pytest fixtures: a temporary notebooks root so tests never touch
the real notebooks/ directory (which holds real, copyrighted course files).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from notecast.config import get_settings


@pytest.fixture(autouse=True)
def _default_provider_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most tests were written against the Anthropic provider. Pin it via the
    environment; tests for AgentAUS pass `provider="agentaus"` explicitly.
    """
    monkeypatch.setenv("NOTECAST_PROVIDER", "anthropic")


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
