"""Rendering helpers for the Streamlit UI: citations, chat answers, and the
audio overview's file listings. The formatting logic is kept in plain
functions (pure, testable without Streamlit); a thin `render_*` wrapper
calls into `streamlit` widgets.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from notecast.chat.models import Citation
    from notecast.index.service import IndexReport
    from notecast.notebook import Notebook


# --- Citations -------------------------------------------------------------


def citation_heading(citation: Citation) -> str:
    """A one-line heading for a citation, e.g.
    `[1] week-02/lecture.pdf — p. 4` (course) or `[2] Some Page Title` (web).
    """
    if citation.kind == "course":
        location = f" — {citation.location_label}" if citation.location_label else ""
        source = citation.source_path or ""
        return f"[{citation.n}] {source}{location}"
    title = citation.title or citation.url or "web result"
    return f"[{citation.n}] {title}"


def citation_body(citation: Citation) -> str:
    """The expandable body for one citation: the quoted passage for a
    course citation, or the link for a web citation.
    """
    if citation.kind == "course":
        header = f"**{citation.header}**\n\n" if citation.header else ""
        quote = citation.cited_text.strip()
        return f"{header}> {quote}" if quote else header.strip() or "(no excerpt)"
    if citation.url:
        return f"[{citation.url}]({citation.url})"
    return "(no link)"


def format_index_summary(report: IndexReport) -> str:
    """A one-line human summary of an `index_notebook` run's `IndexReport`."""
    ingest = report.ingest
    parts = [
        f"added {len(ingest.added)}",
        f"updated {len(ingest.updated)}",
        f"unchanged {len(ingest.unchanged)}",
        f"removed {len(ingest.removed)}",
    ]
    if ingest.failed:
        parts.append(f"failed {len(ingest.failed)}")
    if report.repaired:
        parts.append(f"repaired {len(report.repaired)}")
    summary = ", ".join(parts)
    return f"{summary}. Total chunks in index: {report.total_chunks}."


# --- Audio overview file listings -------------------------------------------


def list_episodes(nb: Notebook) -> list[dict]:
    """Previously rendered episodes in `nb.audio_dir`: every `*.mp3` with a
    matching `.md` transcript, newest first.
    """
    audio_dir = nb.audio_dir
    if not audio_dir.exists():
        return []
    episodes = []
    for mp3_path in sorted(audio_dir.glob("*.mp3")):
        md_path = mp3_path.with_suffix(".md")
        episodes.append(
            {
                "title": mp3_path.stem,
                "mp3_path": mp3_path,
                "transcript_path": md_path if md_path.exists() else None,
                "mtime": mp3_path.stat().st_mtime,
            }
        )
    episodes.sort(key=lambda e: e["mtime"], reverse=True)
    return episodes


def list_saved_scripts(nb: Notebook) -> list[Path]:
    """Every saved `*.script.json` under `nb.audio_dir`, newest first."""
    audio_dir = nb.audio_dir
    if not audio_dir.exists():
        return []
    scripts = sorted(audio_dir.glob("*.script.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return scripts


__all__ = [
    "citation_body",
    "citation_heading",
    "format_index_summary",
    "list_episodes",
    "list_saved_scripts",
]
