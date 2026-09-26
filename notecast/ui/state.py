"""Pure helpers backing the Streamlit UI, plus small session-state
conveniences. Everything here is unit-testable without Streamlit running;
`app.py` wires these into widgets.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from notecast.chat.pricing import estimate_cost
from notecast.chat.session import ChatSession
from notecast.ingest import manifest as manifest_mod
from notecast.ingest.course import CourseConfig, infer_week, load_course_config
from notecast.models import SearchFilters

if TYPE_CHECKING:
    from notecast.config import Settings
    from notecast.notebook import Notebook

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
# \w matches unicode letters/digits/underscore by default for str patterns,
# so accented and non-Latin filenames keep their letters instead of being
# flattened to underscores.
_UNSAFE_FILENAME_RE = re.compile(r"[^\w.\-]+", re.UNICODE)
_MAX_FILENAME_LENGTH = 120
# Windows reserved device names (case-insensitive) -- writing a file with one
# of these stems fails outright on Windows, even with an extension.
_RESERVED_WINDOWS_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


# --- Notebook slug / upload helpers -------------------------------------


def validate_new_slug(slug: str, existing_slugs: list[str]) -> str | None:
    """Return an error message for `slug`, or None if it's a valid, unused
    notebook slug.
    """
    slug = (slug or "").strip()
    if not slug:
        return "Enter a slug."
    if not _SLUG_RE.match(slug):
        return (
            "Slug must be lowercase letters, digits and hyphens, starting with a letter or digit."
        )
    if slug in existing_slugs:
        return f"Notebook {slug!r} already exists."
    return None


def sanitize_filename(name: str) -> str:
    """Make an uploaded filename safe to write to disk: keep only its
    basename (unicode letters/digits preserved), replace unsafe characters,
    dodge Windows-reserved device names, cap the length, and never return
    an empty string.
    """
    base = unicodedata.normalize("NFC", Path(name).name.strip())
    base = _UNSAFE_FILENAME_RE.sub("_", base)
    base = base.strip("._")
    if not base:
        return "file"

    stem = Path(base).stem or base
    ext = Path(base).suffix
    if stem.upper() in _RESERVED_WINDOWS_NAMES:
        stem = f"{stem}_"
    if len(stem) + len(ext) > _MAX_FILENAME_LENGTH:
        stem = stem[: max(1, _MAX_FILENAME_LENGTH - len(ext))]
    return f"{stem}{ext}" or "file"


def week_folder_name(week: int) -> str:
    """The `sources/` subfolder name for a given week number, e.g. `week-03`."""
    return f"week-{week:02d}"


def unique_destination(dest_dir: Path, filename: str) -> Path:
    """A path under `dest_dir` for `filename` that doesn't already exist,
    adding a numeric suffix (` (2)`, ` (3)`, ...) if it does. Never
    overwrites an existing file.
    """
    candidate = dest_dir / filename
    if not candidate.exists():
        return candidate
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    n = 2
    while True:
        candidate = dest_dir / f"{stem} ({n}){suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def save_uploaded_file(nb: Notebook, data: bytes, filename: str, week: int | None) -> Path:
    """Write an uploaded file's bytes into `nb.sources_dir`, under a
    `week-NN/` folder if `week` is given, sanitising the filename and never
    overwriting an existing file. Returns the path written.
    """
    safe_name = sanitize_filename(filename)
    dest_dir = nb.sources_dir / week_folder_name(week) if week else nb.sources_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = unique_destination(dest_dir, safe_name)
    dest.write_bytes(data)
    return dest


# --- Source listing ------------------------------------------------------


def group_sources_by_week(nb: Notebook) -> list[tuple[int | None, list[tuple[str, int]]]]:
    """Group `nb`'s source files by inferred week, with chunk counts from
    the manifest (0 if a file hasn't been ingested yet). Sorted by week
    (files with no week last), then by source path within each week.
    """
    config = load_course_config(nb)
    m = manifest_mod.load_manifest(nb)
    by_week: dict[int | None, list[tuple[str, int]]] = {}
    for path in nb.source_files():
        source_path = nb.relative_source_path(path)
        week = infer_week(source_path, config)
        entry = m.entries.get(source_path)
        chunk_count = entry.chunk_count if entry else 0
        by_week.setdefault(week, []).append((source_path, chunk_count))

    def _sort_key(item: tuple[int | None, list[tuple[str, int]]]) -> tuple[int, int]:
        week = item[0]
        return (1, 0) if week is None else (0, week)

    result = sorted(by_week.items(), key=_sort_key)
    for _week, entries in result:
        entries.sort(key=lambda e: e[0])
    return result


# --- Search filters --------------------------------------------------------


def build_search_filters(weeks: list[int]) -> SearchFilters | None:
    """Build a `SearchFilters` from a week multiselect, or None for "no
    filter".
    """
    if not weeks:
        return None
    return SearchFilters(weeks=sorted(set(weeks)))


def deep_scope_key(weeks: list[int]) -> tuple:
    """A hashable key identifying a deep-mode scope, for tracking which
    scope's cost estimate the user has confirmed. Distinct from `None` (used
    as "nothing confirmed yet") even for the whole-course scope (no weeks
    selected), so an unconfirmed whole-course scope is never mistaken for a
    confirmed one.
    """
    return tuple(sorted(set(weeks))) if weeks else ("__all__",)


# --- Cost / estimate formatting ------------------------------------------


def format_usd(usd: float | None) -> str:
    if usd is None:
        return "n/a"
    if usd < 0.01:
        return f"${usd:.4f}"
    return f"${usd:.2f}"


def deep_cost_breakdown(
    session: ChatSession, filters: SearchFilters | None
) -> tuple[int, float, float]:
    """Tokens, first-question cost and per-follow-up cost for deep mode
    over `filters`'s scope. Raises `ChatError` if the scope is empty/too big
    (see `ChatSession.estimate_deep_cost`).
    """
    tokens, _total = session.estimate_deep_cost(filters)
    model = session.settings.deep_model
    first_cost = estimate_cost(model, {"cache_write_tokens": tokens}) or 0.0
    follow_up_cost = estimate_cost(model, {"cache_read_tokens": tokens}) or 0.0
    return tokens, first_cost, follow_up_cost


def format_deep_estimate(tokens: int, first_cost: float, follow_up_cost: float) -> str:
    return (
        f"Deep mode will send ~{tokens:,} tokens of material: "
        f"~{format_usd(first_cost)} for the first question, "
        f"~{format_usd(follow_up_cost)} for each follow-up in this scope."
    )


def format_overview_estimate(
    total_tokens: int, n_sources: int, low_usd: float, high_usd: float
) -> str:
    return (
        f"~{total_tokens:,} tokens of material across {n_sources} source(s). "
        f"Estimated cost: {format_usd(low_usd)}–{format_usd(high_usd)}."
    )


# --- Notebook / chat session wiring ---------------------------------------


def default_chat_session_factory(retriever: Any, **kwargs: Any) -> ChatSession:
    """The default way to build a `ChatSession`. Overridden in tests via
    `notecast.ui.state.chat_session_factory` so a fake session can be
    injected without a real API key or network access.
    """
    return ChatSession(retriever, **kwargs)


# Module-level lookup, deliberately mutable: tests monkeypatch this
# attribute (not the imported name) so `app.py` must call it as
# `state.chat_session_factory(...)` for the patch to take effect.
chat_session_factory = default_chat_session_factory


def filter_chunks(chunks: list, filters: SearchFilters | None) -> list:
    """Apply a `SearchFilters` to an already-loaded chunk list, for deep
    mode's `chunk_source` callback.
    """
    if filters is None:
        return chunks
    result = chunks
    if filters.weeks:
        weeks = set(filters.weeks)
        result = [c for c in result if c.week in weeks]
    if filters.source_types:
        types = set(filters.source_types)
        result = [c for c in result if c.source_type in types]
    if filters.source_paths:
        paths = set(filters.source_paths)
        result = [c for c in result if c.source_path in paths]
    return result


def api_key_configured(settings: Settings) -> bool:
    return bool(settings.anthropic_api_key)


__all__ = [
    "CourseConfig",
    "api_key_configured",
    "build_search_filters",
    "chat_session_factory",
    "default_chat_session_factory",
    "deep_cost_breakdown",
    "deep_scope_key",
    "filter_chunks",
    "format_deep_estimate",
    "format_overview_estimate",
    "format_usd",
    "group_sources_by_week",
    "sanitize_filename",
    "save_uploaded_file",
    "unique_destination",
    "validate_new_slug",
    "week_folder_name",
]
