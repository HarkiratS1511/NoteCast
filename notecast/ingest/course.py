"""Per-course configuration: display name and week/topic inference from
source paths, with an optional override in `notebooks/<course>/course.yaml`.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from notecast.notebook import Notebook

_WORD_TOKEN = re.compile(r"(?i)^(?:week|wk|w)$")
_WORD_NUM_TOKEN = re.compile(r"(?i)^(?:week|wk|w)(\d{1,2})$")
_NUM_TOKEN = re.compile(r"^(\d{1,2})$")


class CourseConfig(BaseModel):
    """Optional per-course overrides read from `course.yaml`."""

    name: str | None = None
    weeks: dict[str, int] = Field(default_factory=dict)
    topics: dict[str, str] = Field(default_factory=dict)


def load_course_config(nb: Notebook) -> CourseConfig:
    """Read `nb.config_path` (course.yaml). Missing or empty -> defaults.
    Invalid YAML or shape -> ValueError with a friendly message.
    """
    path = nb.config_path
    if not path.exists():
        return CourseConfig()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not read course config at {path}: {exc}") from exc
    if raw is None:
        return CourseConfig()
    if not isinstance(raw, dict):
        raise ValueError(f"Course config at {path} must be a mapping (got {type(raw).__name__})")
    try:
        return CourseConfig.model_validate(raw)
    except Exception as exc:
        raise ValueError(f"Course config at {path} is invalid: {exc}") from exc


def _match_week_component(component: str) -> int | None:
    """Try to find a week number token within one path component (a folder
    name or a filename stem, already delimiter-tokenized).
    """
    tokens = re.split(r"[_\-\s]+", component)
    for i, tok in enumerate(tokens):
        if not tok:
            continue
        m = _WORD_NUM_TOKEN.match(tok)
        if m:
            return int(m.group(1))
        if _WORD_TOKEN.match(tok) and i + 1 < len(tokens):
            num_match = _NUM_TOKEN.match(tokens[i + 1])
            if num_match:
                return int(num_match.group(1))
    return None


def _config_override(source_path: str, mapping: dict[str, int | str]) -> int | str | None:
    for pattern, value in mapping.items():
        if fnmatch.fnmatch(source_path, pattern) or source_path.startswith(pattern):
            return value
    return None


def infer_week(source_path: str, config: CourseConfig | None = None) -> int | None:
    """Infer a week number from `source_path`. Config override wins; then
    folders (checked before the filename); then the filename itself.
    """
    if config is not None:
        override = _config_override(source_path, config.weeks)  # type: ignore[arg-type]
        if override is not None:
            return int(override)

    parts = source_path.split("/")
    folders, filename = parts[:-1], parts[-1]

    for folder in folders:
        week = _match_week_component(folder)
        if week is not None:
            return week

    stem = Path(filename).stem
    return _match_week_component(stem)


def infer_topic(source_path: str, config: CourseConfig | None = None) -> str | None:
    """Infer a topic from `source_path`. Only ever comes from config
    overrides (there is no reliable filename convention for topics).
    """
    if config is None:
        return None
    override = _config_override(source_path, config.topics)  # type: ignore[arg-type]
    return str(override) if override is not None else None


def display_name(nb: Notebook, config: CourseConfig | None = None) -> str:
    """The course's human-readable name: config override, else the slug."""
    if config is not None and config.name:
        return config.name
    return nb.slug
