"""Tests for course.yaml loading and week/topic inference."""

from __future__ import annotations

from pathlib import Path

import pytest

from notecast.ingest.course import (
    CourseConfig,
    display_name,
    infer_topic,
    infer_week,
    load_course_config,
)
from notecast.notebook import Notebook

WEEK_CASES = [
    ("week-03/lecture.pdf", 3),
    ("week_3/lecture.pdf", 3),
    ("Week 3/lecture.pdf", 3),
    ("wk3/lecture.pdf", 3),
    ("w03/lecture.pdf", 3),
    ("W3/lecture.pdf", 3),
    ("DA_2026_S2_W3_v1.0.pdf", 3),
    ("2026-notes.pdf", None),
    ("random-folder/S2-notes.pdf", None),
    ("intro.pdf", None),
]


@pytest.mark.parametrize(("source_path", "expected"), WEEK_CASES)
def test_infer_week_table(source_path: str, expected: int | None) -> None:
    assert infer_week(source_path) == expected


def test_infer_week_folder_beats_filename() -> None:
    # Folder says week 3, filename says something that looks like week 5.
    assert infer_week("week-03/w5-notes.pdf") == 3


def test_infer_week_config_override_wins() -> None:
    config = CourseConfig(weeks={"week-03/*": 99})
    assert infer_week("week-03/lecture.pdf", config) == 99


def test_infer_week_no_config_falls_back_to_inference() -> None:
    config = CourseConfig(weeks={"other/*": 99})
    assert infer_week("week-03/lecture.pdf", config) == 3


def test_infer_topic_config_prefix_override_does_not_match_prefix_collision() -> None:
    # "week-1" must not match "week-10/..." as a prefix.
    config = CourseConfig(topics={"week-1": "graphs"})
    assert infer_topic("week-10/lecture.pdf", config) is None
    assert infer_topic("week-1/lecture.pdf", config) == "graphs"
    assert infer_topic("week-1", config) == "graphs"


def test_infer_topic_config_prefix_override_with_trailing_slash() -> None:
    config = CourseConfig(topics={"week-1/": "graphs"})
    assert infer_topic("week-10/lecture.pdf", config) is None
    assert infer_topic("week-1/lecture.pdf", config) == "graphs"


def test_infer_week_config_prefix_override_does_not_match_prefix_collision() -> None:
    # An override on "week-1" must not match "week-10/..." as a prefix —
    # but since it also doesn't match exactly, inference falls through to
    # the normal folder-name detection, which legitimately reads "week-10"
    # as week 10 (a different mechanism from the override).
    config = CourseConfig(weeks={"week-1": 1})
    assert infer_week("week-10/lecture.pdf", config) == 10
    assert infer_week("week-1/lecture.pdf", config) == 1


def test_infer_topic_only_via_config() -> None:
    assert infer_topic("week-03/lecture.pdf") is None
    config = CourseConfig(topics={"week-03/*": "smoothing"})
    assert infer_topic("week-03/lecture.pdf", config) == "smoothing"


def test_load_course_config_missing_file(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("my-course", root=tmp_notebooks_root)
    config = load_course_config(nb)
    assert config == CourseConfig()


def test_load_course_config_empty_file(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("my-course", root=tmp_notebooks_root)
    nb.config_path.write_text("")
    config = load_course_config(nb)
    assert config == CourseConfig()


def test_load_course_config_reads_values(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("my-course", root=tmp_notebooks_root)
    nb.config_path.write_text(
        "name: COMP4650 Document Analysis\nweeks:\n  special/*: 7\ntopics:\n  special/*: graphs\n"
    )
    config = load_course_config(nb)
    assert config.name == "COMP4650 Document Analysis"
    assert config.weeks == {"special/*": 7}
    assert config.topics == {"special/*": "graphs"}


def test_load_course_config_invalid_yaml(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("my-course", root=tmp_notebooks_root)
    nb.config_path.write_text("weeks: [this, is, a, list, not, a, mapping]\n- broken\n")
    with pytest.raises(ValueError, match="course.yaml|Course config"):
        load_course_config(nb)


def test_load_course_config_not_a_mapping(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("my-course", root=tmp_notebooks_root)
    nb.config_path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError):
        load_course_config(nb)


def test_display_name_uses_config_name(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("my-course", root=tmp_notebooks_root)
    config = CourseConfig(name="COMP4650 Document Analysis")
    assert display_name(nb, config) == "COMP4650 Document Analysis"


def test_display_name_falls_back_to_slug(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("my-course", root=tmp_notebooks_root)
    assert display_name(nb, CourseConfig()) == "my-course"
    assert display_name(nb, None) == "my-course"
