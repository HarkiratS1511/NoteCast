"""Tests for the pure helpers in notecast.ui.state and notecast.ui.components
-- no Streamlit involved.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from notecast.chat.models import Citation
from notecast.index.service import IndexReport
from notecast.ingest.pipeline import IngestReport
from notecast.models import Chunk, Location, Manifest, ManifestEntry, SearchFilters, SourceType
from notecast.notebook import Notebook
from notecast.ui import components, state

# --- state.py --------------------------------------------------------------


def test_validate_new_slug_accepts_valid() -> None:
    assert state.validate_new_slug("comp3000", []) is None


def test_validate_new_slug_rejects_empty() -> None:
    assert state.validate_new_slug("", []) is not None


def test_validate_new_slug_rejects_invalid_chars() -> None:
    assert state.validate_new_slug("COMP 3000!", []) is not None


def test_validate_new_slug_rejects_duplicate() -> None:
    assert state.validate_new_slug("comp3000", ["comp3000"]) is not None


def test_sanitize_filename_strips_path_and_unsafe_chars() -> None:
    assert state.sanitize_filename("../../etc/passwd") == "passwd"
    assert state.sanitize_filename("my file (1)!.pdf") == "my_file_1_.pdf"


def test_sanitize_filename_never_empty() -> None:
    assert state.sanitize_filename("...") == "file"


def test_sanitize_filename_keeps_unicode_letters() -> None:
    assert state.sanitize_filename("café notes.pdf") == "café_notes.pdf"


def test_sanitize_filename_dodges_windows_reserved_names() -> None:
    assert state.sanitize_filename("CON.txt") == "CON_.txt"
    assert state.sanitize_filename("com1.pdf") == "com1_.pdf"
    assert state.sanitize_filename("lecture.pdf") == "lecture.pdf"


def test_sanitize_filename_truncates_long_names() -> None:
    name = ("a" * 300) + ".pdf"
    result = state.sanitize_filename(name)
    assert len(result) <= 120
    assert result.endswith(".pdf")


def test_week_folder_name() -> None:
    assert state.week_folder_name(3) == "week-03"
    assert state.week_folder_name(12) == "week-12"


def test_unique_destination_no_collision(tmp_path: Path) -> None:
    dest = state.unique_destination(tmp_path, "notes.pdf")
    assert dest == tmp_path / "notes.pdf"


def test_unique_destination_adds_suffix_on_collision(tmp_path: Path) -> None:
    (tmp_path / "notes.pdf").write_text("existing")
    dest = state.unique_destination(tmp_path, "notes.pdf")
    assert dest == tmp_path / "notes (2).pdf"
    assert not dest.exists()


def test_unique_destination_never_overwrites_across_multiple_collisions(tmp_path: Path) -> None:
    (tmp_path / "notes.pdf").write_text("a")
    (tmp_path / "notes (2).pdf").write_text("b")
    dest = state.unique_destination(tmp_path, "notes.pdf")
    assert dest == tmp_path / "notes (3).pdf"


def test_save_uploaded_file_no_week(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("comp1000")
    dest = state.save_uploaded_file(nb, b"hello", "lecture 1.pdf", None)
    assert dest == nb.sources_dir / "lecture_1.pdf"
    assert dest.read_bytes() == b"hello"


def test_save_uploaded_file_with_week(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("comp1000")
    dest = state.save_uploaded_file(nb, b"hello", "lecture.pdf", 3)
    assert dest == nb.sources_dir / "week-03" / "lecture.pdf"
    assert dest.read_bytes() == b"hello"


def test_save_uploaded_file_never_overwrites(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("comp1000")
    first = state.save_uploaded_file(nb, b"one", "notes.pdf", None)
    second = state.save_uploaded_file(nb, b"two", "notes.pdf", None)
    assert first != second
    assert first.read_bytes() == b"one"
    assert second.read_bytes() == b"two"


def _write_manifest(nb: Notebook, entries: dict[str, int]) -> None:
    from notecast.ingest import manifest as manifest_mod

    m = Manifest(
        entries={
            path: ManifestEntry(
                source_path=path,
                sha256="x" * 64,
                source_type=SourceType.TXT,
                chunk_count=count,
                ingested_at=datetime.now(UTC),
            )
            for path, count in entries.items()
        }
    )
    manifest_mod.save_manifest(nb, m)


def test_group_sources_by_week(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("comp1000")
    nb.sources_dir.mkdir(exist_ok=True)
    (nb.sources_dir / "week-01").mkdir()
    (nb.sources_dir / "week-01" / "a.txt").write_text("a")
    (nb.sources_dir / "misc.md").write_text("m")
    _write_manifest(nb, {"week-01/a.txt": 5})

    grouped = state.group_sources_by_week(nb)
    weeks = [w for w, _ in grouped]
    assert weeks == [1, None]
    assert grouped[0][1] == [("week-01/a.txt", 5)]
    assert grouped[1][1] == [("misc.md", 0)]


def test_build_search_filters_empty_is_none() -> None:
    assert state.build_search_filters([]) is None


def test_build_search_filters_dedupes_and_sorts() -> None:
    filters = state.build_search_filters([3, 1, 1])
    assert filters == SearchFilters(weeks=[1, 3])


def test_deep_scope_key_whole_course_is_not_none() -> None:
    key = state.deep_scope_key([])
    assert key is not None
    assert key != (None,)


def test_deep_scope_key_distinguishes_whole_course_from_weeks() -> None:
    assert state.deep_scope_key([]) != state.deep_scope_key([1])


def test_deep_scope_key_dedupes_and_sorts() -> None:
    assert state.deep_scope_key([3, 1, 1]) == state.deep_scope_key([1, 3])


def test_format_usd_none() -> None:
    assert state.format_usd(None) == "n/a"


def test_format_usd_small_amount_shows_more_precision() -> None:
    assert state.format_usd(0.0032) == "$0.0032"


def test_format_usd_normal_amount() -> None:
    assert state.format_usd(1.5) == "$1.50"


def test_format_deep_estimate_mentions_tokens_and_costs() -> None:
    text = state.format_deep_estimate(12_345, 0.5, 0.05)
    assert "12,345" in text
    assert "$0.50" in text
    assert "$0.05" in text


def test_format_overview_estimate() -> None:
    text = state.format_overview_estimate(10_000, 3, 0.1, 0.2)
    assert "10,000" in text
    assert "3 source" in text
    assert "$0.10" in text
    assert "$0.20" in text


def test_format_deep_estimate_shows_na_hint_instead_of_zero_when_unpriced() -> None:
    text = state.format_deep_estimate(12_345, None, None)
    assert "$0.00" not in text
    assert "n/a" in text
    assert "NOTECAST_AGENTAUS_PRICE" in text


def test_format_overview_estimate_shows_na_hint_instead_of_zero_when_unpriced() -> None:
    text = state.format_overview_estimate(10_000, 3, None, None)
    assert "$0.00" not in text
    assert "n/a" in text
    assert "NOTECAST_AGENTAUS_PRICE" in text


def _chunk(source_path: str, week: int | None, source_type: SourceType = SourceType.TXT) -> Chunk:
    return Chunk(
        chunk_id=f"{source_path}-{week}",
        course="test",
        source_path=source_path,
        source_type=source_type,
        ordinal=0,
        text="hello",
        location=Location(),
        week=week,
    )


def test_filter_chunks_no_filters_returns_all() -> None:
    chunks = [_chunk("a.txt", 1), _chunk("b.txt", 2)]
    assert state.filter_chunks(chunks, None) == chunks


def test_filter_chunks_by_week() -> None:
    chunks = [_chunk("a.txt", 1), _chunk("b.txt", 2)]
    filtered = state.filter_chunks(chunks, SearchFilters(weeks=[2]))
    assert [c.source_path for c in filtered] == ["b.txt"]


def test_filter_chunks_by_source_type() -> None:
    chunks = [_chunk("a.txt", 1, SourceType.TXT), _chunk("b.md", 1, SourceType.MD)]
    filtered = state.filter_chunks(chunks, SearchFilters(source_types=[SourceType.MD]))
    assert [c.source_path for c in filtered] == ["b.md"]


def test_filter_chunks_by_source_paths() -> None:
    chunks = [_chunk("a.txt", 1), _chunk("b.txt", 1)]
    filtered = state.filter_chunks(chunks, SearchFilters(source_paths=["a.txt"]))
    assert [c.source_path for c in filtered] == ["a.txt"]


def test_api_key_configured(tmp_notebooks_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from notecast.config import Settings

    assert state.api_key_configured(Settings(anthropic_api_key="sk-ant-x")) is True
    assert state.api_key_configured(Settings(anthropic_api_key=None)) is False


def test_api_key_configured_agentaus_needs_key_url_and_model() -> None:
    from notecast.config import Settings

    full = Settings(
        _env_file=None,
        provider="agentaus",
        agentaus_api_key="key",
        agentaus_base_url="https://example.test/v1",
        agentaus_model="trellis-large",
    )
    assert state.api_key_configured(full) is True

    missing_url = Settings(
        _env_file=None,
        provider="agentaus",
        agentaus_api_key="key",
        agentaus_base_url=None,
        agentaus_model="trellis-large",
    )
    assert state.api_key_configured(missing_url) is False


def test_missing_api_key_hint_provider_aware() -> None:
    from notecast.config import Settings

    anthropic_hint = state.missing_api_key_hint(Settings(_env_file=None, provider="anthropic"))
    assert "ANTHROPIC_API_KEY" in anthropic_hint

    agentaus_hint = state.missing_api_key_hint(Settings(_env_file=None, provider="agentaus"))
    assert "AGENTAUS_API_KEY" in agentaus_hint
    assert "AGENTAUS_BASE_URL" in agentaus_hint
    assert "NOTECAST_AGENTAUS_MODEL" in agentaus_hint


def test_provider_label() -> None:
    from notecast.config import Settings

    assert state.provider_label(Settings(_env_file=None, provider="anthropic")) == "Claude"
    assert state.provider_label(Settings(_env_file=None, provider="agentaus")) == "AgentAUS"


def test_deep_cost_breakdown_agentaus_no_cache_discount() -> None:
    from notecast.config import Settings

    class _FakeDeepSession:
        settings = Settings(
            _env_file=None,
            provider="agentaus",
            agentaus_api_key="key",
            agentaus_base_url="https://example.test/v1",
            agentaus_model="trellis-large",
            agentaus_price_input_per_mtok=3.0,
            agentaus_price_output_per_mtok=15.0,
        )

        def estimate_deep_cost(self, filters=None):  # noqa: ANN001
            return 1_000_000, 0.0

    tokens, first_cost, follow_up_cost = state.deep_cost_breakdown(_FakeDeepSession(), None)
    assert tokens == 1_000_000
    assert first_cost == follow_up_cost == pytest.approx(3.0)


def test_deep_cost_breakdown_agentaus_none_when_unpriced() -> None:
    from notecast.config import Settings

    class _FakeDeepSession:
        settings = Settings(
            _env_file=None,
            provider="agentaus",
            agentaus_api_key="key",
            agentaus_base_url="https://example.test/v1",
            agentaus_model="trellis-large",
        )

        def estimate_deep_cost(self, filters=None):  # noqa: ANN001
            return 1_000, 0.0

    _tokens, first_cost, follow_up_cost = state.deep_cost_breakdown(_FakeDeepSession(), None)
    assert first_cost is None
    assert follow_up_cost is None


# --- components.py -----------------------------------------------------


def test_citation_heading_course() -> None:
    citation = Citation(
        n=1, kind="course", source_path="week-01/a.pdf", location_label="p. 4", cited_text="quote"
    )
    assert components.citation_heading(citation) == "[1] week-01/a.pdf — p. 4"


def test_citation_heading_web() -> None:
    citation = Citation(n=2, kind="web", title="Some Page", url="https://example.com")
    assert components.citation_heading(citation) == "[2] Some Page"


def test_citation_body_course_includes_quote() -> None:
    citation = Citation(
        n=1, kind="course", source_path="a.pdf", header="Intro", cited_text="the quoted bit"
    )
    body = components.citation_body(citation)
    assert "Intro" in body
    assert "the quoted bit" in body


def test_citation_body_web_includes_link() -> None:
    citation = Citation(n=1, kind="web", url="https://example.com", title="Example")
    assert "https://example.com" in components.citation_body(citation)


def test_format_index_summary() -> None:
    report = IndexReport(
        ingest=IngestReport(
            added=["a.txt"], updated=[], unchanged=["b.txt"], removed=[], failed={}
        ),
        indexed_chunks=5,
        total_chunks=42,
    )
    summary = components.format_index_summary(report)
    assert "added 1" in summary
    assert "unchanged 1" in summary
    assert "42" in summary


def test_format_index_summary_mentions_failures() -> None:
    report = IndexReport(
        ingest=IngestReport(
            added=[], updated=[], unchanged=[], removed=[], failed={"x.pdf": "boom"}
        ),
        total_chunks=0,
    )
    summary = components.format_index_summary(report)
    assert "failed 1" in summary


def test_list_episodes_pairs_mp3_and_transcript(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("comp1000")
    nb.audio_dir.mkdir(parents=True, exist_ok=True)
    (nb.audio_dir / "week-1-overview.mp3").write_bytes(b"data")
    (nb.audio_dir / "week-1-overview.md").write_text("# transcript")
    (nb.audio_dir / "no-transcript.mp3").write_bytes(b"data")

    episodes = components.list_episodes(nb)
    titles = {e["title"] for e in episodes}
    assert titles == {"week-1-overview", "no-transcript"}
    by_title = {e["title"]: e for e in episodes}
    assert by_title["week-1-overview"]["transcript_path"] is not None
    assert by_title["no-transcript"]["transcript_path"] is None


def test_list_episodes_empty_when_no_audio_dir(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("comp1000")
    assert components.list_episodes(nb) == []


def test_list_saved_scripts(tmp_notebooks_root: Path) -> None:
    nb = Notebook.create("comp1000")
    nb.audio_dir.mkdir(parents=True, exist_ok=True)
    (nb.audio_dir / "comp1000-20260101-1200.script.json").write_text("{}")
    (nb.audio_dir / "notascript.json").write_text("{}")

    scripts = components.list_saved_scripts(nb)
    assert [p.name for p in scripts] == ["comp1000-20260101-1200.script.json"]
