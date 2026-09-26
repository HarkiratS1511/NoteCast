"""Tests for notecast.index.service: wiring ingest into the search index."""

from __future__ import annotations

from pathlib import Path

import pytest

from notecast.index.embedder import HashingEmbedder
from notecast.index.service import IndexNotBuiltError, get_retriever, index_notebook, open_store
from notecast.notebook import Notebook


class CountingEmbedder(HashingEmbedder):
    """A HashingEmbedder that counts how many texts it's asked to embed, so
    tests can assert unchanged files aren't re-embedded.
    """

    def __init__(self, dim: int = 32) -> None:
        super().__init__(dim=dim)
        self.embed_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += len(texts)
        return super().embed_documents(texts)


def _make_notebook(root: Path, slug: str = "my-course") -> Notebook:
    return Notebook.create(slug, root=root)


def _write_source(nb: Notebook, name: str, text: str) -> None:
    path = nb.sources_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_first_ingest_indexes_all(tmp_notebooks_root: Path) -> None:
    nb = _make_notebook(tmp_notebooks_root)
    _write_source(nb, "week-01/a.txt", "Add-one smoothing avoids zero probabilities.\n" * 5)
    _write_source(nb, "week-01/b.txt", "PageRank models a random walk over the web graph.\n" * 5)

    embedder = CountingEmbedder()
    result = index_notebook(nb, embedder=embedder)

    assert result.indexed_chunks > 0
    assert result.total_chunks == result.indexed_chunks
    assert result.reindexed is False
    assert embedder.embed_calls == result.indexed_chunks

    store = open_store(nb, embedder=embedder)
    assert store.count() == result.total_chunks


def test_rerun_unchanged_does_not_reembed(tmp_notebooks_root: Path) -> None:
    nb = _make_notebook(tmp_notebooks_root)
    _write_source(nb, "a.txt", "Some stable course content about smoothing.\n" * 5)

    embedder = CountingEmbedder()
    first = index_notebook(nb, embedder=embedder)
    assert embedder.embed_calls == first.indexed_chunks

    calls_before = embedder.embed_calls
    second = index_notebook(nb, embedder=embedder)

    assert second.indexed_chunks == 0
    assert embedder.embed_calls == calls_before
    assert second.total_chunks == first.total_chunks


def test_modifying_a_file_replaces_its_rows(tmp_notebooks_root: Path) -> None:
    nb = _make_notebook(tmp_notebooks_root)
    _write_source(nb, "a.txt", "Original content about pagerank and random walks.\n" * 5)

    embedder = CountingEmbedder()
    index_notebook(nb, embedder=embedder)

    _write_source(nb, "a.txt", "Completely different content about perplexity now.\n" * 5)
    result = index_notebook(nb, embedder=embedder)

    assert result.indexed_chunks > 0

    store = open_store(nb, embedder=embedder)
    chunks = store.all_chunks()
    assert all("perplexity" in c.text.lower() for c in chunks if c.source_path == "a.txt")
    assert not any("pagerank" in c.text.lower() for c in chunks if c.source_path == "a.txt")


def test_deleting_a_file_removes_its_rows(tmp_notebooks_root: Path) -> None:
    nb = _make_notebook(tmp_notebooks_root)
    _write_source(nb, "a.txt", "Content about smoothing techniques in NLP.\n" * 5)
    _write_source(nb, "b.txt", "Content about pagerank in graphs.\n" * 5)

    embedder = CountingEmbedder()
    index_notebook(nb, embedder=embedder)

    (nb.sources_dir / "a.txt").unlink()
    result = index_notebook(nb, embedder=embedder)

    assert result.removed_sources == ["a.txt"]
    store = open_store(nb, embedder=embedder)
    assert store.sources() == {"b.txt"}


def test_reindex_rebuilds(tmp_notebooks_root: Path) -> None:
    nb = _make_notebook(tmp_notebooks_root)
    _write_source(nb, "a.txt", "Content about smoothing techniques in NLP.\n" * 5)

    embedder = CountingEmbedder()
    first = index_notebook(nb, embedder=embedder)

    result = index_notebook(nb, reindex=True, embedder=embedder)
    assert result.reindexed is True
    assert result.total_chunks == first.total_chunks
    assert result.indexed_chunks == first.total_chunks


def test_deleting_index_dir_then_ingest_auto_reindexes(tmp_notebooks_root: Path) -> None:
    import shutil

    nb = _make_notebook(tmp_notebooks_root)
    _write_source(nb, "a.txt", "Content about smoothing techniques in NLP.\n" * 5)

    embedder = CountingEmbedder()
    first = index_notebook(nb, embedder=embedder)
    assert first.total_chunks > 0

    shutil.rmtree(nb.index_dir)

    result = index_notebook(nb, embedder=embedder)
    assert result.reindexed is True
    assert result.total_chunks == first.total_chunks


def test_model_mismatch_triggers_reindex(tmp_notebooks_root: Path) -> None:
    nb = _make_notebook(tmp_notebooks_root)
    _write_source(nb, "a.txt", "Content about smoothing techniques in NLP.\n" * 5)

    embedder_a = CountingEmbedder(dim=32)
    first = index_notebook(nb, embedder=embedder_a)
    assert first.total_chunks > 0

    embedder_b = CountingEmbedder(dim=48)
    result = index_notebook(nb, embedder=embedder_b)
    assert result.reindexed is True
    assert result.total_chunks == first.total_chunks

    store = open_store(nb, embedder=embedder_b)
    assert store.dim == 48


def test_get_retriever_on_unindexed_notebook_raises(tmp_notebooks_root: Path) -> None:
    nb = _make_notebook(tmp_notebooks_root)
    with pytest.raises(IndexNotBuiltError):
        get_retriever(nb, embedder=HashingEmbedder())


class CrashingEmbedder(HashingEmbedder):
    """A HashingEmbedder whose `embed_documents` raises on a chosen call
    number (1-based), to simulate a crash partway through embedding a
    source with more chunks than one embed batch.
    """

    def __init__(self, dim: int = 32, fail_on_call: int = 2) -> None:
        super().__init__(dim=dim)
        self.calls = 0
        self.fail_on_call = fail_on_call

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated embed crash")
        return super().embed_documents(texts)


def _register_paged_txt_parser(n_pages: int, text_prefix: str) -> None:
    """Register a fake `.txt` parser that produces `n_pages` distinct,
    page-located sections (each long enough not to be folded as a "tiny
    slide"), so a single source reliably chunks into exactly `n_pages`
    chunks -- enough to span more than one embed batch.
    """
    from notecast.ingest.parsers import register
    from notecast.models import Location, ParsedDocument, Section, SourceType

    class PagedParser:
        suffixes = (".txt",)

        def parse(self, path: Path, source_path: str):  # noqa: ANN201
            sections = [
                Section(
                    text=(
                        f"{text_prefix} page {i}: this page has enough words to stand on "
                        "its own as a full chunk for the retrieval index without being "
                        "folded into a neighbouring page."
                    ),
                    location=Location(page=i),
                )
                for i in range(1, n_pages + 1)
            ]
            return ParsedDocument(
                source_path=source_path,
                source_type=SourceType.TXT,
                title=path.stem,
                sections=sections,
            )

    register(PagedParser())


def test_embed_crash_mid_source_keeps_old_rows_and_is_retried(
    tmp_notebooks_root: Path,
) -> None:
    from notecast.ingest import manifest as manifest_mod
    from notecast.ingest.parsers import _PARSERS
    from notecast.ingest.pipeline import load_builtin_parsers

    load_builtin_parsers()
    original_parsers = dict(_PARSERS)
    try:
        nb = _make_notebook(tmp_notebooks_root)
        _register_paged_txt_parser(70, text_prefix="version one")
        (nb.sources_dir / "big.txt").write_text("placeholder v1")

        good_embedder = CountingEmbedder(dim=32)
        first = index_notebook(nb, embedder=good_embedder)
        assert first.total_chunks == 70
        assert first.ingest.failed == {}

        # Change the content (still 70 chunks) and crash the embedder on the
        # second embed batch (chunks 65-70), simulating a failure partway
        # through re-embedding this source.
        _register_paged_txt_parser(70, text_prefix="version two")
        (nb.sources_dir / "big.txt").write_text("placeholder v2")

        crashing_embedder = CrashingEmbedder(dim=32, fail_on_call=2)
        result = index_notebook(nb, embedder=crashing_embedder)

        assert "big.txt" in result.ingest.failed
        assert crashing_embedder.calls == 2

        # The old rows must still be present and untouched (not deleted).
        store = open_store(nb, embedder=crashing_embedder)
        remaining = store.all_chunks()
        assert len(remaining) == 70
        assert all("version one" in c.text for c in remaining)

        m = manifest_mod.load_manifest(nb)
        assert "big.txt" not in m.entries

        # A later normal run with a working embedder fully reprocesses it.
        good_embedder2 = CountingEmbedder(dim=32)
        result2 = index_notebook(nb, embedder=good_embedder2)
        assert result2.ingest.failed == {}
        assert result2.ingest.added == ["big.txt"]

        store2 = open_store(nb, embedder=good_embedder2)
        final = store2.all_chunks()
        assert len(final) == 70
        assert all("version two" in c.text for c in final)
    finally:
        _PARSERS.clear()
        _PARSERS.update(original_parsers)


def test_self_heal_repairs_manually_deleted_rows(tmp_notebooks_root: Path) -> None:
    nb = _make_notebook(tmp_notebooks_root)
    _write_source(nb, "a.txt", "Content about smoothing techniques in NLP.\n" * 5)
    _write_source(nb, "b.txt", "Content about pagerank in graphs.\n" * 5)

    embedder = CountingEmbedder()
    first = index_notebook(nb, embedder=embedder)
    assert first.total_chunks > 0

    store = open_store(nb, embedder=embedder)
    store.delete_source("a.txt")
    store.rebuild_text_index()
    assert "a.txt" not in store.sources()

    result = index_notebook(nb, embedder=embedder)
    assert result.repaired == ["a.txt"]

    store2 = open_store(nb, embedder=embedder)
    assert store2.count() == first.total_chunks
    assert "a.txt" in store2.sources()


def test_search_finds_the_right_chunk(tmp_notebooks_root: Path) -> None:
    nb = _make_notebook(tmp_notebooks_root)
    _write_source(
        nb,
        "week-01/smoothing.txt",
        "Add-one smoothing assigns nonzero probability to unseen n-grams in a language model.\n"
        * 5,
    )
    _write_source(
        nb,
        "week-02/pagerank.txt",
        "PageRank models a random walk over the web graph to rank pages.\n" * 5,
    )

    embedder = CountingEmbedder()
    index_notebook(nb, embedder=embedder)

    retriever = get_retriever(nb, embedder=embedder)
    hits = retriever.search("add-one smoothing unseen n-grams", k=3)

    assert hits
    assert hits[0].chunk.source_path == "week-01/smoothing.txt"
