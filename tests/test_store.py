"""Tests for notecast.index.store.LanceChunkStore."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import pytest

from notecast.index.store import IndexMismatchError, LanceChunkStore
from notecast.models import Chunk, Location, SearchFilters, SourceType

DIM = 32


def _hash_embed(text: str) -> list[float]:
    """A tiny deterministic bag-of-words embedder: hash each token into one
    of DIM buckets and count, then L2-normalise. Good enough to make
    "obviously related" texts land close together in cosine space.
    """
    vec = [0.0] * DIM
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    for tok in tokens:
        idx = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16) % DIM
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class FakeEmbedder:
    dim = DIM

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_hash_embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return _hash_embed(text)


CORPUS = [
    (
        "smoothing.pdf",
        1,
        1,
        "Laplace add-one smoothing assigns nonzero probability to unseen n-grams.",
    ),
    (
        "smoothing.pdf",
        1,
        2,
        "Kneser-Ney smoothing is a sophisticated back-off discounting technique.",
    ),
    ("pagerank.pdf", 2, 1, "PageRank models a random walk over the web graph to rank pages."),
    ("pagerank.pdf", 2, 2, "The PageRank vector is the stationary distribution of a Markov chain."),
    ("perplexity.pdf", 3, 1, "Perplexity measures how well a language model predicts a test set."),
    (
        "perplexity.pdf",
        3,
        2,
        "Lower perplexity means the model is less surprised by the held-out text.",
    ),
    (
        "inverted-index.pdf",
        4,
        1,
        "An inverted index maps each term to the list of documents containing it.",
    ),
    (
        "inverted-index.pdf",
        4,
        2,
        "Postings lists in an inverted index are often compressed with gap encoding.",
    ),
    (
        "tokenization.pdf",
        5,
        1,
        "Tokenization splits raw text into words or subword units before modelling.",
    ),
    (
        "stopwords.pdf",
        5,
        2,
        "Stop word removal drops frequent low-information words like 'the' and 'a'.",
    ),
]


def make_chunks(course: str = "comp4650") -> list[Chunk]:
    chunks = []
    for ordinal, (source_path, week, page, text) in enumerate(CORPUS):
        chunk_id = Chunk.make_id(course, source_path, ordinal, text)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                course=course,
                source_path=source_path,
                source_type=SourceType.PDF,
                ordinal=ordinal,
                text=text,
                header=f"{course} week {week}",
                location=Location(page=page, approx_minute=1.5 * ordinal),
                week=week,
            )
        )
    return chunks


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def populated_store(tmp_path: Path, embedder: FakeEmbedder) -> tuple[LanceChunkStore, list[Chunk]]:
    store = LanceChunkStore(tmp_path / "idx", dim=DIM, model_name="fake-hash")
    chunks = make_chunks()
    vectors = embedder.embed_documents([c.embed_text for c in chunks])
    store.upsert(chunks, vectors)
    store.rebuild_text_index()
    return store, chunks


def test_create_and_reopen(tmp_path: Path):
    idx_dir = tmp_path / "idx"
    store1 = LanceChunkStore(idx_dir, dim=DIM, model_name="fake-hash")
    assert store1.count() == 0
    store2 = LanceChunkStore(idx_dir, dim=DIM, model_name="fake-hash")
    assert store2.count() == 0
    assert (idx_dir / "index_meta.json").exists()


def test_meta_mismatch_raises(tmp_path: Path):
    idx_dir = tmp_path / "idx"
    LanceChunkStore(idx_dir, dim=DIM, model_name="fake-hash")
    with pytest.raises(IndexMismatchError):
        LanceChunkStore(idx_dir, dim=DIM, model_name="other-model")
    with pytest.raises(IndexMismatchError):
        LanceChunkStore(idx_dir, dim=16, model_name="fake-hash")


def test_upsert_then_update_same_chunk_id_no_duplicates(tmp_path: Path, embedder: FakeEmbedder):
    store = LanceChunkStore(tmp_path / "idx", dim=DIM, model_name="fake-hash")
    chunk = make_chunks()[0]
    v1 = embedder.embed_documents([chunk.embed_text])
    store.upsert([chunk], v1)
    assert store.count() == 1

    updated = chunk.model_copy(
        update={"text": "Completely different updated text about smoothing."}
    )
    v2 = embedder.embed_documents([updated.embed_text])
    store.upsert([updated], v2)

    assert store.count() == 1
    [only] = store.all_chunks()
    assert only.text == updated.text


def test_upsert_validates_lengths_and_dims(tmp_path: Path):
    store = LanceChunkStore(tmp_path / "idx", dim=DIM, model_name="fake-hash")
    chunks = make_chunks()[:2]
    with pytest.raises(ValueError):
        store.upsert(chunks, [[0.0] * DIM])
    with pytest.raises(ValueError):
        store.upsert(chunks, [[0.0] * DIM, [0.0] * (DIM - 1)])


def test_upsert_empty_is_noop(tmp_path: Path):
    store = LanceChunkStore(tmp_path / "idx", dim=DIM, model_name="fake-hash")
    store.upsert([], [])
    assert store.count() == 0


def test_delete_source_including_apostrophe_path(tmp_path: Path, embedder: FakeEmbedder):
    store = LanceChunkStore(tmp_path / "idx", dim=DIM, model_name="fake-hash")
    course = "comp4650"
    tricky_path = "week-3/o'brien's notes.pdf"
    chunk = Chunk(
        chunk_id=Chunk.make_id(course, tricky_path, 0, "apostrophe text"),
        course=course,
        source_path=tricky_path,
        source_type=SourceType.PDF,
        ordinal=0,
        text="apostrophe text",
    )
    other = make_chunks()[0]
    vectors = embedder.embed_documents([chunk.embed_text, other.embed_text])
    store.upsert([chunk, other], vectors)
    assert store.count() == 2

    deleted = store.delete_source(tricky_path)
    assert deleted == 1
    assert store.count() == 1
    assert store.sources() == {other.source_path}


def test_delete_all(populated_store):
    store, _chunks = populated_store
    store.delete_all()
    assert store.count() == 0
    assert store.all_chunks() == []


@pytest.mark.parametrize("k", [0, -1, -5])
@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
def test_search_with_non_positive_k_returns_empty(populated_store, embedder, mode, k):
    store, _chunks = populated_store
    qvec = embedder.embed_query("smoothing")
    hits = store.search("smoothing", qvec, k=k, mode=mode)
    assert hits == []


def test_concurrent_upserts_and_searches_do_not_raise(tmp_path: Path, embedder: FakeEmbedder):
    import threading
    import time as _time

    store = LanceChunkStore(tmp_path / "idx", dim=DIM, model_name="fake-hash")
    chunks = make_chunks()
    vectors = embedder.embed_documents([c.embed_text for c in chunks])
    # Seed a bit of data so searches have something to look at from the start.
    store.upsert(chunks[:2], vectors[:2])
    store.rebuild_text_index()

    errors: list[BaseException] = []
    stop_at = _time.monotonic() + 1.0

    def upsert_worker() -> None:
        i = 0
        try:
            while _time.monotonic() < stop_at:
                idx = i % len(chunks)
                store.upsert([chunks[idx]], [vectors[idx]])
                i += 1
        except BaseException as exc:  # noqa: BLE001 -- capture for the main thread to assert on
            errors.append(exc)

    def vector_search_worker() -> None:
        try:
            while _time.monotonic() < stop_at:
                store.search("smoothing", embedder.embed_query("smoothing"), k=3, mode="vector")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def keyword_search_worker() -> None:
        try:
            while _time.monotonic() < stop_at:
                store.search("PageRank", [], k=3, mode="keyword")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=upsert_worker),
        threading.Thread(target=vector_search_worker),
        threading.Thread(target=keyword_search_worker),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert errors == []


def test_vector_search_finds_relevant_chunk(populated_store, embedder):
    store, _chunks = populated_store
    qvec = embedder.embed_query("What smoothing technique handles unseen n-grams?")
    hits = store.search(
        "What smoothing technique handles unseen n-grams?", qvec, k=3, mode="vector"
    )
    assert hits
    assert "smoothing" in hits[0].chunk.text.lower()


def test_keyword_search_finds_relevant_chunk(populated_store, embedder):
    store, _chunks = populated_store
    hits = store.search("PageRank random walk", [], k=3, mode="keyword")
    assert hits
    assert "pagerank" in hits[0].chunk.text.lower()


def test_keyword_search_catches_rare_exact_term(populated_store, embedder):
    store, _chunks = populated_store
    # The fake embedder is bag-of-words hashed, so it has no special notion
    # of a rare compound term -- but FTS should match it exactly.
    hits = store.search("Kneser-Ney", [], k=3, mode="keyword")
    assert hits
    assert "kneser-ney" in hits[0].chunk.text.lower()


def test_hybrid_search_finds_relevant_chunk(populated_store, embedder):
    store, _chunks = populated_store
    query = "How does perplexity relate to a language model's predictions?"
    qvec = embedder.embed_query(query)
    hits = store.search(query, qvec, k=3, mode="hybrid")
    assert hits
    assert "perplexity" in hits[0].chunk.text.lower()


def test_keyword_search_empty_table(tmp_path: Path):
    store = LanceChunkStore(tmp_path / "idx", dim=DIM, model_name="fake-hash")
    assert store.search("anything", [], k=3, mode="keyword") == []


def test_vector_search_empty_table(tmp_path: Path):
    store = LanceChunkStore(tmp_path / "idx", dim=DIM, model_name="fake-hash")
    assert store.search("anything", [0.0] * DIM, k=3, mode="vector") == []


def test_filters_by_week(populated_store, embedder):
    store, _chunks = populated_store
    filters = SearchFilters(weeks=[2])
    hits = store.search("graph", [0.0] * DIM, k=10, filters=filters, mode="keyword")
    assert hits
    assert all(hit.chunk.week == 2 for hit in hits)


def test_filters_by_source_type(populated_store, embedder):
    store, _chunks = populated_store
    filters = SearchFilters(source_types=[SourceType.PDF])
    chunks = store.all_chunks(filters=filters)
    assert len(chunks) == len(CORPUS)


def test_filters_by_source_path(populated_store, embedder):
    store, _chunks = populated_store
    filters = SearchFilters(source_paths=["pagerank.pdf"])
    chunks = store.all_chunks(filters=filters)
    assert chunks
    assert all(c.source_path == "pagerank.pdf" for c in chunks)


def test_empty_filter_list_means_no_constraint(populated_store):
    store, _chunks = populated_store
    filters = SearchFilters(weeks=[], source_types=[], source_paths=[])
    assert len(store.all_chunks(filters=filters)) == len(CORPUS)


def test_chunk_round_trips_exactly(tmp_path: Path, embedder: FakeEmbedder):
    store = LanceChunkStore(tmp_path / "idx", dim=DIM, model_name="fake-hash")
    chunk = Chunk(
        chunk_id=Chunk.make_id("comp4650", "transcript.vtt", 0, "hello"),
        course="comp4650",
        source_path="transcript.vtt",
        source_type=SourceType.VTT,
        ordinal=0,
        text="hello",
        header="Week 6 lecture",
        location=Location(
            t_start=12.5,
            t_end=20.0,
            approx_minute=3.25,
            speaker="Prof X",
            heading_path=["Intro", "Motivation"],
            char_start=10,
            char_end=15,
        ),
        week=6,
        topic="intro",
    )
    store.upsert([chunk], embedder.embed_documents([chunk.embed_text]))
    [round_tripped] = store.all_chunks()
    assert round_tripped == chunk


def test_all_chunks_count_sources(populated_store):
    store, chunks = populated_store
    assert store.count() == len(chunks)
    assert store.sources() == {c.source_path for c in chunks}
    all_chunks = store.all_chunks()
    assert [c.chunk_id for c in all_chunks] == sorted(
        [c.chunk_id for c in chunks],
        key=lambda cid: next((c.source_path, c.ordinal) for c in chunks if c.chunk_id == cid),
    )
    # sorted by (source_path, ordinal)
    pairs = [(c.source_path, c.ordinal) for c in all_chunks]
    assert pairs == sorted(pairs)
