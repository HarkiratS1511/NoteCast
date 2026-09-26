"""Tests for notecast.index.retriever.Retriever."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import pytest

from notecast.index.retriever import Retriever
from notecast.index.store import LanceChunkStore
from notecast.models import Chunk, SearchHit, SourceType

DIM = 32


def _hash_embed(text: str) -> list[float]:
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


class ReversingReranker:
    """A fake reranker that just reverses the incoming hit order, truncated
    to top_k -- enough to prove the retriever actually calls it.
    """

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        return list(reversed(hits))[:top_k]


CORPUS = [
    (
        "smoothing.pdf",
        1,
        "Laplace add-one smoothing assigns nonzero probability to unseen n-grams.",
    ),
    ("smoothing.pdf", 1, "Kneser-Ney smoothing is a sophisticated back-off discounting technique."),
    ("pagerank.pdf", 2, "PageRank models a random walk over the web graph to rank pages."),
    ("pagerank.pdf", 2, "The PageRank vector is the stationary distribution of a Markov chain."),
    ("perplexity.pdf", 3, "Perplexity measures how well a language model predicts a test set."),
    (
        "perplexity.pdf",
        3,
        "Lower perplexity means the model is less surprised by the held-out text.",
    ),
    (
        "inverted-index.pdf",
        4,
        "An inverted index maps each term to the list of documents containing it.",
    ),
    (
        "inverted-index.pdf",
        4,
        "Postings lists in an inverted index are often compressed with gap encoding.",
    ),
    (
        "tokenization.pdf",
        5,
        "Tokenization splits raw text into words or subword units before modelling.",
    ),
    (
        "stopwords.pdf",
        5,
        "Stop word removal drops frequent low-information words like 'the' and 'a'.",
    ),
]


def make_chunks(course: str = "comp4650") -> list[Chunk]:
    chunks = []
    for ordinal, (source_path, week, text) in enumerate(CORPUS):
        chunks.append(
            Chunk(
                chunk_id=Chunk.make_id(course, source_path, ordinal, text),
                course=course,
                source_path=source_path,
                source_type=SourceType.PDF,
                ordinal=ordinal,
                text=text,
                week=week,
            )
        )
    return chunks


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store(tmp_path: Path, embedder: FakeEmbedder) -> LanceChunkStore:
    s = LanceChunkStore(tmp_path / "idx", dim=DIM, model_name="fake-hash")
    chunks = make_chunks()
    vectors = embedder.embed_documents([c.embed_text for c in chunks])
    s.upsert(chunks, vectors)
    s.rebuild_text_index()
    return s


def test_search_returns_relevant_hits(store, embedder):
    retriever = Retriever(store, embedder)
    hits = retriever.search("How does PageRank use a random walk?", k=3)
    assert hits
    assert "pagerank" in hits[0].chunk.text.lower()


def test_blank_query_returns_empty(store, embedder):
    retriever = Retriever(store, embedder)
    assert retriever.search("") == []
    assert retriever.search("   ") == []


@pytest.mark.parametrize("k", [0, -1, -5])
def test_search_with_non_positive_k_returns_empty(store, embedder, k):
    retriever = Retriever(store, embedder)
    assert retriever.search("PageRank random walk", k=k) == []


def test_reranker_is_applied(store, embedder):
    # candidates == k so both retrievers pull the same underlying pool,
    # making the reranker's effect (a plain reversal) directly comparable.
    retriever = Retriever(store, embedder, reranker=ReversingReranker(), candidates=5)

    without_rerank = Retriever(store, embedder, candidates=5).search("PageRank random walk", k=5)
    with_rerank = retriever.search("PageRank random walk", k=5)

    assert with_rerank == list(reversed(without_rerank))


def test_reranker_widens_candidate_pool(store, embedder):
    retriever = Retriever(store, embedder, reranker=ReversingReranker(), candidates=10)
    hits = retriever.search("PageRank random walk", k=2)
    assert len(hits) == 2


def test_keyword_mode_skips_embedding(store):
    class ExplodingEmbedder:
        dim = DIM

        def embed_documents(self, texts):
            raise AssertionError("should not embed documents")

        def embed_query(self, text):
            raise AssertionError("keyword mode should not embed the query")

    retriever = Retriever(store, ExplodingEmbedder())
    hits = retriever.search("Kneser-Ney", k=3, mode="keyword")
    assert hits
    assert "kneser-ney" in hits[0].chunk.text.lower()


def test_dedupe_keeps_best_scoring_duplicate():
    chunk = make_chunks()[0]
    worse = SearchHit(chunk=chunk, score=0.1)
    better = SearchHit(chunk=chunk, score=0.9)
    deduped = Retriever._dedupe([worse, better])
    assert deduped == [better]


def test_dedupe_preserves_distinct_hits():
    chunks = make_chunks()[:3]
    hits = [SearchHit(chunk=c, score=1.0 - i * 0.1) for i, c in enumerate(chunks)]
    deduped = Retriever._dedupe(hits)
    assert deduped == hits
