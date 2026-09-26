"""Tests for notecast.index.embedder: the hashing embedder (real, offline)
and the fastembed embedder (fully mocked -- no network, no real model).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from notecast.index import embedder as embedder_mod
from notecast.index.embedder import (
    EmbedderUnavailableError,
    FastEmbedEmbedder,
    HashingEmbedder,
    get_embedder,
)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class TestHashingEmbedder:
    def test_deterministic_across_calls(self) -> None:
        e1 = HashingEmbedder()
        e2 = HashingEmbedder()
        assert e1.embed_query("PageRank is a link analysis algorithm") == e2.embed_query(
            "PageRank is a link analysis algorithm"
        )

    def test_dim(self) -> None:
        e = HashingEmbedder(dim=128)
        vec = e.embed_query("hello world")
        assert len(vec) == 128
        assert e.dim == 128
        assert e.model_name == "hashing-128"

    def test_default_dim_is_384(self) -> None:
        e = HashingEmbedder()
        assert e.dim == 384
        assert len(e.embed_query("x")) == 384

    def test_normalised(self) -> None:
        e = HashingEmbedder()
        vec = e.embed_query("Laplace smoothing is used in language models")
        norm = math.sqrt(sum(v * v for v in vec))
        assert norm == pytest.approx(1.0, abs=1e-9)

    def test_empty_text_is_zero_vector(self) -> None:
        e = HashingEmbedder()
        vec = e.embed_query("")
        assert vec == [0.0] * e.dim

    def test_whitespace_only_is_zero_vector(self) -> None:
        e = HashingEmbedder()
        assert e.embed_query("   \t\n  ") == [0.0] * e.dim

    def test_similar_texts_score_higher_than_unrelated(self) -> None:
        e = HashingEmbedder()
        query = e.embed_query("add-one Laplace smoothing for language models")
        related = e.embed_query("Laplace add-one smoothing technique in NLP models")
        unrelated = e.embed_query("PageRank computes the stationary distribution of a graph")

        sim_related = _cosine(query, related)
        sim_unrelated = _cosine(query, unrelated)
        assert sim_related > sim_unrelated

    def test_embed_documents_batches(self) -> None:
        e = HashingEmbedder()
        texts = ["first document", "second document", ""]
        vecs = e.embed_documents(texts)
        assert len(vecs) == 3
        assert vecs[2] == [0.0] * e.dim
        assert vecs[0] == e.embed_query("first document")

    def test_case_insensitive(self) -> None:
        e = HashingEmbedder()
        assert e.embed_query("Graph Theory") == e.embed_query("graph theory")


class _FakeArray(list):
    """Stands in for a numpy array: supports len() and .tolist()."""

    def tolist(self):
        return list(self)


class _FakeTextEmbedding:
    """A fake fastembed.TextEmbedding used to test lazy loading, the
    local-first-then-online fallback, batching and list[float] output
    without ever loading a real model.
    """

    calls: list[dict] = []
    fail_local = False
    fail_online = False

    def __init__(self, model_name, cache_dir=None, threads=None, **kwargs):
        self.__class__.calls.append(
            {
                "model_name": model_name,
                "cache_dir": cache_dir,
                "threads": threads,
                "local_files_only": kwargs.get("local_files_only"),
            }
        )
        local_only = kwargs.get("local_files_only")
        if local_only and self.fail_local:
            raise RuntimeError("not found locally")
        if not local_only and self.fail_online:
            raise RuntimeError("no network")
        self.model_name = model_name

    def embed(self, texts, batch_size=256, **kwargs):
        return [_FakeArray([float(len(t)), 0.0, 1.0]) for t in texts]

    def query_embed(self, query, **kwargs):
        if isinstance(query, str):
            query = [query]
        return [_FakeArray([float(len(q)), 1.0, 0.0]) for q in query]

    @staticmethod
    def list_supported_models():
        return [{"model": "BAAI/bge-small-en-v1.5", "dim": 384}]


@pytest.fixture(autouse=True)
def _reset_embedder_cache():
    embedder_mod._EMBEDDER_CACHE.clear()
    yield
    embedder_mod._EMBEDDER_CACHE.clear()


@pytest.fixture
def fake_text_embedding(monkeypatch):
    _FakeTextEmbedding.calls = []
    _FakeTextEmbedding.fail_local = False
    _FakeTextEmbedding.fail_online = False

    import fastembed

    monkeypatch.setattr(fastembed, "TextEmbedding", _FakeTextEmbedding)
    return _FakeTextEmbedding


class TestFastEmbedEmbedder:
    def test_construction_is_lazy(self, fake_text_embedding, tmp_path: Path) -> None:
        FastEmbedEmbedder("BAAI/bge-small-en-v1.5", tmp_path)
        assert fake_text_embedding.calls == []

    def test_dim_known_without_loading(self, fake_text_embedding, tmp_path: Path) -> None:
        e = FastEmbedEmbedder("BAAI/bge-small-en-v1.5", tmp_path)
        assert e.dim == 384
        assert fake_text_embedding.calls == []

    def test_embed_documents_uses_local_first(self, fake_text_embedding, tmp_path: Path) -> None:
        e = FastEmbedEmbedder("BAAI/bge-small-en-v1.5", tmp_path)
        vecs = e.embed_documents(["hi", "hello there"])
        assert vecs == [[2.0, 0.0, 1.0], [11.0, 0.0, 1.0]]
        assert all(isinstance(v, list) for v in vecs)
        assert fake_text_embedding.calls == [
            {
                "model_name": "BAAI/bge-small-en-v1.5",
                "cache_dir": str(tmp_path),
                "threads": None,
                "local_files_only": True,
            }
        ]

    def test_embed_query_returns_list_of_float(self, fake_text_embedding, tmp_path: Path) -> None:
        e = FastEmbedEmbedder("BAAI/bge-small-en-v1.5", tmp_path)
        vec = e.embed_query("hello")
        assert vec == [5.0, 1.0, 0.0]
        assert isinstance(vec, list)

    def test_model_loaded_once_and_reused(self, fake_text_embedding, tmp_path: Path) -> None:
        e = FastEmbedEmbedder("BAAI/bge-small-en-v1.5", tmp_path)
        e.embed_query("a")
        e.embed_query("b")
        assert len(fake_text_embedding.calls) == 1

    def test_falls_back_to_online_when_local_fails(
        self, fake_text_embedding, tmp_path: Path
    ) -> None:
        fake_text_embedding.fail_local = True
        e = FastEmbedEmbedder("BAAI/bge-small-en-v1.5", tmp_path)
        vec = e.embed_query("hello")
        assert vec == [5.0, 1.0, 0.0]
        assert [c["local_files_only"] for c in fake_text_embedding.calls] == [True, False]

    def test_raises_friendly_error_when_unavailable(
        self, fake_text_embedding, tmp_path: Path
    ) -> None:
        fake_text_embedding.fail_local = True
        fake_text_embedding.fail_online = True
        e = FastEmbedEmbedder("BAAI/bge-small-en-v1.5", tmp_path)
        with pytest.raises(EmbedderUnavailableError) as exc_info:
            e.embed_query("hello")
        message = str(exc_info.value)
        assert "BAAI/bge-small-en-v1.5" in message
        assert str(tmp_path) in message
        assert "internet" in message.lower()

    def test_creates_cache_dir(self, fake_text_embedding, tmp_path: Path) -> None:
        cache_dir = tmp_path / "nested" / "models"
        e = FastEmbedEmbedder("BAAI/bge-small-en-v1.5", cache_dir)
        e.embed_query("hello")
        assert cache_dir.exists()

    def test_batch_size_is_passed(self, monkeypatch, tmp_path: Path) -> None:
        captured = {}

        class RecordingFake(_FakeTextEmbedding):
            def embed(self, texts, batch_size=256, **kwargs):
                captured["batch_size"] = batch_size
                return super().embed(texts, batch_size=batch_size, **kwargs)

        import fastembed

        monkeypatch.setattr(fastembed, "TextEmbedding", RecordingFake)
        e = FastEmbedEmbedder("BAAI/bge-small-en-v1.5", tmp_path)
        e.embed_documents(["a", "b"])
        assert captured["batch_size"] == 64


class TestGetEmbedder:
    def test_hashing_backend(self) -> None:
        from notecast.config import Settings

        settings = Settings(embedding_backend="hashing")
        e = get_embedder(settings)
        assert isinstance(e, HashingEmbedder)

    def test_fastembed_backend(self, tmp_path: Path) -> None:
        from notecast.config import Settings

        settings = Settings(
            embedding_backend="fastembed",
            embedding_model="BAAI/bge-small-en-v1.5",
            model_cache_dir=tmp_path,
        )
        e = get_embedder(settings)
        assert isinstance(e, FastEmbedEmbedder)
        assert e.model_name == "BAAI/bge-small-en-v1.5"

    def test_caches_instance_per_backend_model_cache_dir(self, tmp_path: Path) -> None:
        from notecast.config import Settings

        settings = Settings(embedding_backend="hashing")
        e1 = get_embedder(settings)
        e2 = get_embedder(settings)
        assert e1 is e2

    def test_different_settings_give_different_instances(self, tmp_path: Path) -> None:
        from notecast.config import Settings

        s1 = Settings(embedding_backend="fastembed", model_cache_dir=tmp_path / "a")
        s2 = Settings(embedding_backend="fastembed", model_cache_dir=tmp_path / "b")
        e1 = get_embedder(s1)
        e2 = get_embedder(s2)
        assert e1 is not e2

    def test_uses_get_settings_when_none_passed(self, monkeypatch) -> None:
        from notecast.config import Settings, get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("NOTECAST_EMBEDDING_BACKEND", "hashing")
        get_settings.cache_clear()
        try:
            e = get_embedder()
            assert isinstance(e, HashingEmbedder)
        finally:
            get_settings.cache_clear()
            monkeypatch.delenv("NOTECAST_EMBEDDING_BACKEND", raising=False)
            get_settings.cache_clear()
        assert Settings  # keep import used
