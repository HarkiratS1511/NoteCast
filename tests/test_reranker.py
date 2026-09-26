"""Tests for notecast.index.reranker: ordering/top_k behaviour and the
disabled-by-default get_reranker() factory. The cross-encoder is fully
mocked -- no network, no real model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from notecast.index.reranker import (
    FastEmbedReranker,
    RerankerUnavailableError,
    get_reranker,
)
from notecast.models import Chunk, SearchHit, SourceType


def _hit(text: str, ordinal: int, score: float = 0.0) -> SearchHit:
    chunk = Chunk(
        chunk_id=f"c{ordinal}",
        course="comp4650",
        source_path="week-01/slides.pdf",
        source_type=SourceType.PDF,
        ordinal=ordinal,
        text=text,
    )
    return SearchHit(chunk=chunk, score=score)


class _FakeTextCrossEncoder:
    calls: list[dict] = []
    fail_local = False
    fail_online = False
    # Map document text -> score returned by rerank().
    scores_by_text: dict[str, float] = {}

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

    def rerank(self, query, documents, batch_size=64, **kwargs):
        return [self.scores_by_text.get(doc, 0.0) for doc in documents]


@pytest.fixture
def fake_cross_encoder(monkeypatch):
    _FakeTextCrossEncoder.calls = []
    _FakeTextCrossEncoder.fail_local = False
    _FakeTextCrossEncoder.fail_online = False
    _FakeTextCrossEncoder.scores_by_text = {}

    import fastembed.rerank.cross_encoder as ce_mod

    monkeypatch.setattr(ce_mod, "TextCrossEncoder", _FakeTextCrossEncoder)
    return _FakeTextCrossEncoder


class TestFastEmbedReranker:
    def test_construction_is_lazy(self, fake_cross_encoder, tmp_path: Path) -> None:
        FastEmbedReranker("Xenova/ms-marco-MiniLM-L-6-v2", tmp_path)
        assert fake_cross_encoder.calls == []

    def test_reorders_by_reranker_score(self, fake_cross_encoder, tmp_path: Path) -> None:
        hits = [_hit("low relevance", 0, score=0.9), _hit("high relevance", 1, score=0.1)]
        fake_cross_encoder.scores_by_text = {
            hits[0].chunk.embed_text: 0.2,
            hits[1].chunk.embed_text: 0.8,
        }
        reranker = FastEmbedReranker("Xenova/ms-marco-MiniLM-L-6-v2", tmp_path)
        result = reranker.rerank("query", hits, top_k=2)
        assert [h.chunk.ordinal for h in result] == [1, 0]
        assert result[0].score == pytest.approx(0.8)
        assert result[1].score == pytest.approx(0.2)

    def test_respects_top_k(self, fake_cross_encoder, tmp_path: Path) -> None:
        hits = [_hit(f"doc {i}", i) for i in range(5)]
        fake_cross_encoder.scores_by_text = {
            h.chunk.embed_text: float(i) for i, h in enumerate(hits)
        }
        reranker = FastEmbedReranker("Xenova/ms-marco-MiniLM-L-6-v2", tmp_path)
        result = reranker.rerank("query", hits, top_k=2)
        assert len(result) == 2
        assert [h.chunk.ordinal for h in result] == [4, 3]

    def test_empty_hits(self, fake_cross_encoder, tmp_path: Path) -> None:
        reranker = FastEmbedReranker("Xenova/ms-marco-MiniLM-L-6-v2", tmp_path)
        assert reranker.rerank("query", [], top_k=5) == []
        assert fake_cross_encoder.calls == []

    def test_falls_back_to_online_when_local_fails(
        self, fake_cross_encoder, tmp_path: Path
    ) -> None:
        fake_cross_encoder.fail_local = True
        hits = [_hit("doc", 0)]
        reranker = FastEmbedReranker("Xenova/ms-marco-MiniLM-L-6-v2", tmp_path)
        reranker.rerank("query", hits, top_k=1)
        assert [c["local_files_only"] for c in fake_cross_encoder.calls] == [True, False]

    def test_raises_friendly_error_when_unavailable(
        self, fake_cross_encoder, tmp_path: Path
    ) -> None:
        fake_cross_encoder.fail_local = True
        fake_cross_encoder.fail_online = True
        reranker = FastEmbedReranker("Xenova/ms-marco-MiniLM-L-6-v2", tmp_path)
        with pytest.raises(RerankerUnavailableError) as exc_info:
            reranker.rerank("query", [_hit("doc", 0)], top_k=1)
        message = str(exc_info.value)
        assert "Xenova/ms-marco-MiniLM-L-6-v2" in message
        assert str(tmp_path) in message


class TestGetReranker:
    def test_disabled_by_default(self) -> None:
        from notecast.config import Settings

        settings = Settings(rerank_enabled=False)
        assert get_reranker(settings) is None

    def test_enabled_returns_reranker(self, tmp_path: Path) -> None:
        from notecast.config import Settings

        settings = Settings(rerank_enabled=True, model_cache_dir=tmp_path)
        reranker = get_reranker(settings)
        assert isinstance(reranker, FastEmbedReranker)
        assert reranker.model_name == settings.reranker_model
