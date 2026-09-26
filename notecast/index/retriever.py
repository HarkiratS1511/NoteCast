"""The retriever: turns a question into ranked, de-duplicated search hits by
embedding the query, searching the chunk store, and optionally reranking.
"""

from __future__ import annotations

from typing import Protocol

from notecast.index.store import LanceChunkStore, SearchMode
from notecast.models import SearchFilters, SearchHit


class _Reranker(Protocol):
    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]: ...


class Retriever:
    """Retrieves and ranks the best chunks for a query against a
    `LanceChunkStore`.
    """

    def __init__(
        self,
        store: LanceChunkStore,
        embedder,
        reranker: _Reranker | None = None,
        *,
        candidates: int = 30,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.candidates = candidates

    def search(
        self,
        query: str,
        k: int = 10,
        filters: SearchFilters | None = None,
        mode: SearchMode = "hybrid",
    ) -> list[SearchHit]:
        if not query or not query.strip():
            return []
        if k <= 0:
            return []

        fetch_k = max(k, self.candidates) if self.reranker is not None else k

        query_vector: list[float] = []
        if mode != "keyword":
            query_vector = self.embedder.embed_query(query)

        hits = self.store.search(query, query_vector, fetch_k, filters, mode=mode)
        hits = self._dedupe(hits)

        if self.reranker is not None:
            hits = self.reranker.rerank(query, hits, k)
        else:
            hits = hits[:k]
        return hits

    @staticmethod
    def _dedupe(hits: list[SearchHit]) -> list[SearchHit]:
        """Drop near-identical hits (same source_path and identical text),
        keeping the highest-scoring one, while preserving relative order.
        """
        best: dict[tuple[str, str], SearchHit] = {}
        order: list[tuple[str, str]] = []
        for hit in hits:
            key = (hit.chunk.source_path, hit.chunk.text)
            if key not in best:
                best[key] = hit
                order.append(key)
            elif hit.score > best[key].score:
                best[key] = hit
        return [best[key] for key in order]


__all__ = ["Retriever"]
