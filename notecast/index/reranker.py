"""Optional cross-encoder reranking of search hits."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from notecast.models import SearchHit

if TYPE_CHECKING:
    from notecast.config import Settings


class RerankerUnavailableError(RuntimeError):
    """Raised when a fastembed reranker model can neither be loaded from the
    local cache nor downloaded.
    """


@runtime_checkable
class Reranker(Protocol):
    """Re-scores and re-orders a set of search hits for a query."""

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        """Return the `top_k` hits, re-scored (score = reranker score) and
        sorted from most to least relevant.
        """
        ...


class FastEmbedReranker:
    """A local cross-encoder reranker, loaded lazily via `fastembed`.

    Construction is instant: the underlying `TextCrossEncoder` model is only
    loaded the first time `rerank` is used.
    """

    def __init__(self, model_name: str, cache_dir: Path, *, threads: int | None = None) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.threads = threads
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model

        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_dir_str = str(self.cache_dir)
        try:
            model = TextCrossEncoder(
                self.model_name,
                cache_dir=cache_dir_str,
                threads=self.threads,
                local_files_only=True,
            )
        except Exception:
            try:
                model = TextCrossEncoder(
                    self.model_name,
                    cache_dir=cache_dir_str,
                    threads=self.threads,
                    local_files_only=False,
                )
            except Exception as exc:
                raise RerankerUnavailableError(
                    f"Could not load the reranker model {self.model_name!r} from the "
                    f"local cache ({self.cache_dir}) or download it. The first run needs "
                    "internet access to download the model."
                ) from exc
        self._model = model
        return model

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        model = self._load()
        documents = [hit.chunk.embed_text for hit in hits]
        scores = list(model.rerank(query, documents))
        rescored = [
            hit.model_copy(update={"score": float(score)})
            for hit, score in zip(hits, scores, strict=True)
        ]
        rescored.sort(key=lambda hit: hit.score, reverse=True)
        return rescored[:top_k]


def get_reranker(settings: Settings | None = None) -> Reranker | None:
    """Return a Reranker for the configured model, or None if reranking is
    disabled.
    """
    if settings is None:
        from notecast.config import get_settings

        settings = get_settings()

    if not settings.rerank_enabled:
        return None

    return FastEmbedReranker(settings.reranker_model, settings.model_cache_dir)
