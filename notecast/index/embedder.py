"""Text embedders: a real local ONNX model (fastembed) behind a lazy loader,
and a tiny deterministic offline hashing embedder used as a fallback and in
tests that must not touch a network or load a real model.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from notecast.config import Settings

_WORD_RE = re.compile(r"\w+", re.UNICODE)


class EmbedderUnavailableError(RuntimeError):
    """Raised when a fastembed embedding model can neither be loaded from
    the local cache nor downloaded.
    """


def _tokens(text: str) -> list[str]:
    """Lowercase words plus char trigrams (with boundary markers) of each
    word, so short/near-duplicate words still share features.
    """
    lowered = text.lower()
    words = _WORD_RE.findall(lowered)
    tokens: list[str] = list(words)
    for word in words:
        padded = f"^{word}$"
        if len(padded) < 3:
            tokens.append(padded)
            continue
        tokens.extend(padded[i : i + 3] for i in range(len(padded) - 2))
    return tokens


class HashingEmbedder:
    """Deterministic, offline, dependency-free embedder: hashes tokens into
    fixed-size buckets with random signs (a signed feature-hashing / random
    projection scheme), then L2-normalises. Not semantically meaningful, but
    stable across processes and fast, so it's used throughout the test suite
    and as an explicit fallback backend.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.model_name = f"hashing-{dim}"

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = _tokens(text)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket_bits = int.from_bytes(digest[:7], "big")
            bucket = bucket_bits % self.dim
            sign = 1.0 if digest[7] & 1 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


class FastEmbedEmbedder:
    """A real local ONNX embedding model, loaded lazily via `fastembed`.

    Construction is instant: the underlying `TextEmbedding` model is only
    loaded the first time `embed_documents`/`embed_query`/`dim` is used, so
    importing or instantiating this class does not touch disk or network.
    """

    def __init__(self, model_name: str, cache_dir: Path, *, threads: int | None = None) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.threads = threads
        self._model = None
        self._dim: int | None = None

    def _known_dim(self) -> int | None:
        from fastembed import TextEmbedding

        for info in TextEmbedding.list_supported_models():
            if info.get("model") == self.model_name:
                dim = info.get("dim")
                if isinstance(dim, int):
                    return dim
        return None

    def _load(self):
        if self._model is not None:
            return self._model

        from fastembed import TextEmbedding

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_dir_str = str(self.cache_dir)
        try:
            model = TextEmbedding(
                self.model_name,
                cache_dir=cache_dir_str,
                threads=self.threads,
                local_files_only=True,
            )
        except Exception:
            try:
                model = TextEmbedding(
                    self.model_name,
                    cache_dir=cache_dir_str,
                    threads=self.threads,
                    local_files_only=False,
                )
            except Exception as exc:
                raise EmbedderUnavailableError(
                    f"Could not load the embedding model {self.model_name!r} from the "
                    f"local cache ({self.cache_dir}) or download it. The first run needs "
                    "internet access to download the model (~70 MB)."
                ) from exc
        self._model = model
        return model

    @property
    def dim(self) -> int:
        if self._dim is not None:
            return self._dim
        known = self._known_dim()
        if known is not None:
            self._dim = known
            return known
        model = self._load()
        vector = next(iter(model.embed(["dimension probe"])))
        self._dim = len(vector)
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        return [list(vec) for vec in model.embed(texts, batch_size=64)]

    def embed_query(self, text: str) -> list[float]:
        model = self._load()
        return next(iter(model.query_embed(text))).tolist()


_EMBEDDER_CACHE: dict[tuple[str, str, str], object] = {}


def get_embedder(settings: Settings | None = None):
    """Return a cached Embedder for the configured backend, model and cache
    dir. Repeated calls with the same settings reuse the same instance (and
    therefore the same lazily-loaded model), instead of reloading it.
    """
    if settings is None:
        from notecast.config import get_settings

        settings = get_settings()

    backend = settings.embedding_backend
    key = (backend, settings.embedding_model, str(settings.model_cache_dir))
    if key in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[key]

    embedder: object
    if backend == "hashing":
        embedder = HashingEmbedder()
    elif backend == "fastembed":
        embedder = FastEmbedEmbedder(settings.embedding_model, settings.model_cache_dir)
    else:  # pragma: no cover - Settings/Literal should prevent this
        raise ValueError(f"Unknown embedding backend: {backend!r}")

    _EMBEDDER_CACHE[key] = embedder
    return embedder
