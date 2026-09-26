"""Contracts that later phases implement: how to parse a file, how to embed
text, and how to store/search chunks. Defining these here lets Phase 1+
builders write real implementations without needing to agree on anything
beyond these shapes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from notecast.models import Chunk, ParsedDocument, SearchFilters, SearchHit


@runtime_checkable
class Parser(Protocol):
    """Turns one raw source file into a ParsedDocument of sections."""

    suffixes: tuple[str, ...]

    def parse(self, path: Path, source_path: str) -> ParsedDocument:
        """Parse `path` on disk, tagging the result with `source_path` (the
        file's path relative to the notebook's sources directory).
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors for similarity search."""

    @property
    def dim(self) -> int:
        """The length of the vectors this embedder produces."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of chunk texts (for indexing)."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        ...


@runtime_checkable
class ChunkStore(Protocol):
    """Persists chunks and their vectors, and searches over them."""

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert or replace chunks (matched by chunk_id) along with their
        embedding vectors.
        """
        ...

    def delete_source(self, source_path: str) -> int:
        """Remove all chunks belonging to `source_path`. Returns the number
        of chunks deleted.
        """
        ...

    def search(
        self,
        query: str,
        query_vector: list[float],
        k: int,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]:
        """Return the top-k chunks for a query, optionally narrowed by
        `filters`.
        """
        ...

    def all_chunks(self, filters: SearchFilters | None = None) -> list[Chunk]:
        """Return every stored chunk, optionally narrowed by `filters`."""
        ...
