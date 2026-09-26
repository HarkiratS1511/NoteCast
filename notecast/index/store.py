"""LanceDB-backed chunk store: persists chunks and their vectors for one
notebook, and answers vector, keyword and hybrid search queries.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypeVar

import lancedb
import pyarrow as pa
from lancedb.index import FTS
from lancedb.rerankers import RRFReranker

from notecast.models import Chunk, SearchFilters, SearchHit

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_META_FILENAME = "index_meta.json"
_MAX_COMMIT_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 0.05

SearchMode = Literal["hybrid", "vector", "keyword"]

T = TypeVar("T")


class IndexMismatchError(RuntimeError):
    """Raised when an existing on-disk index was built with a different
    embedding model or vector dimension than the one now requested.
    """


def _is_commit_conflict(exc: Exception) -> bool:
    return isinstance(exc, RuntimeError) and "conflict" in str(exc).lower()


def _with_conflict_retry(fn: Callable[[], T]) -> T:
    """Run `fn`, retrying a few times with small backoff if LanceDB reports
    a retryable commit conflict (concurrent writers racing on the same
    table). Re-raises immediately for any other error.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except RuntimeError as exc:
            attempt += 1
            if not _is_commit_conflict(exc) or attempt >= _MAX_COMMIT_RETRIES:
                raise
            logger.warning(
                "LanceDB commit conflict, retrying (%d/%d): %s",
                attempt,
                _MAX_COMMIT_RETRIES,
                exc,
            )
            time.sleep(_RETRY_BACKOFF_SECONDS * attempt)


def _quote_sql_string(value: str) -> str:
    """Escape a string for embedding as a single-quoted SQL literal."""
    return value.replace("'", "''")


def _in_clause(column: str, values: list[str] | list[int]) -> str:
    quoted = [f"'{_quote_sql_string(str(v))}'" if isinstance(v, str) else str(v) for v in values]
    return f"{column} IN ({', '.join(quoted)})"


def _build_where(filters: SearchFilters | None) -> str | None:
    """Build a SQL WHERE clause (ANDed) from search filters.

    `None` or an empty list for any field means "no constraint" on that
    field.
    """
    if filters is None:
        return None
    clauses: list[str] = []
    if filters.weeks:
        clauses.append(_in_clause("week", filters.weeks))
    if filters.source_types:
        clauses.append(_in_clause("source_type", [st.value for st in filters.source_types]))
    if filters.source_paths:
        clauses.append(_in_clause("source_path", filters.source_paths))
    if not clauses:
        return None
    return " AND ".join(clauses)


def _schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("course", pa.string()),
            pa.field("source_path", pa.string()),
            pa.field("source_type", pa.string()),
            pa.field("week", pa.int32(), nullable=True),
            pa.field("ordinal", pa.int32()),
            pa.field("text", pa.string()),
            pa.field("embed_text", pa.string()),
            pa.field("chunk_json", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
    )


def _row_from_chunk(chunk: Chunk, vector: list[float]) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "course": chunk.course,
        "source_path": chunk.source_path,
        "source_type": chunk.source_type.value,
        "week": chunk.week,
        "ordinal": chunk.ordinal,
        "text": chunk.text,
        "embed_text": chunk.embed_text,
        "chunk_json": chunk.model_dump_json(),
        "vector": vector,
    }


def _chunk_from_row(row: dict) -> Chunk:
    return Chunk.model_validate_json(row["chunk_json"])


class LanceChunkStore:
    """A LanceDB-backed store for one notebook's chunks, implementing the
    `ChunkStore` protocol.

    Thread-safe: a single `RLock` serializes writes (upsert, delete,
    text-index rebuilds) and the dirty-check-then-maybe-rebuild that a
    keyword/hybrid search can trigger, so one store instance can safely be
    shared across threads (e.g. a Streamlit app's worker threads).
    """

    def __init__(
        self,
        index_dir: Path,
        *,
        dim: int,
        model_name: str,
        table_name: str = "chunks",
    ) -> None:
        self.index_dir = Path(index_dir)
        self.dim = dim
        self.model_name = model_name
        self.table_name = table_name
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        self._check_meta()

        self._db = lancedb.connect(str(self.index_dir))
        if table_name in set(self._db.list_tables().tables):
            self._table = self._db.open_table(table_name)
        else:
            self._table = self._db.create_table(table_name, schema=_schema(dim))
        # Only force a rebuild if there's data but no FTS index on disk yet
        # (e.g. an index built before FTS existed, or a crash mid-build).
        # A fresh/empty table needs no index, and an index that's already
        # present and current shouldn't be rebuilt on every reopen.
        self._fts_dirty = self._table.count_rows() > 0 and not self._has_fts_index()

    # -- meta -----------------------------------------------------------

    @property
    def _meta_path(self) -> Path:
        return self.index_dir / _META_FILENAME

    def _check_meta(self) -> None:
        if self._meta_path.exists():
            existing = json.loads(self._meta_path.read_text(encoding="utf-8"))
            if existing.get("model_name") != self.model_name or existing.get("dim") != self.dim:
                raise IndexMismatchError(
                    f"index at {self.index_dir} was built with model "
                    f"{existing.get('model_name')!r} (dim={existing.get('dim')}); "
                    f"this run wants {self.model_name!r} (dim={self.dim}). "
                    f"Run `notecast ingest <slug> --reindex` to rebuild it."
                )
        else:
            self._write_meta()

    def _write_meta(self) -> None:
        meta = {
            "model_name": self.model_name,
            "dim": self.dim,
            "schema_version": _SCHEMA_VERSION,
        }
        self._meta_path.write_text(json.dumps(meta), encoding="utf-8")

    def _has_fts_index(self) -> bool:
        try:
            indices = self._table.list_indices()
        except Exception:  # pragma: no cover - defensive, list_indices is cheap/safe
            return False
        return any(idx.index_type == "FTS" and "embed_text" in idx.columns for idx in indices)

    # -- writes -----------------------------------------------------------

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks and vectors must have equal length, got {len(chunks)} and {len(vectors)}"
            )
        for vector in vectors:
            if len(vector) != self.dim:
                raise ValueError(f"expected vectors of dim {self.dim}, got {len(vector)}")
        if not chunks:
            return
        rows = [
            _row_from_chunk(chunk, vector) for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        with self._lock:
            _with_conflict_retry(
                lambda: (
                    self._table.merge_insert("chunk_id")
                    .when_matched_update_all()
                    .when_not_matched_insert_all()
                    .execute(rows)
                )
            )
            self._fts_dirty = True

    def delete_source(self, source_path: str) -> int:
        predicate = f"source_path = '{_quote_sql_string(source_path)}'"
        with self._lock:
            before = self._table.count_rows()
            _with_conflict_retry(lambda: self._table.delete(predicate))
            after = self._table.count_rows()
            self._fts_dirty = True
        return before - after

    def delete_all(self) -> None:
        with self._lock:
            _with_conflict_retry(
                lambda: self._table.delete("chunk_id IS NOT NULL OR chunk_id IS NULL")
            )
            self._fts_dirty = True

    def rebuild_text_index(self) -> None:
        with self._lock:
            if self._table.count_rows() == 0:
                self._fts_dirty = False
                return
            _with_conflict_retry(
                lambda: self._table.create_index("embed_text", config=FTS(), replace=True)
            )
            self._fts_dirty = False

    def _ensure_text_index(self) -> bool:
        """Rebuild the FTS index if it's stale. Returns whether the table
        has any rows (so callers can short-circuit an empty table).
        """
        with self._lock:
            if self._table.count_rows() == 0:
                return False
            if self._fts_dirty:
                self.rebuild_text_index()
            return True

    # -- reads --------------------------------------------------------------

    def count(self) -> int:
        return self._table.count_rows()

    def sources(self) -> set[str]:
        rows = self._table.to_arrow().select(["source_path"]).to_pylist()
        return {row["source_path"] for row in rows}

    def all_chunks(self, filters: SearchFilters | None = None) -> list[Chunk]:
        where = _build_where(filters)
        total = self._table.count_rows()
        if total == 0:
            return []
        if where:
            arrow_tbl = self._table.search().where(where).limit(total).to_arrow()
        else:
            arrow_tbl = self._table.to_arrow()
        rows = arrow_tbl.to_pylist()
        chunks = [_chunk_from_row(row) for row in rows]
        chunks.sort(key=lambda c: (c.source_path, c.ordinal))
        return chunks

    def search(
        self,
        query: str,
        query_vector: list[float],
        k: int,
        filters: SearchFilters | None = None,
        *,
        mode: SearchMode = "hybrid",
    ) -> list[SearchHit]:
        if k <= 0:
            return []
        if self._table.count_rows() == 0:
            return []
        where = _build_where(filters)

        if mode == "vector":
            return self._search_vector(query_vector, k, where)
        if mode == "keyword":
            return self._search_keyword(query, k, where)
        return self._search_hybrid(query, query_vector, k, where)

    def _search_vector(
        self, query_vector: list[float], k: int, where: str | None
    ) -> list[SearchHit]:
        builder = self._table.search(query_vector).metric("cosine")
        if where:
            builder = builder.where(where, prefilter=True)
        rows = _with_conflict_retry(lambda: builder.limit(k).to_list())
        return [SearchHit(chunk=_chunk_from_row(row), score=1.0 - row["_distance"]) for row in rows]

    def _search_keyword(self, query: str, k: int, where: str | None) -> list[SearchHit]:
        if not self._ensure_text_index():
            return []
        try:
            builder = self._table.search(query, query_type="fts", fts_columns="embed_text")
            if where:
                builder = builder.where(where, prefilter=True)
            rows = _with_conflict_retry(lambda: builder.limit(k).to_list())
        except ValueError:
            # No FTS tokens in the query (e.g. only stop words / punctuation).
            return []
        return [SearchHit(chunk=_chunk_from_row(row), score=row["_score"]) for row in rows]

    def _search_hybrid(
        self, query: str, query_vector: list[float], k: int, where: str | None
    ) -> list[SearchHit]:
        if not self._ensure_text_index():
            return self._search_vector(query_vector, k, where)
        try:
            builder = (
                self._table.search(query_type="hybrid", fts_columns="embed_text")
                .text(query)
                .vector(query_vector)
            )
            if where:
                builder = builder.where(where, prefilter=True)
            rows = _with_conflict_retry(
                lambda: builder.limit(k).rerank(reranker=RRFReranker()).to_list()
            )
        except ValueError:
            logger.warning(
                "Hybrid search failed for query %r; falling back to vector search", query
            )
            return self._search_vector(query_vector, k, where)
        return [
            SearchHit(chunk=_chunk_from_row(row), score=row["_relevance_score"]) for row in rows
        ]


__all__ = ["LanceChunkStore", "IndexMismatchError", "SearchMode"]
