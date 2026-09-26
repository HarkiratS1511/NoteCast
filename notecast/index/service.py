"""Wires ingest (parsing/chunking) into the search index: embeds new/changed
chunks, keeps the LanceDB store in sync with the manifest, and hands back a
ready-to-use `Retriever` for chat/search/eval.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from notecast.index.reranker import get_reranker
from notecast.index.retriever import Retriever
from notecast.index.store import IndexMismatchError, LanceChunkStore
from notecast.ingest.pipeline import IngestReport, ingest_notebook, load_chunks, load_source_chunks
from notecast.models import Chunk

if TYPE_CHECKING:
    from notecast.config import Settings
    from notecast.notebook import Notebook

_EMBED_BATCH_SIZE = 64


class IndexNotBuiltError(RuntimeError):
    """Raised when a notebook's search index is missing or empty."""


class IndexReport(BaseModel):
    """A summary of one `index_notebook` run."""

    ingest: IngestReport
    indexed_chunks: int = 0
    removed_sources: list[str] = Field(default_factory=list)
    reindexed: bool = False
    total_chunks: int = 0
    # Sources whose index rows didn't match their manifest (missing, or a
    # different chunk count) and were re-embedded from processed chunks
    # before this run's ingest, e.g. after rows were deleted or lost outside
    # of a normal ingest run.
    repaired: list[str] = Field(default_factory=list)


def open_store(nb: Notebook, *, embedder) -> LanceChunkStore:
    """Open (or create) `nb`'s chunk store, sized/named for `embedder`."""
    return LanceChunkStore(nb.index_dir, dim=embedder.dim, model_name=embedder.model_name)


def _embed_chunks(embedder, chunks: list[Chunk]) -> list[list[float]]:
    vectors: list[list[float]] = []
    texts = [c.embed_text for c in chunks]
    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[start : start + _EMBED_BATCH_SIZE]
        vectors.extend(embedder.embed_documents(batch))
    return vectors


def _repair_missing_sources(nb: Notebook, store: LanceChunkStore, embedder) -> list[str]:
    """Re-embed, from their already-processed `.chunks.jsonl` (no
    re-parsing), any manifested source whose rows are missing from the store
    entirely or whose row count doesn't match its manifest `chunk_count` --
    e.g. rows deleted or lost outside of a normal ingest run, or a run that
    crashed after `store.delete_source` but before `store.upsert`.
    """
    from notecast.ingest import manifest as manifest_mod

    m = manifest_mod.load_manifest(nb)
    if not m.entries:
        return []

    counts: dict[str, int] = {}
    for chunk in store.all_chunks():
        counts[chunk.source_path] = counts.get(chunk.source_path, 0) + 1

    repaired: list[str] = []
    for source_path, entry in m.entries.items():
        if counts.get(source_path, 0) == entry.chunk_count:
            continue
        chunks = load_source_chunks(nb, source_path)
        store.delete_source(source_path)
        if chunks:
            vectors = _embed_chunks(embedder, chunks)
            store.upsert(chunks, vectors)
        repaired.append(source_path)

    if repaired:
        store.rebuild_text_index()
    return repaired


def _rebuild_index(nb: Notebook, embedder, ingest_report: IngestReport) -> IndexReport:
    # Drop any existing index outright rather than opening it: an existing
    # index may have been built with a different embedding model/dim (an
    # `IndexMismatchError` waiting to happen the moment we try to open it),
    # and a plain delete_all() wouldn't touch that model/dim metadata anyway.
    if nb.index_dir.exists():
        shutil.rmtree(nb.index_dir)
    store = LanceChunkStore(nb.index_dir, dim=embedder.dim, model_name=embedder.model_name)
    chunks = load_chunks(nb)
    if chunks:
        vectors = _embed_chunks(embedder, chunks)
        store.upsert(chunks, vectors)
    store.rebuild_text_index()
    return IndexReport(
        ingest=ingest_report,
        indexed_chunks=len(chunks),
        removed_sources=[],
        reindexed=True,
        total_chunks=store.count(),
    )


def index_notebook(
    nb: Notebook,
    *,
    force: bool = False,
    reindex: bool = False,
    settings: Settings | None = None,
    embedder=None,
    progress: Callable[[str, int, int], None] | None = None,
) -> IndexReport:
    """Ingest `nb` (parse/chunk) and keep its search index in sync.

    Normally this incrementally embeds only added/changed chunks and deletes
    rows for changed/removed sources. It falls back to a full rebuild from
    the processed chunks on disk when `reindex=True`, when the existing
    store was built with a different embedding model/dimension, or when the
    store is empty but the manifest already has entries (e.g. the index
    directory was deleted, or this is a fresh clone of already-processed
    data).
    """
    if embedder is None:
        from notecast.index.embedder import get_embedder

        embedder = get_embedder(settings)

    indexed_chunks = 0
    removed_sources: list[str] = []
    changed = False

    try:
        store = LanceChunkStore(nb.index_dir, dim=embedder.dim, model_name=embedder.model_name)
    except IndexMismatchError:
        ingest_report = ingest_notebook(nb, force=force, progress=progress)
        return _rebuild_index(nb, embedder, ingest_report)

    from notecast.ingest import manifest as manifest_mod

    manifest_has_entries = bool(manifest_mod.load_manifest(nb).entries)
    needs_rebuild = reindex or (store.count() == 0 and manifest_has_entries)

    if needs_rebuild:
        ingest_report = ingest_notebook(nb, force=force, progress=progress)
        return _rebuild_index(nb, embedder, ingest_report)

    repaired = _repair_missing_sources(nb, store, embedder)

    def _on_chunks(source_path: str, chunks: list[Chunk]) -> None:
        nonlocal indexed_chunks, changed
        # Embed BEFORE touching the store: if embedding raises partway
        # through (e.g. a later batch), the source's existing rows must be
        # left untouched rather than deleted-and-not-replaced. The exception
        # propagates to `ingest_notebook`, which records it in
        # `report.failed` and drops the source's manifest entry so it's
        # retried (and self-healed if needed) on the next run.
        vectors = _embed_chunks(embedder, chunks) if chunks else []
        store.delete_source(source_path)
        if chunks:
            store.upsert(chunks, vectors)
            indexed_chunks += len(chunks)
        changed = True

    def _on_removed(source_path: str) -> None:
        nonlocal changed
        store.delete_source(source_path)
        removed_sources.append(source_path)
        changed = True

    ingest_report = ingest_notebook(
        nb,
        force=force,
        on_chunks=_on_chunks,
        on_removed=_on_removed,
        progress=progress,
    )

    if changed:
        store.rebuild_text_index()

    return IndexReport(
        ingest=ingest_report,
        indexed_chunks=indexed_chunks,
        removed_sources=removed_sources,
        reindexed=False,
        total_chunks=store.count(),
        repaired=repaired,
    )


def get_retriever(nb: Notebook, *, settings: Settings | None = None, embedder=None) -> Retriever:
    """Return a `Retriever` over `nb`'s existing search index.

    Raises `IndexNotBuiltError` if the notebook hasn't been indexed yet (or
    the index is empty).
    """
    if settings is None:
        from notecast.config import get_settings

        settings = get_settings()
    if embedder is None:
        from notecast.index.embedder import get_embedder

        embedder = get_embedder(settings)

    try:
        store = LanceChunkStore(nb.index_dir, dim=embedder.dim, model_name=embedder.model_name)
    except IndexMismatchError as exc:
        raise IndexNotBuiltError(
            f"Notebook {nb.slug!r} was indexed with a different embedding model. "
            f"Run `notecast ingest {nb.slug} --reindex` first."
        ) from exc

    if store.count() == 0:
        raise IndexNotBuiltError(
            f"Notebook {nb.slug!r} has no search index yet. Run `notecast ingest {nb.slug}` first."
        )

    reranker = get_reranker(settings)
    return Retriever(store, embedder, reranker, candidates=settings.rerank_candidates)


__all__ = ["IndexNotBuiltError", "IndexReport", "open_store", "index_notebook", "get_retriever"]
