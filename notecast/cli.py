"""The `notecast` command-line tool. For now this covers creating and
listing notebooks, plus placeholders for ingest/chat/audio, which arrive in
later phases.
"""

from __future__ import annotations

from pathlib import Path

import typer

from notecast import __version__
from notecast.eval.retrieval import format_report, load_eval_cases, run_retrieval_eval
from notecast.index.embedder import EmbedderUnavailableError
from notecast.index.service import IndexNotBuiltError, get_retriever, index_notebook
from notecast.index.store import IndexMismatchError, SearchMode
from notecast.models import SearchFilters
from notecast.notebook import Notebook

app = typer.Typer(
    add_completion=False, help="NoteCast: notes and audio overviews for your courses."
)
notebooks_app = typer.Typer(help="Create and list notebooks (one per course).")
app.add_typer(notebooks_app, name="notebooks")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"notecast {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the installed version and exit.",
    ),
) -> None:
    """NoteCast: notes and audio overviews for your courses."""


@notebooks_app.command("list")
def notebooks_list() -> None:
    """List every notebook and how many source files it has."""
    notebooks = Notebook.list_all()
    if not notebooks:
        typer.echo("No notebooks yet. Create one with: notecast notebooks create <slug>")
        return
    for nb in notebooks:
        count = len(nb.source_files())
        noun = "file" if count == 1 else "files"
        typer.echo(f"{nb.slug}  ({count} source {noun})")


@notebooks_app.command("create")
def notebooks_create(slug: str) -> None:
    """Create a new, empty notebook."""
    try:
        notebook = Notebook.create(slug)
    except ValueError as exc:
        typer.echo(f"Could not create notebook: {exc}")
        raise typer.Exit(code=1) from None
    typer.echo(f"Created notebook {slug!r}. Drop your course files into:")
    typer.echo(f"  {notebook.sources_dir}")


def _not_implemented(slug: str, phase: int) -> None:
    try:
        Notebook(slug)
    except ValueError as exc:
        typer.echo(f"Could not use notebook: {exc}")
        raise typer.Exit(code=1) from None
    typer.echo(f"Not implemented yet — coming in Phase {phase}.")


def _load_notebook_or_exit(slug: str) -> Notebook:
    try:
        notebook = Notebook(slug)
    except ValueError as exc:
        typer.echo(f"Could not use notebook: {exc}")
        raise typer.Exit(code=1) from None
    return notebook


@app.command("ingest")
def ingest(
    slug: str,
    force: bool = typer.Option(
        False, "--force", help="Reprocess every file even if its content is unchanged."
    ),
    reindex: bool = typer.Option(
        False, "--reindex", help="Rebuild the search index from scratch after ingesting."
    ),
) -> None:
    """Parse and chunk a notebook's source files, then embed and index them
    for search.
    """
    notebook = _load_notebook_or_exit(slug)
    if not notebook.exists():
        typer.echo(f"No notebook named {slug!r}. Create one with: notecast notebooks create {slug}")
        raise typer.Exit(code=1)

    def _progress(source_path: str, index: int, total: int) -> None:
        typer.echo(f"[{index}/{total}] {source_path}")

    try:
        result = index_notebook(notebook, force=force, reindex=reindex, progress=_progress)
    except EmbedderUnavailableError as exc:
        typer.echo(f"Could not load the embedding model: {exc}")
        raise typer.Exit(code=1) from None
    except IndexMismatchError as exc:
        typer.echo(f"Search index error: {exc}")
        raise typer.Exit(code=1) from None
    except Exception as exc:  # noqa: BLE001 -- surface any other failure cleanly, no traceback
        typer.echo(f"Ingest failed ({type(exc).__name__}): {exc}")
        raise typer.Exit(code=1) from None

    report = result.ingest

    typer.echo("")
    typer.echo(
        f"Added {len(report.added)}, updated {len(report.updated)}, "
        f"unchanged {len(report.unchanged)}, removed {len(report.removed)}, "
        f"failed {len(report.failed)}."
    )
    typer.echo(f"Chunks produced this run: {report.chunk_count}")
    typer.echo(
        f"indexed {result.indexed_chunks} chunks (total {result.total_chunks} in index)"
        + (" [reindexed]" if result.reindexed else "")
    )
    if result.repaired:
        typer.echo(f"repaired {len(result.repaired)} source(s): {', '.join(result.repaired)}")
    if report.failed:
        typer.echo("")
        typer.echo("Failures:")
        for source_path, message in report.failed.items():
            typer.echo(f"  {source_path}: {message}")
        raise typer.Exit(code=1)


@app.command("search")
def search(
    slug: str,
    query: str,
    k: int = typer.Option(5, "-k", help="Number of results to show."),
    week: list[int] = typer.Option(  # noqa: B008
        None, "--week", help="Restrict to this week (repeatable)."
    ),
    mode: SearchMode = typer.Option(  # noqa: B008
        "hybrid", "--mode", help="Search mode: hybrid, vector or keyword."
    ),
) -> None:
    """Search a notebook's indexed material."""
    notebook = _load_notebook_or_exit(slug)
    try:
        retriever = get_retriever(notebook)
    except IndexNotBuiltError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from None

    filters = SearchFilters(weeks=list(week)) if week else None
    hits = retriever.search(query, k=k, filters=filters, mode=mode)

    if not hits:
        typer.echo("No results.")
        return

    for rank, hit in enumerate(hits, start=1):
        label = hit.chunk.location.label()
        header = f"{hit.chunk.source_path}" + (f" ({label})" if label else "")
        typer.echo(f"{rank}. [{hit.score:.3f}] {header}")
        snippet = hit.chunk.text.strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200].rstrip() + "..."
        typer.echo(f"   {snippet}")


@app.command("eval")
def eval_cmd(
    slug: str,
    k: int = typer.Option(10, "-k", help="Number of results to consider per question."),
    mode: SearchMode = typer.Option(  # noqa: B008
        "hybrid", "--mode", help="Search mode: hybrid, vector or keyword."
    ),
    file: Path | None = typer.Option(  # noqa: B008
        None, "--file", help="Eval YAML file (default: notebooks/<slug>/eval.yaml)."
    ),
) -> None:
    """Run a retrieval eval set against a notebook's indexed material."""
    notebook = _load_notebook_or_exit(slug)
    eval_path = file if file is not None else notebook.path / "eval.yaml"
    if not eval_path.exists():
        typer.echo(f"No eval file found at {eval_path}.")
        raise typer.Exit(code=1)

    try:
        cases = load_eval_cases(eval_path)
    except ValueError as exc:
        typer.echo(f"Could not load eval file: {exc}")
        raise typer.Exit(code=1) from None

    try:
        retriever = get_retriever(notebook)
    except IndexNotBuiltError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from None

    class _ModeRetriever:
        def search(self, query: str, k: int = 10, filters: SearchFilters | None = None):
            return retriever.search(query, k=k, filters=filters, mode=mode)

    report = run_retrieval_eval(_ModeRetriever(), cases, k=k)
    typer.echo(format_report(report))


@app.command("chat")
def chat(slug: str) -> None:
    """Chat with a notebook's material. (Phase 3)"""
    _not_implemented(slug, phase=3)


@app.command("audio")
def audio(slug: str) -> None:
    """Generate an audio overview for a notebook. (Phase 6)"""
    _not_implemented(slug, phase=6)


if __name__ == "__main__":
    app()
