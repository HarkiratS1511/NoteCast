"""The `notecast` command-line tool. For now this covers creating and
listing notebooks, plus placeholders for ingest/chat/audio, which arrive in
later phases.
"""

from __future__ import annotations

import typer

from notecast import __version__
from notecast.ingest.pipeline import ingest_notebook
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


@app.command("ingest")
def ingest(
    slug: str,
    force: bool = typer.Option(
        False, "--force", help="Reprocess every file even if its content is unchanged."
    ),
) -> None:
    """Parse and chunk a notebook's source files.

    Search indexing (embedding + LanceDB) arrives in Phase 2; this command
    covers parsing, chunking and the manifest.
    """
    try:
        notebook = Notebook(slug)
    except ValueError as exc:
        typer.echo(f"Could not use notebook: {exc}")
        raise typer.Exit(code=1) from None
    if not notebook.exists():
        typer.echo(f"No notebook named {slug!r}. Create one with: notecast notebooks create {slug}")
        raise typer.Exit(code=1)

    def _progress(source_path: str, index: int, total: int) -> None:
        typer.echo(f"[{index}/{total}] {source_path}")

    report = ingest_notebook(notebook, force=force, progress=_progress)

    typer.echo("")
    typer.echo(
        f"Added {len(report.added)}, updated {len(report.updated)}, "
        f"unchanged {len(report.unchanged)}, removed {len(report.removed)}, "
        f"failed {len(report.failed)}."
    )
    typer.echo(f"Chunks produced this run: {report.chunk_count}")
    if report.failed:
        typer.echo("")
        typer.echo("Failures:")
        for source_path, message in report.failed.items():
            typer.echo(f"  {source_path}: {message}")

    typer.echo("")
    typer.echo("Search indexing comes in Phase 2 — this ran parsing and chunking only.")

    if report.failed:
        raise typer.Exit(code=1)


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
