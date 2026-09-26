"""The `notecast` command-line tool. For now this covers creating and
listing notebooks, plus placeholders for ingest/chat/audio, which arrive in
later phases.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click
import typer

from notecast import __version__
from notecast.audio.models import AudioScope, AudioScript
from notecast.audio.render import estimate_render_seconds
from notecast.audio.service import (
    OverviewFailed,
    estimate_overview_cost,
    generate_overview,
    render_saved_script,
)
from notecast.audio.tts import InvalidVoiceError, TTSUnavailableError
from notecast.chat.client import ChatError, MissingApiKeyError, get_client
from notecast.chat.models import ChatAnswer, ChatMode
from notecast.chat.pricing import estimate_cost as estimate_api_cost
from notecast.chat.pricing import provider_label
from notecast.chat.session import ChatSession
from notecast.config import get_settings
from notecast.eval.retrieval import format_report, load_eval_cases, run_retrieval_eval
from notecast.index.embedder import EmbedderUnavailableError
from notecast.index.service import IndexNotBuiltError, get_retriever, index_notebook
from notecast.index.store import IndexMismatchError, SearchMode
from notecast.ingest.course import display_name, load_course_config
from notecast.models import Chunk, SearchFilters
from notecast.notebook import Notebook

_YES_HELP = "Skip confirmation prompts (for API spend)."

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


def _make_filters(weeks: list[int]) -> SearchFilters | None:
    return SearchFilters(weeks=list(weeks)) if weeks else None


def _format_cost_or_na(cost: float | None) -> str:
    if cost is None:
        return "n/a (set NOTECAST_AGENTAUS_PRICE_* to estimate)"
    return f"${cost:.4f}"


def _print_answer(answer: ChatAnswer) -> None:
    typer.echo(answer.text)
    if answer.not_in_sources:
        typer.secho("Not in your course material.", fg=typer.colors.YELLOW)
    if answer.citations:
        typer.echo("")
        typer.echo("Sources:")
        typer.echo(answer.citations_markdown())
    usage = answer.usage
    total_tokens = usage.input_tokens + usage.output_tokens
    cost = _format_cost_or_na(usage.est_cost_usd)
    typer.echo("")
    typer.secho(f"[{total_tokens} tokens, est cost {cost}]", dim=True)


def _get_retriever_or_exit(notebook: Notebook):  # noqa: ANN201
    try:
        return get_retriever(notebook)
    except IndexNotBuiltError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from None


def _filter_chunks(chunks: list[Chunk], filters: SearchFilters | None) -> list[Chunk]:
    """Apply a `SearchFilters` to a full chunk list (weeks/source_types/
    source_paths), for deep mode's `chunk_source`.
    """
    if filters is None:
        return chunks
    result = chunks
    if filters.weeks:
        weeks = set(filters.weeks)
        result = [c for c in result if c.week in weeks]
    if filters.source_types:
        types = set(filters.source_types)
        result = [c for c in result if c.source_type in types]
    if filters.source_paths:
        paths = set(filters.source_paths)
        result = [c for c in result if c.source_path in paths]
    return result


def _make_chat_session(notebook: Notebook, retriever) -> ChatSession:  # noqa: ANN001
    course_name = display_name(notebook, load_course_config(notebook))
    try:
        return ChatSession(
            retriever,
            course_name=course_name,
            chunk_source=lambda filters: _filter_chunks(retriever.store.all_chunks(), filters),
        )
    except MissingApiKeyError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from None


def _confirm_deep_mode(
    session: ChatSession,
    filters: SearchFilters | None,
    yes: bool,
    confirmed_tokens: int | None = None,
) -> tuple[bool, int | None]:
    """Print deep mode's cost estimate and ask for confirmation, unless
    `yes` or the current scope's token estimate hasn't grown past
    `confirmed_tokens` (the size last confirmed in this session, if any --
    a REPL passes this back in after a `/week` change grows the scope, so
    the increase gets its own confirmation instead of riding on an earlier,
    smaller one).

    Returns `(proceed, new_confirmed_tokens)`: `proceed` is True if the
    caller should go ahead with the deep question, and `new_confirmed_tokens`
    is what the caller should remember as "confirmed" for next time (it's
    unchanged when declined or when re-confirmation wasn't needed).
    """
    try:
        tokens, _total_cost = session.estimate_deep_cost(filters)
    except ChatError as exc:
        typer.echo(str(exc))
        return False, confirmed_tokens

    if confirmed_tokens is not None and tokens <= confirmed_tokens:
        return True, confirmed_tokens

    settings = session.settings
    model = settings.deep_model
    if settings.provider == "agentaus":
        cost = estimate_api_cost(model, {"input_tokens": tokens}, settings=settings)
        first_cost, follow_up_cost = cost, cost
    else:
        first_cost = estimate_api_cost(model, {"cache_write_tokens": tokens})
        follow_up_cost = estimate_api_cost(model, {"cache_read_tokens": tokens})
    typer.echo(
        f"Deep mode will send ~{tokens:,} tokens: "
        f"{_format_cost_or_na(first_cost)} first question, "
        f"{_format_cost_or_na(follow_up_cost)} per follow-up."
    )
    if yes:
        return True, tokens
    if not typer.confirm("Continue?"):
        typer.echo("Cancelled.")
        return False, confirmed_tokens
    return True, tokens


@app.command("ask")
def ask(
    slug: str,
    question: str,
    mode: ChatMode = typer.Option(  # noqa: B008
        "sources", "--mode", help="sources, open or deep."
    ),
    week: list[int] = typer.Option(  # noqa: B008
        None, "--week", help="Restrict to this week (repeatable)."
    ),
    k: int = typer.Option(None, "-k", help="Number of chunks to retrieve."),  # noqa: B008
    yes: bool = typer.Option(False, "--yes", "-y", help=_YES_HELP),
) -> None:
    """Ask a single grounded question of a notebook's material."""
    notebook = _load_notebook_or_exit(slug)
    retriever = _get_retriever_or_exit(notebook)
    session = _make_chat_session(notebook, retriever)
    filters = _make_filters(week)

    if mode == "deep":
        proceed, _confirmed_tokens = _confirm_deep_mode(session, filters, yes)
        if not proceed:
            raise typer.Exit(code=1)

    try:
        answer = session.ask(question, mode=mode, filters=filters, k=k)
    except MissingApiKeyError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from None
    except ChatError as exc:
        typer.echo(f"Chat failed: {exc}")
        raise typer.Exit(code=1) from None

    _print_answer(answer)


_CHAT_HELP = "Commands: /mode sources|open|deep, /week N (or /week all), /reset, /quit"


@app.command("chat")
def chat(
    slug: str,
    mode: ChatMode = typer.Option(  # noqa: B008
        "sources", "--mode", help="sources, open or deep."
    ),
    week: list[int] = typer.Option(  # noqa: B008
        None, "--week", help="Restrict to this week (repeatable)."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help=_YES_HELP),
) -> None:
    """Chat interactively with a notebook's material."""
    notebook = _load_notebook_or_exit(slug)
    retriever = _get_retriever_or_exit(notebook)
    session = _make_chat_session(notebook, retriever)

    current_mode: ChatMode = mode
    filters = _make_filters(week)
    total_cost = 0.0
    confirmed_deep_tokens: int | None = None

    typer.echo(f"Chatting with {slug!r} (mode: {current_mode}).")
    typer.echo(_CHAT_HELP)

    while True:
        try:
            line = click.prompt("you", prompt_suffix="> ")
        except (EOFError, click.exceptions.Abort):
            typer.echo("")
            break

        line = line.strip()
        if not line:
            continue

        if line in ("/quit", "/exit"):
            break
        if line == "/reset":
            session.reset()
            typer.echo("History cleared.")
            continue
        if line.startswith("/mode"):
            parts = line.split()
            if len(parts) == 2 and parts[1] in ("sources", "open", "deep"):
                current_mode = parts[1]  # type: ignore[assignment]
                typer.echo(f"Mode set to {current_mode}.")
            else:
                typer.echo("Usage: /mode sources|open|deep")
            continue
        if line.startswith("/week"):
            parts = line.split()
            if len(parts) == 2 and parts[1] == "all":
                filters = None
                typer.echo("Week filter cleared.")
            elif len(parts) == 2 and parts[1].isdigit():
                filters = SearchFilters(weeks=[int(parts[1])])
                typer.echo(f"Restricted to week {parts[1]}.")
            else:
                typer.echo("Usage: /week N or /week all")
            continue

        if current_mode == "deep":
            proceed, confirmed_deep_tokens = _confirm_deep_mode(
                session, filters, yes, confirmed_deep_tokens
            )
            if not proceed:
                continue

        try:
            answer = session.ask(line, mode=current_mode, filters=filters)
        except (MissingApiKeyError, ChatError) as exc:
            typer.echo(f"Error: {exc}")
            continue
        except ValueError as exc:
            typer.echo(str(exc))
            continue

        _print_answer(answer)
        if answer.usage.est_cost_usd:
            total_cost += answer.usage.est_cost_usd

    typer.echo(f"Total est cost this session: ${total_cost:.4f}")


def _print_plan_summary(script: AudioScript) -> None:
    plan = script.plan
    tier_a = sum(1 for point in plan.points if point.tier == "A")
    tier_b = sum(1 for point in plan.points if point.tier == "B")

    typer.echo("")
    typer.echo(script.title)
    typer.echo(f"Target length: {plan.target_minutes:.1f} min, {len(plan.chapters)} chapters")
    typer.echo(f"Tier A points: {tier_a}, Tier B points: {tier_b}")
    for chapter in plan.chapters:
        typer.echo(f"  {chapter.index}. {chapter.title}")
    if plan.notes:
        typer.echo("Notes: " + "; ".join(plan.notes))

    coverage = script.coverage
    typer.echo(
        f"Coverage: {len(coverage.covered)} covered, {len(coverage.missing)} missing, "
        f"{len(coverage.patched)} patched"
    )
    if coverage.missing:
        typer.echo("  Still missing: " + ", ".join(coverage.missing))


def _progress_echo(stage: str, fraction: float) -> None:
    typer.echo(f"[{fraction * 100:5.1f}%] {stage}")


@app.command("audio")
def audio(
    slug: str,
    week: list[int] = typer.Option(  # noqa: B008
        None, "--week", help="Restrict to this week (repeatable)."
    ),
    source: list[str] = typer.Option(  # noqa: B008
        None, "--source", help="Restrict to this source path (repeatable)."
    ),
    focus: str | None = typer.Option(
        None, "--focus", help="Optional focus, e.g. 'exam prep' or 'just the maths'."
    ),
    script_only: bool = typer.Option(
        False, "--script-only", help="Write the script but don't render audio."
    ),
    minutes_max: int | None = typer.Option(
        None, "--minutes-max", help="Override the maximum length (minutes) for this run."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help=_YES_HELP),
) -> None:
    """Generate an audio overview for a notebook."""
    notebook = _load_notebook_or_exit(slug)
    settings = get_settings()
    if minutes_max is not None:
        settings = settings.model_copy(update={"audio_max_minutes": minutes_max})

    scope = AudioScope(
        weeks=list(week) if week else None,
        source_paths=list(source) if source else None,
        focus=focus,
    )

    try:
        estimate = estimate_overview_cost(notebook, scope, settings)
    except IndexNotBuiltError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from None
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from None

    typer.echo(
        f"Estimated cost: ~${estimate.est_cost_usd:.4f} "
        f"(range ${estimate.est_cost_low_usd:.4f}–${estimate.est_cost_high_usd:.4f})"
    )
    if not yes and not typer.confirm("Continue?"):
        typer.echo("Cancelled.")
        raise typer.Exit(code=1)

    try:
        result = generate_overview(
            notebook, scope, settings=settings, render=False, progress=_progress_echo
        )
    except MissingApiKeyError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from None
    except IndexNotBuiltError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from None
    except OverviewFailed as exc:
        typer.echo(
            f"Audio generation failed at {exc.stage}: {exc} (spent ~${exc.est_cost_usd:.4f})"
        )
        raise typer.Exit(code=1) from None
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from None
    except ChatError as exc:
        typer.echo(f"Audio generation failed: {exc}")
        raise typer.Exit(code=1) from None

    _print_plan_summary(result.script)
    typer.echo(f"Estimated API cost: ${result.est_cost_usd:.4f}")
    typer.echo(f"Script saved to: {result.script_json_path}")

    if script_only:
        return

    est_seconds = estimate_render_seconds(result.script)
    typer.echo(f"Estimated render time: ~{est_seconds:.0f}s")

    try:
        saved = render_saved_script(
            notebook, result.script_json_path, settings=settings, progress=_progress_echo
        )
    except (TTSUnavailableError, InvalidVoiceError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from None

    _warn_if_stale(saved.stale_chunk_ids)
    typer.echo(f"Audio: {saved.render.mp3_path}")
    typer.echo(f"Transcript: {saved.render.transcript_path}")
    typer.echo(f"Script JSON: {result.script_json_path}")


def _warn_if_stale(stale_chunk_ids: list[str]) -> None:
    if not stale_chunk_ids:
        return
    typer.secho(
        f"Note: {len(stale_chunk_ids)} cited chunk(s) are no longer in the current index "
        "(the notebook was likely re-ingested since this script was written); the "
        "transcript's Sources lines may be incomplete for those lines.",
        fg=typer.colors.YELLOW,
    )


@app.command("audio-render")
def audio_render(slug: str, script_json: Path) -> None:
    """Re-render a previously saved script (no LLM API calls)."""
    notebook = _load_notebook_or_exit(slug)
    if not script_json.exists():
        typer.echo(f"No script file at {script_json}")
        raise typer.Exit(code=1)

    try:
        script = AudioScript.model_validate_json(script_json.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        typer.echo(f"Could not read script file: {exc}")
        raise typer.Exit(code=1) from None

    est_seconds = estimate_render_seconds(script)
    typer.echo(f"Estimated render time: ~{est_seconds:.0f}s")

    try:
        saved = render_saved_script(notebook, script_json, progress=_progress_echo)
    except IndexNotBuiltError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from None
    except (TTSUnavailableError, InvalidVoiceError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from None

    _warn_if_stale(saved.stale_chunk_ids)
    typer.echo(f"Audio: {saved.render.mp3_path}")
    typer.echo(f"Transcript: {saved.render.transcript_path}")


def _get_field(obj: object, key: str, default: object = None) -> object:
    """Read `key` off `obj`, whether it's a dict-like fake response (used in
    tests) or a real SDK object with attributes.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@app.command("provider-check")
def provider_check() -> None:
    """Check the active LLM provider's configuration with a live request."""
    settings = get_settings()
    typer.echo(f"Provider: {provider_label(settings)} ({settings.provider})")
    if settings.provider == "agentaus":
        typer.echo(f"Base URL: {settings.agentaus_base_url}")
    typer.echo(f"Chat model:   {settings.chat_model}")
    typer.echo(f"Helper model: {settings.helper_model}")
    typer.echo(f"Script model: {settings.script_model}")
    typer.echo(f"Deep model:   {settings.deep_model}")

    try:
        client = get_client(settings)
    except MissingApiKeyError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from None

    if hasattr(client, "list_models"):
        try:
            models = client.list_models()
        except ChatError as exc:
            typer.echo(f"Could not list models: {exc}")
        else:
            typer.echo(f"Available models: {', '.join(models)}")

    typer.echo("")
    typer.echo("Sending a test request...")
    try:
        message = client.messages.create(
            model=settings.chat_model,
            max_tokens=20,
            system="Reply with the single word: ok",
            messages=[{"role": "user", "content": "ping"}],
        )
    except (MissingApiKeyError, ChatError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from None

    text = "".join(
        _get_field(block, "text", "") or ""
        for block in _get_field(message, "content", []) or []
        if _get_field(block, "type") == "text"
    ).strip()
    usage = _get_field(message, "usage")
    input_tokens = _get_field(usage, "input_tokens")
    output_tokens = _get_field(usage, "output_tokens")
    typer.echo(f"Reply: {text!r}")
    typer.echo(f"Usage: input={input_tokens}, output={output_tokens}")


@app.command("ui")
def ui(
    port: int = typer.Option(8501, "--port", help="Port for the local web UI."),
) -> None:
    """Open the NoteCast web UI in your browser (runs locally, Ctrl-C to stop)."""
    app_path = Path(__file__).resolve().parent / "ui" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
    typer.echo(f"Starting NoteCast UI on http://localhost:{port} (Ctrl-C to stop)")
    try:
        result = subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        return
    raise typer.Exit(code=result.returncode)


if __name__ == "__main__":
    app()
