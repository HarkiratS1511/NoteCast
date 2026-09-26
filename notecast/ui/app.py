"""NoteCast's local Streamlit web UI: notebook management, grounded chat,
and audio overviews. Run with `streamlit run notecast/ui/app.py`.
"""

from __future__ import annotations

import streamlit as st

from notecast.audio.models import AudioScope
from notecast.audio.service import (
    OverviewFailed,
    OverviewResult,
    estimate_overview_cost,
    generate_overview,
    render_saved_script,
)
from notecast.audio.tts import InvalidVoiceError, TTSUnavailableError
from notecast.chat.client import ChatError, MissingApiKeyError
from notecast.chat.models import ChatAnswer
from notecast.config import get_settings
from notecast.index.embedder import EmbedderUnavailableError
from notecast.index.service import IndexNotBuiltError, get_retriever, index_notebook
from notecast.index.store import IndexMismatchError
from notecast.ingest.course import display_name, load_course_config
from notecast.notebook import Notebook
from notecast.ui import components, state

st.set_page_config(page_title="NoteCast", page_icon="🎧", layout="wide")


# --- Cached-per-run helpers --------------------------------------------------


def _get_retriever(nb: Notebook):  # noqa: ANN201
    """Return `nb`'s retriever, cached in session state until an ingest run
    invalidates it. Raises `IndexNotBuiltError` if it hasn't been indexed.
    """
    cache = st.session_state.setdefault("retrievers", {})
    if nb.slug not in cache:
        cache[nb.slug] = get_retriever(nb)
    return cache[nb.slug]


def _invalidate_notebook_caches(slug: str) -> None:
    st.session_state.setdefault("retrievers", {}).pop(slug, None)
    st.session_state.setdefault("chat_sessions", {}).pop(slug, None)


def _get_chat_session(nb: Notebook, retriever):  # noqa: ANN201
    sessions = st.session_state.setdefault("chat_sessions", {})
    if nb.slug not in sessions:
        course_name = display_name(nb, load_course_config(nb))
        sessions[nb.slug] = state.chat_session_factory(
            retriever,
            course_name=course_name,
            chunk_source=lambda filters, retr=retriever: state.filter_chunks(
                retr.store.all_chunks(), filters
            ),
        )
    return sessions[nb.slug]


# --- Sidebar -----------------------------------------------------------------


def _render_sidebar() -> str | None:
    st.sidebar.title("NoteCast")

    notebooks = Notebook.list_all()
    slugs = [nb.slug for nb in notebooks]
    labels = {nb.slug: display_name(nb, load_course_config(nb)) for nb in notebooks}

    selected_slug: str | None = None
    if slugs:
        default_slug = st.session_state.get("notebook_slug")
        if default_slug not in slugs:
            default_slug = slugs[0]
        selected_slug = st.sidebar.selectbox(
            "Notebook",
            slugs,
            index=slugs.index(default_slug),
            format_func=lambda s: labels[s],
            key="notebook_slug",
        )
    else:
        st.sidebar.info("No notebooks yet — create one below.")

    with st.sidebar.expander("New notebook", expanded=not slugs):
        new_slug = st.text_input("Slug (e.g. comp3000)", key="new_notebook_slug_input")
        if st.button("Create notebook", key="create_notebook_btn"):
            error = state.validate_new_slug(new_slug, slugs)
            if error:
                st.error(error)
            else:
                Notebook.create(new_slug.strip())
                st.success(f"Created {new_slug.strip()!r}.")
                st.session_state["notebook_slug"] = new_slug.strip()
                st.rerun()

    if selected_slug is None:
        return None

    nb = Notebook(selected_slug)

    st.sidebar.markdown("**Add material**")
    st.sidebar.caption(
        "Files stay on this computer — notebooks/ is git-ignored and never committed."
    )
    week_input = st.sidebar.number_input(
        "Week (optional, 0 = none)",
        min_value=0,
        max_value=52,
        value=0,
        step=1,
        key=f"upload_week_{selected_slug}",
    )
    uploads = st.sidebar.file_uploader(
        "Upload files",
        type=["pdf", "pptx", "docx", "vtt", "srt", "txt", "md"],
        accept_multiple_files=True,
        key=f"uploader_{selected_slug}",
    )
    if uploads and st.sidebar.button("Save files", key=f"save_uploads_{selected_slug}"):
        nb.ensure_dirs()
        try:
            saved_names = [
                state.save_uploaded_file(nb, f.getvalue(), f.name, week_input or None).name
                for f in uploads
            ]
        except (OSError, ValueError) as exc:
            st.sidebar.error(f"Could not save uploaded files: {exc}")
        else:
            st.sidebar.success(f"Saved {len(saved_names)} file(s): {', '.join(saved_names)}")
            st.rerun()

    if st.sidebar.button("Ingest / update index", key=f"ingest_btn_{selected_slug}"):
        progress_bar = st.sidebar.progress(0.0)
        status = st.sidebar.empty()

        def _progress(source_path: str, index: int, total: int) -> None:
            fraction = index / total if total else 1.0
            progress_bar.progress(min(max(fraction, 0.0), 1.0))
            status.text(f"[{index}/{total}] {source_path}")

        try:
            report = index_notebook(nb, progress=_progress)
        except EmbedderUnavailableError as exc:
            st.sidebar.error(f"Could not load the embedding model: {exc}")
        except IndexMismatchError as exc:
            st.sidebar.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - never show a raw traceback in the UI
            st.sidebar.error(f"Ingest failed: {exc}")
        else:
            progress_bar.progress(1.0)
            _invalidate_notebook_caches(selected_slug)
            st.sidebar.success(components.format_index_summary(report))
            if report.ingest.failed:
                for source_path, message in report.ingest.failed.items():
                    st.sidebar.warning(f"{source_path}: {message}")

    with st.sidebar.expander("Source files"):
        grouped = state.group_sources_by_week(nb)
        if not grouped:
            st.caption("No source files yet.")
        for week, entries in grouped:
            label = f"Week {week}" if week is not None else "Unfiled"
            st.markdown(f"**{label}**")
            for source_path, chunk_count in entries:
                st.write(f"{source_path} — {chunk_count} chunk(s)")

    settings = get_settings()
    with st.sidebar.expander("Settings"):
        st.write(f"Chat model: `{settings.chat_model}`")
        st.write(f"Helper model: `{settings.helper_model}`")
        st.write(f"Script model: `{settings.script_model}`")
        st.write(f"Deep model: `{settings.deep_model}`")
        st.write(f"Embedding model: `{settings.embedding_model}` ({settings.embedding_backend})")
        if state.api_key_configured(settings):
            st.success(f"{state.provider_label(settings)} API access is configured.")
        else:
            st.warning(state.missing_api_key_hint(settings))

    st.sidebar.caption(f"LLM: {state.provider_label(settings)} · {settings.chat_model}")

    history = st.session_state.get("chat_history", {}).get(selected_slug, [])
    total_cost = sum(a.usage.est_cost_usd or 0.0 for a in history)
    st.sidebar.caption(f"Session cost ({labels[selected_slug]}): {state.format_usd(total_cost)}")

    return selected_slug


# --- Chat tab ------------------------------------------------------------


def _render_answer(answer: ChatAnswer) -> None:
    st.markdown(answer.text)
    if answer.not_in_sources:
        label = state.provider_label(get_settings())
        st.info(
            f"Not in your course material. Try Open mode for {label}'s general "
            "knowledge and web search."
        )
    if answer.citations:
        with st.expander(f"Sources ({len(answer.citations)})"):
            for citation in answer.citations:
                st.markdown(f"**{components.citation_heading(citation)}**")
                st.markdown(components.citation_body(citation))
    total_tokens = answer.usage.input_tokens + answer.usage.output_tokens
    st.caption(f"{state.format_usd(answer.usage.est_cost_usd)} · {total_tokens} tokens")


def _send_question(nb: Notebook, slug: str, question: str, mode: str, filters) -> bool:  # noqa: ANN001
    """Ask `question` and record the answer. Returns whether it succeeded,
    so the caller only reruns (clearing the pending question) on success --
    rerunning after a failure would wipe the just-shown `st.error` before
    anyone could read it.
    """
    try:
        retriever = _get_retriever(nb)
    except IndexNotBuiltError as exc:
        st.error(str(exc))
        return False
    session = _get_chat_session(nb, retriever)
    try:
        answer = session.ask(question, mode=mode, filters=filters)
    except MissingApiKeyError as exc:
        st.error(str(exc))
        return False
    except ChatError as exc:
        st.error(f"Chat failed: {exc}")
        return False
    except ValueError as exc:
        st.error(str(exc))
        return False
    st.session_state.setdefault("chat_history", {}).setdefault(slug, []).append(answer)
    return True


def _render_chat_tab(nb: Notebook, slug: str) -> None:
    mode_labels = {"Sources only": "sources", "Open (web)": "open", "Deep — whole scope": "deep"}
    mode_label = st.radio("Mode", list(mode_labels), key=f"mode_{slug}", horizontal=True)
    mode = mode_labels[mode_label]

    weeks_present = [w for w, _ in state.group_sources_by_week(nb) if w is not None]
    selected_weeks = st.multiselect("Weeks", weeks_present, key=f"chat_weeks_{slug}")
    filters = state.build_search_filters(selected_weeks)

    if st.button("New conversation", key=f"reset_chat_{slug}"):
        session = st.session_state.get("chat_sessions", {}).get(slug)
        if session is not None:
            session.reset()
        st.session_state.setdefault("chat_history", {})[slug] = []
        st.session_state.pop(f"deep_confirmed_{slug}", None)
        st.rerun()

    for answer in st.session_state.setdefault("chat_history", {}).get(slug, []):
        with st.chat_message("user"):
            st.write(answer.question)
        with st.chat_message("assistant"):
            _render_answer(answer)

    pending_key = f"pending_question_{slug}"
    question = st.chat_input("Ask a question about your course material")
    if question:
        st.session_state[pending_key] = question

    pending = st.session_state.get(pending_key)
    if not pending:
        return

    if mode != "deep":
        with st.chat_message("user"):
            st.write(pending)
        with st.spinner("Thinking..."):
            sent_ok = _send_question(nb, slug, pending, mode, filters)
        st.session_state.pop(pending_key, None)
        if sent_ok:
            st.rerun()
        return

    confirmed_key = f"deep_confirmed_{slug}"
    scope_key = state.deep_scope_key(selected_weeks)

    try:
        retriever = _get_retriever(nb)
    except IndexNotBuiltError as exc:
        st.error(str(exc))
        st.session_state.pop(pending_key, None)
        return
    session = _get_chat_session(nb, retriever)
    try:
        tokens, first_cost, follow_up_cost = state.deep_cost_breakdown(session, filters)
    except ChatError as exc:
        st.error(str(exc))
        st.session_state.pop(pending_key, None)
        return

    confirmed = st.session_state.get(confirmed_key)
    already_confirmed = (
        confirmed is not None
        and confirmed.get("scope_key") == scope_key
        and tokens <= confirmed.get("tokens", -1)
    )
    if already_confirmed:
        with st.chat_message("user"):
            st.write(pending)
        with st.spinner("Thinking..."):
            sent_ok = _send_question(nb, slug, pending, mode, filters)
        st.session_state.pop(pending_key, None)
        if sent_ok:
            st.rerun()
        return

    # No confirmation on file yet for this scope, or the material has grown
    # since it was confirmed (e.g. new material was ingested) -- re-ask.
    st.warning(state.format_deep_estimate(tokens, first_cost, follow_up_cost))
    col1, col2 = st.columns(2)
    if col1.button("Confirm and send", key=f"confirm_deep_{slug}"):
        st.session_state[confirmed_key] = {"scope_key": scope_key, "tokens": tokens}
        st.rerun()
    if col2.button("Cancel", key=f"cancel_deep_{slug}"):
        st.session_state.pop(pending_key, None)
        st.rerun()


# --- Audio tab -----------------------------------------------------------


def _render_plan_summary(script) -> None:  # noqa: ANN001
    plan = script.plan
    tier_a = sum(1 for point in plan.points if point.tier == "A")
    tier_b = sum(1 for point in plan.points if point.tier == "B")
    st.write(
        f"**{len(plan.chapters)} chapters** — {plan.target_minutes:.1f} min target · "
        f"tier A: {tier_a}, tier B: {tier_b}"
    )
    for chapter in plan.chapters:
        st.write(f"{chapter.index}. {chapter.title}")
    coverage = script.coverage
    st.write(
        f"Coverage — covered: {len(coverage.covered)}, missing: {len(coverage.missing)}, "
        f"patched: {len(coverage.patched)}"
    )
    if coverage.missing:
        st.warning("Still missing: " + ", ".join(coverage.missing))


def _generate_overview_run(nb: Notebook, scope: AudioScope, settings) -> None:  # noqa: ANN001
    progress_bar = st.progress(0.0)
    status = st.empty()

    def _progress(stage: str, fraction: float) -> None:
        progress_bar.progress(min(max(fraction, 0.0), 1.0))
        status.text(stage)

    try:
        result: OverviewResult = generate_overview(nb, scope, settings=settings, progress=_progress)
    except MissingApiKeyError as exc:
        st.error(str(exc))
        return
    except IndexNotBuiltError as exc:
        st.error(str(exc))
        return
    except OverviewFailed as exc:
        st.error(
            f"Audio generation failed at {exc.stage}: {exc} "
            f"(spent ~{state.format_usd(exc.est_cost_usd)})"
        )
        return
    except (TTSUnavailableError, InvalidVoiceError) as exc:
        st.error(
            f"Audio generation failed while rendering speech: {exc} "
            "The script itself was already saved, so you can re-render it for free "
            "from “Re-render a saved script” below without calling the API again."
        )
        return
    except EmbedderUnavailableError as exc:
        st.error(f"Could not load the embedding model: {exc}")
        return
    except IndexMismatchError as exc:
        st.error(str(exc))
        return
    except (ChatError, ValueError, OSError) as exc:
        st.error(f"Audio generation failed: {exc}")
        return

    st.success(
        f"Generated “{result.script.title}”. Est. cost: {state.format_usd(result.est_cost_usd)}"
    )
    _render_plan_summary(result.script)

    if result.render is not None:
        st.audio(str(result.render.mp3_path))
        st.download_button(
            "Download MP3",
            result.render.mp3_path.read_bytes(),
            file_name=result.render.mp3_path.name,
            key=f"dl_mp3_{result.render.mp3_path.name}",
        )
        transcript_text = result.render.transcript_path.read_text(encoding="utf-8")
        st.download_button(
            "Download transcript",
            transcript_text,
            file_name=result.render.transcript_path.name,
            key=f"dl_transcript_{result.render.transcript_path.name}",
        )
        with st.expander("Transcript"):
            st.markdown(transcript_text)


def _render_rerender_section(nb: Notebook, slug: str) -> None:
    st.markdown("#### Re-render a saved script (free, no API calls)")
    scripts = components.list_saved_scripts(nb)
    if not scripts:
        st.caption("No saved scripts yet.")
        return
    chosen = st.selectbox(
        "Script", scripts, format_func=lambda p: p.name, key=f"script_select_{slug}"
    )
    if st.button("Re-render", key=f"rerender_btn_{slug}"):
        progress_bar = st.progress(0.0)
        status = st.empty()

        def _progress(stage: str, fraction: float) -> None:
            progress_bar.progress(min(max(fraction, 0.0), 1.0))
            status.text(stage)

        try:
            saved = render_saved_script(nb, chosen, progress=_progress)
        except (TTSUnavailableError, InvalidVoiceError) as exc:
            st.error(str(exc))
            return
        except IndexNotBuiltError as exc:
            st.error(str(exc))
            return
        st.success("Re-rendered.")
        st.audio(str(saved.render.mp3_path))
        if saved.stale_chunk_ids:
            st.warning(
                f"{len(saved.stale_chunk_ids)} cited chunk(s) are no longer in the current "
                "index — the transcript's Sources lines may be incomplete for those lines."
            )


def _render_audio_tab(nb: Notebook, slug: str) -> None:
    settings = get_settings()
    weeks_present = [w for w, _ in state.group_sources_by_week(nb) if w is not None]
    scope_weeks = st.multiselect(
        "Weeks (empty = whole course)", weeks_present, key=f"audio_weeks_{slug}"
    )
    focus = st.text_input("Focus (optional)", key=f"audio_focus_{slug}")
    max_minutes = st.slider(
        "Max length (minutes)",
        5,
        45,
        value=settings.audio_max_minutes,
        key=f"audio_minutes_{slug}",
    )
    scope = AudioScope(weeks=scope_weeks or None, focus=focus or None)
    run_settings = settings.model_copy(update={"audio_max_minutes": max_minutes})

    if st.button("Estimate cost", key=f"estimate_audio_{slug}"):
        try:
            estimate = estimate_overview_cost(nb, scope, run_settings)
        except IndexNotBuiltError as exc:
            st.error(str(exc))
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.info(
                state.format_overview_estimate(
                    estimate.total_material_tokens,
                    estimate.n_sources,
                    estimate.est_cost_low_usd,
                    estimate.est_cost_high_usd,
                )
            )

    confirm_key = f"confirm_generate_{slug}"
    if st.button("Generate", key=f"generate_audio_{slug}"):
        st.session_state[confirm_key] = True

    if st.session_state.get(confirm_key):
        st.warning(
            f"This calls the {state.provider_label(get_settings())} API and may cost money. "
            "Continue?"
        )
        col1, col2 = st.columns(2)
        if col1.button("Yes, generate", key=f"confirm_generate_yes_{slug}"):
            st.session_state[confirm_key] = False
            _generate_overview_run(nb, scope, run_settings)
        if col2.button("Cancel", key=f"confirm_generate_no_{slug}"):
            st.session_state[confirm_key] = False

    st.markdown("---")
    st.markdown("#### Previous episodes")
    episodes = components.list_episodes(nb)
    if not episodes:
        st.caption("No episodes yet.")
    for episode in episodes:
        with st.expander(episode["title"]):
            st.audio(str(episode["mp3_path"]))
            st.download_button(
                "Download MP3",
                episode["mp3_path"].read_bytes(),
                file_name=episode["mp3_path"].name,
                key=f"dl_ep_mp3_{episode['title']}",
            )
            if episode["transcript_path"] is not None:
                st.download_button(
                    "Download transcript",
                    episode["transcript_path"].read_bytes(),
                    file_name=episode["transcript_path"].name,
                    key=f"dl_ep_txt_{episode['title']}",
                )

    st.markdown("---")
    _render_rerender_section(nb, slug)


# --- Main --------------------------------------------------------------


def main() -> None:
    slug = _render_sidebar()
    if slug is None:
        st.title("NoteCast")
        st.info("Create a notebook in the sidebar to get started.")
        return

    nb = Notebook(slug)
    st.title(display_name(nb, load_course_config(nb)))

    tab_chat, tab_audio = st.tabs(["Chat", "Audio overview"])
    with tab_chat:
        _render_chat_tab(nb, slug)
    with tab_audio:
        _render_audio_tab(nb, slug)


main()
