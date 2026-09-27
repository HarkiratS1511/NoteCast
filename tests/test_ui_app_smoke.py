"""Smoke test for notecast/ui/app.py using Streamlit's AppTest harness.

No network, no real API, no Claude client -- the notebook is a tmp dir with
a tiny synthetic .md source, ingested with the hashing embedding backend,
`ChatSession` is swapped for a fake via `notecast.ui.state.chat_session_factory`.

Important AppTest quirk: `AppTest.from_file` re-executes the whole script
(including its top-level `import`/`from ... import ...` statements) on every
`.run()`, exactly like a real Streamlit rerun. That means monkeypatching a
name that `notecast/ui/app.py` pulled in with `from X import name` (e.g.
`notecast.ui.app.generate_overview`) has no effect -- the next rerun just
re-imports the original. Instead we patch the *source* attribute
(`notecast.audio.service.generate_overview`), which the fresh `from ...
import` picks up. `state.chat_session_factory` doesn't have this problem
because `app.py` calls it as `state.chat_session_factory(...)`, a lookup on
the (singleton) `notecast.ui.state` module itself, not a copied-in name.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import notecast.audio.service as audio_service
from notecast.audio.models import (
    AudioPlan,
    AudioScript,
    ChapterPlan,
    ChapterScript,
    CoverageReport,
)
from notecast.audio.service import OverviewFailed, OverviewResult
from notecast.audio.tts import TTSUnavailableError
from notecast.chat.client import MissingApiKeyError
from notecast.chat.models import AnswerSegment, ChatAnswer, Citation, Usage
from notecast.config import get_settings
from notecast.index.service import index_notebook
from notecast.notebook import Notebook
from notecast.ui import state

APP_PATH = str(Path(__file__).resolve().parent.parent / "notecast" / "ui" / "app.py")


class FakeChatSession:
    """A stand-in for `ChatSession` that never touches the network.

    Class-level knobs (`raise_error`, `not_in_sources`) are read fresh by
    each `ask()` call so tests can steer behaviour by monkeypatching them on
    the class *before* the widget interaction that triggers `ask()`; a
    class-level `calls` log lets tests assert exactly how many times (and
    with what mode/filters) `ask()` was actually invoked.
    """

    raise_error: Exception | None = None
    not_in_sources: bool = False
    calls: list[tuple[str, str, object]] = []

    def __init__(self, retriever, **kwargs) -> None:  # noqa: ANN001
        self.retriever = retriever
        self.kwargs = kwargs
        self.reset_calls = 0
        self.settings = get_settings()

    def ask(self, question: str, *, mode: str = "sources", filters=None, k=None) -> ChatAnswer:  # noqa: ANN001
        type(self).calls.append((question, mode, filters))
        if type(self).raise_error is not None:
            raise type(self).raise_error
        if type(self).not_in_sources:
            return ChatAnswer(
                question=question,
                standalone_query=question,
                mode=mode,
                segments=[
                    AnswerSegment(
                        text="The course material doesn't cover this.", citation_numbers=[]
                    )
                ],
                citations=[],
                not_in_sources=True,
                usage=Usage(),
                model="fake-model",
            )
        citation = Citation(
            n=1,
            kind="course",
            source_path="lecture.md",
            location_label="p. 1",
            cited_text="Photosynthesis converts light into chemical energy.",
            header="Week 1",
        )
        return ChatAnswer(
            question=question,
            standalone_query=question,
            mode=mode,
            segments=[
                AnswerSegment(
                    text="Photosynthesis turns light into chemical energy.",
                    citation_numbers=[1],
                )
            ],
            citations=[citation],
            not_in_sources=False,
            usage=Usage(input_tokens=100, output_tokens=20, est_cost_usd=0.001),
            model="fake-model",
        )

    def reset(self) -> None:
        self.reset_calls += 1

    def estimate_deep_cost(self, filters=None):  # noqa: ANN001
        return 1000, 0.01


@pytest.fixture
def notebook_with_index(tmp_notebooks_root: Path, monkeypatch: pytest.MonkeyPatch) -> Notebook:
    monkeypatch.setenv("NOTECAST_EMBEDDING_BACKEND", "hashing")
    get_settings.cache_clear()

    nb = Notebook.create("bio101")
    week_dir = nb.sources_dir / "week-01"
    week_dir.mkdir(parents=True, exist_ok=True)
    (week_dir / "lecture.md").write_text(
        "# Photosynthesis\n\nPlants convert light energy into chemical energy "
        "through photosynthesis, using chlorophyll in their leaves.\n",
        encoding="utf-8",
    )
    index_notebook(nb)
    yield nb
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _reset_fake_chat_session() -> None:
    """Every test starts with a clean `FakeChatSession` (no error/flag
    left over from a previous test, no stale call log).
    """
    FakeChatSession.raise_error = None
    FakeChatSession.not_in_sources = False
    FakeChatSession.calls = []
    yield
    FakeChatSession.raise_error = None
    FakeChatSession.not_in_sources = False
    FakeChatSession.calls = []


def test_app_loads_notebook_and_tabs(notebook_with_index: Notebook) -> None:
    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    select_boxes = at.sidebar.selectbox
    assert any("bio101" in str(sb.value) for sb in select_boxes)
    assert len(at.tabs) == 2


def test_dark_mode_toggle_swaps_theme_css(notebook_with_index: Notebook) -> None:
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert not at.exception

    style_blocks = "\n".join(md.value for md in at.markdown)
    assert "midnight mode" not in style_blocks.lower()
    assert "#26201a" not in style_blocks  # dark override not injected by default

    at.sidebar.toggle(key="notecast_dark_mode").set_value(True).run()
    assert not at.exception
    assert at.session_state["notecast_dark_mode"] is True

    style_blocks = "\n".join(md.value for md in at.markdown)
    assert "#26201a" in style_blocks  # dark "midnight notebook" --paper-bg override

    at.sidebar.toggle(key="notecast_dark_mode").set_value(False).run()
    assert not at.exception
    style_blocks = "\n".join(md.value for md in at.markdown)
    assert "#26201a" not in style_blocks


def test_sidebar_shows_active_provider_and_model(notebook_with_index: Notebook) -> None:
    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    captions = "\n".join(c.value for c in at.sidebar.caption)
    assert "LLM: Claude" in captions
    assert get_settings().chat_model in captions


def test_chat_tab_renders_answer_and_citation(
    notebook_with_index: Notebook, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(state, "chat_session_factory", FakeChatSession)

    at = AppTest.from_file(APP_PATH)
    at.run()
    assert not at.exception

    at.chat_input[0].set_value("What is photosynthesis?").run()
    assert not at.exception

    full_text = "\n".join(md.value for md in at.markdown)
    assert "Photosynthesis turns light into chemical energy" in full_text
    assert "[1]" in full_text
    assert "lecture.md" in full_text


def test_not_in_sources_shows_info_banner(
    notebook_with_index: Notebook, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(state, "chat_session_factory", FakeChatSession)
    FakeChatSession.not_in_sources = True

    at = AppTest.from_file(APP_PATH)
    at.run()
    at.chat_input[0].set_value("Something not covered").run()
    assert not at.exception

    info_text = "\n".join(i.value for i in at.info)
    assert "Not in your course material" in info_text
    assert "Open mode" in info_text


def test_missing_api_key_error_renders_st_error(
    notebook_with_index: Notebook, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(state, "chat_session_factory", FakeChatSession)
    FakeChatSession.raise_error = MissingApiKeyError()

    at = AppTest.from_file(APP_PATH)
    at.run()
    at.chat_input[0].set_value("Hello?").run()
    assert not at.exception
    assert any("ANTHROPIC_API_KEY" in e.value for e in at.error)


def test_deep_mode_whole_course_requires_confirmation_then_sends_once(
    notebook_with_index: Notebook, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(state, "chat_session_factory", FakeChatSession)

    at = AppTest.from_file(APP_PATH)
    at.run()
    at.radio(key="mode_bio101").set_value("Deep — whole scope").run()

    at.chat_input[0].set_value("Summarise everything").run()
    assert not at.exception
    # No week selected (whole-course scope): must NOT have sent yet.
    assert FakeChatSession.calls == []
    assert any("Deep mode will send" in w.value for w in at.warning)

    at.button(key="confirm_deep_bio101").click().run()
    assert not at.exception
    assert len(FakeChatSession.calls) == 1
    assert FakeChatSession.calls[0][1] == "deep"

    # A plain rerun afterwards must not resend the same question.
    at.run()
    assert len(FakeChatSession.calls) == 1


def test_deep_mode_with_week_requires_confirmation_then_sends_once(
    notebook_with_index: Notebook, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(state, "chat_session_factory", FakeChatSession)

    at = AppTest.from_file(APP_PATH)
    at.run()
    at.radio(key="mode_bio101").set_value("Deep — whole scope").run()
    at.multiselect(key="chat_weeks_bio101").set_value([1]).run()

    at.chat_input[0].set_value("Summarise week 1").run()
    assert not at.exception
    assert FakeChatSession.calls == []

    at.button(key="confirm_deep_bio101").click().run()
    assert not at.exception
    assert len(FakeChatSession.calls) == 1
    assert FakeChatSession.calls[0][1] == "deep"

    at.run()
    assert len(FakeChatSession.calls) == 1


def _fake_overview_result(nb: Notebook, scope):  # noqa: ANN001
    plan = AudioPlan(
        scope=scope,
        points=[],
        target_minutes=5.0,
        target_words=750,
        chapters=[ChapterPlan(index=1, title="Intro", point_ids=[], target_words=750)],
    )
    script = AudioScript(
        title="Week 1 overview",
        plan=plan,
        chapters=[ChapterScript(index=1, title="Intro", lines=[])],
        coverage=CoverageReport(),
        est_cost_usd=0.02,
    )
    return OverviewResult(
        script=script,
        render=None,
        script_json_path=nb.audio_dir / "bio101-fake.script.json",
        est_cost_usd=0.02,
        timings={},
    )


def test_audio_generate_confirm_flow_calls_once_and_not_on_plain_rerun(
    notebook_with_index: Notebook, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def _fake_generate_overview(nb, scope, *, settings=None, progress=None, **kwargs):  # noqa: ANN001
        calls.append(1)
        if progress:
            progress("done", 1.0)
        return _fake_overview_result(nb, scope)

    monkeypatch.setattr(audio_service, "generate_overview", _fake_generate_overview)

    at = AppTest.from_file(APP_PATH)
    at.run()

    at.button(key="generate_audio_bio101").click().run()
    assert not at.exception
    assert calls == []  # confirmation gate not yet passed
    assert any("may cost money" in w.value for w in at.warning)

    at.button(key="confirm_generate_yes_bio101").click().run()
    assert not at.exception
    assert calls == [1]
    assert any("Generated" in s.value for s in at.success)

    # A plain rerun must not call generate_overview again.
    at.run()
    assert calls == [1]


def test_overview_failed_renders_st_error(
    notebook_with_index: Notebook, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise_overview_failed(nb, scope, *, settings=None, progress=None, **kwargs):  # noqa: ANN001
        raise OverviewFailed("scripting", 0.05, "the model refused")

    monkeypatch.setattr(audio_service, "generate_overview", _raise_overview_failed)

    at = AppTest.from_file(APP_PATH)
    at.run()
    at.button(key="generate_audio_bio101").click().run()
    at.button(key="confirm_generate_yes_bio101").click().run()

    assert not at.exception
    error_text = "\n".join(e.value for e in at.error)
    assert "scripting" in error_text
    assert "0.05" in error_text


def test_tts_unavailable_error_renders_st_error(
    notebook_with_index: Notebook, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise_tts_unavailable(nb, scope, *, settings=None, progress=None, **kwargs):  # noqa: ANN001
        raise TTSUnavailableError("could not download the Kokoro model files")

    monkeypatch.setattr(audio_service, "generate_overview", _raise_tts_unavailable)

    at = AppTest.from_file(APP_PATH)
    at.run()
    at.button(key="generate_audio_bio101").click().run()
    at.button(key="confirm_generate_yes_bio101").click().run()

    assert not at.exception
    error_text = "\n".join(e.value for e in at.error)
    assert "rendering speech" in error_text
    assert "re-render" in error_text.lower()


def test_upload_save_error_renders_sidebar_error(
    notebook_with_index: Notebook, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise_os_error(nb, data, filename, week):  # noqa: ANN001
        raise OSError("disk full")

    monkeypatch.setattr(state, "save_uploaded_file", _raise_os_error)

    at = AppTest.from_file(APP_PATH)
    at.run()

    at.sidebar.file_uploader(key="uploader_bio101").set_value(
        ("notes.pdf", b"%PDF-1.4 fake", "application/pdf")
    ).run()
    at.sidebar.button(key="save_uploads_bio101").click().run()

    assert not at.exception
    assert any("Could not save uploaded files" in e.value for e in at.sidebar.error)
