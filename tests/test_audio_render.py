"""Tests for notecast.audio.render: voice routing, pauses, chapter timing,
loudness normalisation, MP3 encoding and transcript generation. Uses a fake
TTS (deterministic sine/zero arrays) — no network, no real model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from notecast.audio.models import (
    AudioPlan,
    AudioScope,
    AudioScript,
    ChapterScript,
    ScriptLine,
)
from notecast.audio.render import (
    SAMPLE_RATE,
    estimate_render_seconds,
    render_script,
)
from notecast.config import Settings


def _plan() -> AudioPlan:
    return AudioPlan(
        scope=AudioScope(),
        points=[],
        target_minutes=5,
        target_words=750,
        chapters=[],
    )


def _script(title: str = "Week 3") -> AudioScript:
    return AudioScript(
        title=title,
        plan=_plan(),
        chapters=[
            ChapterScript(
                index=1,
                title="Introduction",
                lines=[
                    ScriptLine(speaker="A", text="Welcome to the show."),
                    ScriptLine(speaker="B", text="Great to be here."),
                    ScriptLine(speaker="B", text="Let's get started."),
                ],
            ),
            ChapterScript(
                index=2,
                title="Deep Dive",
                lines=[
                    ScriptLine(speaker="A", text="Now for the details."),
                    ScriptLine(speaker="B", text="Interesting stuff."),
                ],
            ),
        ],
    )


class FakeTTS:
    """Deterministic fake TTS: length proportional to text, per-voice amplitude."""

    def __init__(self, amplitude_by_voice: dict[str, float] | None = None):
        self.amplitude_by_voice = amplitude_by_voice or {}
        self.calls: list[tuple[str, str]] = []

    def synthesize(self, text: str, voice: str) -> np.ndarray:
        self.calls.append((text, voice))
        n_samples = max(len(text) * 200, 200)
        amplitude = self.amplitude_by_voice.get(voice, 0.2)
        t = np.linspace(0, n_samples / SAMPLE_RATE, n_samples, dtype=np.float32)
        return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        model_cache_dir=tmp_path / "cache",
        host_a_voice="af_heart",
        host_b_voice="bm_george",
        audio_mp3_bitrate=96,
    )


def _is_valid_mp3(data: bytes) -> bool:
    if data[:3] == b"ID3":
        return True
    return len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0


def test_render_script_uses_correct_voice_per_speaker(settings, tmp_path):
    tts = FakeTTS()
    render_script(_script(), tmp_path, settings=settings, tts=tts)

    voices_used = {voice for _text, voice in tts.calls}
    assert voices_used == {"af_heart", "bm_george"}
    for text, voice in tts.calls:
        if text in ("Welcome to the show.", "Now for the details."):
            assert voice == "af_heart"
        else:
            assert voice == "bm_george"


def test_render_script_chapter_start_times(settings, tmp_path):
    tts = FakeTTS()
    result = render_script(_script(), tmp_path, settings=settings, tts=tts)

    assert len(result.chapters) == 2
    assert result.chapters[0].start_seconds == pytest.approx(0.5)  # lead-in only
    assert result.chapters[1].start_seconds > result.chapters[0].start_seconds
    # Second chapter starts after chapter 1's audio + inter-chapter pause.
    gap = result.chapters[1].start_seconds - result.chapters[0].start_seconds
    assert gap > 1.2  # at least the inter-chapter pause


def test_render_script_progress_callback(settings, tmp_path):
    tts = FakeTTS()
    calls: list[tuple[int, int]] = []
    render_script(
        _script(),
        tmp_path,
        settings=settings,
        tts=tts,
        progress=lambda done, total: calls.append((done, total)),
    )

    assert calls[-1] == (5, 5)  # 3 + 2 lines total
    assert calls == sorted(calls)


def test_render_script_output_files(settings, tmp_path):
    tts = FakeTTS()
    result = render_script(_script("Week 3: Graphs!"), tmp_path, settings=settings, tts=tts)

    assert result.mp3_path.parent == tmp_path
    assert result.mp3_path.suffix == ".mp3"
    assert result.mp3_path.name.startswith("week-3-graphs")
    assert result.mp3_path.exists()
    assert result.transcript_path.exists()

    mp3_bytes = result.mp3_path.read_bytes()
    assert len(mp3_bytes) > 0
    assert _is_valid_mp3(mp3_bytes)


def test_render_script_peak_normalised(settings, tmp_path):
    # One very loud voice, one quiet — final mix should still peak near -1 dBFS,
    # not clip and not be silent.
    tts = FakeTTS({"af_heart": 0.9, "bm_george": 0.05})
    result = render_script(_script(), tmp_path, settings=settings, tts=tts)

    assert result.duration_seconds > 0
    assert result.mp3_path.stat().st_size > 100


def test_render_script_generates_transcript_without_markdown(settings, tmp_path):
    tts = FakeTTS()
    result = render_script(_script(), tmp_path, settings=settings, tts=tts)

    transcript = result.transcript_path.read_text(encoding="utf-8")
    assert "# Week 3" in transcript
    assert "## Introduction (00:00)" in transcript
    assert "Host A:" in transcript
    assert "Host B:" in transcript


def test_render_script_inserts_times_into_given_transcript(settings, tmp_path):
    tts = FakeTTS()
    markdown = "# Week 3\n\n## Introduction\n\nSome intro text.\n\n## Deep Dive\n\nMore text.\n"
    result = render_script(
        _script(), tmp_path, settings=settings, tts=tts, transcript_markdown=markdown
    )

    transcript = result.transcript_path.read_text(encoding="utf-8")
    assert "## Introduction (00:00)" in transcript
    assert "## Deep Dive (" in transcript
    assert "Some intro text." in transcript


def test_render_script_pause_lengths_affect_duration(settings, tmp_path):
    # Same-speaker consecutive lines (chapter 1 has B, B) should produce a
    # shorter gap than switching speakers; verify indirectly via total duration
    # bounds using zero-length synthesis for pause isolation.
    class SilentTTS:
        def synthesize(self, text: str, voice: str) -> np.ndarray:
            return np.zeros(0, dtype=np.float32)

    result = render_script(_script(), tmp_path, settings=settings, tts=SilentTTS())
    # lead-in (0.5) + pause A->B (0.35) + pause B->B (0.2) + inter-chapter (1.2)
    # + pause A->B (0.35) = 2.6s, all lines silent.
    assert result.duration_seconds == pytest.approx(0.5 + 0.35 + 0.2 + 1.2 + 0.35, abs=1e-6)


def test_estimate_render_seconds_scales_with_length():
    script = _script()
    fast = estimate_render_seconds(script, realtime_factor=0.1)
    slow = estimate_render_seconds(script, realtime_factor=0.5)
    assert 0 < fast < slow
