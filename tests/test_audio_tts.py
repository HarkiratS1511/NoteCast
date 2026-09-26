"""Tests for notecast.audio.tts: model download, voice validation, sentence
splitting and text normalisation. No network access, no real model loaded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from notecast.audio import tts as tts_module
from notecast.audio.tts import (
    InvalidVoiceError,
    KokoroTTS,
    TTSUnavailableError,
    ensure_kokoro_files,
    split_into_chunks,
    text_for_speech,
)
from notecast.config import Settings


def _settings(tmp_path: Path, **overrides) -> Settings:
    defaults = {"model_cache_dir": tmp_path / "cache"}
    defaults.update(overrides)
    return Settings(**defaults)


# --- ensure_kokoro_files -----------------------------------------------------


def test_ensure_kokoro_files_no_download_when_present(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path)
    model_dir = settings.model_cache_dir / "kokoro"
    model_dir.mkdir(parents=True)
    (model_dir / "kokoro-v1.0.onnx").write_bytes(b"model")
    (model_dir / "voices-v1.0.bin").write_bytes(b"voices")

    def fail_download(*args, **kwargs):
        raise AssertionError("should not attempt a download")

    monkeypatch.setattr(tts_module, "_download_file", fail_download)

    model_path, voices_path = ensure_kokoro_files(settings)
    assert model_path == model_dir / "kokoro-v1.0.onnx"
    assert voices_path == model_dir / "voices-v1.0.bin"


def test_ensure_kokoro_files_downloads_missing(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path, tts_model="int8")
    calls: list[str] = []

    def fake_download(url, dest, *, progress, label):
        calls.append(label)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replace, as the real implementation does.
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(b"fake-bytes")
        import os

        os.replace(tmp, dest)
        if progress is not None:
            progress(label, 10, 10)

    monkeypatch.setattr(tts_module, "_download_file", fake_download)

    model_path, voices_path = ensure_kokoro_files(settings)
    assert model_path.name == "kokoro-v1.0.int8.onnx"
    assert model_path.exists()
    assert voices_path.exists()
    assert set(calls) == {"kokoro-v1.0.int8.onnx", "voices-v1.0.bin"}


def test_ensure_kokoro_files_reports_network_failure(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path)

    class FakeURLError(OSError):
        pass

    def fake_urlopen(*args, **kwargs):
        raise FakeURLError("no network")

    monkeypatch.setattr(tts_module, "urlopen", fake_urlopen)

    with pytest.raises(TTSUnavailableError, match="one-time"):
        ensure_kokoro_files(settings)


# --- text_for_speech ----------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected_substring",
    [
        ("Tom & Jerry", "Tom and Jerry"),
        ("50% done", "50 percent done"),
        ("e.g. this one", "for example this one"),
        ("i.e. that one", "that is that one"),
        ("cats vs. dogs", "cats versus dogs"),
    ],
)
def test_text_for_speech_symbol_expansion(raw, expected_substring):
    assert expected_substring in text_for_speech(raw)


def test_text_for_speech_strips_markdown_and_collapses_whitespace():
    raw = (
        "This is **bold** and _italic_ and `code` and a "
        "[link](https://example.com).\n\n  Extra   spaces."
    )
    result = text_for_speech(raw)
    assert "*" not in result
    assert "_" not in result
    assert "`" not in result
    assert "[" not in result and "](" not in result
    assert "link" in result
    assert "  " not in result


# --- split_into_chunks --------------------------------------------------------


def test_split_into_chunks_respects_max_length():
    sentence = "This is a sentence. " * 40  # long text with clear sentence boundaries
    chunks = split_into_chunks(sentence, max_chars=400)
    assert len(chunks) > 1
    assert all(len(chunk) <= 400 for chunk in chunks)


def test_split_into_chunks_splits_overlong_single_sentence():
    huge_sentence = "word " * 200  # no punctuation, one giant "sentence"
    chunks = split_into_chunks(huge_sentence, max_chars=50)
    assert all(len(chunk) <= 50 for chunk in chunks)
    assert len(chunks) > 1


def test_split_into_chunks_empty_text():
    assert split_into_chunks("") == []
    assert split_into_chunks("   ") == []


# --- KokoroTTS -----------------------------------------------------------------


class FakeKokoro:
    """Deterministic fake of the kokoro_onnx.Kokoro engine."""

    def __init__(self, model_path, voices_path):
        self.model_path = model_path
        self.voices_path = voices_path

    def get_voices(self):
        return ["af_heart", "bm_george", "af_bella"]

    def create(self, text, voice, speed=1.0, lang="en-us", **kwargs):
        # Deterministic length proportional to text, silent sine-ish array.
        n_samples = max(len(text) * 100, 100)
        t = np.linspace(0, 1, n_samples, dtype=np.float32)
        audio = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
        return audio, 24000


@pytest.fixture
def fake_kokoro_tts(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path)
    model_dir = settings.model_cache_dir / "kokoro"
    model_dir.mkdir(parents=True)
    (model_dir / "kokoro-v1.0.onnx").write_bytes(b"model")
    (model_dir / "voices-v1.0.bin").write_bytes(b"voices")

    monkeypatch.setattr("kokoro_onnx.Kokoro", FakeKokoro)
    return KokoroTTS(settings)


def test_kokoro_tts_voices(fake_kokoro_tts):
    assert fake_kokoro_tts.voices() == ["af_bella", "af_heart", "bm_george"]


def test_kokoro_tts_synthesize_returns_float32_mono(fake_kokoro_tts):
    audio = fake_kokoro_tts.synthesize("Hello there, this is a test.", "af_heart")
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert audio.size > 0


def test_kokoro_tts_invalid_voice_raises_friendly_error(fake_kokoro_tts):
    with pytest.raises(InvalidVoiceError, match="not-a-real-voice"):
        fake_kokoro_tts.synthesize("Hello", "not-a-real-voice")


def test_kokoro_tts_synthesize_splits_long_text(fake_kokoro_tts, monkeypatch):
    calls: list[str] = []
    original_create = FakeKokoro.create

    def counting_create(self, text, voice, **kwargs):
        calls.append(text)
        return original_create(self, text, voice, **kwargs)

    monkeypatch.setattr(FakeKokoro, "create", counting_create)

    long_text = "This is a sentence. " * 40
    fake_kokoro_tts.synthesize(long_text, "af_heart")
    assert len(calls) > 1
    assert all(len(chunk) <= 400 for chunk in calls)
