"""Local text-to-speech via Kokoro (kokoro-onnx). Handles one-time model
download, voice validation, text normalisation and sentence-bounded synthesis.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np

if TYPE_CHECKING:
    from notecast.config import Settings

MODEL_RELEASE_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
)
VOICES_FILENAME = "voices-v1.0.bin"

# Approx max characters per synthesis call (kokoro has a phoneme length limit).
MAX_CHUNK_CHARS = 400

# Common English voices bundled with the Kokoro v1.0 voice pack.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_SYMBOL_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\be\.g\.", re.IGNORECASE), "for example"),
    (re.compile(r"\bi\.e\.", re.IGNORECASE), "that is"),
    (re.compile(r"\bvs\.", re.IGNORECASE), "versus"),
    (re.compile(r"&"), "and"),
    (re.compile(r"%"), " percent"),
]

# Leftover markdown markers we strip for speech (bold/italic/code/headers/links).
_MARKDOWN_RE = re.compile(r"[*_`#]+")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_WHITESPACE_RE = re.compile(r"\s+")


class TTSUnavailableError(RuntimeError):
    """Raised when the Kokoro model files can't be downloaded or loaded."""


class InvalidVoiceError(ValueError):
    """Raised when a requested voice name isn't a valid Kokoro voice."""


def _model_filename(tts_model: str) -> str:
    return "kokoro-v1.0.int8.onnx" if tts_model == "int8" else "kokoro-v1.0.onnx"


def _download_file(
    url: str,
    dest: Path,
    *,
    progress: Callable[[str, int, int], None] | None,
    label: str,
) -> None:
    try:
        with urlopen(url) as response:  # noqa: S310 - fixed, trusted GitHub release URL
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            fd, tmp_path_str = tempfile.mkstemp(
                dir=str(dest.parent), prefix=f".{dest.name}.", suffix=".part"
            )
            tmp_path = Path(tmp_path_str)
            try:
                with os.fdopen(fd, "wb") as tmp_file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        tmp_file.write(chunk)
                        downloaded += len(chunk)
                        if progress is not None:
                            progress(label, downloaded, total)
                os.replace(tmp_path, dest)
            finally:
                tmp_path.unlink(missing_ok=True)
    except (URLError, OSError) as exc:
        raise TTSUnavailableError(
            "Couldn't download the Kokoro text-to-speech model files "
            f"({label}). This is a one-time download of roughly 350 MB total "
            "(model + voices). Check your internet connection and try again. "
            f"Original error: {exc}"
        ) from exc


def ensure_kokoro_files(
    settings: Settings,
    *,
    progress: Callable[[str, int, int], None] | None = None,
) -> tuple[Path, Path]:
    """Ensure the Kokoro model + voices files exist locally, downloading any
    that are missing. Returns (model_path, voices_path).
    """
    model_dir = settings.model_cache_dir / "kokoro"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_filename = _model_filename(settings.tts_model)
    model_path = model_dir / model_filename
    voices_path = model_dir / VOICES_FILENAME

    if not model_path.exists():
        _download_file(
            MODEL_RELEASE_URL + model_filename,
            model_path,
            progress=progress,
            label=model_filename,
        )
    if not voices_path.exists():
        _download_file(
            MODEL_RELEASE_URL + VOICES_FILENAME,
            voices_path,
            progress=progress,
            label=VOICES_FILENAME,
        )

    return model_path, voices_path


def text_for_speech(text: str) -> str:
    """Light normalisation of text before it's sent to the TTS engine."""
    result = text
    for pattern, replacement in _SYMBOL_REPLACEMENTS:
        result = pattern.sub(replacement, result)
    result = _MARKDOWN_LINK_RE.sub(r"\1", result)
    result = _MARKDOWN_RE.sub("", result)
    result = _WHITESPACE_RE.sub(" ", result).strip()
    return result


def split_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text on sentence boundaries into pieces no longer than
    `max_chars`. A single sentence longer than `max_chars` is further split
    on word boundaries.
    """
    text = text.strip()
    if not text:
        return []

    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            chunks.append(current.strip())
            current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue
        # Current sentence doesn't fit with what's buffered.
        flush()
        if len(sentence) <= max_chars:
            current = sentence
            continue
        # Sentence itself is too long: split on words.
        words = sentence.split(" ")
        piece = ""
        for word in words:
            candidate_piece = f"{piece} {word}".strip() if piece else word
            if len(candidate_piece) <= max_chars:
                piece = candidate_piece
            else:
                if piece:
                    chunks.append(piece)
                piece = word
        current = piece

    flush()
    return chunks


class KokoroTTS:
    """Lazily-loaded wrapper around the Kokoro ONNX TTS engine."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._kokoro = None
        self._voices: list[str] | None = None

    def _load(self):  # noqa: ANN202 - kokoro_onnx.Kokoro, imported lazily
        if self._kokoro is None:
            from kokoro_onnx import Kokoro

            model_path, voices_path = ensure_kokoro_files(self._settings)
            try:
                self._kokoro = Kokoro(str(model_path), str(voices_path))
            except Exception as exc:  # pragma: no cover - defensive
                raise TTSUnavailableError(
                    f"Couldn't load the Kokoro TTS model from {model_path}: {exc}"
                ) from exc
        return self._kokoro

    def voices(self) -> list[str]:
        if self._voices is None:
            kokoro = self._load()
            self._voices = sorted(kokoro.get_voices())
        return self._voices

    def _validate_voice(self, voice: str) -> None:
        valid = self.voices()
        if voice not in valid:
            raise InvalidVoiceError(
                f"'{voice}' isn't a valid Kokoro voice. Valid English voices: "
                f"{', '.join(v for v in valid if v.startswith(('a', 'b')))}"
            )

    def synthesize(self, text: str, voice: str) -> np.ndarray:
        """Synthesize `text` in `voice`, returning float32 mono audio at 24 kHz."""
        kokoro = self._load()
        self._validate_voice(voice)

        normalised = text_for_speech(text)
        chunks = split_into_chunks(normalised)
        if not chunks:
            return np.zeros(0, dtype=np.float32)

        gap = np.zeros(int(0.05 * 24000), dtype=np.float32)
        pieces: list[np.ndarray] = []
        for i, chunk in enumerate(chunks):
            audio, _sample_rate = kokoro.create(
                chunk,
                voice=voice,
                speed=self._settings.tts_speed,
                lang=self._settings.tts_lang,
            )
            pieces.append(np.asarray(audio, dtype=np.float32))
            if i < len(chunks) - 1:
                pieces.append(gap)

        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
