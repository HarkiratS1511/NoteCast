"""Render an AudioScript to an MP3 (two-voice TTS, stitched with pauses,
loudness-normalised) plus a chapter-marked transcript.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import lameenc
import numpy as np

from notecast.audio.models import AudioScript, RenderedChapter, RenderResult
from notecast.audio.tts import KokoroTTS

if TYPE_CHECKING:
    from notecast.config import Settings

SAMPLE_RATE = 24000

LEAD_IN_SECONDS = 0.5
PAUSE_SAME_SPEAKER_SECONDS = 0.2
PAUSE_DIFFERENT_SPEAKER_SECONDS = 0.35
PAUSE_CHAPTER_SECONDS = 1.2

PEAK_TARGET_DBFS = -1.0
MAX_LINE_GAIN_DB = 6.0

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(title: str) -> str:
    slug = _SLUG_RE.sub("-", title.lower()).strip("-")
    return slug or "notecast-episode"


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(round(seconds * SAMPLE_RATE)), dtype=np.float32)


def _db_to_gain(db: float) -> float:
    return float(10 ** (db / 20))


def _format_mmss(seconds: float) -> str:
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def estimate_render_seconds(script: AudioScript, realtime_factor: float = 0.35) -> float:
    """Estimate wall-clock render time from the script's spoken length."""
    audio_seconds = script.est_minutes() * 60
    return audio_seconds * realtime_factor


def _insert_chapter_times(markdown: str, rendered_chapters: list[RenderedChapter]) -> str:
    lines = markdown.splitlines()
    chapters = iter(rendered_chapters)
    out: list[str] = []
    for line in lines:
        if line.startswith("## "):
            chapter = next(chapters, None)
            if chapter is not None:
                out.append(f"{line} ({_format_mmss(chapter.start_seconds)})")
                continue
        out.append(line)
    result = "\n".join(out)
    if markdown.endswith("\n"):
        result += "\n"
    return result


_SPEAKER_LABELS = {"A": "Host A", "B": "Host B"}


def _generate_transcript(script: AudioScript, rendered_chapters: list[RenderedChapter]) -> str:
    parts = [f"# {script.title}", ""]
    for chapter, rendered in zip(script.chapters, rendered_chapters, strict=True):
        parts.append(f"## {chapter.title} ({_format_mmss(rendered.start_seconds)})")
        parts.append("")
        for line in chapter.lines:
            label = _SPEAKER_LABELS.get(line.speaker, line.speaker)
            parts.append(f"**{label}:** {line.text}")
            parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _encode_mp3(pcm_int16: np.ndarray, *, bitrate: int) -> bytes:
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(SAMPLE_RATE)
    encoder.set_channels(1)
    encoder.set_quality(2)
    data = encoder.encode(pcm_int16.tobytes())
    data += encoder.flush()
    return data


def render_script(
    script: AudioScript,
    out_dir: Path,
    *,
    settings: Settings,
    tts: KokoroTTS | None = None,
    transcript_markdown: str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> RenderResult:
    """Synthesize `script` into an MP3 + transcript in `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if tts is None:
        tts = KokoroTTS(settings)

    voice_for_speaker = {"A": settings.host_a_voice, "B": settings.host_b_voice}

    total_lines = sum(len(chapter.lines) for chapter in script.chapters)
    done_lines = 0

    # Pass 1: synthesize every line, track per-line RMS for loudness levelling.
    line_audios: list[list[np.ndarray]] = []
    line_rms: list[list[float]] = []
    for chapter in script.chapters:
        chapter_audios: list[np.ndarray] = []
        chapter_rms: list[float] = []
        for line in chapter.lines:
            audio = tts.synthesize(line.text, voice_for_speaker[line.speaker])
            chapter_audios.append(audio)
            chapter_rms.append(float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0)
            done_lines += 1
            if progress is not None:
                progress(done_lines, total_lines)
        line_audios.append(chapter_audios)
        line_rms.append(chapter_rms)

    nonzero_rms = [r for chapter_rms in line_rms for r in chapter_rms if r > 0]
    target_rms = float(np.median(nonzero_rms)) if nonzero_rms else 0.0
    min_gain = _db_to_gain(-MAX_LINE_GAIN_DB)
    max_gain = _db_to_gain(MAX_LINE_GAIN_DB)

    # Pass 2: apply per-line RMS levelling and stitch with pauses.
    mix_pieces: list[np.ndarray] = [_silence(LEAD_IN_SECONDS)]
    current_time = LEAD_IN_SECONDS
    rendered_chapters: list[RenderedChapter] = []

    for chapter_idx, chapter in enumerate(script.chapters):
        if chapter_idx > 0:
            pause = _silence(PAUSE_CHAPTER_SECONDS)
            mix_pieces.append(pause)
            current_time += PAUSE_CHAPTER_SECONDS

        rendered_chapters.append(
            RenderedChapter(index=chapter.index, title=chapter.title, start_seconds=current_time)
        )

        prev_speaker: str | None = None
        for line_idx, line in enumerate(chapter.lines):
            audio = line_audios[chapter_idx][line_idx]
            rms = line_rms[chapter_idx][line_idx]
            if rms > 0 and target_rms > 0:
                gain = min(max(target_rms / rms, min_gain), max_gain)
                audio = audio * gain

            if prev_speaker is not None:
                pause_seconds = (
                    PAUSE_SAME_SPEAKER_SECONDS
                    if line.speaker == prev_speaker
                    else PAUSE_DIFFERENT_SPEAKER_SECONDS
                )
                mix_pieces.append(_silence(pause_seconds))
                current_time += pause_seconds

            mix_pieces.append(audio.astype(np.float32, copy=False))
            current_time += audio.size / SAMPLE_RATE
            prev_speaker = line.speaker

    mix = np.concatenate(mix_pieces) if mix_pieces else np.zeros(0, dtype=np.float32)

    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0:
        target_peak = _db_to_gain(PEAK_TARGET_DBFS)
        mix = mix * (target_peak / peak)

    pcm = np.clip(mix * 32767.0, -32768, 32767).astype(np.int16)
    duration_seconds = mix.size / SAMPLE_RATE

    mp3_bytes = _encode_mp3(pcm, bitrate=settings.audio_mp3_bitrate)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    stem = f"{_slugify(script.title)}-{timestamp}"
    mp3_path = out_dir / f"{stem}.mp3"
    transcript_path = out_dir / f"{stem}.md"

    mp3_path.write_bytes(mp3_bytes)

    if transcript_markdown is not None:
        transcript = _insert_chapter_times(transcript_markdown, rendered_chapters)
    else:
        transcript = _generate_transcript(script, rendered_chapters)
    transcript_path.write_text(transcript, encoding="utf-8")

    return RenderResult(
        mp3_path=mp3_path,
        transcript_path=transcript_path,
        duration_seconds=duration_seconds,
        chapters=rendered_chapters,
    )
