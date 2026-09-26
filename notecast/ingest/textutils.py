"""Small text helpers shared by the parsers and the chunker."""

from __future__ import annotations

import re

# Typical lecture speaking rate, used to estimate "≈ N min" positions in
# transcripts that have no timestamps.
SPOKEN_WORDS_PER_MINUTE = 150

_COPYRIGHT_NOTICE = re.compile(
    r"This material has been reproduced and communicated to you by or on behalf of"
    r".*?Do not remove this notice\.?",
    re.IGNORECASE | re.DOTALL,
)


def estimate_tokens(text: str) -> int:
    """Rough token count (≈ 4 characters per token). Good enough for sizing chunks."""
    return max(1, len(text) // 4) if text else 0


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs, normalise newlines, and trim each line.

    Keeps paragraph breaks (at most one blank line in a row).
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    out = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def strip_copyright_notice(text: str) -> str:
    """Remove the standard Australian university s113P copyright notice, which
    appears at the start of lecture recordings and would otherwise pollute search.
    """
    return _COPYRIGHT_NOTICE.sub("", text).strip()
