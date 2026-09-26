"""PDF parser: turns a PDF (regular document or PowerPoint-exported slide
deck) into a ParsedDocument, one Section per non-empty page.

Handles the two shapes we actually see in course material:
- A "normal" document PDF -> Section.location.page is set.
- A landscape slide deck (detected by page aspect ratio) -> location.slide
  is set instead, and the label becomes "slide N".

Repeated headers/footers (the same text near the top/bottom of most pages,
e.g. a university boilerplate footer) are stripped, as are bare page-number
lines, using each line's on-page position so numbered *body* content
(e.g. "Example 3", "Slide 12 title") is never mistaken for boilerplate just
because it shares a shape with other pages once digits are collapsed.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from pathlib import Path

import pymupdf

from notecast.ingest.textutils import normalize_whitespace
from notecast.models import Location, ParsedDocument, Section, SourceType

from . import register

# Fraction of pages that must be landscape for the whole PDF to be treated
# as a slide deck.
_SLIDE_DECK_LANDSCAPE_RATIO = 0.8

# A line recurring on more than this fraction of pages is boilerplate.
_BOILERPLATE_REPEAT_RATIO = 0.5

# Minimum page count before we bother with header/footer stripping (a
# footer that appears on every page of a 2-page doc isn't "repeated"
# boilerplate in any useful sense).
_MIN_PAGES_FOR_BOILERPLATE = 4

# A line is "in the header/footer zone" when its vertical center sits in
# the top or bottom this-much fraction of the page height. A zone line is
# only eligible for digit-collapsed (position-blind) recurrence matching if
# it is *also* small (<= the page's median line font size) and not the
# page's largest-font span -- a big numbered slide title ("Example 3") can
# sit in the zone too, but must not be treated like a small footer/header
# just because of where it is. Everything else must recur verbatim to
# count as boilerplate.
_HEADER_FOOTER_ZONE_FRACTION = 0.12

_MAX_TITLE_LEN = 120

_BULLET_CHARS = "•▪◦●"
_BULLET_LINE_RE = re.compile(r"^[•▪◦●–]\s*")
# A bullet glyph glued directly onto preceding text with no whitespace
# (pymupdf sometimes concatenates adjacent table cells / columns this way).
_GLUED_BULLET_RE = re.compile(r"(?<=\S)([•▪◦●])")

# A line that is *only* a page number, optionally with "Page", "of", "/",
# separators, e.g. "12", "Page 12", "12 / 60", "12 of 60".
_PAGE_NUMBER_LINE_RE = re.compile(
    r"^(page\s*)?\d+\s*((/|of)\s*\d+)?$",
    re.IGNORECASE,
)

_WS_RE = re.compile(r"\s+")
_LEADING_NUM_RE = re.compile(r"^\d+\s+")
_TRAILING_NUM_RE = re.compile(r"\s+\d+$")
_HAS_LETTER_RE = re.compile(r"[A-Za-z]")
# A line ending in a hyphenated word-wrap: an alphanumeric immediately
# followed by "-" at the very end of the line.
_HYPHEN_WRAP_END_RE = re.compile(r"[A-Za-z0-9]-$")

_GENERIC_TITLE_PREFIXES = (
    "microsoft word - ",
    "microsoft powerpoint - ",
)
_GENERIC_TITLES = {
    "powerpoint presentation",
    "presentation1",
    "document1",
}


def _normalize_ws(line: str) -> str:
    """Collapse all whitespace runs (including non-breaking-ish gaps) to a
    single space and strip the ends.
    """
    return _WS_RE.sub(" ", line).strip()


def _is_page_number_line(line: str) -> bool:
    return bool(_PAGE_NUMBER_LINE_RE.match(line.strip()))


def _boilerplate_key(line: str, digit_collapse_eligible: bool) -> str | None:
    """The comparison key used to detect recurring boilerplate, or None if
    `line` isn't eligible to be counted as boilerplate at all.

    Lines eligible for digit collapsing (small text in the header/footer
    zone, not the page's largest span -- see `_digit_collapse_eligible`)
    get a page-number stripped off either edge and remaining digits
    collapsed to '#', so "9  ANU SCHOOL OF COMPUTING | DOCUMENT ANALYSIS"
    and "ANU SCHOOL OF COMPUTING | DOCUMENT ANALYSIS" (missing its page
    number) key the same. Every other line -- including a large, numbered
    slide title sitting in that same zone -- must match verbatim (after
    whitespace normalisation only) to be considered boilerplate, so
    numbered content (body or title) is never treated as recurring just
    because digits look alike once collapsed.
    """
    normalized = _normalize_ws(line)
    if not normalized:
        return None
    if digit_collapse_eligible:
        normalized = _TRAILING_NUM_RE.sub("", normalized)
        normalized = _LEADING_NUM_RE.sub("", normalized).strip()
        if not normalized or not _HAS_LETTER_RE.search(normalized):
            return None
        return re.sub(r"\d+", "#", normalized)
    if not _HAS_LETTER_RE.search(normalized):
        return None
    return normalized


def _in_header_footer_zone(y0: float, y1: float, page_height: float) -> bool:
    if page_height <= 0:
        return False
    center = (y0 + y1) / 2
    return (
        center <= _HEADER_FOOTER_ZONE_FRACTION * page_height
        or center >= (1 - _HEADER_FOOTER_ZONE_FRACTION) * page_height
    )


def _digit_collapse_eligible(
    y0: float, y1: float, page_height: float, font_size: float, median_size: float, max_size: float
) -> bool:
    """Whether a line may be matched as boilerplate by collapsing its
    digits: it must sit in the header/footer zone AND be a small line (at
    or below the page's median line font size) AND not be the page's
    single largest-font span. A numbered slide title like "Example 3" can
    be large and sit near the top of the page, but must fail this check so
    it only ever matches (and gets kept) by verbatim, digit-sensitive
    comparison.
    """
    if not _in_header_footer_zone(y0, y1, page_height):
        return False
    return font_size <= median_size and font_size < max_size


def _extract_dict_lines(page: pymupdf.Page) -> list[tuple[str, float, float, float, float]]:
    """Every visual text line on the page as (text, y0, y1, x0, font_size),
    in reading order (top-to-bottom, then left-to-right). `font_size` is the
    largest span size within that line.
    """
    try:
        page_dict = page.get_text("dict")
    except Exception:
        return []
    lines: list[tuple[str, float, float, float, float]] = []
    for block in page_dict.get("blocks", []):
        if block.get("type", 0) != 0:
            continue  # image block, no text
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(span.get("text", "") for span in spans)
            if not text.strip():
                continue
            sizes = [float(s.get("size", 0.0)) for s in spans if s.get("text", "").strip()]
            font_size = max(sizes) if sizes else 0.0
            bbox = line.get("bbox", (0.0, 0.0, 0.0, 0.0))
            lines.append((text, bbox[1], bbox[3], bbox[0], font_size))
    lines.sort(key=lambda item: (round(item[1]), item[3]))
    return lines


def _merge_hyphenated_wraps(
    lines: list[tuple[str, float, float, float, float]],
) -> list[tuple[str, float, float, float, float]]:
    """Join a line ending "...30-" with a following line starting "minute
    ..." into "...30-minute ..." -- a word wrapped across two visual lines.
    """
    merged: list[list] = []
    for text, y0, y1, x0, font_size in lines:
        if merged:
            prev_text = merged[-1][0]
            prev_stripped = prev_text.rstrip()
            continuation = text.lstrip()
            if _HYPHEN_WRAP_END_RE.search(prev_stripped) and continuation[:1].islower():
                merged[-1][0] = prev_stripped + continuation
                merged[-1][2] = y1
                continue
        merged.append([text, y0, y1, x0, font_size])
    return [(t, y0, y1, x0, fs) for t, y0, y1, x0, fs in merged]


def _split_glued_bullets(text: str) -> list[str]:
    """Split a line at any bullet glyph glued onto preceding text (no
    whitespace before it), turning e.g. "Relevant Not relevant• Example:"
    into ["Relevant Not relevant", "• Example:"].
    """
    parts = _GLUED_BULLET_RE.split(text)
    if len(parts) == 1:
        return [text]
    out = [parts[0]]
    for i in range(1, len(parts), 2):
        bullet = parts[i]
        rest = parts[i + 1] if i + 1 < len(parts) else ""
        out.append(bullet + rest)
    return out


def _convert_bullets(text: str) -> str:
    """Convert leading bullet glyphs on each line to a plain "- " marker."""
    out_lines = []
    for line in text.split("\n"):
        stripped = line.lstrip()
        if stripped and (stripped[0] in _BULLET_CHARS or stripped[0] == "–"):
            stripped = _BULLET_LINE_RE.sub("", stripped)
            out_lines.append("- " + stripped)
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def _font_size_ranked_texts(page: pymupdf.Page) -> list[str]:
    """Text grouped by font size, largest first: each entry is the text of
    all spans sharing one font size, joined by spaces.
    """
    try:
        page_dict = page.get_text("dict")
    except Exception:
        return []
    spans_by_size: dict[float, list[str]] = {}
    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                size = round(float(span.get("size", 0.0)), 1)
                spans_by_size.setdefault(size, []).append(text)
    return [" ".join(spans_by_size[size]).strip() for size in sorted(spans_by_size, reverse=True)]


def _clean_title_candidate(candidate: str) -> str | None:
    candidate = normalize_whitespace(candidate).replace("\n", " ").strip()
    if not candidate or len(candidate) > _MAX_TITLE_LEN:
        return None
    return candidate


class PdfParser:
    """Parses PDF files (documents and slide decks) into ParsedDocuments."""

    suffixes = (".pdf",)

    def parse(self, path: Path, source_path: str) -> ParsedDocument:
        try:
            doc = pymupdf.open(path)
        except Exception as exc:  # pragma: no cover - pymupdf raises varied types
            raise ValueError(f"Could not open PDF {source_path!r}: {exc}") from exc

        try:
            if doc.is_encrypted:
                raise ValueError(f"PDF {source_path!r} is password-protected/encrypted.")

            page_count = doc.page_count
            if page_count == 0:
                raise ValueError(f"PDF {source_path!r} has no pages.")

            landscape_count = sum(1 for page in doc if page.rect.width > page.rect.height)
            is_slide_deck = (
                page_count > 0 and (landscape_count / page_count) >= _SLIDE_DECK_LANDSCAPE_RATIO
            )

            page_line_data: list[list[tuple[str, bool]]] = []
            page_font_ranked: list[list[str]] = []
            for page in doc:
                height = page.rect.height
                raw_lines = _merge_hyphenated_wraps(_extract_dict_lines(page))
                sizes = [font_size for *_rest, font_size in raw_lines if font_size > 0]
                median_size = statistics.median(sizes) if sizes else 0.0
                max_size = max(sizes) if sizes else 0.0
                expanded: list[tuple[str, bool]] = []
                for text, y0, y1, _x0, font_size in raw_lines:
                    eligible = _digit_collapse_eligible(
                        y0, y1, height, font_size, median_size, max_size
                    )
                    for part in _split_glued_bullets(text):
                        if part.strip():
                            expanded.append((part, eligible))
                page_line_data.append(expanded)
                page_font_ranked.append(_font_size_ranked_texts(page))

            boilerplate_lines = self._find_boilerplate_lines(page_line_data)

            sections: list[Section] = []
            for index, lines in enumerate(page_line_data):
                page_number = index + 1
                cleaned = self._clean_page_lines(lines, boilerplate_lines)
                if not cleaned.strip():
                    continue

                title = self._section_title(page_font_ranked[index], cleaned, boilerplate_lines)

                location = (
                    Location(slide=page_number) if is_slide_deck else Location(page=page_number)
                )
                sections.append(Section(text=cleaned, location=location, title=title))

            metadata = {
                "is_slide_deck": is_slide_deck,
                "page_count": page_count,
            }
            if not sections:
                metadata["warning"] = "no extractable text (scanned PDF?)"

            first_page_fonts = page_font_ranked[0] if page_font_ranked else []
            title = self._document_title(doc, first_page_fonts, boilerplate_lines, path)

            return ParsedDocument(
                source_path=source_path,
                source_type=SourceType.PDF,
                title=title,
                sections=sections,
                metadata=metadata,
            )
        finally:
            doc.close()

    def _find_boilerplate_lines(self, page_line_data: list[list[tuple[str, bool]]]) -> set[str]:
        """Boilerplate keys (see `_boilerplate_key`) that recur on more than
        half the pages.
        """
        if len(page_line_data) < _MIN_PAGES_FOR_BOILERPLATE:
            return set()
        counts: Counter[str] = Counter()
        for lines in page_line_data:
            seen_this_page: set[str] = set()
            for text, eligible in lines:
                stripped = text.strip()
                if not stripped or _is_page_number_line(stripped):
                    continue
                key = _boilerplate_key(stripped, eligible)
                if key is None:
                    continue
                if key not in seen_this_page:
                    counts[key] += 1
                    seen_this_page.add(key)
        threshold = _BOILERPLATE_REPEAT_RATIO * len(page_line_data)
        return {key for key, count in counts.items() if count > threshold}

    @staticmethod
    def _is_boilerplate_text(candidate: str, boilerplate_lines: set[str]) -> bool:
        """True if `candidate` matches, or is a component of, a known
        boilerplate line (e.g. the standalone title-slide occurrence of a
        header phrase that is otherwise merged into a footer bar on every
        other page: "DOCUMENT ANALYSIS" vs "ANU SCHOOL OF COMPUTING |
        DOCUMENT ANALYSIS").
        """
        ws = _normalize_ws(candidate)
        if not ws:
            return False
        for key in (ws, re.sub(r"\d+", "#", ws)):
            if key in boilerplate_lines:
                return True
            if any(key in line for line in boilerplate_lines):
                return True
        return False

    def _clean_page_lines(self, lines: list[tuple[str, bool]], boilerplate_lines: set[str]) -> str:
        kept: list[str] = []
        for text, eligible in lines:
            stripped = text.strip()
            if not stripped:
                continue
            if _is_page_number_line(stripped):
                continue
            key = _boilerplate_key(stripped, eligible)
            if key is not None and key in boilerplate_lines:
                continue
            kept.append(text)
        joined = "\n".join(kept)
        joined = normalize_whitespace(joined)
        return _convert_bullets(joined)

    def _section_title(
        self,
        font_ranked_texts: list[str],
        cleaned_text: str,
        boilerplate_lines: set[str],
    ) -> str | None:
        for text in font_ranked_texts:
            candidate = _clean_title_candidate(text)
            if candidate and not self._is_boilerplate_text(candidate, boilerplate_lines):
                return candidate
        for line in cleaned_text.split("\n"):
            stripped = line.strip()
            if stripped:
                return _clean_title_candidate(stripped) or stripped[:_MAX_TITLE_LEN]
        return None

    def _document_title(
        self,
        doc: pymupdf.Document,
        font_ranked_texts_page1: list[str],
        boilerplate_lines: set[str],
        path: Path,
    ) -> str:
        metadata_title = (doc.metadata or {}).get("title", "") or ""
        metadata_title = metadata_title.strip()
        for prefix in _GENERIC_TITLE_PREFIXES:
            if metadata_title.lower().startswith(prefix):
                metadata_title = metadata_title[len(prefix) :].strip()
                break
        if metadata_title and metadata_title.lower() not in _GENERIC_TITLES:
            return metadata_title

        # Prefer the largest non-boilerplate text on page 1, walking down
        # font sizes as needed: PowerPoint title slides sometimes give the
        # course/unit name its own (largest) font size, separate from the
        # actual topic, so the biggest span alone can be boilerplate even
        # when a smaller one on the same page isn't (see brief for the
        # "N-gram Language Models" example).
        for text in font_ranked_texts_page1:
            candidate = _clean_title_candidate(text)
            if candidate and not self._is_boilerplate_text(candidate, boilerplate_lines):
                return candidate
        return path.stem


register(PdfParser())
