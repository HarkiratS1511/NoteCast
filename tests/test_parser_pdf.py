"""Tests for the PDF parser (notecast.ingest.parsers.pdf).

All PDFs used here are synthetic, built in-memory/tmp_path with pymupdf.
Real course material under notebooks/ is never read in tests.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

import notecast.ingest.parsers.pdf  # noqa: F401 -- import registers PdfParser
from notecast.ingest.parsers import get_parser
from notecast.ingest.parsers.pdf import PdfParser

_DEJAVU_SANS = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

PORTRAIT = (612, 792)  # width, height
LANDSCAPE = (792, 612)


def _make_pdf(
    path: Path,
    pages: list[list[tuple[str, float]]],
    size: tuple[float, float] = PORTRAIT,
    title: str | None = None,
) -> None:
    """Build a PDF at `path`. `pages` is a list of pages, each a list of
    (text, fontsize) lines placed top-to-bottom.
    """
    doc = pymupdf.open()
    width, height = size
    for lines in pages:
        page = doc.new_page(width=width, height=height)
        y = 60.0
        for text, fontsize in lines:
            page.insert_text((50, y), text, fontsize=fontsize)
            y += fontsize + 20
    if title is not None:
        doc.set_metadata({"title": title})
    doc.save(path)
    doc.close()


def test_portrait_uses_page_numbers(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(
        pdf_path,
        pages=[
            [("Intro", 20), ("Some body text here.", 12)],
            [("Chapter 2", 20), ("More body text.", 12)],
        ],
    )
    parsed = PdfParser().parse(pdf_path, "doc.pdf")
    assert len(parsed.sections) == 2
    assert parsed.sections[0].location.page == 1
    assert parsed.sections[0].location.slide is None
    assert parsed.sections[1].location.page == 2
    assert parsed.sections[0].location.label() == "p. 1"
    assert parsed.metadata["is_slide_deck"] is False
    assert parsed.metadata["page_count"] == 2


def test_landscape_uses_slide_numbers(tmp_path: Path) -> None:
    pdf_path = tmp_path / "deck.pdf"
    _make_pdf(
        pdf_path,
        pages=[
            [("Slide One", 24), ("bullet content", 12)],
            [("Slide Two", 24), ("more content", 12)],
        ],
        size=LANDSCAPE,
    )
    parsed = PdfParser().parse(pdf_path, "deck.pdf")
    assert parsed.metadata["is_slide_deck"] is True
    assert parsed.sections[0].location.slide == 1
    assert parsed.sections[0].location.page is None
    assert parsed.sections[0].location.label() == "slide 1"
    assert parsed.sections[1].location.slide == 2


def test_repeated_footer_and_page_numbers_stripped(tmp_path: Path) -> None:
    pdf_path = tmp_path / "footer.pdf"
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    pages = []
    for i, word in enumerate(words, start=1):
        pages.append(
            [
                ("ANU SCHOOL OF COMPUTING", 10),
                ("DOCUMENT ANALYSIS", 10),
                (f"Content about {word}", 14),
                (str(i), 8),
            ]
        )
    _make_pdf(pdf_path, pages=pages)
    parsed = PdfParser().parse(pdf_path, "footer.pdf")
    assert len(parsed.sections) == 6
    for i, (section, word) in enumerate(zip(parsed.sections, words, strict=True), start=1):
        assert "ANU SCHOOL OF COMPUTING" not in section.text
        assert "DOCUMENT ANALYSIS" not in section.text
        # bare page number line should be gone too
        assert section.text.strip().splitlines()[-1] != str(i)
        assert f"Content about {word}" in section.text


def test_line_repeated_on_minority_of_pages_is_kept(tmp_path: Path) -> None:
    pdf_path = tmp_path / "partial_repeat.pdf"
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    pages = []
    for i, word in enumerate(words, start=1):
        lines = [(f"Body text about {word}", 14)]
        if i <= 2:
            lines.append(("Occasional note", 12))
        pages.append(lines)
    _make_pdf(pdf_path, pages=pages)
    parsed = PdfParser().parse(pdf_path, "partial_repeat.pdf")
    assert "Occasional note" in parsed.sections[0].text
    assert "Occasional note" in parsed.sections[1].text
    assert "Occasional note" not in parsed.sections[2].text


def test_title_from_largest_font(tmp_path: Path) -> None:
    pdf_path = tmp_path / "titled.pdf"
    _make_pdf(
        pdf_path,
        pages=[
            [("Big Important Title", 28), ("smaller supporting text", 12)],
        ],
    )
    parsed = PdfParser().parse(pdf_path, "titled.pdf")
    assert parsed.sections[0].title == "Big Important Title"


@pytest.mark.skipif(
    not _DEJAVU_SANS.exists(), reason="requires a DejaVu Sans font to render bullet glyphs"
)
def test_bullets_normalised(tmp_path: Path) -> None:
    pdf_path = tmp_path / "bullets.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    y = 60.0
    for text, fontsize in [
        ("Heading", 20),
        ("• first point", 12),
        ("▪ second point", 12),
        ("◦ third point", 12),
        ("● fourth point", 12),
    ]:
        page.insert_text(
            (50, y), text, fontsize=fontsize, fontfile=str(_DEJAVU_SANS), fontname="dejavu"
        )
        y += fontsize + 20
    doc.save(pdf_path)
    doc.close()
    parsed = PdfParser().parse(pdf_path, "bullets.pdf")
    text = parsed.sections[0].text
    assert "- first point" in text
    assert "- second point" in text
    assert "- third point" in text
    assert "- fourth point" in text
    for glyph in "•▪◦●":
        assert glyph not in text


def test_empty_page_skipped(tmp_path: Path) -> None:
    pdf_path = tmp_path / "gap.pdf"
    doc = pymupdf.open()
    doc.new_page(width=612, height=792)  # page 1: empty
    page2 = doc.new_page(width=612, height=792)
    page2.insert_text((50, 60), "Real content", fontsize=14)
    doc.save(pdf_path)
    doc.close()

    parsed = PdfParser().parse(pdf_path, "gap.pdf")
    assert len(parsed.sections) == 1
    assert parsed.sections[0].location.page == 2


def test_all_empty_document_warns(tmp_path: Path) -> None:
    pdf_path = tmp_path / "blank.pdf"
    doc = pymupdf.open()
    doc.new_page(width=612, height=792)
    doc.new_page(width=612, height=792)
    doc.save(pdf_path)
    doc.close()

    parsed = PdfParser().parse(pdf_path, "blank.pdf")
    assert parsed.sections == []
    assert parsed.metadata["warning"] == "no extractable text (scanned PDF?)"


def test_corrupt_pdf_raises_value_error(tmp_path: Path) -> None:
    bad_path = tmp_path / "corrupt.pdf"
    bad_path.write_bytes(b"not a real pdf" * 20)
    with pytest.raises(ValueError):
        PdfParser().parse(bad_path, "corrupt.pdf")


def test_registry_resolves_pdf_case_insensitively() -> None:
    # notecast.ingest.parsers.pdf is imported at module level above, which
    # registers PdfParser as a side effect.
    parser = get_parser(Path("x.PDF"))
    assert isinstance(parser, PdfParser)


def test_document_title_falls_back_to_stem_when_no_metadata_or_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "my-notes.pdf"
    doc = pymupdf.open()
    doc.new_page(width=612, height=792)
    doc.save(pdf_path)
    doc.close()

    parsed = PdfParser().parse(pdf_path, "my-notes.pdf")
    assert parsed.title == "my-notes"


def test_document_title_prefers_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "meta.pdf"
    _make_pdf(
        pdf_path,
        pages=[[("Some Content", 14)]],
        title="A Real Title",
    )
    parsed = PdfParser().parse(pdf_path, "meta.pdf")
    assert parsed.title == "A Real Title"


def test_document_title_ignores_generic_powerpoint_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "generic.pdf"
    _make_pdf(
        pdf_path,
        pages=[[("Actual Topic Title", 24), ("body", 12)]],
        title="PowerPoint Presentation",
    )
    parsed = PdfParser().parse(pdf_path, "generic.pdf")
    assert parsed.title == "Actual Topic Title"


def test_numbered_body_content_kept_on_every_page(tmp_path: Path) -> None:
    """Regression: a line that only *looks* the same across pages once
    digits are collapsed (e.g. "Body content 3") must NOT be treated as
    boilerplate when it sits in the body of the page, not the header or
    footer zone. Uses a large page count to match the case that failed
    verification.
    """
    pdf_path = tmp_path / "numbered_body.pdf"
    doc = pymupdf.open()
    for i in range(1, 301):
        page = doc.new_page(width=792, height=612)  # landscape slide
        # Well clear of the header/footer zones (12% of 612 = ~73.4px).
        page.insert_text((50, 300), f"Body content {i}", fontsize=14)
    doc.save(pdf_path)
    doc.close()

    parsed = PdfParser().parse(pdf_path, "numbered_body.pdf")
    assert len(parsed.sections) == 300
    for i, section in enumerate(parsed.sections, start=1):
        assert section.text == f"Body content {i}"


def test_numbered_body_content_kept_small_deck(tmp_path: Path) -> None:
    """Same regression as above with a small, easy-to-inspect deck: e.g.
    "Example 1".."Example 3" and "Slide N title" must survive on every page.
    """
    pdf_path = tmp_path / "small_numbered.pdf"
    doc = pymupdf.open()
    for i in range(1, 5):
        page = doc.new_page(width=612, height=792)  # portrait document
        page.insert_text((50, 300), f"Example {i}: some worked example", fontsize=14)
        page.insert_text((50, 330), f"Slide {i} title", fontsize=14)
    doc.save(pdf_path)
    doc.close()

    parsed = PdfParser().parse(pdf_path, "small_numbered.pdf")
    assert len(parsed.sections) == 4
    for i, section in enumerate(parsed.sections, start=1):
        assert f"Example {i}: some worked example" in section.text
        assert f"Slide {i} title" in section.text


def test_footer_whitespace_and_missing_page_number_variants_stripped(tmp_path: Path) -> None:
    """Footer variants that differ only in internal whitespace, and one
    page whose footer is missing its page number entirely, must all still
    be recognised as the same boilerplate and stripped.
    """
    pdf_path = tmp_path / "footer_variants.pdf"
    words = ["alpha", "bravo", "charlie", "delta"]
    footers = [
        "1              ANU SCHOOL OF COMPUTING  | DOCUMENT ANALYSIS",
        "2  ANU SCHOOL OF COMPUTING | DOCUMENT ANALYSIS",
        "3   ANU SCHOOL OF COMPUTING   |  DOCUMENT ANALYSIS",
        # week-2-slide-84 case: footer present but page number missing.
        "ANU SCHOOL OF COMPUTING | DOCUMENT ANALYSIS",
    ]
    doc = pymupdf.open()
    for word, footer in zip(words, footers, strict=True):
        page = doc.new_page(width=612, height=792)
        page.insert_text((50, 300), f"Content about {word}", fontsize=14)
        # Bottom footer zone (12% of 792 = ~95px from the bottom -> y > 697).
        page.insert_text((50, 760), footer, fontsize=8)
    doc.save(pdf_path)
    doc.close()

    parsed = PdfParser().parse(pdf_path, "footer_variants.pdf")
    assert len(parsed.sections) == 4
    for section, word in zip(parsed.sections, words, strict=True):
        assert "ANU SCHOOL OF COMPUTING" not in section.text
        assert "DOCUMENT ANALYSIS" not in section.text
        assert f"Content about {word}" in section.text


def test_watermark_repeated_verbatim_midpage_stripped(tmp_path: Path) -> None:
    """A line outside the header/footer zone that recurs verbatim on most
    pages (e.g. a watermark placed mid-page) is still boilerplate.
    """
    pdf_path = tmp_path / "watermark.pdf"
    words = ["alpha", "bravo", "charlie", "delta"]
    doc = pymupdf.open()
    for word in words:
        page = doc.new_page(width=612, height=792)
        page.insert_text((50, 300), "CONFIDENTIAL DRAFT", fontsize=10)
        page.insert_text((50, 400), f"Content about {word}", fontsize=14)
    doc.save(pdf_path)
    doc.close()

    parsed = PdfParser().parse(pdf_path, "watermark.pdf")
    assert len(parsed.sections) == 4
    for section, word in zip(parsed.sections, words, strict=True):
        assert "CONFIDENTIAL DRAFT" not in section.text
        assert f"Content about {word}" in section.text


def test_numbered_footer_in_bottom_zone_stripped(tmp_path: Path) -> None:
    """A footer that includes a running page count ("Page 3 of 60") in the
    bottom zone is recognised as boilerplate across pages via digit
    collapsing, while distinct body content elsewhere on the page survives.
    """
    pdf_path = tmp_path / "numbered_footer.pdf"
    words = ["alpha", "bravo", "charlie", "delta", "echo"]
    doc = pymupdf.open()
    for i, word in enumerate(words, start=1):
        page = doc.new_page(width=612, height=792)
        page.insert_text((50, 300), f"Content about {word}", fontsize=14)
        page.insert_text((50, 760), f"Page {i} of {len(words)} - COMP4650", fontsize=8)
    doc.save(pdf_path)
    doc.close()

    parsed = PdfParser().parse(pdf_path, "numbered_footer.pdf")
    assert len(parsed.sections) == 5
    for section, word in zip(parsed.sections, words, strict=True):
        assert "COMP4650" not in section.text
        assert f"Content about {word}" in section.text


@pytest.mark.skipif(
    not _DEJAVU_SANS.exists(), reason="requires a DejaVu Sans font to render bullet glyphs"
)
def test_mid_line_glued_bullet_is_split(tmp_path: Path) -> None:
    """A bullet glyph glued directly onto preceding text with no
    whitespace (common in tables/multi-column slide layouts) is split onto
    its own "- " line instead of surviving mid-sentence.
    """
    pdf_path = tmp_path / "glued_bullet.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text(
        (50, 300),
        "Relevant Not relevant• Example: cats and dogs",
        fontsize=12,
        fontfile=str(_DEJAVU_SANS),
        fontname="dejavu",
    )
    doc.save(pdf_path)
    doc.close()

    parsed = PdfParser().parse(pdf_path, "glued_bullet.pdf")
    text = parsed.sections[0].text
    assert "Relevant Not relevant" in text
    assert "- Example: cats and dogs" in text
    assert "•" not in text
    lines = text.split("\n")
    assert any(line.strip() == "Relevant Not relevant" for line in lines)


def test_hyphenated_word_wrap_is_joined(tmp_path: Path) -> None:
    """A word wrapped across two lines with a trailing hyphen, followed by
    a lowercase continuation, is joined back into one word.
    """
    pdf_path = tmp_path / "hyphen_wrap.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((50, 300), "Join us for a 30-", fontsize=12)
    page.insert_text((50, 320), "minute drop-in session next week.", fontsize=12)
    doc.save(pdf_path)
    doc.close()

    parsed = PdfParser().parse(pdf_path, "hyphen_wrap.pdf")
    text = parsed.sections[0].text
    assert "30-minute" in text
    assert "30-\nminute" not in text
    assert "30- minute" not in text


def test_title_slide_title_prefers_next_largest_non_boilerplate_span(tmp_path: Path) -> None:
    """If a section's largest-font text is itself (a component of) a
    detected boilerplate line -- e.g. the course-name kicker that's only
    standalone-sized on the title slide, but merged into every other page's
    footer -- fall through to the next-largest non-boilerplate span.
    """
    pdf_path = tmp_path / "title_slide.pdf"
    words = ["bravo", "charlie", "delta"]
    doc = pymupdf.open()
    for i in range(1, 5):
        page = doc.new_page(width=792, height=612)  # landscape slide deck
        if i == 1:
            page.insert_text((50, 100), "DOCUMENT ANALYSIS", fontsize=24)
            page.insert_text((50, 140), "N-gram Language Models", fontsize=16)
        else:
            page.insert_text((50, 140), f"Topic {words[i - 2]}", fontsize=20)
        page.insert_text(
            (50, 590), f"{i}    ANU SCHOOL OF COMPUTING | DOCUMENT ANALYSIS", fontsize=8
        )
        page.insert_text((50, 300), "Body content here", fontsize=12)
    doc.save(pdf_path)
    doc.close()

    parsed = PdfParser().parse(pdf_path, "title_slide.pdf")
    assert parsed.sections[0].title == "N-gram Language Models"
    assert parsed.title == "N-gram Language Models"


def test_numbered_slide_titles_in_top_zone_kept_in_text(tmp_path: Path) -> None:
    """Regression: a large, numbered slide title sitting in the top zone
    (e.g. "Example 1".."Example 5") must NOT be stripped as boilerplate --
    digit collapsing only applies to small header/footer text, never to
    the page's largest-font span, even when that span sits in the zone.
    """
    pdf_path = tmp_path / "example_titles.pdf"
    doc = pymupdf.open()
    for i in range(1, 6):
        page = doc.new_page(width=792, height=612)  # landscape slide deck
        page.insert_text((50, 40), f"Example {i}", fontsize=24)
        page.insert_text((50, 300), f"Unique body line {i} about something", fontsize=12)
    doc.save(pdf_path)
    doc.close()

    parsed = PdfParser().parse(pdf_path, "example_titles.pdf")
    assert len(parsed.sections) == 5
    for i, section in enumerate(parsed.sections, start=1):
        assert f"Example {i}" in section.text
        assert f"Unique body line {i} about something" in section.text
        assert section.title == f"Example {i}"


def test_small_font_footer_in_top_zone_still_stripped(tmp_path: Path) -> None:
    """A small-font "Page N of M" footer placed in the *top* zone (not just
    the bottom) is still recognised and stripped via digit collapsing,
    since it's small relative to the page's body/title text and isn't the
    page's largest span.
    """
    pdf_path = tmp_path / "top_footer.pdf"
    words = ["alpha", "bravo", "charlie", "delta", "echo"]
    doc = pymupdf.open()
    for i, word in enumerate(words, start=1):
        page = doc.new_page(width=612, height=792)
        page.insert_text((50, 30), f"Page {i} of {len(words)}", fontsize=8)
        page.insert_text((50, 300), f"Content about {word}", fontsize=14)
    doc.save(pdf_path)
    doc.close()

    parsed = PdfParser().parse(pdf_path, "top_footer.pdf")
    assert len(parsed.sections) == 5
    for section, word in zip(parsed.sections, words, strict=True):
        assert "Page" not in section.text
        assert "of 5" not in section.text
        assert f"Content about {word}" in section.text
