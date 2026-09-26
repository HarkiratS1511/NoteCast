"""Tests for the parser registry in notecast.ingest.parsers."""

from __future__ import annotations

from pathlib import Path

import pytest

from notecast.ingest.parsers import (
    UnsupportedFileError,
    get_parser,
    register,
    registered_suffixes,
)
from notecast.models import ParsedDocument, SourceType


class FakeParser:
    suffixes = (".fake", ".FAK")

    def parse(self, path: Path, source_path: str) -> ParsedDocument:
        return ParsedDocument(
            source_path=source_path,
            source_type=SourceType.TXT,
            sections=[],
        )


@pytest.fixture(autouse=True)
def _register_fake() -> None:
    register(FakeParser())


def test_lookup_case_insensitive() -> None:
    parser = get_parser(Path("notes.FAKE"))
    assert isinstance(parser, FakeParser)
    parser2 = get_parser(Path("notes.fake"))
    assert isinstance(parser2, FakeParser)


def test_registered_suffixes_includes_registered() -> None:
    assert ".fake" in registered_suffixes()


def test_unsupported_raises() -> None:
    with pytest.raises(UnsupportedFileError):
        get_parser(Path("video.mp4"))
