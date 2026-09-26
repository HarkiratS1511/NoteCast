"""A tiny registry that maps file extensions to the Parser that handles
them. Each parser module (added in Phase 1) calls `register()` with its own
parser instance at import time, so parallel builders can each own a
separate file without editing this one.
"""

from __future__ import annotations

from pathlib import Path

from notecast.interfaces import Parser

_PARSERS: dict[str, Parser] = {}


class UnsupportedFileError(ValueError):
    """Raised when no registered parser handles a file's extension."""


def register(parser: Parser) -> None:
    """Register `parser` for every extension in its `suffixes`."""
    for suffix in parser.suffixes:
        _PARSERS[suffix.lower()] = parser


def get_parser(path: Path) -> Parser:
    """Return the registered parser for `path`'s extension.

    Raises UnsupportedFileError if nothing is registered for it.
    """
    suffix = path.suffix.lower()
    parser = _PARSERS.get(suffix)
    if parser is None:
        raise UnsupportedFileError(f"No parser registered for extension {suffix!r} ({path})")
    return parser


def registered_suffixes() -> set[str]:
    """The set of extensions (e.g. {'.pdf', '.pptx'}) with a registered parser."""
    return set(_PARSERS.keys())
