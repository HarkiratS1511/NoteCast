"""Reading and writing a notebook's manifest.json: the record of ingested
files (content hash, parser, chunk count) that makes re-ingesting cheap.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from notecast.models import Manifest

if TYPE_CHECKING:
    from notecast.notebook import Notebook

logger = logging.getLogger(__name__)

_READ_CHUNK_SIZE = 1024 * 1024


def file_sha256(path: Path) -> str:
    """Stream-hash a file's contents (sha256, hex digest)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(_READ_CHUNK_SIZE)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_manifest(nb: Notebook) -> Manifest:
    """Load `nb.manifest_path`. Missing -> empty manifest. Corrupt JSON or
    an invalid shape -> back it up to manifest.json.bak, log a warning, and
    return an empty manifest.
    """
    path = nb.manifest_path
    if not path.exists():
        return Manifest()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return Manifest.model_validate(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        backup = path.with_name(path.name + ".bak")
        try:
            path.replace(backup)
        except OSError:
            logger.warning("Could not back up corrupt manifest at %s", path)
        else:
            logger.warning(
                "Manifest at %s was corrupt (%s); backed up to %s and starting fresh",
                path,
                exc,
                backup,
            )
        return Manifest()


def save_manifest(nb: Notebook, manifest: Manifest) -> None:
    """Write the manifest atomically: a temp file in the same directory,
    then an OS-level rename (works on Windows too).
    """
    path = nb.manifest_path
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".manifest-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
