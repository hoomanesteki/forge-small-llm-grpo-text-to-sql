"""Where the Spider files live and what they must hash to.

The data directory is ``FORGE_DATA_DIR`` if set, else ``data/spider`` in the repo. It
holds Spider's JSON files and ``database/<db>/<db>.sqlite``, either the full download
(``forge data fetch``) or the committed fixture subset ``tests/fixtures/spider_mini``.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

from forge.ops.paths import repo_root

DATA_DIR_ENV = "FORGE_DATA_DIR"

# The full download's files and their sha256, in ``sha256sum`` format, so
# ``shasum -a 256 -c`` run inside the data directory checks them too.
CHECKSUMS: Traversable = files("forge.data") / "spider.sha256"

_CHECKSUM_LINE = re.compile(r"([0-9a-f]{64})  (\S.*)")
_CHUNK = 1 << 20


class ChecksumMismatch(ValueError):
    """Files in the data directory are missing or differ from their expected sha256."""

    def __init__(self, root: Path, problems: Mapping[str, str]) -> None:
        self.files = sorted(problems)
        details = "; ".join(f"{rel}: {problems[rel]}" for rel in self.files)
        super().__init__(f"{len(self.files)} file(s) in {root} failed verification: {details}")


def resolve_data_dir(data_dir: Path | str | None = None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    if os.environ.get(DATA_DIR_ENV):
        return Path(os.environ[DATA_DIR_ENV])
    return repo_root() / "data" / "spider"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def load_checksums(source: Traversable | None = None) -> dict[str, str]:
    """Parse a checksum list: one ``<sha256>  <path relative to the data dir>`` per line."""
    source = source or CHECKSUMS
    checksums: dict[str, str] = {}
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"{source.name} line {number} is not '<sha256>  <path>': {line!r}")
        digest, rel = match.groups()
        if rel in checksums:
            raise ValueError(f"{source.name} line {number} repeats {rel}")
        checksums[rel] = digest
    if not checksums:
        raise ValueError(f"{source.name} lists no files")
    return checksums


def check_files(root: Path, expected: Mapping[str, str]) -> dict[str, str]:
    """What is wrong with each listed file, by relative path; empty when all match."""
    problems: dict[str, str] = {}
    for rel, digest in sorted(expected.items()):
        path = root / rel
        if not path.is_file():
            problems[rel] = "missing"
            continue
        actual = file_sha256(path)
        if actual != digest:
            problems[rel] = f"sha256 {actual}, expected {digest}"
    return problems


def verify_files(root: Path, expected: Mapping[str, str]) -> None:
    problems = check_files(root, expected)
    if problems:
        raise ChecksumMismatch(root, problems)
