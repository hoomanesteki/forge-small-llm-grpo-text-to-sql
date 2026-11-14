"""Spider on disk: where it lives, what it must hash to, and loading it as examples.

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
from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError
from test_suite_eval.evaluation import Evaluator

from forge.core.types import Example, Split
from forge.ops.paths import repo_root

DATA_DIR_ENV = "FORGE_DATA_DIR"

# Question files in load order, with the split and id prefix of their examples. Ids
# number each file's items from 0 in file order, so they never change between runs.
# Val is carved out of train by database later, in forge.data.splits.
QUESTION_FILES: tuple[tuple[str, Split, str], ...] = (
    ("train_spider.json", "train", "spider-train"),
    ("train_others.json", "train", "spider-others"),
    ("dev.json", "dev", "spider-dev"),
)
TABLES_FILE = "tables.json"

# Gold queries that fail on SQLite (FG-101 spike): 03153 (assets_maintenance) names a
# table the database lacks, and 04513 and 04514 (document_management) put ORDER BY
# before INTERSECT. The official evaluator asserts that the gold runs, so FG-108 skips
# them.
FAILING_GOLD_IDS = frozenset({"spider-train-03153", "spider-train-04513", "spider-train-04514"})

# Training databases whose tables hold no rows (FG-101 spike): the six train_others
# databases and music_2. Every gold result on them is empty or a single 0 or NULL, so
# execution accuracy there says nothing, and val never takes them (ADR-0004).
ROWLESS_DBS = frozenset({"academic", "geo", "imdb", "music_2", "restaurants", "scholar", "yelp"})

# The full download's files and their sha256, in ``sha256sum`` format, so
# ``shasum -a 256 -c`` run inside the data directory checks them too.
CHECKSUMS: Traversable = files("forge.data") / "spider.sha256"

_CHECKSUM_LINE = re.compile(r"([0-9a-f]{64})  (\S.*)")
_CHUNK = 1 << 20


class DataNotFound(FileNotFoundError):
    pass


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


def db_path(db_id: str, data_dir: Path | None = None) -> Path:
    return resolve_data_dir(data_dir) / "database" / db_id / f"{db_id}.sqlite"


def raw_files(data_dir: Path | None = None) -> list[str]:
    """The JSON files and every ``database/<db>/<db>.sqlite``, relative to the data dir."""
    root = resolve_data_dir(data_dir)
    json_files = [name for name, _, _ in QUESTION_FILES] + [TABLES_FILE]
    missing = [name for name in json_files if not (root / name).is_file()]
    if missing:
        raise DataNotFound(_not_found(root, ", ".join(missing)))
    databases = [
        path.relative_to(root).as_posix()
        for path in (root / "database").glob("*/*.sqlite")
        if path.stem == path.parent.name
    ]
    return sorted(json_files + databases)


class _RawItem(BaseModel):
    db_id: str
    question: str
    query: str
    sql: dict[str, Any]  # the gold query, pre-parsed by Spider's authors


_ITEMS = TypeAdapter(list[_RawItem])


def load_raw(data_dir: Path | None = None) -> list[Example]:
    """Every Spider question, in file order: train_spider, train_others, then dev.

    Hardness is the official evaluator's, computed from each item's pre-parsed ``sql``
    field, which gives the published labels without parsing SQL (or needing nltk).
    """
    root = resolve_data_dir(data_dir)
    evaluator = Evaluator()
    examples: list[Example] = []
    for name, split, prefix in QUESTION_FILES:
        for index, item in enumerate(_read_items(root, name)):
            try:
                hardness = evaluator.eval_hardness(item.sql)
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError(f"{name} item {index}: unreadable sql field ({exc!r})") from exc
            examples.append(
                Example(
                    id=f"{prefix}-{index:05d}",
                    db_id=item.db_id,
                    question=item.question,
                    gold_sql=item.query,
                    hardness=hardness,
                    split=split,
                )
            )
    missing = sorted({e.db_id for e in examples if not db_path(e.db_id, root).is_file()})
    if missing:
        raise DataNotFound(
            _not_found(root, ", ".join(f"database/{db}/{db}.sqlite" for db in missing))
        )
    return examples


def _read_items(root: Path, name: str) -> list[_RawItem]:
    path = root / name
    if not path.is_file():
        raise DataNotFound(_not_found(root, name))
    try:
        return _ITEMS.validate_json(path.read_bytes())
    except ValidationError as exc:
        raise ValueError(f"{path} is not a list of Spider examples: {exc}") from exc


def _not_found(root: Path, what: str) -> str:
    return (
        f"{what} not found in {root}; run `forge data fetch`, or point {DATA_DIR_ENV} "
        "at a Spider data directory"
    )
