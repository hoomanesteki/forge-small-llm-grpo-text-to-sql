from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from forge.data.spider import (
    ChecksumMismatch,
    DataNotFound,
    load_checksums,
    load_raw,
    resolve_data_dir,
    verify_files,
)
from forge.ops.paths import repo_root

# Spider's pre-parsed form of "SELECT count(*) FROM head WHERE age > 56", an easy query.
EASY_SQL: dict[str, Any] = {
    "from": {"table_units": [["table_unit", 1]], "conds": []},
    "select": [False, [[3, [0, [0, 0, False], None]]]],
    "where": [[False, 3, [0, [0, 10, False], None], 56.0, None]],
    "groupBy": [],
    "having": [],
    "orderBy": [],
    "limit": None,
    "intersect": None,
    "union": None,
    "except": None,
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _item(db_id: str, question: str) -> dict[str, Any]:
    return {
        "db_id": db_id,
        "question": question,
        "query": f"SELECT 1 -- {question}",
        "sql": EASY_SQL,
    }


def _data_dir(root: Path, files: dict[str, list[dict[str, Any]]]) -> Path:
    """A Spider-shaped directory with these question files and an empty db for each db_id."""
    root.mkdir(parents=True, exist_ok=True)
    for name, items in files.items():
        (root / name).write_text(json.dumps(items))
        for item in items:
            db = root / "database" / item["db_id"]
            db.mkdir(parents=True, exist_ok=True)
            (db / f"{item['db_id']}.sqlite").touch()
    return root


@pytest.mark.phase(1)
def test_checksum_mismatch_raises(tmp_path: Path) -> None:
    (tmp_path / "database" / "geo").mkdir(parents=True)
    (tmp_path / "tables.json").write_bytes(b"[]")
    (tmp_path / "database" / "geo" / "geo.sqlite").write_bytes(b"SQLite format 3\x00 edited")
    expected = {
        "tables.json": _sha(b"[]"),
        "database/geo/geo.sqlite": _sha(b"SQLite format 3\x00 original"),
        "dev.json": _sha(b"[]"),
    }

    with pytest.raises(ChecksumMismatch) as caught:
        verify_files(tmp_path, expected)

    # Only the bad files are named, each with what is wrong with it.
    assert caught.value.files == ["database/geo/geo.sqlite", "dev.json"]
    message = str(caught.value)
    assert "database/geo/geo.sqlite: sha256 " in message
    assert f"expected {expected['database/geo/geo.sqlite']}" in message
    assert "dev.json: missing" in message
    assert "tables.json" not in message

    (tmp_path / "database" / "geo" / "geo.sqlite").write_bytes(b"SQLite format 3\x00 original")
    (tmp_path / "dev.json").write_bytes(b"[]")
    verify_files(tmp_path, expected)


@pytest.mark.phase(1)
def test_committed_checksums_cover_the_spider_files() -> None:
    checksums = load_checksums()
    assert len(checksums) == 170
    databases = [rel for rel in checksums if rel.startswith("database/")]
    assert len(databases) == 166
    assert all(re.fullmatch(r"database/([^/]+)/\1\.sqlite", rel) for rel in databases)
    # The JSON values recorded by the FG-101 spike (docs/spikes/data.md).
    assert {rel: digest for rel, digest in checksums.items() if rel.endswith(".json")} == {
        "train_spider.json": "c43d0d72e59e1a9e1a60837da9bf70d5a6277226bdb7f634d544f380646f527a",
        "train_others.json": "7adb04af470b3c9be653504e03c9a36c1b963a861f308ecf25d436472284e10f",
        "dev.json": "30d64a3fccde493226df79687aed9e4a1c0129525baf44f29c0573d914d758a4",
        "tables.json": "61bb20aa401f03164e2d7f3b16509b7b5f79cc9c943ca7bd159046df1159e2ed",
    }
    # test.json, test_tables.json and test_database/ are not training or dev data.
    assert not any(rel.startswith("test") for rel in checksums)


@pytest.mark.phase(1)
def test_checksum_list_rejects_malformed_lines(tmp_path: Path) -> None:
    good = f"{_sha(b'')}  dev.json"
    listing = tmp_path / "list.sha256"

    listing.write_text(f"{good}\n{_sha(b'')} tables.json\n")
    with pytest.raises(ValueError, match="line 2 is not"):
        load_checksums(listing)

    listing.write_text(f"{good}\n{good}\n")
    with pytest.raises(ValueError, match=r"line 2 repeats dev\.json"):
        load_checksums(listing)

    listing.write_text("")
    with pytest.raises(ValueError, match="lists no files"):
        load_checksums(listing)

    listing.write_text(f"{good}\n")
    assert load_checksums(listing) == {"dev.json": _sha(b"")}


@pytest.mark.phase(1)
def test_ids_number_each_file_in_order(tmp_path: Path) -> None:
    root = _data_dir(
        tmp_path / "spider",
        {
            "train_spider.json": [_item("a", "q0"), _item("b", "q1")],
            "train_others.json": [_item("c", "q2")],
            "dev.json": [_item("d", "q3"), _item("d", "q4")],
        },
    )

    examples = load_raw(root)

    assert [(e.id, e.split, e.db_id, e.question) for e in examples] == [
        ("spider-train-00000", "train", "a", "q0"),
        ("spider-train-00001", "train", "b", "q1"),
        ("spider-others-00000", "train", "c", "q2"),
        ("spider-dev-00000", "dev", "d", "q3"),
        ("spider-dev-00001", "dev", "d", "q4"),
    ]
    assert examples[0].gold_sql == "SELECT 1 -- q0"
    assert {e.hardness for e in examples} == {"easy"}


@pytest.mark.phase(1)
def test_missing_files_are_named(tmp_path: Path) -> None:
    root = _data_dir(
        tmp_path / "spider",
        {"train_spider.json": [_item("a", "q0")], "train_others.json": []},
    )
    with pytest.raises(DataNotFound, match=r"dev\.json not found in .*forge data fetch"):
        load_raw(root)

    (root / "dev.json").write_text(json.dumps([_item("gone", "q1")]))
    with pytest.raises(DataNotFound, match=r"database/gone/gone\.sqlite not found"):
        load_raw(root)


@pytest.mark.phase(1)
def test_malformed_items_name_the_file_and_item(tmp_path: Path) -> None:
    no_sql = {key: value for key, value in _item("a", "q1").items() if key != "sql"}
    root = _data_dir(
        tmp_path / "spider",
        {"train_spider.json": [_item("a", "q0"), no_sql], "train_others.json": [], "dev.json": []},
    )
    with pytest.raises(ValueError, match=r"train_spider\.json is not a list of Spider .*\n1\.sql"):
        load_raw(root)

    broken = {**_item("a", "q0"), "sql": {"select": [False, []]}}
    (root / "train_spider.json").write_text(json.dumps([broken]))
    with pytest.raises(ValueError, match=r"train_spider\.json item 0: unreadable sql field"):
        load_raw(root)


@pytest.mark.phase(1)
def test_data_dir_comes_from_the_argument_env_or_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FORGE_DATA_DIR", raising=False)
    assert resolve_data_dir() == repo_root() / "data" / "spider"
    monkeypatch.setenv("FORGE_DATA_DIR", str(tmp_path / "elsewhere"))
    assert resolve_data_dir() == tmp_path / "elsewhere"
    assert resolve_data_dir(tmp_path / "given") == tmp_path / "given"
