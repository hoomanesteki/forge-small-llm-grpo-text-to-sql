from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from forge.data.spider import ChecksumMismatch, load_checksums, verify_files


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
