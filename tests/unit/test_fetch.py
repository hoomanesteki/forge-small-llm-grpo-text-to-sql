"""``forge data fetch`` against small zips built here, with the Hub and Drive mocked."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import re
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from forge.data import fetch as fetch_mod
from forge.data import spider
from forge.data.fetch import FetchError, extract, fetch
from forge.data.spider import ChecksumMismatch

# What a pinned zip holds that Forge keeps, keyed by path inside the Spider folder.
WANTED = {
    "train_spider.json": b'[{"db_id": "alpha"}]',
    "train_others.json": b"[]",
    "dev.json": b'[{"db_id": "beta"}]',
    "tables.json": b"[]",
    "database/alpha/alpha.sqlite": b"SQLite format 3\x00alpha",
    "database/beta/beta.sqlite": b"SQLite format 3\x00beta",
}
# What the real zips also hold and Forge leaves behind.
UNWANTED = {
    "README.txt": b"citations",
    "train_gold.sql": b"SELECT 1\talpha",
    "test.json": b"[]",
    ".DS_Store": b"\x00",
    "database/alpha/schema.sql": b"CREATE TABLE t (x)",
    "database/formula_1/formula_1.splite": b"",
    "test_database/gamma/gamma.sqlite": b"SQLite format 3\x00gamma",
}
OUTSIDE = {
    "__MACOSX/spider/._train_spider.json": b"resource fork",
    "__MACOSX/spider_data/._dev.json": b"resource fork",
    ".DS_Store": b"\x00",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _spider_zip(folder: str, files: dict[str, bytes] | None = None) -> bytes:
    inside = {**WANTED, **UNWANTED} if files is None else files
    return _zip_bytes({**{f"{folder}/{rel}": data for rel, data in inside.items()}, **OUTSIDE})


def _files_under(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def pin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[str, bytes], None]:
    """Make the test zips the pinned ones: list WANTED in spider.sha256, pin a source."""
    listing = tmp_path / "spider.sha256"
    listing.write_text("".join(f"{_sha(data)}  {rel}\n" for rel, data in WANTED.items()))
    monkeypatch.setattr(spider, "CHECKSUMS", listing)

    def pin_source(name: str, data: bytes) -> None:
        source = dataclasses.replace(fetch_mod.SOURCES[name], sha256=_sha(data))
        monkeypatch.setitem(fetch_mod.SOURCES, name, source)

    return pin_source


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("tried to download")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", refuse)
    monkeypatch.setattr("urllib.request.urlopen", refuse)


@pytest.mark.phase(1)
@pytest.mark.parametrize(("folder", "source"), [("spider", "hf"), ("spider_data", "mirror")])
def test_extracts_only_the_listed_files(
    tmp_path: Path, pin: Callable[[str, bytes], None], no_network: None, folder: str, source: str
) -> None:
    data = _spider_zip(folder)
    pin(source, data)
    zip_path = tmp_path / "upload.zip"
    zip_path.write_bytes(data)
    data_dir = tmp_path / "data" / "spider"

    result = fetch(data_dir, zip_path=zip_path)

    assert result == fetch_mod.FetchResult(data_dir, len(WANTED), zip_path)
    assert _files_under(data_dir) == WANTED
    assert not (tmp_path / "data" / "__MACOSX").exists()


@pytest.mark.phase(1)
def test_fetch_is_a_no_op_when_the_files_match(
    tmp_path: Path, pin: Callable[[str, bytes], None], no_network: None
) -> None:
    data = _spider_zip("spider")
    pin("hf", data)
    zip_path = tmp_path / "spider.zip"
    zip_path.write_bytes(data)
    data_dir = tmp_path / "spider"
    fetch(data_dir, zip_path=zip_path)
    zip_path.unlink()
    stamps = {p: p.stat().st_mtime_ns for p in data_dir.rglob("*") if p.is_file()}

    # No zip and no network needed: the files already verify.
    assert fetch(data_dir).zip_path is None
    assert {p: p.stat().st_mtime_ns for p in data_dir.rglob("*") if p.is_file()} == stamps


@pytest.mark.phase(1)
def test_stale_files_are_extracted_again(
    tmp_path: Path, pin: Callable[[str, bytes], None], no_network: None
) -> None:
    data = _spider_zip("spider")
    pin("hf", data)
    zip_path = tmp_path / "spider.zip"
    zip_path.write_bytes(data)
    data_dir = tmp_path / "spider"
    fetch(data_dir, zip_path=zip_path)

    (data_dir / "database" / "beta" / "beta.sqlite").write_bytes(b"tampered")
    (data_dir / "dev.json").unlink()
    assert fetch(data_dir, zip_path=zip_path).zip_path == zip_path
    assert _files_under(data_dir) == WANTED

    # --force extracts even when everything already matches.
    assert fetch(data_dir, zip_path=zip_path, force=True).zip_path == zip_path
    assert _files_under(data_dir) == WANTED


@pytest.mark.phase(1)
def test_unknown_zip_is_refused(
    tmp_path: Path, pin: Callable[[str, bytes], None], no_network: None
) -> None:
    pin("hf", _spider_zip("spider"))
    other = tmp_path / "other.zip"
    other.write_bytes(_spider_zip("spider", {**WANTED, "README.txt": b"a different release"}))
    data_dir = tmp_path / "spider"

    with pytest.raises(FetchError, match=f"sha256 {_sha(other.read_bytes())}, which is not a"):
        fetch(data_dir, zip_path=other)
    assert not data_dir.exists()


@pytest.mark.phase(1)
def test_extracted_file_with_the_wrong_sha256_is_named(
    tmp_path: Path, pin: Callable[[str, bytes], None], no_network: None
) -> None:
    data = _spider_zip("spider", {**WANTED, "dev.json": b"[]"})
    pin("hf", data)
    zip_path = tmp_path / "spider.zip"
    zip_path.write_bytes(data)

    with pytest.raises(ChecksumMismatch, match=r"dev\.json: sha256") as caught:
        fetch(tmp_path / "spider", zip_path=zip_path)
    assert caught.value.files == ["dev.json"]


@pytest.mark.phase(1)
def test_zip_missing_a_listed_file_is_refused(
    tmp_path: Path, pin: Callable[[str, bytes], None], no_network: None
) -> None:
    partial = {rel: data for rel, data in WANTED.items() if rel != "database/beta/beta.sqlite"}
    data = _spider_zip("spider", partial)
    pin("hf", data)
    zip_path = tmp_path / "spider.zip"
    zip_path.write_bytes(data)

    with pytest.raises(FetchError, match=r"lacks 1 file\(s\): database/beta/beta.sqlite"):
        fetch(tmp_path / "spider", zip_path=zip_path)


@pytest.mark.phase(1)
@pytest.mark.parametrize(
    ("members", "found"),
    [
        ({"data/train_spider.json": b"[]"}, "neither"),
        ({"spider/dev.json": b"[]", "spider_data/dev.json": b"[]"}, "spider/, spider_data/"),
    ],
)
def test_zip_needs_exactly_one_spider_folder(
    tmp_path: Path, members: dict[str, bytes], found: str
) -> None:
    zip_path = tmp_path / "odd.zip"
    zip_path.write_bytes(_zip_bytes(members))
    message = f"one Spider folder (spider/ or spider_data/), found {found}"
    with pytest.raises(FetchError, match=re.escape(message)):
        extract(zip_path, tmp_path / "out", ["dev.json"])
    assert not (tmp_path / "out").exists()


@pytest.mark.phase(1)
def test_member_names_cannot_escape_the_data_dir(tmp_path: Path) -> None:
    dest = tmp_path / "box" / "spider"
    hostile = {
        "spider/dev.json": b"[]",
        "spider/../escaped.json": b"[]",
        "spider/database/../../../escaped.sqlite": b"x",
        "/spider/absolute.json": b"[]",
    }
    zip_path = tmp_path / "hostile.zip"
    zip_path.write_bytes(_zip_bytes(hostile))

    # Output paths come from the wanted list, so stray members are never written.
    assert extract(zip_path, dest, ["dev.json"]) == ["dev.json"]
    assert _files_under(tmp_path / "box") == {"spider/dev.json": b"[]"}

    # A wanted path that climbs out is refused before anything at all is written.
    (dest / "dev.json").unlink()
    with pytest.raises(FetchError, match=r"outside .*: \.\./escaped\.json"):
        extract(zip_path, dest, ["dev.json", "../escaped.json"])
    assert _files_under(tmp_path / "box") == {}


@pytest.mark.phase(1)
def test_hub_download_is_pinned_and_reused(
    tmp_path: Path, pin: Callable[[str, bytes], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _spider_zip("spider")
    pin("hf", data)
    calls: list[dict[str, Any]] = []

    def fake_hf_hub_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return str(target)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)
    data_dir = tmp_path / "data" / "spider"

    result = fetch(data_dir)

    assert result.zip_path == tmp_path / "data" / "download" / "data" / "spider.zip"
    assert calls == [
        {
            "repo_id": "xlangai/spider",
            "filename": "data/spider.zip",
            "repo_type": "dataset",
            "revision": "6232cc3fad6d54c62b3ba23a364083a98ff36a17",
            "local_dir": tmp_path / "data" / "download",
            "force_download": False,
        }
    ]
    assert _files_under(data_dir) == WANTED

    # The verified zip on disk is used again without asking the Hub.
    assert fetch(data_dir, force=True).zip_path == result.zip_path
    assert len(calls) == 1


@pytest.mark.phase(1)
def test_drive_download_lands_at_the_pinned_path(
    tmp_path: Path, pin: Callable[[str, bytes], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _spider_zip("spider_data")
    pin("gdrive", data)
    urls: list[str] = []

    def fake_urlopen(url: str, timeout: float) -> io.BytesIO:
        urls.append(url)
        return io.BytesIO(data)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    download_dir = tmp_path / "download"

    result = fetch(tmp_path / "spider", source="gdrive", download_dir=download_dir)

    assert len(urls) == 1
    assert "id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J" in urls[0]
    assert result.zip_path == download_dir / "spider_data.zip"
    assert sorted(p.name for p in download_dir.iterdir()) == ["spider_data.zip"]
    assert _files_under(tmp_path / "spider") == WANTED


@pytest.mark.phase(1)
def test_download_with_the_wrong_sha256_is_deleted(
    tmp_path: Path, pin: Callable[[str, bytes], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    pin("gdrive", _spider_zip("spider_data"))
    # Drive answers a throttled download with an HTML page, not the zip.
    page = b"<html>Too many users have viewed or downloaded this file recently</html>"
    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout: io.BytesIO(page))
    download_dir = tmp_path / "download"

    with pytest.raises(FetchError, match=f"gdrive served a file with sha256 {_sha(page)}"):
        fetch(tmp_path / "spider", source="gdrive", download_dir=download_dir)
    assert list(download_dir.iterdir()) == []
    assert not (tmp_path / "spider").exists()
