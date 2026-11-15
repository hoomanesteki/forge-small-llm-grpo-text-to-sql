"""``forge data fetch`` against small zips built here, with the Hub and Drive mocked."""

from __future__ import annotations

import dataclasses
import errno
import hashlib
import http.client
import io
import re
import urllib.error
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
import pytest
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from typer.testing import CliRunner

from forge import cli
from forge.data import fetch as fetch_mod
from forge.data import spider
from forge.data.fetch import FetchError, extract, fetch
from forge.data.spider import ChecksumMismatch
from tests.conftest import SPIDER_MINI

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


@pytest.fixture(autouse=True)
def repo_root_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The repo root for these tests, so a download can never land in the real data/."""
    monkeypatch.setenv("FORGE_ROOT", str(tmp_path))
    return tmp_path


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

    # A missing file is put back without asking.
    (data_dir / "dev.json").unlink()
    assert fetch(data_dir, zip_path=zip_path).zip_path == zip_path
    assert _files_under(data_dir) == WANTED

    # A file that is there but differs is only replaced with --force.
    (data_dir / "database" / "beta" / "beta.sqlite").write_bytes(b"tampered")
    (data_dir / "dev.json").unlink()
    with pytest.raises(FetchError, match=r"1 file\(s\) in .* differ .*database/beta/beta\.sqlite"):
        fetch(data_dir, zip_path=zip_path)
    assert (data_dir / "database" / "beta" / "beta.sqlite").read_bytes() == b"tampered"
    assert not (data_dir / "dev.json").exists()
    assert fetch(data_dir, zip_path=zip_path, force=True).zip_path == zip_path
    assert _files_under(data_dir) == WANTED

    # --force extracts even when everything already matches.
    assert fetch(data_dir, zip_path=zip_path, force=True).zip_path == zip_path
    assert _files_under(data_dir) == WANTED


@pytest.mark.phase(1)
def test_fetch_leaves_the_fixture_alone(spider_mini: Path, no_network: None) -> None:
    # FORGE_DATA_DIR still pointing at the fixture after a test run: its files differ from
    # the full download's, so fetch must neither download next to it nor extract over it.
    before = _files_under(SPIDER_MINI)

    result = CliRunner().invoke(cli.app, ["data", "fetch"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert f"file(s) in {SPIDER_MINI} differ from spider.sha256" in result.stderr
    assert "train_spider.json" in result.stderr
    assert "--force" in result.stderr
    assert _files_under(SPIDER_MINI) == before
    assert not (SPIDER_MINI.parent / "download").exists()


@pytest.mark.phase(1)
def test_download_lands_in_the_repo_data_dir(
    tmp_path: Path, pin: Callable[[str, bytes], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _spider_zip("spider")
    pin("hf", data)

    def fake_hf_hub_download(**kwargs: Any) -> str:
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return str(target)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)
    repo = tmp_path / "repo"
    monkeypatch.setenv("FORGE_ROOT", str(repo))
    data_dir = tmp_path / "elsewhere" / "spider"

    result = fetch(data_dir)

    # data/ is git-ignored; a folder next to FORGE_DATA_DIR may not be.
    assert result.zip_path == repo / "data" / "download" / "data" / "spider.zip"
    assert sorted(p.name for p in data_dir.parent.iterdir()) == ["spider"]
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


@pytest.mark.phase(1)
def test_fetch_cli_extracts_once_and_names_bad_files(
    tmp_path: Path,
    pin: Callable[[str, bytes], None],
    no_network: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _spider_zip("spider")
    pin("hf", data)
    zip_path = tmp_path / "spider.zip"
    zip_path.write_bytes(data)
    data_dir = tmp_path / "spider"
    monkeypatch.setenv("FORGE_DATA_DIR", str(data_dir))
    runner = CliRunner()

    first = runner.invoke(cli.app, ["data", "fetch", "--from-zip", str(zip_path)])
    assert first.exit_code == 0, first.output
    assert f"extracted 6 files from {zip_path} into {data_dir}" in first.stdout
    assert _files_under(data_dir) == WANTED

    again = runner.invoke(cli.app, ["data", "fetch"])
    assert again.exit_code == 0, again.output
    assert "all 6 files match spider.sha256, nothing to do" in again.stdout

    # A pinned zip whose dev.json differs from spider.sha256: the CLI names the file.
    bad = _spider_zip("spider", {**WANTED, "dev.json": b"[]"})
    pin("hf", bad)
    zip_path.write_bytes(bad)
    failed = runner.invoke(cli.app, ["data", "fetch", "--from-zip", str(zip_path), "--force"])
    assert failed.exit_code == 1
    assert failed.stdout == ""
    assert "error: 1 file(s) in" in failed.stderr
    assert "dev.json: sha256" in failed.stderr


def _hub_status_error() -> HfHubHTTPError:
    request = httpx2.Request("GET", "https://huggingface.co/datasets/xlangai/spider")
    return HfHubHTTPError("503 Server Error", response=httpx2.Response(503, request=request))


def _raise(exc: BaseException) -> Callable[..., Any]:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise exc

    return fail


@pytest.mark.phase(1)
@pytest.mark.parametrize(
    ("source", "target", "exc"),
    [
        # Offline: no connection and nothing in the Hub cache.
        ("hf", "huggingface_hub.hf_hub_download", LocalEntryNotFoundError("cannot reach the Hub")),
        ("hf", "huggingface_hub.hf_hub_download", _hub_status_error()),
        # Still failing after the Hub client's own retries; not an OSError.
        ("mirror", "huggingface_hub.hf_hub_download", httpx2.ConnectError("connection reset")),
        # hf_xet, which moves the bytes of the Hub zips, raises a bare RuntimeError when a
        # transfer fails; this is its message when the storage server can't be reached.
        (
            "hf",
            "huggingface_hub.hf_hub_download",
            RuntimeError(
                "Task error: File reconstruction error: CAS Client Error: Request middleware "
                "error: error sending request for url (http://127.0.0.1:9/v2/reconstructions/0)"
            ),
        ),
        ("gdrive", "urllib.request.urlopen", urllib.error.URLError("timed out")),
        # A body cut short; not an OSError either.
        ("gdrive", "urllib.request.urlopen", http.client.IncompleteRead(b"PK\x03\x04", 10)),
        ("hf", "huggingface_hub.hf_hub_download", OSError(errno.ENOSPC, "No space left on device")),
    ],
)
def test_fetch_cli_reports_download_failures(
    tmp_path: Path,
    pin: Callable[[str, bytes], None],
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    target: str,
    exc: BaseException,
) -> None:
    pin(source, _spider_zip("spider"))
    monkeypatch.setattr(target, _raise(exc))
    monkeypatch.setenv("FORGE_DATA_DIR", str(tmp_path / "spider"))

    result = CliRunner().invoke(cli.app, ["data", "fetch", "--source", source])

    # A clean exit with the hint, not a traceback.
    assert isinstance(result.exception, SystemExit), result.exception
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr.startswith(f"error: downloading the {source} zip failed: ")
    assert "Try another --source, or download the zip by hand and pass --from-zip" in result.stderr
    assert not (tmp_path / "spider").exists()


@pytest.mark.phase(1)
def test_fetch_cli_reports_a_full_disk(
    tmp_path: Path,
    pin: Callable[[str, bytes], None],
    no_network: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _spider_zip("spider")
    pin("hf", data)
    zip_path = tmp_path / "spider.zip"
    zip_path.write_bytes(data)
    data_dir = tmp_path / "spider"
    monkeypatch.setenv("FORGE_DATA_DIR", str(data_dir))
    monkeypatch.setattr(
        fetch_mod.shutil, "copyfileobj", _raise(OSError(errno.ENOSPC, "No space left on device"))
    )

    result = CliRunner().invoke(cli.app, ["data", "fetch", "--from-zip", str(zip_path)])

    assert isinstance(result.exception, SystemExit), result.exception
    assert result.exit_code == 1
    assert result.stderr.startswith(f"error: extracting {zip_path} into {data_dir} failed: ")
    assert "No space left on device" in result.stderr
    # The file being written is removed, so no torn copy is left behind.
    assert [p for p in data_dir.rglob("*") if p.is_file()] == []
