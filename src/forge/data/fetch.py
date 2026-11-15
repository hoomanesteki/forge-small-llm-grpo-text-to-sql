"""``forge data fetch``: get the pinned Spider zip, extract what Forge uses, verify it.

Each source serves a zip with a known sha256, and every extracted file must then match
``spider.sha256``, so every machine ends up with byte-identical data whichever source it
used. Only the files listed there are extracted: Spider's four JSON files and
``database/<db>/<db>.sqlite``. README, schema dumps, CSVs, ``test_database/``,
``.DS_Store`` and ``__MACOSX/`` stay in the zip.

Fetch fills in missing files, but replaces a file that is there and differs only with
``force``: the directory may hold other data, such as the committed fixture subset.
Zips are downloaded to ``data/download`` in the repo, which git ignores.
"""

from __future__ import annotations

import http.client
import os
import shutil
import tempfile
import urllib.request
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from forge.data.spider import check_files, file_sha256, load_checksums, verify_files
from forge.ops.paths import repo_root

SourceName = Literal["hf", "mirror", "gdrive"]

# The Spider folder at the top of each zip: spider.zip uses spider/, the official
# spider_data.zip (and its mirror) spider_data/.
ZIP_FOLDERS = ("spider", "spider_data")

_CHUNK = 1 << 20
_TIMEOUT_S = 60
_RETRY_HINT = "Try another --source, or download the zip by hand and pass --from-zip"


class FetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class ZipSource:
    """A pinned Spider zip. ``path`` is where it lands, relative to the download dir."""

    path: str
    sha256: str
    repo_id: str | None = None  # a Hugging Face dataset repo, pinned to ``revision``
    revision: str | None = None
    url: str | None = None  # a plain HTTPS download, for sources outside the Hub


SOURCES: dict[str, ZipSource] = {
    # The Spider authors' lab org, at the last revision before its parquet conversion
    # dropped the databases (docs/spikes/data.md).
    "hf": ZipSource(
        path="data/spider.zip",
        sha256="5ddff97bb1d421282c593e8d30ce0ce107270f4dd4a21d60eba4bf287d5956b1",
        repo_id="xlangai/spider",
        revision="6232cc3fad6d54c62b3ba23a364083a98ff36a17",
    ),
    # A third-party re-host of the official Drive file, which adds the test set.
    "mirror": ZipSource(
        path="spider_data.zip",
        sha256="00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b",
        repo_id="HAL-9001/spider-databases",
        revision="4a01bbac6520cd35b216db9e1724e5e1ada60aa4",
    ),
    # The official Google Drive file. The mirror matches its size and central
    # directory, so it should have the mirror's sha256; if Drive's copy ever changes,
    # or Drive answers with a quota page, the zip check fails instead of extracting it.
    "gdrive": ZipSource(
        path="spider_data.zip",
        sha256="00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b",
        url=(
            "https://drive.usercontent.google.com/download"
            "?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download&confirm=t"
        ),
    ),
}


@dataclass(frozen=True)
class FetchResult:
    data_dir: Path
    files: int
    zip_path: Path | None  # None when the files already matched and nothing was extracted


def verify_zip(zip_path: Path) -> None:
    """Refuse a zip that isn't one of the pinned sources."""
    actual = file_sha256(zip_path)
    if actual not in {source.sha256 for source in SOURCES.values()}:
        raise FetchError(
            f"{zip_path} has sha256 {actual}, which is not a known Spider zip "
            f"({', '.join(f'{name}: {s.sha256}' for name, s in SOURCES.items())})"
        )


def default_download_dir() -> Path:
    """``data/download`` in the repo, wherever the data directory is: git ignores it."""
    return repo_root() / "data" / "download"


def download(source_name: SourceName, download_dir: Path) -> Path:
    """The source's zip under ``download_dir``, downloaded only if no valid copy is there."""
    source = SOURCES[source_name]
    target = download_dir / source.path
    if target.is_file() and file_sha256(target) == source.sha256:
        return target
    if source.repo_id is not None:
        _download_hub(source_name, source.repo_id, source, download_dir, force=target.exists())
    elif source.url is not None:
        _download_url(source_name, source.url, target)
    else:
        raise FetchError(f"source {source_name} has neither a repo nor a url")
    actual = file_sha256(target)
    if actual != source.sha256:
        target.unlink()
        raise FetchError(
            f"{source_name} served a file with sha256 {actual}, expected {source.sha256}; "
            f"it was deleted. {_RETRY_HINT}"
        )
    return target


def _download_failed(source_name: str, exc: BaseException) -> FetchError:
    return FetchError(f"downloading the {source_name} zip failed: {exc}. {_RETRY_HINT}")


def _download_hub(
    source_name: str, repo_id: str, source: ZipSource, download_dir: Path, *, force: bool
) -> None:
    # Imported here: --from-zip and an already valid copy never need the Hub client.
    import httpx2
    from huggingface_hub import hf_hub_download

    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=source.path,
            repo_type="dataset",
            revision=source.revision,
            local_dir=download_dir,
            # A copy that failed the sha256 check must not be reused from the local cache.
            force_download=force,
        )
    # The Hub's own errors (offline, HTTP status) are OSErrors; a connection that still
    # fails after the client's retries raises a bare httpx2 error.
    except (OSError, httpx2.HTTPError) as exc:
        raise _download_failed(source_name, exc) from exc


def _download_url(source_name: str, url: str, target: Path) -> None:
    partial = target.with_name(target.name + ".part")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out, _CHUNK)
        os.replace(partial, target)
    # URLError and timeouts are OSErrors; a body cut short raises IncompleteRead.
    except (OSError, http.client.HTTPException) as exc:
        raise _download_failed(source_name, exc) from exc
    finally:
        partial.unlink(missing_ok=True)


def _spider_members(archive: zipfile.ZipFile, zip_path: Path) -> dict[str, zipfile.ZipInfo]:
    """Members of the zip's Spider folder, keyed by their path inside that folder."""
    folders: dict[str, dict[str, zipfile.ZipInfo]] = {}
    for info in archive.infolist():
        top, _, rel = info.filename.partition("/")
        if top in ZIP_FOLDERS and rel and not info.is_dir():
            folders.setdefault(top, {})[rel] = info
    if len(folders) != 1:
        expected = " or ".join(f"{folder}/" for folder in ZIP_FOLDERS)
        found = ", ".join(f"{folder}/" for folder in sorted(folders)) or "neither"
        raise FetchError(f"{zip_path} should hold one Spider folder ({expected}), found {found}")
    return next(iter(folders.values()))


def extract(zip_path: Path, dest: Path, wanted: Iterable[str]) -> list[str]:
    """Extract the ``wanted`` paths of the zip's Spider folder into ``dest``.

    Output paths come from ``wanted``, never from member names, and must resolve inside
    ``dest``, so a hostile member name can't write elsewhere (zip slip). Each file is
    written to a temporary name and renamed, so an interrupted run leaves no torn file.
    """
    root = dest.resolve()
    targets = {rel: (root / rel).resolve() for rel in sorted(wanted)}
    escaping = [rel for rel, target in targets.items() if not target.is_relative_to(root)]
    if escaping:
        raise FetchError(f"refusing to extract outside {dest}: {', '.join(escaping)}")
    try:
        with zipfile.ZipFile(zip_path) as archive:
            members = _spider_members(archive, zip_path)
            missing = [rel for rel in targets if rel not in members]
            if missing:
                raise FetchError(f"{zip_path} lacks {len(missing)} file(s): {', '.join(missing)}")
            for rel, target in targets.items():
                target.parent.mkdir(parents=True, exist_ok=True)
                _extract_member(archive, members[rel], target)
    # A full disk or a read error, or a member whose CRC32 fails.
    except (OSError, zipfile.BadZipFile) as exc:
        raise FetchError(f"extracting {zip_path} into {dest} failed: {exc}") from exc
    return list(targets)


def _extract_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".part")
    tmp = Path(tmp_name)
    try:
        # zipfile checks each member's CRC32 as it reads, so a damaged zip raises here.
        with os.fdopen(fd, "wb") as out, archive.open(info) as member:
            shutil.copyfileobj(member, out, _CHUNK)
        tmp.chmod(0o644)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def fetch(
    data_dir: Path,
    *,
    zip_path: Path | None = None,
    source: SourceName = "hf",
    force: bool = False,
    download_dir: Path | None = None,
) -> FetchResult:
    """Put every file listed in ``spider.sha256`` into ``data_dir`` and verify it.

    A no-op when they all match already, unless ``force``. Missing files are filled in,
    but a file that is there and differs is replaced only with ``force``. The zip comes
    from ``zip_path`` or is downloaded into ``download_dir`` (default:
    ``default_download_dir()``), reusing a valid copy already there.
    """
    expected = load_checksums()
    if not force:
        problems = check_files(data_dir, expected)
        if not problems:
            return FetchResult(data_dir, len(expected), None)
        differing = sorted(rel for rel in problems if (data_dir / rel).exists())
        if differing:
            shown = ", ".join(differing[:5]) + (", ..." if len(differing) > 5 else "")
            raise FetchError(
                f"{len(differing)} file(s) in {data_dir} differ from spider.sha256 ({shown}). "
                "Fetch only adds missing files: pass --force to replace these, or if this is "
                "other data, such as the tests/fixtures/spider_mini subset, point "
                "FORGE_DATA_DIR at the Spider download instead"
            )
    if zip_path is None:
        zip_path = download(source, download_dir or default_download_dir())
    else:
        verify_zip(zip_path)
    extract(zip_path, data_dir, expected)
    verify_files(data_dir, expected)
    return FetchResult(data_dir, len(expected), zip_path)
