"""The run store: where a run's files live once the machine that made them is gone.

A Kaggle session lasts 12 hours at most, so every run keeps its files in a store off the
machine. ``HubRunStore`` is the private Hugging Face model repo that ``FORGE_RUN_STORE``
names (``hoomanesteki/forge-runs`` by default), and ``LocalRunStore`` a folder such as
``runs/``. Both hold plan section 3.9's layout, one folder a run:

    runs/<run_id>/manifest.json, config.yaml, metrics.jsonl, eval/..., samples/...
    runs/<run_id>/checkpoints/step-NNNNNN/   the latest KEEP_CHECKPOINTS are kept
    runs/<run_id>/checkpoints/final/         the adapter a training run ended with
    runs/<run_id>/checkpoints/best/

In a local store the folder is ``<root>/<run_id>/``. Paths inside a run are POSIX paths
relative to its folder, such as ``eval/val-scores.jsonl``, in either store.

``RunStore`` is plan section 3.5's protocol with four methods more: ``put_folder``, since
a checkpoint goes up as a folder; ``read_file``, one file of a run; ``runs``, for
``forge runs list``; and ``prune_checkpoints``, the retention rule, which keeps the latest
checkpoints, ``best/`` and the steps a job names to keep, such as one half way through a
run for val to choose from (FG-404), and deletes the other ``step-NNNNNN`` folders.
``put_folder`` adds to what the store holds, or with ``replace`` makes it hold the
folder's files and no others, as a run that ended must: an earlier attempt's
``error.txt`` would otherwise outlive it.

A Hub call that fails on the network (``network_error`` says which failures count) is
tried again by one layer, so the waits don't multiply, but for one rare case below.
huggingface_hub 2.4 already tries some failures of most requests 5 more times, 1, 2, 4, 8
and 8 s apart, and a failure that comes through those retries is ``StoreUnavailable`` at
once. ``client_retried`` tells which from the request that failed and how. The client
never retries a commit or the repo lookup a snapshot download starts with, nor a 5xx
other than 500, 502, 503 and 504. It retries a download's first request, for the file's
metadata, only after a refused connection, a timeout or one of its statuses (408, 429,
500, 502, 503 and 504); the download itself only after a refused connection, a timeout, a
broken answer (``RemoteProtocolError``), 408 or 429; and any other request after any
network error or one of its statuses. Any other failure the store tries again, the whole
call ``RETRIES`` more times, after ``BACKOFF_S`` seconds and then twice as long each time,
2, 4 and 8 s: a dropped connection on a download, for one. So a call that keeps failing
waits at most 14 s between the store's tries, and at most 23 s inside the client's
retries of one request, unless the Hub's answer says how long to wait (``Retry-After``,
or a 429's ``RateLimit`` header): the client then waits that long and 1 s more before
each of its 5 tries, so 305 s for a ``Retry-After: 60``. ``client_retried`` judges by
the failure that ended the call, so when a download's first metadata request is refused,
times out or gets one of the client's statuses, and the client's retries of it then drop
their connections, both layers retry: each of the store's 4 tries waits out the client's
23 s, 106 s of waiting in all with the store's 14 s. A missing run is ``RunNotFound``, and
anything else, such as a missing repo or a refused token, fails at once.

``pull`` makes a local folder hold the store's copy of a run, as ``forge kaggle output``
does with a kernel's (``replace_run``): what the copy lacks goes, such as a failed
attempt's ``error.txt`` beside the manifest of the attempt that finished the run. What
only a machine keeps stays (``LOCAL_ONLY``): an eval run's lock file, which never leaves
the machine it was taken on, and ``kernels.json``, the kernels ``forge kaggle run``
pushed for the run, with ``kernel/``, the rest of a kernel's output, unless the copy's
``kernels.json`` lists more pushes than this machine's: ``forge runs push`` uploads them,
so the store may hold an older record. ``HubRunStore.get_folder`` fetches one folder of a
run, such as the checkpoint a kernel's eval job loads as its adapter.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Collection, Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import ValidationError

from forge.core.types import RunManifest
from forge.eval.config import check_run_id
from forge.eval.run_files import LOCK_FILE, MANIFEST_FILE, RunCorrupt, write_atomic

log = logging.getLogger(__name__)

T = TypeVar("T")

REPO_TYPE = "model"
RUNS = "runs"
CHECKPOINTS = "checkpoints"
FINAL = f"{CHECKPOINTS}/final"
BEST = f"{CHECKPOINTS}/best"
METRICS_FILE = "metrics.jsonl"
KEEP_CHECKPOINTS = 2
RETRIES = 3
BACKOFF_S = 2.0
# Request timeout and rate limiting: the Hub's answer may differ a moment later.
RETRY_STATUS = frozenset({408, 429})
# The statuses huggingface_hub 2.4's http_backoff retries (_DEFAULT_RETRY_ON_STATUS_CODES).
CLIENT_RETRY_STATUS = RETRY_STATUS | {500, 502, 503, 504}
# The paths of the two requests huggingface_hub 2.4 sends without retrying them: the
# repo lookup (``/api/models/<repo>``, with ``/revision/<rev>`` from snapshot_download)
# and a commit (``/commit/<rev>``).
_SENT_ONCE = re.compile(r"/api/models/[^/]+/[^/]+(?:/revision/[^/]+|/commit/[^/]+)?$")
# A Hub API request; a download's own GET, of the file or a CDN copy, is not one.
_API = re.compile(r"/api/")
# The names checkpoint_dir writes: six digits, or more without a leading zero, so no two
# names stand for one step.
_STEP = re.compile(r"step-(\d{6}|[1-9]\d{6,})")
KERNELS_FILE = "kernels.json"  # forge kaggle run's record of the run's kernels
KERNEL_DIR = "kernel"  # where forge kaggle output puts what a kernel wrote beside the run
LOCAL_ONLY = frozenset({LOCK_FILE, KERNELS_FILE, KERNEL_DIR})


class RunNotFound(FileNotFoundError):
    """The store holds no run of that id."""


class StoreUnavailable(OSError):
    """A Hub call still failed on the network after every retry."""


class RunStore(Protocol):
    """Where runs' files are kept: plan section 3.5's five methods, and the three this
    module's docstring describes. Paths inside a run are relative POSIX paths."""

    def put_file(self, run_id: str, local_path: str | Path, dest: str) -> str: ...
    def put_folder(
        self, run_id: str, local_dir: str | Path, dest: str = "", *, replace: bool = False
    ) -> str: ...
    def put_manifest(self, manifest: RunManifest) -> None: ...
    def get_manifest(self, run_id: str) -> RunManifest: ...
    def read_file(self, run_id: str, dest: str) -> bytes | None: ...
    def latest_checkpoint(self, run_id: str) -> str | None: ...
    def prune_checkpoints(
        self, run_id: str, keep: int = KEEP_CHECKPOINTS, *, keep_steps: Collection[int] = ()
    ) -> list[str]: ...
    def runs(self) -> list[str]: ...
    def pull(self, run_id: str, local_dir: str | Path) -> None: ...


def checkpoint_dir(step: int) -> str:
    """Where in a run the checkpoint saved at ``step`` goes: ``checkpoints/step-000050``."""
    if step < 0:
        raise ValueError(f"a checkpoint's step can't be negative, got {step}")
    return f"{CHECKPOINTS}/step-{step:06d}"


def checkpoint_step(rel: str) -> int | None:
    """The step of the checkpoint ``rel`` names, a path in a run as ``checkpoint_dir`` writes
    one, or None for any other path."""
    folder, _, name = rel.partition("/")
    match = _STEP.fullmatch(name)
    return int(match.group(1)) if folder == CHECKPOINTS and match else None


def _steps(names: Iterable[str]) -> dict[int, str]:
    """The step checkpoints among folder names in ``checkpoints/``, by step."""
    found = {}
    for name in names:
        match = _STEP.fullmatch(name)
        if match:
            found[int(match.group(1))] = f"{CHECKPOINTS}/{name}"
    return found


def _latest(names: Iterable[str]) -> str | None:
    steps = _steps(names)
    return steps[max(steps)] if steps else None


def _to_prune(names: Iterable[str], keep: int, keep_steps: Collection[int]) -> list[str]:
    """The step checkpoints retention deletes, oldest first: never ``best/``, the latest
    ``keep`` or one of ``keep_steps``."""
    if keep < 1:
        raise ValueError(f"keep must be at least 1, so the latest checkpoint stays; got {keep}")
    steps = _steps(names)
    return [steps[step] for step in sorted(steps)[:-keep] if step not in keep_steps]


def _inside(dest: str, *, root_ok: bool = False) -> str:
    """``dest`` once it is checked to name a path inside a run's folder.

    An empty ``dest`` is the folder itself, which only ``root_ok`` allows.
    """
    if dest == "" and root_ok:
        return dest
    parts = dest.split("/")
    if "\\" in dest or any(part in ("", ".", "..") for part in parts):
        raise ValueError(
            f"{dest!r} must be a relative path inside the run's folder, such as "
            "eval/val-report.md, with no '.', '..' or empty parts"
        )
    return dest


def manifest_bytes(manifest: RunManifest) -> bytes:
    """A manifest as a run's manifest.json holds it, as forge.eval.run_files writes it."""
    return (manifest.model_dump_json(indent=2) + "\n").encode()


def _read_manifest(data: bytes, run_id: str, where: str) -> RunManifest:
    try:
        manifest = RunManifest.model_validate_json(data)
    except ValidationError as exc:
        raise RunCorrupt(f"{where} is not a run manifest: {exc}") from exc
    if manifest.run_id != run_id:
        raise RunCorrupt(f"{where} is the manifest of run {manifest.run_id}, not of {run_id}")
    return manifest


def _same(a: Path, b: Path) -> bool:
    return a.resolve() == b.resolve()


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _empty(folder: Path) -> None:
    for child in folder.iterdir():
        _remove(child)


def replace_run(source: Path, target: Path, *, move: bool) -> None:
    """Make the run folder ``target`` hold the copy of the run in ``source``.

    Each top-level item of ``source`` replaces ``target``'s of that name, and every other
    item of ``target`` goes, unless it is ``LOCAL_ONLY``. The lock file is never copied
    or touched, and nor are ``kernels.json`` and ``kernel/`` when ``target``'s record
    lists as many pushes as ``source``'s or more, so a pull never puts back an older one.
    ``move`` moves the items out of ``source`` instead of copying them.
    """
    target.mkdir(parents=True, exist_ok=True)
    ours, theirs = target / KERNELS_FILE, source / KERNELS_FILE
    keep_ours = ours.is_file() and (not theirs.is_file() or _pushes(ours) >= _pushes(theirs))
    kept = LOCAL_ONLY if keep_ours else frozenset({LOCK_FILE})
    incoming = sorted(item for item in source.iterdir() if item.name not in kept)
    names = {item.name for item in incoming}
    for there in sorted(target.iterdir()):
        if there.name in names or there.name not in LOCAL_ONLY:
            _remove(there)
    for item in incoming:
        there = target / item.name
        if move:
            shutil.move(item, there)
        elif item.is_dir():
            shutil.copytree(item, there, ignore=shutil.ignore_patterns(LOCK_FILE))
        else:
            shutil.copy2(item, there)


def _pushes(path: Path) -> int:
    """How many pushes the ``kernels.json`` at ``path`` lists. Each push adds one and none
    is ever taken out, so of two records of a run the longer is the newer."""
    try:
        kernels = json.loads(path.read_text(encoding="utf-8"))["kernels"]
        if not isinstance(kernels, list):
            raise TypeError(f"its kernels are {type(kernels).__name__}, not a list")
    except (KeyError, TypeError, ValueError) as exc:
        raise RunCorrupt(f"{path} is not forge kaggle run's record of kernels: {exc}") from exc
    return len(kernels)


class LocalRunStore:
    """A run store in a folder: run ``<run_id>`` lives in ``<root>/<run_id>/``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def __str__(self) -> str:
        return str(self.root)

    def _run(self, run_id: str) -> Path:
        check_run_id(run_id)
        return self.root / run_id

    def put_file(self, run_id: str, local_path: str | Path, dest: str) -> str:
        """Copy the file ``local_path`` to ``dest`` in the run; returns where it is now."""
        source, target = Path(local_path), self._run(run_id) / _inside(dest)
        if not source.is_file():
            raise FileNotFoundError(f"{source} is not a file, so it can't go to run {run_id}")
        if not _same(source, target):
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(f"{target.name}.tmp")
            shutil.copyfile(source, partial)
            os.replace(partial, target)
        return str(target)

    def put_folder(
        self, run_id: str, local_dir: str | Path, dest: str = "", *, replace: bool = False
    ) -> str:
        """Copy the folder ``local_dir`` into ``dest`` in the run, the run's own folder when
        ``dest`` is empty. Files there that it lacks stay, unless ``replace``. Returns where
        it is now."""
        source, target = Path(local_dir), self._run(run_id) / _inside(dest, root_ok=True)
        if not source.is_dir():
            raise NotADirectoryError(f"{source} is not a folder, so it can't go to run {run_id}")
        if not _same(source, target):
            if replace and target.is_dir():
                shutil.rmtree(target)
            ignore = shutil.ignore_patterns(LOCK_FILE)
            shutil.copytree(source, target, ignore=ignore, dirs_exist_ok=True)
        return str(target)

    def put_manifest(self, manifest: RunManifest) -> None:
        write_atomic(self._run(manifest.run_id) / MANIFEST_FILE, manifest_bytes(manifest))

    def get_manifest(self, run_id: str) -> RunManifest:
        path = self._run(run_id) / MANIFEST_FILE
        if not path.is_file():
            raise RunNotFound(f"no run {run_id} in {self.root}: it has no {MANIFEST_FILE}")
        return _read_manifest(path.read_bytes(), run_id, str(path))

    def read_file(self, run_id: str, dest: str) -> bytes | None:
        """The file ``dest`` of the run, or None when the run has no such file."""
        path = self._run(run_id) / _inside(dest)
        return path.read_bytes() if path.is_file() else None

    def _checkpoints(self, run_id: str) -> list[str]:
        """Checkpoint folders holding a file: the Hub has no empty folders to match."""
        folder = self._run(run_id) / CHECKPOINTS
        if not folder.is_dir():
            return []
        return [
            child.name
            for child in folder.iterdir()
            if child.is_dir() and any(path.is_file() for path in child.rglob("*"))
        ]

    def latest_checkpoint(self, run_id: str) -> str | None:
        """The run's highest ``checkpoints/step-NNNNNN``, or None before its first."""
        return _latest(self._checkpoints(run_id))

    def prune_checkpoints(
        self, run_id: str, keep: int = KEEP_CHECKPOINTS, *, keep_steps: Collection[int] = ()
    ) -> list[str]:
        """Delete the step checkpoints but the latest ``keep`` and those of ``keep_steps``;
        returns what went."""
        doomed = _to_prune(self._checkpoints(run_id), keep, keep_steps)
        for rel in doomed:
            shutil.rmtree(self._run(run_id) / rel)
        return doomed

    def runs(self) -> list[str]:
        """The ids of the runs here, the folders with a manifest, sorted (by date first)."""
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if (p / MANIFEST_FILE).is_file())

    def pull(self, run_id: str, local_dir: str | Path) -> None:
        """Mirror the run into ``local_dir``, as the module docstring describes."""
        source, target = self._run(run_id), Path(local_dir)
        if not source.is_dir():
            raise RunNotFound(f"no run {run_id} in {self.root}")
        if not _same(source, target):
            replace_run(source, target, move=False)


class HubApi(Protocol):
    """The part of ``huggingface_hub.HfApi`` the Hub run store calls."""

    def upload_file(
        self,
        *,
        path_or_fileobj: str | Path | bytes,
        path_in_repo: str,
        repo_id: str,
        repo_type: str | None = None,
        commit_message: str | None = None,
    ) -> object: ...

    def upload_folder(
        self,
        *,
        repo_id: str,
        folder_path: str | Path,
        path_in_repo: str | None = None,
        commit_message: str | None = None,
        repo_type: str | None = None,
        ignore_patterns: list[str] | str | None = None,
        delete_patterns: list[str] | str | None = None,
    ) -> object: ...

    def list_repo_files(self, repo_id: str, *, repo_type: str | None = None) -> list[str]: ...

    def delete_files(
        self,
        repo_id: str,
        delete_patterns: list[str],
        *,
        repo_type: str | None = None,
        commit_message: str | None = None,
    ) -> object: ...

    def hf_hub_download(
        self,
        repo_id: str,
        filename: str,
        *,
        repo_type: str | None = None,
        local_dir: str | Path | None = None,
    ) -> object: ...

    def snapshot_download(
        self,
        repo_id: str,
        *,
        repo_type: str | None = None,
        local_dir: str | Path | None = None,
        allow_patterns: list[str] | str | None = None,
    ) -> object: ...


def network_error(exc: BaseException) -> bool:
    """Whether a Hub call that raised ``exc`` failed on the network, so it may work a
    moment later.

    That is a dropped or refused connection, a timeout, or the Hub answering 408, 429 or
    5xx, raised as itself or as the cause of what huggingface_hub wraps it in: a
    ``LocalEntryNotFoundError`` for a download, a bare ``RuntimeError`` for an LFS
    upload. Offline mode, missing repos and files, refused tokens, any other
    ``RuntimeError`` (a 403 the LFS upload wrapped, ``NotImplementedError``) and a
    misconfigured proxy are not.
    """
    return _network_cause(exc) is not None


def _network_cause(exc: BaseException) -> BaseException | None:
    """The network failure ``exc`` is or wraps, as ``network_error`` describes; None for
    any other error."""
    import httpx2
    from huggingface_hub.errors import (
        HfHubHTTPError,
        LocalEntryNotFoundError,
        OfflineModeIsEnabled,
    )

    if isinstance(exc, OfflineModeIsEnabled):
        return None
    if isinstance(exc, HfHubHTTPError):
        status = exc.response.status_code
        return exc if status >= 500 or status in RETRY_STATUS else None
    if isinstance(exc, LocalEntryNotFoundError) or type(exc) is RuntimeError:
        return None if exc.__cause__ is None else _network_cause(exc.__cause__)
    # The errors huggingface_hub's own retries take as passing, and the builtin ones a
    # socket raises.
    passing = (
        httpx2.NetworkError,
        httpx2.TimeoutException,
        httpx2.RemoteProtocolError,
        ConnectionError,
        TimeoutError,
    )
    return exc if isinstance(exc, passing) else None


def client_retried(exc: BaseException) -> bool:
    """Whether huggingface_hub 2.4 already tried again the request whose failure ``exc``
    is, by what it retries for that request, as the module docstring says.

    It decides by the failure that ended the call: in the rare call whose first metadata
    request is refused, times out or gets one of the client's statuses (408, 429, 500,
    502, 503 or 504), and whose retries then drop their connections, both layers retry.
    """
    import httpx2
    from huggingface_hub.errors import HfHubHTTPError

    status = None
    try:
        if isinstance(exc, HfHubHTTPError):
            request, status = exc.response.request, exc.response.status_code
        elif isinstance(exc, httpx2.RequestError):
            request = exc.request
        else:
            return False
    # httpx2's answer when the error was made without its request: one the store can't
    # place, so it retries the call itself.
    except RuntimeError:
        return False
    path = request.url.path
    errors: tuple[type[Exception], ...]
    if _SENT_ONCE.search(path):
        return False
    if request.method == "HEAD":
        # hf_hub_download's metadata request goes once, and again with retries only after
        # these (file_download._get_metadata_or_catch_error).
        errors, statuses = (httpx2.ConnectError, httpx2.TimeoutException), CLIENT_RETRY_STATUS
    elif request.method == "GET" and not _API.search(path):
        # The download itself, which http_get retries on its own terms.
        errors = (httpx2.ConnectError, httpx2.TimeoutException, httpx2.RemoteProtocolError)
        statuses = RETRY_STATUS
    else:
        # Everything else goes through http_backoff, with its defaults.
        errors = (httpx2.TimeoutException, httpx2.NetworkError, httpx2.RemoteProtocolError)
        statuses = CLIENT_RETRY_STATUS
    return status in statuses if status is not None else isinstance(exc, errors)


class HubRunStore:
    """A run store in a private Hugging Face model repo: run ``<run_id>`` lives in
    ``runs/<run_id>/``.

    ``api`` stands in for ``HfApi(token=token)``. Without a token the client finds one as
    it always does (``HF_TOKEN``, or a saved login). ``sleep`` waits between retries.
    """

    def __init__(
        self,
        repo_id: str,
        token: str | None = None,
        api: HubApi | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if api is None:
            # Imported here: only a Hub store needs the client.
            from huggingface_hub import HfApi

            api = HfApi(token=token)
        self.repo_id = repo_id
        self.api = api
        self.sleep = sleep

    def __str__(self) -> str:
        return self.repo_id

    def _path(self, run_id: str, dest: str = "") -> str:
        check_run_id(run_id)
        return f"{RUNS}/{run_id}/{dest}" if dest else f"{RUNS}/{run_id}"

    def _call(self, method: Callable[..., T], what: str, **kwargs: Any) -> T:
        """``method`` on this repo, tried again on network errors as the module says."""
        attempt = 0
        while True:
            try:
                return method(repo_id=self.repo_id, repo_type=REPO_TYPE, **kwargs)
            except Exception as exc:
                cause = _network_cause(exc)
                if cause is None:
                    raise
                last = f"{type(cause).__name__}: {cause}"
                if client_retried(cause):
                    raise StoreUnavailable(
                        f"{what} failed on the network after huggingface_hub's own retries: {last}"
                    ) from exc
                if attempt == RETRIES:
                    raise StoreUnavailable(
                        f"{what} failed {RETRIES + 1} times; the last time {last}"
                    ) from exc
                wait = BACKOFF_S * 2**attempt
                log.warning("%s: %s; trying again in %g s", what, last, wait)
                self.sleep(wait)
                attempt += 1

    def put_file(self, run_id: str, local_path: str | Path, dest: str) -> str:
        """Upload the file ``local_path`` to ``dest`` in the run; returns its path in the repo."""
        source, path = Path(local_path), self._path(run_id, _inside(dest))
        if not source.is_file():
            raise FileNotFoundError(f"{source} is not a file, so it can't go to run {run_id}")
        self._call(
            self.api.upload_file,
            f"uploading {source} to {self.repo_id}/{path}",
            path_or_fileobj=str(source),
            path_in_repo=path,
            commit_message=f"Add {dest} to run {run_id}",
        )
        return path

    def put_folder(
        self, run_id: str, local_dir: str | Path, dest: str = "", *, replace: bool = False
    ) -> str:
        """Upload the folder ``local_dir`` into ``dest`` in the run, the run's own folder
        when ``dest`` is empty. Files there that it lacks stay, unless ``replace`` deletes
        them in the same commit. Returns its path in the repo.

        A folder of a few hundred files goes up in one commit; huggingface_hub splits a
        larger one, and the same call made again carries on where it stopped.
        """
        source, path = Path(local_dir), self._path(run_id, _inside(dest, root_ok=True))
        if not source.is_dir():
            raise NotADirectoryError(f"{source} is not a folder, so it can't go to run {run_id}")
        self._call(
            self.api.upload_folder,
            f"uploading {source} to {self.repo_id}/{path}",
            folder_path=str(source),
            path_in_repo=path,
            ignore_patterns=[LOCK_FILE],
            # Matched inside path_in_repo; a file being uploaded is never deleted.
            delete_patterns=["*"] if replace else None,
            commit_message=f"Add {dest} to run {run_id}" if dest else f"Add run {run_id}",
        )
        return path

    def put_manifest(self, manifest: RunManifest) -> None:
        path = self._path(manifest.run_id, MANIFEST_FILE)
        self._call(
            self.api.upload_file,
            f"uploading {self.repo_id}/{path}",
            path_or_fileobj=manifest_bytes(manifest),
            path_in_repo=path,
            commit_message=f"Write the {manifest.status} manifest of run {manifest.run_id}",
        )

    def get_manifest(self, run_id: str) -> RunManifest:
        """The run's manifest. Raises ``RunNotFound`` when the repo has none for it, and
        ``StoreUnavailable`` when the Hub can't be reached to say."""
        path = self._path(run_id, MANIFEST_FILE)
        data = self.read_file(run_id, MANIFEST_FILE)
        if data is None:
            raise RunNotFound(f"no run {run_id} in {self.repo_id}: it has no {path}")
        return _read_manifest(data, run_id, f"{self.repo_id}/{path}")

    def read_file(self, run_id: str, dest: str) -> bytes | None:
        """The file ``dest`` of the run, or None when the repo has no such file."""
        from huggingface_hub.errors import RemoteEntryNotFoundError

        path = self._path(run_id, _inside(dest))
        # A folder of its own rather than the Hub cache, which would keep every copy.
        with tempfile.TemporaryDirectory(prefix="forge-file-") as folder:
            try:
                self._call(
                    self.api.hf_hub_download,
                    f"downloading {self.repo_id}/{path}",
                    filename=path,
                    local_dir=folder,
                )
            # The Hub's 404 for the file. An outage is a LocalEntryNotFoundError instead,
            # which _call retries or turns into StoreUnavailable.
            except RemoteEntryNotFoundError:
                return None
            return Path(folder, path).read_bytes()

    def _files(self, run_id: str) -> list[str]:
        """Every file of the run, as a path inside its folder."""
        prefix = self._path(run_id) + "/"
        files = self._call(self.api.list_repo_files, f"listing the files of {self.repo_id}")
        return [name.removeprefix(prefix) for name in files if name.startswith(prefix)]

    def _checkpoints(self, run_id: str) -> set[str]:
        return {
            parts[1]
            for name in self._files(run_id)
            if len(parts := name.split("/")) > 2 and parts[0] == CHECKPOINTS
        }

    def latest_checkpoint(self, run_id: str) -> str | None:
        """The run's highest ``checkpoints/step-NNNNNN``, or None before its first."""
        return _latest(self._checkpoints(run_id))

    def prune_checkpoints(
        self, run_id: str, keep: int = KEEP_CHECKPOINTS, *, keep_steps: Collection[int] = ()
    ) -> list[str]:
        """Delete the step checkpoints but the latest ``keep`` and those of ``keep_steps``,
        in one commit; returns what went."""
        doomed = _to_prune(self._checkpoints(run_id), keep, keep_steps)
        if doomed:
            self._call(
                self.api.delete_files,
                f"deleting old checkpoints of run {run_id} from {self.repo_id}",
                delete_patterns=[f"{self._path(run_id, rel)}/*" for rel in doomed],
                commit_message=f"Keep the latest {keep} checkpoints of run {run_id}",
            )
        return doomed

    def get_folder(self, run_id: str, rel: str, local_dir: str | Path) -> Path:
        """Download the run's folder ``rel``, such as a checkpoint, to ``local_dir``, which
        it replaces; returns ``local_dir``."""
        path, target = self._path(run_id, _inside(rel)), Path(local_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Staged beside the target, as pull does, so a failed download leaves it as it was.
        with tempfile.TemporaryDirectory(prefix=".get-", dir=target.parent) as staging:
            fetched = self._download(path, staging)
            if not fetched.is_dir():
                raise RunNotFound(f"run {run_id} in {self.repo_id} has no {rel}")
            if target.is_dir():
                shutil.rmtree(target)
            shutil.move(fetched, target)
        return target

    def runs(self) -> list[str]:
        """The ids of the runs in the repo, those with a manifest, sorted (by date first)."""
        files = self._call(self.api.list_repo_files, f"listing the files of {self.repo_id}")
        return sorted(
            parts[1]
            for name in files
            if len(parts := name.split("/")) == 3 and parts[0] == RUNS and parts[2] == MANIFEST_FILE
        )

    def pull(self, run_id: str, local_dir: str | Path) -> None:
        """Mirror the run into ``local_dir``, as the module docstring describes."""
        path, target = self._path(run_id), Path(local_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Staged beside the target, so moving the files in is a rename, and nothing of a
        # download that fails reaches it.
        with tempfile.TemporaryDirectory(prefix=".pull-", dir=target.parent) as staging:
            fetched = self._download(path, staging)
            if not fetched.is_dir():
                raise RunNotFound(f"no run {run_id} in {self.repo_id}")
            replace_run(fetched, target, move=True)

    def _download(self, path: str, staging: str) -> Path:
        """Download the repo's folder ``path`` into the folder ``staging``, and return
        where it lands there, which is no folder when the repo has none at ``path``."""

        def download(**kwargs: Any) -> object:
            # snapshot_download that can't reach the Hub returns a folder that isn't
            # empty as if it were complete, so no try starts on what one before left.
            _empty(Path(staging))
            return self.api.snapshot_download(**kwargs)

        self._call(
            download,
            f"downloading {self.repo_id}/{path}",
            local_dir=staging,
            allow_patterns=[f"{path}/*"],
        )
        return Path(staging, path)


def hub_store(env: Mapping[str, str] | None = None) -> HubRunStore:
    """The Hub run store ``FORGE_RUN_STORE`` names, with ``HF_TOKEN`` when it is set, both
    read from the environment and the repo's ``.env``."""
    from forge.ops.doctor import DEFAULT_RUN_STORE
    from forge.ops.dotenv import forge_env

    values = forge_env() if env is None else env
    return HubRunStore(
        values.get("FORGE_RUN_STORE", DEFAULT_RUN_STORE), token=values.get("HF_TOKEN") or None
    )


def open_store(name: str) -> RunStore:
    """The store a command's ``--store`` names: ``hub``, or ``local`` for ``runs/``
    (``FORGE_RUNS_DIR``)."""
    from forge.ops.paths import runs_dir

    if name == "hub":
        return hub_store()
    if name == "local":
        return LocalRunStore(runs_dir())
    raise ValueError(f"a run store is hub or local, not {name!r}")


def describe(manifest: RunManifest, latest: str | None) -> list[str]:
    """What ``forge runs show`` prints of a run: its manifest, and its latest checkpoint."""
    ended = manifest.ended_at.isoformat() if manifest.ended_at else "not yet"
    lines = [
        f"run {manifest.run_id}: {manifest.kind}, {manifest.status}",
        f"model {manifest.model_ref}"
        + (f", adapter {manifest.adapter_ref}" if manifest.adapter_ref else ""),
        f"platform {manifest.platform}"
        + (f" on {manifest.gpu}" if manifest.gpu else "")
        + f", {manifest.gpu_hours:.2f} GPU h, ${manifest.cost_usd:.2f}",
        f"commit {manifest.git_sha}, seed {manifest.seed}",
        f"config hash {manifest.config_hash}, data manifest hash {manifest.data_manifest_hash}",
        f"started {manifest.started_at.isoformat()}, ended {ended}",
    ]
    if manifest.parent_run:
        lines.append(f"parent run {manifest.parent_run}")
    if manifest.outputs:
        lines.append("outputs:")
        lines += [f"  {name}: {path}" for name, path in sorted(manifest.outputs.items())]
    lines.append(f"latest checkpoint: {latest or 'none'}")
    return lines
