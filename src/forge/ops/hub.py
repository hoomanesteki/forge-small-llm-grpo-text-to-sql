"""The run store: where a run's files live once the machine that made them is gone.

A Kaggle session lasts 12 hours at most, so every run keeps its files in a store off the
machine. ``HubRunStore`` is the private Hugging Face model repo that ``FORGE_RUN_STORE``
names (``hoomanesteki/forge-runs`` by default), and ``LocalRunStore`` a folder such as
``runs/``. Both hold plan section 3.9's layout, one folder a run:

    runs/<run_id>/manifest.json, config.yaml, metrics.jsonl, eval/..., samples/...
    runs/<run_id>/checkpoints/step-NNNNNN/   the latest KEEP_CHECKPOINTS are kept
    runs/<run_id>/checkpoints/best/

In a local store the folder is ``<root>/<run_id>/``. Paths inside a run are POSIX paths
relative to its folder, such as ``eval/val-scores.jsonl``, in either store.

``RunStore`` is plan section 3.5's protocol with three methods more: ``put_folder``, since
a checkpoint goes up as a folder; ``runs``, for ``forge runs list``; and
``prune_checkpoints``, the retention rule, which keeps the latest checkpoints and
``best/`` and deletes the older ``step-NNNNNN`` folders. ``put_folder`` adds to what the
store holds, or with ``replace`` makes it hold the folder's files and no others, as a run
that ended must: an earlier attempt's ``error.txt`` would otherwise outlive it.

A Hub call that fails on the network is tried again ``RETRIES`` times, after
``BACKOFF_S`` seconds and then twice as long each time (``network_error`` says which
failures count). Anything else, such as a missing repo or a refused token, fails at once.

``pull`` mirrors a run into a local folder as ``forge kaggle output`` does: each file or
folder at the top of the run that the store holds replaces the local one, and what only
the local folder holds, such as ``kernels.json``, stays. An eval run's lock file never
leaves the machine it was taken on.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
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
BEST = f"{CHECKPOINTS}/best"
KEEP_CHECKPOINTS = 2
RETRIES = 3
BACKOFF_S = 2.0
# Request timeout and rate limiting: the Hub's answer may differ a moment later.
RETRY_STATUS = frozenset({408, 429})
# The names checkpoint_dir writes: six digits, or more without a leading zero, so no two
# names stand for one step.
_STEP = re.compile(r"step-(\d{6}|[1-9]\d{6,})")


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
    def latest_checkpoint(self, run_id: str) -> str | None: ...
    def prune_checkpoints(self, run_id: str, keep: int = KEEP_CHECKPOINTS) -> list[str]: ...
    def runs(self) -> list[str]: ...
    def pull(self, run_id: str, local_dir: str | Path) -> None: ...


def checkpoint_dir(step: int) -> str:
    """Where in a run the checkpoint saved at ``step`` goes: ``checkpoints/step-000050``."""
    if step < 0:
        raise ValueError(f"a checkpoint's step can't be negative, got {step}")
    return f"{CHECKPOINTS}/step-{step:06d}"


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


def _to_prune(names: Iterable[str], keep: int) -> list[str]:
    """The step checkpoints retention deletes, oldest first; never ``best/``."""
    if keep < 1:
        raise ValueError(f"keep must be at least 1, so the latest checkpoint stays; got {keep}")
    steps = _steps(names)
    return [steps[step] for step in sorted(steps)[:-keep]]


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


def _mirror(source: Path, target: Path, *, move: bool) -> None:
    """Replace each top-level item of ``target`` that ``source`` holds, lock file aside."""
    target.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.iterdir()):
        if item.name == LOCK_FILE:
            continue
        there = target / item.name
        if there.is_dir() and not there.is_symlink():
            shutil.rmtree(there)
        elif there.exists() or there.is_symlink():
            there.unlink()
        if move:
            shutil.move(item, there)
        elif item.is_dir():
            shutil.copytree(item, there, ignore=shutil.ignore_patterns(LOCK_FILE))
        else:
            shutil.copy2(item, there)


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

    def prune_checkpoints(self, run_id: str, keep: int = KEEP_CHECKPOINTS) -> list[str]:
        """Delete all but the latest ``keep`` step checkpoints; returns what went."""
        doomed = _to_prune(self._checkpoints(run_id), keep)
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
            _mirror(source, target, move=False)


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
    """Whether a Hub call that raised ``exc`` is worth trying again.

    A connection the client's own retries gave up on raises an httpx2 transport error or
    a ``ConnectionError``; the Hub answers 408, 429 or 5xx when the trouble is on its
    side; and huggingface_hub reports a failed upload or Xet transfer as a bare
    ``RuntimeError``. Offline mode, missing repos and files, and refused tokens are not.
    """
    import httpx2
    from huggingface_hub.errors import HfHubHTTPError, OfflineModeIsEnabled

    if isinstance(exc, OfflineModeIsEnabled):
        return False
    if isinstance(exc, HfHubHTTPError):
        status = exc.response.status_code
        return status >= 500 or status in RETRY_STATUS
    return isinstance(exc, httpx2.TransportError | ConnectionError | TimeoutError | RuntimeError)


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
                if not network_error(exc):
                    raise
                if attempt == RETRIES:
                    raise StoreUnavailable(
                        f"{what} failed {RETRIES + 1} times; the last time "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                wait = BACKOFF_S * 2**attempt
                log.warning("%s: %s: %s; trying again in %g s", what, type(exc).__name__, exc, wait)
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
        from huggingface_hub.errors import EntryNotFoundError

        path = self._path(run_id, MANIFEST_FILE)
        # A folder of its own rather than the Hub cache, which would keep every copy.
        with tempfile.TemporaryDirectory(prefix="forge-manifest-") as folder:
            try:
                self._call(
                    self.api.hf_hub_download,
                    f"downloading {self.repo_id}/{path}",
                    filename=path,
                    local_dir=folder,
                )
            except EntryNotFoundError as exc:
                raise RunNotFound(f"no run {run_id} in {self.repo_id}: it has no {path}") from exc
            data = Path(folder, path).read_bytes()
        return _read_manifest(data, run_id, f"{self.repo_id}/{path}")

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

    def prune_checkpoints(self, run_id: str, keep: int = KEEP_CHECKPOINTS) -> list[str]:
        """Delete all but the latest ``keep`` step checkpoints, in one commit; returns what
        went."""
        doomed = _to_prune(self._checkpoints(run_id), keep)
        if doomed:
            self._call(
                self.api.delete_files,
                f"deleting old checkpoints of run {run_id} from {self.repo_id}",
                delete_patterns=[f"{self._path(run_id, rel)}/*" for rel in doomed],
                commit_message=f"Keep the latest {keep} checkpoints of run {run_id}",
            )
        return doomed

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
            self._call(
                self.api.snapshot_download,
                f"downloading {self.repo_id}/{path}",
                local_dir=staging,
                allow_patterns=[f"{path}/*"],
            )
            fetched = Path(staging, path)
            if not fetched.is_dir():
                raise RunNotFound(f"no run {run_id} in {self.repo_id}")
            _mirror(fetched, target, move=True)


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
