"""The run store: where a run's files live once the machine that made them is gone.

A Kaggle session lasts 12 hours at most, so every run keeps its files in a store off the
machine. ``LocalRunStore`` is a folder such as ``runs/``, and holds plan section 3.9's
layout, one folder a run:

    runs/<run_id>/manifest.json, config.yaml, metrics.jsonl, eval/..., samples/...
    runs/<run_id>/checkpoints/step-NNNNNN/   the latest KEEP_CHECKPOINTS are kept
    runs/<run_id>/checkpoints/best/

In a local store the folder is ``<root>/<run_id>/``. Paths inside a run are POSIX paths
relative to its folder, such as ``eval/val-scores.jsonl``, in every store.

``RunStore`` is plan section 3.5's protocol with three methods more: ``put_folder``, since
a checkpoint goes up as a folder; ``runs``, for ``forge runs list``; and
``prune_checkpoints``, the retention rule, which keeps the latest checkpoints and
``best/`` and deletes the older ``step-NNNNNN`` folders. ``put_folder`` adds to what the
store holds, or with ``replace`` makes it hold the folder's files and no others, as a run
that ended must: an earlier attempt's ``error.txt`` would otherwise outlive it.

``pull`` mirrors a run into a local folder as ``forge kaggle output`` does: each file or
folder at the top of the run that the store holds replaces the local one, and what only
the local folder holds, such as ``kernels.json``, stays. An eval run's lock file never
leaves the machine it was taken on.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from forge.core.types import RunManifest
from forge.eval.config import check_run_id
from forge.eval.run_files import LOCK_FILE, MANIFEST_FILE, RunCorrupt, write_atomic

CHECKPOINTS = "checkpoints"
BEST = f"{CHECKPOINTS}/best"
KEEP_CHECKPOINTS = 2
# The names checkpoint_dir writes: six digits, or more without a leading zero, so no two
# names stand for one step.
_STEP = re.compile(r"step-(\d{6}|[1-9]\d{6,})")


class RunNotFound(FileNotFoundError):
    """The store holds no run of that id."""


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


def _mirror(source: Path, target: Path) -> None:
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
        if item.is_dir():
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
            _mirror(source, target)
