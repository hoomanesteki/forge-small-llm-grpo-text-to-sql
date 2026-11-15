"""An eval run's folder: its layout, its manifest's life, and outputs that survive a kill.

The layout is the run store's (plan section 3.9), under ``runs/<run_id>/``:
``manifest.json``, ``config.yaml``, ``eval/<split>-generations.jsonl`` and
``eval/<split>-scores.jsonl``, plus ``error.txt`` when the last attempt failed. A run
that retries failed queries (``forge.eval.retry``) keeps the first replies of the samples
it retried in ``eval/<split>-first-tries.jsonl``; their generations line holds the
second reply, the one scored (ADR-0015). The manifest's ``outputs`` lists whichever of
these exist, as paths inside the folder. Reports of a finished run
(``forge.eval.report``) add ``eval/<split>-report.md`` and its chart, which the finished
manifest doesn't list.

``start`` writes the manifest with status ``running`` before any work, and ``finish``
writes it again at the end: ``finished``, or ``failed`` with the traceback in
``error.txt``. A run killed outright stays ``running``, and its id can be resumed: the
same run with the same config, commit and data. A finished run, one made with other
options, code or data, or one of another kind is refused, and ``hold`` keeps a second
process off a run while one works on it.

``RunFiles`` treats the scores file as the record of what is done. Each batch's first
tries are appended before its generations, and those before its scores, every line
flushed to disk, so a kill can leave first tries or generations without scores, or a
last line cut short, but never a score without its generation, nor a retried score
without its first try. On opening, those leftovers are dropped (their questions run
again) and the files rewritten. Anything else out of place, such as a line twice or a
first try for a score that took one reply, raises ``RunCorrupt``.

``read_run`` reads a finished run back for reports and changes nothing. A finished run
has none of those leftovers, so there each is ``RunCorrupt`` too.
"""

from __future__ import annotations

import fcntl
import logging
import os
import traceback
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from forge.core.types import Generation, RunManifest, Score, Split
from forge.data.splits import SPLITS
from forge.ops.manifest import config_hash

log = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)
Key = tuple[str, int]  # (example_id, sample_idx)

LOCK_FILE = ".lock"
MANIFEST_FILE = "manifest.json"

# How read_run says what an unfinished run's status means.
_UNFINISHED = {
    "running": "is still running, or was stopped before it could say so",
    "failed": "failed (its error.txt says why)",
    "stopped_time_guard": "was stopped by the time guard",
}


class RunConflict(RuntimeError):
    """The run id names a run that this one can't resume."""


class RunLocked(RuntimeError):
    """Another process is working on the run."""


class RunCorrupt(ValueError):
    """A run's files can't be read, or contradict each other."""


class RunUnfinished(RuntimeError):
    """The run hasn't finished, so its files don't hold all of its results yet."""


@dataclass(frozen=True)
class RunPaths:
    """Where one eval run's files go."""

    root: Path
    split: Split

    @property
    def manifest(self) -> Path:
        return self.root / MANIFEST_FILE

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def generations(self) -> Path:
        return self.root / "eval" / f"{self.split}-generations.jsonl"

    @property
    def scores(self) -> Path:
        return self.root / "eval" / f"{self.split}-scores.jsonl"

    @property
    def first_tries(self) -> Path:
        return self.root / "eval" / f"{self.split}-first-tries.jsonl"

    @property
    def error(self) -> Path:
        return self.root / "error.txt"

    @property
    def report(self) -> Path:
        return self.root / "eval" / f"{self.split}-report.md"

    @property
    def chart(self) -> Path:
        return self.root / "eval" / f"{self.split}-hardness.png"

    def outputs(self) -> dict[str, str]:
        """The run's files that exist, by logical name, as paths inside the run folder."""
        files = {
            "config": self.config,
            "generations": self.generations,
            "scores": self.scores,
            "first_tries": self.first_tries,
            "error": self.error,
        }
        return {
            name: path.relative_to(self.root).as_posix()
            for name, path in files.items()
            if path.is_file()
        }


def key(item: Generation | Score) -> Key:
    return (item.example_id, item.sample_idx)


@contextmanager
def hold(paths: RunPaths) -> Iterator[None]:
    """Keep the run to this process; the lock is gone with the process, however it ends."""
    paths.root.mkdir(parents=True, exist_ok=True)
    with (paths.root / LOCK_FILE).open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RunLocked(f"run {paths.root.name} is running in another process") from None
        yield


def start(paths: RunPaths, fresh: RunManifest) -> RunManifest:
    """Write the manifest as running: ``fresh``, or the stored one when this resumes it.

    A resumed run keeps its first start time. Raises ``RunConflict`` when the stored run
    can't be resumed by ``fresh``, and ``RunCorrupt`` for outputs with no manifest.
    """
    if paths.manifest.is_file():
        try:
            previous = RunManifest.model_validate_json(paths.manifest.read_bytes())
        except ValidationError as exc:
            raise RunCorrupt(f"{paths.manifest} is not a run manifest: {exc}") from exc
        check_resumable(previous, fresh)
        manifest = previous.model_copy(update={"status": "running", "ended_at": None})
    elif any(path.exists() for path in (paths.scores, paths.generations, paths.first_tries)):
        raise RunCorrupt(f"{paths.root} has outputs but no manifest to say what made them")
    else:
        manifest = fresh
    write_atomic(paths.config, yaml.safe_dump(manifest.config, sort_keys=False).encode())
    manifest = manifest.model_copy(update={"outputs": paths.outputs()})
    _write_manifest(paths, manifest)
    return manifest


def check_resumable(previous: RunManifest, fresh: RunManifest) -> None:
    """Raise ``RunConflict`` unless ``fresh`` describes the same unfinished eval run."""
    run = previous.run_id
    if previous.kind != "eval":
        raise RunConflict(f"run {run} is a {previous.kind} run; give the eval its own --run-id")
    if previous.status == "finished":
        raise RunConflict(
            f"run {run} has finished and its results stand; pass a new --run-id to evaluate again"
        )
    if previous.config_hash != fresh.config_hash:
        changed = sorted(
            name
            for name in previous.config.keys() | fresh.config.keys()
            if previous.config.get(name) != fresh.config.get(name)
        )
        raise RunConflict(
            f"run {run} was made with other options ({', '.join(changed)}); resume it with "
            "the same ones, or pass a new --run-id"
        )
    if previous.git_sha != fresh.git_sha:
        raise RunConflict(
            f"run {run} was made at commit {previous.git_sha}, not {fresh.git_sha}; resume "
            "it from that commit, or pass a new --run-id"
        )
    if previous.data_manifest_hash != fresh.data_manifest_hash:
        raise RunConflict(
            f"run {run} was made on other data (data manifest hash "
            f"{previous.data_manifest_hash[:12]}, not {fresh.data_manifest_hash[:12]}); "
            "pass a new --run-id"
        )


def finish(paths: RunPaths, manifest: RunManifest, error: BaseException | None) -> RunManifest:
    """Write the final manifest, with the traceback beside it when the run failed."""
    if error is None:
        # An earlier attempt's failure no longer describes the run.
        paths.error.unlink(missing_ok=True)
    else:
        write_atomic(paths.error, "".join(traceback.format_exception(error)).encode())
    final = manifest.model_copy(
        update={
            "status": "finished" if error is None else "failed",
            "ended_at": datetime.now(UTC),
            "outputs": paths.outputs(),
        }
    )
    _write_manifest(paths, final)
    return final


class RunFiles:
    """A run's generations, scores and first tries, with what earlier attempts finished.

    ``expected`` holds every (example id, sample) the run asks for; a stored score for
    anything else means the files belong to another run.
    """

    def __init__(self, paths: RunPaths, expected: Collection[Key]) -> None:
        self.paths = paths
        self.expected = expected
        scores = _read_lines(paths.scores, Score)
        generations = _read_lines(paths.generations, Generation)
        first_tries = _read_lines(paths.first_tries, Generation)
        self.done = _paired(paths, generations, scores, expected)
        unscored = len(generations) - len(self.done)
        if unscored:
            log.info("dropping %d generations stored without a score; they run again", unscored)
        self.first_tries = _first_tries(paths, first_tries, scores, whole=False)
        if len(first_tries) > len(self.first_tries):
            log.info(
                "dropping %d first tries stored without a score; they run again",
                len(first_tries) - len(self.first_tries),
            )
        self.scores = scores
        self.generations = {key(g): g for g in generations if key(g) in self.done}
        self.resumed = len(scores)
        if paths.scores.exists() or paths.generations.exists():
            _write_lines(paths.generations, list(self.generations.values()))
            _write_lines(paths.scores, self.scores)
        if paths.first_tries.exists():
            _write_lines(paths.first_tries, list(self.first_tries.values()))

    def append(
        self,
        generations: Sequence[Generation],
        scores: Sequence[Score],
        first_tries: Sequence[Generation] = (),
    ) -> None:
        """Store one batch: the first tries of the samples it retried, then its
        generations, then its scores, so a kill between them leaves what the next attempt
        drops, never a score without its generation or a retried one without its first try.

        Raises ``ValueError`` unless ``first_tries`` are those of the retried scores, in
        order, before anything is written.
        """
        retried = [key(score) for score in scores if score.attempts > 1]
        if [key(first) for first in first_tries] != retried:
            raise ValueError("a batch's first tries must be its retried scores' ones, in order")
        if first_tries:
            _append_lines(self.paths.first_tries, first_tries)
        _append_lines(self.paths.generations, generations)
        _append_lines(self.paths.scores, scores)
        self.first_tries.update((key(first), first) for first in first_tries)
        self.generations.update((key(g), g) for g in generations)
        self.scores.extend(scores)
        self.done.update(key(score) for score in scores)

    def replace_scores(self, scores: Sequence[Score]) -> None:
        """Rewrite the scores in one step, the same ones in the same order with new values."""
        if [key(score) for score in scores] != [key(score) for score in self.scores]:
            raise ValueError("replacement scores must be for the same samples, in order")
        _write_lines(self.paths.scores, scores)
        self.scores = list(scores)


@dataclass(frozen=True)
class StoredRun:
    """A finished eval run as its folder holds it: what reports are made from."""

    manifest: RunManifest
    paths: RunPaths
    scores: tuple[Score, ...]  # in file order
    generations: Mapping[Key, Generation]  # the reply each score is for
    first_tries: Mapping[Key, Generation]  # the first reply of each retried score


def read_run(root: Path) -> StoredRun:
    """The finished eval run in ``root``, read without changing a file.

    Raises ``FileNotFoundError`` when ``root`` has no manifest, ``RunUnfinished`` unless
    the run finished, ``ValueError`` for a run of another kind, and ``RunCorrupt`` for
    files that can't be read, a config that doesn't hash to the manifest's
    ``config_hash``, scores and generations that don't pair up one to one, or first tries
    that don't pair up one to one with the retried scores.
    """
    path = root / MANIFEST_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no run in {root}: it has no {MANIFEST_FILE}")
    try:
        manifest = RunManifest.model_validate_json(path.read_bytes())
    except ValidationError as exc:
        raise RunCorrupt(f"{path} is not a run manifest: {exc}") from exc
    run = manifest.run_id
    if run != root.name:
        raise RunCorrupt(f"{path} is the manifest of run {run}, not of {root.name}")
    if manifest.kind != "eval":
        raise ValueError(f"run {run} is a {manifest.kind} run, not an eval run")
    if manifest.status != "finished":
        raise RunUnfinished(
            f"run {run} {_UNFINISHED[manifest.status]}, so its scores are not all in; "
            f"resume it with `forge eval run --run-id {run}` and the options it started with"
        )
    # Reports show the config, so one edited by hand must not pass for what the run used.
    if config_hash(manifest.config) != manifest.config_hash:
        raise RunCorrupt(f"{path}: its config doesn't hash to its config_hash")
    split = next((name for name in SPLITS if name == manifest.config.get("split")), None)
    if split is None:
        raise RunCorrupt(f"{path}: its config names no split, so its files can't be found")
    paths = RunPaths(root, split)
    for output in (paths.scores, paths.generations):
        if not output.is_file():
            raise RunCorrupt(f"run {run} finished, but {output} is missing")
    scores = _read_lines(paths.scores, Score, whole=True)
    generations = _read_lines(paths.generations, Generation, whole=True)
    scored = _paired(paths, generations, scores)
    unscored = [g for g in generations if key(g) not in scored]
    if unscored:
        raise RunCorrupt(f"{paths.generations} has {_name(unscored[0])}, which has no score")
    if not scores:
        raise RunCorrupt(f"run {run} finished, but {paths.scores} holds no scores")
    first_tries = _read_lines(paths.first_tries, Generation, whole=True)
    return StoredRun(
        manifest,
        paths,
        tuple(scores),
        {key(g): g for g in generations},
        _first_tries(paths, first_tries, scores, whole=True),
    )


def write_atomic(path: Path, data: bytes) -> None:
    """Replace ``path`` with ``data`` in one step, so a kill leaves the old file or the new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # The run's lock keeps every other writer out, so one temporary name is enough.
    partial = path.with_name(f"{path.name}.tmp")
    with partial.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _write_manifest(paths: RunPaths, manifest: RunManifest) -> None:
    write_atomic(paths.manifest, (manifest.model_dump_json(indent=2) + "\n").encode())


def _name(item: Generation | Score) -> str:
    return f"{item.example_id} sample {item.sample_idx}"


def _paired(
    paths: RunPaths,
    generations: Sequence[Generation],
    scores: Sequence[Score],
    expected: Collection[Key] | None = None,
) -> set[Key]:
    """The samples scored, once no sample is stored twice and each score has its
    generation, and is in ``expected`` when that is given."""
    generated: set[Key] = set()
    for generation in generations:
        if key(generation) in generated:
            raise RunCorrupt(f"{paths.generations} has {_name(generation)} twice")
        generated.add(key(generation))
    done: set[Key] = set()
    for score in scores:
        if key(score) in done:
            raise RunCorrupt(f"{paths.scores} has {_name(score)} twice")
        if expected is not None and key(score) not in expected:
            raise RunCorrupt(f"{paths.scores} has {_name(score)}, which this run never asks")
        if key(score) not in generated:
            raise RunCorrupt(f"{paths.scores} has {_name(score)}, which was never generated")
        done.add(key(score))
    return done


def _first_tries(
    paths: RunPaths, first_tries: Sequence[Generation], scores: Sequence[Score], *, whole: bool
) -> dict[Key, Generation]:
    """The first try of each retried score, once each one has one and no other score does.

    The first try of a sample not scored yet is left out, as a kill leaves one, unless
    ``whole`` says there can't be any.
    """
    scored = {key(score): score for score in scores}
    kept: dict[Key, Generation] = {}
    seen: set[Key] = set()
    for first in first_tries:
        if key(first) in seen:
            raise RunCorrupt(f"{paths.first_tries} has {_name(first)} twice")
        seen.add(key(first))
        score = scored.get(key(first))
        if score is None:
            if whole:
                raise RunCorrupt(f"{paths.first_tries} has {_name(first)}, which has no score")
        elif score.attempts > 1:
            kept[key(first)] = first
        else:
            raise RunCorrupt(f"{paths.first_tries} has {_name(first)}, whose score wasn't retried")
    for score in scores:
        if score.attempts > 1 and key(score) not in kept:
            raise RunCorrupt(
                f"{paths.scores} has {_name(score)} retried, but {paths.first_tries} lacks its "
                "first try"
            )
    return kept


def _read_lines(path: Path, model: type[M], *, whole: bool = False) -> list[M]:
    """Every complete line of a JSON lines file. A last line a kill cut short is left out,
    or raises ``RunCorrupt`` when ``whole`` says the file can't have one."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return []
    *lines, tail = data.split(b"\n")
    if tail:
        if whole:
            raise RunCorrupt(f"{path} ends in a line cut short")
        log.warning("%s: dropping a last line cut short by a stopped run", path)
    items: list[M] = []
    for number, line in enumerate(lines, 1):
        try:
            items.append(model.model_validate_json(line))
        except ValidationError as exc:
            raise RunCorrupt(f"{path} line {number} is not a {model.__name__}: {exc}") from exc
    return items


def _append_lines(path: Path, items: Sequence[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(b"".join(item.model_dump_json().encode() + b"\n" for item in items))
        handle.flush()
        os.fsync(handle.fileno())


def _write_lines(path: Path, items: Sequence[BaseModel]) -> None:
    write_atomic(path, b"".join(item.model_dump_json().encode() + b"\n" for item in items))
