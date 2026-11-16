"""Two copies of one run, this machine's folder and the run store's, and which is newer.

A run's ``sessions.json`` lists its attempts oldest first, one session each
(``forge.ops.sessions``), and an attempt carries on from the copy of the run it started
with. So a copy's sessions are its history. A session is known by when it started, which
no copy changes once it holds the session: a kernel stretches its session back to the
kernel's start before the run leaves the kernel. A copy without the file has the one
session its manifest spans.

When one copy's sessions begin with all of the other's, it is the newer, or the same copy
when they match. When neither's do, two attempts carried on from one point on two
machines, and the copies went separate ways: each holds work the other lacks, so neither
may replace the other, and a person chooses, keeping the store's with ``forge runs pull``
or evaluating again under a new run id.

``fetch_newer`` makes a run's folder the store's copy before an attempt resumes the run,
when the store's is the newer, and ``put_run`` uploads a folder only when the store holds
no session it lacks, so no upload loses progress the store has.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

from forge.eval.run_files import MANIFEST_FILE, RunConflict
from forge.ops.hub import LocalRunStore, RunNotFound, RunStore
from forge.ops.sessions import (
    SESSIONS_FILE,
    Session,
    manifest_session,
    parse_sessions,
    read_sessions,
)

History = tuple[datetime, ...]  # when each session of a copy started, oldest first
Relation = Literal["same", "newer", "older", "diverged"]  # one copy, to the other


class StoreNewer(OSError):
    """The run store's copy of a run holds a session the folder to upload lacks."""


def history(run_dir: Path) -> History | None:
    """The history of the run in ``run_dir``, or None when the folder has no manifest."""
    if not (run_dir / MANIFEST_FILE).is_file():
        return None
    sessions = read_sessions(run_dir)
    if sessions is None:
        sessions = [manifest_session(LocalRunStore(run_dir.parent).get_manifest(run_dir.name))]
    return _starts(sessions)


def stored_history(store: RunStore, run_id: str) -> History | None:
    """The history of the store's copy of the run, or None when the store has no such run."""
    try:
        manifest = store.get_manifest(run_id)
    except RunNotFound:
        return None
    data = store.read_file(run_id, SESSIONS_FILE)
    if data is None:
        return _starts([manifest_session(manifest)])
    where = f"{SESSIONS_FILE} of run {run_id} in {store}"
    return _starts(parse_sessions(data.decode("utf-8"), where))


def compare(here: Sequence[datetime], there: Sequence[datetime]) -> Relation:
    """How the copy with history ``here`` stands to the one with history ``there``."""
    if tuple(here) == tuple(there):
        return "same"
    shared = min(len(here), len(there))
    if tuple(here[:shared]) != tuple(there[:shared]):
        return "diverged"
    return "newer" if len(here) > len(there) else "older"


def fetch_newer(store: RunStore, run_dir: Path) -> bool:
    """Make ``run_dir`` the store's copy of its run when that is the newer, or when the
    folder holds no run; returns whether it did.

    Raises ``RunConflict`` when the two went separate ways, and changes neither.
    """
    run = run_dir.name
    there = stored_history(store, run)
    if there is None:
        return False
    here = history(run_dir)
    if here is not None:
        relation = compare(here, there)
        if relation == "diverged":
            raise RunConflict(
                f"run {run} in {run_dir} and in {store} {_parting(here, there, run_dir, store)}; "
                f"keep the store's with forge runs pull {run}, which replaces the folder, or "
                "pass a new --run-id"
            )
        if relation != "older":
            return False
    store.pull(run, run_dir)
    return True


def put_run(store: RunStore, run_dir: Path, *, replace: bool = False) -> str:
    """Upload the run in ``run_dir`` as ``store.put_folder`` does, unless the store's copy
    holds a session the folder lacks; returns where the store holds the run.

    Raises ``RunNotFound`` for a folder with no manifest, and ``StoreNewer``, having
    uploaded nothing, when the store's copy is the newer or went its own way.
    """
    run = run_dir.name
    here = history(run_dir)
    if here is None:
        raise RunNotFound(f"no run in {run_dir}: it has no {MANIFEST_FILE}")
    there = stored_history(store, run)
    if there is not None:
        relation = compare(here, there)
        if relation == "older":
            extra = len(there) - len(here)
            more = f"holds {extra} session{'' if extra == 1 else 's'} more than {run_dir}"
            raise StoreNewer(_kept(run, store, more))
        if relation == "diverged":
            parting = _parting(here, there, run_dir, store)
            raise StoreNewer(_kept(run, store, f"and {run_dir} {parting}"))
    return store.put_folder(run, run_dir, replace=replace)


def _kept(run: str, store: RunStore, held: str) -> str:
    return (
        f"the copy of run {run} in {store} {held}; nothing was uploaded, so the store keeps "
        f"its copy, which forge runs pull {run} takes in place of the folder's"
    )


def _starts(sessions: Sequence[Session]) -> History:
    return tuple(session.started_at for session in sessions)


def _parting(here: History, there: History, folder: Path, store: RunStore) -> str:
    """Where two histories that went separate ways part."""
    pairs = enumerate(zip(here, there, strict=False))
    at = next(i for i, (mine, theirs) in pairs if mine != theirs)
    return (
        f"went separate ways at session {at + 1}, which started at {here[at].isoformat()} "
        f"in {folder} and at {there[at].isoformat()} in {store}"
    )
