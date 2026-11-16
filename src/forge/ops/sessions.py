"""A run's sessions: each attempt at the run, from when it started to when it ended.

A run that stops before it is done, by the time guard, a failure or a kernel's time
limit, resumes in a later session, and the time between two sessions holds no GPU. So a
run's folder keeps its sessions in ``sessions.json``, oldest first: ``begin`` adds one as
an attempt starts, and ``end`` closes it with how the attempt ended. The ledger gives
every session that ended a row of its own, with that session's own hours
(``forge.ops.ledger.record_run``), and invariant I7 wants a row for every session of a
run on a budget platform.

In a Kaggle or Colab kernel, ``kaggle/entry.py`` then stretches the session its job ran
in to the whole kernel session, from the kernel's start to its end, since the GPU is held
through the setup, the preflight, the model load and the upload as well.

A folder without the file, such as one made before runs kept their sessions, has one
session: its manifest's ``started_at`` to its ``ended_at``. Times are UTC.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from forge.core.types import RunManifest

SESSIONS_FILE = "sessions.json"
# A session is running, or has ended as its attempt did: RunManifest's statuses.
STATUSES = frozenset({"running", "stopped_time_guard", "finished", "failed"})


@dataclass(frozen=True)
class Session:
    """One attempt at a run. An attempt killed before it could say when it ended is
    ``failed`` with no end: its hours were never known."""

    started_at: datetime
    ended_at: datetime | None = None
    status: str = "running"

    def __post_init__(self) -> None:
        for name, moment in (("started_at", self.started_at), ("ended_at", self.ended_at)):
            if moment is not None and moment.utcoffset() is None:
                raise ValueError(f"a session's {name} must be timezone-aware, got {moment}")
        if self.status not in STATUSES:
            raise ValueError(f"a session's status must be one of {sorted(STATUSES)}")
        if self.status == "running" and self.ended_at is not None:
            raise ValueError("a running session has no end yet")
        if self.ended_at is not None and self.ended_at < self.started_at:
            raise ValueError(f"a session can't end ({self.ended_at}) before it starts")

    def to_json(self) -> dict[str, Any]:
        end = None if self.ended_at is None else self.ended_at.isoformat()
        return {"started_at": self.started_at.isoformat(), "ended_at": end, "status": self.status}

    @classmethod
    def from_json(cls, data: object) -> Session:
        if not isinstance(data, dict) or set(data) != {"started_at", "ended_at", "status"}:
            raise ValueError(f"a session has started_at, ended_at and status, got {data!r}")
        end = data["ended_at"]
        return cls(
            started_at=datetime.fromisoformat(data["started_at"]),
            ended_at=None if end is None else datetime.fromisoformat(end),
            status=data["status"],
        )


def read_sessions(run_dir: Path) -> list[Session] | None:
    """The sessions ``run_dir`` records, or None when it keeps no sessions file.

    Raises ``ValueError`` for a file that isn't a list of sessions.
    """
    path = run_dir / SESSIONS_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        data = json.loads(text)
        if not isinstance(data, list) or not data:
            raise ValueError("it must list one session or more")
        return [Session.from_json(item) for item in data]
    # JSONDecodeError is a ValueError; a field of the wrong type raises TypeError.
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} is not a list of sessions: {exc}") from exc


def sessions_of(manifest: RunManifest, run_dir: Path) -> list[Session]:
    """The run's sessions as its folder records them, or else the one its manifest spans."""
    recorded = read_sessions(run_dir)
    return [manifest_session(manifest)] if recorded is None else recorded


def begin(run_dir: Path, started_at: datetime, *, resumes: RunManifest | None = None) -> None:
    """Record a session of the run in ``run_dir`` that starts at ``started_at``.

    ``resumes`` is the stored manifest of the run this attempt carries on. When the folder
    records no sessions, that manifest's span stands for the earlier ones. A session an
    attempt left running, killed before it could end it, becomes ``failed`` with no end.
    """
    sessions = read_sessions(run_dir)
    if sessions is None:
        sessions = [] if resumes is None else [manifest_session(resumes)]
    sessions = [replace(s, status="failed") if s.status == "running" else s for s in sessions]
    _write(run_dir, [*sessions, Session(started_at)])


def end(run_dir: Path, ended_at: datetime, status: str) -> list[Session]:
    """Close the session running in ``run_dir`` at ``ended_at`` with ``status``, and return
    the run's sessions. Raises ``ValueError`` when no session is running there."""
    if status == "running":
        raise ValueError("a session ends finished, failed or stopped_time_guard")
    sessions = read_sessions(run_dir) or []
    if not sessions or sessions[-1].status != "running":
        raise ValueError(f"{run_dir} has no session running to end")
    sessions[-1] = replace(sessions[-1], ended_at=ended_at, status=status)
    _write(run_dir, sessions)
    return sessions


def manifest_session(manifest: RunManifest) -> Session:
    """The one session a manifest spans, from its ``started_at`` to its ``ended_at``."""
    return Session(manifest.started_at, manifest.ended_at, manifest.status)


def _write(run_dir: Path, sessions: list[Session]) -> None:
    # In one step, so a kill leaves the old list or the new one.
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / SESSIONS_FILE
    partial = path.with_name(f"{SESSIONS_FILE}.tmp")
    text = json.dumps([session.to_json() for session in sessions], indent=2) + "\n"
    with partial.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
