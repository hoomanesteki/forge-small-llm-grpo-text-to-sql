"""Run untrusted SQL in worker processes that are killed when they pass a hard deadline.

``SqliteExecutor.run`` stops a query from inside the process, and SQLite only looks at
its progress handler and interrupt flag at VM loop jumps, so it can't stop work packed
into one row: under FG-191's limits a one-row ``SELECT`` adding up 100 ``instr`` calls on
a 99 kB string ran 8.7 s against a 300 ms timeout. A ``SandboxPool`` keeps a few
long-lived worker processes, each with its own ``SqliteExecutor``, and sends each one
job at a time over a pipe. A job still running ``KILL_GRACE_MS`` past its deadline gets
its worker killed with SIGKILL, comes back as a ``timeout``, and a fresh worker starts
the next time that slot has work.

Which path for which SQL (ADR-0014):

- Untrusted SQL, anything a model wrote or a reward or evaluation scores, runs through
  ``SandboxPool.run``, ``SandboxPool.run_many`` or ``SqliteExecutor.run_many``, which
  starts a pool.
- Trusted SQL, Spider's gold queries, schema reads and tests, may call
  ``SqliteExecutor.run`` in process, whose deadline is soft.

Workers start with the spawn method on every platform, because forking a process that
holds SQLite connections (or CUDA, in a trainer) is unsafe; spawn imports the calling
script again in each worker, so a script that makes a pool needs
``if __name__ == "__main__":``. A pool is not thread-safe.
"""

from __future__ import annotations

import importlib
import logging
import pickle
import signal
import time
import traceback
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from pathlib import Path
from types import TracebackType
from typing import Self, TypeVar, TypeVarTuple, cast

from forge.core.types import ExecResult
from forge.data.spider import resolve_data_dir
from forge.sql.executor import (
    DEFAULT_MAX_ROWS,
    DEFAULT_TIMEOUT_MS,
    SqliteExecutor,
    check_run_args,
)

log = logging.getLogger(__name__)

# Past its own deadline a query normally stops within milliseconds (the executor's
# watchdog); the grace leaves room for that, and for a busy machine, before the kill.
KILL_GRACE_MS = 250
# Spawning Python and importing Forge takes well under a second; this is for a cold,
# overloaded CI machine.
START_TIMEOUT_S = 60.0
_READY = "ready"

T = TypeVar("T")
Ts = TypeVarTuple("Ts")


class SandboxTimeout(TimeoutError):
    """A job ran past its deadline, so its worker was killed."""


class SandboxError(RuntimeError):
    """A worker failed outside the job it ran: it didn't start, or it died."""


@dataclass(frozen=True)
class _Done:
    value: object


@dataclass(frozen=True)
class _Killed:
    ms: float


@dataclass(frozen=True)
class _Died:
    exitcode: int | None
    ms: float


_Outcome = _Done | _Killed | _Died


@dataclass
class _Worker:
    process: BaseProcess
    conn: Connection
    started: float
    ready: bool = False


@dataclass
class _Job:
    index: int
    sent: float


class SandboxPool:
    """Up to ``workers`` killable processes running SQL on one Spider data directory.

    Workers start when there is work for them and stay up until ``close`` (or the end of
    a ``with`` block). ``imports`` are modules each worker imports before it takes a
    job, so a job function's module doesn't load inside the job's deadline.
    ``restarts`` counts the workers replaced after a kill or a crash.
    """

    def __init__(
        self, workers: int = 1, *, data_dir: Path | None = None, imports: Sequence[str] = ()
    ) -> None:
        if workers < 1:
            raise ValueError(f"workers must be at least 1, got {workers}")
        # Absolute, so workers find the same files whatever their working directory.
        self.data_dir = resolve_data_dir(data_dir).resolve()
        self.restarts = 0
        self._imports = tuple(imports)
        self._slots: list[_Worker | None] = [None] * workers
        self._context = get_context("spawn")
        self._closed = False

    def run(
        self,
        db_id: str,
        sql: str,
        *,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        max_rows: int = DEFAULT_MAX_ROWS,
    ) -> ExecResult:
        """``SqliteExecutor.run`` in a worker, with a hard deadline."""
        (result,) = self.run_many([(db_id, sql)], timeout_ms=timeout_ms, max_rows=max_rows)
        return result

    def run_many(
        self,
        items: Sequence[tuple[str, str]],
        *,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        max_rows: int = DEFAULT_MAX_ROWS,
    ) -> list[ExecResult]:
        """Run ``(db_id, sql)`` pairs across the workers; results keep their order.

        A query still running ``KILL_GRACE_MS`` after its ``timeout_ms`` comes back as a
        ``timeout`` that says it was killed. A worker that dies under a query gives an
        ``other`` error. An exception a query raises, such as ``DataNotFound`` for a
        missing database, is raised here once the queries already running are done.
        """
        check_run_args(timeout_ms, max_rows)
        jobs = [(db_id, sql, timeout_ms, max_rows) for db_id, sql in items]
        outcomes = self._map(_run_job, jobs, (timeout_ms + KILL_GRACE_MS) / 1000)
        return [_exec_result(outcome, timeout_ms) for outcome in outcomes]

    def call(
        self,
        fn: Callable[[SqliteExecutor, *Ts], T],
        *args: *Ts,
        timeout_s: float,
    ) -> T:
        """``fn(executor, *args)`` in a worker, killed after ``timeout_s``.

        ``fn`` must be a module-level function, which is how a job crosses to a worker,
        and ``executor`` is the worker's own. Raises ``SandboxTimeout`` when the worker
        had to be killed and ``SandboxError`` when it died; an exception ``fn`` raises is
        raised here, with the worker's traceback as a note.
        """
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s}")
        (outcome,) = self._map(fn, [args], timeout_s)
        if isinstance(outcome, _Killed):
            raise SandboxTimeout(f"killed after {outcome.ms:.0f} ms (timeout {timeout_s:g} s)")
        if isinstance(outcome, _Died):
            raise SandboxError(f"sandbox worker died with exit code {outcome.exitcode}")
        return cast(T, outcome.value)

    def close(self) -> None:
        """Stop every worker; the pool can't be used after this."""
        workers = [worker for worker in self._slots if worker is not None]
        for worker in workers:
            # An idle worker reads the end of its pipe and exits.
            worker.conn.close()
        for worker in workers:
            worker.process.join(timeout=1)
            if worker.process.is_alive():
                worker.process.kill()
                worker.process.join()
        self._slots = [None] * len(self._slots)
        if not self._closed:
            log.debug(
                "closed a sandbox pool of %d workers after %d restarts",
                len(self._slots),
                self.restarts,
            )
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _map(
        self, fn: Callable[..., object], jobs: Sequence[tuple[object, ...]], timeout_s: float
    ) -> list[_Outcome]:
        """Each job's outcome, in order; each one runs ``fn(executor, *job)`` in a worker."""
        if self._closed:
            raise RuntimeError("the sandbox pool is closed")
        outcomes: list[_Outcome | None] = [None] * len(jobs)
        pending = deque(range(len(jobs)))
        busy: dict[int, _Job] = {}
        error: BaseException | None = None
        try:
            while busy or (pending and error is None):
                if error is None:
                    self._dispatch(fn, jobs, pending, busy)
                for slot, message in self._receive(busy, timeout_s):
                    job = busy.pop(slot)
                    if isinstance(message, _Died):
                        outcomes[job.index] = message
                    elif message[0] == "error":
                        error = error or cast(BaseException, message[1])
                    else:
                        outcomes[job.index] = _Done(message[1])
                now = time.monotonic()
                for slot, job in list(busy.items()):
                    if now - job.sent >= timeout_s:
                        del busy[slot]
                        outcomes[job.index] = _Killed((now - job.sent) * 1000)
                        self._replace(slot, f"killed {(now - job.sent) * 1000:.0f} ms into a job")
        except BaseException:
            # Their replies would be read as answers to the next jobs.
            for slot in busy:
                self._replace(slot, "stopped with the call that gave it a job")
            raise
        if error is not None:
            raise error
        return [outcome for outcome in outcomes if outcome is not None]

    def _dispatch(
        self,
        fn: Callable[..., object],
        jobs: Sequence[tuple[object, ...]],
        pending: deque[int],
        busy: dict[int, _Job],
    ) -> None:
        """Give waiting jobs to idle workers, and start workers for the jobs left over."""
        for slot, worker in enumerate(self._slots):
            if pending and slot not in busy and worker is not None and worker.ready:
                index = pending.popleft()
                try:
                    worker.conn.send((fn, jobs[index]))
                except (BrokenPipeError, ConnectionResetError):
                    # It died while idle; the job goes to the next worker.
                    pending.appendleft(index)
                    self._replace(slot, "was found dead while idle")
                    continue
                busy[slot] = _Job(index, time.monotonic())
        starting = sum(1 for worker in self._slots if worker is not None and not worker.ready)
        for slot, worker in enumerate(self._slots):
            if starting >= len(pending):
                break
            if worker is None:
                self._spawn(slot)
                starting += 1

    def _receive(
        self, busy: dict[int, _Job], timeout_s: float
    ) -> list[tuple[int, tuple[str, object] | _Died]]:
        """Wait for the first reply or deadline; the busy workers' replies that came.

        A starting worker's ready message is taken here too.
        """
        waiting = {
            slot: worker
            for slot, worker in enumerate(self._slots)
            if worker is not None and (slot in busy or not worker.ready)
        }
        deadlines = [job.sent + timeout_s for job in busy.values()]
        deadlines += [w.started + START_TIMEOUT_S for w in waiting.values() if not w.ready]
        if not deadlines:
            return []
        ready = wait(
            [worker.conn for worker in waiting.values()],
            max(0.0, min(deadlines) - time.monotonic()),
        )
        replies: list[tuple[int, tuple[str, object] | _Died]] = []
        for slot, worker in waiting.items():
            if worker.conn in ready:
                try:
                    message = worker.conn.recv()
                except (EOFError, OSError):
                    worker.process.join()
                    if not worker.ready:
                        self._replace(slot, "exited while starting")
                        raise SandboxError(
                            f"a sandbox worker exited while starting, with exit code "
                            f"{worker.process.exitcode}"
                        ) from None
                    ms = (time.monotonic() - busy[slot].sent) * 1000
                    replies.append((slot, _Died(worker.process.exitcode, ms)))
                    self._replace(slot, f"died with exit code {worker.process.exitcode}")
                    continue
                if worker.ready:
                    replies.append((slot, cast(tuple[str, object], message)))
                else:
                    worker.ready = message == _READY
            elif not worker.ready and time.monotonic() - worker.started >= START_TIMEOUT_S:
                self._replace(slot, "didn't start")
                raise SandboxError(f"a sandbox worker didn't start in {START_TIMEOUT_S:g} s")
        return replies

    def _spawn(self, slot: int) -> None:
        conn, child_conn = self._context.Pipe()
        process = self._context.Process(
            target=_serve,
            args=(child_conn, self.data_dir, self._imports),
            name=f"forge-sandbox-{slot}",
            # Stopped with the interpreter, should the pool never be closed.
            daemon=True,
        )
        started = time.monotonic()
        process.start()
        # Closed here, so the parent reads EOF once the worker is gone.
        child_conn.close()
        self._slots[slot] = _Worker(process, conn, started)

    def _replace(self, slot: int, why: str) -> None:
        """Kill the worker in ``slot``, which a later job will start afresh."""
        worker = self._slots[slot]
        if worker is None:
            return
        worker.process.kill()
        worker.process.join()
        worker.conn.close()
        self._slots[slot] = None
        self.restarts += 1
        log.debug(
            "sandbox worker %d (pid %s) %s; %d restarts so far",
            slot,
            worker.process.pid,
            why,
            self.restarts,
        )


def _exec_result(outcome: _Outcome, timeout_ms: int) -> ExecResult:
    if isinstance(outcome, _Killed):
        return ExecResult(
            ok=False,
            error=f"killed after {outcome.ms:.0f} ms (timeout {timeout_ms} ms)",
            error_kind="timeout",
            ms=outcome.ms,
        )
    if isinstance(outcome, _Died):
        return ExecResult(
            ok=False,
            error=f"sandbox worker died with exit code {outcome.exitcode}",
            error_kind="other",
            ms=outcome.ms,
        )
    return cast(ExecResult, outcome.value)


def _run_job(
    executor: SqliteExecutor, db_id: str, sql: str, timeout_ms: int, max_rows: int
) -> ExecResult:
    return executor.run(db_id, sql, timeout_ms=timeout_ms, max_rows=max_rows)


def _serve(conn: Connection, data_dir: Path, imports: tuple[str, ...]) -> None:
    """A worker's life: import, open an executor, then run jobs until the pipe closes."""
    # Ctrl-C reaches the whole process group; the parent decides what happens to workers.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    for module in imports:
        importlib.import_module(module)
    with SqliteExecutor(data_dir) as executor:
        try:
            conn.send(_READY)
            while True:
                try:
                    payload = conn.recv_bytes()
                except EOFError:
                    return
                try:
                    fn, args = pickle.loads(payload)
                    reply: tuple[str, object] = ("done", fn(executor, *args))
                except Exception as exc:
                    reply = ("error", _portable(exc))
                conn.send(reply)
        except BrokenPipeError:
            # The pool was closed before this worker was ready or done.
            return


def _portable(exc: Exception) -> Exception:
    """``exc`` in a form that crosses the pipe, with the worker's traceback as a note."""
    exc.add_note("in a sandbox worker:\n" + "".join(traceback.format_tb(exc.__traceback__)))
    try:
        pickle.loads(pickle.dumps(exc))
    except Exception:
        return SandboxError(f"{type(exc).__name__}: {exc}")
    return exc
