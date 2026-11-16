"""Jobs for sandbox workers in tests.

A job reaches a worker as its module and name, so the worker imports this module to run
it; the jobs live here, where every test that needs one can share it.
"""

from __future__ import annotations

import os
import resource
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from forge.core.types import ExecResult
from forge.eval import testsuite
from forge.eval.testsuite import MatchItem
from forge.sql.executor import SqliteExecutor

# The limits FG-191 set, under which the review found one row holding seconds of work.
FG191_LIMITS = {
    sqlite3.SQLITE_LIMIT_LENGTH: 100_000,
    sqlite3.SQLITE_LIMIT_LIKE_PATTERN_LENGTH: 1_000,
    sqlite3.SQLITE_LIMIT_SQL_LENGTH: 100_000,
}


def review_query(calls: int) -> str:
    """The FG-191 review's query: one row, ``calls`` instr calls on a 99 kB string."""
    one = "instr(x, hex(zeroblob(25000)) || '1')"
    return (
        "WITH t(x) AS MATERIALIZED (SELECT hex(zeroblob(49500))) SELECT "
        + " + ".join([one] * calls)
        + " FROM t"
    )


def run_with_fg191_limits(
    executor: SqliteExecutor, db_id: str, sql: str, timeout_ms: int, max_rows: int
) -> ExecResult:
    """``forge.sql.sandbox._run_job`` on a connection with FG-191's limits.

    The review's queries need those limits to run at all; what is tested is the deadline.
    """
    conn = executor._connection(db_id)
    for limit, value in FG191_LIMITS.items():
        conn.setlimit(limit, value)
    return executor.run(db_id, sql, timeout_ms=timeout_ms, max_rows=max_rows)


# A cap on SQLite's heap far below what HOARD asks for, and far above what Spider needs.
HEAP_LIMIT = 32 * 2**20


def run_under_heap_limit(
    executor: SqliteExecutor, db_id: str, sql: str, timeout_ms: int, max_rows: int
) -> ExecResult:
    """``forge.sql.sandbox._run_job`` once SQLite's heap is capped at ``HEAP_LIMIT``.

    The cap is SQLite's own, for the whole process, so a query past it gets SQLITE_NOMEM
    on any platform, as one past ``RLIMIT_AS`` does on Linux. A pragma can lower it but
    never lift it, so the worker keeps it for the rest of its life.
    """
    with closing(sqlite3.connect(":memory:")) as scratch:
        scratch.execute(f"PRAGMA hard_heap_limit = {HEAP_LIMIT}")
    return executor.run(db_id, sql, timeout_ms=timeout_ms, max_rows=max_rows)


def raise_memory_error(executor: SqliteExecutor) -> None:
    raise MemoryError


def address_space_limit(executor: SqliteExecutor) -> tuple[int, int]:
    """The worker's soft and hard ``RLIMIT_AS``."""
    return resource.getrlimit(resource.RLIMIT_AS)


def address_space(executor: SqliteExecutor) -> int:
    """The bytes of address space the worker maps now, on Linux."""
    pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[0])
    return pages * resource.getpagesize()


def exit_with_three(executor: SqliteExecutor, *args: object) -> ExecResult:
    os._exit(3)


def worker_pid(executor: SqliteExecutor) -> int:
    return os.getpid()


def fail(executor: SqliteExecutor, message: str) -> None:
    raise ValueError(message)


class Unpicklable(Exception):
    """Pickles from its message alone, so it pickles but can't be rebuilt from that."""

    def __init__(self, first: str, second: str) -> None:
        super().__init__(f"{first} and {second}")


def fail_unpicklably(executor: SqliteExecutor) -> None:
    raise Unpicklable("one", "two")


def returns_unloadable(executor: SqliteExecutor) -> Unpicklable:
    return Unpicklable("one", "two")


class Unsendable:
    """A value that raises when pickled, so a job returning it can't reply over the pipe."""

    def __reduce__(self) -> tuple[object, ...]:
        raise TypeError("Unsendable cannot be pickled")


def returns_unsendable(executor: SqliteExecutor) -> Unsendable:
    return Unsendable()


def sleep(executor: SqliteExecutor, seconds: float) -> None:
    time.sleep(seconds)


def match_or_die(executor: SqliteExecutor, root: Path, item: MatchItem, die_on: str) -> bool:
    """``forge.eval.testsuite._match``, except that the worker dies on ``die_on``."""
    if item.pred_sql == die_on:
        os._exit(3)
    return testsuite._match(executor, root, item)
