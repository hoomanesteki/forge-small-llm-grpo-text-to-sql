"""Jobs for sandbox workers in tests.

A job reaches a worker as its module and name, so the worker imports this module to run
it; the jobs live here, where every test that needs one can share it.
"""

from __future__ import annotations

import os
import sqlite3
import time

from forge.core.types import ExecResult
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


def exit_with_three(executor: SqliteExecutor, *args: object) -> ExecResult:
    os._exit(3)


def worker_pid(executor: SqliteExecutor) -> int:
    return os.getpid()


def fail(executor: SqliteExecutor, message: str) -> None:
    raise ValueError(message)


class Unpicklable(Exception):
    def __init__(self, first: str, second: str) -> None:
        super().__init__(f"{first} and {second}")


def fail_unpicklably(executor: SqliteExecutor) -> None:
    raise Unpicklable("one", "two")


class Unsendable:
    """A value that raises when pickled, so a job returning it can't reply over the pipe."""

    def __reduce__(self) -> tuple[object, ...]:
        raise TypeError("Unsendable cannot be pickled")


def returns_unsendable(executor: SqliteExecutor) -> Unsendable:
    return Unsendable()


def sleep(executor: SqliteExecutor, seconds: float) -> None:
    time.sleep(seconds)
