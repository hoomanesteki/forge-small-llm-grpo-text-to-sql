"""The safety changes made to the vendored evaluator's executor (see its CHANGES.md)."""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import time
from pathlib import Path

import pytest
from test_suite_eval.exec_eval import exec_on_db

RUNAWAY = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT count(*) FROM r"


@pytest.fixture
def scratch_db(tmp_path: Path) -> Path:
    # A throwaway database, so a regression can never damage a committed fixture.
    path = tmp_path / "scratch" / "scratch.sqlite"
    path.parent.mkdir()
    con = sqlite3.connect(path)
    with con:
        con.execute("CREATE TABLE t (x INTEGER)")
        con.executemany("INSERT INTO t VALUES (?)", [(1,), (2,), (3,)])
    con.close()
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.phase(1)
def test_writes_fail_and_leave_the_file_unchanged(scratch_db: Path) -> None:
    before = _sha256(scratch_db)
    for statement in ("DROP TABLE t", "DELETE FROM t", "INSERT INTO t VALUES (4)"):
        flag, error = asyncio.run(exec_on_db(str(scratch_db), statement))
        assert flag == "exception", statement
        assert isinstance(error, sqlite3.OperationalError)
        assert "readonly" in str(error)
    assert _sha256(scratch_db) == before
    assert asyncio.run(exec_on_db(str(scratch_db), "SELECT x FROM t ORDER BY x")) == (
        "result",
        [(1,), (2,), (3,)],
    )


@pytest.mark.phase(1)
def test_missing_database_is_an_error_not_a_new_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing.sqlite"
    flag, error = asyncio.run(exec_on_db(str(missing), "SELECT 1"))
    assert flag == "exception"
    assert isinstance(error, sqlite3.OperationalError)
    assert not missing.exists()
    assert capsys.readouterr() == ("", "")


@pytest.mark.phase(1)
def test_runaway_query_is_interrupted_at_the_timeout(scratch_db: Path) -> None:
    start = time.monotonic()
    flag, error = asyncio.run(exec_on_db(str(scratch_db), RUNAWAY, timeout=1))
    elapsed = time.monotonic() - start
    assert (flag, error) == ("exception", TimeoutError)
    # The deadline is checked every 1,000 VM steps, so it stops right after one second;
    # the upper bound only leaves room for a slow CI runner.
    assert 1.0 <= elapsed < 4.0
