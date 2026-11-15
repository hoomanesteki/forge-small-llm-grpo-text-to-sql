"""The safety changes made to the vendored evaluator's executor (see its CHANGES.md)."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import pytest
from test_suite_eval.evaluation import isValidSQL
from test_suite_eval.exec_eval import MAX_ROWS, ResultTooBig, exec_on_db, get_cursor_from_path

from tests.sandbox_jobs import review_query

ROOT = Path(__file__).resolve().parents[2]
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
        assert isinstance(error, sqlite3.DatabaseError)
        assert str(error) == "not authorized"
    # Without the authorizer, the read-only URI still refuses them as they run.
    cursor = get_cursor_from_path(str(scratch_db))
    with closing(cursor.connection) as conn:
        conn.set_authorizer(None)
        with pytest.raises(sqlite3.OperationalError, match="readonly database"):
            conn.execute("DELETE FROM t")
    assert _sha256(scratch_db) == before
    assert asyncio.run(exec_on_db(str(scratch_db), "SELECT x FROM t ORDER BY x")) == (
        "result",
        [(1,), (2,), (3,)],
    )


@pytest.mark.phase(1)
def test_attach_and_vacuum_into_write_nothing(
    scratch_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A read-only connection still lets ATTACH and VACUUM INTO create the files they name.
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    folder = scratch_db.parent
    for statement in (
        f"ATTACH DATABASE '{folder}/attached.sqlite' AS a",
        "ATTACH DATABASE 'relative.sqlite' AS a",
        f"ATTACH DATABASE 'file:{work}/uri.sqlite?mode=rwc' AS a",
        f"VACUUM INTO '{work}/copy.sqlite'",
        f"VACUUM INTO '{folder}/copy.sqlite'",
        "PRAGMA user_version = 7",
    ):
        flag, error = asyncio.run(exec_on_db(str(scratch_db), statement))
        assert flag == "exception", statement
        assert isinstance(error, sqlite3.DatabaseError), statement
    assert sorted(p.name for p in folder.iterdir()) == ["scratch.sqlite"]
    assert list(work.iterdir()) == []


@pytest.mark.phase(1)
def test_values_are_capped(scratch_db: Path) -> None:
    for query in (
        "SELECT length(randomblob(900000000))",
        "SELECT count(*) FROM t WHERE length(randomblob(300000000)) > 0",
        "SELECT printf('%.*c', 4000, 'a') LIKE ('%' || printf('%.*c', 200, 'a') || 'b')",
        # The FG-191 review's one-row query, 8.7 s under 100 kB values.
        review_query(100),
    ):
        start = time.monotonic()
        flag, error = asyncio.run(exec_on_db(str(scratch_db), query))
        assert flag == "exception", query
        assert isinstance(error, sqlite3.Error), query
        assert time.monotonic() - start < 0.5, query


@pytest.mark.phase(1)
def test_results_are_capped(scratch_db: Path) -> None:
    counting = (
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < {n}) "
        "SELECT n{columns} FROM r"
    )
    blobs = ", " + ", ".join(["zeroblob(4000)"] * 10)
    for query, error in (
        (counting.format(n=MAX_ROWS + 1, columns=""), "more than 100,000 rows"),
        # About 40 kB a row, so 2,000 rows take some 81 MB.
        (counting.format(n=2000, columns=blobs), "rows take more than 64 MiB"),
    ):
        flag, value = asyncio.run(exec_on_db(str(scratch_db), query))
        assert (flag, type(value), str(value)) == ("exception", ResultTooBig, error)

    # At the ceilings is not past them.
    flag, rows = asyncio.run(exec_on_db(str(scratch_db), counting.format(n=MAX_ROWS, columns="")))
    assert (flag, len(rows)) == ("result", MAX_ROWS)
    flag, rows = asyncio.run(exec_on_db(str(scratch_db), counting.format(n=1000, columns=blobs)))
    assert (flag, len(rows), len(rows[0])) == ("result", 1000, 11)


@pytest.mark.phase(1)
def test_read_write_helpers_are_documented_and_unused(scratch_db: Path) -> None:
    # isValidSQL runs its SQL on a read-write connection, so it commits DDL.
    assert isValidSQL("DROP TABLE t", str(scratch_db)) is True
    with closing(sqlite3.connect(scratch_db)) as conn:
        assert conn.execute("SELECT name FROM sqlite_master").fetchall() == []

    # Whitespace collapsed, so rewrapping the file doesn't matter.
    changes = " ".join(
        (ROOT / "third_party" / "test_suite_eval" / "CHANGES.md").read_text().split()
    )
    assert "`isValidSQL` runs its SQL on a read-write connection" in changes
    assert "never call it on a prediction" in changes
    # Forge calls neither helper that connects read-write.
    used = {
        node.id if isinstance(node, ast.Name) else node.attr
        for path in sorted((ROOT / "src" / "forge").rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Name | ast.Attribute)
    }
    assert not used & {"isValidSQL", "get_schema"}


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
