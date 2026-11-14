"""The safe executor on real Spider databases: reads work, nothing else does, nothing hangs."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from forge.data.spider import DataNotFound
from forge.sql import executor as executor_module
from forge.sql.executor import DEFAULT_MAX_ROWS, SqlExecutor, SqliteExecutor, connect_readonly

DB = "department_management"
RUNAWAY = "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) SELECT count(*) FROM r"


def _counting(n: int) -> str:
    return (
        f"WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r WHERE x < {n}) "
        "SELECT x FROM r"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _db_file(root: Path, db_id: str = DB) -> Path:
    return root / "database" / db_id / f"{db_id}.sqlite"


@pytest.fixture
def executor(spider_mini: Path) -> Any:
    with SqliteExecutor() as sql:
        yield sql


@pytest.mark.phase(1)
def test_select_returns_rows(
    executor: SqliteExecutor, spider_mini: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query = "SELECT Department_ID, Name, Budget_in_Billions FROM department ORDER BY 1 LIMIT 3"
    checked: SqlExecutor = executor  # the class satisfies the plan's protocol

    result = checked.run(DB, query)

    assert result.ok, result.error
    assert result.columns == ["Department_ID", "Name", "Budget_in_Billions"]
    assert result.rows == [(1, "State", 9.96), (2, "Treasury", 11.1), (3, "Defense", 439.3)]
    assert (result.error, result.error_kind, result.truncated) == (None, None, False)
    assert result.ms > 0

    # Exactly what a plain read-only connection returns, whole tables included.
    with closing(sqlite3.connect(f"{_db_file(spider_mini).as_uri()}?mode=ro", uri=True)) as ref:
        for table in ("department", "head", "management"):
            full = executor.run(DB, f"SELECT * FROM {table}")
            reference = ref.execute(f"SELECT * FROM {table}")
            assert full.rows == reference.fetchall()
            assert full.columns == [d[0] for d in reference.description]

    # One connection per database, opened on first use and reused after that.
    opened: list[str] = []
    real_connect = sqlite3.connect

    def spy(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        opened.append(args[0])
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(executor_module.sqlite3, "connect", spy)
    for _ in range(3):
        assert executor.run(DB, "SELECT count(*) FROM head").rows == [(10,)]
        assert executor.run("flight_1", "SELECT count(*) FROM aircraft").ok
    assert opened == [f"{_db_file(spider_mini, 'flight_1').as_uri()}?mode=ro&immutable=1"]


@pytest.mark.phase(1)
def test_writes_denied(
    executor: SqliteExecutor, spider_mini: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A file that ATTACH or VACUUM INTO managed to create would land here.
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "copy.sqlite"
    before = _sha256(_db_file(spider_mini))
    statements = [
        "INSERT INTO head VALUES (99, 'x', 'y', 1.0)",
        "INSERT INTO head SELECT * FROM head",
        "REPLACE INTO head VALUES (1, 'x', 'y', 1.0)",
        "UPDATE head SET age = 0",
        "DELETE FROM head",
        "WITH gone AS (SELECT 1) DELETE FROM head",
        "DROP TABLE head",
        "DROP TABLE IF EXISTS forge_missing",
        "DROP INDEX IF EXISTS forge_missing",
        "DROP VIEW IF EXISTS forge_missing",
        "DROP TRIGGER IF EXISTS forge_missing",
        "ALTER TABLE head RENAME TO boss",
        "ALTER TABLE head ADD COLUMN z INTEGER",
        "CREATE TABLE z (a INTEGER)",
        "CREATE TEMP TABLE z (a INTEGER)",
        "CREATE INDEX z ON head (age)",
        "CREATE VIEW z AS SELECT 1",
        "CREATE TRIGGER z AFTER INSERT ON head BEGIN SELECT 1; END",
        f"ATTACH DATABASE '{target}' AS evil",
        "ATTACH DATABASE ':memory:' AS scratch",
        "DETACH DATABASE main",
        "PRAGMA writable_schema=1",
        "PRAGMA journal_mode=WAL",
        "PRAGMA user_version=7",
        "PRAGMA table_info(head)",
        "SELECT * FROM pragma_table_info('head')",
        "VACUUM",
        f"VACUUM INTO '{target}'",
        "ANALYZE",
        "REINDEX",
        "BEGIN",
        "SAVEPOINT s",
        "SELECT load_extension('evil')",
    ]

    for statement in statements:
        result = executor.run(DB, statement)
        assert (result.ok, result.error_kind) == (False, "denied"), (statement, result)
        assert result.columns is None
        assert result.rows is None

    assert executor.run(DB, "SELECT count(*) FROM head").rows == [(10,)]
    assert _sha256(_db_file(spider_mini)) == before
    assert list(tmp_path.iterdir()) == []


@pytest.mark.phase(1)
def test_connect_readonly_guards_on_two_layers(spider_mini: Path, tmp_path: Path) -> None:
    path = _db_file(spider_mini)
    with closing(connect_readonly(path)) as conn:
        assert conn.execute("SELECT count(*) FROM head").fetchall() == [(10,)]
        # The authorizer refuses writes as they are prepared...
        with pytest.raises(sqlite3.DatabaseError, match=r"^not authorized$"):
            conn.execute("DELETE FROM head")
        # ...and without it, the read-only URI still refuses them as they run.
        conn.set_authorizer(None)
        with pytest.raises(sqlite3.OperationalError, match="readonly database"):
            conn.execute("DELETE FROM head")
        assert conn.execute("SELECT count(*) FROM head").fetchall() == [(10,)]

    # A missing file is reported up front, not as a failing query later.
    with pytest.raises(DataNotFound, match=r"gone\.sqlite not found; run `forge data fetch`"):
        connect_readonly(tmp_path / "gone.sqlite")
    with SqliteExecutor(tmp_path) as executor, pytest.raises(DataNotFound):
        executor.run("gone", "SELECT 1")


@pytest.mark.phase(1)
def test_timeout(executor: SqliteExecutor) -> None:
    started = time.perf_counter()
    result = executor.run(DB, RUNAWAY, timeout_ms=300)
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert (result.ok, result.error_kind) == (False, "timeout")
    assert result.error == "timed out after 300 ms"
    assert 300 <= result.ms <= elapsed_ms
    # The deadline is checked every 1,000 VM steps, which is microseconds of work; the
    # margin is for slow CI machines, not for the check.
    assert elapsed_ms < 300 + 700

    # The deadline covers fetching too: the first row comes at once, the next never does.
    one_row = RUNAWAY.replace("count(*) FROM r", "x FROM r WHERE x < 2")
    assert executor.run(DB, one_row, timeout_ms=200).error_kind == "timeout"

    # Interrupting leaves the connection usable, and without the old deadline.
    assert executor.run(DB, _counting(3)).rows == [(1,), (2,), (3,)]


@pytest.mark.phase(1)
def test_row_cap_truncates(executor: SqliteExecutor) -> None:
    capped = executor.run(DB, _counting(20_000))
    assert capped.ok, capped.error
    assert capped.truncated
    assert DEFAULT_MAX_ROWS == 10_000
    assert capped.rows == [(x,) for x in range(1, DEFAULT_MAX_ROWS + 1)]

    # At the cap is not past it.
    whole = executor.run(DB, _counting(20_000), max_rows=20_000)
    assert (whole.truncated, len(whole.rows or ())) == (False, 20_000)
    assert executor.run(DB, _counting(5), max_rows=5).truncated is False
    assert executor.run(DB, _counting(6), max_rows=5).truncated is True

    # A query with no end is stopped by the cap, not the timeout.
    endless = executor.run(DB, RUNAWAY.replace("count(*)", "x"), max_rows=50)
    assert (endless.ok, endless.truncated, len(endless.rows or ())) == (True, True, 50)


@pytest.mark.phase(1)
def test_bad_utf8_handled(tmp_path: Path) -> None:
    # Only wta_1, a dev database, has text that isn't UTF-8, so make one.
    path = _db_file(tmp_path, "odd")
    path.parent.mkdir(parents=True)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT)")
        conn.execute("INSERT INTO notes (body) VALUES (CAST(X'41ff42' AS TEXT))")
        conn.execute("INSERT INTO notes (body) VALUES ('café')")

    with SqliteExecutor(tmp_path) as executor:
        result = executor.run("odd", "SELECT body FROM notes ORDER BY id")
        assert result.ok, result.error
        assert result.rows == [("A�B",), ("café",)]
        # Bad bytes made by the query itself decode the same way.
        made = executor.run("odd", "SELECT 'x' || CAST(X'ff' AS TEXT)")
        assert made.rows == [("x�",)]


@pytest.mark.phase(1)
def test_errors_classified(executor: SqliteExecutor) -> None:
    cases = {
        "SELEC * FROM head": "syntax",
        "SELECT * FROM head WHERE": "syntax",
        "SELECT 'unclosed FROM head": "syntax",
        "SELECT * FROM nowhere": "no_such_table",
        "SELECT nobody FROM head": "no_such_column",
        "SELECT [not authorized] FROM head": "no_such_column",
        "SELECT 1 FROM [no such column: x]": "no_such_table",
        "SELECT no_such_function(1)": "other",
        "SELECT * FROM head WHERE age > ?": "other",
        "SELECT 1\x00": "other",
        "SELECT '\ud800'": "other",
    }
    for sql, kind in cases.items():
        result = executor.run(DB, sql)
        assert (result.ok, result.error_kind) == (False, kind), (sql, result)
        assert result.error
