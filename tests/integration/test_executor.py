"""The safe executor on real Spider databases: reads work, nothing else does, nothing hangs."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from test_suite_eval import exec_eval
from test_suite_eval.parse import remove_distinct
from typer.testing import CliRunner

from forge import cli
from forge.data.spider import FAILING_GOLD_IDS, DataNotFound, db_path, load_raw, resolve_data_dir
from forge.data.splits import split_examples
from forge.sql import executor as executor_module
from forge.sql.executor import (
    DEFAULT_MAX_ROWS,
    LIMITS,
    MAX_RESULT_BYTES,
    MAX_VALUE_BYTES,
    SqlExecutor,
    SqliteExecutor,
    connect_readonly,
    row_size,
)
from forge.sql.sandbox import KILL_GRACE_MS, SandboxPool
from tests.sandbox_jobs import review_query

DB = "department_management"
RUNAWAY = "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) SELECT count(*) FROM r"
# Endless, and each row spends milliseconds inside SQLite calls on 4 kB strings, so several
# rows pass between two progress-handler checks.
_INSTR = "instr(printf('%.*c', 4000, 'a') || x, printf('%.*c', 2000, 'a') || 'b')"
_LIKE = "(printf('%.*c', 4000, 'a') || x LIKE '%' || printf('%.*c', 90, 'a') || 'b')"
SLOW_ROWS = (
    RUNAWAY + " WHERE " + " + ".join([_INSTR] * 20) + " > 0",
    RUNAWAY + " WHERE " + " + ".join([_LIKE] * 20) + " > 0",
)

# Ten rows, each a handful of bounded-but-slow calls (ltrim with a large character set),
# together under PROGRESS_STEPS virtual-machine steps. The progress handler fires only
# every PROGRESS_STEPS steps, so on its own it never reaches a check and the query runs to
# completion; only the watchdog's interrupt at the deadline stops it.
_WATCHDOG_CALL = "length(ltrim(printf('%.*c', 3600, 'a') || x, printf('%.*c', 340, 'b') || 'a'))"
WATCHDOG_ROWS = (
    "WITH r(x) AS (VALUES "
    + ", ".join(f"({i})" for i in range(1, 11))
    + ") SELECT count(*) FROM r WHERE ("
    + " + ".join([_WATCHDOG_CALL] * 10)
    + ") >= 0"
)


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
        # SQLite resolves names before it asks the authorizer, so these fail on the
        # missing name first; they are still writes.
        "DELETE FROM forge_missing",
        "INSERT INTO forge_missing VALUES (1)",
        "UPDATE head SET forge_missing = 1",
        "ALTER TABLE head DROP COLUMN forge_missing",
        "DROP INDEX forge_missing",
        "WITH s AS (SELECT 1) DELETE FROM forge_missing",
        "DELETE FROM head WHERE name = 'unclosed",
        # Statements that read but are not queries.
        "EXPLAIN SELECT * FROM head",
        "EXPLAIN QUERY PLAN SELECT * FROM head",
        # Keywords only SQLite's own reading finds: behind a byte order mark, after a
        # table named with a long s (U+017F), or after a parameter whose Tcl-style suffix
        # holds parentheses or a quote.
        "\ufeffDELETE FROM forge_missing",
        "\ufeffEXPLAIN SELECT * FROM head",
        "WITH \u017felect AS (SELECT 1) DELETE FROM forge_missing",
        "WITH s AS (SELECT $a(() ) DELETE FROM forge_missing",
        "WITH s AS (SELECT @a(') ) UPDATE head SET forge_missing = 1 -- '",
    ]

    for statement in statements:
        result = executor.run(DB, statement)
        assert (result.ok, result.error_kind) == (False, "denied"), (statement, result)
        assert result.columns is None
        assert result.rows is None

    assert executor.run(DB, "SELECT count(*) FROM head").rows == [(10,)]
    # A table named like a statement keyword is still a query.
    named = "WITH replace AS (SELECT count(*) AS n FROM head) SELECT n FROM replace"
    assert executor.run(DB, named).rows == [(10,)]
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
def test_run_many_keeps_order_across_workers(executor: SqliteExecutor) -> None:
    items = [
        (DB, "SELECT count(*) FROM department"),
        ("flight_1", "SELECT count(*) FROM aircraft"),
        (DB, "DELETE FROM head"),
        (DB, RUNAWAY),
        ("dorm_1", "SELECT 1; SELECT 2"),
        (DB, _counting(30)),
    ] * 4
    one = executor.run_many(items, workers=1, timeout_ms=100, max_rows=20)
    two = executor.run_many(items, workers=2, timeout_ms=100, max_rows=20)

    def outcome(result: Any) -> tuple[Any, ...]:
        return (result.ok, result.error_kind, result.columns, result.rows, result.truncated)

    # The same as running them here, one at a time.
    assert [outcome(r) for r in one] == [
        outcome(executor.run(*item, timeout_ms=100, max_rows=20)) for item in items
    ]
    assert [outcome(r) for r in two] == [outcome(r) for r in one]
    assert [r.error_kind for r in one[:6]] == [None, None, "denied", "timeout", "denied", None]
    assert one[5].truncated

    # A worker's error reaches the caller rather than becoming a result.
    with pytest.raises(DataNotFound, match="no_such_db"):
        executor.run_many([(DB, "SELECT 1"), ("no_such_db", "SELECT 1")], workers=2)


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
def test_timeout_holds_for_expensive_rows(executor: SqliteExecutor) -> None:
    for sql in SLOW_ROWS:
        started = time.perf_counter()
        result = executor.run(DB, sql, timeout_ms=300)
        elapsed_ms = (time.perf_counter() - started) * 1000

        assert (result.ok, result.error_kind) == (False, "timeout"), (sql, result)
        # These many-row queries run enough VM steps that the progress handler stops them
        # near the deadline; the watchdog's separate job (rows the handler never sees) is
        # test_watchdog_stops_a_query_the_handler_never_checks. The margin is for slow CI.
        assert elapsed_ms < 300 + 700, sql
    assert executor.run(DB, _counting(3)).rows == [(1,), (2,), (3,)]


@pytest.mark.phase(1)
def test_watchdog_stops_a_query_the_handler_never_checks(executor: SqliteExecutor) -> None:
    # This fails if the watchdog is removed: without it only the progress handler is left,
    # and for this query (under PROGRESS_STEPS steps) it never fires, so the query runs to
    # completion instead of stopping at the deadline.
    started = time.perf_counter()
    result = executor.run(DB, WATCHDOG_ROWS, timeout_ms=100)
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert (result.ok, result.error_kind) == (False, "timeout"), result
    assert elapsed_ms < 100 + 700
    # Interrupting leaves the connection usable, with no deadline left over.
    assert executor.run(DB, _counting(3)).rows == [(1,), (2,), (3,)]


def _one_row(x: str, y: str, call: str, op: str) -> str:
    """One row adding up compact calls on ``x`` and ``y``, as many as 8 kB of SQL holds.

    Each argument is a one-row materialized CTE, so every call reads it from a column:
    identical calls on constants would be worked out once.
    """
    head = f"WITH t(x) AS MATERIALIZED (SELECT {x}), u(y) AS MATERIALIZED (SELECT {y}) SELECT "
    tail = " AS v FROM t, u"  # named, so the column's name isn't its 8 kB of text
    sql_bytes = LIMITS[sqlite3.SQLITE_LIMIT_SQL_LENGTH]
    n = (sql_bytes - len(head) - len(tail) + len(op)) // (len(call) + len(op))
    sql = head + op.join([call] * n) + tail
    assert len(sql.encode()) <= sql_bytes
    return sql


def _repeat(n: int, char: str) -> str:
    return f"printf('%.*c', {n}, '{char}')"


# The longest value a one-column materialized row holds under the 4 kB length limit.
LONGEST = MAX_VALUE_BYTES - 16


@pytest.mark.phase(1)
def test_in_process_deadline_is_soft_for_one_row(executor: SqliteExecutor) -> None:
    # SQLite can't stop the work inside one row, and the limits don't bound how much a row
    # holds. ltrim with a large character set costs far more a call than the functions
    # test_limits_bound_the_work_in_one_row measures, and a CTE that is referenced several
    # times is copied into each reference, so nesting multiplies even cheap calls past
    # what the statement's text holds: here 10,000 instr calls in under 400 bytes.
    trims = _one_row(_repeat(1500, "a"), _repeat(340, "b") + " || 'a'", "ltrim(x,y)", "||")
    nested = (
        f"WITH t(x) AS MATERIALIZED (SELECT {_repeat(LONGEST, 'a')}), "
        f"u(y) AS MATERIALIZED (SELECT {_repeat(LONGEST // 2 - 1, 'a')} || 'b'), "
        f"c1(v) AS (SELECT {'+'.join(['instr(x,y)'] * 10)} FROM t, u), "
        f"c2(v) AS (SELECT {'+'.join(['v'] * 10)} FROM c1), "
        f"c3(v) AS (SELECT {'+'.join(['v'] * 10)} FROM c2) "
        f"SELECT {'+'.join(['v'] * 10)} AS v FROM c3"
    )
    assert len(nested) < 400
    with SandboxPool(1, data_dir=executor.data_dir) as pool:
        pool.start()
        for sql in (trims, nested):
            started = time.perf_counter()
            soft = executor.run(DB, sql, timeout_ms=100)
            in_process_ms = (time.perf_counter() - started) * 1000
            assert soft.error_kind == "timeout", sql[:60]
            # The one row runs past even the pool's hard deadline, the timeout and the
            # kill grace.
            assert in_process_ms > 100 + KILL_GRACE_MS, (sql[:60], in_process_ms)

            # What bounds the same row is the pool's kill of its worker.
            killed = pool.run(DB, sql, timeout_ms=100)
            assert killed.error_kind == "timeout", sql[:60]
            assert killed.error is not None
            assert killed.error.startswith("killed after "), killed.error
            assert killed.ms < 100 + KILL_GRACE_MS + 1000, killed.ms
        assert pool.restarts == 2


@pytest.mark.phase(1)
def test_limits_bound_each_value(executor: SqliteExecutor, spider_mini: Path) -> None:
    with closing(connect_readonly(_db_file(spider_mini))) as conn:
        assert {limit: conn.getlimit(limit) for limit in LIMITS} == LIMITS
    assert MAX_VALUE_BYTES == 4096

    # Each of these used to run for seconds, or allocate gigabytes, and return ok.
    cases = {
        "SELECT length(randomblob(900000000))": "string or blob too big",
        "SELECT count(*) FROM head WHERE length(randomblob(300000000)) > 0": (
            "string or blob too big"
        ),
        "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r WHERE x < 150) "
        "SELECT zeroblob(2000000) FROM r": "string or blob too big",
        "SELECT length(zeroblob(4097))": "string or blob too big",
        # The FG-191 review's queries, which ran for seconds under 100 kB (ADR-0014).
        review_query(100): "string or blob too big",
        review_query(200): "string or blob too big",
        "SELECT printf('%.*c', 4000, 'a') LIKE ('%' || printf('%.*c', 100, 'a') || 'b')": (
            "LIKE or GLOB pattern too complex"
        ),
        "SELECT " + " + ".join(["1"] * 30_000): "query string is too large",
        "SELECT 1" + " " * 8185: "query string is too large",
        "SELECT " + ", ".join(["1"] * 201): "too many columns in result set",
    }
    for sql, error in cases.items():
        started = time.perf_counter()
        result = executor.run(DB, sql, timeout_ms=300)
        elapsed_ms = (time.perf_counter() - started) * 1000

        assert (result.ok, result.error_kind, result.error) == (False, "other", error), sql[:60]
        assert elapsed_ms < 300, sql[:60]

    # Values, patterns, statements and rows up to the limits still work.
    assert executor.run(DB, "SELECT length(randomblob(4096))").rows == [(4096,)]
    like = "SELECT printf('%.*c', 4000, 'a') LIKE ('%' || printf('%.*c', 99, 'a'))"
    assert executor.run(DB, like).rows == [(1,)]
    assert executor.run(DB, "SELECT 1" + " " * 8184).rows == [(1,)]
    assert executor.run(DB, "SELECT " + ", ".join(["1"] * 200)).rows == [(1,) * 200]


@pytest.mark.phase(1)
def test_limits_bound_the_work_in_one_row(executor: SqliteExecutor) -> None:
    # The cost of instr, replace, LIKE and GLOB grows with the product of their inputs'
    # lengths, so the length and pattern limits keep each call cheap. Here each call takes
    # the longest inputs those allow from a column, in compact calls that fill the 8 kB
    # statement limit: 0.07 s for instr and 0.34 to 0.38 s for LIKE or GLOB on the laptop
    # (Apple M4), where with FG-191's 100 kB values the instr row ran 32 s. That bounds
    # only calls written out in the statement; see
    # test_in_process_deadline_is_soft_for_one_row for the rest.
    pattern = LIMITS[sqlite3.SQLITE_LIMIT_LIKE_PATTERN_LENGTH]
    haystack = _repeat(LONGEST, "a")
    needle = _repeat(LONGEST // 2 - 1, "a") + " || 'b'"
    wildcard = {"like": "'%'", "glob": "'*'"}
    rows = {
        "instr": _one_row(haystack, needle, "instr(x,y)", "+"),
        "replace": _one_row(haystack, needle, "replace(x,y,'')", "+"),
        **{
            name: _one_row(
                haystack, f"{first} || {_repeat(pattern - 2, 'a')} || 'b'", f"{name}(y,x)", "+"
            )
            for name, first in wildcard.items()
        },
    }
    for name, sql in rows.items():
        started = time.perf_counter()
        result = executor.run(DB, sql, timeout_ms=10_000)
        elapsed_ms = (time.perf_counter() - started) * 1000

        assert result.ok, (name, result.error)
        assert elapsed_ms < 2000, (name, elapsed_ms)


@pytest.mark.phase(1)
def test_result_bytes_capped(executor: SqliteExecutor) -> None:
    blobs = (
        "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r WHERE x < {n}) "
        "SELECT x, " + ", ".join(["zeroblob(4000)"] * 10) + " FROM r"
    )
    assert MAX_RESULT_BYTES == 64 * 2**20

    # 2,000 rows of 40 kB is about 81 MB, under the row cap but over the budget.
    over = executor.run(DB, blobs.format(n=2000))
    assert (over.ok, over.error_kind) == (False, "other")
    assert over.error == "result too big: its rows take more than 64 MiB"
    assert over.rows is None

    under = executor.run(DB, blobs.format(n=1000))
    assert under.ok, under.error
    assert [row[0] for row in under.rows or ()] == list(range(1, 1001))


@pytest.mark.phase(1)
def test_wide_rows_count_what_they_take(executor: SqliteExecutor) -> None:
    # 200 numbers a row: 1.6 kB at 8 bytes a value, as FG-191 counted, but 7.2 kB as Python
    # objects (a 1.6 kB tuple and 200 ints of 28 bytes). 10,000 rows take 72 MB.
    wide = (
        "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r WHERE x < {n}) SELECT "
        + ", ".join(f"x + {1000 + i}" for i in range(200))
        + " FROM r"
    )
    under = executor.run(DB, wide.format(n=5000))
    assert under.ok, under.error
    rows = under.rows or []
    assert len(rows) == 5000
    assert sum(map(row_size, rows)) == 5000 * (sys.getsizeof(rows[0]) + 8 + 200 * 28)

    over = executor.run(DB, wide.format(n=10_000))
    assert (over.ok, over.error) == (False, "result too big: its rows take more than 64 MiB")


@pytest.mark.phase(1)
def test_run_many_bounds_each_item(executor: SqliteExecutor) -> None:
    items = [(DB, SLOW_ROWS[0]), (DB, "SELECT count(*) FROM head"), (DB, SLOW_ROWS[1])] * 2

    results = executor.run_many(items, workers=2, timeout_ms=300)

    assert [r.error_kind for r in results] == ["timeout", None, "timeout"] * 2
    # Each item stops at its own deadline inside the worker.
    assert all(r.ms < 300 + 700 for r in results), [round(r.ms) for r in results]


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
def test_multiple_statements_rejected(executor: SqliteExecutor) -> None:
    for sql in (
        "SELECT 1; SELECT 2",
        "SELECT count(*) FROM head; DELETE FROM head",
        "SELECT 1;\nDROP TABLE head;",
        "SELECT 1;;",
        "SELECT ';'; SELECT 2",
        "SELECT 1 /* ; */; SELECT 2 -- ;",
    ):
        result = executor.run(DB, sql)
        assert (result.ok, result.error_kind) == (False, "denied"), (sql, result)
        assert result.error == "more than one statement; only a single statement may run"
    assert executor.run(DB, "SELECT count(*) FROM head").rows == [(10,)]

    # One statement, however it is dressed: semicolons in strings, names and comments,
    # stray leading semicolons, and a trailing semicolon or comment.
    for sql, rows in (
        ("SELECT 1;", [(1,)]),
        ("  SELECT 1 ;  -- done", [(1,)]),
        ("SELECT 1; /* unterminated", [(1,)]),
        (";SELECT 1", [(1,)]),
        ("SELECT 'a;b', 'it''s; fine'", [("a;b", "it's; fine")]),
        ('SELECT 1 AS [x;y], 2 AS "p;q", 3 AS `r;s`', [(1, 2, 3)]),
        ("SELECT /* ; */ 1 -- ;\n", [(1,)]),
    ):
        result = executor.run(DB, sql)
        assert (result.ok, result.rows) == (True, rows), (sql, result)


@pytest.mark.phase(1)
def test_empty_sql(executor: SqliteExecutor) -> None:
    for sql in ("", "   \n\t", "-- just a comment", "/* nothing */", ";", " ; ; "):
        result = executor.run(DB, sql)
        assert (result.ok, result.error_kind, result.error) == (False, "empty_sql", "no SQL"), sql


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


@pytest.mark.phase(1)
@pytest.mark.requires_full_data
def test_demo_delete_is_denied_on_full_data() -> None:
    # The Phase 1 demo command, on the full download.
    result = CliRunner().invoke(
        cli.app, ["sql", "run", "--db", "concert_singer", "DELETE FROM singer"]
    )

    assert result.exit_code == 1
    assert result.stdout.startswith("denied: ")


@pytest.mark.phase(1)
@pytest.mark.requires_full_data
def test_every_training_gold_query_runs() -> None:
    # The allowlist and the statement count must refuse nothing Spider's authors wrote.
    # Train and val, in sandbox workers; the limits test below adds dev.
    examples = [example for example in load_raw() if example.split == "train"]
    items = [(example.db_id, example.gold_sql) for example in examples]
    with SqliteExecutor() as executor:
        # A generous timeout: this checks what runs, not how fast (the slowest takes ~1 s).
        results = executor.run_many(items, workers=min(4, os.cpu_count() or 1), timeout_ms=20_000)

    failed = {e.id: r.error for e, r in zip(examples, results, strict=True) if not r.ok}
    assert set(failed) == FAILING_GOLD_IDS, failed


@pytest.mark.phase(1)
@pytest.mark.requires_full_data
def test_every_gold_query_runs_under_the_limits() -> None:
    # Train, val and dev, through the executor and through the evaluator's own connections,
    # which have the same limits. The evaluator runs gold as eval_exec_match does, with
    # every DISTINCT removed (keep_distinct=False), so that is what its row and byte
    # ceilings meet. Gold SQL only, no model, so this is not a dev use (ADR-0004).
    examples = load_raw()
    assert len(examples) == 9693
    failed: dict[str, str | None] = {}
    shapes: list[tuple[int, int, int, str]] = []
    as_run: list[tuple[int, int, str]] = []
    with SqliteExecutor() as executor:
        for e in examples:
            result = executor.run(e.db_id, e.gold_sql, timeout_ms=20_000, max_rows=10**6)
            if not result.ok:
                failed[e.id] = result.error
                continue
            rows = result.rows or []
            shapes.append((len(result.columns or ()), len(rows), sum(map(row_size, rows)), e.id))
            gold = remove_distinct(exec_eval.postprocess(e.gold_sql))
            flag, value = asyncio.run(exec_eval.exec_on_db(str(db_path(e.db_id)), gold))
            if flag != "result":
                failed[e.id] = f"evaluator: {value}"
                continue
            as_run.append((len(value), sum(map(exec_eval.row_size, value)), e.id))
    assert set(failed) == FAILING_GOLD_IDS, failed

    # What the limits were set from (ADR-0014), each well inside its limit.
    longest_sql = max(len(e.gold_sql.encode()) for e in examples)
    like = re.compile(r"""\blike\s+('(?:[^']|'')*'|"(?:[^"]|"")*")""", re.IGNORECASE)
    longest_like = max(len(m.group(1)) - 2 for e in examples for m in like.finditer(e.gold_sql))
    widest = max(shapes)
    longest = max(shapes, key=lambda shape: shape[1])
    biggest = max(shapes, key=lambda shape: shape[2])
    assert (longest_sql, longest_like) == (577, 42)
    assert (widest[0], widest[3]) == (19, "spider-dev-00755")
    assert longest[1:] == biggest[1:] == (20_662, 3_305_817, "spider-dev-00455")
    assert 10 * longest_sql < LIMITS[sqlite3.SQLITE_LIMIT_SQL_LENGTH]
    assert 2 * longest_like < LIMITS[sqlite3.SQLITE_LIMIT_LIKE_PATTERN_LENGTH]
    assert 10 * widest[0] < LIMITS[sqlite3.SQLITE_LIMIT_COLUMN]
    assert 10 * biggest[2] < MAX_RESULT_BYTES == exec_eval.MAX_RESULT_BYTES

    # Without its DISTINCT a soccer_1 training gold returns 93,475 rows, close under the
    # evaluator's 100,000; the val and dev golds, which official EX scores, keep 4 times
    # the room.
    most_rows = max(as_run)
    assert most_rows == max(as_run, key=lambda shape: shape[1])
    assert most_rows == (93_475, 10_329_330, "spider-train-01297")
    assert most_rows[0] < exec_eval.MAX_ROWS
    assert 6 * most_rows[1] < exec_eval.MAX_RESULT_BYTES
    splits = split_examples(examples)
    held_out = {e.id for name in ("val", "dev") for e in splits[name]}
    assert 4 * max(rows for rows, _, id_ in as_run if id_ in held_out) < exec_eval.MAX_ROWS


@pytest.mark.phase(1)
@pytest.mark.requires_full_data
def test_spider_data_fits_the_limits() -> None:
    # Every table of every database, dev's too: no data that SQLite has to read can be cut
    # off by the limits, with room to spare.
    paths = sorted((resolve_data_dir() / "database").glob("*/*.sqlite"))
    assert len(paths) == 166
    largest_value = largest_schema = widest_row = most_columns = 0
    for path in paths:
        with closing(connect_readonly(path)) as conn:
            conn.text_factory = bytes  # sizes as stored
            for (sql,) in conn.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL"):
                largest_schema = max(largest_schema, len(sql))
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            for (table,) in tables.fetchall():
                quoted = '"' + table.decode().replace('"', '""') + '"'
                # DISTINCT stores each row as a record, which the length limit also bounds.
                cursor = conn.execute(f"SELECT DISTINCT * FROM {quoted}")
                most_columns = max(most_columns, len(cursor.description))
                for row in cursor:
                    sizes = [len(v) if isinstance(v, bytes) else 8 for v in row]
                    largest_value = max(largest_value, *sizes)
                    widest_row = max(widest_row, sum(sizes))

    # The largest value is in cre_Doc_Tracking_DB, the largest CREATE statement in
    # soccer_1, and the widest table is baseball_1's team, which holds the widest row.
    assert (largest_value, largest_schema, widest_row, most_columns) == (213, 1191, 355, 48)
    assert 3 * largest_schema < MAX_VALUE_BYTES
    assert 4 * most_columns < LIMITS[sqlite3.SQLITE_LIMIT_COLUMN]
