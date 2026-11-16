"""Executor logic that needs no Spider database: statement counting, error kinds, limits."""

from __future__ import annotations

import ast
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from forge.sql.executor import (
    SqliteExecutor,
    classify_error,
    comments,
    split_statements,
    statement_count,
    statement_keyword,
    strip_leading,
)

SRC = Path(__file__).resolve().parents[2] / "src" / "forge"

STATEMENT_COUNTS = {
    "": 0,
    " \n\t": 0,
    "-- comment": 0,
    "/* comment */": 0,
    "/* never closed": 0,
    ";": 0,
    " ; ; ": 0,
    "SELECT 1": 1,
    "SELECT 1;": 1,
    ";SELECT 1": 1,
    ";; SELECT 1 ;": 1,
    "SELECT 1; -- done": 1,
    "SELECT 1; /* done */ ": 1,
    "SELECT 1; /* never closed": 1,
    "SELECT 4/2, 3 - -1": 1,
    "SELECT ';'": 1,
    "SELECT 'it''s; fine'": 1,
    'SELECT "a;b"': 1,
    "SELECT `a;b`": 1,
    "SELECT [a;b]": 1,
    "SELECT 1 -- ;\n": 1,
    "SELECT /* ; */ 1": 1,
    "SELECT 'never closed;": 1,
    "SELECT [never closed;": 1,
    "SELECT 1;;": 2,
    "SELECT 1; SELECT 2": 2,
    "SELECT 1 -- ;\n; SELECT 2": 2,
    "SELECT 1;\nDROP TABLE t;": 2,
    "SELECT 1; x": 2,
    "SELECT 1; SELECT 2; SELECT 3": 3,
    # SQLite skips a vertical tab only inside a run of whitespace that starts with one of
    # the other five, and the check after the first statement never skips it.
    " \v": 0,
    "\t\v\v\nSELECT 1": 1,
    " \v/* x */;SELECT 1;": 1,
    "; \vSELECT 1": 1,
    "SELECT 1 \v": 1,
    "SELECT 1 -- c\n\v;": 1,
    "\v": 1,
    "SELECT 1;\v": 2,
    "SELECT 1; \v": 2,
    "SELECT 1;\n\v": 2,
    # SQLite skips a byte order mark wherever a token starts; Python's check after the
    # first statement doesn't.
    "\ufeff": 0,
    "\ufeff;\ufeffSELECT 1": 1,
    "SELECT \ufeff1 \ufeff": 1,
    "SELECT 1;\ufeff": 2,
    "SELECT 1; \ufeff": 2,
    # A parameter's Tcl-style suffix runs to whitespace or ")", and takes a ";" with it.
    "SELECT $a(;)": 1,
    "SELECT :a(x y);": 1,
    "SELECT $a(') ; SELECT 2 -- '": 2,
}


@pytest.mark.phase(1)
@pytest.mark.parametrize(("sql", "count"), STATEMENT_COUNTS.items())
def test_statement_count(sql: str, count: int) -> None:
    assert statement_count(sql) == count


@pytest.mark.phase(1)
def test_statement_count_agrees_with_sqlite3() -> None:
    # CPython's execute refuses anything but whitespace and comments after the first
    # statement (3.12 raises ProgrammingError, 3.11 sqlite3.Warning). Wherever it does,
    # statement_count must see more than one, so its check is never the weaker one.
    with closing(sqlite3.connect(":memory:")) as conn:
        for sql, count in STATEMENT_COUNTS.items():
            try:
                conn.execute(sql).close()
                refused = False
            except (sqlite3.ProgrammingError, sqlite3.Warning) as exc:
                refused = "one statement at a time" in str(exc)
            except sqlite3.OperationalError:
                refused = False  # unclosed quotes and unknown names: a single bad statement
            assert refused == (count > 1), sql


# The keyword that decides what a statement does: None when the first word is no statement
# keyword, which SQLite then reports as a syntax error.
STATEMENT_KEYWORDS: dict[str, str | None] = {
    "SELECT 1": "SELECT",
    "  select 1": "SELECT",
    "SELECT*FROM t": "SELECT",
    ";; /* c */ -- d\n VALUES (1)": "VALUES",
    "WITH s AS (SELECT 1) SELECT * FROM s": "SELECT",
    "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) SELECT x FROM r": "SELECT",
    'WITH "delete" AS (SELECT 1) SELECT 2': "SELECT",
    "WITH s AS (SELECT 1) VALUES (2)": "VALUES",
    "with s(a, b) as not materialized (select 1, 2) delete from t": "DELETE",
    "WITH s AS (SELECT 1), u AS (SELECT 2) INSERT INTO t SELECT * FROM u": "INSERT",
    "WITH s AS (SELECT 1)UPDATE t SET a = 1": "UPDATE",
    "WITH s AS (SELECT (1)) REPLACE INTO t VALUES (1)": "REPLACE",
    "DELETE FROM nowhere": "DELETE",
    "Drop Table t": "DROP",
    "ALTER TABLE t DROP COLUMN nowhere": "ALTER",
    "ATTACH DATABASE CASE WHEN 1 > = 1 THEN 'x' END AS e": "ATTACH",
    "explain select 1": "EXPLAIN",
    "PRAGMA user_version": "PRAGMA",
    "VACUUM INTO 'x'": "VACUUM",
    "END": "END",
    "SELEC * FROM t": None,
    "SELECT1": None,
    "DISTINCT DELETE FROM t": None,
    "(SELECT 1)": None,
    "'SELECT'": None,
    '"DELETE" FROM t': None,
    "éSELECT 1": None,
    "\x00DROP TABLE t": None,
    "\vDROP TABLE t": None,
    "WITH s AS (SELECT 1)": None,
    "": None,
    "-- DELETE FROM t": None,
    # Keywords are ASCII to SQLite: a long s (U+017F) or a dotless i (U+0131) upper-cases
    # to a keyword's letters in Python, but the word is a name.
    "\u017felect 1": None,
    "WITH \u017felect AS (SELECT 1) DELETE FROM nowhere": "DELETE",
    "WITH value\u017f AS (SELECT 1) DELETE FROM nowhere": "DELETE",
    "WITH \u0131nsert AS (SELECT 1 AS a) SELECT a FROM \u0131nsert": "SELECT",
    # A table may be named like a keyword that can follow the WITH clause.
    "WITH replace AS (SELECT 1 AS a) SELECT a FROM replace": "SELECT",
    "WITH a AS (SELECT 1), replace(x) AS (SELECT 2) SELECT x FROM replace": "SELECT",
    "WITH 'delete' AS MATERIALIZED (SELECT 1) SELECT 2": "SELECT",
    "WITH s AS (SELECT 1) , u AS NOT MATERIALIZED (SELECT 2) UPDATE t SET a = 1": "UPDATE",
    "WITH s AS (SELECT 1) nothing": None,
    "WITH s AS (SELECT 1; DELETE FROM t": None,
    "WITH recursive AS (SELECT 1) SELECT 2": None,
    # A byte order mark is whitespace where a token starts and part of a word inside one.
    "\ufeffDELETE FROM nowhere": "DELETE",
    "\ufeff\ufeffEXPLAIN SELECT 1": "EXPLAIN",
    "\ufeff; \ufeffSELECT 1": "SELECT",
    "WITH s AS (SELECT 1)\ufeffDELETE FROM nowhere": "DELETE",
    "SELECT\ufeff1": None,
    # A parameter's Tcl-style suffix holds parentheses and quotes that SQLite never sees.
    "WITH s AS (SELECT $a(() ) DELETE FROM nowhere": "DELETE",
    "WITH s AS (SELECT @a(') ) DELETE FROM nowhere -- '": "DELETE",
    "WITH s AS (SELECT :a(x)) DELETE FROM nowhere": "DELETE",
    "WITH s AS (SELECT #a()), u AS (SELECT $b::c(()) INSERT INTO t VALUES (1)": "INSERT",
}


@pytest.mark.phase(1)
@pytest.mark.parametrize(("sql", "keyword"), STATEMENT_KEYWORDS.items())
def test_statement_keyword(sql: str, keyword: str | None) -> None:
    assert statement_keyword(sql) == keyword


_SQLITE_READS = frozenset(
    {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
)
_SQLITE_WRITES = frozenset({sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE})


def _what_sqlite_prepares(sql: str) -> str:
    """``read``, ``write``, ``nothing`` or ``error``: what SQLite makes of ``sql``, by what
    it asks the authorizer while it prepares the statement on a scratch database."""
    actions: list[int] = []

    def record(action: int, *_: object) -> int:
        actions.append(action)
        return sqlite3.SQLITE_OK if action in _SQLITE_READS else sqlite3.SQLITE_DENY

    with closing(sqlite3.connect(":memory:")) as conn:
        conn.executescript("CREATE TABLE t (a); CREATE TABLE nowhere (a);")
        conn.set_authorizer(record)
        try:
            conn.execute(sql).close()
            failed = False
        except sqlite3.ProgrammingError as exc:
            # Parameters are only bound after the statement is prepared.
            failed = "Incorrect number of bindings" not in str(exc)
        except (sqlite3.Error, ValueError):
            failed = True
    if _SQLITE_WRITES.intersection(actions):
        return "write"
    if failed:
        return "error"
    return "read" if actions else "nothing"


@pytest.mark.phase(1)
def test_statement_keywords_agree_with_sqlite() -> None:
    # The table above, checked against SQLite itself: a query keyword must be a statement
    # SQLite prepares as a read, a write keyword one it prepares as a write, and None one
    # it can't prepare or that holds nothing. Other keywords are the statement's first
    # word, which needs no check.
    expected = {"SELECT": "read", "VALUES": "read", "DELETE": "write", "INSERT": "write"}
    expected |= {"REPLACE": "write", "UPDATE": "write"}
    for sql, keyword in STATEMENT_KEYWORDS.items():
        if keyword is None:
            assert _what_sqlite_prepares(sql) in ("error", "nothing"), sql
        elif keyword in expected:
            assert _what_sqlite_prepares(sql) == expected[keyword], sql


@pytest.mark.phase(1)
def test_strip_leading() -> None:
    # What SQLite skips before the first statement, and nothing else.
    cases = {
        "SELECT 1": "SELECT 1",
        "; SELECT 1;": "SELECT 1;",
        " ;\n-- note\n/* c */ ;  SELECT 1 -- done": "SELECT 1 -- done",
        " \v; SELECT 1": "SELECT 1",
        "\vSELECT 1": "\vSELECT 1",
        "  ; ": "",
        "": "",
    }
    for sql, stripped in cases.items():
        assert strip_leading(sql) == stripped, sql


@pytest.mark.phase(5)
def test_comments() -> None:
    cases: dict[str, list[str]] = {
        "SELECT 1": [],
        "SELECT 1 -- note": [" note"],
        "-- only a comment\n": [" only a comment"],
        "SELECT /* a */ 1 /* b */;": [" a ", " b "],
        "SELECT 1 --a\n--b": ["a", "b"],
        "SELECT 1; -- after": [" after"],
        "/**/": [""],
        # Left open, a block comment runs to the end, as SQLite reads it.
        "SELECT 1 /* never closed": [" never closed"],
        "/*/": ["/"],
        "SELECT '-- no', \"/* nor */\", [-- this], `/* or */`": [],
        "SELECT 4/2 - -1": [],
    }
    for sql, found in cases.items():
        assert comments(sql) == found, sql
    # SQLite agrees that the open comment is one: the query runs.
    with closing(sqlite3.connect(":memory:")) as conn:
        assert conn.execute("SELECT 1 /* never closed").fetchall() == [(1,)]


@pytest.mark.phase(5)
def test_split_statements() -> None:
    cases: dict[str, list[str]] = {
        "SELECT 1": ["SELECT 1"],
        "SELECT 1; -- done": ["SELECT 1"],
        "-- c\n;SELECT 1": ["SELECT 1"],
        "SELECT 1;;; DELETE FROM t ;": ["SELECT 1", "DELETE FROM t"],
        "SELECT /* a; */ 1 -- b;\n, 2; SELECT ';'": ["SELECT /* a; */ 1 -- b;\n, 2", "SELECT ';'"],
        # Left open, the comment swallows the rest, semicolons and all.
        "DELETE FROM t; SELECT 1 /* x; y": ["DELETE FROM t", "SELECT 1"],
        "SELECT 1;﻿PRAGMA x": ["SELECT 1", "PRAGMA x"],
        " ; -- nothing\n": [],
    }
    for sql, pieces in cases.items():
        assert split_statements(sql) == pieces, sql
        assert len(pieces) <= statement_count(sql), sql


@pytest.mark.phase(1)
def test_classify_error() -> None:
    cases = {
        "not authorized": "denied",
        "authorization denied": "denied",
        "You can only execute one statement at a time.": "denied",
        "no such table: main.flight": "no_such_table",
        "no such column: t.name": "no_such_column",
        'near "SELEC": syntax error': "syntax",
        'unrecognized token: "\'abc"': "syntax",
        "incomplete input": "syntax",
        # A name inside the message can't change its kind.
        "no such column: not authorized": "no_such_column",
        'near "no such table: x": syntax error': "syntax",
        "no such table: no such column: x": "no_such_table",
        # Timeouts come from the executor's own deadline, never from a message.
        "interrupted": "other",
        "no such function: median": "other",
        "ambiguous column name: id": "other",
        "": "other",
    }
    for message, kind in cases.items():
        assert classify_error(message) == kind, message


@pytest.mark.phase(1)
def test_limits_must_be_positive(tmp_path: Path) -> None:
    executor = SqliteExecutor(tmp_path)
    with pytest.raises(ValueError, match="timeout_ms must be positive"):
        executor.run("any", "SELECT 1", timeout_ms=0)
    with pytest.raises(ValueError, match="max_rows must be zero or more"):
        executor.run("any", "SELECT 1", max_rows=-1)
    with pytest.raises(ValueError, match="workers must be at least 1"):
        executor.run_many([("any", "SELECT 1")], workers=0)


@pytest.mark.phase(1)
def test_only_the_executor_connects_to_sqlite() -> None:
    # CONTRIBUTING rule 9: Spider databases are opened in one place, read-only.
    allowed = SRC / "sql" / "executor.py"
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if path == allowed:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            calls_connect = (
                isinstance(node, ast.Attribute)
                and node.attr == "connect"
                and isinstance(node.value, ast.Name)
                and node.value.id == "sqlite3"
            )
            imports_connect = (
                isinstance(node, ast.ImportFrom)
                and node.module == "sqlite3"
                and any(alias.name == "connect" for alias in node.names)
            )
            if calls_connect or imports_connect:
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert offenders == []
