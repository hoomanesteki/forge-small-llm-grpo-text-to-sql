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
}


@pytest.mark.phase(1)
@pytest.mark.parametrize(("sql", "keyword"), STATEMENT_KEYWORDS.items())
def test_statement_keyword(sql: str, keyword: str | None) -> None:
    assert statement_keyword(sql) == keyword


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
