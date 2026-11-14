"""Executor logic that needs no Spider database: statement counting, error kinds, limits."""

from __future__ import annotations

import ast
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from forge.sql.executor import SqliteExecutor, classify_error, statement_count

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
