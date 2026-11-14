"""Executor logic that needs no Spider database: error kinds and limits."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from forge.sql.executor import SqliteExecutor, classify_error

SRC = Path(__file__).resolve().parents[2] / "src" / "forge"


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
