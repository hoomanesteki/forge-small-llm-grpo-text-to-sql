"""Rules for SQLite access that need no Spider database."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "forge"


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
