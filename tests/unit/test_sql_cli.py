"""``forge sql run``: prints the result of one query, and fails when the query does."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from forge import cli

DB = "department_management"


def _run(*args: str) -> tuple[int, list[str], str]:
    result = CliRunner().invoke(cli.app, ["sql", "run", "--db", DB, *args])
    return result.exit_code, result.stdout.splitlines(), result.stderr


@pytest.mark.phase(1)
def test_sql_run_prints_rows(spider_mini: Path) -> None:
    code, lines, _ = _run("SELECT Department_ID, Name FROM department ORDER BY 1 LIMIT 2")

    assert code == 0
    assert lines[:3] == ["Department_ID | Name", "1 | State", "2 | Treasury"]
    assert re.fullmatch(r"2 rows \(\d+\.\d ms\)", lines[3])

    code, lines, _ = _run("SELECT NULL, X'00ff', 1.5, 'one row'")
    assert code == 0
    assert lines[1] == "NULL | X'00ff' | 1.5 | one row"
    assert lines[2].startswith("1 row (")

    code, lines, _ = _run("SELECT Name FROM department ORDER BY 1", "--max-rows", "2")
    assert code == 0
    assert len(lines) == 4
    assert lines[3].startswith("2 rows, truncated at --max-rows 2 (")


@pytest.mark.phase(1)
def test_sql_run_reports_failures(spider_mini: Path) -> None:
    code, lines, _ = _run("DELETE FROM department")
    assert code == 1
    denied = r"denied: DELETE is not a query; only SELECT and VALUES statements may run"
    assert re.fullmatch(denied + r" \(\d+\.\d ms\)", lines[0])

    runaway = "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) SELECT max(x) FROM r"
    code, lines, _ = _run(runaway, "--timeout-ms", "50")
    assert code == 1
    assert lines[0].startswith("timeout: timed out after 50 ms (")

    result = CliRunner().invoke(cli.app, ["sql", "run", "--db", "no_such_db", "SELECT 1"])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "no_such_db.sqlite not found" in result.stderr
    assert "forge data fetch" in result.stderr
