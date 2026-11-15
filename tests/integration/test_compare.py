"""Pipeline stages 5 to 7 on the fixture databases: extract, execute, compare."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from forge.data.spider import load_raw
from forge.sql.compare import order_matters, results_match
from forge.sql.executor import SqliteExecutor
from forge.sql.extract import extract_sql

DB = "department_management"
GOLD = "SELECT Name, Budget_in_Billions FROM department ORDER BY Name"


@pytest.fixture
def executor(spider_mini: Path) -> Iterator[SqliteExecutor]:
    with SqliteExecutor() as sql:
        yield sql


@pytest.mark.phase(1)
def test_fixture_gold_matches_itself(executor: SqliteExecutor) -> None:
    examples = load_raw()

    for example in examples:
        result = executor.run(example.db_id, example.gold_sql)
        assert result.ok, (example.id, result.error)
        assert results_match(result, result, order_matters=order_matters(example.gold_sql))
    assert len(examples) == 60


@pytest.mark.phase(1)
@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (f"```sql\n{GOLD};\n```", True),
        # Columns in another order, rows in the same order.
        (
            "<think>\n\n</think>\n\n```sql\nSELECT Budget_in_Billions, Name FROM department "
            "ORDER BY Name\n```",
            True,
        ),
        ("Here:\nselect name, budget_in_billions from department order by 1", True),
        # The gold orders its rows, so the same rows in table order are wrong.
        ("```sql\nSELECT Name, Budget_in_Billions FROM department\n```", False),
        ("```sql\nSELECT Name FROM department ORDER BY Name\n```", False),
        ("```sql\nSELECT Name, Budget FROM department ORDER BY Name\n```", False),
        ("```sql\nDELETE FROM department\n```", False),
        ("I don't know.", False),
    ],
)
def test_reply_to_verdict(executor: SqliteExecutor, reply: str, expected: bool) -> None:
    gold = executor.run(DB, GOLD)
    sql = extract_sql(reply)

    matched = sql is not None and results_match(
        gold, executor.run(DB, sql), order_matters=order_matters(GOLD)
    )

    assert matched is expected
