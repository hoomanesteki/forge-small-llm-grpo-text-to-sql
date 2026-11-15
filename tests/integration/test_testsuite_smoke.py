"""The vendored official evaluator runs on a fixture Spider database."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from test_suite_eval.evaluation import Evaluator
from test_suite_eval.exec_eval import eval_exec_match

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "spider_mini"
DB = FIXTURE / "database" / "department_management" / "department_management.sqlite"

# Two gold queries for department_management from Spider's train_spider.json.
COUNT_GOLD = "SELECT count(*) FROM head WHERE age  >  56"
ORDERED_GOLD = "SELECT name ,  born_state ,  age FROM head ORDER BY age"

# The pre-parsed `sql` field of those examples (train_spider.json items 0 and 8), which
# is what hardness is computed from, so no SQL parsing or nltk is involved.
COUNT_GOLD_SQL: dict[str, Any] = {
    "from": {"table_units": [["table_unit", 1]], "conds": []},
    "select": [False, [[3, [0, [0, 0, False], None]]]],
    "where": [[False, 3, [0, [0, 10, False], None], 56.0, None]],
    "groupBy": [],
    "having": [],
    "orderBy": [],
    "limit": None,
    "intersect": None,
    "union": None,
    "except": None,
}
TOP_YEAR_GOLD_SQL: dict[str, Any] = {
    "from": {"table_units": [["table_unit", 0]], "conds": []},
    "select": [False, [[0, [0, [0, 3, False], None]]]],
    "where": [],
    "groupBy": [[0, 3, False]],
    "having": [],
    "orderBy": ["desc", [[0, [3, 0, False], None]]],
    "limit": 1,
    "intersect": None,
    "union": None,
    "except": None,
}


def _exec_match(pred: str, gold: str) -> int:
    return eval_exec_match(
        db=str(DB),
        p_str=pred,
        g_str=gold,
        plug_value=False,
        keep_distinct=False,
        progress_bar_for_each_datapoint=False,
    )


@pytest.mark.phase(1)
def test_official_eval_runs_on_fixture(capsys: pytest.CaptureFixture[str]) -> None:
    assert _exec_match(COUNT_GOLD, COUNT_GOLD) == 1
    assert _exec_match(ORDERED_GOLD, ORDERED_GOLD) == 1

    # 4 heads are younger than 56, 5 are older: same shape, different answer.
    assert _exec_match("SELECT count(*) FROM head WHERE age < 56", COUNT_GOLD) == 0
    # The gold has ORDER BY, so row order counts; column order never does.
    reversed_rows = "SELECT name, born_state, age FROM head ORDER BY age DESC"
    swapped_columns = "SELECT age, name, born_state FROM head ORDER BY age"
    assert _exec_match(reversed_rows, ORDERED_GOLD) == 0
    assert _exec_match(swapped_columns, ORDERED_GOLD) == 1

    evaluator = Evaluator()
    assert evaluator.eval_hardness(COUNT_GOLD_SQL) == "easy"
    assert evaluator.eval_hardness(TOP_YEAR_GOLD_SQL) == "hard"

    assert capsys.readouterr() == ("", "")
