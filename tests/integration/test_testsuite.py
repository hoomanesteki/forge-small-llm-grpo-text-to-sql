"""Official execution match through the vendored evaluator, and the guards around it."""

from __future__ import annotations

import asyncio
import gc
import threading
import time
from pathlib import Path

import pytest
from test_suite_eval import exec_eval

from forge.data.spider import DataNotFound, db_path
from forge.eval.testsuite import GoldQueryFailed, OfficialEvalTimeout, official_exec_match

DB = "department_management"
# No head is older than 1000, so this gold result is empty.
EMPTY_GOLD = "SELECT name FROM head WHERE age > 1000"
COUNT_GOLD = "SELECT count(*) FROM head WHERE age  >  56"


def _upstream(pred: str, gold: str) -> int:
    """The vendored evaluator called bare, without the wrapper's guards."""
    return exec_eval.eval_exec_match(
        db=str(db_path(DB)),
        p_str=pred,
        g_str=gold,
        plug_value=False,
        keep_distinct=False,
        progress_bar_for_each_datapoint=False,
    )


@pytest.mark.phase(1)
def test_refused_predictions_never_reach_the_evaluator(
    spider_mini: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    def recorder(**kwargs: object) -> int:
        calls.append(kwargs)
        return 1

    monkeypatch.setattr(exec_eval, "eval_exec_match", recorder)
    refused: list[str | None] = [
        None,
        "",
        "  ;  ",
        "-- SELECT name FROM head",
        "/* */",
        "DELETE FROM head",
        "DROP TABLE head",
        "DROP TABLE IF EXISTS forge_missing",
        "SELECT 1; SELECT 2",
    ]

    assert official_exec_match([(DB, pred, EMPTY_GOLD) for pred in refused]) == [False] * 9
    assert calls == []

    assert official_exec_match([(DB, "SELECT name FROM head", EMPTY_GOLD)]) == [True]
    assert calls == [
        {
            "db": str(db_path(DB).resolve()),
            "p_str": "SELECT name FROM head",
            "g_str": EMPTY_GOLD,
            "plug_value": False,
            "keep_distinct": False,
            "progress_bar_for_each_datapoint": False,
        }
    ]


@pytest.mark.phase(1)
def test_official_match_closes_upstream_holes(spider_mini: Path) -> None:
    # Upstream issue #33: SQL that runs nothing matches any empty gold result.
    for pred in ("-- nothing", "/* */", "DROP TABLE IF EXISTS forge_missing"):
        assert _upstream(pred, EMPTY_GOLD) == 1
        assert official_exec_match([(DB, pred, EMPTY_GOLD)]) == [False]
    # A real query with an empty result still matches.
    assert official_exec_match([(DB, "SELECT name FROM head WHERE age > 2000", EMPTY_GOLD)]) == [
        True
    ]
    # evaluate() would turn the prediction's 'value' into '1' (upstream issue #21).
    assert official_exec_match([(DB, "SELECT 'value'", "SELECT 'value'")]) == [True]
    # Past sqlparse's grouping limits the evaluator can't read the prediction.
    huge = "SELECT " + " + ".join(["1"] * 6000)
    assert official_exec_match([(DB, huge, "SELECT 6000")]) == [False]


@pytest.mark.phase(1)
def test_official_match_runs_inside_an_event_loop(spider_mini: Path) -> None:
    async def in_a_notebook_cell(call: str) -> object:
        if call == "upstream":
            return _upstream(COUNT_GOLD, COUNT_GOLD)
        return official_exec_match([(DB, COUNT_GOLD, COUNT_GOLD)])

    # The bare call fails, and leaves behind a coroutine that was never awaited, which
    # Python warns about once it is collected.
    with pytest.warns(RuntimeWarning, match="'exec_on_db' was never awaited"):
        with pytest.raises(RuntimeError, match="cannot be called from a running event loop"):
            asyncio.run(in_a_notebook_cell("upstream"))
        gc.collect()
    assert asyncio.run(in_a_notebook_cell("wrapped")) == [True]


@pytest.mark.phase(1)
def test_official_match_has_an_overall_timeout(
    spider_mini: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def slow(**kwargs: object) -> int:
        calls.append(str(kwargs["p_str"]))
        time.sleep(0.05)
        return 1

    monkeypatch.setattr(exec_eval, "eval_exec_match", slow)
    items = [(DB, f"SELECT {i}", f"SELECT {i}") for i in range(100)]

    start = time.monotonic()
    with pytest.raises(OfficialEvalTimeout, match=r"100 items took longer than 0\.3 s"):
        official_exec_match(items, timeout_s=0.3)
    assert time.monotonic() - start < 1.0

    # The worker stops after the item it was on.
    time.sleep(0.2)
    settled = len(calls)
    time.sleep(0.2)
    assert len(calls) == settled < len(items)
    assert not any(t.name == "forge-official-eval" for t in threading.enumerate())

    with pytest.raises(ValueError, match="timeout_s must be positive"):
        official_exec_match(items, timeout_s=0)


@pytest.mark.phase(1)
def test_official_match_raises_for_broken_gold_or_data(spider_mini: Path) -> None:
    with pytest.raises(GoldQueryFailed, match="gold query fails on department_management"):
        official_exec_match([(DB, "SELECT 1", "SELECT * FROM forge_missing")])
    with pytest.raises(DataNotFound, match=r"no_such_db\.sqlite not found"):
        official_exec_match([("no_such_db", "SELECT 1", "SELECT 1")])
