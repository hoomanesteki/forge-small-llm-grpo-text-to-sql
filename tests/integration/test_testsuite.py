"""Official execution match, gold scored against itself, and the agreement check."""

from __future__ import annotations

import asyncio
import gc
import logging
import multiprocessing
import re
import time
from pathlib import Path
from typing import Any

import pytest
from test_suite_eval import exec_eval
from typer.testing import CliRunner

from forge import cli
from forge.data.spider import DataNotFound, db_path, load_raw
from forge.data.splits import load_val
from forge.eval import testsuite
from forge.eval.agreement import (
    GOLD_TIMEOUT_MS,
    agreement_mutations,
    agreement_rate,
    gold_mutations,
    score,
)
from forge.eval.testsuite import GoldQueryFailed, OfficialEvalTimeout, official_exec_match
from forge.sql import sandbox
from forge.sql.sandbox import SandboxError, SandboxPool
from tests import sandbox_jobs as jobs
from tests.fakes import InlineSandbox
from tests.sandbox_jobs import review_query

DB = "department_management"
# No head is older than 1000, so this gold result is empty.
EMPTY_GOLD = "SELECT name FROM head WHERE age > 1000"
COUNT_GOLD = "SELECT count(*) FROM head WHERE age  >  56"


class _PoolThatLosesAWorker(SandboxPool):
    """A real pool whose worker dies (exit code 3) when an item has one chosen prediction."""

    def __init__(self, data_dir: Path, die_on: str) -> None:
        super().__init__(1, data_dir=data_dir, imports=("tests.sandbox_jobs",))
        self._die_on = die_on

    def call(self, fn: Any, *args: Any, timeout_s: float) -> Any:
        assert fn is testsuite._match
        return super().call(jobs.match_or_die, *args, self._die_on, timeout_s=timeout_s)


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
        # Writes SQLite rejects for a syntax or name error before it asks the authorizer.
        "ATTACH DATABASE CASE WHEN 1 > = 1 THEN 'planted.sqlite' END AS e",
        "ATTACH DATABASE YEAR(CURDATE()) || '.sqlite' AS e",
        "DELETE FROM forge_missing",
        # Writes only once the evaluator has dropped DISTINCT, or cut the input at the
        # first statement sqlparse finds.
        "DISTINCT ATTACH DATABASE 'hidden.sqlite' AS e",
        "DISTINCT DELETE FROM head",
        "DISTINCT VACUUM INTO 'copy.sqlite'",
    ]

    # In this process, where the recorder replaces the evaluator.
    inline = InlineSandbox(spider_mini)
    verdicts = official_exec_match([(DB, pred, EMPTY_GOLD) for pred in refused], pool=inline)
    assert verdicts == [False] * len(refused)
    assert calls == []

    assert official_exec_match([(DB, "SELECT name FROM head", EMPTY_GOLD)], pool=inline) == [True]
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
def test_official_match_sees_the_statement_sqlite_runs(spider_mini: Path) -> None:
    # SQLite skips leading semicolons and comments; remove_distinct keeps only the first
    # statement sqlparse finds, which for these is the bare ';' and runs to no rows.
    names = "; SELECT name FROM head"
    assert _upstream(names, EMPTY_GOLD) == 1
    counted = "/* count */ ; SELECT count(*) FROM head WHERE age  >  56"
    assert _upstream(counted, COUNT_GOLD) == 0

    assert official_exec_match(
        [
            (DB, names, EMPTY_GOLD),
            (DB, counted, COUNT_GOLD),
            (DB, "; -- empty\n SELECT name FROM head WHERE age > 2000", EMPTY_GOLD),
        ]
    ) == [False, True, True]


@pytest.mark.phase(3)
def test_official_match_can_keep_distinct(
    spider_mini: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Official EX drops DISTINCT from both queries, so five distinct states match ten rows
    # of them. Kept, as scripts/check_i5_distinct.py asks, they don't, as fast EX says.
    items = [
        (DB, "SELECT DISTINCT born_state FROM head", "SELECT born_state FROM head"),
        (DB, "SELECT count(DISTINCT born_state) FROM head", "SELECT count(*) FROM head"),
    ]
    assert official_exec_match(items) == [True, True]
    assert official_exec_match(items, keep_distinct=True) == [False, False]

    # Kept, what the evaluator would run is still vetted first, and the option reaches it.
    calls: list[dict[str, object]] = []

    def recorder(**kwargs: object) -> int:
        calls.append(kwargs)
        return 1

    monkeypatch.setattr(exec_eval, "eval_exec_match", recorder)
    planted = "ATTACH DATABASE CASE WHEN 1 > = 1 THEN 'planted.sqlite' END AS e"
    names = "SELECT name FROM head"
    verdicts = official_exec_match(
        [(DB, planted, EMPTY_GOLD), (DB, names, EMPTY_GOLD)],
        pool=InlineSandbox(spider_mini),
        keep_distinct=True,
    )
    assert verdicts == [False, True]
    assert [(call["p_str"], call["keep_distinct"]) for call in calls] == [(names, True)]


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
def test_official_match_has_an_overall_timeout(spider_mini: Path) -> None:
    # About 0.1 s an item: the vetting look and the evaluator each count to 300,000.
    slow = (
        "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r WHERE x < 300000) "
        "SELECT count(*) FROM r"
    )
    items = [(DB, slow, COUNT_GOLD)] * 100

    start = time.monotonic()
    with pytest.raises(OfficialEvalTimeout, match=r"100 items took longer than 0\.3 s"):
        official_exec_match(items, timeout_s=0.3)
    # Worker start-up aside, which can take a second on a busy machine.
    assert time.monotonic() - start < 0.3 + 2.0
    # No worker is left running an item.
    assert multiprocessing.active_children() == []

    with pytest.raises(ValueError, match="timeout_s must be positive"):
        official_exec_match(items, timeout_s=0)
    with pytest.raises(ValueError, match="item_timeout_s must be positive"):
        official_exec_match(items, item_timeout_s=0)


@pytest.mark.phase(1)
def test_official_match_kills_a_pathological_prediction(
    spider_mini: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # The evaluator's own deadline gives a query 60 s; only the kill stops this sooner.
    runaway = (
        "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) SELECT count(*) FROM r"
    )
    cartesian = "SELECT count(*) FROM head AS a, head AS b, head AS c, head AS d, head AS e, " + (
        "head AS f, head AS g, head AS h, head AS i"
    )
    items = [(DB, runaway, COUNT_GOLD), (DB, cartesian, COUNT_GOLD), (DB, COUNT_GOLD, COUNT_GOLD)]
    # The FG-191 review's queries now fail on the length limit at once, here too.
    items += [(DB, review_query(100), COUNT_GOLD), (DB, review_query(200), COUNT_GOLD)]
    caplog.set_level(logging.DEBUG, logger="forge")

    start = time.monotonic()
    verdicts = official_exec_match(items, item_timeout_s=0.5)
    elapsed = time.monotonic() - start

    assert verdicts == [False, False, True, False, False]
    # Two kills of 0.5 s and three worker start-ups, where the evaluator's own deadline
    # would have taken two minutes.
    assert elapsed < 10
    assert "official EX item 0 on department_management killed after 0.5 s" in caplog.text
    assert "2 restarts so far" in caplog.text


@pytest.mark.phase(1)
def test_official_match_raises_for_broken_gold_or_data(spider_mini: Path) -> None:
    with pytest.raises(GoldQueryFailed, match="gold query fails on department_management"):
        official_exec_match([(DB, "SELECT 1", "SELECT * FROM forge_missing")])
    with pytest.raises(DataNotFound, match=r"no_such_db\.sqlite not found"):
        official_exec_match([("no_such_db", "SELECT 1", "SELECT 1")])


@pytest.mark.phase(1)
def test_official_match_survives_a_worker_death(
    spider_mini: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A worker that dies under one item used to raise SandboxError out of the whole call,
    # ending the scoring of every other item. Now the item is counted as no match and the
    # rest are scored.
    good = "SELECT name FROM head WHERE age > 2000"  # empty, so it matches EMPTY_GOLD
    crash = "SELECT name FROM head"
    items = [(DB, good, EMPTY_GOLD), (DB, crash, EMPTY_GOLD), (DB, good, EMPTY_GOLD)]
    caplog.set_level(logging.WARNING, logger="forge.eval.testsuite")

    with _PoolThatLosesAWorker(spider_mini, die_on=crash) as pool:
        verdicts = official_exec_match(items, pool=pool)
        assert pool.restarts == 1

    assert verdicts == [True, False, True]
    assert (
        "official EX item 1 on department_management: sandbox worker died with exit code 3"
        in caplog.text
    )


@pytest.mark.phase(1)
def test_official_match_raises_when_no_worker_starts(
    spider_mini: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Unlike a death under one item, a worker that can't start would score every item as a
    # miss, so it is raised for the CLIs to report instead of counted.
    items = [(DB, COUNT_GOLD, COUNT_GOLD)] * 3
    with (
        SandboxPool(1, data_dir=spider_mini, imports=("forge_no_such_module",)) as broken,
        pytest.raises(SandboxError, match="a sandbox worker exited while starting"),
    ):
        official_exec_match(items, pool=broken)

    # The pool official_exec_match makes for itself, when its worker is never ready.
    monkeypatch.setattr(sandbox, "START_TIMEOUT_S", 0.01)
    with pytest.raises(SandboxError, match=r"a sandbox worker didn't start in 0\.01 s"):
        official_exec_match(items)
    assert multiprocessing.active_children() == []


def _cells(text: str) -> list[list[str]]:
    return [line.split() for line in text.splitlines()]


@pytest.mark.phase(1)
def test_gold_scores_perfect(spider_mini: Path) -> None:
    examples = load_raw()
    scores = score(gold_mutations(examples), pred_timeout_ms=GOLD_TIMEOUT_MS)

    assert len(scores) == 60
    misses = [s.mutation.example_id for s in scores if not (s.fast and s.official)]
    assert misses == []

    for split, count, databases in (("val", 20, "1 database"), ("train", 40, "4 databases")):
        result = CliRunner().invoke(cli.app, ["eval", "gold", "--split", split])
        assert result.exit_code == 0, result.output
        lines = result.stdout.splitlines()
        assert lines[1] == f"split {split}: {count} gold queries over {databases}"
        assert _cells(result.stdout)[2:] == [
            ["fast", "EX", f"{count}/{count}", "100.00%"],
            ["official", "EX", f"{count}/{count}", "100.00%"],
        ]

    # The fixture has no dev questions.
    result = CliRunner().invoke(cli.app, ["eval", "gold", "--split", "dev"])
    assert result.exit_code == 1
    assert "the dev split has no questions" in result.stderr


@pytest.mark.phase(1)
def test_gold_on_dev_is_not_a_dev_use(
    with_dev: Path, dev_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_DATA_DIR", str(with_dev))

    result = CliRunner().invoke(cli.app, ["eval", "gold", "--split", "dev"])

    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert lines[:3] == [
        f"data: {with_dev}",
        cli.NOT_A_DEV_USE,
        "split dev: 10 gold queries over 1 database",
    ]
    assert _cells(result.stdout)[3:] == [
        ["fast", "EX", "10/10", "100.00%"],
        ["official", "EX", "10/10", "100.00%"],
    ]
    assert not dev_log.exists()


@pytest.mark.phase(1)
@pytest.mark.requires_full_data
def test_gold_scores_perfect_on_full_val() -> None:
    scores = score(gold_mutations(load_val()), pred_timeout_ms=GOLD_TIMEOUT_MS)

    assert len(scores) == 872
    assert [s.mutation.example_id for s in scores if not (s.fast and s.official)] == []


@pytest.mark.phase(1)
@pytest.mark.requires_full_data
def test_eval_gold_on_full_val_stays_perfect_under_the_limits() -> None:
    # The phase 1 demo, under the SQLite limits measured from Spider (ADR-0014).
    result = CliRunner().invoke(cli.app, ["eval", "gold", "--split", "val"])

    assert result.exit_code == 0, result.output
    assert _cells(result.stdout)[2:] == [
        ["fast", "EX", "872/872", "100.00%"],
        ["official", "EX", "872/872", "100.00%"],
    ]


@pytest.mark.phase(1)
def test_gold_check_fails_when_a_checker_misses(
    spider_mini: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def never(items: list[tuple[str, str | None, str]], **_: object) -> list[bool]:
        return [False] * len(items)

    monkeypatch.setattr("forge.eval.agreement.official_exec_match", never)
    result = CliRunner().invoke(cli.app, ["eval", "gold", "--split", "val"])

    assert result.exit_code == 1
    assert _cells(result.stdout)[3] == ["official", "EX", "0/20", "0.00%"]
    assert "20 gold queries don't match themselves" in result.stderr
    assert "fast True, official False" in result.stderr


@pytest.mark.phase(1)
def test_eval_gold_reports_a_sandbox_failure(
    spider_mini: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A pool that can't start its workers raises SandboxError; the command prints a clean
    # error, not a traceback.
    def boom(items: list[tuple[str, str | None, str]], **_: object) -> list[bool]:
        raise SandboxError("a sandbox worker didn't start in 60 s")

    monkeypatch.setattr("forge.eval.agreement.official_exec_match", boom)
    result = CliRunner().invoke(cli.app, ["eval", "gold", "--split", "val"])

    assert result.exit_code == 1
    assert "error: a sandbox worker didn't start in 60 s" in result.stderr


@pytest.mark.phase(1)
def test_eval_agreement_reports_a_sandbox_failure(
    spider_mini: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(items: list[tuple[str, str | None, str]], **_: object) -> list[bool]:
        raise SandboxError("a sandbox worker didn't start in 60 s")

    monkeypatch.setattr("forge.eval.agreement.official_exec_match", boom)
    out = tmp_path / "agreement.md"
    result = CliRunner().invoke(
        cli.app, ["eval", "agreement", "--mutations", "20", "--out", str(out)]
    )

    assert result.exit_code == 1
    assert "error: a sandbox worker didn't start in 60 s" in result.stderr
    assert not out.exists()


@pytest.mark.phase(1)
def test_agreement_on_mutations(spider_mini: Path, tmp_path: Path) -> None:
    out = tmp_path / "audit" / "agreement.md"
    args = ["eval", "agreement", "--mutations", "150", "--split", "val", "--seed", "3"]

    result = CliRunner().invoke(cli.app, [*args, "--out", str(out)])

    assert result.exit_code == 0, result.output
    scores = score(agreement_mutations(load_val(), 150, seed=3))
    rate = agreement_rate(scores)
    assert rate >= 0.98
    # Mutations give both verdicts, so agreement isn't just everything failing.
    assert 0.2 < sum(s.official for s in scores) / len(scores) < 0.8
    assert f"agreement {rate:.2%} ({sum(s.agree for s in scores)} of 150)" in result.stdout
    assert f"report: {out}" in result.stdout

    report = out.read_text()
    assert report.startswith(
        f"# Fast and official execution match agree on {rate:.2%} of 150 items\n"
    )
    assert "Generated by `forge eval agreement --mutations 150 --split val --seed 3` on " in report
    for heading in ("## Verdicts", "## By mutation", "## Disagreements"):
        assert f"\n{heading}\n" in report
    distinct = {(s.mutation.db_id, s.mutation.gold_sql, s.mutation.pred_sql) for s in scores}
    assert f"\n| Distinct items | {len(distinct)} |\n" in report
    assert "CC BY-SA 4.0" in report

    # The same command writes the same report.
    again = tmp_path / "again.md"
    assert CliRunner().invoke(cli.app, [*args, "--out", str(again)]).exit_code == 0
    assert again.read_text() == report


@pytest.mark.phase(1)
def test_agreement_report_goes_to_docs_audit_by_default(
    spider_mini: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A repo root without git: the report still gets written, and says it has no commit.
    monkeypatch.setenv("FORGE_ROOT", str(tmp_path))

    result = CliRunner().invoke(cli.app, ["eval", "agreement", "--mutations", "20"])

    assert result.exit_code == 0, result.output
    (report,) = (tmp_path / "docs" / "audit").iterdir()
    assert re.fullmatch(r"agreement-\d{4}-\d{2}-\d{2}\.md", report.name)
    assert "| Git commit | `unknown (not run from a git checkout)` |" in report.read_text()


@pytest.mark.phase(1)
def test_agreement_under_target_fails(
    spider_mini: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def contrary(items: list[tuple[str, str | None, str]], **kwargs: object) -> list[bool]:
        return [not verdict for verdict in official_exec_match(items)]

    monkeypatch.setattr("forge.eval.agreement.official_exec_match", contrary)
    out = tmp_path / "agreement.md"
    result = CliRunner().invoke(
        cli.app, ["eval", "agreement", "--mutations", "40", "--out", str(out)]
    )

    assert result.exit_code == 1
    assert "agreement 0.00% (0 of 40)" in result.stdout
    assert "under the 98% that I5 needs" in result.stderr
    # The report is still written, so the disagreements can be read, each check once.
    mutations = agreement_mutations(load_val(), 40, seed=0)
    distinct = len({(m.db_id, m.gold_sql, m.pred_sql) for m in mutations})
    assert 20 < distinct < 40
    report = out.read_text()
    assert f"Disagreeing items: 40 ({distinct} distinct), the first 20 shown below" in report
    assert report.count("\n### ") == 20
