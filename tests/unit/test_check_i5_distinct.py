"""scripts/check_i5_distinct.py: which of I5's disagreements the evaluator's DISTINCT explains."""

from __future__ import annotations

from pathlib import Path

import pytest
from check_i5_distinct import (
    HEADER,
    Check,
    Disagreement,
    check_run,
    drops_distinct,
    main,
    pooled,
    render,
)

from forge.core.types import GenParams, Prompt
from forge.data.splits import load_val
from forge.eval.config import EvalConfig
from forge.eval.report import load_run
from forge.eval.runner import run_eval
from forge.infer.base import FakeBackend
from forge.sql.sandbox import SandboxPool

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "docs" / "runs"
# FG-307's baselines (ADR-0012), whose disagreements ADR-0008's note of 2026-11-17 settles.
FG307_RUNS = (
    "2026-11-16-qwen3-1-7b-val-c8f4b41",
    "2026-11-16-qwen3-4b-val-c8f4b41",
    "2026-11-16-qwen2-5-coder-1-5b-val-c8f4b41",
    "2026-11-16-qwen2-5-coder-1-5b-p2-c8f4b41",
    "2026-11-16-qwen2-5-coder-1-5b-i6-c8f4b41",
    "2026-11-17-qwen3-8b-awq-val-2f77281",
    "2026-11-17-qwen2-5-coder-1-5b-dev-2f77281",
    "2026-11-17-qwen3-8b-awq-dev-d2d6a3a",
)
# Replies the two checkers read differently, on the fixture's val questions; every other
# question gets its gold query. Gold is the one origin with the most flights, and only
# fast EX matches the one origin with DISTINCT, which the evaluator drops to give eight
# rows of it. Gold counts 16 aircraft, and only official EX matches a count of one value
# with DISTINCT, which runs as a count of 16 once DISTINCT is dropped. The average a
# ten-millionth off is the same to 6 decimals, where fast EX rounds and the evaluator
# doesn't: DISTINCT plays no part, so it disagrees whatever the evaluator does with it.
REPLIES = {
    "spider-train-00024": "SELECT DISTINCT origin FROM Flight WHERE origin = 'Los Angeles'",
    "spider-train-00010": "SELECT count(DISTINCT 'aircraft') FROM Aircraft",
    "spider-train-00019": (
        "SELECT min(distance), avg(distance) + 0.0000001, max(distance) FROM Aircraft"
    ),
}


@pytest.mark.phase(3)
def test_distinct_kept_settles_only_the_disagreements_it_causes(
    spider_mini: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    gold = {example.id: example.gold_sql for example in load_val()}

    def reply(prompt: Prompt, params: GenParams, sample_idx: int) -> str:
        return f"```sql\n{REPLIES.get(prompt.example_id, gold[prompt.example_id])}\n```"

    runs = tmp_path / "runs"
    run_eval(
        EvalConfig(model="gold", backend="fake", run_id="replies"),
        backend=FakeBackend(reply, model_ref="gold"),
        allow_dirty=True,
        runs_root=runs,
        cache_root=tmp_path / "cache",
        workers=1,
    )
    run = load_run("replies", runs_root=runs)
    with SandboxPool(1, data_dir=spider_mini, imports=("forge.eval.testsuite",)) as pool:
        check = check_run(run, pool)

    # Only the three replies disagree, and each is scored again as the run scored it.
    verdicts = {d.example_id: (d.fast, d.again, d.kept) for d in check.disagreements}
    assert verdicts == {
        "spider-train-00010": (False, True, False),
        "spider-train-00019": (True, False, False),
        "spider-train-00024": (True, False, True),
    }
    assert [d.example_id for d in check.disagreements if d.remains] == ["spider-train-00019"]
    # Two of the replies hold a DISTINCT, and none of the gold queries the rest repeat.
    assert check.row() == "| replies | 20 | 2 | 3 | 2 | 1 | 85.00% | 3 | 1 |"
    assert pooled([check, check]).row() == (
        "| 2 runs pooled | 40 | 4 | 6 | 4 | 2 | 85.00% | 6 | 2 |"
    )
    assert render([check]) == "\n".join(
        [
            HEADER,
            check.row(),
            "",
            "Disagreements that remain with DISTINCT kept:",
            "- replies, spider-train-00019 sample 0: only fast EX matches",
            "",
            "Disagreements that don't score as stored again:",
            "- none",
        ]
    )

    # A reply that gets another verdict when scored again is listed apart: something
    # besides DISTINCT changed between the two scorings.
    odd = Check("odd", 1, 0, (Disagreement("odd", "q", 0, fast=True, again=True, kept=True),))
    assert render([odd]).endswith("as stored again:\n- odd, q sample 0: only fast EX matches")

    # The command prints the same, and refuses a folder that holds no run.
    assert main([str(runs / "replies")]) == 0
    assert capsys.readouterr().out == render([check]) + "\n"
    assert main([str(runs / "missing")]) == 1
    assert capsys.readouterr().err.startswith("error: ")


@pytest.mark.phase(3)
def test_replies_with_distinct_are_those_the_evaluator_rewrites() -> None:
    # Only a DISTINCT token of the first statement goes, as remove_distinct reads it.
    assert drops_distinct("SELECT count(DISTINCT name) FROM head")
    assert drops_distinct("select distinct name from head")
    for sql in ("SELECT 'distinct' FROM head", "SELECT 1; SELECT DISTINCT 2", None):
        assert not drops_distinct(sql), sql


@pytest.mark.phase(3)
@pytest.mark.requires_full_data
def test_adr_0008_quotes_the_check_on_fg307s_runs(capsys: pytest.CaptureFixture[str]) -> None:
    # The I5 decision rests on the script's output on the committed runs, never typed by
    # hand, and none of their disagreements outlasts DISTINCT kept.
    assert main([str(RUNS / run) for run in FG307_RUNS]) == 0
    printed = capsys.readouterr().out
    adr = (ROOT / "docs" / "adr" / "0008-fast-comparator-for-rewards.md").read_text(
        encoding="utf-8"
    )
    assert printed in adr.split("\n## Notes\n", 1)[1]
    assert f"\n| {len(FG307_RUNS)} runs pooled | " in printed
    assert "\nDisagreements that remain with DISTINCT kept:\n- none\n" in printed
