"""The GRPO dataset (FG-505): the filter that picks GRPO's rows from a greedy eval of
train questions."""

from __future__ import annotations

import dataclasses
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from forge.core.types import ExecResult, Hardness, Split
from forge.data.spider import ROWLESS_DBS
from forge.data.splits import load_train
from forge.eval.config import EvalConfig
from forge.eval.gold_cache import split_gold
from forge.eval.run_files import StoredRun, read_run
from forge.eval.runner import EvalRun, run_eval
from forge.infer.base import FakeBackend
from forge.train.curriculum import (
    KEEP_SOLVED,
    LEVELS,
    MAX_GOLD_MS,
    check_filter_run,
    filter_hash,
    select_rows,
)

MODEL = "org/tiny-sft@0123456789abcdef0123456789abcdef01234567"
EVAL = "2026-11-20-tiny-sft-eval-abc1234"


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dev_log: Path) -> Path:
    """Runs and the gold cache under tmp_path, and a scratch dev log."""
    monkeypatch.setenv("FORGE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("FORGE_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "runs"


def _right() -> set[str]:
    """The train questions the stand-in model answers with their gold query: two in
    three, in the loader's order; it has no answer for the rest."""
    return {example.id for i, example in enumerate(load_train()) if i % 3 != 2}


def _evaluate(run_id: str, right: set[str], *, split: Split = "train", **options: Any) -> EvalRun:
    """A finished eval run of the stand-in model, as ``forge eval run`` stores one."""
    gold = {example.id: example.gold_sql for example in load_train()}

    def reply(prompt: Any, params: Any, sample_idx: int) -> str:
        if prompt.example_id in right:
            return f"```sql\n{gold[prompt.example_id]}\n```"
        return "I can't tell from this schema."

    config = EvalConfig(model=MODEL, split=split, run_id=run_id, **options)
    return run_eval(
        config, backend=FakeBackend(reply, model_ref=MODEL), allow_dirty=True, workers=1
    )


def _timed(gold: dict[str, ExecResult], ms: dict[str, float]) -> dict[str, ExecResult]:
    return {**gold, **{i: gold[i].model_copy(update={"ms": t}) for i, t in ms.items()}}


@pytest.mark.phase(5)
def test_filter_proportions(spider_mini: Path, runs: Path, tmp_path: Path) -> None:
    # The fixture's 40 train questions, the stand-in right on two in three of them.
    train = load_train()
    right = _right()
    _evaluate(EVAL, right)
    stored = read_run(runs / EVAL)
    assert len(stored.scores) == len(train) == 40
    # One right and one wrong question whose gold took 1.5 s, and one that took 1 s
    # exactly, which isn't slower than the limit.
    ordered = [example.id for example in train]
    slow = {ordered[0]: 1500.0, ordered[2]: 1500.0}
    gold = _timed(split_gold("train", cache_root=tmp_path / "cache").results, slow)
    gold = _timed(gold, {ordered[1]: MAX_GOLD_MS})
    assert ordered[0] in right and ordered[2] not in right

    found = select_rows(stored, train, gold, seed=0)

    rowless = {e.id for e in train if e.db_id in ROWLESS_DBS}
    usable = [e for e in train if e.id not in rowless and e.id not in slow]
    wrong = {e.id for e in usable if e.id not in right}
    solved = [e.id for e in usable if e.id in right]
    kept = {row.example_id for row in found.rows}
    kept_solved = kept & set(solved)
    # Every wrong question stays, 30% of the 19 right ones, 5.7 rounded to 6, and none on
    # a rowless database or with slow gold.
    assert (len(rowless), len(wrong), len(solved)) == (10, 9, 19)
    assert wrong <= kept and kept <= wrong | set(solved)
    assert len(kept_solved) == 6 == round(KEEP_SOLVED * len(solved))
    assert not kept & (rowless | set(slow))
    assert ordered[1] in solved  # 1 s exactly is not slower than the limit
    assert [row.example_id for row in found.rows] == [i for i in ordered if i in kept]
    for row in found.rows:
        assert row.solved == (row.example_id in right)
        assert row.gold_ms == gold[row.example_id].ms

    # The summary's tallies, by hardness and for all, are those same counts.
    summary = found.summary
    total = summary.total.model_dump()
    assert total == {
        "scored": 40,
        "rowless": 10,
        "slow_gold": 2,
        "wrong": 9,
        "solved": 19,
        "solved_kept": 6,
        "kept": 15,
    }
    by_level: dict[Hardness, Counter[str]] = {level: Counter() for level in LEVELS}
    for example in train:
        tally = by_level[example.hardness]
        tally["scored"] += 1
        tally["rowless"] += example.id in rowless
        tally["slow_gold"] += example.id in slow
        tally["wrong"] += example.id in wrong
        tally["solved"] += example.id in solved
        tally["solved_kept"] += example.id in kept_solved
        tally["kept"] += example.id in kept
    for level in LEVELS:
        counted = summary.by_hardness[level].model_dump()
        assert counted == {name: by_level[level][name] for name in counted}, level
    assert sorted(summary.slow_gold_ids) == sorted(slow)
    assert (summary.eval_run, summary.model_ref, summary.adapter_ref) == (EVAL, MODEL, None)
    assert (summary.keep_solved, summary.max_gold_ms, summary.seed) == (0.3, 1000.0, 0)
    assert summary.data_manifest_hash == stored.manifest.data_manifest_hash
    assert summary.hash == filter_hash(summary, found.rows)

    # The same seed keeps the same questions; another keeps as many, but others.
    assert select_rows(stored, train, gold, seed=0) == found
    other = select_rows(stored, train, gold, seed=1)
    other_solved = {row.example_id for row in other.rows if row.solved}
    assert len(other_solved) == 6 and other_solved != kept_solved
    assert {row.example_id for row in other.rows if not row.solved} == wrong
    assert other.summary.hash != summary.hash


def _stored_with(stored: StoredRun, **config: Any) -> StoredRun:
    changed = {**stored.manifest.config, **config}
    return dataclasses.replace(
        stored, manifest=stored.manifest.model_copy(update={"config": changed})
    )


@pytest.mark.phase(5)
def test_filter_reads_greedy_train_evals_only(spider_mini: Path, runs: Path) -> None:
    _evaluate(EVAL, _right(), limit=6)
    stored = read_run(runs / EVAL)
    data_hash = stored.manifest.data_manifest_hash
    check_filter_run(stored, data_hash)
    generation = stored.manifest.config["generation"]
    refused = [
        (_stored_with(stored, split="val"), "evaluated val, and the GRPO filter reads an eval"),
        (
            _stored_with(stored, generation={**generation, "n": 4, "temperature": 0.8}),
            "drew 4 replies a question at temperature 0.8",
        ),
        (_stored_with(stored, retry=1), "scores are of second replies"),
    ]
    for run, message in refused:
        with pytest.raises(ValueError, match=message):
            check_filter_run(run, data_hash)
    with pytest.raises(ValueError, match="its example ids may name other questions"):
        check_filter_run(stored, "f" * 64)

    # A question train doesn't hold is refused, before anything is kept.
    gold = split_gold("train").results
    with pytest.raises(ValueError, match="scored 6 questions train doesn't hold here"):
        select_rows(stored, [], gold)
    with pytest.raises(ValueError, match="has no result in the train gold cache"):
        select_rows(stored, load_train(), {})
