"""scripts/check_choice_extraction.py: ADR-0012's rule on stored scores and cut queries."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from check_choice_extraction import (
    Candidate,
    apply_rule,
    describe,
    load_candidates,
    main,
    read_whole_scores,
)

from forge.core.types import GenParams, Prompt
from forge.data.spider import resolve_data_dir
from forge.data.splits import load_val
from forge.eval.config import EvalConfig
from forge.eval.report import load_run
from forge.eval.runner import run_eval
from forge.infer.base import FakeBackend

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "docs" / "runs"
# FG-307's three candidates on val with p1 (ADR-0012).
CANDIDATE_RUNS = (
    "2026-11-16-qwen3-1-7b-val-c8f4b41",
    "2026-11-16-qwen3-4b-val-c8f4b41",
    "2026-11-16-qwen2-5-coder-1-5b-val-c8f4b41",
)


def _fenced(sql: str) -> str:
    return f"```sql\n{sql}\n```"


def _over_two_lines(sql: str) -> str:
    # Unfenced, the select list on a line of its own: rule 3 keeps only that line.
    return re.sub(r"\s+FROM\s+", "\nFROM ", sql, count=1, flags=re.IGNORECASE)


@pytest.mark.phase(3)
def test_rule_reads_cut_queries_whole_and_breaks_ties_by_size(
    spider_mini: Path, tmp_path: Path
) -> None:
    val = sorted(load_val(), key=lambda example: example.id)
    order = [example.id for example in val]
    gold = {example.id: example.gold_sql for example in val}

    def make(run_id: str, reply: Callable[[int, str], str]) -> Path:
        def fn(prompt: Prompt, params: GenParams, sample_idx: int) -> str:
            return reply(order.index(prompt.example_id), gold[prompt.example_id])

        run = run_eval(
            EvalConfig(model="gold", backend="fake", run_id=run_id),
            backend=FakeBackend(fn, model_ref="gold"),
            allow_dirty=True,
            runs_root=tmp_path / "runs",
            cache_root=tmp_path / "cache",
            workers=1,
        )
        return run.paths.root

    # The coder fences the gold query for the first 13 of 20 questions; the general model
    # gets one more right but writes each over two lines, unfenced; the chat model fences
    # every one, and leaves no headroom.
    folders = {
        "org/coder": make("coder", lambda i, sql: _fenced(sql if i < 13 else "SELECT 1")),
        "org/general": make(
            "general", lambda i, sql: _over_two_lines(sql) if i < 14 else "SELECT 1"
        ),
        "org/chat": make("chat", lambda i, sql: _fenced(sql)),
    }
    params = {"org/coder": 1_500_000_000, "org/general": 4_000_000_000, "org/chat": 2_000_000_000}
    data = resolve_data_dir()
    candidates = [
        Candidate(name, params[name], load_run(folder.name, runs_root=folder.parent))
        for name, folder in folders.items()
    ]
    stored = {c.name: list(c.run.stored.scores) for c in candidates}
    read = {c.name: read_whole_scores(c, data) for c in candidates}
    whole = {name: scores for name, (scores, _) in read.items()}
    cut = {name: count for name, (_, count) in read.items()}
    assert cut == {"org/coder": 0, "org/general": 14, "org/chat": 0}

    # As stored, every cut query misses: the coder leads, ahead by a significant margin,
    # and the chat model, at 100%, is not eligible.
    as_stored = apply_rule(candidates, stored)
    hits = {name: rate.hits for name, rate in as_stored.rates.items()}
    assert hits == {"org/coder": 13, "org/general": 0, "org/chat": 20}
    assert as_stored.ranked == ("org/coder", "org/general")
    assert as_stored.test is not None and as_stored.test.pvalue < 0.05
    assert as_stored.chosen == "org/coder"

    # Read whole, the general model leads by one question, which is no significant
    # difference, so the one with fewer parameters is chosen.
    read_whole = apply_rule(candidates, whole)
    assert read_whole.rates["org/general"].hits == 14
    assert read_whole.ranked == ("org/general", "org/coder")
    assert read_whole.test is not None and read_whole.test.pvalue >= 0.05
    assert read_whole.test.delta.value == pytest.approx(1 / 20)
    assert read_whole.chosen == "org/coder"
    lines = describe("Read whole", candidates, read_whole, cut)
    assert lines[0] == "Read whole:"
    assert lines[1].startswith("- org/coder (1.50B parameters): official EX 65.0% [")
    assert lines[2].endswith(", eligible, 14 queries read whole")
    assert lines[3].endswith("no headroom, 0 queries read whole")
    assert lines[4].startswith("- org/general against org/coder: +5.0 points [")
    assert lines[5] == (
        "- chosen: org/coder, the one with fewer parameters, as the best two don't differ "
        "significantly (p of 0.05 or more)"
    )

    # A significant lead goes to the best, whatever its size; with one eligible candidate
    # it is chosen, and with none the rule has no answer.
    behind = [s.model_copy(update={"ex_official": False}) for s in stored["org/coder"]]
    assert apply_rule(candidates, {**whole, "org/coder": behind}).chosen == "org/general"
    alone = apply_rule(candidates, {**stored, "org/general": whole["org/chat"]})
    assert (alone.ranked, alone.test, alone.chosen) == (("org/coder",), None, "org/coder")
    full = dict.fromkeys(folders, whole["org/chat"])
    assert apply_rule(candidates, full).chosen is None

    # Only a model the models file pins as a candidate is read.
    models = tmp_path / "models.yaml"
    models.write_text(
        yaml.safe_dump({"candidates": {"x": {"repo": "org/x", "revision": "a" * 40, "params": 1}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="run coder's model gold is not a candidate"):
        load_candidates([folders["org/coder"]], models=models, data=data)


@pytest.mark.phase(3)
@pytest.mark.requires_full_data
def test_adr_0012_quotes_the_choice_check(capsys: pytest.CaptureFixture[str]) -> None:
    # The choice under both readings of rule 3 comes from the script, never typed by hand.
    assert main([str(RUNS / run) for run in CANDIDATE_RUNS]) == 0
    printed = capsys.readouterr().out
    adr = (ROOT / "docs" / "adr" / "0012-model-choice.md").read_text(encoding="utf-8")
    assert printed in adr
    chosen = re.search(r"\n- Chosen model: `([^`]+)`\n", adr)
    assert chosen is not None
    assert printed.count(f"- chosen: {chosen.group(1)}, ") == 2
