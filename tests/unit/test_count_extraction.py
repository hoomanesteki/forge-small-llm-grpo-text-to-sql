"""scripts/count_extraction.py: how extract_sql read a stored run's replies."""

from __future__ import annotations

from pathlib import Path

import pytest
from count_extraction import HEADER, count, main, table

from forge.core.types import GenParams, Prompt
from forge.data.splits import load_val
from forge.eval.config import EvalConfig
from forge.eval.run_files import read_run
from forge.eval.runner import run_eval
from forge.infer.base import FakeBackend

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "docs" / "runs" / "2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7"
# One reply of each kind, in turn: a sql block, a query on one line, a query on three lines
# that rule 3 cuts to its first, a query with a sentence after it, which is no cut, a
# query drafted in a think block and given again after it, and no SQL.
REPLIES = (
    "```sql\nSELECT count(*) FROM flight\n```",
    "SELECT count(*) FROM flight",
    "Here it is:\nSELECT name\nFROM aircraft\nWHERE distance > 100\n\nDone.",
    "SELECT count(*) FROM flight\nThis counts the flights.",
    "<think>\nSELECT 1\n</think>\nSELECT count(*) FROM aircraft",
    "I don't know.",
)


@pytest.mark.phase(3)
def test_replies_are_counted_by_the_rule_that_read_them(spider_mini: Path, tmp_path: Path) -> None:
    order = sorted(example.id for example in load_val())

    def reply(prompt: Prompt, params: GenParams, sample_idx: int) -> str:
        return REPLIES[order.index(prompt.example_id) % len(REPLIES)]

    run = run_eval(
        EvalConfig(model="gold", backend="fake", limit=len(REPLIES), run_id="kinds"),
        backend=FakeBackend(reply, model_ref="gold"),
        allow_dirty=True,
        runs_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
        workers=1,
    )

    counts = count(read_run(run.paths.root))
    assert (counts.replies, counts.fenced, counts.rule_3, counts.no_sql) == (6, 1, 4, 1)
    assert counts.cut == sum(counts.cut_classes.values()) == 1
    assert table([run.paths.root]) == f"{HEADER}\n{counts.row()}"
    assert counts.row().startswith("| kinds | 6 | 1 | 4 | 1 | 1 | ")


@pytest.mark.phase(3)
def test_adr_0007_quotes_the_pilots_counts(capsys: pytest.CaptureFixture[str]) -> None:
    # The evidence for the prompt format comes from the script, never typed by hand.
    assert main([str(PILOT)]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith(HEADER)
    adr = (ROOT / "docs" / "adr" / "0007-prompt-format.md").read_text(encoding="utf-8")
    assert printed in adr


@pytest.mark.phase(3)
def test_baseline_adrs_quote_the_runs_counts(capsys: pytest.CaptureFixture[str]) -> None:
    # FG-307's rows: the chosen model with p1 and p2 in ADR-0007, the candidates in
    # ADR-0012, and in a table of its own there the reference, which ran with p2.
    quoted = (
        (
            "0007-prompt-format",
            (
                "2026-11-16-qwen2-5-coder-1-5b-val-c8f4b41",
                "2026-11-16-qwen2-5-coder-1-5b-p2-c8f4b41",
            ),
        ),
        (
            "0012-model-choice",
            (
                "2026-11-16-qwen3-1-7b-val-c8f4b41",
                "2026-11-16-qwen3-4b-val-c8f4b41",
                "2026-11-16-qwen2-5-coder-1-5b-val-c8f4b41",
            ),
        ),
        ("0012-model-choice", ("2026-11-17-qwen3-8b-awq-val-2f77281",)),
    )
    for adr, runs in quoted:
        assert main([str(ROOT / "docs" / "runs" / run) for run in runs]) == 0
        printed = capsys.readouterr().out
        text = (ROOT / "docs" / "adr" / f"{adr}.md").read_text(encoding="utf-8")
        assert printed in text.split("\n## Measured\n", 1)[1], adr
