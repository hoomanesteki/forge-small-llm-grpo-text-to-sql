"""Check whether the evaluator dropping DISTINCT explains where I5's two checkers disagree.

Usage: FORGE_DATA_DIR=data/spider uv run python scripts/check_i5_distinct.py RUN [RUN ...]

Each RUN is a stored eval run folder, read as its report reads it, so a dev run needs its
row in docs/dev-usage.md and adds none. Its scores hold both verdicts on every reply: the
fast comparator's, which GRPO's reward uses, and official EX, which removes every
DISTINCT from both queries before it runs them (ADR-0008). Each sample where the two
differ is scored by the official evaluator again, once as the run scored it and once
with DISTINCT kept (``official_exec_match(keep_distinct=True)``), nothing else changed.

It prints a row a run, then one for the runs pooled: the samples, the replies whose query
holds a DISTINCT the evaluator drops, the disagreements, by which checker alone matched,
the agreement, how many disagreements score as stored when scored again, and how many
remain with DISTINCT kept. Each sample that remains is listed after the table. ADR-0008
quotes the output.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import sqlparse
from sqlparse.exceptions import SQLParseError

from forge.data.spider import resolve_data_dir
from forge.eval.config import OFFICIAL_ITEM_TIMEOUT_S
from forge.eval.report import ReportRun, load_run
from forge.eval.run_files import RunUnfinished, key
from forge.eval.testsuite import official_exec_match
from forge.sql.sandbox import SandboxError, SandboxPool

HEADER = (
    "| Run | Samples | Replies with DISTINCT | Disagree | Only fast EX | Only official EX "
    "| Agreement | Scored as stored again | Disagree with DISTINCT kept |\n"
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
)


@dataclass(frozen=True)
class Disagreement:
    """A sample whose stored verdicts differ, so official EX is the opposite of ``fast``."""

    run_id: str
    example_id: str
    sample_idx: int
    fast: bool  # the stored fast verdict
    again: bool  # official EX scored again as the run scored it
    kept: bool  # official EX with DISTINCT kept

    @property
    def remains(self) -> bool:
        return self.kept != self.fast

    def describe(self) -> str:
        alone = "only fast EX matches" if self.fast else "only official EX matches"
        return f"- {self.run_id}, {self.example_id} sample {self.sample_idx}: {alone}"


@dataclass(frozen=True)
class Check:
    """One run's disagreements, or several runs' pooled under ``label``."""

    label: str
    samples: int
    distinct: int  # replies whose query holds a DISTINCT the evaluator drops
    disagreements: tuple[Disagreement, ...]

    def row(self) -> str:
        found = self.disagreements
        fast_only = sum(d.fast for d in found)
        counts = " | ".join(f"{n:,}" for n in (self.samples, self.distinct, len(found), fast_only))
        agreement = (self.samples - len(found)) / self.samples
        return (
            f"| {self.label} | {counts} | {len(found) - fast_only:,} | {agreement:.2%} "
            f"| {sum(d.again != d.fast for d in found):,} | {sum(d.remains for d in found):,} |"
        )


def drops_distinct(sql: str | None) -> bool:
    """Whether the evaluator's ``remove_distinct`` drops a token of ``sql``: one that reads
    ``distinct`` in the first statement sqlparse finds."""
    if sql is None:
        return False
    try:
        statements = sqlparse.parse(sql)
    except SQLParseError:
        # Past sqlparse's limits the evaluator can't read the query, so it drops nothing.
        return False
    return bool(statements) and any(t.value.lower() == "distinct" for t in statements[0].flatten())


def check_run(run: ReportRun, pool: SandboxPool) -> Check:
    """``run``'s disagreeing samples, each scored again with and without DISTINCT."""
    if not run.stored.scores:
        raise ValueError(f"run {run.run_id} has no scores to check")
    replies = {key(score): run.stored.generations[key(score)].sql for score in run.stored.scores}
    disagreeing = [s for s in run.stored.scores if s.ex_fast != s.ex_official]
    items: list[tuple[str, str | None, str]] = []
    for score in disagreeing:
        example = run.examples[score.example_id]
        items.append((example.db_id, replies[key(score)], example.gold_sql))
    # The runner's item timeout, so a reply scored again gets the bound it was scored under.
    again = official_exec_match(items, item_timeout_s=OFFICIAL_ITEM_TIMEOUT_S, pool=pool)
    kept = official_exec_match(
        items, item_timeout_s=OFFICIAL_ITEM_TIMEOUT_S, pool=pool, keep_distinct=True
    )
    found = tuple(
        Disagreement(run.run_id, s.example_id, s.sample_idx, s.ex_fast, a, k)
        for s, a, k in zip(disagreeing, again, kept, strict=True)
    )
    distinct = sum(map(drops_distinct, replies.values()))
    return Check(run.run_id, len(run.stored.scores), distinct, found)


def pooled(checks: Sequence[Check]) -> Check:
    return Check(
        f"{len(checks)} runs pooled",
        sum(check.samples for check in checks),
        sum(check.distinct for check in checks),
        tuple(d for check in checks for d in check.disagreements),
    )


def render(checks: Sequence[Check]) -> str:
    lines = [HEADER, *(check.row() for check in checks)]
    if len(checks) > 1:
        lines.append(pooled(checks).row())
    found = [d for check in checks for d in check.disagreements]
    lines += ["", "Disagreements that remain with DISTINCT kept:"]
    lines += [d.describe() for d in found if d.remains] or ["- none"]
    # Scored again the way the run scored it, a reply should get its stored verdict; one
    # that doesn't says something besides DISTINCT changed between the two scorings.
    lines += ["", "Disagreements that don't score as stored again:"]
    lines += [d.describe() for d in found if d.again == d.fast] or ["- none"]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="stored run folders")
    folders = parser.parse_args(argv).runs
    try:
        data = resolve_data_dir().resolve()
        with SandboxPool(1, data_dir=data, imports=("forge.eval.testsuite",)) as pool:
            checks = [
                check_run(load_run(folder.name, runs_root=folder.parent, data_dir=data), pool)
                for folder in folders
            ]
    except (ValueError, OSError, RunUnfinished, SandboxError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(render(checks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
