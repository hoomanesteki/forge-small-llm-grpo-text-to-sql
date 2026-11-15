"""Fast against official execution match: gold self-checks and the agreement check.

Rewards use the fast comparator (``forge.sql.compare``) and reported numbers the
official evaluator (``forge.eval.testsuite``), per ADR-0008. ``score`` runs both on the
same predictions. ``forge eval gold`` gives every gold query to both as its own
prediction, which must score 100%; ``forge eval agreement`` gives them mutated gold
queries (``forge.eval.mutations``) and writes how often they agree, with the first
disagreements, to ``docs/audit/agreement-<date>.md``. Invariant I5 asks for 98%.

Items repeat: an unchanged gold query comes up again on each pass over the examples,
and questions that share a gold query can get the same mutation. An item with the
database, gold and prediction of an earlier one is the same check run again, so the
report also counts each distinct item once.

The fast side runs as a reward would: gold and prediction through ``SqliteExecutor`` with
its row cap, the prediction under the default 2-second timeout and the gold under a
longer one, since a slow gold query is not the prediction's fault.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from forge.core.types import Example, ExecResult
from forge.eval.mutations import Mutation, MutationKind, make_mutations
from forge.eval.testsuite import official_exec_match
from forge.sql.compare import order_matters, results_match
from forge.sql.executor import DEFAULT_TIMEOUT_MS, SqliteExecutor

AGREEMENT_TARGET = 0.98
GOLD_TIMEOUT_MS = 10_000
SHOWN_DISAGREEMENTS = 20
SHOWN_ROWS = 5
# The data spike's license section, relative to docs/audit where reports go.
SPIDER_ATTRIBUTION = "../spikes/data.md#2-license-and-attribution"


@dataclass(frozen=True)
class ItemScore:
    """Both verdicts on one prediction, with the executor's results behind the fast one."""

    mutation: Mutation
    gold: ExecResult
    pred: ExecResult
    order_matters: bool
    fast: bool
    official: bool

    @property
    def agree(self) -> bool:
        return self.fast == self.official


@dataclass(frozen=True)
class AgreementRun:
    """What a report describes: the command, its inputs, and where the code and data were."""

    command: str
    split: str
    examples: int
    databases: int
    seed: int
    day: date
    git_sha: str
    data_hash: str


def gold_mutations(examples: Sequence[Example]) -> list[Mutation]:
    """Each example's gold query as its own prediction."""
    return [Mutation(e.id, e.db_id, "unchanged", e.gold_sql, e.gold_sql) for e in examples]


def agreement_mutations(
    examples: Sequence[Example], n: int, *, seed: int, data_dir: Path | None = None
) -> list[Mutation]:
    """``n`` mutations of these examples' gold queries, with columns from their databases."""
    with SqliteExecutor(data_dir) as executor:
        columns = {db: table_columns(executor, db) for db in sorted({e.db_id for e in examples})}
    return make_mutations(examples, n, seed=seed, columns=columns)


def table_columns(executor: SqliteExecutor, db_id: str) -> dict[str, tuple[str, ...]]:
    """Each table of the database, lowercased, with its column names in table order."""
    listing = _query(
        executor,
        db_id,
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name",
    )
    columns: dict[str, tuple[str, ...]] = {}
    for (table,) in listing.rows or ():
        quoted = '"' + str(table).replace('"', '""') + '"'
        columns[str(table).lower()] = tuple(
            _query(executor, db_id, f"SELECT * FROM {quoted} LIMIT 0").columns or ()
        )
    return columns


def score(
    mutations: Sequence[Mutation],
    *,
    data_dir: Path | None = None,
    pred_timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> list[ItemScore]:
    """The fast and the official verdict on each prediction, in order."""
    gold_results: dict[tuple[str, str], ExecResult] = {}
    fast: list[tuple[ExecResult, ExecResult, bool, bool]] = []
    with SqliteExecutor(data_dir) as executor:
        for m in mutations:
            key = (m.db_id, m.gold_sql)
            if key not in gold_results:
                gold_results[key] = executor.run(m.db_id, m.gold_sql, timeout_ms=GOLD_TIMEOUT_MS)
            gold = gold_results[key]
            pred = executor.run(m.db_id, m.pred_sql, timeout_ms=pred_timeout_ms)
            ordered = order_matters(m.gold_sql)
            fast.append((gold, pred, ordered, results_match(gold, pred, order_matters=ordered)))
    official = official_exec_match(
        [(m.db_id, m.pred_sql, m.gold_sql) for m in mutations], data_dir=data_dir
    )
    return [
        ItemScore(m, gold, pred, ordered, matched, verdict)
        for m, (gold, pred, ordered, matched), verdict in zip(
            mutations, fast, official, strict=True
        )
    ]


def agreement_rate(scores: Sequence[ItemScore]) -> float:
    if not scores:
        raise ValueError("agreement needs at least one scored item")
    return sum(s.agree for s in scores) / len(scores)


def group_repeats(scores: Sequence[ItemScore]) -> list[list[ItemScore]]:
    """The items grouped by database, gold and prediction, in order of first appearance.

    The first item of a group stands for it wherever distinct items are counted.
    """
    groups: dict[tuple[str, str, str], list[ItemScore]] = {}
    for s in scores:
        m = s.mutation
        groups.setdefault((m.db_id, m.gold_sql, m.pred_sql), []).append(s)
    return list(groups.values())


def render_report(scores: Sequence[ItemScore], run: AgreementRun) -> str:
    """The agreement report as markdown."""
    rate = agreement_rate(scores)
    distinct = [group[0] for group in group_repeats(scores)]
    disagreements = group_repeats([s for s in scores if not s.agree])
    lines = [
        f"# Fast and official execution match agree on {rate:.2%} of {len(scores):,} items",
        "",
        f"Generated by `{run.command}` on {run.day.isoformat()}.",
        "Don't edit it by hand; run the command again.",
        "",
        "This report quotes Spider 1.0 (Yu et al., EMNLP 2018) gold queries and database",
        "rows, so like Spider it is licensed",
        "[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); attribution in",
        f"[docs/spikes/data.md]({SPIDER_ATTRIBUTION}).",
        "",
        "| | |",
        "|---|---|",
        f"| Split | {run.split}: {run.examples:,} questions over {run.databases} "
        f"database{'' if run.databases == 1 else 's'} |",
        f"| Items | {len(scores):,} predictions made from gold queries, seed {run.seed} |",
        f"| Distinct items | {len(distinct):,} |",
        f"| Agreement | {rate:.2%} ({sum(s.agree for s in scores):,} of {len(scores):,}); "
        f"I5 needs {AGREEMENT_TARGET:.0%} |",
        f"| Agreement on distinct items | {agreement_rate(distinct):.2%} "
        f"({sum(s.agree for s in distinct):,} of {len(distinct):,}) |",
        f"| Git commit | `{run.git_sha}` |",
        f"| Data manifest hash | `{run.data_hash}` |",
        "",
        "The fast verdict is `forge.sql.compare.results_match` on `forge.sql.executor`",
        "results; the official one is `forge.eval.testsuite.official_exec_match`. An item",
        "with the database, gold and prediction of an earlier one is the same check again;",
        "counts in parentheses take each distinct item once.",
        "",
        "## Verdicts",
        "",
        "| | official match | official no match |",
        "|---|---:|---:|",
        f"| fast match | {_counts(scores, distinct, lambda s: s.fast and s.official)} | "
        f"{_counts(scores, distinct, lambda s: s.fast and not s.official)} |",
        f"| fast no match | {_counts(scores, distinct, lambda s: not s.fast and s.official)} | "
        f"{_counts(scores, distinct, lambda s: not s.fast and not s.official)} |",
        "",
        "## By mutation",
        "",
        "| mutation | items | fast match | official match | disagreements |",
        "|---|---:|---:|---:|---:|",
    ]
    distinct_by_kind = _by_kind(distinct)
    for kind, group in sorted(_by_kind(scores).items()):
        once = distinct_by_kind.get(kind, [])
        lines.append(
            f"| {kind} | {_counts(group, once, lambda s: True)} | "
            f"{_counts(group, once, lambda s: s.fast)} | "
            f"{_counts(group, once, lambda s: s.official)} | "
            f"{_counts(group, once, lambda s: not s.agree)} |"
        )
    lines += ["", "## Disagreements", ""]
    if not disagreements:
        lines.append("None.")
    else:
        shown = disagreements[:SHOWN_DISAGREEMENTS]
        listed = (
            "each shown once below"
            if len(shown) == len(disagreements)
            else f"the first {len(shown)} shown below, each once"
        )
        lines += [
            f"Disagreeing items: {sum(map(len, disagreements)):,} "
            f"({len(disagreements):,} distinct), {listed}.",
            "Results are the executor's. The official evaluator runs the queries itself,",
            "after removing every DISTINCT from both.",
        ]
        for number, group in enumerate(shown, 1):
            lines += _disagreement(number, group)
    return "\n".join(lines) + "\n"


def _by_kind(scores: Sequence[ItemScore]) -> dict[MutationKind, list[ItemScore]]:
    by_kind: dict[MutationKind, list[ItemScore]] = {}
    for s in scores:
        by_kind.setdefault(s.mutation.kind, []).append(s)
    return by_kind


def _counts(
    items: Sequence[ItemScore], distinct: Sequence[ItemScore], test: Callable[[ItemScore], bool]
) -> str:
    """How many items pass ``test``, then how many distinct items in parentheses."""
    return f"{sum(map(test, items)):,} ({sum(map(test, distinct)):,})"


def _disagreement(number: int, group: Sequence[ItemScore]) -> list[str]:
    s = group[0]
    m = s.mutation
    fast = "fast match" if s.fast else "fast no match"
    official = "official match" if s.official else "official no match"
    order = "row order counts" if s.order_matters else "row order is free"
    ids = ", ".join(r.mutation.example_id for r in group)
    repeats = ["", f"{len(group)} items have this gold and prediction: {ids}."]
    return [
        "",
        f"### {number}. {m.example_id} on {m.db_id}: {m.kind}, {fast}, {official}",
        *(repeats if len(group) > 1 else []),
        "",
        "Gold:",
        "",
        "```sql",
        m.gold_sql.strip(),
        "```",
        "",
        "Prediction:",
        "",
        "```sql",
        m.pred_sql.strip(),
        "```",
        "",
        f"Gold result ({order} for the fast comparator):",
        "",
        *_result_block(s.gold),
        "",
        "Predicted result:",
        "",
        *_result_block(s.pred),
    ]


def _result_block(result: ExecResult) -> list[str]:
    if not result.ok:
        return ["```text", f"{result.error_kind}: {result.error}", "```"]
    rows = result.rows or []
    size = f"{len(rows):,} row{'' if len(rows) == 1 else 's'}"
    if result.truncated:
        size += " (truncated at the row cap)"
    lines = ["```text", f"columns: {', '.join(result.columns or [])}; {size}"]
    lines += [repr(row) for row in rows[:SHOWN_ROWS]]
    if len(rows) > SHOWN_ROWS:
        lines.append(f"... {len(rows) - SHOWN_ROWS:,} more")
    return [*lines, "```"]


def _query(executor: SqliteExecutor, db_id: str, sql: str) -> ExecResult:
    result = executor.run(db_id, sql)
    if not result.ok:
        raise RuntimeError(f"reading the schema of {db_id} failed: {result.error}")
    return result
