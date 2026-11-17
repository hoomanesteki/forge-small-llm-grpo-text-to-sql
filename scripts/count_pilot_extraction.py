"""Count the pilot replies that extract_sql read by rule 3, and score the queries it cut.

Usage: FORGE_DATA_DIR=data/spider uv run python scripts/count_pilot_extraction.py [RUN]

RUN is a stored run's folder, by default the committed pilot's. A reply with no complete
```sql block is read by rule 3, which keeps one line. Where more lines follow that line,
the whole query, up to a blank line or a code fence, is scored as a run scores a query: run
in a sandbox pool, then by score_generation and official EX. The counts use
extract_sql's own patterns, so they follow its rules. docs/progress.md quotes the output,
and scripts/check_choice_extraction.py scores runs the same way through ``read_whole``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from forge.core.types import Example, ExecResult, Generation, Score
from forge.data.spider import load_raw, resolve_data_dir
from forge.data.splits import split_examples
from forge.eval import testsuite
from forge.eval.agreement import table_columns
from forge.eval.config import OFFICIAL_ITEM_TIMEOUT_S, PRED_TIMEOUT_MS
from forge.eval.gold_cache import MAX_ROWS, run_gold
from forge.eval.run_files import StoredRun, read_run
from forge.eval.scoring import score_generation
from forge.sql.executor import SqliteExecutor
from forge.sql.extract import _QUERY_LINE, _SQL_BLOCK, _THINK_BLOCK, extract_sql
from forge.sql.sandbox import SandboxPool

PILOT = Path("docs/runs/2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7")


@dataclass(frozen=True)
class Whole:
    """A query rule 3 cut, read whole and scored as a run scores a query."""

    generation: Generation  # the stored reply, whose sql is the line rule 3 kept
    query: str
    pred: ExecResult
    verdict: bool  # official EX of the whole query
    score: Score  # the sample's score had the run read the whole query


def unfenced(generation: Generation) -> bool:
    return not _SQL_BLOCK.search(_THINK_BLOCK.sub("", generation.text))


def whole_query(generation: Generation) -> str | None:
    """The query rule 3 cut, from its kept line up to a blank line or a code fence, or
    None where the kept line is the whole of it."""
    lines = _THINK_BLOCK.sub("", generation.text).splitlines()
    kept = max(
        i
        for i, line in enumerate(lines)
        if _QUERY_LINE.match(line) and extract_sql(line) == generation.sql
    )
    end = next(
        (i for i in range(kept + 1, len(lines)) if lines[i].strip() in ("", "```")),
        len(lines),
    )
    if end == kept + 1:
        return None
    return "\n".join(lines[kept:end]).strip().removesuffix(";")


def read_whole(stored: StoredRun, examples: Mapping[str, Example], data: Path) -> list[Whole]:
    """Each reply of ``stored`` whose query rule 3 cut, with the whole query scored.

    ``examples`` holds at least the questions of those replies, by id, and ``data`` is the
    Spider folder the run was made on.
    """
    rule_3 = [g for g in stored.generations.values() if unfenced(g) and g.sql is not None]
    cut = [(g, whole) for g in rule_3 if (whole := whole_query(g)) is not None]
    chosen = [examples[g.example_id] for g, _ in cut]
    gold = run_gold(chosen, data_dir=data)
    with SqliteExecutor(data) as executor:
        columns = {db: table_columns(executor, db) for db in {e.db_id for e in chosen}}
    with SandboxPool(1, data_dir=data, imports=(testsuite.__name__,)) as pool:
        ran = pool.run_many(
            [(e.db_id, whole) for e, (_, whole) in zip(chosen, cut, strict=True)],
            timeout_ms=PRED_TIMEOUT_MS,
            max_rows=MAX_ROWS,
        )
        official = testsuite.official_exec_match(
            [(e.db_id, whole, e.gold_sql) for e, (_, whole) in zip(chosen, cut, strict=True)],
            data_dir=data,
            item_timeout_s=OFFICIAL_ITEM_TIMEOUT_S,
            pool=pool,
        )
    found = []
    for (g, whole), example, pred, verdict in zip(cut, chosen, ran, official, strict=True):
        score = score_generation(
            example,
            g.model_copy(update={"sql": whole}),
            pred,
            gold[example.id],
            columns=columns[example.db_id],
        )
        # As a run fills official EX: only where the fast pass left it open, so a query
        # that timed out stays a miss (forge.eval.runner).
        if score.ex_official is None:
            score = score.model_copy(update={"ex_official": verdict})
        found.append(Whole(g, whole, pred, verdict, score))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", nargs="?", type=Path, default=PILOT)
    args = parser.parse_args()
    data = resolve_data_dir().resolve()
    stored = read_run(args.run)
    scores = {(s.example_id, s.sample_idx): s for s in stored.scores}
    split = stored.manifest.config["split"]
    examples = {e.id: e for e in split_examples(load_raw(data))[split]}

    replies = list(stored.generations.values())
    no_block = [g for g in replies if unfenced(g)]
    rule_3 = [g for g in no_block if g.sql is not None]
    no_query = Counter(
        scores[(g.example_id, g.sample_idx)].error_class for g in no_block if g.sql is None
    )
    cut = read_whole(stored, examples, data)
    print(f"replies: {len(replies)}")
    print(f"with no complete sql block: {len(no_block)}")
    print(f"  read by rule 3: {len(rule_3)}")
    print(f"  with no SELECT or WITH line: {len(no_block) - len(rule_3)}, classes {dict(no_query)}")
    schema_link = sum(score.error_class == "schema_link" for score in stored.scores)
    print(f"schema_link failures in the run: {schema_link}")
    classes = Counter(
        scores[(w.generation.example_id, w.generation.sample_idx)].error_class for w in cut
    )
    print(f"rule 3 kept the first line of a longer query: {len(cut)}, classes {dict(classes)}")

    outcomes: Counter[str] = Counter()
    for whole in cut:
        pred, score = whole.pred, whole.score
        result = "ran" if pred.ok else f"{pred.error_kind}: {pred.error}"
        outcome = "matches gold" if whole.verdict else f"class {score.error_class}"
        print(f"  {whole.generation.example_id} run whole: {result}; {outcome}")
        outcomes["still name a missing table or column"] += pred.error_kind in (
            "no_such_table",
            "no_such_column",
        )
        outcomes["change class"] += score.error_class != "schema_link"
        outcomes["match gold by official EX"] += whole.verdict
    print("run whole, the cut queries " + ", ".join(f"{n} {what}" for what, n in outcomes.items()))


if __name__ == "__main__":
    main()
