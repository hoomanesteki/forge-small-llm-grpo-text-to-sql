"""Count how extract_sql read each stored run's replies, as a table for ADR-0007.

Usage: uv run python scripts/count_extraction.py RUN_FOLDER [RUN_FOLDER ...]

For each run, from its stored generations and scores only: the replies, those with a
```sql block (rule 2 of plan section 3.7), those read by rule 3 (no block, the last line
that starts with SELECT or WITH and parses), and those with no SQL. Of the rule 3 replies,
"cut" counts the ones whose kept line opens a query that goes on over the next lines, up
to a blank line or a code fence, and still parses whole: rule 3 kept only its first line.
The last column gives the error classes of the cut replies. It uses extract_sql's own
patterns, so the counts follow its rules, and ADR-0007 quotes its output for the runs it
compares.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from forge.core.types import Generation
from forge.eval.run_files import StoredRun, key, read_run
from forge.sql.extract import _QUERY_LINE, _SQL_BLOCK, _THINK_BLOCK, extract_sql

HEADER = (
    "| Run | Replies | sql block | Rule 3 | No SQL | Rule 3 cut a query | Cut, by class |\n"
    "|---|---|---|---|---|---|---|"
)


@dataclass(frozen=True)
class Counts:
    run_id: str
    replies: int
    fenced: int
    rule_3: int
    no_sql: int
    cut: int
    cut_classes: Counter[str]

    def row(self) -> str:
        classes = ", ".join(f"{name} {n}" for name, n in sorted(self.cut_classes.items()))
        cells = (self.replies, self.fenced, self.rule_3, self.no_sql, self.cut)
        numbers = " | ".join(f"{n:,}" for n in cells)
        return f"| {self.run_id} | {numbers} | {classes or '-'} |"


def cut_query(generation: Generation) -> str | None:
    """The whole query rule 3 cut from ``generation``'s reply, or None when it cut none."""
    lines = _THINK_BLOCK.sub("", generation.text).splitlines()
    kept = max(
        i
        for i, line in enumerate(lines)
        if _QUERY_LINE.match(line) and extract_sql(line) == generation.sql
    )
    end = kept + 1
    while end < len(lines) and lines[end].strip() and not lines[end].lstrip().startswith("```"):
        end += 1
    # As one line, rule 3 reads the whole query, if it parses as one.
    whole = extract_sql(" ".join(lines[kept:end]))
    return whole if whole is not None and whole != generation.sql else None


def count(run: StoredRun) -> Counts:
    classes = {key(score): score.error_class for score in run.scores}
    fenced = rule_3 = no_sql = cut = 0
    cut_classes: Counter[str] = Counter()
    for item, generation in run.generations.items():
        if _SQL_BLOCK.search(_THINK_BLOCK.sub("", generation.text)):
            fenced += 1
        elif generation.sql is None:
            no_sql += 1
        else:
            rule_3 += 1
            if cut_query(generation) is not None:
                cut += 1
                cut_classes[classes[item] or "match"] += 1
    return Counts(
        run.manifest.run_id, len(run.generations), fenced, rule_3, no_sql, cut, cut_classes
    )


def table(folders: Sequence[Path]) -> str:
    return "\n".join([HEADER, *(count(read_run(folder)).row() for folder in folders)])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="stored run folders")
    print(table(parser.parse_args(argv).runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
