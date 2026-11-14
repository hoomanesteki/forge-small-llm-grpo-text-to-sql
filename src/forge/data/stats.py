"""``forge data stats``: question counts by split, by database and by hardness."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import get_args

from forge.core.types import Example, Hardness, Split

LEVELS: tuple[Hardness, ...] = get_args(Hardness)
SPLITS: tuple[Split, ...] = get_args(Split)


def render_stats(examples: Sequence[Example]) -> str:
    """A table per split (with a total), then one row per database, each with its hardness mix."""
    by_split: dict[Split, list[Example]] = defaultdict(list)
    by_db: dict[tuple[Split, str], list[Example]] = defaultdict(list)
    for example in examples:
        by_split[example.split].append(example)
        by_db[(example.split, example.db_id)].append(example)

    split_rows = [
        _row([split], by_split[split], databases=True) for split in SPLITS if split in by_split
    ]
    split_rows.append(_row(["total"], examples, databases=True))
    db_keys = sorted(by_db, key=lambda key: (SPLITS.index(key[0]), key[1]))
    db_rows = [
        _row([db_id, split], by_db[(split, db_id)], databases=False) for split, db_id in db_keys
    ]
    return "\n\n".join(
        [
            _table(["split", "questions", "databases", *LEVELS], split_rows, text_columns=1),
            _table(["database", "split", "questions", *LEVELS], db_rows, text_columns=2),
        ]
    )


def _row(labels: list[str], group: Sequence[Example], *, databases: bool) -> list[str]:
    levels = Counter(example.hardness for example in group)
    counts = [len(group)]
    if databases:
        counts.append(len({example.db_id for example in group}))
    counts += [levels[level] for level in LEVELS]
    return [*labels, *map(str, counts)]


def _table(header: list[str], rows: list[list[str]], *, text_columns: int) -> str:
    """Plain aligned columns: text left-aligned, counts right-aligned."""
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]

    def line(cells: list[str]) -> str:
        padded = [
            cell.ljust(width) if i < text_columns else cell.rjust(width)
            for i, (cell, width) in enumerate(zip(cells, widths, strict=True))
        ]
        return "  ".join(padded).rstrip()

    return "\n".join(line(row) for row in [header, *rows])
