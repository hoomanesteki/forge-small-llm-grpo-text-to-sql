"""Decide whether a predicted query's results match the gold query's, the same way every time.

``results_match`` follows the official evaluator's execution match except where noted
below. The evaluator also drops ``DISTINCT`` from both queries before running them, which
a comparator of results can't do. FG-108 measures how often the two disagree.

- A result that isn't ``ok`` never matches.
- Values are normalized first. Numbers compare by value, so ``1`` matches ``1.0``, and
  floats are rounded to 6 decimals, where the evaluator compares them exactly. A
  ``bool`` counts as the integer SQLite stores for it (``True`` is ``1``); SQLite itself
  never returns one. Strings and blobs compare exactly, and ``None`` (SQL ``NULL``)
  equals only ``None``. Any other type raises ``TypeError``.
- Without ``order_matters`` the rows compare as multisets, so duplicates count. With it
  they compare as lists.
- Column names are ignored. When both results have the same number of columns, 6 or
  fewer, a reordering of the predicted columns that makes the rows match is accepted, as
  in the official evaluator. Wider results must match column for column, where the
  evaluator still searches.
- Two empty results match whatever their columns, as in the official evaluator.
- A result cut at the executor's row cap (``truncated``) never matches one that wasn't:
  with the same cap, the full row counts differ. Two truncated results compare on the
  rows fetched, which is exact for the same query and a best guess otherwise, so gold
  should run with a cap above its size.

``order_matters(gold_sql)`` says whether row order counts: when the gold query has an
``ORDER BY`` at its top level. One in a subquery or a CTE does not count, unlike the
official evaluator, which looks for ``order by`` anywhere in the gold string.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter
from collections.abc import Hashable, Iterator, Sequence
from typing import TypeVar

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from forge.core.types import ExecResult

FLOAT_DECIMALS = 6
# Past 6 columns (720 orders) only the columns as returned are compared.
MAX_PERMUTED_COLUMNS = 6

Value = int | float | str | bytes | None
Row = tuple[Value, ...]
H = TypeVar("H", bound=Hashable)

# For the fallback when sqlglot can't parse the gold query: quoted text and comments
# are blanked so they can't hide or fake an ORDER BY, then nested parentheses are
# removed from the inside out, which leaves the outer query.
_QUOTED_OR_COMMENT = re.compile(
    r"'(?:[^']|'')*'?|\"(?:[^\"]|\"\")*\"?|`(?:[^`]|``)*`?|\[[^\]]*\]?|--[^\n]*|/\*.*?(?:\*/|\Z)",
    re.DOTALL,
)
_INNERMOST_PARENS = re.compile(r"\([^()]*\)")
_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)


def results_match(gold: ExecResult, pred: ExecResult, *, order_matters: bool) -> bool:
    """Whether ``pred``'s rows are ``gold``'s, by the rules in the module docstring."""
    if not (gold.ok and pred.ok) or gold.truncated != pred.truncated:
        return False
    gold_width, gold_rows = _normalized(gold, "gold")
    pred_width, pred_rows = _normalized(pred, "pred")
    if len(gold_rows) != len(pred_rows):
        return False
    expected = _arranged(gold_rows, order_matters)
    if _arranged(pred_rows, order_matters) == expected:
        return True
    if not gold_rows or gold_width != pred_width or gold_width > MAX_PERMUTED_COLUMNS:
        return False
    return any(
        _arranged([tuple(row[i] for i in order) for row in pred_rows], order_matters) == expected
        for order in _column_orders(gold_rows, pred_rows, gold_width, order_matters)
    )


def order_matters(gold_sql: str) -> bool:
    """Whether the gold query has a top-level ``ORDER BY``, so row order counts."""
    try:
        tree = sqlglot.parse_one(gold_sql, read="sqlite")
    except SqlglotError:
        return _has_outer_order_by(gold_sql)
    if not isinstance(tree, exp.Query):
        # sqlglot hands back a Command or a Block for SQL it only half understands.
        return _has_outer_order_by(gold_sql)
    # A compound query (UNION, INTERSECT, EXCEPT) holds its ORDER BY on the compound node.
    return tree.args.get("order") is not None


def _normalize_value(value: object) -> Value:
    if value is None or isinstance(value, str | bytes):
        return value
    # bool before int: bool is a subclass of int, and stays one after this.
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, FLOAT_DECIMALS)
    raise TypeError(f"cannot compare a {type(value).__name__} value: {value!r}")


def _normalized(result: ExecResult, name: str) -> tuple[int, list[Row]]:
    if result.columns is None or result.rows is None:
        raise ValueError(f"{name} result is ok but has no columns or rows")
    width = len(result.columns)
    if any(len(row) != width for row in result.rows):
        raise ValueError(f"{name} result has rows that are not {width} values wide")
    rows = [tuple(_normalize_value(value) for value in row) for row in result.rows]
    return width, rows


def _column_orders(
    gold: Sequence[Row], pred: Sequence[Row], width: int, order_matters: bool
) -> Iterator[tuple[int, ...]]:
    """Orders of ``pred``'s columns worth trying, other than the order as returned.

    Two necessary conditions keep the search small. Reordering columns keeps the values
    in each row, so the rows' contents must agree before any order is tried, as in the
    official evaluator. And a predicted column can only stand in for a gold column that
    holds the same values (the same sequence, or with order free the same multiset), so
    each gold column gets only those candidates.
    """
    gold_contents = _arranged([_contents(row) for row in gold], order_matters)
    if _arranged([_contents(row) for row in pred], order_matters) != gold_contents:
        return
    gold_columns = [_arranged([row[i] for row in gold], order_matters) for i in range(width)]
    pred_columns = [_arranged([row[i] for row in pred], order_matters) for i in range(width)]
    candidates = [
        [j for j, column in enumerate(pred_columns) if column == gold_column]
        for gold_column in gold_columns
    ]
    as_returned = tuple(range(width))
    for order in itertools.product(*candidates):
        if order != as_returned and len(set(order)) == width:
            yield order


def _arranged(items: list[H], order_matters: bool) -> list[H] | Counter[H]:
    """``items`` as a list when order matters, else as a multiset."""
    return items if order_matters else Counter(items)


def _contents(row: Row) -> frozenset[tuple[Value, int]]:
    """The row's values with their counts, whatever their column order."""
    return frozenset(Counter(row).items())


def _has_outer_order_by(sql: str) -> bool:
    outer = _QUOTED_OR_COMMENT.sub(" ", sql)
    while True:
        stripped = _INNERMOST_PARENS.sub(" ", outer)
        if stripped == outer:
            break
        outer = stripped
    return _ORDER_BY.search(outer) is not None
