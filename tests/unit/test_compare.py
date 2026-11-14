"""The result comparator and the ORDER BY rule, on results built by hand."""

from __future__ import annotations

from typing import Any

import pytest
import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from forge.core.types import ExecResult
from forge.sql.compare import MAX_PERMUTED_COLUMNS, order_matters, results_match


def _ok(rows: list[tuple[Any, ...]], width: int | None = None, **fields: Any) -> ExecResult:
    if width is None:
        width = len(rows[0]) if rows else 1
    return ExecResult(ok=True, columns=[f"c{i}" for i in range(width)], rows=rows, **fields)


def _both_ways(gold: ExecResult, pred: ExecResult) -> tuple[bool, bool]:
    return (
        results_match(gold, pred, order_matters=False),
        results_match(gold, pred, order_matters=True),
    )


@pytest.mark.phase(1)
def test_multiset_ignores_order() -> None:
    gold = _ok([(1, "a"), (2, "b"), (2, "b"), (3, None)])

    shuffled = _ok([(2, "b"), (3, None), (2, "b"), (1, "a")])
    assert results_match(gold, shuffled, order_matters=False)
    # Duplicates count: the same set of rows with a different multiplicity is no match.
    other_counts = _ok([(1, "a"), (1, "a"), (2, "b"), (3, None)])
    assert not results_match(gold, other_counts, order_matters=False)
    assert not results_match(gold, _ok([(1, "a"), (2, "b"), (3, None)]), order_matters=False)
    # Column names don't take part.
    renamed = ExecResult(ok=True, columns=["id", "name"], rows=shuffled.rows)
    assert results_match(gold, renamed, order_matters=False)


@pytest.mark.phase(1)
def test_respects_order_by() -> None:
    gold = _ok([(3, "c"), (2, "b"), (1, "a")])

    assert _both_ways(gold, _ok([(3, "c"), (2, "b"), (1, "a")])) == (True, True)
    assert _both_ways(gold, _ok([(1, "a"), (2, "b"), (3, "c")])) == (True, False)
    # A column swap is still accepted when order matters, if the rows keep their order.
    assert _both_ways(gold, _ok([("c", 3), ("b", 2), ("a", 1)])) == (True, True)
    assert _both_ways(gold, _ok([("a", 1), ("b", 2), ("c", 3)])) == (True, False)


@pytest.mark.phase(1)
@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT name FROM singer ORDER BY age DESC", True),
        ("select count(*) from t group by a order by count(*) desc limit 1", True),
        ("SELECT a FROM t UNION SELECT a FROM u ORDER BY a", True),
        ("SELECT a FROM t EXCEPT SELECT a FROM u ORDER BY 1", True),
        ("SELECT name FROM singer", False),
        # Only the top level counts: an ORDER BY in a subquery or a CTE does not.
        ("SELECT a FROM t WHERE b = (SELECT b FROM u ORDER BY c DESC LIMIT 1)", False),
        ("SELECT * FROM (SELECT a FROM t ORDER BY a)", False),
        ("WITH x AS (SELECT a FROM t ORDER BY a LIMIT 3) SELECT a FROM x", False),
        ("SELECT a FROM t INTERSECT SELECT a FROM (SELECT a FROM u ORDER BY a)", False),
        # Nor does the phrase inside a string or a quoted name.
        ("SELECT a FROM t WHERE note = 'order by date'", False),
        ('SELECT "order by" FROM t', False),
    ],
)
def test_order_matters_needs_top_level_order_by(sql: str, expected: bool) -> None:
    assert order_matters(sql) is expected


@pytest.mark.phase(1)
@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        # sqlglot makes no single query of these, so a regex over the outer query decides.
        ("SELECT a FROM t ORDER BY a LIMIT 1 OFFSET", True),
        ("SELECT a FROM t WHERE b IN (SELECT b FROM u ORDER BY b) AND", False),
        ("SELECT a FROM t WHERE x = 'order by' AND", False),
        ("SELECT a FROM t -- order by a\nWHERE", False),
        ("SELECT 1; SELECT 2 ORDER BY 1", True),
        ("", False),
    ],
)
def test_order_matters_falls_back_to_regex(sql: str, expected: bool) -> None:
    try:
        parsed_as_query = isinstance(sqlglot.parse_one(sql, read="sqlite"), exp.Query)
    except SqlglotError:
        parsed_as_query = False
    assert not parsed_as_query, "the case must reach the fallback"

    assert order_matters(sql) is expected


@pytest.mark.phase(1)
def test_numeric_normalization() -> None:
    def same(gold: Any, pred: Any) -> bool:
        return results_match(_ok([(gold,)]), _ok([(pred,)]), order_matters=True)

    # Numbers compare by value, floats to 6 decimals.
    assert same(1, 1.0)
    assert same(0.3, 0.1 + 0.2)
    assert same(2, 1.9999999)
    assert same(-0.0, 0)
    assert not same(1.0, 1.00001)
    # Integers stay exact: no detour through float, which can't tell these apart.
    assert not same(10**18, 10**18 + 1)
    assert not same(2**53 + 1, float(2**53))
    # A bool is the integer SQLite stores for it.
    assert same(True, 1)
    assert same(False, 0.0)
    assert not same(True, 2)
    # Strings and blobs compare exactly, and never as numbers.
    assert not same("1", 1)
    assert not same("1.0", "1")
    assert not same("Abc", "abc")
    assert not same("a ", "a")
    assert not same(b"a", "a")
    assert same(b"\x00\xff", b"\x00\xff")
    # NULL equals only NULL.
    assert same(None, None)
    for value in (0, 0.0, False, "", "None", "NULL", b""):
        assert not same(None, value)
        assert not same(value, None)


@pytest.mark.phase(1)
def test_column_permutation_accepted() -> None:
    gold = _ok([(1, "a", 2.5), (2, "b", 3.5)])

    assert _both_ways(gold, _ok([("a", 2.5, 1), ("b", 3.5, 2)])) == (True, True)
    assert _both_ways(gold, _ok([("b", 3.5, 2), ("a", 2.5, 1)])) == (True, False)
    # One order for every row: a swap in only some rows is no match.
    assert _both_ways(gold, _ok([("a", 1, 2.5), (2, "b", 3.5)])) == (False, False)
    # A column repeated in the prediction can't stand in for two different gold columns.
    assert _both_ways(gold, _ok([(1, 1, 2.5), (2, 2, 3.5)])) == (False, False)
    # Repeated values, in gold or in the prediction, still find their order.
    assert _both_ways(_ok([(1, 1, 2), (3, 3, 4)]), _ok([(2, 1, 1), (4, 3, 3)])) == (True, True)


@pytest.mark.phase(1)
def test_column_permutation_capped_at_six_columns() -> None:
    def reversed_columns(width: int) -> tuple[ExecResult, ExecResult]:
        rows = [tuple(range(row, row + width)) for row in range(0, 30, 10)]
        return _ok(rows), _ok([tuple(reversed(row)) for row in rows])

    assert MAX_PERMUTED_COLUMNS == 6
    assert _both_ways(*reversed_columns(6)) == (True, True)
    assert _both_ways(*reversed_columns(7)) == (False, False)
    # Seven columns still match column for column.
    gold, _ = reversed_columns(7)
    assert _both_ways(gold, _ok(list(reversed(gold.rows or [])))) == (True, False)


@pytest.mark.phase(1)
def test_widths_and_empty_results() -> None:
    # Two empty results match whatever their columns, as in the official evaluator.
    assert _both_ways(_ok([], width=2), _ok([], width=1)) == (True, True)
    assert _both_ways(_ok([], width=1), _ok([(None,)])) == (False, False)
    # Rows that differ only by an extra column don't match.
    assert _both_ways(_ok([(1,), (2,)]), _ok([(1, 1), (2, 2)])) == (False, False)


@pytest.mark.phase(1)
def test_failed_results_never_match() -> None:
    error = ExecResult(ok=False, error="no such table: t", error_kind="no_such_table")
    empty = _ok([])

    assert _both_ways(error, error) == (False, False)
    assert _both_ways(empty, error) == (False, False)
    assert _both_ways(error, empty) == (False, False)
    # An empty query is a failure too, not an empty result that could match one.
    blank = ExecResult(ok=False, error="no SQL", error_kind="empty_sql")
    assert _both_ways(empty, blank) == (False, False)


@pytest.mark.phase(1)
def test_truncated_results() -> None:
    rows = [(i,) for i in range(5)]
    cut = _ok(rows, truncated=True)

    # Cut at the same cap, one result has more rows than the other.
    assert _both_ways(_ok(rows), cut) == (False, False)
    assert _both_ways(cut, _ok(rows)) == (False, False)
    # Two cut results compare on the rows they kept.
    assert _both_ways(cut, _ok(list(reversed(rows)), truncated=True)) == (True, False)


@pytest.mark.phase(1)
def test_malformed_results_raise() -> None:
    good = _ok([(1,)])

    with pytest.raises(ValueError, match="no columns or rows"):
        results_match(good, ExecResult(ok=True), order_matters=False)
    with pytest.raises(ValueError, match="not 2 values wide"):
        results_match(_ok([(1, 2), (3,)], width=2), good, order_matters=False)
    with pytest.raises(TypeError, match="cannot compare a list value"):
        results_match(good, _ok([([1],)]), order_matters=False)
