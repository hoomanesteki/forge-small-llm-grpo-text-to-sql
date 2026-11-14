"""Predictions made from gold queries, mostly wrong, for the agreement check.

``make_mutations`` changes gold queries the way a model might get them wrong, so the
fast comparator and the official evaluator can be compared on predictions that match
and on predictions that don't. About 30% of the items keep the gold query; each of the
others gets one of these changes, made with sqlglot:

- ``drop_where``: one condition of a WHERE clause goes, and the clause with its last.
  Conditions are what AND and OR join; a parenthesized group counts as one.
- ``swap_column``: a column in a select list becomes another column of the tables in
  that SELECT's FROM and JOIN clauses.
- ``change_aggregate``: COUNT becomes SUM (``COUNT(*)`` sums a column of its SELECT's
  tables), SUM becomes AVG, AVG becomes SUM, MIN becomes MAX and MAX becomes MIN.
- ``flip_order``: one ORDER BY key changes direction, NULLs with it, so its order
  reverses exactly.
- ``change_literal``: a number goes up by one; a quoted string has its letter case
  swapped, or gains an ``x`` when it has no letters. SQLite reads a double-quoted name
  that matches no column as a string, as Spider often writes them, but sqlglot reads it
  as a name, so only single-quoted strings change.

Each change picks its target at random among the candidates in the whole query,
subqueries included. The changes are tried in a random order and the first that has a
candidate is used; a query that none of them fits stays unchanged. A changed query is
sqlglot's SQLite rendering of the edited tree, so its spelling can differ from the gold's
elsewhere too; an unchanged one is the gold text itself.

Everything random comes from ``random.Random`` seeded with the seed and the item's
position, so a seed always gives the same items, and a shorter run is the start of a
longer one.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple, get_args

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from forge.core.types import Example

MutationKind = Literal[
    "unchanged", "drop_where", "swap_column", "change_aggregate", "flip_order", "change_literal"
]
Change = Literal["drop_where", "swap_column", "change_aggregate", "flip_order", "change_literal"]
CHANGES: tuple[Change, ...] = get_args(Change)
UNCHANGED_SHARE = 0.3

# A database's tables, lowercased, and the names of their columns.
Columns = Mapping[str, Sequence[str]]

_AGGREGATE_SWAPS: dict[type[exp.AggFunc], type[exp.AggFunc]] = {
    exp.Count: exp.Sum,
    exp.Sum: exp.Avg,
    exp.Avg: exp.Sum,
    exp.Min: exp.Max,
    exp.Max: exp.Min,
}


@dataclass(frozen=True)
class Mutation:
    """A prediction for one example: the gold query, changed by ``kind`` or not at all."""

    example_id: str
    db_id: str
    kind: MutationKind
    gold_sql: str
    pred_sql: str


def make_mutations(
    examples: Sequence[Example], n: int, *, seed: int, columns: Mapping[str, Columns]
) -> list[Mutation]:
    """``n`` predictions, cycling through ``examples`` in an order shuffled by ``seed``.

    ``columns`` maps each database id to its tables' columns, for the changes that need
    a column the query doesn't name.
    """
    if n < 0:
        raise ValueError(f"n must be zero or more, got {n}")
    if not examples:
        raise ValueError("mutations need at least one example")
    order = list(examples)
    random.Random(f"forge-mutations-{seed}").shuffle(order)
    mutations: list[Mutation] = []
    for i in range(n):
        example = order[i % len(order)]
        rng = random.Random(f"forge-mutations-{seed}-{i}")
        mutations.append(_mutation(example, rng, columns.get(example.db_id, {})))
    return mutations


def mutate(sql: str, change: Change, rng: random.Random, columns: Columns) -> str | None:
    """``sql`` with ``change`` made, or ``None`` when it has nothing to make it to."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except SqlglotError:
        return None
    if not _CHANGERS[change](tree, rng, columns):
        return None
    return tree.sql(dialect="sqlite")


def _mutation(example: Example, rng: random.Random, columns: Columns) -> Mutation:
    if rng.random() >= UNCHANGED_SHARE:
        for change in rng.sample(CHANGES, len(CHANGES)):
            sql = mutate(example.gold_sql, change, rng, columns)
            if sql is not None:
                return Mutation(example.id, example.db_id, change, example.gold_sql, sql)
    return Mutation(example.id, example.db_id, "unchanged", example.gold_sql, example.gold_sql)


def _drop_where(tree: exp.Expr, rng: random.Random, columns: Columns) -> bool:
    conditions = [c for where in tree.find_all(exp.Where) for c in _conditions(where.this)]
    if not conditions:
        return False
    condition = rng.choice(conditions)
    parent = condition.parent
    if isinstance(parent, exp.Connector):
        parent.replace(parent.expression if condition is parent.this else parent.this)
    else:
        # The clause's only condition: _conditions starts at the WHERE.
        assert isinstance(parent, exp.Where)
        parent.pop()
    return True


def _swap_column(tree: exp.Expr, rng: random.Random, columns: Columns) -> bool:
    choices: list[tuple[exp.Column, list[exp.Column]]] = []
    for select in tree.find_all(exp.Select):
        scope = _scope_columns(select, columns)
        for projection in select.expressions:
            for column in projection.find_all(exp.Column):
                # A column of a subquery in the select list belongs to that subquery.
                if isinstance(column.this, exp.Star) or column.parent_select is not select:
                    continue
                same = (_source_of(column, scope), column.name.lower())
                others = [c.column for c in scope if (c.source, c.name) != same]
                if others:
                    choices.append((column, others))
    if not choices:
        return False
    column, others = rng.choice(choices)
    column.replace(rng.choice(others).copy())
    return True


def _change_aggregate(tree: exp.Expr, rng: random.Random, columns: Columns) -> bool:
    choices: list[tuple[exp.AggFunc, list[exp.Expression]]] = []
    for node in tree.find_all(exp.AggFunc):
        # Exact types only, and one argument only: SQLite's two-argument min and max
        # are scalar functions.
        if type(node) not in _AGGREGATE_SWAPS or node.expressions:
            continue
        if isinstance(node.this, exp.Star):
            select = node.parent_select
            scope: list[exp.Expression] = (
                [c.column for c in _scope_columns(select, columns)] if select is not None else []
            )
            if scope:
                choices.append((node, scope))
        else:
            choices.append((node, [node.this]))
    if not choices:
        return False
    node, arguments = rng.choice(choices)
    node.replace(_AGGREGATE_SWAPS[type(node)](this=rng.choice(arguments).copy()))
    return True


def _flip_order(tree: exp.Expr, rng: random.Random, columns: Columns) -> bool:
    keys = list(tree.find_all(exp.Ordered))
    if not keys:
        return False
    key = rng.choice(keys)
    desc = not key.args.get("desc")
    key.set("desc", desc)
    # SQLite puts NULLs first ascending and last descending, so this reverses the whole
    # order and renders without a NULLS clause.
    key.set("nulls_first", not desc)
    return True


def _change_literal(tree: exp.Expr, rng: random.Random, columns: Columns) -> bool:
    choices = [
        (literal, changed)
        for literal in tree.find_all(exp.Literal)
        if (changed := _changed_literal(literal)) is not None
    ]
    if not choices:
        return False
    literal, changed = rng.choice(choices)
    literal.replace(changed)
    return True


_CHANGERS: dict[Change, Callable[[exp.Expr, random.Random, Columns], bool]] = {
    "drop_where": _drop_where,
    "swap_column": _swap_column,
    "change_aggregate": _change_aggregate,
    "flip_order": _flip_order,
    "change_literal": _change_literal,
}


def _conditions(node: exp.Expression) -> list[exp.Expression]:
    """The conditions an AND and OR chain joins, without looking inside parentheses."""
    if isinstance(node, exp.Connector):
        return [*_conditions(node.this), *_conditions(node.expression)]
    return [node]


class _InScope(NamedTuple):
    source: str  # the table's alias, else its name, lowercased
    name: str  # lowercased
    column: exp.Column  # ready to put in the query


def _scope_columns(select: exp.Select, columns: Columns) -> list[_InScope]:
    """Every column of the tables in ``select``'s FROM and JOIN clauses.

    Each is qualified by its table's alias, or by the table's name when there are
    several tables, so a swapped-in column is never ambiguous.
    """
    sources: list[exp.Expression] = []
    from_ = select.args.get("from_")
    if isinstance(from_, exp.From):
        sources.append(from_.this)
    sources.extend(join.this for join in select.args.get("joins") or ())
    tables = [source for source in sources if isinstance(source, exp.Table)]
    qualify = len(tables) > 1
    scope: list[_InScope] = []
    for table in tables:
        qualifier = table.alias or (table.name if qualify else None)
        for name in columns.get(table.name.lower(), ()):
            column = exp.column(name, table=qualifier)
            scope.append(_InScope(table.alias_or_name.lower(), name.lower(), column))
    return scope


def _source_of(column: exp.Column, scope: Sequence[_InScope]) -> str:
    """Which table in scope ``column`` reads: its qualifier, else the first that has it."""
    if column.table:
        return column.table.lower()
    name = column.name.lower()
    return next((c.source for c in scope if c.name == name), "")


def _changed_literal(literal: exp.Literal) -> exp.Expression | None:
    text: str = literal.this
    if literal.is_string:
        swapped = text.swapcase()
        return exp.Literal.string(swapped if swapped != text else f"{text}x")
    for parse in (int, float):
        try:
            return exp.Literal.number(parse(text) + 1)
        except ValueError:
            continue  # not an integer: try it as a float
    return None
