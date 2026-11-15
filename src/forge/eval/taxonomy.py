"""Name what a wrong prediction got wrong (stage 8): the error taxonomy.

``classify`` gives a prediction that doesn't match gold the first of these classes that
fits it:

1. ``no_sql``: the reply has no query, or only comments and semicolons where one should be.
2. ``syntax``: SQLite can't parse the query.
3. ``schema_link``: it names a table or column the database doesn't have.
4. ``wrong_tables``: it reads another set of tables.
5. ``join``: it joins the same tables on other conditions, or by another kind of join.
6. ``projection``: a SELECT lists other columns, or differs in ``DISTINCT``.
7. ``aggregation``: a select list calls other aggregates: another function, argument or
   ``DISTINCT`` inside the call.
8. ``where``: a WHERE clause has other conditions.
9. ``group_order``: GROUP BY, HAVING, ORDER BY or LIMIT differ.
10. ``value``: only literals differ.
11. ``other``: none of these, such as another set operation (``UNION`` for
    ``INTERSECT``), or a query sqlglot can't read.

The first three come from the executor's error kind. The rest compare the structure of
the two queries as sqlglot reads them, once each is in one form:

- Names are case-insensitive, and a column is known by the table it reads, so ``T1.name``,
  ``aircraft.name`` and a bare ``name`` that only ``aircraft`` has in its scope are one
  column. That takes the database's columns, which the caller passes.
- Aliases are dropped, those of derived tables and common table expressions too, and a
  select alias used in WHERE, GROUP BY, HAVING or ORDER BY stands for the expression it
  names. Where a table in scope also has a column by that name, the column wins in WHERE,
  GROUP BY and HAVING, and the alias in a whole ORDER BY term, as SQLite reads them.
- A double-quoted name that no table in its scope has is a string, as SQLite reads it,
  so Spider's ``"Boeing 737-800"`` is the value ``'Boeing 737-800'``.
- Order doesn't count where it can't change the rows: the conditions that AND or OR
  joins, the items of a select list or a GROUP BY, the conditions of the joins, and the
  two sides of ``=`` and ``!=``. Parentheses that change nothing are dropped.

A query is compared SELECT by SELECT: its own, each side of a set operation, and those of
its derived tables and common table expressions. A query nested in an expression, such
as a subquery in a WHERE condition, is part of that expression and compares by its own
shape, so a filter whose subquery is missing or selects the wrong column is a ``where``
error. Each comparison before ``value`` masks literals, so a query wrong only in its
constants (a WHERE value, a LIMIT, a HAVING threshold) is ``value``.

A prediction sqlglot can't read as a single query has only its error kind to go on, and
classes 1 to 3 have taken every kind that names a class, so it is ``other``. So is one
whose gold query sqlglot can't read, though sqlglot reads every Spider training gold query.

Known limits: a comma join with its condition in WHERE is not the same join as
``JOIN ... ON``, and both sides of a self-join resolve to one table, so a query that
swaps them can come out ``other``. So can one that swaps two derived tables, or common
table expressions, that have columns of one name.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Literal, NamedTuple, get_args

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from forge.core.types import Example, ExecResult, Generation
from forge.eval.mutations import Columns

ErrorClass = Literal[
    "no_sql",
    "syntax",
    "schema_link",
    "wrong_tables",
    "join",
    "projection",
    "aggregation",
    "where",
    "group_order",
    "value",
    "other",
]
ERROR_CLASSES: tuple[ErrorClass, ...] = get_args(ErrorClass)

# A table's columns by lowercased name, keyed by the lowercased table name.
_Known = Mapping[str, frozenset[str]]
# What each name in one SELECT's FROM and JOIN clauses stands for: a table's own name,
# or None for a derived table or a common table expression.
_Sources = Mapping[str, str | None]

# What an aggregate call in a select list shows as when the projection is compared.
_AGGREGATE = "AGG"
# Stands in for the qualifier of a derived table's or a CTE's column until select aliases
# are expanded, then goes. No table name can be it.
_DERIVED = "(derived)"


class _Shape(NamedTuple):
    """What the classes compare, read off one query in canonical form."""

    tables: tuple[str, ...]  # sorted, each once
    joins: tuple[str, ...]
    projection: tuple[str, ...]
    aggregation: tuple[str, ...]
    where: tuple[str, ...]
    group_order: tuple[str, ...]
    set_operations: tuple[str, ...]


def classify(
    example: Example, generation: Generation, exec_result: ExecResult, *, columns: Columns
) -> ErrorClass:
    """The class of a prediction that doesn't match gold, by the module docstring's rules.

    ``exec_result`` is the predicted query's result, and ``columns`` holds the tables of
    the example's database with their column names, as ``table_columns`` reads them.
    """
    if generation.sql is None or exec_result.error_kind == "empty_sql":
        return "no_sql"
    if exec_result.error_kind == "syntax":
        return "syntax"
    if exec_result.error_kind in ("no_such_table", "no_such_column"):
        return "schema_link"
    known = {table.lower(): frozenset(c.lower() for c in names) for table, names in columns.items()}
    pred = _canonical(generation.sql, known)
    gold = _canonical(example.gold_sql, known)
    if pred is None or gold is None:
        return "other"
    mine, theirs = _shape(pred, masked=True), _shape(gold, masked=True)
    if mine.tables != theirs.tables:
        return "wrong_tables"
    if mine.joins != theirs.joins:
        return "join"
    if mine.projection != theirs.projection:
        return "projection"
    if mine.aggregation != theirs.aggregation:
        return "aggregation"
    if mine.where != theirs.where:
        return "where"
    if mine.group_order != theirs.group_order:
        return "group_order"
    # Alike in every part above; with their set operations alike too, only the literals
    # can still tell the two apart.
    if mine == theirs and _shape(pred, masked=False) != _shape(gold, masked=False):
        return "value"
    return "other"


def _canonical(sql: str, known: _Known) -> exp.Query | None:
    """``sql`` read as one query and put in the module docstring's form, or ``None`` when
    sqlglot can't read it as one."""
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    # sqlglot recurses per nesting level and runs out of stack at about 45 parentheses,
    # which SQLite still runs.
    except (SqlglotError, RecursionError):
        return None
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        return None
    query = statements[0]
    _resolve_names(query, known)
    _expand_alias_refs(query)
    _drop_derived_qualifiers(query)
    _normalize(query)
    return query


def _shape(query: exp.Query, *, masked: bool) -> _Shape:
    """What the classes compare in a canonical query, literals as ``?`` when ``masked``."""
    blocks = list(_blocks(query))
    selects = [block for block in blocks if isinstance(block, exp.Select)]
    ctes = _cte_names(query)
    return _Shape(
        tables=_sorted({t.name for t in query.find_all(exp.Table) if t.name not in ctes}),
        joins=_sorted(
            part
            for select in selects
            for join in select.args.get("joins") or ()
            for part in _join(join, masked)
        ),
        projection=_sorted(_projection(select, masked) for select in selects),
        aggregation=_sorted(
            _render(call, masked=masked) for select in selects for call in _aggregates(select)
        ),
        where=_sorted(
            _render(condition, masked=masked)
            for select in selects
            for condition in _conditions(select.args.get("where"))
        ),
        group_order=_sorted(text for block in blocks if (text := _group_order(block, masked))),
        set_operations=_sorted(
            _set_operation(block) for block in blocks if isinstance(block, exp.SetOperation)
        ),
    )


def _blocks(node: exp.Expr) -> Iterator[exp.Query]:
    """The parts of a query compared in their own right: the query, each side of its set
    operations, and its derived tables and common table expressions, with theirs."""
    if isinstance(node, exp.Subquery):
        yield from _blocks(node.this)
        return
    if not isinstance(node, exp.Query):
        return
    yield node
    with_ = node.args.get("with_")
    for cte in with_.expressions if with_ is not None else ():
        yield from _blocks(cte.this)
    if isinstance(node, exp.SetOperation):
        yield from _blocks(node.this)
        yield from _blocks(node.expression)
    elif isinstance(node, exp.Select):
        for source in _source_nodes(node):
            yield from _blocks(source)


def _signature(query: exp.Query, masked: bool) -> str:
    """A nested query as text that any query of the same shape shares."""
    fields = zip(_Shape._fields, _shape(query, masked=masked), strict=True)
    return "(" + " | ".join(f"{name}: {'; '.join(parts)}" for name, parts in fields if parts) + ")"


def _resolve_names(query: exp.Query, known: _Known) -> None:
    """Qualify each column with the table it reads, or with ``_DERIVED`` where a derived
    table or a CTE's name qualifies it, leave a bare ORDER BY alias for the alias, and turn
    each double-quoted name that no table in its scope has into the string SQLite reads it
    as."""
    ctes = _cte_names(query)
    scopes: dict[int, _Sources] = {}

    def sources(select: exp.Select) -> _Sources:
        if id(select) not in scopes:
            scopes[id(select)] = _sources(select, ctes)
        return scopes[id(select)]

    for column in list(query.find_all(exp.Column)):
        select = column.parent_select
        # Outside any SELECT, as in a set operation's ORDER BY, a name is an output column.
        if select is None:
            continue
        chain = [sources(scope) for scope in _outward(select)]
        if column.table:
            qualifier = column.table.lower()
            scope = next((found for found in chain if qualifier in found), None)
            if scope is not None:
                column.set("table", exp.to_identifier(scope[qualifier] or _DERIVED))
            continue
        name = column.name.lower()
        if _is_order_by_alias(column, select):
            # _expand_alias_refs puts the expression the alias names in its place.
            continue
        in_scope, owner = _lookup(name, chain, known)
        if owner is not None:
            column.set("table", exp.to_identifier(owner))
        elif not in_scope and _quoted(column) and name not in _aliases(select):
            column.replace(exp.Literal.string(column.name))


def _is_order_by_alias(column: exp.Column, select: exp.Select) -> bool:
    """Whether ``column`` is a whole ORDER BY term of ``select`` that names one of its
    aliases, which SQLite reads as the alias before any column by that name."""
    term = column.parent
    return (
        isinstance(term, exp.Ordered)
        and term.parent is select.args.get("order")
        and column.name.lower() in _aliases(select)
    )


def _lookup(name: str, chain: Sequence[_Sources], known: _Known) -> tuple[bool, str | None]:
    """Whether a table in scope could have the column ``name``, and the table that has it
    when only one can.

    The nearest scope with a candidate decides, as in SQLite, and a scope that holds a
    derived table or a common table expression decides nothing more: their columns
    aren't looked up.
    """
    for found in chain:
        owners = {t for t in found.values() if t is not None and name in known.get(t, ())}
        derived = None in found.values()
        if owners or derived:
            return True, owners.pop() if len(owners) == 1 and not derived else None
    return False, None


def _expand_alias_refs(query: exp.Query) -> None:
    """Put each select alias used in WHERE, GROUP BY, HAVING or ORDER BY back as the
    expression it names."""
    for select in list(query.find_all(exp.Select)):
        named = {
            item.alias.lower(): item.this
            for item in select.expressions
            if isinstance(item, exp.Alias)
        }
        if not named:
            continue
        for clause in ("where", "group", "having", "order"):
            node = select.args.get(clause)
            if node is None:
                continue
            for column in list(node.find_all(exp.Column)):
                name = column.name.lower()
                if column.parent_select is select and not column.table and name in named:
                    column.replace(named[name].copy())


def _drop_derived_qualifiers(query: exp.Query) -> None:
    """Drop the name of a derived table or a CTE from the columns it qualifies, once no
    select alias can be read for one of them."""
    for column in list(query.find_all(exp.Column)):
        if column.table == _DERIVED:
            column.set("table", None)


def _normalize(query: exp.Query) -> None:
    """Lowercase names, drop needless parentheses and sort what can be sorted.

    Nodes are visited children first, so each sort key is read off parts already in
    canonical form.
    """
    for node in reversed(list(query.walk(bfs=False))):
        if isinstance(node, exp.Identifier):
            node.set("this", node.name.lower())
            node.set("quoted", False)
        elif isinstance(node, exp.Anonymous):
            # A function sqlglot doesn't know keeps its name as written.
            node.set("this", node.name.lower())
        elif isinstance(node, exp.Paren):
            _unwrap(node)
        elif isinstance(node, exp.EQ | exp.NEQ):
            if _sort_key(node.expression) < _sort_key(node.this):
                node.replace(type(node)(this=node.expression, expression=node.this))
        elif isinstance(node, exp.And | exp.Or) and type(node.parent) is not type(node):
            operands = sorted(_operands(node, type(node)), key=_sort_key)
            chain = operands[0]
            for operand in operands[1:]:
                chain = type(node)(this=chain, expression=operand)
            node.replace(chain)


def _unwrap(paren: exp.Paren) -> None:
    """Drop parentheses that can't change what their contents mean."""
    inner = paren.this
    while isinstance(inner, exp.Paren):
        inner = inner.this
    outer = paren.parent
    if (
        isinstance(inner, exp.Column | exp.Literal)
        or isinstance(outer, exp.Where | exp.Having)
        or (isinstance(outer, exp.Join) and paren.arg_key == "on")
        or (isinstance(outer, exp.Connector | exp.Not) and not isinstance(inner, exp.Connector))
        or (isinstance(inner, exp.Connector) and type(inner) is type(outer))
    ):
        paren.replace(inner)
    elif inner is not paren.this:
        paren.set("this", inner)


def _sources(select: exp.Select, ctes: frozenset[str]) -> _Sources:
    found: dict[str, str | None] = {}
    for node in _source_nodes(select):
        table = isinstance(node, exp.Table) and node.name.lower() not in ctes
        found[node.alias_or_name.lower()] = node.name.lower() if table else None
    return found


def _source_nodes(select: exp.Select) -> list[exp.Expr]:
    """What ``select``'s FROM and JOIN clauses read: tables and derived tables."""
    from_ = select.args.get("from_")
    nodes = [from_.this] if isinstance(from_, exp.From) else []
    nodes.extend(join.this for join in select.args.get("joins") or ())
    return nodes


def _outward(select: exp.Select) -> Iterator[exp.Select]:
    """``select`` and then each SELECT around it, innermost first."""
    scope: exp.Select | None = select
    while scope is not None:
        yield scope
        scope = scope.parent_select


def _aliases(select: exp.Select) -> set[str]:
    return {item.alias.lower() for item in select.expressions if isinstance(item, exp.Alias)}


def _quoted(column: exp.Column) -> bool:
    identifier = column.this
    return isinstance(identifier, exp.Identifier) and bool(identifier.quoted)


def _cte_names(query: exp.Query) -> frozenset[str]:
    return frozenset(cte.alias.lower() for cte in query.find_all(exp.CTE))


def _operands(node: exp.Expr, kind: type[exp.Connector]) -> list[exp.Expr]:
    """What a chain of one connector joins: ``a AND b AND c`` gives a, b and c."""
    if type(node) is kind:
        return [*_operands(node.this, kind), *_operands(node.expression, kind)]
    return [node]


def _conditions(clause: exp.Expr | None) -> list[exp.Expr]:
    """The conditions that AND joins in a WHERE or HAVING clause."""
    return [] if clause is None else _operands(clause.this, exp.And)


def _aggregates(select: exp.Select) -> Iterator[exp.AggFunc]:
    """The aggregate calls in ``select``'s list, not those of the queries nested in it."""
    for item in select.expressions:
        for call in item.find_all(exp.AggFunc):
            if call.parent_select is select:
                yield call


def _projection(select: exp.Select, masked: bool) -> str:
    items = sorted(
        _render(item.this if isinstance(item, exp.Alias) else item, masked=masked, aggregates=True)
        for item in select.expressions
    )
    distinct = "DISTINCT " if select.args.get("distinct") else ""
    return distinct + ", ".join(items)


def _join(join: exp.Join, masked: bool) -> list[str]:
    """A join's kind with each of its conditions. An inner join on nothing is a cross join,
    and ``JOIN`` is ``INNER JOIN``."""
    words = [w for w in (join.method, join.side, join.kind) if w and w not in ("INNER", "OUTER")]
    on, using = join.args.get("on"), join.args.get("using")
    if on is None and not using and not words:
        words = ["CROSS"]
    kind = " ".join([*words, "JOIN"])
    if on is not None:
        return [f"{kind} ON {_render(c, masked=masked)}" for c in _operands(on, exp.And)]
    if using:
        return [f"{kind} USING " + ", ".join(sorted(_render(u, masked=masked) for u in using))]
    return [kind]


def _group_order(block: exp.Query, masked: bool) -> str:
    """The GROUP BY, HAVING, ORDER BY, LIMIT and OFFSET of a SELECT or a set operation."""
    parts: list[str] = []
    group = block.args.get("group")
    if group is not None:
        keys = sorted(_render(key, masked=masked) for key in group.expressions)
        parts.append("GROUP BY " + ", ".join(keys))
    having = block.args.get("having")
    if having is not None:
        conditions = sorted(_render(c, masked=masked) for c in _conditions(having))
        parts.append("HAVING " + " AND ".join(conditions))
    order = block.args.get("order")
    if order is not None:
        keys = [
            _render(key.this, masked=masked) + (" DESC" if key.args.get("desc") else " ASC")
            for key in order.expressions
        ]
        parts.append("ORDER BY " + ", ".join(keys))
    parts.extend(
        _render(clause, masked=masked)
        for name in ("limit", "offset")
        if (clause := block.args.get(name)) is not None
    )
    return " ".join(parts)


def _set_operation(op: exp.SetOperation) -> str:
    return type(op).__name__.upper() + ("" if op.args.get("distinct") else " ALL")


def _sorted(texts: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(texts))


def _sort_key(node: exp.Expr) -> tuple[str, str]:
    """Orders the parts of a canonical query by their shape first, so a part that only
    differs in its literals keeps its place. Parts that tie read the same either way."""
    return _render(node, masked=True), _render(node, masked=False)


def _render(node: exp.Expr, *, masked: bool, aggregates: bool = False) -> str:
    """``node`` as SQLite text, each query nested in it as its signature, its literals as
    ``?`` when ``masked`` and its aggregate calls as ``AGG`` when ``aggregates``."""
    root = node.copy()

    def stand_in(child: exp.Expr) -> exp.Expr:
        if child is not root and isinstance(child, exp.Query):
            return exp.Var(this=_signature(child, masked))
        if masked and isinstance(child, exp.Literal):
            return exp.Placeholder()
        if aggregates and isinstance(child, exp.AggFunc):
            return exp.Var(this=_AGGREGATE)
        return child

    rendered: exp.Expr = root.transform(stand_in, copy=False)
    return rendered.sql(dialect="sqlite")
