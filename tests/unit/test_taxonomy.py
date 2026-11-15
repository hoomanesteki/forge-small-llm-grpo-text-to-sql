"""The error taxonomy: the class a wrong prediction gets, and what reads as the same query.

Gold queries are the fixture's flight_1 questions (``spider-train-000NN``), and error
results carry the messages SQLite gives for the predicted queries on that database.
"""

from __future__ import annotations

import pytest

from forge.core.types import Example, ExecResult, Generation
from forge.eval.scoring import no_sql
from forge.eval.taxonomy import ERROR_CLASSES, ErrorClass, classify
from forge.sql.executor import MULTIPLE_STATEMENTS

# flight_1 as table_columns reads it. A name, a distance and an aid are each in more than
# one table, so a bare column only resolves where one table in scope has it.
FLIGHT_1 = {
    "aircraft": ("aid", "name", "distance"),
    "certificate": ("eid", "aid"),
    "employee": ("eid", "name", "salary"),
    "flight": (
        *("flno", "origin", "destination", "distance"),
        *("departure_date", "arrival_date", "price", "aid"),
    ),
}
# A query that ran; the taxonomy never looks at the rows.
RAN = ExecResult(ok=True, columns=["n"], rows=[(1,)])

GOLD_10 = "SELECT count(*) FROM Aircraft"
GOLD_14 = "SELECT aid FROM Aircraft WHERE distance  >  1000"
GOLD_16 = "SELECT count(*) FROM Aircraft WHERE distance BETWEEN 1000 AND 5000"
GOLD_17 = "SELECT name ,  distance FROM Aircraft WHERE aid  =  12"
GOLD_19 = "SELECT min(distance) ,  avg(distance) ,  max(distance) FROM Aircraft"
GOLD_20 = "SELECT name FROM Aircraft WHERE distance  >  (SELECT avg(distance) FROM Aircraft)"
GOLD_22 = "SELECT name FROM Employee WHERE salary  >  (SELECT avg(salary) FROM Employee)"
GOLD_24 = "SELECT origin FROM Flight GROUP BY origin ORDER BY count(*) DESC LIMIT 1"
GOLD_25 = (
    "SELECT T1.name FROM Employee AS T1 JOIN Certificate AS T2 ON T1.eid  =  T2.eid "
    'JOIN Aircraft AS T3 ON T3.aid  =  T2.aid WHERE T3.name  =  "Boeing 737-800" '
    "INTERSECT SELECT T1.name FROM Employee AS T1 JOIN Certificate AS T2 ON T1.eid  =  "
    'T2.eid JOIN Aircraft AS T3 ON T3.aid  =  T2.aid WHERE T3.name  =  "Airbus A340-300"'
)
GOLD_27 = (
    "SELECT T2.name FROM Certificate AS T1 JOIN Aircraft AS T2 ON T2.aid  =  T1.aid "
    "GROUP BY T1.aid ORDER BY count(*) DESC LIMIT 1"
)

# One case per class, in class order: (class, gold query, predicted query, its result).
CASES: list[tuple[ErrorClass, str, str | None, ExecResult]] = [
    ("no_sql", GOLD_10, None, no_sql()),
    (
        "syntax",
        GOLD_14,
        "SELECT aid FROM Aircraft WHERE distance >",
        ExecResult(ok=False, error="incomplete input", error_kind="syntax"),
    ),
    (
        "schema_link",
        GOLD_17,
        "SELECT name, flight_range FROM Aircraft WHERE aid = 12",
        ExecResult(ok=False, error="no such column: flight_range", error_kind="no_such_column"),
    ),
    (
        "wrong_tables",
        GOLD_22,
        "SELECT name FROM Aircraft WHERE distance > (SELECT avg(distance) FROM Aircraft)",
        RAN,
    ),
    (
        "join",
        GOLD_27,
        "SELECT T2.name FROM Certificate AS T1 JOIN Aircraft AS T2 ON T2.aid = T1.eid "
        "GROUP BY T1.aid ORDER BY count(*) DESC LIMIT 1",
        RAN,
    ),
    ("projection", GOLD_17, "SELECT name FROM Aircraft WHERE aid = 12", RAN),
    (
        "aggregation",
        GOLD_19,
        "SELECT min(distance), sum(distance), max(distance) FROM Aircraft",
        RAN,
    ),
    ("where", GOLD_14, "SELECT aid FROM Aircraft WHERE distance < 1000", RAN),
    (
        "group_order",
        GOLD_24,
        "SELECT origin FROM Flight GROUP BY origin ORDER BY count(*) LIMIT 1",
        RAN,
    ),
    ("value", GOLD_16, "SELECT count(*) FROM Aircraft WHERE distance BETWEEN 1000 AND 6000", RAN),
    ("other", GOLD_25, GOLD_25.replace(" INTERSECT ", " UNION "), RAN),
]


def _classify(gold_sql: str, pred_sql: str | None, result: ExecResult = RAN) -> ErrorClass:
    example = Example(
        id="spider-train-00010",
        db_id="flight_1",
        question="How many aircrafts do we have?",
        gold_sql=gold_sql,
        hardness="easy",
        split="val",
    )
    generation = Generation(
        example_id=example.id,
        model_ref="test",
        text=f"```sql\n{pred_sql}\n```" if pred_sql is not None else "I can't answer that.",
        sql=pred_sql,
        tokens_in=10,
        tokens_out=5,
        latency_ms=1.0,
    )
    return classify(example, generation, result, columns=FLIGHT_1)


@pytest.mark.phase(2)
@pytest.mark.parametrize(("expected", "gold", "pred", "result"), CASES, ids=[c[0] for c in CASES])
def test_taxonomy_cases(
    expected: ErrorClass, gold: str, pred: str | None, result: ExecResult
) -> None:
    assert _classify(gold, pred, result) == expected


@pytest.mark.phase(2)
def test_every_class_has_a_case_in_the_plan_order() -> None:
    plan_order = (
        *("no_sql", "syntax", "schema_link", "wrong_tables", "join", "projection"),
        *("aggregation", "where", "group_order", "value", "other"),
    )
    assert plan_order == ERROR_CLASSES
    assert tuple(case[0] for case in CASES) == ERROR_CLASSES


@pytest.mark.phase(2)
def test_the_first_class_that_fits_wins() -> None:
    # The error kind comes first, whatever else differs: these also read another table.
    syntax = ExecResult(ok=False, error="incomplete input", error_kind="syntax")
    assert _classify(GOLD_14, "SELECT eid FROM Employee WHERE salary >", syntax) == "syntax"
    missing = ExecResult(ok=False, error="no such table: Planes", error_kind="no_such_table")
    assert _classify(GOLD_14, "SELECT eid FROM Planes", missing) == "schema_link"
    # Then the structure, part by part in class order.
    assert _classify(GOLD_14, "SELECT eid FROM Employee WHERE salary < 1000") == "wrong_tables"
    assert _classify(GOLD_14, "SELECT name FROM Aircraft WHERE distance < 1000") == "projection"
    assert _classify(GOLD_14, "SELECT count(aid) FROM Aircraft WHERE distance > 1000") == (
        "projection"
    )
    pred = "SELECT min(distance), sum(distance), max(distance) FROM Aircraft WHERE aid > 2"
    assert _classify(GOLD_19, pred) == "aggregation"
    pred = "SELECT origin FROM Flight WHERE price > 100 GROUP BY origin ORDER BY count(*) LIMIT 1"
    assert _classify(GOLD_24, pred) == "where"
    # A difference in a literal on top of one in structure is the structure's class.
    pred = "SELECT origin FROM Flight GROUP BY origin ORDER BY count(*) LIMIT 5"
    assert _classify(GOLD_24, pred) == "group_order"


@pytest.mark.phase(2)
def test_names_resolve_to_the_tables_they_read() -> None:
    # Each prediction differs from its gold query in one literal only, however it names
    # tables and columns: case, aliases, table names, or bare names one table in scope has.
    pred = "select T9.AID from AIRCRAFT as T9 where T9.Distance > 2000"
    assert _classify(GOLD_14, pred) == "value"
    pred = (
        "SELECT T2.name FROM Aircraft T2 JOIN Certificate T1 ON T1.aid = T2.aid "
        "GROUP BY T1.aid ORDER BY count(*) DESC LIMIT 3"
    )
    assert _classify(GOLD_27, pred) == "value"
    gold = (
        "SELECT T1.name FROM Employee AS T1 JOIN Certificate AS T2 ON T1.eid = T2.eid "
        "WHERE T1.salary > 100000"
    )
    pred = (
        "SELECT employee.name FROM certificate JOIN employee "
        "ON certificate.eid = employee.eid WHERE salary > 50000"
    )
    assert _classify(gold, pred) == "value"
    # A bare name that two tables in scope have stays unresolved, as SQLite refuses it.
    ambiguous = ExecResult(ok=False, error="ambiguous column name: name", error_kind="other")
    pred = (
        "SELECT name FROM Employee AS T1 JOIN Certificate AS T2 ON T1.eid = T2.eid "
        'JOIN Aircraft AS T3 ON T3.aid = T2.aid WHERE T3.name = "Boeing 737-800"'
    )
    assert _classify(GOLD_25.split(" INTERSECT ")[0], pred, ambiguous) == "projection"


@pytest.mark.phase(2)
def test_double_quoted_names_read_as_sqlite_reads_them() -> None:
    gold = 'SELECT aid FROM Aircraft WHERE name = "Boeing 737-800"'
    # No table has a column by that name, so it is a string, as single quotes would be.
    assert _classify(gold, "SELECT aid FROM aircraft WHERE name = 'Airbus A340-300'") == "value"
    assert _classify(gold, 'SELECT aid FROM aircraft WHERE name = "Airbus A340-300"') == "value"
    assert _classify(gold, 'SELECT aid FROM aircraft WHERE name = "boeing 737-800"') == "value"
    assert _classify(gold, "SELECT aid FROM aircraft WHERE name = 'Boeing 737-800'") == "other"
    # A name a table in scope has is that column, quoted or not.
    assert _classify(GOLD_14, 'SELECT "aid" FROM "Aircraft" WHERE "distance" > 5000') == "value"


@pytest.mark.phase(2)
def test_order_that_cannot_change_the_rows_does_not_count() -> None:
    # Again one literal differs, while conditions, items and the sides of = move around.
    gold = "SELECT name FROM Aircraft WHERE distance > 1000 AND aid < 10"
    assert _classify(gold, "SELECT name FROM aircraft WHERE (aid < 20) AND (distance > 1000)") == (
        "value"
    )
    assert _classify(GOLD_17, "SELECT distance, name FROM aircraft WHERE 13 = aid") == "value"
    gold = "SELECT name FROM Aircraft WHERE aid = 1 OR distance > 1000"
    assert _classify(gold, "SELECT name FROM aircraft WHERE (distance > 2000 OR aid = 1)") == (
        "value"
    )
    gold = "SELECT name FROM Aircraft WHERE (aid = 1 AND distance > 1000) OR name = 'A'"
    pred = "SELECT name FROM aircraft WHERE name = 'B' OR (distance > 1000 AND aid = 1)"
    assert _classify(gold, pred) == "value"
    gold = (
        "SELECT origin, destination, count(*) FROM Flight GROUP BY origin, destination "
        "HAVING count(*) > 1"
    )
    pred = (
        "SELECT destination, origin, count(*) FROM flight GROUP BY destination, origin "
        "HAVING count(*) > 2"
    )
    assert _classify(gold, pred) == "value"
    # A select alias in ORDER BY stands for its expression.
    gold = "SELECT origin, count(*) FROM Flight GROUP BY origin ORDER BY count(*) DESC LIMIT 1"
    pred = "SELECT origin, count(*) AS n FROM flight GROUP BY origin ORDER BY n DESC LIMIT 2"
    assert _classify(gold, pred) == "value"
    # Parentheses around a whole ORDER BY term change nothing.
    pred = "SELECT origin, count(*) FROM flight GROUP BY origin ORDER BY (count(*)) DESC LIMIT 2"
    assert _classify(gold, pred) == "value"
    # The order of ORDER BY keys does change the rows.
    gold = "SELECT name FROM Aircraft ORDER BY distance, name"
    assert _classify(gold, "SELECT name FROM aircraft ORDER BY name, distance") == "group_order"


@pytest.mark.phase(2)
def test_an_order_by_alias_wins_over_a_column_as_in_sqlite() -> None:
    # A whole ORDER BY term that names a select alias is the alias, though a table in scope
    # has a column by that name: SQLite orders these by max(distance) and by -aid.
    gold = "SELECT aid, max(distance) FROM Flight GROUP BY aid ORDER BY max(distance) DESC LIMIT 1"
    pred = (
        "SELECT aid, max(distance) AS distance FROM flight GROUP BY aid "
        "ORDER BY distance DESC LIMIT 3"
    )
    assert _classify(gold, pred) == "value"
    gold = "SELECT aid, -aid AS distance FROM Aircraft ORDER BY distance LIMIT 3"
    assert _classify(gold, "SELECT aid, -aid AS distance FROM aircraft ORDER BY -aid LIMIT 5") == (
        "value"
    )
    # In parentheses, or with COLLATE, the alias is still the whole term: SQLite's parser
    # drops the parentheses and its resolver skips COLLATE.
    pred = "SELECT aid, -aid AS distance FROM aircraft ORDER BY (distance) LIMIT 5"
    assert _classify(gold, pred) == "value"
    collated = "SELECT name AS distance FROM Aircraft ORDER BY name COLLATE NOCASE LIMIT 3"
    for term in (
        "distance COLLATE NOCASE",
        "(distance) COLLATE NOCASE",
        "(distance COLLATE NOCASE)",
    ):
        pred = f"SELECT name AS distance FROM aircraft ORDER BY {term} LIMIT 5"
        assert _classify(collated, pred) == "value", term
    # Qualified, or inside an expression, the name is the column, as it is in WHERE.
    pred = "SELECT aid, -aid AS distance FROM aircraft ORDER BY aircraft.distance LIMIT 5"
    assert _classify(gold, pred) == "group_order"
    gold = "SELECT aid, -aid AS distance FROM Aircraft ORDER BY distance + 1 LIMIT 3"
    pred = "SELECT aid, -aid AS distance FROM aircraft ORDER BY aircraft.distance + 1 LIMIT 5"
    assert _classify(gold, pred) == "value"
    gold = "SELECT aid, -aid AS distance FROM Aircraft WHERE distance > 1000"
    pred = "SELECT aid, -aid AS distance FROM aircraft WHERE aircraft.distance > 2000"
    assert _classify(gold, pred) == "value"


@pytest.mark.phase(2)
def test_derived_tables_and_ctes_lose_their_names() -> None:
    # The columns of a derived table or a common table expression drop its name, as a
    # table's columns drop its alias, so here only a literal differs.
    gold = "SELECT T1.origin FROM (SELECT origin FROM Flight WHERE price > 100) AS T1"
    pred = "SELECT sub.origin FROM (SELECT origin FROM flight WHERE price > 200) AS sub"
    assert _classify(gold, pred) == "value"
    pred = "SELECT origin FROM (SELECT origin FROM Flight WHERE price > 200) AS T1"
    assert _classify(gold, pred) == "value"
    gold = "WITH c AS (SELECT origin FROM Flight WHERE price > 100) SELECT c.origin FROM c"
    pred = "WITH d AS (SELECT origin FROM Flight WHERE price > 200) SELECT d.origin FROM d"
    assert _classify(gold, pred) == "value"
    pred = "WITH d AS (SELECT origin FROM Flight WHERE price > 200) SELECT x.origin FROM d AS x"
    assert _classify(gold, pred) == "value"
    # A qualified column still isn't the select alias by its name: gold orders by the
    # derived table's n, and the prediction by its alias, n + 1.
    derived = "(SELECT count(*) AS n FROM Flight GROUP BY origin)"
    gold = f"SELECT T1.n + 1 AS n FROM {derived} AS T1 ORDER BY T1.n"
    assert _classify(gold, f"SELECT sub.n + 1 AS n FROM {derived} AS sub ORDER BY n") == (
        "group_order"
    )
    assert _classify(gold, f"SELECT sub.n + 2 AS n FROM {derived} AS sub ORDER BY sub.n") == (
        "value"
    )


@pytest.mark.phase(2)
def test_a_derived_column_wins_over_an_alias_as_in_sqlite() -> None:
    # A bare name that a derived table's or a CTE's select list outputs is that column in
    # WHERE, GROUP BY, HAVING and inside an ORDER BY expression, though a select alias has
    # the name too: SQLite filters these gold queries on origin, not on origin || 'x'.
    derived = "SELECT T1.origin || 'x' AS origin FROM (SELECT origin FROM Flight) AS T1"
    cte = "WITH T1 AS (SELECT origin FROM Flight) SELECT T1.origin || 'x' AS origin FROM T1"
    for query in (derived, cte):
        gold = f"{query} WHERE origin = 'Chicago'"
        assert _classify(gold, f"{query} WHERE T1.origin = 'Boston'") == "value"
        assert _classify(gold, f"{query} WHERE T1.origin || 'x' = 'Boston'") == "where"
    grouped = "SELECT count(*) AS origin FROM (SELECT origin FROM Flight) AS T1 GROUP BY"
    assert _classify(f"{grouped} origin LIMIT 3", f"{grouped} T1.origin LIMIT 5") == "value"
    counts = "SELECT T1.n + 1 AS n FROM (SELECT count(*) AS n FROM Flight GROUP BY origin) AS T1"
    gold = f"{counts} GROUP BY T1.n HAVING n > 1"
    assert _classify(gold, f"{counts} GROUP BY T1.n HAVING T1.n > 2") == "value"
    gold = f"{counts} ORDER BY n * -1 LIMIT 1"
    assert _classify(gold, f"{counts} ORDER BY T1.n * -1 LIMIT 2") == "value"
    # A CTE's column list names its columns.
    listed = "WITH c(place) AS (SELECT origin FROM Flight) SELECT place || 'x' AS place FROM c"
    assert _classify(f"{listed} WHERE place = 'Chicago'", f"{listed} WHERE c.place = 'Boston'") == (
        "value"
    )
    # A name the derived table doesn't output isn't its column: a double-quoted one is a
    # string, and a bare one that a table in scope has is that table's.
    named = "SELECT T1.aid FROM (SELECT aid, name FROM Aircraft) AS T1 WHERE name ="
    assert _classify(f"{named} 'Boeing 737-800'", f'{named} "Airbus A340-300"') == "value"
    join = "SELECT T2.name FROM (SELECT aid FROM Certificate) AS T1 JOIN Aircraft AS T2"
    gold = f"{join} ON T1.aid = T2.aid WHERE T2.distance > 1000"
    assert _classify(gold, f"{join} ON T1.aid = T2.aid WHERE distance > 2000") == "value"
    # Where a * hides the derived table's columns, a name it may have is left as written,
    # neither its column nor the alias.
    star = "SELECT T1.origin || 'x' AS origin FROM (SELECT * FROM Flight) AS T1"
    gold = f"{star} WHERE origin = 'Chicago'"
    assert _classify(gold, f"{star} WHERE origin = 'Boston'") == "value"
    assert _classify(gold, f"{star} WHERE T1.origin || 'x' = 'Boston'") == "where"


@pytest.mark.phase(2)
def test_constants_alone_make_a_value_error() -> None:
    assert _classify(GOLD_14, "SELECT aid FROM Aircraft WHERE distance > 1500") == "value"
    assert _classify(GOLD_24, GOLD_24.replace("LIMIT 1", "LIMIT 3")) == "value"
    nested = "SELECT name FROM Aircraft WHERE distance > (SELECT avg(distance) FROM Aircraft {})"
    assert _classify(nested.format("WHERE aid > 2"), nested.format("WHERE aid > 3")) == "value"
    gold = "SELECT origin FROM Flight GROUP BY origin HAVING count(*) >= 2"
    assert _classify(gold, "SELECT origin FROM flight GROUP BY origin HAVING count(*) >= 3") == (
        "value"
    )


@pytest.mark.phase(2)
def test_a_nested_query_counts_in_the_expression_that_holds_it() -> None:
    # Anything wrong in the filter's subquery, or a filter done without one, is a where error.
    assert _classify(GOLD_20, "SELECT name FROM Aircraft ORDER BY distance DESC") == "where"
    pred = "SELECT name FROM Aircraft WHERE distance > (SELECT max(distance) FROM Aircraft)"
    assert _classify(GOLD_20, pred) == "where"
    pred = "SELECT name FROM Aircraft WHERE distance > (SELECT avg(aid) FROM Aircraft)"
    assert _classify(GOLD_20, pred) == "where"
    # Its tables still count among the query's own.
    pred = "SELECT name FROM Aircraft WHERE distance > (SELECT avg(distance) FROM Flight)"
    assert _classify(GOLD_20, pred) == "wrong_tables"
    # A derived table and each side of a set operation are SELECTs in their own right.
    gold = "SELECT count(*) FROM (SELECT origin FROM Flight GROUP BY origin)"
    pred = "SELECT count(*) FROM (SELECT DISTINCT origin FROM Flight)"
    assert _classify(gold, pred) == "projection"
    halves = GOLD_25.split(" INTERSECT ")
    pred = f"{halves[0]} INTERSECT {halves[1].replace('SELECT T1.name', 'SELECT T1.eid', 1)}"
    assert _classify(GOLD_25, pred) == "projection"


@pytest.mark.phase(2)
def test_a_query_sqlglot_cannot_read_falls_back_to_its_error_kind() -> None:
    # SQLite runs 60 nested parentheses; sqlglot runs out of stack on them.
    deep = "SELECT " + "(" * 60 + "aid" + ")" * 60 + " FROM Aircraft"
    assert _classify(GOLD_14, deep) == "other"
    timeout = ExecResult(ok=False, error="timed out after 2000 ms", error_kind="timeout")
    assert _classify(GOLD_14, deep, timeout) == "other"
    syntax = ExecResult(ok=False, error='near ")": syntax error', error_kind="syntax")
    assert _classify(GOLD_14, deep + ")", syntax) == "syntax"
    # Two statements, or a statement that is no query, are only ever denied.
    denied = ExecResult(ok=False, error=MULTIPLE_STATEMENTS, error_kind="denied")
    assert _classify(GOLD_14, f"{GOLD_14}; SELECT 1", denied) == "other"
    assert _classify(GOLD_14, "DELETE FROM Aircraft", denied) == "other"
    # Nor can a gold query sqlglot can't read be compared with.
    assert _classify(deep, GOLD_14) == "other"
