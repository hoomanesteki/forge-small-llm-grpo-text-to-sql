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
    return classify(example, generation, result)


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
    # The order of ORDER BY keys does change the rows.
    gold = "SELECT name FROM Aircraft ORDER BY distance, name"
    assert _classify(gold, "SELECT name FROM aircraft ORDER BY name, distance") == "group_order"


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
