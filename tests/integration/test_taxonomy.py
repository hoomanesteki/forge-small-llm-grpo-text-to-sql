"""The error taxonomy on the fixture's real Spider queries and the agreement check's
mutations of them, against schemas read from the fixture databases."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pytest

from forge.core.types import Example, ExecResult, Generation
from forge.data.splits import load_train, load_val
from forge.eval.agreement import table_columns
from forge.eval.mutations import Columns, MutationKind, make_mutations
from forge.eval.taxonomy import ErrorClass, classify
from forge.sql.executor import SqliteExecutor

# A query that ran; the taxonomy never looks at the rows.
RAN = ExecResult(ok=True, columns=["n"], rows=[])

# The classes each change can land in, the one it is named for first. A change inside a
# query nested in a condition counts in that condition, and dropping the condition that
# held a table's only use changes the tables read.
LANDS_IN: dict[MutationKind, tuple[ErrorClass, ...]] = {
    "unchanged": ("other",),
    "drop_where": ("where", "wrong_tables"),
    "swap_column": ("projection", "aggregation", "where"),
    "change_aggregate": ("aggregation", "group_order", "where"),
    "flip_order": ("group_order", "where"),
    "change_literal": ("value",),
}

Questions = tuple[list[Example], dict[str, Columns]]


@pytest.fixture
def questions(spider_mini: Path) -> Questions:
    """Every fixture question outside dev, with the tables and columns of each database."""
    examples = [*load_train(), *load_val()]
    with SqliteExecutor() as executor:
        schemas = {db: table_columns(executor, db) for db in sorted({e.db_id for e in examples})}
    return examples, schemas


def _as_prediction(example: Example, sql: str, schemas: dict[str, Columns]) -> ErrorClass:
    generation = Generation(
        example_id=example.id,
        model_ref="test",
        text=f"```sql\n{sql}\n```",
        sql=sql,
        tokens_in=1,
        tokens_out=1,
        latency_ms=0.0,
    )
    return classify(example, generation, RAN, columns=schemas[example.db_id])


@pytest.mark.phase(2)
def test_a_gold_query_differs_from_itself_in_nothing(questions: Questions) -> None:
    examples, schemas = questions
    assert len(examples) == 60
    classes = {e.id: _as_prediction(e, e.gold_sql, schemas) for e in examples}
    assert classes == {e.id: "other" for e in examples}


@pytest.mark.phase(2)
def test_mutations_land_in_the_class_of_their_change(questions: Questions) -> None:
    examples, schemas = questions
    by_id = {e.id: e for e in examples}
    landed: defaultdict[MutationKind, Counter[ErrorClass]] = defaultdict(Counter)
    # sqlglot writes each changed query out again, in its own spelling of Spider's SQL.
    for mutation in make_mutations(examples, 300, seed=0, columns=schemas):
        example = by_id[mutation.example_id]
        landed[mutation.kind][_as_prediction(example, mutation.pred_sql, schemas)] += 1
    assert set(landed) == set(LANDS_IN)
    for kind, classes in landed.items():
        assert set(classes) <= set(LANDS_IN[kind]), (kind, classes)
        assert classes.most_common(1)[0][0] == LANDS_IN[kind][0], (kind, classes)
