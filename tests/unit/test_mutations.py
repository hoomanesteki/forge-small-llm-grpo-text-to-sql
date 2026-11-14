"""Mutations of gold queries for the agreement check: what each change does, and seeding."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from forge.core.types import Example
from forge.eval.mutations import CHANGES, UNCHANGED_SHARE, Change, make_mutations, mutate

COLUMNS = {
    "head": ("head_ID", "name", "born_state", "age"),
    "department": ("Department_ID", "Name", "Budget_in_Billions"),
}


def _all(
    sql: str, change: Change, columns: dict[str, tuple[str, ...]] = COLUMNS
) -> set[str | None]:
    """Every result the change gives over a range of seeds."""
    return {mutate(sql, change, random.Random(seed), columns) for seed in range(40)}


@pytest.mark.phase(1)
@pytest.mark.parametrize(
    ("change", "gold", "expected"),
    [
        ("drop_where", "SELECT name FROM head WHERE age > 56", "SELECT name FROM head"),
        ("change_aggregate", "SELECT max(age) FROM head", "SELECT MIN(age) FROM head"),
        ("change_aggregate", "SELECT min(age) FROM head", "SELECT MAX(age) FROM head"),
        ("change_aggregate", "SELECT avg(age) FROM head", "SELECT SUM(age) FROM head"),
        ("change_aggregate", "SELECT sum(age) FROM head", "SELECT AVG(age) FROM head"),
        (
            "change_aggregate",
            "SELECT count(DISTINCT born_state) FROM head",
            "SELECT SUM(DISTINCT born_state) FROM head",
        ),
        (
            "flip_order",
            "SELECT name FROM head ORDER BY age",
            "SELECT name FROM head ORDER BY age DESC",
        ),
        (
            "flip_order",
            "SELECT name FROM head ORDER BY age DESC LIMIT 3",
            "SELECT name FROM head ORDER BY age ASC LIMIT 3",
        ),
        (
            "change_literal",
            "SELECT name FROM head WHERE age > 56",
            "SELECT name FROM head WHERE age > 57",
        ),
        (
            "change_literal",
            "SELECT name FROM head WHERE age > 1.5",
            "SELECT name FROM head WHERE age > 2.5",
        ),
        (
            "change_literal",
            "SELECT name FROM head WHERE name = 'Ab'",
            "SELECT name FROM head WHERE name = 'aB'",
        ),
        (
            "change_literal",
            "SELECT name FROM head WHERE name LIKE '%'",
            "SELECT name FROM head WHERE name LIKE '%x'",
        ),
    ],
)
def test_single_candidate_changes(change: Change, gold: str, expected: str) -> None:
    assert _all(gold, change) == {expected}


@pytest.mark.phase(1)
def test_changes_pick_among_candidates() -> None:
    # A parenthesized group is one condition, and the other side of the AND stays.
    assert _all(
        "SELECT name FROM head WHERE age > 56 AND (born_state = 'Alabama' OR age < 10)",
        "drop_where",
    ) == {
        "SELECT name FROM head WHERE age > 56",
        "SELECT name FROM head WHERE (born_state = 'Alabama' OR age < 10)",
    }
    # COUNT(*) has nothing SUM can take, so it sums one of the table's columns.
    assert _all("SELECT count(*) FROM head", "change_aggregate") == {
        f"SELECT SUM({c}) FROM head" for c in COLUMNS["head"]
    }
    # Any other column of the table, never the same one.
    assert _all("SELECT name FROM head", "swap_column") == {
        "SELECT head_ID FROM head",
        "SELECT born_state FROM head",
        "SELECT age FROM head",
    }
    # The same column under its alias doesn't count as another column.
    assert _all("SELECT name FROM head AS T1", "swap_column") == {
        f"SELECT T1.{c} FROM head AS T1" for c in ("head_ID", "born_state", "age")
    }
    # With several tables a swapped-in column is qualified, so it can't be ambiguous.
    swapped = _all(
        "SELECT T1.name FROM head AS T1 JOIN department AS T2 ON T1.head_ID = T2.Department_ID",
        "swap_column",
    )
    joined = " FROM head AS T1 JOIN department AS T2 ON T1.head_ID = T2.Department_ID"
    assert swapped == {f"SELECT {c}{joined}" for c in ("T1.head_ID", "T1.born_state", "T1.age")} | {
        f"SELECT T2.{c}{joined}" for c in COLUMNS["department"]
    }
    # A subquery's select list draws on the subquery's own tables.
    assert _all(
        "SELECT name FROM head WHERE head_ID IN (SELECT Department_ID FROM department)",
        "swap_column",
    ) == {
        "SELECT name FROM head WHERE head_ID IN (SELECT Name FROM department)",
        "SELECT name FROM head WHERE head_ID IN (SELECT Budget_in_Billions FROM department)",
        *(
            f"SELECT {c} FROM head WHERE head_ID IN (SELECT Department_ID FROM department)"
            for c in ("head_ID", "born_state", "age")
        ),
    }


@pytest.mark.phase(1)
@pytest.mark.parametrize(
    ("change", "gold"),
    [
        ("drop_where", "SELECT name FROM head"),
        ("flip_order", "SELECT name FROM head"),
        ("change_aggregate", "SELECT name FROM head"),
        ("change_literal", "SELECT name FROM head"),
        # Spider writes many strings in double quotes, which sqlglot reads as names.
        ("change_literal", 'SELECT name FROM head WHERE born_state = "Alabama"'),
        ("drop_where", "SELECT FROM WHERE"),
    ],
)
def test_change_without_a_candidate_is_none(change: Change, gold: str) -> None:
    assert _all(gold, change) == {None}


@pytest.mark.phase(1)
def test_column_changes_need_the_schema() -> None:
    assert _all("SELECT name FROM head", "swap_column", {}) == {None}
    assert _all("SELECT count(*) FROM head", "change_aggregate", {}) == {None}


def _examples(n: int) -> list[Example]:
    return [
        Example(
            id=f"spider-train-{i:05d}",
            db_id="department_management",
            question="q",
            gold_sql=f"SELECT name, count(*) FROM head WHERE age > {i} GROUP BY name ORDER BY age",
            hardness="medium",
            split="val",
        )
        for i in range(n)
    ]


@pytest.mark.phase(1)
def test_mutations_are_deterministic_per_seed() -> None:
    examples = _examples(50)
    columns = {"department_management": COLUMNS}

    first = make_mutations(examples, 1000, seed=7, columns=columns)
    assert make_mutations(examples, 1000, seed=7, columns=columns) == first
    assert make_mutations(examples, 1000, seed=8, columns=columns) != first
    # A shorter run is the start of a longer one.
    assert make_mutations(examples, 300, seed=7, columns=columns) == first[:300]

    # Every example comes up before any comes up twice.
    assert len({m.example_id for m in first[:50]}) == 50
    kinds = Counter(m.kind for m in first)
    assert set(kinds) == {"unchanged", *CHANGES}
    assert abs(kinds["unchanged"] / len(first) - UNCHANGED_SHARE) < 0.05
    for m in first:
        if m.kind == "unchanged":
            assert m.pred_sql == m.gold_sql
        else:
            assert m.pred_sql != m.gold_sql


@pytest.mark.phase(1)
def test_make_mutations_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="at least one example"):
        make_mutations([], 10, seed=0, columns={})
    with pytest.raises(ValueError, match="zero or more"):
        make_mutations(_examples(1), -1, seed=0, columns={})
    assert make_mutations(_examples(1), 0, seed=0, columns={}) == []
