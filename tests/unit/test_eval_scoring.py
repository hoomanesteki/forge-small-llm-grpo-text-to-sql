"""One score per generation, from its query's result and its gold result."""

from __future__ import annotations

import pytest

from forge.core.types import Example, ExecResult, Generation, Score
from forge.eval.scoring import no_sql, score_generation

EXAMPLE = Example(
    id="spider-train-00001",
    db_id="concert_singer",
    question="How many singers are there?",
    gold_sql="SELECT count(*) FROM singer",
    hardness="easy",
    split="train",
)
GOLD = ExecResult(ok=True, columns=["count(*)"], rows=[(6,)])
# concert_singer's singer table, as table_columns reads it.
COLUMNS = {
    "singer": ("Singer_ID", "Name", "Country", "Song_Name", "Song_release_year", "Age", "Is_male")
}


def _generation(sql: str | None, sample_idx: int = 0, example_id: str = EXAMPLE.id) -> Generation:
    return Generation(
        example_id=example_id,
        model_ref="gold",
        sample_idx=sample_idx,
        text="reply",
        sql=sql,
        tokens_in=10,
        tokens_out=5,
        latency_ms=1.0,
    )


@pytest.mark.phase(2)
def test_score_reads_the_results() -> None:
    right = score_generation(
        EXAMPLE,
        _generation("SELECT count(*) FROM singer"),
        ExecResult(ok=True, columns=["n"], rows=[(6,)]),
        GOLD,
        columns=COLUMNS,
    )
    assert right == Score(
        example_id=EXAMPLE.id, hardness="easy", valid=True, ex_fast=True, error_class=None
    )
    wrong = score_generation(
        EXAMPLE,
        _generation("SELECT 1", sample_idx=2),
        ExecResult(ok=True, columns=["1"], rows=[(1,)]),
        GOLD,
        columns=COLUMNS,
    )
    assert (wrong.sample_idx, wrong.valid, wrong.ex_fast) == (2, True, False)
    # A wrong prediction gets its error class: this one reads no table at all.
    assert (wrong.ex_official, wrong.error_class) == (None, "wrong_tables")

    broken = ExecResult(ok=False, error="no such column: nope", error_kind="no_such_column")
    failed = score_generation(
        EXAMPLE, _generation("SELECT nope FROM singer"), broken, GOLD, columns=COLUMNS
    )
    assert (failed.valid, failed.ex_fast, failed.ex_official) == (False, False, None)
    assert failed.error_class == "schema_link"

    # No SQL in the reply: nothing ran, and the official evaluator gives its own verdict.
    assert no_sql().error_kind == "empty_sql" and not no_sql().ok
    silent = score_generation(EXAMPLE, _generation(None), no_sql(), GOLD, columns=COLUMNS)
    assert (silent.valid, silent.ex_official, silent.error_class) == (False, None, "no_sql")


@pytest.mark.phase(2)
def test_a_timed_out_query_is_no_official_match() -> None:
    slow = ExecResult(
        ok=False, error="killed after 2253 ms (timeout 2000 ms)", error_kind="timeout"
    )
    score = score_generation(EXAMPLE, _generation("SELECT 1"), slow, GOLD, columns=COLUMNS)
    assert (score.valid, score.ex_fast, score.ex_official) == (False, False, False)


@pytest.mark.phase(2)
def test_row_order_counts_only_under_an_outer_order_by() -> None:
    gold = ExecResult(ok=True, columns=["id"], rows=[(1,), (2,)])
    backwards = ExecResult(ok=True, columns=["id"], rows=[(2,), (1,)])
    ordered = EXAMPLE.model_copy(update={"gold_sql": "SELECT id FROM singer ORDER BY id"})
    unordered = EXAMPLE.model_copy(update={"gold_sql": "SELECT id FROM singer"})
    assert not score_generation(ordered, _generation("x"), backwards, gold, columns=COLUMNS).ex_fast
    assert score_generation(unordered, _generation("x"), backwards, gold, columns=COLUMNS).ex_fast


@pytest.mark.phase(2)
def test_a_generation_scores_only_against_its_own_example() -> None:
    other = _generation("SELECT 1", example_id="spider-train-00002")
    with pytest.raises(ValueError, match="for spider-train-00002 can't be scored against"):
        score_generation(EXAMPLE, other, GOLD, GOLD, columns=COLUMNS)
