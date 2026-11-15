"""One ``Score`` per generation, from its query's result and its gold query's (stage 8).

``valid`` says the predicted query ran without error, and ``ex_fast`` that its rows match
gold's under ``forge.sql.compare``. ``ex_official`` is left to the official evaluator,
which the eval runner calls in one batch at the end, except for a query that timed out
in the sandbox: that one is no match without the evaluator, which would otherwise give
it up to its item timeout again (ADR-0008). ``error_class`` is ``None`` for a match
and ``UNCLASSIFIED`` for any other result until the error taxonomy (FG-204) names the
cause.
"""

from __future__ import annotations

from forge.core.types import Example, ExecResult, Generation, Score
from forge.sql.compare import order_matters, results_match

# What error_class holds for a wrong prediction until FG-204 classifies it.
UNCLASSIFIED = "unclassified"
NO_SQL = "the reply has no SQL query"


def no_sql() -> ExecResult:
    """The result given to a reply with no SQL in it, which has nothing to run."""
    return ExecResult(ok=False, error=NO_SQL, error_kind="empty_sql")


def score_generation(
    example: Example, generation: Generation, pred: ExecResult, gold: ExecResult
) -> Score:
    """Score one generation: ``pred`` is its query's result, ``gold`` its example's."""
    if generation.example_id != example.id:
        raise ValueError(
            f"a generation for {generation.example_id} can't be scored against {example.id}"
        )
    ex_fast = results_match(gold, pred, order_matters=order_matters(example.gold_sql))
    return Score(
        example_id=example.id,
        sample_idx=generation.sample_idx,
        hardness=example.hardness,
        valid=pred.ok,
        ex_fast=ex_fast,
        ex_official=False if pred.error_kind == "timeout" else None,
        error_class=None if ex_fast else UNCLASSIFIED,
    )
