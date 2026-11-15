"""Execution-guided retry (FG-206): one more turn, with the error, for a query that fails
to run, and never for one that ran."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from forge.core.types import ExecResult, Generation, GenParams, Prompt, Score
from forge.eval.retry import (
    RETRY_MESSAGE,
    ask_again,
    retried,
    retry_message,
    retry_params,
    without_retry,
)
from forge.infer.base import FakeBackend, sample_seed


@pytest.mark.phase(2)
def test_retry_message_names_the_timeout_not_the_kill() -> None:
    killed = ExecResult(
        ok=False, error="killed after 2253 ms (timeout 2000 ms)", error_kind="timeout"
    )
    stopped = ExecResult(ok=False, error="timed out after 2000 ms", error_kind="timeout")
    # The kill's measured time varies, and a resumed run must ask the same question.
    assert (
        retry_message(killed)
        == retry_message(stopped)
        == RETRY_MESSAGE.format(error="timed out after 2000 ms")
    )
    with pytest.raises(ValueError, match="no error to show"):
        retry_message(ExecResult(ok=False, error_kind="other"))

    score = Score(example_id="e1", hardness="easy", valid=True, ex_fast=True)
    # A first query that timed out is no official match without the evaluator.
    assert (retried(score, killed).attempts, retried(score, killed).first_try_ex) == (2, False)
    syntax = ExecResult(ok=False, error='near "x": syntax error', error_kind="syntax")
    assert retried(score, syntax).first_try_ex is None
    assert without_retry(score) is score


@pytest.mark.phase(2)
def test_each_retry_is_drawn_with_a_seed_of_its_own() -> None:
    calls: list[tuple[list[str], GenParams]] = []

    class Recording(FakeBackend):
        def generate(self, prompts: Sequence[Prompt], params: GenParams) -> list[Generation]:
            calls.append(([p.example_id for p in prompts], params))
            return super().generate(prompts, params)

    def generation(example_id: str, sample_idx: int) -> Generation:
        return Generation(
            example_id=example_id,
            model_ref="fake",
            sample_idx=sample_idx,
            text="SELEC 1",
            sql="SELEC 1",
            tokens_in=1,
            tokens_out=2,
            latency_ms=0.0,
        )

    failed = ExecResult(ok=False, error='near "SELEC": syntax error', error_kind="syntax")
    asks = [
        (
            Prompt(example_id=example_id, messages=(), prompt_version="p1"),
            generation(example_id, i),
            failed,
        )
        for example_id, i in (("q1", 1), ("q1", 0), ("q2", 1))
    ]
    params = GenParams(temperature=0.7, n=2, seed=7)
    backend = Recording(lambda prompt, params, sample_idx: "```sql\nSELECT 1\n```")
    seconds = ask_again(backend, asks, params)
    assert [(g.example_id, g.sample_idx, g.sql) for g in seconds] == [
        ("q1", 1, "SELECT 1"),
        ("q1", 0, "SELECT 1"),
        ("q2", 1, "SELECT 1"),
    ]
    # One call per sample index, one reply a prompt, so a question's retries never share
    # a row seed, and none shares the first replies'.
    assert [(ids, p.n) for ids, p in calls] == [(["q1"], 1), (["q1", "q2"], 1)]
    assert [p for _, p in calls] == [retry_params(params, 0), retry_params(params, 1)]
    row_seeds = {sample_seed(p.seed, "q1", 0) for _, p in calls}
    first_seeds = {sample_seed(params.seed, "q1", i) for i in range(params.n)}
    assert len(row_seeds) == 2 and not row_seeds & first_seeds
    assert retry_params(params, 0).model_dump() == {
        **params.model_dump(),
        "n": 1,
        "seed": sample_seed(7, "retry", 0),
    }

    class Silent:
        name = "silent"

        def generate(self, prompts: Sequence[Prompt], params: GenParams) -> list[Generation]:
            return []

    with pytest.raises(RuntimeError, match="returned 0 replies to 1 retry prompts"):
        ask_again(Silent(), asks[:1], params)
