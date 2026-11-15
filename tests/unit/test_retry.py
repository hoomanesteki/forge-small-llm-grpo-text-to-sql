"""Execution-guided retry (FG-206): one more turn, with the error, for a query that fails
to run, and never for one that ran.

The runs answer the fixture's val questions (``spider-train-00010`` to ``00029`` on
flight_1) with fake backends whose first replies fail in chosen ways.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from forge.core.types import Example, ExecResult, Generation, GenParams, Prompt, Score
from forge.data.splits import load_val
from forge.eval.config import EvalConfig
from forge.eval.retry import (
    RETRY_MESSAGE,
    ask_again,
    retried,
    retry_message,
    retry_params,
    retry_prompt,
    without_retry,
)
from forge.eval.run_files import read_run
from forge.eval.runner import BATCH_SIZE, EvalRun, run_eval
from forge.infer.base import FakeBackend, ReplyFn, sample_seed

NO_SQL_REPLY = "There are some aircraft."
# First replies whose queries fail to run, and the error each one is told.
FAILING = {
    "spider-train-00011": ("SELEC count(*) FROM Aircraft", 'near "SELEC": syntax error'),
    "spider-train-00012": ("SELECT name, range_km FROM Aircraft", "no such column: range_km"),
    "spider-train-00014": (None, "the reply has no SQL query"),
    "spider-train-00016": (
        "DROP TABLE Aircraft",
        "DROP is not a query; only SELECT and VALUES statements may run",
    ),
}


@pytest.fixture
def runs(spider_mini: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dev_log: Path) -> Path:
    """Runs and the gold cache under tmp_path, and a scratch dev log."""
    monkeypatch.setenv("FORGE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("FORGE_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "runs"


def _val() -> dict[str, Example]:
    return {example.id: example for example in sorted(load_val(), key=lambda e: e.id)}


def _block(sql: str | None) -> str:
    return NO_SQL_REPLY if sql is None else f"```sql\n{sql}\n```"


def _run(run_id: str, reply: ReplyFn, *, retry: int, batch_size: int = BATCH_SIZE) -> EvalRun:
    return run_eval(
        EvalConfig(model="gold", backend="fake", run_id=run_id, retry=retry),
        backend=FakeBackend(reply, model_ref="gold"),
        allow_dirty=True,
        workers=1,
        batch_size=batch_size,
    )


def _official(scores: tuple[Score, ...] | list[Score]) -> dict[str, bool | None]:
    return {score.example_id: score.ex_official for score in scores}


def _turn(prompt: Prompt) -> int:
    """1 for a prompt as built, 2 for a retry: two messages more."""
    return (len(prompt.messages) - 2) // 2 + 1


@pytest.mark.phase(2)
def test_retry_uses_error_message(runs: Path) -> None:
    val = _val()
    firsts: dict[str, Prompt] = {}
    retries: list[tuple[Prompt, GenParams, int]] = []

    def fixes_on_retry(prompt: Prompt, params: GenParams, sample_idx: int) -> str:
        if _turn(prompt) == 1:
            firsts[prompt.example_id] = prompt
            if prompt.example_id in FAILING:
                return _block(FAILING[prompt.example_id][0])
        else:
            retries.append((prompt, params, sample_idx))
        return _block(val[prompt.example_id].gold_sql)

    once = _run("once", fixes_on_retry, retry=0)
    assert not retries
    assert all(score.attempts == 1 and score.first_try_ex is None for score in once.scores)
    assert sum(score.ex_official is True for score in once.scores) == 16

    run = _run("retry", fixes_on_retry, retry=1)
    # One retry for each query that failed to run, holding the whole conversation: the
    # prompt as first asked, the first reply, then the error.
    assert [prompt.example_id for prompt, _, _ in retries] == sorted(FAILING)
    for prompt, params, sample_idx in retries:
        sql, error = FAILING[prompt.example_id]
        assert prompt.messages == (
            *firsts[prompt.example_id].messages,
            {"role": "assistant", "content": _block(sql)},
            {
                "role": "user",
                "content": f"That query failed with: {error}. Reply with a corrected query.",
            },
        )
        assert prompt.prompt_version == "p1"
        assert (params.n, sample_idx) == (1, 0)

    # The second answers are the ones scored, and they lift official EX from 16 of 20.
    assert len(run.scores) == 20
    assert all(score.ex_fast and score.ex_official for score in run.scores)
    for score in run.scores:
        retried_here = score.example_id in FAILING
        assert (score.attempts, score.first_try_ex) == ((2, False) if retried_here else (1, None))
    stored = read_run(run.paths.root)
    assert [g.sql for g in stored.generations.values()] == [e.gold_sql for e in val.values()]
    assert {first.example_id: first.sql for first in stored.first_tries.values()} == {
        example_id: sql for example_id, (sql, _) in FAILING.items()
    }
    assert run.manifest.outputs["first_tries"] == "eval/val-first-tries.jsonl"
    assert run.manifest.config["retry"] == 1
    # Without retry, the run scores as the run that never retried.
    assert _official([without_retry(score) for score in run.scores]) == _official(once.scores)


@pytest.mark.phase(2)
def test_retry_only_on_error(runs: Path) -> None:
    val = _val()
    first_replies = {
        # Ran, with wrong rows: never retried.
        "spider-train-00013": "SELECT 1",
        "spider-train-00017": "SELECT name, distance FROM Aircraft WHERE aid = 13",
        "spider-train-00024": "SELECT origin FROM Flight",
        # SQLite can't parse it, but official EX drops DISTINCT before it runs, which
        # makes it count(*): retried, and its first reply was right after all.
        "spider-train-00010": "SELECT count(DISTINCT *) FROM Aircraft",
        # Fails to run, and fails again on retry.
        "spider-train-00022": "SELECT name FROM Employees",
    }
    second_replies = {
        "spider-train-00010": "SELECT 1",
        "spider-train-00022": "SELECT name FROM Employes",
    }
    asked_again: list[str] = []

    def reply(prompt: Prompt, params: GenParams, sample_idx: int) -> str:
        if _turn(prompt) == 2:
            asked_again.append(prompt.example_id)
            return _block(second_replies[prompt.example_id])
        return _block(first_replies.get(prompt.example_id, val[prompt.example_id].gold_sql))

    run = _run("only-errors", reply, retry=1)
    scores = {score.example_id: score for score in run.scores}
    assert asked_again == ["spider-train-00010", "spider-train-00022"]
    stored = read_run(run.paths.root)
    for example_id in ("spider-train-00013", "spider-train-00017", "spider-train-00024"):
        score = scores[example_id]
        assert score.valid and not score.ex_fast and score.ex_official is False
        assert (score.attempts, score.first_try_ex) == (1, None)
        assert stored.generations[example_id, 0].sql == first_replies[example_id]
    assert all(scores[e].attempts == 1 for e in val if e not in second_replies)

    # Official EX judges the first reply too: it can be right where the executor failed,
    # and the second reply can lose it.
    lost = scores["spider-train-00010"]
    assert (lost.attempts, lost.first_try_ex, lost.valid, lost.ex_official) == (
        2,
        True,
        True,
        False,
    )
    again = scores["spider-train-00022"]
    assert (again.attempts, again.first_try_ex, again.valid) == (2, False, False)
    assert again.error_class == "schema_link"
    assert stored.generations["spider-train-00022", 0].sql == "SELECT name FROM Employes"
    assert without_retry(lost).ex_official is True and without_retry(again).ex_official is False

    # Nothing builds a retry for a query that ran.
    ran = ExecResult(ok=True, columns=["1"], rows=[(1,)])
    prompt = Prompt(example_id="e1", messages=(), prompt_version="p1")
    first = stored.generations["spider-train-00013", 0].model_copy(update={"example_id": "e1"})
    for build in (
        lambda: retry_message(ran),
        lambda: retry_prompt(prompt, first, ran),
        lambda: retried(scores["spider-train-00013"], ran),
    ):
        with pytest.raises(ValueError, match="a query that ran is never retried"):
            build()


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


@pytest.mark.phase(2)
def test_a_failed_retry_run_resumes_with_its_first_tries(runs: Path) -> None:
    val = _val()
    ids = list(val)

    def failing_first(breaks: str | None) -> ReplyFn:
        def reply(prompt: Prompt, params: GenParams, sample_idx: int) -> str:
            if prompt.example_id == breaks:
                raise RuntimeError("the backend fell over")
            if _turn(prompt) == 1 and prompt.example_id in FAILING:
                return _block(FAILING[prompt.example_id][0])
            return _block(val[prompt.example_id].gold_sql)

        return reply

    # Batches of 4: the first three hold every failing question, and the fourth breaks.
    with pytest.raises(RuntimeError, match="the backend fell over"):
        _run("flaky", failing_first(ids[12]), retry=1, batch_size=4)
    stopped = runs / "flaky" / "eval"
    stored_first = [
        Generation.model_validate_json(line).example_id
        for line in (stopped / "val-first-tries.jsonl").read_text().splitlines()
    ]
    assert stored_first == sorted(FAILING)
    unjudged = [
        Score.model_validate_json(line)
        for line in (stopped / "val-scores.jsonl").read_text().splitlines()
    ]
    # The official batch never ran, so the first tries still wait for their verdicts.
    assert [s.example_id for s in unjudged if s.attempts == 2] == sorted(FAILING)
    assert all(s.first_try_ex is None and s.ex_official is None for s in unjudged)

    run = _run("flaky", failing_first(None), retry=1, batch_size=4)
    assert run.resumed == 12 and run.manifest.status == "finished"
    assert all(score.ex_official for score in run.scores)
    assert {s.example_id: s.first_try_ex for s in run.scores if s.attempts == 2} == dict.fromkeys(
        sorted(FAILING), False
    )
    assert list(read_run(run.paths.root).first_tries) == [(e, 0) for e in sorted(FAILING)]
