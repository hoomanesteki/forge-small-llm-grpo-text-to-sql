"""GRPO telemetry (FG-506): the rollouts the reward functions record, and the telemetry a
step's line holds."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest

from forge.train.config import RewardsConfig
from forge.train.rewards import Rewards
from forge.train.telemetry import (
    RATE_SOURCES,
    RolloutRecorder,
    percentile,
    step_metrics,
)


class FakeRewards:
    """Two reward functions with TRL's signature, the second a method of a class."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def match(
        self,
        prompts: Sequence[Any],
        completions: Sequence[Any],
        completion_ids: Any = None,
        **kw: Any,
    ) -> list[float]:
        self.calls.append({"completion_ids": completion_ids, **kw})
        return [1.0 if "right" in str(c) else 0.0 for c in completions]

    def penalty(
        self,
        prompts: Sequence[Any],
        completions: Sequence[Any],
        completion_ids: Any = None,
        **kw: Any,
    ) -> list[float]:
        return [-0.5 if "bad" in str(c) else 0.0 for c in completions]


def _batch(
    functions: Sequence[Any], texts: Sequence[str], ids: Sequence[str], *, step: int = 0
) -> list[list[float]]:
    """Score ``texts`` with each of ``functions`` in turn, as TRL calls them: by keyword,
    the same completions to each, and the trainer's own keywords beside the columns."""
    prompts = [[{"role": "user", "content": f"question {i}"}] for i in ids]
    completions = [[{"role": "assistant", "content": text}] for text in texts]
    columns = {
        "example_id": list(ids),
        "db_id": ["dorm_1"] * len(ids),
        "hardness": ["easy"] * len(ids),
        "gold_sql": ["SELECT 1"] * len(ids),
        "stage": [0] * len(ids),  # a column a sample doesn't keep
        "trainer_state": SimpleNamespace(global_step=step),
        "log_extra": lambda column, values: None,
        "log_metric": lambda name, value: None,
    }
    return [
        function(
            prompts=prompts,
            completions=completions,
            completion_ids=[[7] * len(text.split()) for text in texts],
            **columns,
        )
        for function in functions
    ]


@pytest.mark.phase(5)
def test_recorder_records_what_the_rewards_scored() -> None:
    fake = FakeRewards()
    recorder = RolloutRecorder(group_size=2)
    match, penalty = recorder.wrap([fake.match, functools.partial(fake.penalty)], [1.0, 2.0])
    # TRL logs each by its name, which the wrapped functions keep.
    assert (match.__name__, penalty.__name__) == ("match", "penalty")
    assert recorder.names == ("match", "penalty")

    scores = _batch([match, penalty], ["right", "bad one", "wrong", "right bad"], list("aabb"))
    assert scores == [[1.0, 0.0, 0.0, 1.0], [0.0, -0.5, 0.0, -0.5]]  # as they scored
    assert fake.calls[0]["example_id"] == list("aabb")  # the keywords went through
    first, second = recorder.take()
    assert [r.columns["example_id"] for r in first + second] == list("aabb")
    assert "stage" not in first[0].columns and first[0].columns["db_id"] == "dorm_1"
    assert [r.tokens for r in first + second] == [1, 2, 1, 2]
    assert [r.rewards for r in first] == [
        {"match": 1.0, "penalty": 0.0},
        {"match": 0.0, "penalty": -0.5},
    ]
    assert [r.reward for r in first + second] == [1.0, -1.0, 0.0, 0.0]  # 1 x match + 2 x penalty
    assert first[0].completion == [{"role": "assistant", "content": "right"}]
    assert recorder.take() == []  # taken once

    # Two batches before a take come out in order; one function's call alone is no batch.
    _batch([match, penalty], ["right", "right"], ["c", "c"])
    _batch([match, penalty], ["x", "y"], ["d", "d"])
    _batch([match], ["z", "z"], ["e", "e"])
    assert [g[0].columns["example_id"] for g in recorder.take()] == ["c", "d"]

    with pytest.raises(ValueError, match="isn't whole groups of 2"):
        _batch([match, penalty], ["a", "b", "c"], ["f", "f", "f"])
    with pytest.raises(ValueError, match="one recorder a run"):
        recorder.wrap([fake.match])
    with pytest.raises(ValueError, match="two share one"):
        RolloutRecorder(2).wrap([fake.match, fake.match])
    with pytest.raises(ValueError, match="1 weights came with 2"):
        RolloutRecorder(2).wrap([fake.match, fake.penalty], [1.0])

    async def scored(prompts: Any, completions: Any, **kwargs: Any) -> list[float]:
        await asyncio.sleep(0)
        return [0.0] * len(completions)

    with pytest.raises(TypeError, match="scored is async"):
        RolloutRecorder(2).wrap([scored])


@pytest.mark.phase(5)
def test_step_lines_hold_the_telemetry() -> None:
    recorder = RolloutRecorder(group_size=2)
    fake = FakeRewards()
    functions = recorder.wrap([fake.match, fake.penalty])
    _batch(functions, ["right", "right", "right one", "a b c d e"], ["a", "a", "b", "b"])
    groups = recorder.take()
    logs = {
        "rewards/execution_reward/mean": 0.25,
        "rewards/validity_reward/mean": 0.75,
        "frac_reward_zero_std": 0.5,
        "learning_rate": 5e-6,
    }
    # The rates from TRL's means, and the 95th percentile of the lengths 1, 1, 2 and 5.
    assert step_metrics(logs, groups) == {
        "sql/valid_rate": 0.75,
        "sql/execution_match_rate": 0.25,
        "completions/p95_length": pytest.approx(4.55),
    }
    # The zero-variance share from the rewards when the trainer logged none: group a's
    # two completions scored the same, group b's didn't.
    assert step_metrics({"frac_reward_zero_std": None}, groups) == {
        "completions/p95_length": pytest.approx(4.55),
        "frac_reward_zero_std": 0.5,
    }
    assert step_metrics({}, []) == {}
    assert percentile([3.0], 0.95) == 3.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5


@pytest.mark.phase(5)
def test_telemetry_names_match_their_sources() -> None:
    # The rates read the means TRL logs under the reward functions' own names.
    names = {reward.__name__ for reward in Rewards({}, None, RewardsConfig()).functions}  # type: ignore[arg-type]
    for source in RATE_SOURCES.values():
        assert source.split("/")[1] in names
