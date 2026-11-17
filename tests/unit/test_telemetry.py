"""GRPO telemetry (FG-506): the alert rules on synthetic metric streams, the rollouts
the reward functions record, and the telemetry a step's line holds."""

from __future__ import annotations

import asyncio
import functools
import math
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from forge.eval.run_files import RunCorrupt
from forge.train.config import RewardsConfig
from forge.train.rewards import Rewards
from forge.train.telemetry import (
    LENGTH_FACTOR,
    RATE_SOURCES,
    VALID_DROP,
    ZERO_STD_LIMIT,
    ZERO_STD_STEPS,
    AlertMonitor,
    RolloutRecorder,
    jsonl_lines,
    merged_steps,
    percentile,
    read_file_lines,
    step_metrics,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "telemetry" / "grpo-metrics.jsonl"


def _stream(monitor: AlertMonitor, key: str, values: Sequence[float], start: int = 1) -> list[Any]:
    """Feed ``values`` of ``key`` in, one step each from ``start``; the events, in order."""
    events = []
    for step, value in enumerate(values, start=start):
        events += monitor.observe(step, {key: value})
    return events


@pytest.mark.phase(5)
def test_alerts_trigger() -> None:
    # Zero-variance groups: above 0.8 at every step for 20 steps fires, at the 20th; a
    # step at exactly 0.8 is not above it, and starts the count again.
    monitor = AlertMonitor()
    assert _stream(monitor, "frac_reward_zero_std", [0.9] * 10 + [ZERO_STD_LIMIT]) == []
    assert _stream(monitor, "frac_reward_zero_std", [0.95] * (ZERO_STD_STEPS - 1), 12) == []
    (fired,) = monitor.observe(31, {"frac_reward_zero_std": 1.0})
    assert (fired.step, fired.rule, fired.state) == (31, "zero_variance", "fired")
    assert "above 80% at every step since step 12: no learning signal" in fired.message
    assert monitor.observe(32, {"frac_reward_zero_std": 1.0}) == []  # once, while it holds
    assert monitor.active == {"zero_variance": fired}
    (cleared,) = monitor.observe(33, {"frac_reward_zero_std": 0.5})
    assert (cleared.rule, cleared.state) == ("zero_variance", "cleared") and not monitor.active

    # Completion length: over twice the first logged step's mean fires; twice is not over.
    monitor = AlertMonitor()
    lengths = [80.0, 120.0, LENGTH_FACTOR * 80.0, 161.0, 150.0]
    events = _stream(monitor, "completions/mean_length", lengths)
    assert [(e.step, e.rule, e.state) for e in events] == [
        (4, "length_growth", "fired"),
        (5, "length_growth", "cleared"),
    ]
    assert events[0].message == "mean completion length 161 tokens, over 2 times step 1's 80"

    # Valid rate: more than 20 points below its peak so far fires; 20 points is not more.
    monitor = AlertMonitor()
    rates = [0.5, 0.9, 0.9 - VALID_DROP, 0.69, 0.95, 0.74]
    events = _stream(monitor, "sql/valid_rate", rates)
    assert [(e.step, e.rule, e.state) for e in events] == [
        (4, "valid_drop", "fired"),
        (5, "valid_drop", "cleared"),
        (6, "valid_drop", "fired"),
    ]
    assert events[0].message == "valid-SQL rate 69%, 21.0 points below its peak of 90% at step 2"
    assert "peak of 95% at step 5" in events[2].message

    # All three from one stream of whole lines, as a run logs them; a step that lacks a
    # metric or holds no number for it, as TRL's null, leaves its rule as it was.
    lines = []
    for step in range(1, 26):
        line: dict[str, Any] = {
            "step": step,
            "frac_reward_zero_std": 0.9 if step >= 3 else 0.2,
            "completions/mean_length": 40.0 if step < 10 else 90.0,
            "sql/valid_rate": 0.8 if step < 15 else 0.55,
        }
        if step == 7:
            line["frac_reward_zero_std"] = None
        lines.append(line)
    events = AlertMonitor().replay(lines)
    assert [(e.step, e.rule, e.state) for e in events] == [
        (10, "length_growth", "fired"),
        (15, "valid_drop", "fired"),
        (22, "zero_variance", "fired"),
    ]
    # Up to a step, as a resumed run replays its metrics.
    assert [e.step for e in AlertMonitor().replay(lines, up_to=14)] == [10]
    # A healthy run sets none off.
    assert AlertMonitor().replay(read_file_lines(FIXTURE)) == []


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


@pytest.mark.phase(5)
def test_metrics_lines_merge_by_step() -> None:
    data = b'{"step": 1, "a": 1}\n{"step": 2, "a": 2}\n\n{"step": 2, "b": 3}\n{"step": 1, "a": 9}\n'
    # A last line with no newline yet is being written: the next read takes it.
    lines = jsonl_lines(data + b'{"step": 3, "a"', "metrics.jsonl")
    assert len(lines) == 4
    assert merged_steps([*lines, {"a": 0}, {"step": True, "a": 0}]) == [
        (1, {"step": 1, "a": 9}),
        (2, {"step": 2, "a": 2, "b": 3}),
    ]
    with pytest.raises(RunCorrupt, match=r"line 2 of metrics\.jsonl is not JSON"):
        jsonl_lines(b'{"step": 1}\n{"step": \n', "metrics.jsonl")
    with pytest.raises(RunCorrupt, match=r"line 1 of metrics\.jsonl is not a JSON object"):
        jsonl_lines(b"[1]\n", "metrics.jsonl")
    assert math.isnan(jsonl_lines(b'{"step": 1, "loss": NaN}\n', "m")[0]["loss"])
