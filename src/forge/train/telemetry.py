"""What a GRPO run's metrics say about whether it is learning, while it trains and after
(FG-506): the telemetry each step's line holds.

Each step's line in a GRPO run's ``metrics.jsonl`` holds the trainer's logs. TRL 1.18
logs these at every step, with ``logging_steps=1``:

- ``rewards/<name>/mean`` and ``rewards/<name>/std`` of each reward function, and
  ``reward`` and ``reward_std`` of their weighted sum;
- ``frac_reward_zero_std``: the share of groups whose completions all scored the same, so
  their advantages are 0 and they teach nothing;
- ``completions/mean_length``, with the minimum, maximum and clipped share;
- ``kl`` when beta isn't 0 (ADR-0011's beta is TRL's 0, so it logs none), and
  transformers' ``learning_rate``, ``loss`` and ``grad_norm``.

``step_metrics`` adds what TRL doesn't log, from the rollouts ``RolloutRecorder`` saw:

- ``sql/valid_rate`` and ``sql/execution_match_rate``: the means of
  ``forge.train.rewards``' validity and execution rewards, which are 0 or 1, so the share
  of completions whose query ran without error and set off no guard, and the share whose
  rows matched gold's;
- ``completions/p95_length``: the 95th percentile of the step's completion lengths, in
  tokens, from the ids TRL passes the rewards;
- ``frac_reward_zero_std`` from the recorded rewards, when the trainer logged none.

``RolloutRecorder`` wraps the reward functions a run gives the trainer, and records each
batch's prompts, completions, token counts and every function's reward as TRL scored
them: TRL keeps its own table of them private, and writes it only to trackers. A wrapped
function keeps its name, which is what TRL logs it by.

Nothing here imports transformers or torch, so reading a run's metrics needs no cpu
extra.
"""

from __future__ import annotations

import functools
import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from forge.train.backends import RewardFunc

ZERO_STD_KEY = "frac_reward_zero_std"
P95_LENGTH_KEY = "completions/p95_length"
VALID_KEY = "sql/valid_rate"
MATCH_KEY = "sql/execution_match_rate"
# The rewards of forge.train.rewards whose means are the SQL rates; a test keeps the
# names in step with its functions.
RATE_SOURCES = {
    VALID_KEY: "rewards/validity_reward/mean",
    MATCH_KEY: "rewards/execution_reward/mean",
}

# TRL counts a group flat when its rewards' spread is 0 under torch.isclose.
_FLAT = 1e-8
# The dataset columns a sample keeps beside each prompt, when the batch has them.
SAMPLE_COLUMNS = ("example_id", "db_id", "hardness", "gold_sql")


@dataclass(frozen=True)
class Rollout:
    """One completion as the rewards scored it."""

    prompt: Any  # as TRL passed it: the messages, or the text
    completion: Any  # likewise
    tokens: int | None  # its length in tokens, when TRL passed its ids
    rewards: dict[str, float]  # each reward function's, by the name TRL logs it under
    reward: float  # their weighted sum, which the group's advantages come from
    columns: dict[str, Any]  # the dataset's SAMPLE_COLUMNS it came with


RolloutGroup = tuple[Rollout, ...]  # one prompt's completions, in TRL's order


@dataclass
class _Batch:
    """A batch whose rewards are still coming in, one function at a time."""

    prompts: Sequence[Any]
    completions: Sequence[Any]
    completion_ids: Sequence[Sequence[int]] | None
    columns: Mapping[str, Any]
    rewards: dict[int, list[float]] = field(default_factory=dict)  # by function index


def reward_name(function: Callable[..., Any]) -> str:
    """The name TRL 1.18 logs a reward function under (``trl.trainer.utils``'s
    ``get_callable_name``): a partial's function's, else its own, else its class's."""
    while isinstance(function, functools.partial):
        function = function.func
    return str(getattr(function, "__name__", type(function).__name__))


class RolloutRecorder:
    """The side channel from a run's reward functions to its telemetry.

    ``wrap`` gives the trainer functions that score as the given ones do and record each
    batch, and ``take`` hands over the prompts' groups of the batches scored since the
    last take. TRL calls every function on one batch before the next batch, passing the
    same completions to each, which is how the recorder tells the batches apart.
    ``group_size`` is the run's ``num_generations``, the completions of one prompt.
    """

    def __init__(self, group_size: int) -> None:
        if group_size < 1:
            raise ValueError(f"a group holds 1 completion or more, not {group_size}")
        self.group_size = group_size
        self.names: tuple[str, ...] = ()
        self.weights: tuple[float, ...] = ()
        self._batch: _Batch | None = None
        self._done: list[RolloutGroup] = []

    def wrap(
        self, functions: Sequence[RewardFunc], weights: Sequence[float] | None = None
    ) -> list[RewardFunc]:
        """``functions``, recorded; ``weights`` are TRL's ``reward_weights``, all 1 when
        None, which the recorded sums take."""
        names = tuple(reward_name(function) for function in functions)
        chosen = tuple(1.0 for _ in names) if weights is None else tuple(map(float, weights))
        if not names:
            raise ValueError("there are no reward functions to record")
        if len(set(names)) < len(names):
            raise ValueError(
                f"TRL logs each reward function by its name, and two share one: {names}"
            )
        if len(chosen) != len(names):
            raise ValueError(f"{len(chosen)} weights came with {len(names)} reward functions")
        if self.names:
            raise ValueError(f"this recorder records {self.names} already: one recorder a run")
        for function in functions:
            if inspect.iscoroutinefunction(function):
                raise TypeError(f"{reward_name(function)} is async, and the recorder waits on none")
        self.names, self.weights = names, chosen
        return [self._recorded(index, function) for index, function in enumerate(functions)]

    def take(self) -> list[RolloutGroup]:
        """The groups of every batch scored since the last take, oldest first."""
        groups, self._done = self._done, []
        return groups

    def _recorded(self, index: int, function: RewardFunc) -> RewardFunc:
        @functools.wraps(function)
        def recorded(
            prompts: Sequence[Any],
            completions: Sequence[Any],
            completion_ids: Sequence[Sequence[int]] | None = None,
            **kwargs: Any,
        ) -> list[float]:
            rewards = function(
                prompts=prompts, completions=completions, completion_ids=completion_ids, **kwargs
            )
            # One reward a completion, or TRL refuses the batch itself.
            if len(rewards) == len(completions):
                self._record(index, prompts, completions, completion_ids, kwargs, rewards)
            return rewards

        # The name TRL logs it under, which wraps leaves as it was for a partial.
        recorded.__name__ = recorded.__qualname__ = self.names[index]
        return recorded

    def _record(
        self,
        index: int,
        prompts: Sequence[Any],
        completions: Sequence[Any],
        completion_ids: Sequence[Sequence[int]] | None,
        columns: Mapping[str, Any],
        rewards: Sequence[float | None],
    ) -> None:
        batch = self._batch
        if batch is None or batch.completions is not completions:
            batch = self._batch = _Batch(prompts, completions, completion_ids, columns)
        # TRL reads a None as NaN, a completion the function doesn't score.
        batch.rewards[index] = [math.nan if value is None else float(value) for value in rewards]
        if len(batch.rewards) == len(self.names):
            self._done.extend(self._groups(batch))
            self._batch = None

    def _groups(self, batch: _Batch) -> list[RolloutGroup]:
        size, group = len(batch.completions), self.group_size
        if size % group:
            raise ValueError(
                f"a batch of {size} completions isn't whole groups of {group}: the recorder "
                "needs each prompt's completions in one batch, as on one GPU"
            )
        columns = {
            name: values
            for name, values in batch.columns.items()
            if name in SAMPLE_COLUMNS
            and isinstance(values, Sequence)
            and not isinstance(values, str)
            and len(values) == size
        }
        rollouts = []
        for i in range(size):
            scores = {name: batch.rewards[k][i] for k, name in enumerate(self.names)}
            # nansum, as TRL sums them.
            total = math.fsum(
                weight * scores[name]
                for weight, name in zip(self.weights, self.names, strict=True)
                if not math.isnan(scores[name])
            )
            ids = None if batch.completion_ids is None else batch.completion_ids[i]
            rollouts.append(
                Rollout(
                    prompt=batch.prompts[i],
                    completion=batch.completions[i],
                    tokens=None if ids is None else len(ids),
                    rewards=scores,
                    reward=total,
                    columns={name: values[i] for name, values in columns.items()},
                )
            )
        return [tuple(rollouts[i : i + group]) for i in range(0, size, group)]


def sample_line(step: int, group: RolloutGroup) -> dict[str, Any]:
    """One prompt of a sample dump: its step, columns and prompt, and each completion
    with its length and rewards."""
    first = group[0]
    return {
        "step": step,
        **first.columns,
        "prompt": first.prompt,
        "completions": [
            {
                "completion": rollout.completion,
                "tokens": rollout.tokens,
                "rewards": rollout.rewards,
                "reward": rollout.reward,
            }
            for rollout in group
        ],
    }


def zero_std_share(groups: Sequence[RolloutGroup]) -> float:
    """The share of ``groups`` whose completions' rewards are all the same."""
    if not groups:
        raise ValueError("no groups to count")
    flat = 0
    for group in groups:
        rewards = [rollout.reward for rollout in group]
        flat += max(rewards) - min(rewards) <= _FLAT
    return flat / len(groups)


def percentile(values: Sequence[float], share: float) -> float:
    """The ``share`` quantile of ``values``, interpolated between the two nearest as
    numpy's default does."""
    if not values:
        raise ValueError("no values to take a quantile of")
    ordered = sorted(values)
    at = share * (len(ordered) - 1)
    low = math.floor(at)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (at - low)


def step_metrics(logs: Mapping[str, Any], groups: Sequence[RolloutGroup]) -> dict[str, float]:
    """What a step's line holds beyond the trainer's ``logs``, from the ``groups`` the
    step's rollouts made, as the module docstring lists it."""
    found: dict[str, float] = {}
    for key, source in RATE_SOURCES.items():
        value = finite(logs.get(source))
        if value is not None:
            found[key] = value
    lengths = [r.tokens for group in groups for r in group if r.tokens is not None]
    if lengths:
        found[P95_LENGTH_KEY] = percentile(lengths, 0.95)
    if finite(logs.get(ZERO_STD_KEY)) is None and groups:
        found[ZERO_STD_KEY] = zero_std_share(groups)
    return found


def finite(value: object) -> float | None:
    """``value`` as a float when it is a finite number, else None: the trainer writes
    null for a metric with no value, and an overflow as NaN or Infinity."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None
