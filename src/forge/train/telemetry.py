"""What a GRPO run's metrics say about whether it is learning, while it trains and after
(FG-506): the telemetry each step's line holds, the alerts it sets off, and the plots of
``forge runs plot``.

``forge.train.callbacks.TelemetryCallback`` writes each step's line in ``metrics.jsonl``.
TRL 1.18 logs these at every step, with ``logging_steps=1``:

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

``AlertMonitor`` reads the lines step by step, and its rules say when a run has stopped
learning, each as an event when it starts holding and another when it stops:

- ``zero_variance``: ``frac_reward_zero_std`` above ``ZERO_STD_LIMIT`` (0.8) at every
  step for ``ZERO_STD_STEPS`` (20) steps;
- ``length_growth``: ``completions/mean_length`` above ``LENGTH_FACTOR`` (2) times the
  run's first logged step's, which stands for step 0, the length before any update;
- ``valid_drop``: ``sql/valid_rate`` more than ``VALID_DROP`` (20 points) below its peak.

An event is a line of the run's ``alerts.jsonl`` (``forge.ops.hub.ALERTS_FILE``), which
the manifest's outputs name ``alerts``: ``RunManifest`` has no field for them, and the
contract stays as it is. The events follow from the metrics alone, so the training
callback and anyone reading a run's metrics find the same ones.

Nothing here imports transformers or torch, so ``forge runs plot`` runs without the cpu
extra.
"""

from __future__ import annotations

import functools
import inspect
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from forge.eval.run_files import RunCorrupt
from forge.ops.hub import METRICS_FILE, RunStore

if TYPE_CHECKING:
    from forge.train.backends import RewardFunc

REWARD_KEY = "reward"  # in every training step's log, and in no other
ZERO_STD_KEY = "frac_reward_zero_std"
MEAN_LENGTH_KEY = "completions/mean_length"
P95_LENGTH_KEY = "completions/p95_length"
VALID_KEY = "sql/valid_rate"
MATCH_KEY = "sql/execution_match_rate"
# The rewards of forge.train.rewards whose means are the SQL rates; a test keeps the
# names in step with its functions.
RATE_SOURCES = {
    VALID_KEY: "rewards/validity_reward/mean",
    MATCH_KEY: "rewards/execution_reward/mean",
}
KL_KEY = "kl"
LR_KEY = "learning_rate"
# The in-training eval's fast EX (forge.eval.callbacks.EX_KEY), spelled out here since that
# module imports transformers; a test keeps the two in step.
VAL_EX_KEY = "val/ex_fast"

ZERO_STD_LIMIT = 0.8
ZERO_STD_STEPS = 20
LENGTH_FACTOR = 2.0
VALID_DROP = 0.20
# A drop of exactly 20 points is no alert, though 0.9 - 0.7 comes out a hair above 0.2.
_DROP_SLACK = 1e-9
# TRL counts a group flat when its rewards' spread is 0 under torch.isclose.
_FLAT = 1e-8
# The dataset columns a sample keeps beside each prompt, when the batch has them.
SAMPLE_COLUMNS = ("example_id", "db_id", "hardness", "gold_sql")

AlertRule = Literal["zero_variance", "length_growth", "valid_drop"]
AlertState = Literal["fired", "cleared"]


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


@dataclass(frozen=True)
class AlertEvent:
    """An alert rule starting or stopping to hold, at ``step``."""

    step: int
    rule: AlertRule
    state: AlertState
    message: str

    def line(self) -> dict[str, Any]:
        """The event as a line of ``alerts.jsonl``."""
        return asdict(self)

    def describe(self) -> str:
        what = "alert" if self.state == "fired" else "cleared"
        return f"step {self.step}: {what} {self.rule}: {self.message}"


class AlertMonitor:
    """The alert rules of the module docstring, fed one step's metrics at a time."""

    def __init__(self) -> None:
        self.active: dict[AlertRule, AlertEvent] = {}  # fired, and not cleared since
        self._flat_since: int | None = None  # the first step of the run above the limit
        self._first: tuple[int, float] | None = None  # the first logged mean length
        self._peak: tuple[int, float] | None = None  # the highest valid rate so far

    def observe(self, step: int, values: Mapping[str, Any]) -> list[AlertEvent]:
        """The events the metrics ``values`` of ``step`` set off; a rule whose metric
        they lack or hold no number for stays as it was."""
        events: list[AlertEvent] = []
        share = finite(values.get(ZERO_STD_KEY))
        if share is not None:
            above = share > ZERO_STD_LIMIT
            if not above:
                self._flat_since = None
            elif self._flat_since is None:
                self._flat_since = step
            since = self._flat_since
            held = since is not None and step - since + 1 >= ZERO_STD_STEPS
            message = (
                f"{share:.0%} of groups had zero reward variance, above {ZERO_STD_LIMIT:.0%} "
                f"at every step since step {since}: no learning signal"
                if held
                else f"{share:.0%} of groups had zero reward variance"
            )
            events += self._turn("zero_variance", held, step, message)
        length = finite(values.get(MEAN_LENGTH_KEY))
        if length is not None:
            if self._first is None:
                self._first = (step, length)
            first_step, first = self._first
            longer = length > LENGTH_FACTOR * first
            message = (
                f"mean completion length {length:.0f} tokens, "
                f"{'over' if longer else 'within'} {LENGTH_FACTOR:g} times step {first_step}'s "
                f"{first:.0f}"
            )
            events += self._turn("length_growth", longer, step, message)
        rate = finite(values.get(VALID_KEY))
        if rate is not None:
            if self._peak is None or rate > self._peak[1]:
                self._peak = (step, rate)
            peak_step, peak = self._peak
            drop = peak - rate
            dropped = drop > VALID_DROP + _DROP_SLACK
            message = (
                f"valid-SQL rate {rate:.0%}, {100 * drop:.1f} points below its peak of "
                f"{peak:.0%} at step {peak_step}"
            )
            events += self._turn("valid_drop", dropped, step, message)
        return events

    def replay(
        self, lines: Iterable[Mapping[str, Any]], *, up_to: int | None = None
    ) -> list[AlertEvent]:
        """Feed a run's metrics lines in, step by step as ``merged_steps`` gives them, up
        to step ``up_to`` when given; returns the events they set off."""
        events: list[AlertEvent] = []
        for step, values in merged_steps(lines):
            if up_to is not None and step > up_to:
                break
            events += self.observe(step, values)
        return events

    def _turn(self, rule: AlertRule, on: bool, step: int, message: str) -> list[AlertEvent]:
        if on == (rule in self.active):
            return []
        event = AlertEvent(step, rule, "fired" if on else "cleared", message)
        if on:
            self.active[rule] = event
        else:
            del self.active[rule]
        return [event]


def jsonl_lines(data: bytes, where: str) -> list[dict[str, Any]]:
    """The JSON objects of a JSONL file's bytes, ``where`` naming it in errors. A last
    line with no newline yet is being written, and is left for the next read."""
    complete, _, _ = data.decode("utf-8").rpartition("\n")
    lines: list[dict[str, Any]] = []
    for number, text in enumerate(complete.split("\n") if complete else [], start=1):
        if not text.strip():
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RunCorrupt(f"line {number} of {where} is not JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise RunCorrupt(f"line {number} of {where} is not a JSON object")
        lines.append(value)
    return lines


def read_file_lines(path: Path) -> list[dict[str, Any]]:
    """``jsonl_lines`` of the file at ``path``; none when there is no file."""
    return jsonl_lines(path.read_bytes(), str(path)) if path.is_file() else []


def read_metrics(store: RunStore, run_id: str) -> list[dict[str, Any]] | None:
    """The lines of the run's ``metrics.jsonl`` in ``store``, or None when it has none."""
    data = store.read_file(run_id, METRICS_FILE)
    return None if data is None else jsonl_lines(data, f"{METRICS_FILE} of run {run_id}")


def merged_steps(lines: Iterable[Mapping[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    """Each step's metrics, in step order: the lines of a step merged, a later line's
    values over an earlier's. So the in-training eval's line joins its step's, and the
    steps a session that failed logged past its last checkpoint give way to those the
    resumed session logged again. A line without a whole-number step is left out."""
    steps: dict[int, dict[str, Any]] = {}
    for line in lines:
        step = line.get("step")
        if isinstance(step, int) and not isinstance(step, bool):
            steps.setdefault(step, {}).update(line)
    return sorted(steps.items())


# The eval report chart's surface and ink (forge.eval.report), and three series hues
# taken in this order, so a chart's series always get the same ones.
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_GRID = "#e1e0d9"
_BASELINE = "#c3c2b7"
_SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
_INCHES = (6.4, 3.6)
_PANEL_INCHES = (4.8, 2.6)
_HEADING_INCHES = 0.75  # above the panels, for the title and the line under it
_DPI = 100
_LINE_PT = 1.44  # 2 px at 100 dpi
_MARKED = 20  # a series of this many points or fewer shows each one


@dataclass(frozen=True)
class _Chart:
    stem: str  # the file is <stem>.png
    title: str
    series: tuple[tuple[str, str], ...]  # each line's key and label
    percent: bool = False  # shares, drawn from 0 to 100%
    from_zero: bool = False  # counts, drawn from 0


CHARTS = (
    _Chart(
        "sql-rates",
        "SQL rates",
        (
            (MATCH_KEY, "execution match, rollouts"),
            (VALID_KEY, "valid SQL, rollouts"),
            (VAL_EX_KEY, "fast EX on val, greedy"),
        ),
        percent=True,
    ),
    _Chart(
        "zero-variance",
        "Groups with zero reward variance",
        ((ZERO_STD_KEY, "share of groups"),),
        percent=True,
    ),
    _Chart(
        "completion-length",
        "Completion length, tokens",
        ((MEAN_LENGTH_KEY, "mean"), (P95_LENGTH_KEY, "95th percentile")),
        from_zero=True,
    ),
    _Chart("kl", "KL to the reference model", ((KL_KEY, "kl"),)),
    _Chart("learning-rate", "Learning rate", ((LR_KEY, "learning rate"),)),
    _Chart("loss", "Training loss", (("loss", "loss"),)),
)


def _points(
    steps: Sequence[tuple[int, Mapping[str, Any]]], key: str
) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for step, values in steps:
        value = finite(values.get(key))
        if value is not None:
            xs.append(step)
            ys.append(value)
    return xs, ys


def _limit(
    chart: _Chart, steps: Sequence[tuple[int, Mapping[str, Any]]]
) -> tuple[float, str] | None:
    """The alert threshold a chart shows, and its label."""
    if chart.stem == "zero-variance":
        return ZERO_STD_LIMIT, f"alert above {ZERO_STD_LIMIT:.0%} for {ZERO_STD_STEPS} steps"
    if chart.stem == "completion-length":
        xs, ys = _points(steps, MEAN_LENGTH_KEY)
        if ys:
            return LENGTH_FACTOR * ys[0], f"alert above {LENGTH_FACTOR:g}x step {xs[0]}'s mean"
    return None


def _style(axes: Any, *, percent: bool) -> None:
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    axes.set_facecolor(_SURFACE)
    axes.grid(axis="y", color=_GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(_BASELINE)
    axes.tick_params(length=0, labelcolor=_INK_SECONDARY, labelsize=9)
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes.set_xlabel("step", color=_INK_SECONDARY, fontsize=9)
    if percent:
        axes.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))


def _line(axes: Any, xs: list[int], ys: list[float], color: str, label: str) -> None:
    marked = len(xs) <= _MARKED
    axes.plot(
        xs,
        ys,
        color=color,
        linewidth=_LINE_PT,
        label=label,
        marker="o" if marked else None,
        markersize=6,
        markeredgecolor=_SURFACE,
        markeredgewidth=1.4,
        zorder=3,
    )


def _heading(figure: Any, title: str, subtitle: str, *, left: float, tall: float) -> None:
    """A chart's title and the line under it, a fixed distance from the figure's top."""
    figure.text(left, 1 - 0.12 / tall, title, color=_INK, fontsize=11, va="top")
    figure.text(left, 1 - 0.36 / tall, subtitle, color=_INK_SECONDARY, fontsize=8, va="top")


def _save(figure: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="png", dpi=_DPI, facecolor=_SURFACE)
    return path


def _line_chart(
    chart: _Chart, steps: Sequence[tuple[int, Mapping[str, Any]]], subtitle: str, path: Path
) -> Path | None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    drawn = [(key, label, *_points(steps, key)) for key, label in chart.series]
    drawn = [entry for entry in drawn if entry[2]]
    if not drawn:
        return None
    figure = Figure(figsize=_INCHES, dpi=_DPI, facecolor=_SURFACE)
    FigureCanvasAgg(figure)
    axes = figure.add_axes((0.11, 0.16, 0.85, 0.62))
    _style(axes, percent=chart.percent)
    colors = dict(zip((key for key, _ in chart.series), _SERIES, strict=False))
    for key, label, xs, ys in drawn:
        _line(axes, xs, ys, colors[key], label)
    limit = _limit(chart, steps)
    if limit is not None:
        value, label = limit
        axes.axhline(value, color=_INK_SECONDARY, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        axes.annotate(
            label,
            (1, value),
            xycoords=("axes fraction", "data"),
            xytext=(0, 4),
            textcoords="offset points",
            ha="right",
            va="bottom",
            color=_INK_SECONDARY,
            fontsize=8,
        )
    if chart.percent:
        top = max(max(ys) for *_, ys in drawn)
        axes.set_ylim(0, max(1.0, top) * 1.05)
    elif chart.from_zero:
        axes.set_ylim(bottom=0)
    if len(drawn) > 1:
        legend = axes.legend(
            loc="lower left",
            bbox_to_anchor=(0, 1.0),
            ncols=len(drawn),
            frameon=False,
            fontsize=8,
            labelcolor=_INK_SECONDARY,
            handlelength=1.6,
        )
        legend.set_zorder(4)
    _heading(figure, chart.title, subtitle, left=0.11, tall=_INCHES[1])
    return _save(figure, path)


def _rewards_chart(
    steps: Sequence[tuple[int, Mapping[str, Any]]], subtitle: str, path: Path
) -> Path | None:
    """Small multiples, one panel for each reward's mean and one for their weighted sum,
    each on its own scale, with a band one standard deviation either side."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    means = [
        key
        for key in dict.fromkeys(k for _, values in steps for k in values)
        if key.startswith("rewards/") and key.endswith("/mean")
    ]
    panels = [
        (
            key,
            key.removeprefix("rewards/").removesuffix("/mean"),
            key.removesuffix("/mean") + "/std",
        )
        for key in means
    ]
    panels.append((REWARD_KEY, "reward, weighted sum", "reward_std"))
    panels = [panel for panel in panels if _points(steps, panel[0])[0]]
    if not panels:
        return None
    columns = 2 if len(panels) > 1 else 1
    rows = math.ceil(len(panels) / columns)
    width, height = _PANEL_INCHES
    tall = height * rows + _HEADING_INCHES
    figure = Figure(figsize=(width * columns, tall), dpi=_DPI, facecolor=_SURFACE)
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(
        rows,
        columns,
        left=0.08,
        right=0.97,
        bottom=0.45 / tall,
        top=1 - _HEADING_INCHES / tall,
        hspace=0.6,
        wspace=0.22,
    )
    for index, (key, name, spread) in enumerate(panels):
        axes = figure.add_subplot(grid[index // columns, index % columns])
        _style(axes, percent=False)
        xs, ys = _points(steps, key)
        band = dict(zip(*_points(steps, spread), strict=True))
        if all(x in band for x in xs):
            axes.fill_between(
                xs,
                [y - band[x] for x, y in zip(xs, ys, strict=True)],
                [y + band[x] for x, y in zip(xs, ys, strict=True)],
                color=_SERIES[0],
                alpha=0.16,
                linewidth=0,
                zorder=2,
            )
        _line(axes, xs, ys, _SERIES[0], name)
        axes.set_title(name, loc="left", color=_INK, fontsize=9)
    subtitle = f"{subtitle}; the band is one standard deviation either side"
    _heading(figure, "Mean reward per step", subtitle, left=0.08, tall=tall)
    return _save(figure, path)


def plot_metrics(lines: Iterable[Mapping[str, Any]], out_dir: Path, *, title: str) -> list[Path]:
    """Draw a run's metrics lines as PNGs in ``out_dir``: ``rewards.png`` and one for
    each of ``CHARTS`` whose metrics the lines hold, with the alert thresholds; returns
    the files written. ``title`` names the run under each chart's heading. Raises
    ``ValueError`` when the lines hold none of them."""
    import matplotlib.style

    steps = merged_steps(lines)
    written: list[Path] = []
    # matplotlib's defaults rather than any matplotlibrc's, as the eval report's chart.
    with matplotlib.style.context("default"):
        rewards = _rewards_chart(steps, title, out_dir / "rewards.png")
        if rewards is not None:
            written.append(rewards)
        for chart in CHARTS:
            path = _line_chart(chart, steps, title, out_dir / f"{chart.stem}.png")
            if path is not None:
                written.append(path)
    if not written:
        raise ValueError(f"{title}: the metrics hold nothing to plot")
    return written
