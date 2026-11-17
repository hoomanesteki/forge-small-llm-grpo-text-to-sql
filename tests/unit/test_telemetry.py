"""GRPO telemetry (FG-506): the alert rules on synthetic metric streams, the plots of a
fixture run's metrics, the rollouts the reward functions record, the telemetry callback
driven by hand, event by event, and ``forge runs plot`` on a local store."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.image
import pytest
from transformers import TrainerControl, TrainerState
from typer.testing import CliRunner

from forge import cli
from forge.core.types import RunManifest
from forge.eval.callbacks import EX_KEY
from forge.eval.run_files import RunCorrupt
from forge.ops.hub import LocalRunStore, StoreUnavailable
from forge.train.callbacks import TelemetryCallback, training_callbacks
from forge.train.config import OpsConfig, RewardsConfig
from forge.train.rewards import Rewards
from forge.train.telemetry import (
    LENGTH_FACTOR,
    RATE_SOURCES,
    VAL_EX_KEY,
    VALID_DROP,
    ZERO_STD_LIMIT,
    ZERO_STD_STEPS,
    AlertMonitor,
    RolloutRecorder,
    jsonl_lines,
    merged_steps,
    percentile,
    plot_metrics,
    read_file_lines,
    step_metrics,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "telemetry" / "grpo-metrics.jsonl"
RUN = "2026-11-20-qwen3-1p7b-grpo-abc1234"
PNG = b"\x89PNG\r\n\x1a\n"


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


@pytest.mark.phase(5)
def test_plots_render(tmp_path: Path) -> None:
    lines = read_file_lines(FIXTURE)
    written = plot_metrics(lines, tmp_path / "plots", title="run fixture, grpo, finished")
    names = [path.name for path in written]
    # The fixture run had beta 0, so TRL logged no KL, and no chart of it is drawn.
    assert names == [
        "rewards.png",
        "sql-rates.png",
        "zero-variance.png",
        "completion-length.png",
        "learning-rate.png",
        "loss.png",
    ]
    sizes = {}
    for path in written:
        assert path.read_bytes().startswith(PNG)
        image = matplotlib.image.imread(path)
        sizes[path.name] = image.shape[:2]
        # Something was drawn: not every pixel is the chart's surface.
        assert image.reshape(-1, image.shape[-1]).std(axis=0).max() > 0.05, path.name
    assert sizes["sql-rates.png"] == (360, 640)
    # Five panels, the four rewards and their sum, two to a row.
    assert sizes["rewards.png"] == (855, 960)

    # Only what is there: a run that logged nothing but its learning rate gets that alone.
    sparse = [{"step": 1, "learning_rate": 1e-5}, {"step": 2, "learning_rate": 5e-6}]
    (only,) = plot_metrics(sparse, tmp_path / "sparse", title="run sparse")
    assert only.name == "learning-rate.png"
    with pytest.raises(ValueError, match="run empty: the metrics hold nothing to plot"):
        plot_metrics([{"step": 1, "epoch": 1.0}], tmp_path / "empty", title="run empty")


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
    assert VAL_EX_KEY == EX_KEY


def _log(callback: TelemetryCallback, state: TrainerState, step: int, **logs: Any) -> None:
    """The trainer's log of ``step``, which reads none of its arguments but the logs."""
    state.global_step = step
    callback.on_log(SimpleNamespace(), state, TrainerControl(), logs=logs)


@pytest.mark.phase(5)
def test_telemetry_callback_writes_lines_alerts_and_samples(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalRunStore(tmp_path / "store")
    run_dir = tmp_path / "run"
    seconds = iter(range(1000))
    recorder = RolloutRecorder(group_size=2)
    fake = FakeRewards()
    functions = recorder.wrap([fake.match, fake.penalty])
    callback = TelemetryCallback(
        run_dir / "metrics.jsonl",
        recorder,
        store=store,
        run_id=RUN,
        now=lambda: datetime(2026, 11, 20, 9, 0, next(seconds) % 60, tzinfo=UTC),
        samples_every=2,
        sample_prompts=3,
    )
    args: Any = SimpleNamespace(output_dir=str(tmp_path / "out"))
    state, control = TrainerState(max_steps=40), TrainerControl()
    callback.on_train_begin(args, state, control)
    assert (run_dir / "alerts.jsonl").read_bytes() == b""  # no alerts is an empty file

    for step in range(1, 25):
        # Two prompts a step, each with two completions that score the same.
        _batch(functions, ["right"] * 4, [f"q{step}", f"q{step}", f"r{step}", f"r{step}"])
        state.global_step = step
        callback.on_step_end(args, state, control)
        with caplog.at_level(logging.INFO, logger="forge.train.callbacks"):
            _log(
                callback,
                state,
                step,
                reward=1.0,
                frac_reward_zero_std=None,  # left to the rollouts: every group is flat
                **{"rewards/validity_reward/mean": 1.0, "completions/mean_length": 1.0},
            )
        if step == 4:
            callback.on_save(args, state, control)
    callback.write(24, {"val/ex_fast": 0.4})  # the in-training eval's line, as it is
    _log(callback, state, 24, train_runtime=3.0)  # the run's summary, as it is

    lines = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    first = lines[0]
    assert (first["step"], first["frac_reward_zero_std"], first["sql/valid_rate"]) == (1, 1.0, 1.0)
    assert first["completions/p95_length"] == 1.0
    assert lines[-2] == {"step": 24, "time": lines[-2]["time"], "val/ex_fast": 0.4}
    assert lines[-1]["train_runtime"] == 3.0 and "sql/valid_rate" not in lines[-1]

    # Every group flat from step 1, so the alert at step 20: in the log and in alerts.jsonl.
    alerts = [json.loads(line) for line in (run_dir / "alerts.jsonl").read_text().splitlines()]
    assert [(a["step"], a["rule"], a["state"]) for a in alerts] == [(20, "zero_variance", "fired")]
    assert f"telemetry: run {RUN}, step 20: alert zero_variance: 100% of groups" in caplog.text

    # A dump every 2 steps, of the latest 3 prompts with both their completions each.
    samples = run_dir / "samples"
    assert sorted(p.name for p in samples.iterdir())[:2] == [
        "step-000002.jsonl",
        "step-000004.jsonl",
    ]
    dump = [json.loads(line) for line in (samples / "step-000004.jsonl").read_text().splitlines()]
    assert [(d["step"], d["example_id"]) for d in dump] == [(3, "r3"), (4, "q4"), (4, "r4")]
    assert dump[0]["db_id"] == "dorm_1" and dump[0]["gold_sql"] == "SELECT 1"
    assert dump[0]["prompt"] == [{"role": "user", "content": "question r3"}]
    assert [c["rewards"] for c in dump[0]["completions"]] == [{"match": 1.0, "penalty": 0.0}] * 2
    assert [c["reward"] for c in dump[0]["completions"]] == [1.0, 1.0]

    # The save at step 4 pushed what there was, and the end pushes the rest.
    stored = store.root / RUN
    assert sorted(p.name for p in (stored / "samples").iterdir()) == [
        "step-000002.jsonl",
        "step-000004.jsonl",
    ]

    def unavailable(*args: object, **kwargs: object) -> str:
        raise StoreUnavailable("uploading failed 4 times")

    monkeypatch.setattr(store, "put_file", unavailable)
    callback.on_train_end(args, state, control)
    assert len(callback.unpushed) == 10  # steps 6 to 24 wait for the next push
    monkeypatch.undo()
    callback.on_train_end(args, state, control)
    assert callback.unpushed == {}
    assert (stored / "alerts.jsonl").read_bytes() == (run_dir / "alerts.jsonl").read_bytes()
    assert len(list((stored / "samples").iterdir())) == 12

    # A resumed session replays the run's metrics up to its checkpoint's step: from step
    # 10 the alert hasn't fired, and alerts.jsonl says so until the steps after say more.
    again = TelemetryCallback(run_dir / "metrics.jsonl", RolloutRecorder(2))
    state.global_step = 10
    again.on_train_begin(args, state, control)
    assert (run_dir / "alerts.jsonl").read_bytes() == b""
    state.global_step = 21
    again.on_train_begin(args, state, control)
    assert again.monitor.active.keys() == {"zero_variance"}
    assert len((run_dir / "alerts.jsonl").read_text().splitlines()) == 1

    with pytest.raises(ValueError, match="every 0 of 12"):
        TelemetryCallback(run_dir / "metrics.jsonl", RolloutRecorder(2), samples_every=0)


@pytest.mark.phase(5)
def test_training_callbacks_record_telemetry_for_grpo(tmp_path: Path) -> None:
    store, ops = LocalRunStore(tmp_path / "store"), OpsConfig()
    path = tmp_path / "run" / "metrics.jsonl"
    plain = training_callbacks(ops, store, RUN, metrics_path=path, env={})
    assert not any(isinstance(c, TelemetryCallback) for c in plain)
    recorder = RolloutRecorder(6)
    grpo = training_callbacks(ops, store, RUN, metrics_path=path, env={}, rollouts=recorder)
    (telemetry,) = [c for c in grpo if isinstance(c, TelemetryCallback)]
    assert (telemetry.rollouts, telemetry.path, telemetry.store) == (recorder, path, store)
    assert (telemetry.samples_every, telemetry.sample_prompts) == (50, 12)


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


def _manifest(status: str = "running") -> RunManifest:
    return RunManifest(
        run_id=RUN,
        kind="grpo",
        git_sha="a" * 40,
        config={"grpo": {"max_steps": 600}},
        config_hash="1" * 64,
        data_manifest_hash="2" * 64,
        model_ref="Qwen/Qwen3-1.7B",
        platform="kaggle",
        gpu="Tesla T4",
        seed=42,
        started_at=datetime(2026, 11, 20, 9, 0, tzinfo=UTC),
        status=status,  # type: ignore[arg-type]
    )


@pytest.mark.phase(5)
def test_runs_plot_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_RUNS_DIR", str(tmp_path / "runs"))
    store = LocalRunStore(tmp_path / "runs")
    store.put_manifest(_manifest("finished"))
    runner = CliRunner()

    missing = runner.invoke(cli.app, ["runs", "plot", RUN, "--store", "local"])
    assert missing.exit_code == 1
    assert f"error: run {RUN} in {store} has no metrics.jsonl yet" in missing.output

    store.put_file(RUN, FIXTURE, "metrics.jsonl")
    plotted = runner.invoke(cli.app, ["runs", "plot", RUN, "--store", "local"])
    assert plotted.exit_code == 0, plotted.output
    plots = tmp_path / "runs" / RUN / "plots"
    assert plotted.output.splitlines()[0] == str(plots / "rewards.png")
    assert (plots / "zero-variance.png").read_bytes().startswith(PNG)
    out = tmp_path / "docs-runs"
    elsewhere = runner.invoke(cli.app, ["runs", "plot", RUN, "--store", "local", "--out", str(out)])
    assert elsewhere.exit_code == 0 and (out / "sql-rates.png").is_file()
    unknown = runner.invoke(cli.app, ["runs", "plot", "2026-11-20-none", "--store", "local"])
    assert unknown.exit_code == 1 and "error: no run 2026-11-20-none" in unknown.output
