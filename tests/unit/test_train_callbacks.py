"""The training callbacks driven by hand, event by event, as a trainer would drive them."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import transformers.integrations
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from forge.core.types import RunManifest
from forge.ops.hub import LocalRunStore, StoreUnavailable
from forge.ops.manifest import DirtyTreeError, git_sha
from forge.train.callbacks import (
    WANDB_KEY,
    CheckpointCallback,
    MetricsCallback,
    TimeGuardCallback,
    kept_steps,
    training_callbacks,
    wandb_callback,
)
from forge.train.config import OpsConfig

ROOT = Path(__file__).resolve().parents[2]
RUN = "2026-11-15-qwen3-4b-grpo-abc1234"


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _manifest() -> RunManifest:
    return RunManifest(
        run_id=RUN,
        kind="grpo",
        git_sha="a" * 40,
        config={},
        config_hash="1" * 64,
        data_manifest_hash="2" * 64,
        model_ref="Qwen/Qwen3-4B",
        platform="kaggle",
        gpu="Tesla T4",
        seed=0,
        # Two hours ago, so the guard's stop, at the real time now, comes after it.
        started_at=datetime.now(UTC) - timedelta(hours=2),
    )


def _steps(
    guard: TimeGuardCallback, clock: Clock, seconds: list[float], *, max_steps: int = 100
) -> tuple[TrainerState, TrainerControl]:
    """Train begins, then one step per entry of ``seconds``, each that long."""
    args: Any = SimpleNamespace(output_dir="unused")
    state, control = TrainerState(max_steps=max_steps), TrainerControl()
    guard.on_train_begin(args, state, control)
    for step_s in seconds:
        clock.now += step_s
        state.global_step += 1
        guard.on_step_end(args, state, control)
        if control.should_training_stop:
            break
    return state, control


@pytest.mark.phase(3)
def test_guard_without_a_store_saves_and_stops() -> None:
    clock = Clock()
    guard = TimeGuardCallback(10, margin_s=120, clock=clock)
    clock.now += 300  # loading the model, before training begins
    state, control = _steps(guard, clock, [60, 30, 30, 30, 30])
    # Of 600 s, 300 went to loading and 120 to steps of 60, 30 and 30 s. The 180 s left
    # are the longest step and the margin, so training stops after the third.
    assert guard.stopped_at == state.global_step == 3
    assert control.should_save and control.should_training_stop
    args: Any = SimpleNamespace(output_dir="unused")
    guard.on_save(args, state, control)
    assert guard.pushed is None


@pytest.mark.phase(3)
def test_guard_runs_nothing_but_its_save_after_the_stop() -> None:
    args: Any = SimpleNamespace(output_dir="unused")
    clock = Clock()
    guard = TimeGuardCallback(1, clock=clock)
    state, control = TrainerState(max_steps=100), TrainerControl()
    guard.on_train_begin(args, state, control)
    # An epoch that ends before the deadline is the trainer's to evaluate and save.
    control.should_evaluate = control.should_save = True
    guard.on_epoch_end(args, state, control)
    assert control.should_evaluate and control.should_save

    # The trainer's own callback asked to evaluate the step the guard stops after.
    control = TrainerControl(should_evaluate=True)
    clock.now += 45
    state.global_step += 1
    guard.on_step_end(args, state, control)
    assert guard.stopped_at == 1
    assert control.should_save and control.should_training_stop
    assert not control.should_evaluate

    # Something cleared the save before the trainer made it, and the epoch's end asks to
    # evaluate: the guard asks for its save there, and no evaluation.
    control.should_save = False
    control.should_evaluate = True
    guard.on_epoch_end(args, state, control)
    assert control.should_save and not control.should_evaluate
    guard.on_save(args, state, control)
    assert guard.saved and guard.pushed is None

    # Once the step is saved, the epoch's end saves it no second time.
    control.should_save = control.should_evaluate = True
    guard.on_epoch_end(args, state, control)
    assert not control.should_save and not control.should_evaluate


@pytest.mark.phase(3)
def test_guard_leaves_the_last_step_to_the_trainer() -> None:
    clock = Clock()
    guard = TimeGuardCallback(1, clock=clock)
    state, control = _steps(guard, clock, [40], max_steps=1)
    assert state.global_step == 1 and guard.stopped_at is None
    assert not control.should_save and not control.should_training_stop


@pytest.mark.phase(3)
def test_guard_pushes_its_own_checkpoint_once(tmp_path: Path) -> None:
    store = LocalRunStore(tmp_path / "store")
    store.put_manifest(_manifest())
    out = tmp_path / "out"
    for step in (2, 4):
        (out / f"checkpoint-{step}").mkdir(parents=True)
        (out / f"checkpoint-{step}" / "adapter_model.safetensors").write_text(f"step {step}")
    args: Any = SimpleNamespace(output_dir=str(out))
    clock = Clock()
    guard = TimeGuardCallback(1, store=store, run_id=RUN, clock=clock)
    state, control = _steps(guard, clock, [20, 20, 20])
    assert guard.stopped_at == 2

    # A save of another step, such as the trainer's own every N steps, isn't the guard's.
    guard.on_save(args, TrainerState(global_step=4), control)
    assert store.latest_checkpoint(RUN) is None
    assert store.get_manifest(RUN).status == "running"

    # An earlier session's checkpoints, of which retention keeps the latest beside best/.
    for name in ("step-000000", "step-000001", "best"):
        (store.root / RUN / "checkpoints" / name).mkdir(parents=True)
        (store.root / RUN / "checkpoints" / name / "adapter_model.safetensors").write_text(name)
    guard.on_save(args, state, control)
    assert store.latest_checkpoint(RUN) == "checkpoints/step-000002"
    kept = sorted(p.name for p in (store.root / RUN / "checkpoints").iterdir())
    assert kept == ["best", "step-000001", "step-000002"]
    pushed = Path(str(guard.pushed))
    assert (pushed / "adapter_model.safetensors").read_text() == "step 2"
    manifest = store.get_manifest(RUN)
    assert manifest.status == "stopped_time_guard" and manifest.ended_at is not None
    assert manifest.gpu_hours == pytest.approx(2, abs=0.1)  # a Kaggle run's wall-clock hours

    # The trainer saves the same step again at its epoch's end: nothing is pushed twice.
    (pushed / "adapter_model.safetensors").write_text("pushed once")
    guard.on_save(args, state, control)
    assert (pushed / "adapter_model.safetensors").read_text() == "pushed once"


@pytest.mark.phase(3)
def test_failed_push_says_where_the_checkpoint_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalRunStore(tmp_path / "store")
    store.put_manifest(_manifest())
    (tmp_path / "out" / "checkpoint-1").mkdir(parents=True)
    (tmp_path / "out" / "checkpoint-1" / "adapter_model.safetensors").write_text("step 1")

    def unavailable(*args: object, **kwargs: object) -> str:
        raise StoreUnavailable("uploading failed 4 times")

    monkeypatch.setattr(store, "put_folder", unavailable)
    clock = Clock()
    guard = TimeGuardCallback(1, store=store, run_id=RUN, clock=clock)
    state, control = _steps(guard, clock, [45])
    args: Any = SimpleNamespace(output_dir=str(tmp_path / "out"))
    with pytest.raises(OSError, match=r"the time guard saved .*checkpoint-1, but pushing it"):
        guard.on_save(args, state, control)
    # The manifest isn't marked stopped while its checkpoint isn't in the store.
    assert store.get_manifest(RUN).status == "running"

    # Once the push and the manifest are done, a failed prune only leaves old checkpoints.
    monkeypatch.undo()
    monkeypatch.setattr(store, "prune_checkpoints", unavailable)
    guard.on_save(args, state, control)
    assert store.latest_checkpoint(RUN) == "checkpoints/step-000001"
    assert store.get_manifest(RUN).status == "stopped_time_guard"


@pytest.mark.phase(3)
def test_store_and_run_id_go_together(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="store and run_id together"):
        TimeGuardCallback(1, store=LocalRunStore(tmp_path))
    with pytest.raises(ValueError, match="store and run_id together"):
        TimeGuardCallback(1, run_id=RUN)
    with pytest.raises(ValueError, match="positive number of minutes"):
        TimeGuardCallback(0)


class Ticker(TrainerCallback):
    """Moves the clock at each step's end, before the guard reads it."""

    def __init__(self, clock: Clock, step_s: float) -> None:
        self.clock = clock
        self.step_s = step_s

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        self.clock.now += self.step_s


class FakeWandb(TrainerCallback):
    pass


def _train(
    callbacks: Sequence[TrainerCallback], out: Path, steps: int, *, max_steps: int = 100
) -> TrainerState:
    """Steps as a trainer runs them: each step's end, then the save one asked for, which
    leaves checkpoint-<step> in the output folder."""
    args: Any = SimpleNamespace(output_dir=str(out))
    state, control = TrainerState(max_steps=max_steps), TrainerControl()
    for callback in callbacks:
        callback.on_train_begin(args, state, control)
    for _ in range(steps):
        state.global_step += 1
        for callback in callbacks:
            callback.on_step_end(args, state, control)
        if control.should_save:
            folder = out / f"checkpoint-{state.global_step}"
            folder.mkdir(parents=True)
            (folder / "adapter_model.safetensors").write_text(f"step {state.global_step}")
            for callback in callbacks:
                callback.on_save(args, state, control)
            control.should_save = False
        if control.should_training_stop:
            break
    return state


@pytest.mark.phase(4)
def test_checkpoints_every_n_steps_go_to_the_store(tmp_path: Path) -> None:
    store = LocalRunStore(tmp_path / "store")
    out = tmp_path / "out"
    checkpoints = CheckpointCallback(store, RUN, every=2)
    _train([checkpoints], out, 7)
    assert sorted(checkpoints.pushed) == [2, 4, 6]
    assert sorted(p.name for p in out.iterdir()) == ["checkpoint-2", "checkpoint-4", "checkpoint-6"]
    # Retention keeps the latest two in the store.
    kept = sorted(p.name for p in (store.root / RUN / "checkpoints").iterdir())
    assert kept == ["step-000004", "step-000006"]
    assert store.latest_checkpoint(RUN) == "checkpoints/step-000006"
    pushed = Path(checkpoints.pushed[6], "adapter_model.safetensors")
    assert pushed.read_text() == "step 6"
    with pytest.raises(ValueError, match="every 1 step or more, not every 0"):
        CheckpointCallback(store, RUN, every=0)


@pytest.mark.phase(4)
def test_checkpoints_at_shares_of_the_run_outlive_retention(tmp_path: Path) -> None:
    # The step each share keeps: the last it covers, never the run's end, which is
    # checkpoints/final.
    assert kept_steps([0.5], 486) == {0.5: 243}
    assert kept_steps([0.25, 0.5], 3) == {0.25: 1, 0.5: 1}
    assert kept_steps([0.5], 1) == {}
    with pytest.raises(ValueError, match="a run of 0 steps keeps no checkpoint"):
        kept_steps([0.5], 0)

    # Half way through an 18-step run, between the cadence's checkpoints, step 9 is saved
    # and pushed, and pruning keeps it beside the latest two.
    store = LocalRunStore(tmp_path / "store")
    checkpoints = CheckpointCallback(store, RUN, every=4, keep_at=[0.5])
    out = tmp_path / "out"
    _train([checkpoints], out, 18, max_steps=18)
    assert checkpoints.kept == {0.5: 9}
    assert sorted(checkpoints.pushed) == [4, 8, 9, 12, 16]
    kept = sorted(p.name for p in (store.root / RUN / "checkpoints").iterdir())
    assert kept == ["step-000009", "step-000012", "step-000016"]
    assert (store.root / RUN / "checkpoints/step-000009/adapter_model.safetensors").read_text() == (
        "step 9"
    )

    # The time guard's pruning keeps it too: a step every 10 s of a minute stops after
    # step 5, whose push leaves the latest two and the kept step 2.
    other = LocalRunStore(tmp_path / "other")
    other.put_manifest(_manifest())
    clock = Clock()
    guard = TimeGuardCallback(1, store=other, run_id=RUN, keep_at=[0.1], clock=clock)
    every = CheckpointCallback(other, RUN, every=1, guard=guard, keep_at=[0.1])
    state = _train([Ticker(clock, 10), guard, every], tmp_path / "guarded", 20, max_steps=20)
    assert state.global_step == guard.stopped_at == 5
    assert guard.kept == every.kept == {0.1: 2}
    kept = sorted(p.name for p in (other.root / RUN / "checkpoints").iterdir())
    assert kept == ["step-000002", "step-000004", "step-000005"]


@pytest.mark.phase(4)
def test_the_guards_stop_is_the_guards_to_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalRunStore(tmp_path / "store")
    store.put_manifest(_manifest())
    pushes: list[str] = []
    put_folder = store.put_folder

    def counted(
        run_id: str, local_dir: str | Path, dest: str = "", *, replace: bool = False
    ) -> str:
        pushes.append(dest)
        return put_folder(run_id, local_dir, dest, replace=replace)

    monkeypatch.setattr(store, "put_folder", counted)
    clock = Clock()
    guard = TimeGuardCallback(1, store=store, run_id=RUN, clock=clock)
    checkpoints = CheckpointCallback(store, RUN, every=2, guard=guard)
    # Steps of 20 s: after the second, 20 s are left of the minute, and the guard stops.
    state = _train([Ticker(clock, 20), guard, checkpoints], tmp_path / "out", 10)
    assert state.global_step == guard.stopped_at == 2
    assert pushes == ["checkpoints/step-000002"]  # once, by the guard
    assert checkpoints.pushed == {}
    assert store.get_manifest(RUN).status == "stopped_time_guard"


@pytest.mark.phase(4)
def test_checkpoint_push_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = LocalRunStore(tmp_path / "store")
    put_folder = store.put_folder
    failures: list[OSError] = [StoreUnavailable("uploading failed 4 times")]

    def flaky(run_id: str, local_dir: str | Path, dest: str = "", *, replace: bool = False) -> str:
        if failures:
            raise failures.pop()
        return put_folder(run_id, local_dir, dest, replace=replace)

    monkeypatch.setattr(store, "put_folder", flaky)
    checkpoints = CheckpointCallback(store, RUN, every=2)
    out = tmp_path / "out"
    with caplog.at_level(logging.WARNING, logger="forge.train.callbacks"):
        _train([checkpoints], out, 4)
    # The network failed for step 2: it stays local, training goes on, step 4 is pushed.
    assert sorted(checkpoints.pushed) == [4]
    assert "step 2 of run" in caplog.text and "training goes on" in caplog.text
    assert (out / "checkpoint-2" / "adapter_model.safetensors").is_file()
    assert store.latest_checkpoint(RUN) == "checkpoints/step-000004"

    # A refusal fails again on every push, so it stops the job.
    failures.append(PermissionError("403 Forbidden"))
    with pytest.raises(OSError, match=r"checkpoint .*checkpoint-2 could not go to .*403"):
        _train([CheckpointCallback(store, RUN, every=2)], tmp_path / "again", 2)


@pytest.mark.phase(4)
def test_a_kept_checkpoint_waits_out_of_rotation_until_the_store_takes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = LocalRunStore(tmp_path / "store")
    put_folder = store.put_folder
    down = [True]

    def flaky(run_id: str, local_dir: str | Path, dest: str = "", *, replace: bool = False) -> str:
        if down[0]:
            raise StoreUnavailable(f"uploading {dest} failed 4 times")
        return put_folder(run_id, local_dir, dest, replace=replace)

    monkeypatch.setattr(store, "put_folder", flaky)
    out = tmp_path / "out"
    checkpoints = CheckpointCallback(store, RUN, every=4, keep_at=[0.25])
    with caplog.at_level(logging.WARNING, logger="forge.train.callbacks"):
        _train([checkpoints], out, 8, max_steps=8)
    # Step 2, kept for val, left the trainer's checkpoint-* folders, which its rotation
    # prunes, for a name it leaves alone; step 4 and 8 stay where the next push supersedes.
    aside = out / "unpushed-checkpoint-2"
    assert checkpoints.unpushed == {2: aside} and checkpoints.pushed == {}
    assert (aside / "adapter_model.safetensors").read_text() == "step 2"
    assert sorted(p.name for p in out.glob("checkpoint-*")) == ["checkpoint-4", "checkpoint-8"]
    assert "out of the trainer's rotation, since pushing it to checkpoints/step-000002" in (
        caplog.text
    )

    # Still down at the end: an error names the folder, which stays for the job to keep.
    args: Any = SimpleNamespace(output_dir=str(out))
    state, control = TrainerState(max_steps=8, global_step=8), TrainerControl()
    with caplog.at_level(logging.ERROR, logger="forge.train.callbacks"):
        checkpoints.on_train_end(args, state, control)
    assert f"ends with kept steps 2 that the store never took, only in {aside}" in caplog.text
    assert aside.is_dir()
    # Back at the end: the store takes it, beside the latest checkpoints it holds.
    down[0] = False
    checkpoints.on_train_end(args, state, control)
    assert checkpoints.unpushed == {} and not aside.exists()
    stored = store.root / RUN / "checkpoints" / "step-000002" / "adapter_model.safetensors"
    assert stored.read_text() == "step 2"


@pytest.mark.phase(4)
def test_metrics_go_to_jsonl_and_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = LocalRunStore(tmp_path / "store")
    path = tmp_path / "run" / "metrics.jsonl"
    seconds = iter(range(60))
    metrics = MetricsCallback(
        path,
        store=store,
        run_id=RUN,
        now=lambda: datetime(2026, 11, 15, 12, 0, next(seconds), tzinfo=UTC),
    )
    args: Any = SimpleNamespace(output_dir=str(tmp_path / "out"))
    state, control = TrainerState(), TrainerControl()
    state.global_step = 1
    metrics.on_log(args, state, control, logs={"loss": 1.5, "learning_rate": 2e-4})
    metrics.on_log(args, state, control, logs={})
    state.global_step = 2
    metrics.on_log(args, state, control, logs={"loss": float("nan")})
    metrics.on_save(args, state, control)

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines[0] == {
        "step": 1,
        "time": "2026-11-15T12:00:00+00:00",
        "loss": 1.5,
        "learning_rate": 2e-4,
    }
    assert lines[1]["step"] == 2 and math.isnan(lines[1]["loss"])  # an overflow shows
    stored = store.root / RUN / "metrics.jsonl"
    assert stored.read_bytes() == path.read_bytes()

    state.global_step = 3
    metrics.on_log(args, state, control, logs={"train_loss": 1.0})

    def unavailable(*args: object, **kwargs: object) -> str:
        raise StoreUnavailable("uploading failed 4 times")

    monkeypatch.setattr(store, "put_file", unavailable)
    with caplog.at_level(logging.WARNING, logger="forge.train.callbacks"):
        metrics.on_train_end(args, state, control)
    assert "in " + str(path) + " only for now" in caplog.text
    assert len(stored.read_text(encoding="utf-8").splitlines()) == 2
    monkeypatch.undo()
    metrics.on_train_end(args, state, control)
    assert len(stored.read_text(encoding="utf-8").splitlines()) == 3

    # Without a store the lines are written, and nothing is pushed.
    local = MetricsCallback(tmp_path / "local.jsonl")
    local.on_log(args, state, control, logs={"loss": 2.0})
    local.on_train_end(args, state, control)
    assert json.loads((tmp_path / "local.jsonl").read_text(encoding="utf-8"))["loss"] == 2.0
    with pytest.raises(ValueError, match="store and run_id together"):
        MetricsCallback(path, store=store)


@pytest.mark.phase(4)
def test_wandb_only_with_a_key_and_the_package(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Set, then deleted, so the test's own setting is undone whatever the code does.
    monkeypatch.setenv(WANDB_KEY, "unset")
    monkeypatch.delenv(WANDB_KEY)
    key = "-".join(("wandb", "test", "key"))
    assert wandb_callback({}) is None
    assert wandb_callback({WANDB_KEY: ""}) is None

    monkeypatch.setattr(transformers.integrations, "is_wandb_available", lambda: False)
    with caplog.at_level(logging.INFO, logger="forge.train.callbacks"):
        assert wandb_callback({WANDB_KEY: key}) is None
    assert "wandb is not installed: metrics go to JSONL only" in caplog.text
    assert WANDB_KEY not in os.environ

    monkeypatch.setattr(transformers.integrations, "is_wandb_available", lambda: True)
    monkeypatch.setattr(transformers.integrations, "WandbCallback", FakeWandb)
    assert isinstance(wandb_callback({WANDB_KEY: key}), FakeWandb)
    # wandb reads the key from the environment, where a key from .env now is too.
    assert os.environ[WANDB_KEY] == key


@pytest.mark.phase(4)
def test_training_callbacks_from_ops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = LocalRunStore(tmp_path / "store")
    path = tmp_path / "run" / "metrics.jsonl"
    ops = OpsConfig(checkpoint_every=25, time_guard_minutes=30)
    guard, checkpoints, metrics = training_callbacks(ops, store, RUN, metrics_path=path, env={})
    assert isinstance(guard, TimeGuardCallback)
    assert (guard.deadline.minutes, guard.store, guard.run_id) == (30, store, RUN)
    assert isinstance(checkpoints, CheckpointCallback)
    assert (checkpoints.every, checkpoints.guard, checkpoints.store) == (25, guard, store)
    assert isinstance(metrics, MetricsCallback)
    assert (metrics.path, metrics.store, metrics.run_id) == (path, store, RUN)

    monkeypatch.setenv(WANDB_KEY, "-".join(("wandb", "test", "key")))
    monkeypatch.setattr(transformers.integrations, "is_wandb_available", lambda: True)
    monkeypatch.setattr(transformers.integrations, "WandbCallback", FakeWandb)
    with_wandb = training_callbacks(ops, store, RUN, metrics_path=path, env=dict(os.environ))
    assert [type(c).__name__ for c in with_wandb] == [
        "TimeGuardCallback",
        "CheckpointCallback",
        "MetricsCallback",
        "FakeWandb",
    ]


@pytest.mark.phase(4)
def test_wandb_files_leave_the_tree_clean(tmp_path: Path) -> None:
    # W&B writes wandb/ in the folder training starts in, the checkout's top for forge
    # train sft, and the next run refuses a dirty tree, so the repo's ignore rules take it.
    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.copy2(ROOT / ".gitignore", repo / ".gitignore")
    for args in (
        ["init", "-q"],
        ["config", "user.email", "test@example.com"],
        ["config", "user.name", "Test"],
        ["add", "."],
        ["commit", "-q", "-m", "init"],
    ):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    head = git_sha(repo)
    written = [
        "wandb/run-20261116_035420-abc123/files/wandb-metadata.json",
        "kaggle/wandb/latest-run/files/config.yaml",
    ]
    for rel in written:
        (repo / rel).parent.mkdir(parents=True)
        (repo / rel).write_text("{}\n", encoding="utf-8")
    ignored = subprocess.run(
        ["git", "check-ignore", *written], cwd=repo, check=True, capture_output=True, text=True
    )
    assert ignored.stdout.splitlines() == written
    assert git_sha(repo) == head
    # Only a folder of that name: a module called wandb is code, and still counts.
    (repo / "wandb.py").write_text("import wandb\n", encoding="utf-8")
    with pytest.raises(DirtyTreeError):
        git_sha(repo)
