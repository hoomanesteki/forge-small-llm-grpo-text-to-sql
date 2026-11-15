"""``TimeGuardCallback`` driven by hand, event by event, as a trainer would drive it."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from transformers import TrainerControl, TrainerState

from forge.core.types import RunManifest
from forge.ops.hub import LocalRunStore, StoreUnavailable
from forge.train.callbacks import TimeGuardCallback

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
