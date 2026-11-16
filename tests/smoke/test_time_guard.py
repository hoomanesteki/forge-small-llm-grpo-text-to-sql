"""The time guard on a real transformers ``Trainer``: the tiny model pinned in
configs/models.yaml trains on CPU against a one-minute deadline, on a clock the test moves
15 s a step, so the guard stops it at the same step every time and in seconds."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from forge.core.types import RunManifest
from forge.infer.hf_backend import split_ref
from forge.ops.hub import LocalRunStore, checkpoint_dir
from forge.train.callbacks import TimeGuardCallback

RUN = "2026-11-15-tiny-sft-abc1234"
STEP_S = 15.0


class Ticker(TrainerCallback):
    """Moves the fake clock at the end of each step, before the guard reads it."""

    def __init__(self, now: list[float]) -> None:
        self.now = now

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        self.now[0] += STEP_S


def _manifest() -> RunManifest:
    return RunManifest(
        run_id=RUN,
        kind="sft",
        git_sha="a" * 40,
        config={"max_steps": 10},
        config_hash="1" * 64,
        data_manifest_hash="2" * 64,
        model_ref="tiny",
        platform="local-cpu",
        seed=0,
        started_at=datetime.now(UTC),
    )


def _trainer(
    tmp_path: Path, tiny_model_ref: str, **options: Any
) -> tuple[Trainer, TimeGuardCallback, LocalRunStore]:
    """A trainer of the tiny model, for 10 steps of 2 rows, with the guard on a one-minute
    deadline and its run in a local store; ``options`` are more ``TrainingArguments``."""
    repo, revision = split_ref(tiny_model_ref)
    model = AutoModelForCausalLM.from_pretrained(repo, revision=revision)
    generator = torch.Generator().manual_seed(0)
    ids = torch.randint(model.config.vocab_size, (16, 12), generator=generator).tolist()
    rows = [{"input_ids": row, "labels": row} for row in ids]
    store = LocalRunStore(tmp_path / "store")
    store.put_manifest(_manifest())
    now = [0.0]
    guard = TimeGuardCallback(1, store=store, run_id=RUN, clock=lambda: now[0])
    args = TrainingArguments(
        output_dir=str(tmp_path / "out"),
        max_steps=10,
        per_device_train_batch_size=2,
        use_cpu=True,
        logging_strategy="no",
        report_to="none",
        disable_tqdm=True,
        dataloader_num_workers=0,
        seed=0,
        **options,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=rows,
        eval_dataset=rows[:4],
        callbacks=[Ticker(now), guard],
    )
    return trainer, guard, store


@pytest.mark.phase(3)
def test_time_guard_stops_and_saves(tmp_path: Path, tiny_model_ref: str) -> None:
    out = tmp_path / "out"
    # The only save is the guard's.
    trainer, guard, store = _trainer(
        tmp_path, tiny_model_ref, learning_rate=1e-3, save_strategy="no"
    )

    trainer.train()

    # After 3 steps of 15 s, 15 s are left of the minute: another step could pass it.
    assert guard.stopped_at == trainer.state.global_step == 3
    assert sorted(p.name for p in out.iterdir()) == ["checkpoint-3"]
    assert guard.pushed == str(store.root / RUN / checkpoint_dir(3))
    assert store.latest_checkpoint(RUN) == "checkpoints/step-000003"
    pushed = Path(guard.pushed)
    assert json.loads((pushed / "trainer_state.json").read_text())["global_step"] == 3
    # The pushed weights are the ones training stopped with.
    saved = load_file(pushed / "model.safetensors")
    trained = trainer.model.state_dict()
    assert saved.keys() <= trained.keys() and saved
    assert all(torch.equal(saved[name], trained[name]) for name in saved)

    manifest = store.get_manifest(RUN)
    assert manifest.status == "stopped_time_guard"
    assert manifest.ended_at is not None and manifest.ended_at >= manifest.started_at
    assert manifest.gpu_hours == 0.0  # a local-cpu run holds no GPU


class Evaluations(TrainerCallback):
    """Records the step of each evaluation the trainer runs."""

    def __init__(self) -> None:
        self.steps: list[int] = []

    def on_evaluate(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        self.steps.append(state.global_step)


@pytest.mark.phase(3)
def test_time_guard_saves_under_save_strategy_best(tmp_path: Path, tiny_model_ref: str) -> None:
    # Keeping only the best checkpoint, the trainer saves a step when its evaluation is a
    # new best. At learning rate 0 the eval loss never beats step 1's, so no later step is
    # one, the guard's stop step included, which an evaluation there would leave unsaved.
    trainer, guard, store = _trainer(
        tmp_path,
        tiny_model_ref,
        learning_rate=0.0,
        eval_strategy="steps",
        eval_steps=1,
        save_strategy="best",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    evaluations = Evaluations()
    trainer.add_callback(evaluations)

    trainer.train()

    assert guard.stopped_at == trainer.state.global_step == 3
    # The stop step isn't evaluated: the deadline kept no time for it.
    assert evaluations.steps == [1, 2]
    assert trainer.state.best_global_step == 1
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == ["checkpoint-1", "checkpoint-3"]
    assert guard.pushed == str(store.root / RUN / checkpoint_dir(3))
    assert store.latest_checkpoint(RUN) == "checkpoints/step-000003"
    assert store.get_manifest(RUN).status == "stopped_time_guard"
