"""Callbacks for Hugging Face trainers; TRL's SFT and GRPO trainers take them too.

``training_callbacks`` makes the ones every training job runs with, on either backend
(``forge.train.backends``), from the config's ops section:

- ``TimeGuardCallback`` is the time guard of plan section 3.8 (``forge.ops.timeguard``).
  It times each optimizer step, from one step's end to the next, so logging, evaluation
  and saving in between count too. Once the deadline is within the longest step so far
  plus the margin, it sets ``control.should_save`` and ``control.should_training_stop``,
  and the trainer saves ``checkpoint-<step>`` in its output folder and stops after that
  step. Then, with a run store, the callback pushes that folder to
  ``checkpoints/step-NNNNNN`` in the run (``forge.ops.hub.checkpoint_dir``), where the
  next session resumes from it, marks the run's stored manifest ``stopped_time_guard``,
  and prunes the run's older checkpoints as retention says. The last step needs no
  guard, since the trainer stops after it anyway.

  Once the guard stops training, nothing but that save runs: no evaluation, at the stop
  step or at its epoch's end, since the deadline kept no time for one. That also keeps
  the save whatever the save strategy: under ``save_strategy="best"`` an evaluation would
  make the trainer save the step only if its metric were a new best. The trainer runs its
  own ``DefaultFlowCallback`` before any callback it is given, so the guard's flags are
  the last set. Should something still clear the save, the epoch's end asks for it again.
- ``CheckpointCallback`` has the trainer save every ``checkpoint_every`` steps and pushes
  each checkpoint to the same folders, pruning as the guard does; a stop of the guard's
  is the guard's to push. A push that still fails on the network after the store's
  retries is logged and training goes on, since the checkpoint stays in the output
  folder and the next one tries again. Any other failure, such as a refused token,
  stops the job: no later push would work either.
- ``MetricsCallback`` appends each log the trainer writes to ``metrics.jsonl`` as one
  JSON line, and pushes the file to the run at every save and at the end, failing as
  ``CheckpointCallback`` does.
- ``wandb_callback`` adds transformers' W&B callback when ``WANDB_API_KEY`` is set and
  wandb is installed. Neither is needed (ADR-0013): the JSONL metrics are always there.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from forge.ops.dotenv import forge_env
from forge.ops.hub import RunStore, StoreUnavailable, checkpoint_dir
from forge.ops.manifest import close
from forge.ops.timeguard import Clock, Deadline
from forge.train.config import OpsConfig

log = logging.getLogger(__name__)

METRICS_FILE = "metrics.jsonl"  # its name in a run, plan section 3.9
WANDB_KEY = "WANDB_API_KEY"


class TimeGuardCallback(TrainerCallback):
    """Stop training ``deadline_minutes`` after this callback is made, saving first.

    Make it when the job starts, since the session's clock is already running. ``store``
    and ``run_id`` name where the checkpoint and the manifest go; without them the guard
    only saves and stops, and the job marks its own manifest by ``stopped_at``.
    ``margin_s`` keeps time back for the save and the push.
    """

    def __init__(
        self,
        deadline_minutes: float,
        *,
        store: RunStore | None = None,
        run_id: str | None = None,
        margin_s: float = 0.0,
        clock: Clock = time.monotonic,
    ) -> None:
        if (store is None) != (run_id is None):
            raise ValueError("pass store and run_id together: the checkpoint goes to that run")
        self.deadline = Deadline(deadline_minutes, margin_s=margin_s, clock=clock)
        self.store = store
        self.run_id = run_id
        self.stopped_at: int | None = None  # the step training stopped after
        self.saved = False  # whether the trainer has saved that step
        self.pushed: str | None = None  # where the store holds that step's checkpoint

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        # Loading the model is no step's time.
        self.deadline.mark()

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        self.deadline.lap()
        if self.stopped_at is not None or state.global_step >= state.max_steps:
            return
        if self.deadline.near():
            log.warning(
                "time guard: saving and stopping after step %d; %s",
                state.global_step,
                self.deadline.describe(),
            )
            self.stopped_at = state.global_step
            control.should_save = True
            control.should_training_stop = True
            control.should_evaluate = False

    def on_epoch_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        # The trainer evaluates and saves once more after the stop step breaks its epoch.
        if self.stopped_at is None:
            return
        control.should_evaluate = False
        control.should_save = not self.saved

    def on_save(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        # Only the save the guard asked for, and only once: another one of the same step
        # pushes nothing again.
        if self.stopped_at != state.global_step:
            return
        self.saved = True
        if self.pushed is not None or self.store is None or self.run_id is None:
            return
        if args.output_dir is None:  # the trainer fills it in before it saves anything
            raise ValueError("the trainer saved a checkpoint, but its args name no output_dir")
        folder = Path(args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        dest = checkpoint_dir(state.global_step)
        try:
            self.pushed = self.store.put_folder(self.run_id, folder, dest)
        except OSError as exc:
            raise OSError(
                f"the time guard saved {folder}, but pushing it to {dest} of run "
                f"{self.run_id} in {self.store} failed: {exc}"
            ) from exc
        manifest = self.store.get_manifest(self.run_id)
        self.store.put_manifest(close(manifest, "stopped_time_guard"))
        log.warning("time guard: run %s stopped; its checkpoint is %s", self.run_id, self.pushed)
        try:
            self.store.prune_checkpoints(self.run_id)
        # Logged, not raised: the checkpoint and the manifest are stored, and the next push
        # prunes again.
        except OSError as exc:
            log.warning("time guard: run %s keeps its older checkpoints: %s", self.run_id, exc)


class CheckpointCallback(TrainerCallback):
    """Save every ``every`` steps, and push each of those checkpoints to the run store.

    A step ``guard`` stops at is the guard's to push, since it marks the manifest once
    the checkpoint is stored.
    """

    def __init__(
        self,
        store: RunStore,
        run_id: str,
        every: int,
        *,
        guard: TimeGuardCallback | None = None,
    ) -> None:
        if every < 1:
            raise ValueError(f"a checkpoint comes every 1 step or more, not every {every}")
        self.store = store
        self.run_id = run_id
        self.every = every
        self.guard = guard
        self.pushed: dict[int, str] = {}  # step: where the store holds its checkpoint

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        if state.global_step % self.every == 0:
            control.should_save = True

    def on_save(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        step = state.global_step
        if step % self.every or step in self.pushed:
            return
        if self.guard is not None and self.guard.stopped_at == step:
            return
        if args.output_dir is None:  # the trainer fills it in before it saves anything
            raise ValueError("the trainer saved a checkpoint, but its args name no output_dir")
        folder = Path(args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{step}"
        dest = checkpoint_dir(step)
        try:
            self.pushed[step] = self.store.put_folder(self.run_id, folder, dest)
        except StoreUnavailable as exc:
            log.warning(
                "checkpoint: step %d of run %s is only in %s, since pushing it to %s failed: "
                "%s; training goes on, and the next checkpoint is pushed as usual",
                step,
                self.run_id,
                folder,
                dest,
                exc,
            )
            return
        except OSError as exc:
            raise OSError(
                f"checkpoint {folder} could not go to {dest} of run {self.run_id} in "
                f"{self.store}: {exc}"
            ) from exc
        log.info("checkpoint: step %d of run %s is %s", step, self.run_id, self.pushed[step])
        try:
            self.store.prune_checkpoints(self.run_id)
        # Logged, not raised: the checkpoint is stored, and the next push prunes again.
        except OSError as exc:
            log.warning("checkpoint: run %s keeps its older checkpoints: %s", self.run_id, exc)


class MetricsCallback(TrainerCallback):
    """Append each log of the trainer to the JSONL file ``path``, and push the file to
    the run as ``metrics.jsonl`` at every save and at the end of training.

    A line holds the step, the wall-clock time and what the trainer logged. A loss that
    overflowed is written as NaN or Infinity, which Python's json reads back, so the
    telemetry sees it. On a resume ``path`` must be the file the run already has, such
    as the earlier session's output holds, so the new lines follow the old ones: the
    push replaces the stored file.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        store: RunStore | None = None,
        run_id: str | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if (store is None) != (run_id is None):
            raise ValueError("pass store and run_id together: the metrics go to that run")
        self.path = Path(path)
        self.store = store
        self.run_id = run_id
        self.now = now

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return
        line = {"step": state.global_step, "time": self.now().isoformat(), **logs}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(line) + "\n")

    def on_save(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        self._push()

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        self._push()

    def _push(self) -> None:
        if self.store is None or self.run_id is None or not self.path.is_file():
            return
        try:
            self.store.put_file(self.run_id, self.path, METRICS_FILE)
        except StoreUnavailable as exc:
            log.warning(
                "metrics: run %s has them in %s only for now, since pushing failed: %s",
                self.run_id,
                self.path,
                exc,
            )
        except OSError as exc:
            raise OSError(
                f"metrics {self.path} could not go to run {self.run_id} in {self.store}: {exc}"
            ) from exc


def wandb_callback(env: Mapping[str, str] | None = None) -> TrainerCallback | None:
    """transformers' W&B callback when ``WANDB_API_KEY`` is set, in the environment or
    the repo's ``.env``, and wandb is installed; otherwise None."""
    values = forge_env() if env is None else env
    key = values.get(WANDB_KEY)
    if not key:
        return None
    from transformers.integrations import WandbCallback, is_wandb_available

    if not is_wandb_available():
        log.info("%s is set, but wandb is not installed: metrics go to JSONL only", WANDB_KEY)
        return None
    # wandb reads its key from the environment, and the runbook lets it come from .env.
    os.environ.setdefault(WANDB_KEY, key)
    callback: TrainerCallback = WandbCallback()
    return callback


def training_callbacks(
    ops: OpsConfig,
    store: RunStore,
    run_id: str,
    *,
    metrics_path: str | Path,
    env: Mapping[str, str] | None = None,
    clock: Clock = time.monotonic,
) -> list[TrainerCallback]:
    """The callbacks of every training job, as the module docstring lists them, for the
    run ``run_id`` in ``store``.

    Make them when the job starts: the time guard's deadline counts from then.
    """
    guard = TimeGuardCallback(ops.time_guard_minutes, store=store, run_id=run_id, clock=clock)
    callbacks: list[TrainerCallback] = [
        guard,
        CheckpointCallback(store, run_id, ops.checkpoint_every, guard=guard),
        MetricsCallback(metrics_path, store=store, run_id=run_id),
    ]
    tracker = wandb_callback(env)
    if tracker is not None:
        callbacks.append(tracker)
    return callbacks
