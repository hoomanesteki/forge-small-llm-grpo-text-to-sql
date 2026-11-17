"""Callbacks for Hugging Face trainers; TRL's SFT and GRPO trainers take them too.

``TimeGuardCallback`` is the time guard of plan section 3.8 (``forge.ops.timeguard``).
It times each optimizer step, from one step's end to the next, so logging, evaluation and
saving in between count too. Once the deadline is within the longest step so far plus
the margin, it sets ``control.should_save`` and ``control.should_training_stop``, and the
trainer saves ``checkpoint-<step>`` in its output folder and stops after that step. Then,
with a run store, the callback pushes that folder to ``checkpoints/step-NNNNNN`` in the
run (``forge.ops.hub.checkpoint_dir``), where the next session resumes from it, marks
the run's stored manifest ``stopped_time_guard``, and prunes the run's older checkpoints
as retention says. The last step needs no guard, since the trainer stops after it anyway.

Once the guard stops training, nothing but that save runs: no evaluation, at the stop step
or at its epoch's end, since the deadline kept no time for one. That also keeps the save
whatever the save strategy: under ``save_strategy="best"`` an evaluation would make the
trainer save the step only if its metric were a new best. The trainer runs its own
``DefaultFlowCallback`` before any callback it is given, so the guard's flags are the last
set. Should something still clear the save, the epoch's end asks for it again.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from forge.ops.hub import RunStore, checkpoint_dir
from forge.ops.manifest import close
from forge.ops.timeguard import Clock, Deadline

log = logging.getLogger(__name__)


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
