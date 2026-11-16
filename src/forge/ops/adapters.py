"""What an eval run's adapter names: a training run's checkpoint, or something else.

An adapter is a local folder or a Hub repo id with an optional ``@revision``
(``forge.infer.base.split_ref``). A folder that is a training run's checkpoint is
``<folder>/<run_id>/checkpoints/<name>``, the name ``final``, ``best`` or
``step-NNNNNN`` (plan section 3.9): ``runs/<run_id>/checkpoints/final`` from the repo
root on a laptop, and the same path in a kernel's work folder, where an eval job fetches
it from the run store first. ``checkpoint_ref`` is the one parser of that form, for
``forge kaggle run --adapter`` before a push, for the kernel's fetch and for
``forge runs promote``, so the three never disagree on a spelling. A trailing ``/``, as
a shell's completion adds, and a leading ``./`` name the same checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

from forge.eval.config import check_run_id
from forge.ops.hub import BEST, CHECKPOINTS, FINAL, RUNS, checkpoint_step


@dataclass(frozen=True)
class CheckpointRef:
    """A training run's checkpoint, as an adapter names it."""

    folder: str  # the folder that holds the run's folder, as the adapter writes it; "" for none
    run_id: str
    rel: str  # the checkpoint's path in its run, such as checkpoints/final

    @property
    def in_runs(self) -> bool:
        """Whether the adapter names it in ``runs/``, from where the command runs: the form
        a kernel fetches from the run store."""
        return self.folder == RUNS

    @property
    def path(self) -> str:
        """The checkpoint's folder, spelled the one way: ``runs/<run_id>/checkpoints/final``."""
        return str(PurePosixPath(self.folder or ".", self.run_id, self.rel))


def checkpoint_ref(adapter: str) -> CheckpointRef | None:
    """The training run's checkpoint ``adapter`` names, or None when it names none, as a
    Hub repo id or another folder does.

    Raises ``ValueError`` for a path whose last folders are ``<run_id>/checkpoints/<name>``
    with a name no checkpoint has, or a run id no run has.
    """
    parts = PurePosixPath(adapter).parts
    if len(parts) < 3 or parts[-2] != CHECKPOINTS:
        return None
    run_id, rel = parts[-3], f"{CHECKPOINTS}/{parts[-1]}"
    if rel not in (FINAL, BEST) and checkpoint_step(rel) is None:
        raise ValueError(
            f"{adapter} names {rel} of run {run_id}, and a run's checkpoints are {FINAL}, "
            f"{BEST} and {CHECKPOINTS}/step-NNNNNN"
        )
    try:
        check_run_id(run_id)
    except ValueError as exc:
        raise ValueError(f"{adapter} names a checkpoint of no run: {exc}") from None
    folder = PurePosixPath(*parts[:-3]).as_posix() if len(parts) > 3 else ""
    return CheckpointRef(folder, run_id, rel)
