"""``forge.ops.adapters``: which training run's checkpoint an adapter names."""

from __future__ import annotations

import pytest

from forge.ops.adapters import CheckpointRef, checkpoint_ref

RUN = "2026-11-20-qwen3-4b-sft-abc1234"


@pytest.mark.phase(4)
def test_checkpoint_ref_reads_every_spelling_of_a_checkpoint() -> None:
    final = CheckpointRef("runs", RUN, "checkpoints/final")
    # A shell's trailing slash, a leading ./ and a doubled slash name the same folder.
    for spelling in (
        f"runs/{RUN}/checkpoints/final",
        f"runs/{RUN}/checkpoints/final/",
        f"./runs/{RUN}/checkpoints/final",
        f"runs//{RUN}/checkpoints/final",
    ):
        assert checkpoint_ref(spelling) == final, spelling
    assert final.in_runs and final.path == f"runs/{RUN}/checkpoints/final"

    # best/ and a step, and a run's folder anywhere, as a kernel's work folder holds it.
    assert checkpoint_ref(f"runs/{RUN}/checkpoints/best") == CheckpointRef(
        "runs", RUN, "checkpoints/best"
    )
    kernel = checkpoint_ref(f"/kaggle/working/runs/{RUN}/checkpoints/step-000242/")
    assert kernel == CheckpointRef("/kaggle/working/runs", RUN, "checkpoints/step-000242")
    assert kernel is not None and not kernel.in_runs
    assert kernel.path == f"/kaggle/working/runs/{RUN}/checkpoints/step-000242"
    bare = checkpoint_ref(f"{RUN}/checkpoints/final")
    assert bare == CheckpointRef("", RUN, "checkpoints/final") and not bare.in_runs
    assert bare.path == f"{RUN}/checkpoints/final"

    # A Hub repo, a folder of another shape and nothing at all name no checkpoint.
    for other in ("org/adapter@abc123", "org/adapter", f"runs/{RUN}", "adapters/one", ""):
        assert checkpoint_ref(other) is None, other


@pytest.mark.phase(4)
def test_checkpoint_ref_refuses_what_no_run_holds() -> None:
    for name in ("latest", "step-42", "step-0000001", "checkpoint-242"):
        with pytest.raises(ValueError, match="a run's checkpoints are checkpoints/final, "):
            checkpoint_ref(f"runs/{RUN}/checkpoints/{name}")
    with pytest.raises(ValueError, match="names a checkpoint of no run: run id 'Run_1'"):
        checkpoint_ref("runs/Run_1/checkpoints/final")
