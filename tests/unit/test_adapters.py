"""``forge.ops.adapters``: which training run's checkpoint an adapter names, and the
digest that ties a score to the adapter's files."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from forge.ops.adapters import (
    CheckpointRef,
    adapter_folder,
    adapter_sha256,
    checkpoint_digests,
    checkpoint_ref,
    digest_output,
)

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


def _adapter(folder: Path, weights: str) -> Path:
    folder.mkdir(parents=True)
    (folder / "adapter_config.json").write_text('{"r": 16}\n')
    (folder / "adapter_model.safetensors").write_text(weights)
    return folder


@pytest.mark.phase(4)
def test_adapter_sha256_is_the_files_wherever_they_are(tmp_path: Path) -> None:
    run_dir = tmp_path / RUN
    final = _adapter(run_dir / "checkpoints" / "final", "final weights")
    digest = adapter_sha256(final)
    # A copy anywhere hashes alike, and a byte of the weights or the settings changes it.
    assert adapter_sha256(Path(shutil.copytree(final, tmp_path / "pulled"))) == digest
    (tmp_path / "pulled" / "adapter_config.json").write_text('{"r": 8}\n')
    assert adapter_sha256(tmp_path / "pulled") != digest
    half = _adapter(run_dir / "checkpoints" / "step-000242", "half weights")
    assert adapter_sha256(half) != digest
    (run_dir / "checkpoints" / "partial").mkdir()
    (run_dir / "checkpoints" / "partial" / "adapter_model.safetensors").write_text("cut short")
    with pytest.raises(FileNotFoundError, match=r"partial has no adapter_config\.json, so it"):
        adapter_sha256(run_dir / "checkpoints" / "partial")

    # A training run records each checkpoint that holds an adapter, and nothing else.
    assert checkpoint_digests(run_dir) == {
        "adapter_sha256:checkpoints/final": digest,
        "adapter_sha256:checkpoints/step-000242": adapter_sha256(half),
    }
    assert digest_output("checkpoints/final") == "adapter_sha256:checkpoints/final"
    assert checkpoint_digests(tmp_path / "no-run") == {}


@pytest.mark.phase(4)
def test_adapter_folder_is_where_a_backend_loads_it_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = _adapter(tmp_path / "local", "weights")
    assert adapter_folder(str(local)) == local
    # A run's checkpoint is never on the Hub, so a missing one says how to get it.
    with pytest.raises(FileNotFoundError, match=f"no such folder here; forge runs pull {RUN}"):
        adapter_folder(f"runs/{RUN}/checkpoints/final")

    # A Hub repo's two files, at its revision, through the Hub's cache.
    cached = _adapter(tmp_path / "cache", "hub weights")
    calls: list[tuple[str, dict[str, object]]] = []

    def snapshot_download(repo: str, **kwargs: object) -> str:
        calls.append((repo, kwargs))
        return str(cached)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    assert adapter_folder("org/adapter@abc123") == cached
    patterns = ["adapter_config.json", "adapter_model.safetensors"]
    assert calls == [("org/adapter", {"revision": "abc123", "allow_patterns": patterns})]
    with pytest.raises(FileNotFoundError, match="neither a directory nor a Hub repo id"):
        adapter_folder("no such adapter")
