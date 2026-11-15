"""The run stores (``forge.ops.hub``), each through the same tests: a folder for now."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from forge.core.types import RunManifest
from forge.ops.hub import (
    BEST,
    LocalRunStore,
    RunNotFound,
    RunStore,
    checkpoint_dir,
    manifest_bytes,
)

RUN = "2026-11-15-tiny-grpo-abc1234"
OTHER = "2026-11-16-tiny-grpo-abc1234"


def _manifest(run_id: str = RUN) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        kind="grpo",
        git_sha="a" * 40,
        config={"learning_rate": 5e-6, "num_generations": 6},
        config_hash="0" * 64,
        data_manifest_hash="1" * 64,
        model_ref="org/tiny",
        platform="kaggle",
        gpu="T4",
        seed=42,
        started_at=datetime(2026, 11, 15, 8, 0, tzinfo=UTC),
        outputs={"metrics": "metrics.jsonl"},
    )


def _checkpoint(root: Path, step: int) -> Path:
    """A checkpoint folder as the trainer writes one, its files marked with the step."""
    folder = root / f"saved-{step}"
    (folder / "nested").mkdir(parents=True)
    (folder / "adapter_model.safetensors").write_bytes(f"weights at {step}".encode())
    (folder / "nested" / "trainer_state.json").write_text(f'{{"global_step": {step}}}')
    return folder


def _files(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.fixture(params=["local"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> RunStore:
    return LocalRunStore(tmp_path / "store")


@pytest.mark.phase(3)
def test_local_store_roundtrip(tmp_path: Path) -> None:
    store = LocalRunStore(tmp_path / "store")
    work = tmp_path / "work"
    work.mkdir()
    (work / "metrics.jsonl").write_text('{"step": 50, "reward": 0.5}\n')
    manifest = _manifest()

    store.put_manifest(manifest)
    assert store.put_file(RUN, work / "metrics.jsonl", "metrics.jsonl") == str(
        tmp_path / "store" / RUN / "metrics.jsonl"
    )
    store.put_folder(RUN, _checkpoint(work, 50), checkpoint_dir(50))
    # A lock is an eval run's hold on its folder on one machine, so it never travels.
    (tmp_path / "store" / RUN / ".lock").write_text("")

    assert store.runs() == [RUN]
    assert store.get_manifest(RUN) == manifest
    assert store.latest_checkpoint(RUN) == "checkpoints/step-000050"

    local = tmp_path / "runs" / RUN
    local.mkdir(parents=True)
    (local / "kernels.json").write_text('{"kernels": []}\n')
    (local / "metrics.jsonl").write_text("from an earlier pull\n")
    store.pull(RUN, local)

    assert _files(local) == {
        "checkpoints/step-000050/adapter_model.safetensors": b"weights at 50",
        "checkpoints/step-000050/nested/trainer_state.json": b'{"global_step": 50}',
        "kernels.json": b'{"kernels": []}\n',
        "manifest.json": manifest_bytes(manifest),
        "metrics.jsonl": b'{"step": 50, "reward": 0.5}\n',
    }
    assert RunManifest.model_validate_json((local / "manifest.json").read_bytes()) == manifest
    # Pulling copies: the store keeps its run.
    assert store.get_manifest(RUN) == manifest and (tmp_path / "store" / RUN / ".lock").exists()


@pytest.mark.phase(3)
def test_latest_checkpoint_selects_highest_step(store: RunStore, tmp_path: Path) -> None:
    assert store.latest_checkpoint(RUN) is None
    for step in (50, 1_000_000, 150, 999_999, 200):
        store.put_folder(RUN, _checkpoint(tmp_path, step), checkpoint_dir(step))
    store.put_folder(RUN, _checkpoint(tmp_path, 1), BEST)
    # Not names checkpoint_dir writes, so not checkpoints retention or resume may pick.
    store.put_folder(RUN, _checkpoint(tmp_path, 2), "checkpoints/step-0000300")
    store.put_folder(RUN, _checkpoint(tmp_path, 3), "checkpoints/step-12a")
    store.put_folder(RUN, _checkpoint(tmp_path, 4), "samples/step-9000000")
    store.put_folder(OTHER, _checkpoint(tmp_path, 5), checkpoint_dir(5_000_000))
    if isinstance(store, LocalRunStore):
        # A folder the trainer made but put nothing in: the Hub has no such folders.
        (store.root / RUN / checkpoint_dir(2_000_000)).mkdir()

    # By step, not by name: "step-999999" sorts after "step-1000000".
    assert store.latest_checkpoint(RUN) == "checkpoints/step-1000000"
    assert store.latest_checkpoint(OTHER) == "checkpoints/step-5000000"
    assert checkpoint_dir(50) == "checkpoints/step-000050"
    with pytest.raises(ValueError, match="can't be negative"):
        checkpoint_dir(-1)


@pytest.mark.phase(3)
def test_retention_keeps_latest_two_and_best(store: RunStore, tmp_path: Path) -> None:
    for step in (100, 50, 200, 150):
        store.put_folder(RUN, _checkpoint(tmp_path, step), checkpoint_dir(step))
    store.put_folder(RUN, _checkpoint(tmp_path, 75), BEST)
    store.put_folder(OTHER, _checkpoint(tmp_path, 25), checkpoint_dir(25))
    store.put_manifest(_manifest())

    assert store.prune_checkpoints(RUN) == ["checkpoints/step-000050", "checkpoints/step-000100"]

    pulled = tmp_path / "pulled"
    store.pull(RUN, pulled)
    assert sorted(p.name for p in (pulled / "checkpoints").iterdir()) == [
        "best",
        "step-000150",
        "step-000200",
    ]
    assert (pulled / BEST / "adapter_model.safetensors").read_bytes() == b"weights at 75"
    assert (pulled / "checkpoints/step-000200/nested/trainer_state.json").is_file()
    assert store.latest_checkpoint(OTHER) == "checkpoints/step-000025"
    # Nothing more to delete, and a rule that would delete the latest is refused.
    assert store.prune_checkpoints(RUN) == []
    assert store.prune_checkpoints(RUN, keep=1) == ["checkpoints/step-000150"]
    with pytest.raises(ValueError, match="keep must be at least 1"):
        store.prune_checkpoints(RUN, keep=0)


@pytest.mark.phase(3)
def test_missing_runs_are_named(store: RunStore, tmp_path: Path) -> None:
    with pytest.raises(RunNotFound, match=f"no run {RUN}"):
        store.get_manifest(RUN)
    with pytest.raises(RunNotFound, match=f"no run {RUN}"):
        store.pull(RUN, tmp_path / "runs" / RUN)
    assert store.runs() == []
    assert not (tmp_path / "runs").exists() or list((tmp_path / "runs").iterdir()) == []


@pytest.mark.phase(3)
@pytest.mark.parametrize("dest", ["", "../escape", "/etc/passwd", "eval//x", "eval/./x", "a\\b"])
def test_paths_outside_the_run_are_refused(store: RunStore, dest: str, tmp_path: Path) -> None:
    (tmp_path / "file").write_text("x")
    with pytest.raises(ValueError, match="relative path inside the run's folder"):
        store.put_file(RUN, tmp_path / "file", dest)
    if dest:
        with pytest.raises(ValueError, match="relative path inside the run's folder"):
            store.put_folder(RUN, tmp_path, dest)
    for run_id in ("../up", "Run", ""):
        with pytest.raises(ValueError, match="names a folder"):
            store.put_file(run_id, tmp_path / "file", "file")
    assert store.runs() == []


@pytest.mark.phase(3)
def test_a_corrupt_manifest_is_refused(store: RunStore, tmp_path: Path) -> None:
    other = tmp_path / "other.json"
    other.write_bytes(manifest_bytes(_manifest(OTHER)))
    store.put_file(RUN, other, "manifest.json")
    with pytest.raises(ValueError, match=f"is the manifest of run {OTHER}, not of {RUN}"):
        store.get_manifest(RUN)
    other.write_text("{}")
    store.put_file(RUN, other, "manifest.json")
    with pytest.raises(ValueError, match="is not a run manifest"):
        store.get_manifest(RUN)


@pytest.mark.phase(3)
def test_replace_leaves_only_the_folders_files(store: RunStore, tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "eval").mkdir(parents=True)
    (run / "error.txt").write_text("Traceback: the first attempt failed\n")
    (run / "eval" / "val-scores.jsonl").write_text("{}\n")
    store.put_folder(RUN, run)
    store.put_folder(RUN, _checkpoint(tmp_path, 50), checkpoint_dir(50))
    store.put_folder(OTHER, run)

    # The second attempt finished, so its folder has no error.txt.
    (run / "error.txt").unlink()
    (run / "eval" / "val-scores.jsonl").write_text("{}\n{}\n")
    (run / ".lock").write_text("")
    store.put_folder(RUN, run)
    kept, replaced = tmp_path / "kept", tmp_path / "replaced"
    store.pull(RUN, kept)
    store.put_folder(RUN, run, replace=True)
    store.pull(RUN, replaced)

    assert sorted(_files(kept)) == [
        "checkpoints/step-000050/adapter_model.safetensors",
        "checkpoints/step-000050/nested/trainer_state.json",
        "error.txt",
        "eval/val-scores.jsonl",
    ]
    assert _files(replaced) == {"eval/val-scores.jsonl": b"{}\n{}\n"}
    # Another run is another folder, which replace never reaches.
    store.pull(OTHER, tmp_path / "other")
    assert "error.txt" in _files(tmp_path / "other")
