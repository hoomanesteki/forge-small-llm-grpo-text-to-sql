"""The run stores (``forge.ops.hub``): a folder, and the Hub repo with its client stood in
for by ``FakeHub``, so no test needs a token or the network."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx2
import pytest
from huggingface_hub.errors import OfflineModeIsEnabled, RepositoryNotFoundError

from forge.core.types import RunManifest
from forge.ops.hub import (
    BEST,
    HubRunStore,
    LocalRunStore,
    RunNotFound,
    RunStore,
    StoreUnavailable,
    checkpoint_dir,
    manifest_bytes,
)
from tests.fakes import FakeHub, hub_error

RUN = "2026-11-15-tiny-grpo-abc1234"
OTHER = "2026-11-16-tiny-grpo-abc1234"
REPO = "owner/forge-runs"


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


def _never_sleep(seconds: float) -> None:
    raise AssertionError(f"waited {seconds} s for a call that should not be retried")


@pytest.fixture(params=["local", "hub"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> RunStore:
    """Each store, the Hub's with its client faked and nothing to retry."""
    if request.param == "local":
        return LocalRunStore(tmp_path / "store")
    return HubRunStore(REPO, api=FakeHub(), sleep=_never_sleep)


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
def test_hub_store_roundtrip(tmp_path: Path) -> None:
    fake = FakeHub()
    store = HubRunStore(REPO, api=fake, sleep=_never_sleep)
    run = tmp_path / "made" / RUN
    (run / "eval").mkdir(parents=True)
    (run / "eval" / "val-scores.jsonl").write_text("{}\n")
    (run / ".lock").write_text("")
    manifest = _manifest()
    (run / "manifest.json").write_bytes(manifest_bytes(manifest))

    assert store.put_folder(RUN, run) == f"runs/{RUN}"
    assert sorted(fake.files) == [f"runs/{RUN}/eval/val-scores.jsonl", f"runs/{RUN}/manifest.json"]
    assert store.runs() == [RUN]
    assert store.get_manifest(RUN) == manifest

    local = tmp_path / "runs" / RUN
    (local / "eval").mkdir(parents=True)
    (local / "eval" / "old.jsonl").write_text("from an earlier pull\n")
    (local / "kernels.json").write_text("{}\n")
    store.pull(RUN, local)

    # As forge kaggle output does: eval/ comes whole from the store, kernels.json stays.
    assert _files(local) == {
        "eval/val-scores.jsonl": b"{}\n",
        "kernels.json": b"{}\n",
        "manifest.json": manifest_bytes(manifest),
    }
    assert sorted(p.name for p in local.parent.iterdir()) == [RUN]


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
def test_hub_store_calls_api(tmp_path: Path) -> None:
    fake = FakeHub()
    # No token: the client is stood in for, and a real one would find its own.
    store = HubRunStore(REPO, api=fake, sleep=_never_sleep)
    (tmp_path / "metrics.jsonl").write_text("{}\n")
    manifest = _manifest()
    for step in (50, 100, 150):
        store.put_folder(RUN, _checkpoint(tmp_path, step), checkpoint_dir(step))
    fake.calls.clear()

    store.put_manifest(manifest)
    assert store.put_file(RUN, tmp_path / "metrics.jsonl", "metrics.jsonl") == (
        f"runs/{RUN}/metrics.jsonl"
    )
    assert store.put_folder(RUN, tmp_path / "saved-150", "checkpoints/best") == (
        f"runs/{RUN}/checkpoints/best"
    )
    assert store.get_manifest(RUN) == manifest
    assert store.latest_checkpoint(RUN) == "checkpoints/step-000150"
    assert store.prune_checkpoints(RUN) == ["checkpoints/step-000050"]
    assert store.runs() == [RUN]
    store.pull(RUN, tmp_path / "runs" / RUN)

    names = [method for method, _ in fake.calls]
    assert names == [
        "upload_file",
        "upload_file",
        "upload_folder",
        "hf_hub_download",
        "list_repo_files",
        "list_repo_files",
        "delete_files",
        "list_repo_files",
        "snapshot_download",
    ]
    # Every call goes to the run store's model repo, the private one doctor checks.
    assert {(kw["repo_id"], kw["repo_type"]) for _, kw in fake.calls} == {(REPO, "model")}
    calls = [
        {name: value for name, value in kw.items() if name not in ("repo_id", "repo_type")}
        for _, kw in fake.calls
    ]
    assert calls[0] == {
        "path_or_fileobj": manifest_bytes(manifest),
        "path_in_repo": f"runs/{RUN}/manifest.json",
        "commit_message": f"Write the running manifest of run {RUN}",
    }
    assert calls[1] == {
        "path_or_fileobj": str(tmp_path / "metrics.jsonl"),
        "path_in_repo": f"runs/{RUN}/metrics.jsonl",
        "commit_message": f"Add metrics.jsonl to run {RUN}",
    }
    assert calls[2] == {
        "folder_path": str(tmp_path / "saved-150"),
        "path_in_repo": f"runs/{RUN}/checkpoints/best",
        "ignore_patterns": [".lock"],
        "delete_patterns": None,
        "commit_message": f"Add checkpoints/best to run {RUN}",
    }
    # The manifest is read through a folder of its own, not the shared Hub cache.
    assert calls[3]["filename"] == f"runs/{RUN}/manifest.json"
    assert not Path(calls[3]["local_dir"]).exists()
    assert calls[4] == calls[5] == calls[7] == {}
    assert calls[6] == {
        "delete_patterns": [f"runs/{RUN}/checkpoints/step-000050/*"],
        "commit_message": f"Keep the latest 2 checkpoints of run {RUN}",
    }
    assert calls[8]["allow_patterns"] == [f"runs/{RUN}/*"]
    assert Path(calls[8]["local_dir"]).parent == tmp_path / "runs"


@pytest.mark.phase(3)
@pytest.mark.parametrize(
    "error",
    [
        httpx2.ConnectError("connection refused"),
        httpx2.ReadTimeout("read timed out"),
        hub_error(503),
        hub_error(429),
        RuntimeError("Error while uploading 'runs/x/metrics.jsonl' to the Hub."),
    ],
)
def test_hub_store_retries_network_errors_with_backoff(
    error: BaseException, tmp_path: Path
) -> None:
    (tmp_path / "metrics.jsonl").write_text("{}\n")
    waits: list[float] = []
    fake = FakeHub()
    store = HubRunStore(REPO, api=fake, sleep=waits.append)

    fake.fail["upload_file"] = [error, error]
    store.put_file(RUN, tmp_path / "metrics.jsonl", "metrics.jsonl")
    assert waits == [2.0, 4.0] and len(fake.calls) == 3
    assert fake.files == {f"runs/{RUN}/metrics.jsonl": b"{}\n"}

    waits.clear()
    fake.calls.clear()
    fake.fail["list_repo_files"] = [error] * 4
    with pytest.raises(StoreUnavailable, match="failed 4 times") as raised:
        store.runs()
    assert raised.value.__cause__ is error
    assert waits == [2.0, 4.0, 8.0] and len(fake.calls) == 4


@pytest.mark.phase(3)
@pytest.mark.parametrize(
    "error",
    [
        hub_error(401, RepositoryNotFoundError),
        hub_error(403),
        OfflineModeIsEnabled("HF_HUB_OFFLINE is set"),
        ValueError("Invalid repo id"),
    ],
)
def test_hub_store_fails_at_once_on_other_errors(error: BaseException) -> None:
    fake = FakeHub()
    store = HubRunStore(REPO, api=fake, sleep=_never_sleep)
    fake.fail["list_repo_files"] = [error]
    with pytest.raises(type(error)) as raised:
        store.latest_checkpoint(RUN)
    assert raised.value is error and len(fake.calls) == 1


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
