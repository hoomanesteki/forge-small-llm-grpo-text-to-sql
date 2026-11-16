"""The run stores (``forge.ops.hub``): a folder, and the Hub repo with its client stood in
for by ``FakeHub``, so no test needs a token or the network."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import huggingface_hub
import pytest
from huggingface_hub import HfApi
from huggingface_hub.errors import (
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
)
from huggingface_hub.utils import _http

from forge.core.types import RunManifest
from forge.eval.run_files import RunCorrupt
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
from tests.fakes import (
    REPO_URL,
    RESOLVE_URL,
    TREE_URL,
    FakeHub,
    connect_error,
    dropped_error,
    hub_error,
)

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


@pytest.mark.phase(4)
def test_retention_keeps_the_steps_a_job_names(store: RunStore, tmp_path: Path) -> None:
    # A run's checkpoint half way, which val compares with its final adapter (FG-404).
    for step in (50, 100, 150, 200):
        store.put_folder(RUN, _checkpoint(tmp_path, step), checkpoint_dir(step))

    assert store.prune_checkpoints(RUN, keep_steps=[100]) == ["checkpoints/step-000050"]
    assert store.prune_checkpoints(RUN, keep=1, keep_steps={100}) == ["checkpoints/step-000150"]
    pulled = tmp_path / "pulled"
    store.pull(RUN, pulled)
    assert sorted(p.name for p in (pulled / "checkpoints").iterdir()) == [
        "step-000100",
        "step-000200",
    ]
    # A kept step is no reason to keep fewer of the latest, nor is it the latest.
    store.put_folder(RUN, _checkpoint(tmp_path, 250), checkpoint_dir(250))
    assert store.prune_checkpoints(RUN, keep_steps={100}) == []
    assert store.latest_checkpoint(RUN) == "checkpoints/step-000250"


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


def _lfs_failure(cause: BaseException) -> RuntimeError:
    """A failed LFS upload as huggingface_hub reports it: a bare RuntimeError from its cause."""
    error = RuntimeError("Error while uploading 'runs/x/metrics.jsonl' to the Hub.")
    error.__cause__ = cause
    return error


def _not_located(cause: BaseException) -> LocalEntryNotFoundError:
    """A download whose metadata request failed, as huggingface_hub reports it."""
    error = LocalEntryNotFoundError("An error happened while trying to locate the file.")
    error.__cause__ = cause
    return error


def _chain(error: BaseException | None) -> list[BaseException]:
    found = []
    while error is not None:
        found.append(error)
        error = error.__cause__
    return found


@pytest.mark.phase(3)
@pytest.mark.parametrize(
    "error",
    [
        httpx2.ConnectError("connection refused"),
        httpx2.ReadTimeout("read timed out"),
        hub_error(503),
        hub_error(429),
        _lfs_failure(httpx2.ConnectError("connection reset")),
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

    # Downloads too, which huggingface_hub reports as LocalEntryNotFoundError: a blip is
    # survived, and an outage is no missing run.
    manifest = _manifest()
    store.put_manifest(manifest)
    waits.clear()
    fake.fail["hf_hub_download"] = [error, error]
    assert store.get_manifest(RUN) == manifest
    assert waits == [2.0, 4.0]
    waits.clear()
    fake.fail["hf_hub_download"] = [error] * 4
    with pytest.raises(StoreUnavailable, match="failed 4 times") as raised:
        store.get_manifest(RUN)
    assert error in _chain(raised.value) and not isinstance(raised.value, RunNotFound)
    assert waits == [2.0, 4.0, 8.0]
    waits.clear()
    fake.fail["snapshot_download"] = [error] * 4
    with pytest.raises(StoreUnavailable, match="failed 4 times") as raised:
        store.pull(RUN, tmp_path / "runs" / RUN)
    assert error in _chain(raised.value) and waits == [2.0, 4.0, 8.0]
    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == []
    fake.fail["snapshot_download"] = [error]
    store.pull(RUN, tmp_path / "runs" / RUN)
    assert (tmp_path / "runs" / RUN / "manifest.json").read_bytes() == manifest_bytes(manifest)


@pytest.mark.phase(3)
@pytest.mark.parametrize(
    "error",
    [
        connect_error(TREE_URL),
        hub_error(503, url=TREE_URL),
        dropped_error(TREE_URL),
        hub_error(429, url=f"{REPO_URL}/preupload/main"),
        _lfs_failure(connect_error("https://lfs.huggingface.co/upload/part-1", "PUT")),
        _lfs_failure(dropped_error("https://lfs.huggingface.co/upload/part-1", "PUT")),
        # A download's metadata request, which the client sends again with retries after
        # a refused connection, a timeout or one of its statuses.
        _not_located(connect_error(RESOLVE_URL, "HEAD")),
        _not_located(hub_error(503, url=RESOLVE_URL, method="HEAD")),
        # The download itself, which it retries on its own terms.
        connect_error(RESOLVE_URL),
        hub_error(429, url=RESOLVE_URL, method="GET"),
    ],
)
def test_hub_store_leaves_to_the_client_what_it_retried(error: BaseException) -> None:
    # huggingface_hub tried these requests 5 more times before it raised, so the store
    # tries no more; a commit or the repo lookup it sends once, and the store retries.
    fake = FakeHub()
    store = HubRunStore(REPO, api=fake, sleep=_never_sleep)
    fake.fail["list_repo_files"] = [error]
    with pytest.raises(StoreUnavailable, match="after huggingface_hub's own retries") as raised:
        store.runs()
    assert raised.value.__cause__ is error and len(fake.calls) == 1

    waits: list[float] = []
    store = HubRunStore(REPO, api=fake, sleep=waits.append)
    fake.fail["list_repo_files"] = [connect_error(f"{REPO_URL}/commit/main")] * 2
    assert store.runs() == []
    assert waits == [2.0, 4.0]


@pytest.mark.phase(3)
@pytest.mark.parametrize(
    "error",
    [
        # A dropped connection on a download's metadata request, which goes once.
        dropped_error(RESOLVE_URL, "HEAD"),
        httpx2.RemoteProtocolError(
            "server disconnected", request=httpx2.Request("HEAD", RESOLVE_URL)
        ),
        # A 5xx the client never retries, on the metadata request or any other.
        _not_located(hub_error(501, url=RESOLVE_URL, method="HEAD")),
        _not_located(hub_error(520, url=RESOLVE_URL, method="HEAD")),
        # The download itself, whose GET the client retries only on 408 and 429.
        dropped_error(RESOLVE_URL),
        hub_error(503, url=RESOLVE_URL, method="GET"),
    ],
)
def test_hub_store_retries_what_the_client_sent_once(error: BaseException) -> None:
    fake = FakeHub()
    waits: list[float] = []
    store = HubRunStore(REPO, api=fake, sleep=waits.append)
    manifest = _manifest()
    store.put_manifest(manifest)

    fake.fail["hf_hub_download"] = [error, error]
    assert store.get_manifest(RUN) == manifest
    assert waits == [2.0, 4.0]

    waits.clear()
    fake.fail["hf_hub_download"] = [error] * 4
    with pytest.raises(StoreUnavailable, match="failed 4 times") as raised:
        store.get_manifest(RUN)
    assert error in _chain(raised.value) and waits == [2.0, 4.0, 8.0]

    # Any other call's 5xx outside the client's list, too.
    waits.clear()
    fake.fail["list_repo_files"] = [hub_error(501, url=TREE_URL, method="GET")] * 2
    assert store.runs() == [RUN] and waits == [2.0, 4.0]


@pytest.mark.phase(3)
@pytest.mark.parametrize(
    "error",
    [
        hub_error(401, RepositoryNotFoundError),
        hub_error(403),
        OfflineModeIsEnabled("HF_HUB_OFFLINE is set"),
        ValueError("Invalid repo id"),
        # Not every RuntimeError is the network: a refused LFS upload, a bug.
        _lfs_failure(hub_error(403)),
        RuntimeError("Error while uploading 'runs/x/metrics.jsonl' to the Hub."),
        NotImplementedError("not built yet"),
        httpx2.ProxyError("407 Proxy Authentication Required"),
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
def test_a_retried_pull_starts_from_an_empty_folder(tmp_path: Path) -> None:
    class HalfwayHub(FakeHub):
        """Fails its first download halfway through, then works."""

        def __init__(self) -> None:
            super().__init__()
            self.found: list[list[str]] = []  # what each download found in its folder

        def snapshot_download(self, **kwargs: Any) -> str:
            root = Path(kwargs["local_dir"])
            self.found.append(sorted(p.relative_to(root).as_posix() for p in root.rglob("*")))
            if len(self.found) > 1:
                return super().snapshot_download(**kwargs)
            (root / "runs" / RUN / "eval").mkdir(parents=True)
            (root / "runs" / RUN / "eval" / "partial.jsonl").write_text("a line cut sh")
            raise LocalEntryNotFoundError("the Hub went away") from connect_error(REPO_URL)

    fake = HalfwayHub()
    store = HubRunStore(REPO, api=fake, sleep=lambda seconds: None)
    store.put_manifest(_manifest())

    store.pull(RUN, tmp_path / "runs" / RUN)

    # huggingface_hub returns a folder that isn't empty as it is when it can't reach the
    # Hub, so a second try on the first one's leftovers could pass for a whole run.
    assert fake.found == [[], []]
    assert _files(tmp_path / "runs" / RUN) == {"manifest.json": manifest_bytes(_manifest())}


@pytest.mark.phase(4)
def test_get_folder_fetches_one_folder_whole_from_an_empty_start(tmp_path: Path) -> None:
    class HalfwayHub(FakeHub):
        """Fails its first download halfway through, then works."""

        def __init__(self) -> None:
            super().__init__()
            self.found: list[list[str]] = []  # what each download found in its folder

        def snapshot_download(self, **kwargs: Any) -> str:
            root = Path(kwargs["local_dir"])
            self.found.append(sorted(p.relative_to(root).as_posix() for p in root.rglob("*")))
            if len(self.found) > 1:
                return super().snapshot_download(**kwargs)
            cut = root / "runs" / RUN / checkpoint_dir(2) / "cut.safetensors"
            cut.parent.mkdir(parents=True)
            cut.write_text("weights cut sh")
            raise LocalEntryNotFoundError("the Hub went away") from connect_error(REPO_URL)

    fake = HalfwayHub()
    store = HubRunStore(REPO, api=fake, sleep=lambda seconds: None)
    saved = _checkpoint(tmp_path, 2)
    store.put_folder(RUN, saved, checkpoint_dir(2))
    store.put_folder(RUN, _checkpoint(tmp_path, 4), checkpoint_dir(4))
    target = tmp_path / "work" / "runs" / RUN / checkpoint_dir(2)
    target.mkdir(parents=True)
    (target / "stale.txt").write_text("from before")

    assert store.get_folder(RUN, checkpoint_dir(2), target) == target

    # As in a pull, the second try starts from nothing, and the folder comes whole and
    # alone: the run's other checkpoint stays in the store.
    assert fake.found == [[], []]
    assert _files(target) == _files(saved)
    assert sorted(p.name for p in target.parent.iterdir()) == ["step-000002"]
    missing = tmp_path / "work" / "missing"
    with pytest.raises(RunNotFound, match=f"run {RUN} in {REPO} has no checkpoints/step-000003"):
        store.get_folder(RUN, checkpoint_dir(3), missing)
    assert not missing.exists()


Handler = Callable[[httpx2.Request], httpx2.Response]


@pytest.fixture
def answered(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[Handler], list[float]]]:
    """huggingface_hub for real, with every request it sends answered by a handler in
    this process. Returns a function that installs a handler and gives the list the
    client's own retries record their waits in."""
    waits: list[float] = []
    monkeypatch.setattr(_http, "time", SimpleNamespace(sleep=waits.append))

    def use(handler: Handler) -> list[float]:
        huggingface_hub.set_client_factory(
            lambda: httpx2.Client(transport=httpx2.MockTransport(handler))
        )
        return waits

    yield use
    huggingface_hub.set_client_factory(_http.default_client_factory)


def _refused(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ConnectError("connection refused", request=request)


def _unavailable(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(503, request=request)


def _dropped(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ReadError("connection reset by peer", request=request)


def _disconnected(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.RemoteProtocolError("server disconnected", request=request)


def _not_implemented(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(501, request=request)


def _rate_limited(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(429, headers={"Retry-After": "60"}, request=request)


def _busy(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(503, headers={"Retry-After": "30"}, request=request)


def _commit_refused(request: httpx2.Request) -> httpx2.Response:
    """A Hub that takes the upload's preupload request and drops its commit."""
    if "/preupload/" in request.url.path:
        mode = {"path": f"runs/{RUN}/metrics.jsonl", "uploadMode": "regular", "shouldIgnore": False}
        return httpx2.Response(200, json={"files": [mode]}, request=request)
    return _refused(request)


CLIENT_WAITS = [1, 2, 4, 8, 8]  # huggingface_hub 2.4's own, between its 6 tries
STORE_WAITS = [2.0, 4.0, 8.0]  # the store's, between its 4


@pytest.mark.phase(3)
@pytest.mark.parametrize(
    ("call", "handler", "client_waits", "store_waits"),
    [
        ("runs", _refused, CLIENT_WAITS, []),
        ("runs", _unavailable, CLIENT_WAITS, []),
        ("get_manifest", _refused, CLIENT_WAITS, []),
        ("get_manifest", _unavailable, CLIENT_WAITS, []),
        ("put_file", _refused, CLIENT_WAITS, []),
        ("put_file", _commit_refused, [], STORE_WAITS),
        ("pull", _refused, [], STORE_WAITS),
        ("pull", _unavailable, [], STORE_WAITS),
        # The listing retries a dropped connection; a download's metadata request doesn't.
        ("runs", _dropped, CLIENT_WAITS, []),
        ("get_manifest", _dropped, [], STORE_WAITS),
        ("get_manifest", _disconnected, [], STORE_WAITS),
        # A 5xx outside the client's 500, 502, 503 and 504.
        ("runs", _not_implemented, [], STORE_WAITS),
        ("get_manifest", _not_implemented, [], STORE_WAITS),
        # The Hub says how long to wait, and the client waits that and 1 s more each time.
        ("runs", _rate_limited, [61.0] * 5, []),
        ("get_manifest", _busy, [31.0] * 5, []),
    ],
)
def test_store_and_client_retries_never_stack(
    answered: Callable[[Handler], list[float]],
    tmp_path: Path,
    call: str,
    handler: Handler,
    client_waits: list[float],
    store_waits: list[float],
) -> None:
    # The real client, so what it retries itself is measured rather than assumed.
    client = answered(handler)
    waits: list[float] = []
    api = HfApi(endpoint="https://hub.test", token=False)
    store = HubRunStore(REPO, api=api, sleep=waits.append)
    (tmp_path / "metrics.jsonl").write_text("{}\n")
    calls: dict[str, Callable[[], object]] = {
        "runs": store.runs,
        "get_manifest": lambda: store.get_manifest(RUN),
        "put_file": lambda: store.put_file(RUN, tmp_path / "metrics.jsonl", "metrics.jsonl"),
        "pull": lambda: store.pull(RUN, tmp_path / "runs" / RUN),
    }

    with pytest.raises(StoreUnavailable):
        calls[call]()

    # Each failure is retried by one of the two, so a call waits 14 s at most between the
    # store's tries, or 23 s inside the client's unless the Hub's answer names a wait, as
    # the module docstring says.
    assert (client, waits) == (client_waits, store_waits)


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


@pytest.mark.phase(3)
def test_a_pull_leaves_no_file_the_store_lacks(store: RunStore, tmp_path: Path) -> None:
    run = tmp_path / "finished"
    (run / "eval").mkdir(parents=True)
    (run / "manifest.json").write_bytes(manifest_bytes(_manifest()))
    (run / "eval" / "val-scores.jsonl").write_text("{}\n{}\n")
    store.put_folder(RUN, run)
    # This machine's copy is the first attempt's, which failed.
    local = tmp_path / "runs" / RUN
    (local / "eval").mkdir(parents=True)
    (local / "error.txt").write_text("Traceback: the first attempt failed\n")
    (local / "eval" / "val-scores.jsonl").write_text("{}\n")
    (local / "eval" / "val-report.md").write_text("a report of the first attempt\n")
    # What the store's copy lacks and only this machine has: the kernels it pushed, their
    # output, and its lock.
    (local / "kernels.json").write_text('{"kernels": []}\n')
    (local / "kernel").mkdir()
    (local / "kernel" / "forge-run.log").write_text("log\n")
    (local / ".lock").write_text("")
    beside = tmp_path / "runs" / OTHER
    beside.mkdir()
    (beside / "error.txt").write_text("another run's\n")

    store.pull(RUN, local)

    assert _files(local) == {
        ".lock": b"",
        "eval/val-scores.jsonl": b"{}\n{}\n",
        "kernel/forge-run.log": b"log\n",
        "kernels.json": b'{"kernels": []}\n',
        "manifest.json": manifest_bytes(_manifest()),
    }
    assert _files(beside) == {"error.txt": b"another run's\n"}


def _kernels(*names: str) -> str:
    """A kernels.json as forge kaggle run writes it, with a push for each kernel named."""
    pushes = [{"kernel": f"owner/{name}", "version": 1} for name in names]
    return json.dumps({"run_id": RUN, "job": "eval", "kernels": pushes}) + "\n"


@pytest.mark.phase(3)
def test_a_pull_never_puts_back_an_older_record_of_kernels(store: RunStore, tmp_path: Path) -> None:
    # forge runs push uploads the record of kernels and their output with the run.
    local = tmp_path / "runs" / RUN
    (local / "kernel").mkdir(parents=True)
    (local / "manifest.json").write_bytes(manifest_bytes(_manifest()))
    (local / "kernels.json").write_text(_kernels("forge-k1"))
    (local / "kernel" / "forge-k1.log").write_text("k1\n")
    store.put_folder(RUN, local, replace=True)
    # Then a resume pushes a second kernel from here, and forge kaggle output fetches it.
    (local / "kernels.json").write_text(_kernels("forge-k1", "forge-k2"))
    shutil.rmtree(local / "kernel")
    (local / "kernel").mkdir()
    (local / "kernel" / "forge-k2.log").write_text("k2\n")

    store.pull(RUN, local)

    assert _files(local) == {
        "kernel/forge-k2.log": b"k2\n",
        "kernels.json": _kernels("forge-k1", "forge-k2").encode(),
        "manifest.json": manifest_bytes(_manifest()),
    }
    # A machine with an older record, or none, takes the store's.
    for name, record in (("older", _kernels()), ("fresh", None)):
        other = tmp_path / name / RUN
        other.mkdir(parents=True)
        if record is not None:
            (other / "kernels.json").write_text(record)
        store.pull(RUN, other)
        assert _files(other) == {
            "kernel/forge-k1.log": b"k1\n",
            "kernels.json": _kernels("forge-k1").encode(),
            "manifest.json": manifest_bytes(_manifest()),
        }, name
    # A record that is no record is named, not taken as either.
    broken = tmp_path / "broken" / RUN
    broken.mkdir(parents=True)
    (broken / "kernels.json").write_text("[]\n")
    with pytest.raises(RunCorrupt, match="is not forge kaggle run's record of kernels"):
        store.pull(RUN, broken)
