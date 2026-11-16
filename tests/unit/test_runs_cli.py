"""``forge runs list|show|pull|push``, with the Hub client stood in for by ``FakeHub``."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from forge import cli
from forge.core.types import RunManifest
from forge.ops import kaggle_job
from forge.ops.hub import manifest_bytes, open_store
from tests.fakes import FakeHub, hub_error

RUN = "2026-11-15-qwen3-4b-eval-abc1234"
REPO = "owner/forge-runs"


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    """A Hub run store named by FORGE_RUN_STORE, runs/ under tmp_path, and no token."""
    fake = FakeHub()
    monkeypatch.setattr("huggingface_hub.HfApi", fake.client)
    monkeypatch.setenv("FORGE_RUN_STORE", REPO)
    monkeypatch.setenv("FORGE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("FORGE_ROOT", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return fake


def _manifest(run_id: str = RUN, **changes: Any) -> RunManifest:
    fields: dict[str, Any] = {
        "run_id": run_id,
        "kind": "eval",
        "git_sha": "b" * 40,
        "config": {"split": "val"},
        "config_hash": "2" * 64,
        "data_manifest_hash": "3" * 64,
        "model_ref": "Qwen/Qwen3-4B",
        "platform": "kaggle",
        "gpu": "T4",
        "seed": 0,
        "started_at": datetime(2026, 11, 15, 9, 0, tzinfo=UTC),
        "ended_at": datetime(2026, 11, 15, 9, 40, tzinfo=UTC),
        "gpu_hours": 0.67,
        "status": "finished",
        "outputs": {"scores": "eval/val-scores.jsonl", "config": "config.yaml"},
        **changes,
    }
    return RunManifest(**fields)


def _in_hub(hub: FakeHub, manifest: RunManifest, files: dict[str, bytes] | None = None) -> None:
    """Put the run in the fake repo: its manifest, and ``files`` by path inside it."""
    hub.files[f"runs/{manifest.run_id}/manifest.json"] = manifest_bytes(manifest)
    for name, data in (files or {}).items():
        hub.files[f"runs/{manifest.run_id}/{name}"] = data


def _invoke(*args: str) -> Any:
    return CliRunner().invoke(cli.app, ["runs", *args])


@pytest.mark.phase(3)
def test_list_prints_the_ids_of_each_store(hub: FakeHub, tmp_path: Path) -> None:
    _in_hub(hub, _manifest())
    _in_hub(hub, _manifest("2026-11-14-qwen3-1-7b-eval-abc1234"))
    hub.files["runs/2026-11-16-half-written/eval/val-scores.jsonl"] = b"{}\n"
    local = tmp_path / "runs" / "2026-11-13-pilot-eval-abc1234"
    local.mkdir(parents=True)
    (local / "manifest.json").write_text("{}")
    (tmp_path / "runs" / "no-manifest").mkdir()

    result = _invoke("list")
    assert result.exit_code == 0, result.output
    assert result.stdout == f"2026-11-14-qwen3-1-7b-eval-abc1234\n{RUN}\n"
    assert result.stderr == f"2 runs in {REPO}\n"
    # The run store's client finds its own token: none is needed to build it.
    assert hub.tokens == [None]

    result = _invoke("list", "--store", "local")
    assert result.exit_code == 0, result.output
    assert result.stdout == "2026-11-13-pilot-eval-abc1234\n"
    with pytest.raises(ValueError, match="a run store is hub or local, not 'kaggle'"):
        open_store("kaggle")


@pytest.mark.phase(3)
def test_show_prints_the_manifest_and_latest_checkpoint(hub: FakeHub, tmp_path: Path) -> None:
    grpo = _manifest(kind="grpo", adapter_ref="runs/sft/checkpoints/best", status="running")
    _in_hub(
        hub,
        grpo,
        {
            "checkpoints/step-000050/adapter_model.safetensors": b"a",
            "checkpoints/step-000100/adapter_model.safetensors": b"b",
            "checkpoints/best/adapter_model.safetensors": b"c",
        },
    )

    result = _invoke("show", RUN)

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        f"run {RUN}: grpo, running",
        "model Qwen/Qwen3-4B, adapter runs/sft/checkpoints/best",
        "platform kaggle on T4, 0.67 GPU h, $0.00",
        f"commit {'b' * 40}, seed 0",
        f"config hash {'2' * 64}, data manifest hash {'3' * 64}",
        "started 2026-11-15T09:00:00+00:00, ended 2026-11-15T09:40:00+00:00",
        "outputs:",
        "  config: config.yaml",
        "  scores: eval/val-scores.jsonl",
        "latest checkpoint: checkpoints/step-000100",
    ]

    result = _invoke("show", "2026-01-01-missing-eval-abc1234")
    assert result.exit_code == 1
    assert result.stderr == (
        f"error: no run 2026-01-01-missing-eval-abc1234 in {REPO}: it has no "
        "runs/2026-01-01-missing-eval-abc1234/manifest.json\n"
    )


@pytest.mark.phase(3)
def test_pull_mirrors_the_hub_run_into_runs(hub: FakeHub, tmp_path: Path) -> None:
    manifest = _manifest()
    _in_hub(hub, manifest, {"eval/val-scores.jsonl": b"{}\n"})
    local = tmp_path / "runs" / RUN
    local.mkdir(parents=True)
    (local / "kernels.json").write_text('{"kernels": []}\n')

    result = _invoke("pull", RUN)

    assert result.exit_code == 0, result.output
    assert result.stdout == f"run {RUN}: {local}, from {REPO}\neval run, finished\n"
    assert sorted(p.relative_to(local).as_posix() for p in local.rglob("*") if p.is_file()) == [
        "eval/val-scores.jsonl",
        "kernels.json",
        "manifest.json",
    ]
    assert (local / "manifest.json").read_bytes() == manifest_bytes(manifest)
    assert sorted(p.name for p in local.parent.iterdir()) == [RUN]

    result = _invoke("pull", "2026-01-01-missing-eval-abc1234")
    assert result.exit_code == 1
    assert result.stderr == f"error: no run 2026-01-01-missing-eval-abc1234 in {REPO}\n"


@pytest.mark.phase(3)
def test_pull_from_kaggle_is_forge_kaggle_output(
    hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    (runs / RUN).mkdir(parents=True)
    record = {"run_id": RUN, "job": "eval", "kernels": [{"kernel": "o/forge-x", "version": 3}]}
    (runs / RUN / "kernels.json").write_text(json.dumps(record))
    fetched: list[tuple[str, str, Path]] = []

    def fetch_output(client: object, kernel: str, run_id: str, runs_root: Path) -> Path:
        fetched.append((kernel, run_id, runs_root))
        return runs_root / run_id

    monkeypatch.setattr(kaggle_job, "fetch_output", fetch_output)

    result = _invoke("pull", RUN, "--store", "kaggle")

    assert result.exit_code == 0, result.output
    assert fetched == [("o/forge-x/3", RUN, runs)]
    assert result.stdout.startswith(f"run {RUN}: {runs / RUN}; the kernel's log")
    assert hub.calls == []


@pytest.mark.phase(3)
def test_push_uploads_a_local_run(hub: FakeHub, tmp_path: Path) -> None:
    run = tmp_path / "runs" / RUN
    (run / "eval").mkdir(parents=True)
    (run / "manifest.json").write_bytes(manifest_bytes(_manifest()))
    (run / "eval" / "val-scores.jsonl").write_text("{}\n")
    (run / ".lock").write_text("")

    result = _invoke("push", RUN)

    assert result.exit_code == 0, result.output
    assert result.stdout == f"run {RUN} (eval, finished): uploaded to {REPO} as runs/{RUN}\n"
    assert sorted(hub.files) == [f"runs/{RUN}/eval/val-scores.jsonl", f"runs/{RUN}/manifest.json"]

    # A run with no manifest isn't one, and a Hub that refuses says so.
    (tmp_path / "runs" / "not-a-run").mkdir()
    result = _invoke("push", "not-a-run")
    assert result.exit_code == 1 and "no run not-a-run in" in result.stderr
    hub.fail["upload_folder"] = [hub_error(403)]
    result = _invoke("push", RUN)
    assert result.exit_code == 1 and result.stderr == "error: 403 from the Hub\n"
