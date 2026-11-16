"""``forge eval run --store hub`` on the fixture data: the run's folder goes to the Hub run
store when the run ends, and a run another machine started resumes from it (FG-304)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from typer.testing import CliRunner

from forge import cli
from forge.core.types import GenParams, Prompt, RunManifest
from forge.data.splits import load_val
from forge.eval.config import EvalConfig
from forge.eval.runner import run_eval
from forge.infer.base import FakeBackend
from forge.ops.hub import HubRunStore
from forge.ops.manifest import CODE_SHA_ENV
from tests.fakes import FakeHub, hub_error

REPO = "owner/forge-runs"


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dev_log: Path) -> Path:
    """Runs and the gold cache under tmp_path, the run store named, and no token."""
    monkeypatch.setenv("FORGE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("FORGE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("FORGE_RUN_STORE", REPO)
    # The run store's settings come from here, never from a real .env.
    monkeypatch.setenv("FORGE_ROOT", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    # tmp_path is no checkout, so the code's commit is named as a kernel names it. Built at
    # run time, so no literal here looks like a real commit.
    monkeypatch.setenv(CODE_SHA_ENV, "".join("0123456789abcdef"[(i * 7) % 16] for i in range(40)))
    return tmp_path / "runs"


def _hub_run(hub: FakeHub, run_id: str) -> dict[str, bytes]:
    prefix = f"runs/{run_id}/"
    return {
        name.removeprefix(prefix): data
        for name, data in hub.files.items()
        if name.startswith(prefix)
    }


def _local_run(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != ".lock"
    }


@pytest.mark.phase(3)
def test_eval_run_store_hub_puts_the_run_in_the_hub(
    spider_mini: Path, runs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = FakeHub()
    monkeypatch.setattr("huggingface_hub.HfApi", hub.client)
    args = ["eval", "run", "--backend", "fake", "--model", "gold", "--limit", "4"]
    args += ["--store", "hub", "--run-id", "hub-run", "--workers", "1", "--allow-dirty"]

    result = CliRunner().invoke(cli.app, args)

    assert result.exit_code == 0, result.output
    assert f"run store {REPO}: runs/hub-run\n" in result.stdout
    stored = _hub_run(hub, "hub-run")
    # The whole folder, as it was when the run ended, and never its lock.
    assert stored == _local_run(runs / "hub-run")
    assert sorted(stored) == [
        "config.yaml",
        "eval/val-generations.jsonl",
        "eval/val-scores.jsonl",
        "manifest.json",
        "sessions.json",
    ]
    assert RunManifest.model_validate_json(stored["manifest.json"]).status == "finished"
    # It looked for an earlier attempt first, then uploaded once, with no token to give.
    assert [method for method, _ in hub.calls] == ["snapshot_download", "upload_folder"]
    assert hub.tokens == [None]


@pytest.mark.phase(3)
def test_a_failed_run_resumes_from_the_hub_on_another_machine(
    spider_mini: Path, tmp_path: Path, runs: Path
) -> None:
    val = sorted(load_val(), key=lambda example: example.id)
    gold = {example.id: example.gold_sql for example in val}
    hub = FakeHub()
    store = HubRunStore(REPO, api=hub, sleep=lambda seconds: None)

    def reply(prompt: Prompt, params: GenParams, sample_idx: int) -> str:
        return f"```sql\n{gold[prompt.example_id]}\n```"

    def breaks_on_sixth(prompt: Prompt, params: GenParams, sample_idx: int) -> str:
        if prompt.example_id == val[5].id:
            raise RuntimeError("the session ended")
        return reply(prompt, params, sample_idx)

    config = EvalConfig(model="gold", backend="fake", run_id="moved", store="hub")
    with pytest.raises(RuntimeError, match="the session ended"):
        run_eval(
            config,
            backend=FakeBackend(breaks_on_sixth, model_ref="gold"),
            allow_dirty=True,
            runs_root=tmp_path / "first-machine",
            workers=1,
            batch_size=4,
            store=store,
        )
    failed = RunManifest.model_validate_json(hub.files["runs/moved/manifest.json"])
    assert failed.status == "failed" and "error.txt" in _hub_run(hub, "moved")

    second = tmp_path / "second-machine"
    run = run_eval(
        config,
        backend=FakeBackend(reply, model_ref="gold"),
        allow_dirty=True,
        runs_root=second,
        workers=1,
        store=store,
    )

    assert run.resumed == 4 and run.manifest.status == "finished"
    assert [score.example_id for score in run.scores] == [example.id for example in val]
    assert run.manifest.started_at == failed.started_at
    assert run.stored == REPO
    assert _hub_run(hub, "moved") == _local_run(second / "moved")
    assert "error.txt" not in _hub_run(hub, "moved")


@pytest.mark.phase(3)
def test_a_run_the_hub_cannot_take_says_how_to_push_it(
    spider_mini: Path,
    runs: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    hub = FakeHub()
    store = HubRunStore(REPO, api=hub, sleep=lambda seconds: None)
    hub.fail["upload_folder"] = [hub_error(403)]

    config = EvalConfig(model="gold", backend="fake", run_id="refused", limit=2, store="hub")
    with pytest.raises(OSError, match="upload it with forge runs push refused") as raised:
        run_eval(config, allow_dirty=True, runs_root=runs, workers=1, store=store)

    assert str(raised.value).startswith(
        f"run refused is in {runs / 'refused'}, but storing it in {REPO} failed: 403"
    )
    finished = RunManifest.model_validate_json((runs / "refused" / "manifest.json").read_bytes())
    assert finished.status == "finished" and hub.files == {}
    monkeypatch.setattr("huggingface_hub.HfApi", hub.client)
    result = CliRunner().invoke(cli.app, ["runs", "push", "refused"])
    assert result.exit_code == 0, result.output
    assert _hub_run(hub, "refused") == _local_run(runs / "refused")

    # A failed run whose upload fails too raises its own error, and logs the other.
    def falls_over(prompt: Prompt, params: GenParams, sample_idx: int) -> str:
        raise RuntimeError("the backend fell over")

    hub.fail["upload_folder"] = [hub_error(503)] * 4
    with (
        caplog.at_level(logging.WARNING, logger="forge.eval.runner"),
        pytest.raises(RuntimeError, match="the backend fell over"),
    ):
        run_eval(
            EvalConfig(model="gold", backend="fake", run_id="both", store="hub"),
            backend=FakeBackend(falls_over, model_ref="gold"),
            allow_dirty=True,
            runs_root=runs,
            workers=1,
            store=store,
        )
    assert "run both is in" in caplog.text and "failed 4 times" in caplog.text
    assert "runs/both/manifest.json" not in hub.files
