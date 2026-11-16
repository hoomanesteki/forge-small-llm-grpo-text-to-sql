"""kaggle/entry.py running a real eval job on the fixture data, with pip stood in for.

The job is what a kernel runs: ``run_eval`` with the job's options, a preflight on five
examples first, and the report at the end, outside any git checkout, so the manifest's
commit comes from the code dataset. A vLLM job runs with vLLM faked.
"""

from __future__ import annotations

import functools
import json
from datetime import datetime
from pathlib import Path

import pytest

from forge.core.types import RunManifest, Score
from forge.data.splits import load_val
from forge.eval import runner
from forge.eval.config import EvalConfig
from forge.ops.dev_usage import read_uses
from forge.ops.sessions import read_sessions
from tests.fakes import FakeHub, FakeVllm, gold_reply
from tests.kernel import SHA, argv, build_inputs, kernel_folder, mount_output
from tests.spike import entry

script = entry()
# Built at run time, so no literal here looks like a real token.
TOKEN = "hf_" + "".join(chr(97 + (i * 11) % 26) for i in range(30))


def _scores(run_dir: Path, split: str = "val") -> list[Score]:
    lines = (run_dir / "eval" / f"{split}-scores.jsonl").read_text().splitlines()
    return [Score.model_validate_json(line) for line in lines]


@pytest.mark.phase(3)
def test_entry_runs_an_eval_job_end_to_end(
    tmp_path: Path, spider_mini: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel_env = kernel_folder(tmp_path, monkeypatch)
    build_inputs(kernel_env, spider_mini, zipped=True)
    options = {"model": "gold", "backend": "fake", "limit": 6, "allow_over_budget": True}

    assert script.main(argv(kernel_env, options=options), run=lambda command: None) == 0

    work = kernel_env / "working"
    status = json.loads((work / "runs" / "run-1" / "status.json").read_text())
    assert (status["status"], status["error"]) == ("finished", None), status["error"]
    run = RunManifest.model_validate_json((work / "runs" / "run-1" / "manifest.json").read_text())
    assert (run.status, run.git_sha, run.platform) == ("finished", SHA, "kaggle")
    laptop = EvalConfig(model="gold", backend="fake", limit=6, host="kaggle")
    assert run.config == {**laptop.manifest_config(), "allow_over_budget": True}
    assert (work / "runs" / "run-1" / "eval" / "val-report.md").is_file()
    scores = (work / "runs" / "run-1" / "eval" / "val-scores.jsonl").read_text().splitlines()
    assert len(scores) == 6
    # The run's one session is the whole kernel's, which held the GPU from start to end.
    (session,) = read_sessions(work / "runs" / "run-1") or []
    assert (session.status, status["session"]) == ("finished", 1)
    assert session.started_at == datetime.fromisoformat(status["started_at"])
    assert session.ended_at == datetime.fromisoformat(status["ended_at"])
    assert run.started_at >= session.started_at and run.gpu_hours > 0

    # The preflight ran the same job on five examples, in a run of its own.
    preflight = work / "preflight" / "run-1-preflight"
    first = RunManifest.model_validate_json((preflight / "manifest.json").read_text())
    assert (first.status, first.config["limit"], first.git_sha) == ("finished", 5, SHA)
    assert len((preflight / "eval" / "val-scores.jsonl").read_text().splitlines()) == 5


@pytest.mark.phase(3)
def test_entry_runs_a_vllm_job_on_one_engine(
    tmp_path: Path, spider_mini: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # vLLM faked. A second engine might not fit on the T4 beside the preflight's, so the
    # preflight and the job share the one the preflight loaded.
    vllm = FakeVllm(gold_reply(load_val(spider_mini))).install(monkeypatch)
    kernel_env = kernel_folder(tmp_path, monkeypatch)
    build_inputs(kernel_env, spider_mini)
    options = {"model": "org/model@abc123", "backend": "vllm", "limit": 6}

    assert script.main(argv(kernel_env, options=options), run=lambda command: None) == 0

    work = kernel_env / "working"
    status = json.loads((work / "runs" / "run-1" / "status.json").read_text())
    assert (status["status"], status["error"]) == ("finished", None), status["error"]
    assert len(vllm.engines) == 1
    # Five preflight prompts, then the job's six, one request a batch.
    assert [len(call["sampling_params"]) for _, call in vllm.calls] == [5, 6]
    run = RunManifest.model_validate_json((work / "runs" / "run-1" / "manifest.json").read_text())
    assert (run.status, run.platform, run.config["backend"]) == ("finished", "kaggle", "vllm")
    scores = (work / "runs" / "run-1" / "eval" / "val-scores.jsonl").read_text().splitlines()
    assert len(scores) == 6
    preflight = work / "preflight" / "run-1-preflight" / "manifest.json"
    assert RunManifest.model_validate_json(preflight.read_text()).status == "finished"


@pytest.mark.phase(3)
def test_eval_job_stops_at_its_deadline_then_uploads_and_resumes(
    tmp_path: Path, spider_mini: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [0.0]
    gold = gold_reply(load_val(spider_mini))

    def slow(text: str, sample_idx: int) -> str:
        now[0] += 10.0  # each reply takes 10 s of the kernel's time
        return gold(text, sample_idx)

    FakeVllm(slow).install(monkeypatch)
    hub = FakeHub()
    monkeypatch.setattr("huggingface_hub.HfApi", hub.client)
    # Two questions a batch, so the guard has batches to stop between.
    monkeypatch.setattr(runner, "run_eval", functools.partial(runner.run_eval, batch_size=2))
    options = {"model": "org/model@abc123", "backend": "vllm", "limit": 6}
    first = kernel_folder(tmp_path / "first", monkeypatch)
    build_inputs(first, spider_mini)
    monkeypatch.setenv("HF_TOKEN", TOKEN)

    # Pushed with a two-minute timeout. The preflight's five replies take 50 s, then each
    # batch 20 s: after two, 30 s are left, within a batch and the 12 s kept back.
    kernel_argv = argv(first, "--timeout-s", "120", options=options)
    assert script.main(kernel_argv, run=lambda c: None, clock=lambda: now[0]) == 0

    run_dir = first / "working" / "runs" / "run-1"
    status = json.loads((run_dir / "status.json").read_text())
    assert (status["status"], status["stage"], status["error"]) == (
        "stopped_time_guard",
        "upload",
        None,
    )
    assert status["stages"]["job"]["result"] == {"status": "stopped_time_guard"}
    assert status["stages"]["upload"]["result"] == "uploaded to forge-owner/forge-runs"
    stopped = RunManifest.model_validate_json((run_dir / "manifest.json").read_text())
    assert stopped.status == "stopped_time_guard"
    assert [s.ex_official for s in _scores(run_dir)] == [None] * 4
    # A stopped run gets its report once it finishes; the store has it, ready to resume.
    assert not (run_dir / "eval" / "val-report.md").exists()
    uploaded = RunManifest.model_validate_json(hub.files["runs/run-1/manifest.json"])
    assert uploaded.status == "stopped_time_guard"
    assert len(hub.files["runs/run-1/eval/val-scores.jsonl"].splitlines()) == 4

    second = kernel_folder(tmp_path / "second", monkeypatch)
    build_inputs(second, spider_mini)
    mount_output(first, second, "forge-run-1")
    resume = argv(second, "--timeout-s", "3600", "--resume", options=options)
    assert script.main(resume, run=lambda c: None, clock=lambda: now[0]) == 0

    run_dir = second / "working" / "runs" / "run-1"
    status = json.loads((run_dir / "status.json").read_text())
    assert (status["status"], status["error"], status["session"]) == ("finished", None, 2)
    scores = _scores(run_dir)
    assert len(scores) == 6 and all(s.ex_fast and s.ex_official for s in scores)
    assert (run_dir / "eval" / "val-report.md").is_file()
    sessions = read_sessions(run_dir) or []
    assert [s.status for s in sessions] == ["stopped_time_guard", "finished"]


@pytest.mark.phase(3)
def test_entry_loads_dev_once_in_a_dev_job(
    tmp_path: Path, with_dev: Path, dev_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel_env = kernel_folder(tmp_path, monkeypatch)
    build_inputs(kernel_env, with_dev)
    options = {"model": "gold", "backend": "fake", "split": "dev", "reason": "baseline"}

    assert script.main(argv(kernel_env, options=options), run=lambda command: None) == 0

    work = kernel_env / "working"
    status = json.loads((work / "runs" / "run-1" / "status.json").read_text())
    assert (status["status"], status["error"]) == ("finished", None), status["error"]
    run = RunManifest.model_validate_json((work / "runs" / "run-1" / "manifest.json").read_text())
    assert (run.status, run.config["split"]) == ("finished", "dev")
    assert (work / "runs" / "run-1" / "eval" / "dev-report.md").is_file()
    # The preflight answered val, so the job's load was the one look, logged in the
    # kernel's output; the repo's log is forge kaggle run's to write, at the push.
    preflight = work / "preflight" / "run-1-preflight" / "manifest.json"
    first = RunManifest.model_validate_json(preflight.read_text())
    assert (first.status, first.config["split"], first.config["limit"]) == ("finished", "val", 5)
    assert [(run_id, reason) for _, run_id, reason in read_uses(work / "dev-usage.md")] == [
        ("run-1", "baseline")
    ]
    assert not dev_log.exists()
