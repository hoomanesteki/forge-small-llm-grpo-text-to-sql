"""kaggle/entry.py running a real eval job on the fixture data, with pip stood in for.

The job is what a kernel runs: ``run_eval`` with the job's options, a preflight on five
examples first, and the report at the end, outside any git checkout, so the manifest's
commit comes from the code dataset. A vLLM job runs with vLLM faked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.core.types import RunManifest
from forge.data.splits import load_val
from forge.eval.config import EvalConfig
from forge.ops.dev_usage import read_uses
from tests.fakes import FakeVllm, gold_reply
from tests.kernel import SHA, argv, build_inputs, kernel_folder
from tests.spike import entry

script = entry()


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
