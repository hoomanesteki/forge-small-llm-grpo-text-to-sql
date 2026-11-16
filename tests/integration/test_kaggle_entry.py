"""kaggle/entry.py running a real eval job on the fixture data, with pip stood in for.

The job is what a kernel runs: ``run_eval`` with the job's options, a preflight on five
examples first, and the report at the end, outside any git checkout, so the manifest's
commit comes from the code dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.core.types import RunManifest
from forge.eval.config import EvalConfig
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
