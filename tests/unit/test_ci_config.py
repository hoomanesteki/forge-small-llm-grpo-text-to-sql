from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _load() -> dict[str, Any]:
    data = yaml.safe_load(WORKFLOW.read_text())
    # PyYAML reads the bare key `on` as boolean True.
    data["on"] = data.pop(True, data.get("on"))
    return data


@pytest.mark.phase(0)
def test_workflow_has_required_jobs() -> None:
    workflow = _load()
    jobs = workflow["jobs"]
    assert {"lint-type", "unit", "integration", "security"} <= set(jobs)
    for name in ("unit", "integration"):
        # 3.13 is the Python of Kaggle's GPU image (FG-303).
        assert jobs[name]["strategy"]["matrix"]["python"] == ["3.11", "3.12", "3.13"]
    assert "pull_request" in workflow["on"]


@pytest.mark.phase(0)
def test_protection_script_lists_every_ci_check() -> None:
    script = (ROOT / "scripts" / "protect_branches.sh").read_text()
    workflow = _load()
    for job_id, job in workflow["jobs"].items():
        matrix = job.get("strategy", {}).get("matrix", {}).get("python")
        names = [f"{job_id} ({py})" for py in matrix] if matrix else [job_id]
        for name in names:
            assert f'"{name}"' in script, f"{name} is not a required check"


@pytest.mark.phase(2)
def test_smoke_job_runs_make_smoke_with_the_tiny_model_cached() -> None:
    job = _load()["jobs"]["smoke"]
    steps = job["steps"]
    commands = [step.get("run", "") for step in steps]
    # The plan's bound for the whole CPU pipeline, setup included.
    assert job["timeout-minutes"] <= 10
    setup = next(s for s in steps if s.get("uses", "").startswith("astral-sh/setup-uv@"))
    assert setup["with"]["python-version"] == "3.12"
    assert commands.index("uv sync --locked --extra cpu") < commands.index("make smoke")
    assert commands[-1] == 'test -f ".smoke-ok-$(git rev-parse HEAD)"'

    # The Hugging Face cache is keyed on the tiny model's pinned revision, read before
    # the cache is restored and the smoke downloads anything.
    read = next(i for i, s in enumerate(steps) if s.get("id") == "tiny")
    cache = next(i for i, s in enumerate(steps) if s.get("uses", "").startswith("actions/cache@"))
    assert read < cache < commands.index("make smoke")
    assert steps[cache]["with"]["path"] == "~/.cache/huggingface"
    assert "${{ steps.tiny.outputs.revision }}" in steps[cache]["with"]["key"]
    *command, program, append, output = shlex.split(commands[read])
    assert (command, append, output) == (["uv", "run", "python", "-c"], ">>", "$GITHUB_OUTPUT")
    printed = subprocess.run(
        [sys.executable, "-c", program], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    tiny = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))["tiny"]
    assert printed == f"revision={tiny['revision']}\n"
