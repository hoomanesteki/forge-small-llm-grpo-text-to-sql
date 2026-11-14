from __future__ import annotations

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
        assert jobs[name]["strategy"]["matrix"]["python"] == ["3.11", "3.12"]
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
