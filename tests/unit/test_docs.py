from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

REQUIRED = [
    "docs/PLAN.md",
    "CONTRIBUTING.md",
    "docs/adr/0001-record-decisions.md",
    "docs/adr/0002-dependency-pins.md",
    "docs/progress.md",
    "docs/compute-ledger.md",
    "docs/dev-usage.md",
    "docs/runbook.md",
    ".github/pull_request_template.md",
]


@pytest.mark.phase(0)
def test_required_docs_present() -> None:
    missing = [p for p in REQUIRED if not (ROOT / p).is_file()]
    assert not missing, f"missing docs: {missing}"
    assert "#### FG-001 · Repo scaffold and toolchain" in (ROOT / "docs/PLAN.md").read_text()


@pytest.mark.phase(0)
def test_progress_lists_every_plan_story() -> None:
    plan = (ROOT / "docs/PLAN.md").read_text()
    progress = (ROOT / "docs/progress.md").read_text()
    story_ids = {line.split()[1] for line in plan.splitlines() if line.startswith("#### FG-")}
    assert story_ids, "no stories found in the plan"
    missing = sorted(s for s in story_ids if f"| {s} |" not in progress)
    assert not missing, f"stories missing from progress.md: {missing}"
