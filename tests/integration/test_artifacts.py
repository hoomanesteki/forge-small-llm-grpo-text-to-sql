"""Runs committed under docs/runs/: whole, made as their configs say, and rebuildable."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from forge.data.spider import FAILING_GOLD_IDS, resolve_data_dir
from forge.data.splits import read_split_ids
from forge.eval.config import EvalConfig, read_config_file
from forge.eval.report import rebuild
from forge.eval.run_files import read_run
from forge.ops.ledger import read_rows
from tests.report_trace import untraced

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "docs" / "runs"
# The local pilot of FG-291 (ADR-0016), made from the commit its id ends with.
PILOT_RUN = "2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7"
PILOT_CONFIG = ROOT / "configs" / "eval" / "pilot-qwen25coder05b-val.yaml"


@pytest.mark.phase(2)
def test_pilot_run_is_committed_whole() -> None:
    # read_run refuses a run that isn't finished, whose config doesn't hash to the
    # manifest's, or whose scores and generations don't pair up one to one.
    stored = read_run(RUNS / PILOT_RUN)
    manifest = stored.manifest
    assert (manifest.platform, manifest.config["device"]) == ("local-mps", "mps")
    # Made from a clean commit, the one in its id.
    assert re.fullmatch(r"[0-9a-f]{40}", manifest.git_sha), manifest.git_sha
    assert PILOT_RUN.endswith(manifest.git_sha[:7])

    # The committed pilot config, with the model pinned in configs/models.yaml.
    config = EvalConfig(**read_config_file(PILOT_CONFIG))
    pilot = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))["pilot"]
    assert manifest.model_ref == config.model == f"{pilot['repo']}@{pilot['revision']}"
    for name in ("backend", "device", "split", "limit", "prompt_version", "retry"):
        assert manifest.config[name] == getattr(config, name), name
    assert manifest.config["generation"] == config.params.model_dump()

    # Every val question but the known failing golds, once each: val only, never dev.
    val = {example_id for example_id, _ in read_split_ids("val")} - FAILING_GOLD_IDS
    assert sorted(score.example_id for score in stored.scores) == sorted(val)

    report = stored.paths.report.read_text(encoding="utf-8")
    assert "\n| Platform | `local-mps` |\n" in report
    assert f"\n| Git commit | `{manifest.git_sha}` |\n" in report
    assert stored.paths.chart.is_file()
    assert f"({PILOT_RUN}/" in (RUNS / "README.md").read_text(encoding="utf-8")
    # Logged, on a platform that spends no GPU budget.
    rows = [
        row for row in read_rows(ROOT / "docs" / "compute-ledger.md") if row.run_id == PILOT_RUN
    ]
    assert [row.platform for row in rows] == ["local-mps"]


@pytest.mark.phase(2)
@pytest.mark.requires_full_data
def test_pilot_report_rebuilds_and_traces_to_its_files() -> None:
    data = resolve_data_dir()
    checks = rebuild(PILOT_RUN, runs_root=RUNS, data_dir=data)
    assert [check.path.name for check in checks] == ["val-report.md", "val-hardness.png"]
    report, chart = checks
    # The report as the stored files render it, with each of its numbers traced to them.
    assert (report.diff, report.skipped) == ((), None)
    assert untraced(report.path.read_text(encoding="utf-8"), RUNS / PILOT_RUN, data_dir=data) == []
    # The chart too, where this machine draws as the one that drew it did. Elsewhere, or
    # after a release of what draws it changes, its bytes can't be compared, and the test
    # says so instead of passing.
    if chart.skipped is not None:
        pytest.skip(f"the pilot's chart wasn't compared: {chart.skipped}")
    assert chart.diff == ()
