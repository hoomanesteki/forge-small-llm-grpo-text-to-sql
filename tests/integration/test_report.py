"""``forge report`` on runs that ``forge eval run`` stored with the fake backend (FG-205)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from forge import cli


@pytest.fixture
def runs(spider_mini: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dev_log: Path) -> Path:
    """Runs and the gold cache under tmp_path, and a scratch dev log."""
    monkeypatch.setenv("FORGE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("FORGE_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "runs"


@pytest.mark.phase(2)
def test_reports_on_eval_runs_through_the_cli(runs: Path) -> None:
    cli_runner = CliRunner()
    for model, run_id in (("gold", "gold-run"), ("select-1", "select-one")):
        args = ["eval", "run", "--backend", "fake", "--model", model, "--run-id", run_id]
        evaluated = cli_runner.invoke(cli.app, [*args, "--allow-dirty"])
        assert evaluated.exit_code == 0, evaluated.output
        reported = cli_runner.invoke(cli.app, ["report", "run", run_id])
        assert reported.exit_code == 0, reported.output
        assert reported.stdout == (
            f"report: {runs / run_id / 'eval' / 'val-report.md'}\n"
            f"chart: {runs / run_id / 'eval' / 'val-hardness.png'}\n"
        )

    gold = (runs / "gold-run" / "eval" / "val-report.md").read_text()
    assert gold.startswith("# Run gold-run: official EX 100.0% [100.0%, 100.0%] on val\n")
    assert "\n| official EX | 20 | 20 | 100.0% [100.0%, 100.0%] |\n" in gold
    assert "\nNone: official EX matches every sample.\n" in gold
    assert gold.endswith("\n## Example failures\n\nNone.\n")
    constant = (runs / "select-one" / "eval" / "val-report.md").read_text()
    assert "\n| official EX | 0 | 20 | 0.0% [0.0%, 0.0%] |\n" in constant
    # SELECT 1 reads no table, and every val gold query reads one.
    assert "\n| wrong_tables | 20 | 100.0% |\n" in constant
    assert "\n### wrong_tables: 5 of 20 shown\n" in constant

    missing = cli_runner.invoke(cli.app, ["report", "run", "no-such-run"])
    assert missing.exit_code == 1
    assert missing.stderr.startswith(f"error: no run in {runs / 'no-such-run'}")
