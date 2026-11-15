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
def test_reports_on_eval_runs_through_the_cli(runs: Path, tmp_path: Path) -> None:
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

    compared = cli_runner.invoke(cli.app, ["report", "compare", "gold-run", "select-one"])
    assert compared.exit_code == 0, compared.output
    path = runs / "select-one" / "eval" / "compare-gold-run.md"
    assert compared.stdout == f"report: {path}\n"
    text = path.read_text()
    # Run a gets more right in each of the 12 clusters: p = 2 / 2**12.
    assert text.startswith(
        "# Official EX on val, run b against run a: -100.0 points [-100.0, -100.0], "
        "McNemar p = 0.0005 on clusters\n"
    )
    assert "| run a right | 0 | 20 |\n| run a wrong | 0 | 0 |" in text
    assert "\n| wrong_tables | 0 | 20 | +20 |\n" in text
    out = tmp_path / "elsewhere.md"
    elsewhere = cli_runner.invoke(
        cli.app, ["report", "compare", "gold-run", "select-one", "--out", str(out)]
    )
    assert elsewhere.exit_code == 0, elsewhere.output
    assert "| run a right | 0 | 20 |" in out.read_text()

    rebuilt = cli_runner.invoke(cli.app, ["report", "rebuild", "select-one"])
    assert rebuilt.exit_code == 0, rebuilt.output
    assert rebuilt.stdout == (
        f"{runs / 'select-one' / 'eval' / 'val-report.md'}: matches its rebuild\n"
        f"{path}: matches its rebuild\n"
    )

    # A number edited by hand fails the rebuild, which shows where.
    report = runs / "select-one" / "eval" / "val-report.md"
    report.write_text(constant.replace("| wrong_tables | 20 |", "| wrong_tables | 19 |"))
    edited = cli_runner.invoke(cli.app, ["report", "rebuild", "select-one"])
    assert edited.exit_code == 1
    assert f"{report}: differs from its rebuild\n" in edited.stdout
    assert "\n-| wrong_tables | 19 | 100.0% |\n+| wrong_tables | 20 | 100.0% |\n" in edited.stdout
    assert f"{path}: matches its rebuild\n" in edited.stdout
    assert edited.stderr == "error: 1 of 2 reports differ from their rebuild\n"

    missing = cli_runner.invoke(cli.app, ["report", "run", "no-such-run"])
    assert missing.exit_code == 1
    assert missing.stderr.startswith(f"error: no run in {runs / 'no-such-run'}")
