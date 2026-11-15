"""Which commit an eval run records: the checkout FORGE_ROOT names, wherever the run is
started from, and a clear error when there is no checkout (FG-292)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from forge import cli
from forge.core.types import RunManifest

GOLD_RUN = ["eval", "run", "--backend", "fake", "--model", "gold", "--limit", "2"]


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dev_log: Path) -> Path:
    """Runs and the gold cache under tmp_path, outside every checkout the tests make."""
    monkeypatch.setenv("FORGE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("FORGE_CACHE_DIR", str(tmp_path / "cache"))
    # Keeps git from finding a checkout that happens to hold the temp folder.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    return tmp_path / "runs"


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return done.stdout.strip()


def _checkout(path: Path, name: str) -> str:
    """A one-commit checkout at ``path`` holding ``docs/PLAN.md``; return its HEAD."""
    (path / "docs").mkdir(parents=True)
    (path / "docs" / "PLAN.md").write_text(f"# {name}\n")
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    _git(path, "add", ".")
    _git(path, "commit", "-q", "-m", name)
    return _git(path, "rev-parse", "HEAD")


@pytest.mark.phase(2)
def test_run_records_the_commit_forge_root_names(
    spider_mini: Path, runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    forge_head = _checkout(tmp_path / "forge", "forge")
    other_head = _checkout(tmp_path / "elsewhere", "elsewhere")
    assert forge_head != other_head
    monkeypatch.setenv("FORGE_ROOT", str(tmp_path / "forge"))
    # Started from another checkout, the run still describes the one FORGE_ROOT names.
    monkeypatch.chdir(tmp_path / "elsewhere")
    cli_runner = CliRunner()

    clean = cli_runner.invoke(cli.app, [*GOLD_RUN, "--run-id", "clean"])
    assert clean.exit_code == 0, clean.output
    manifest = RunManifest.model_validate_json((runs / "clean" / "manifest.json").read_bytes())
    assert manifest.git_sha == forge_head

    # Changes in the cwd's checkout don't count, and changes in FORGE_ROOT's do.
    (tmp_path / "elsewhere" / "stray.py").write_text("x = 1\n")
    assert cli_runner.invoke(cli.app, [*GOLD_RUN, "--run-id", "stray"]).exit_code == 0
    (tmp_path / "forge" / "new_module.py").write_text("x = 1\n")
    dirty = cli_runner.invoke(cli.app, [*GOLD_RUN, "--run-id", "dirty"])
    assert dirty.exit_code == 1
    assert dirty.stderr == (
        "error: working tree has uncommitted changes; commit first or pass --allow-dirty\n"
    )
    assert not (runs / "dirty").exists()


@pytest.mark.phase(2)
def test_eval_run_outside_a_checkout_is_one_error_line(
    spider_mini: Path, runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = tmp_path / "plain"
    (plain / "docs").mkdir(parents=True)
    (plain / "docs" / "PLAN.md").write_text("# not a checkout\n")
    monkeypatch.setenv("FORGE_ROOT", str(plain))
    monkeypatch.chdir(plain)

    result = CliRunner().invoke(cli.app, [*GOLD_RUN, "--allow-dirty"])

    assert result.exit_code == 1
    # A handled error, not a traceback from git.
    assert isinstance(result.exception, SystemExit), result.exception
    (line,) = result.stderr.splitlines()
    assert line.startswith(f"error: {plain} is not in a git checkout"), line
    assert "FORGE_ROOT" in line
    assert not runs.exists()
