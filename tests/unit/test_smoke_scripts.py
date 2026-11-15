"""The scripts under ``scripts/smoke`` stay runnable: ``make smoke``'s pipeline, and the
integration checkpoint scripts."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "scripts" / "smoke"

# uv as the smoke pipeline calls it, without the work: `uv run python` is the tests' own
# Python, and every other command only logs its arguments. `forge eval run` also stores
# the manifest of a run made at FAKE_MADE_AT, `forge report run` a one-line report, and
# the tool FAKE_FAIL names fails.
FAKE_UV = """#!/usr/bin/env bash
set -euo pipefail
shift
tool="$1"
shift
if [[ "$tool" == python ]]; then
    exec "$FAKE_PYTHON" "$@"
fi
echo "$tool $*" >> "$FAKE_LOG"
if [[ "$tool" == "${FAKE_FAIL:-}" ]]; then
    exit 1
fi
if [[ "$tool $1 $2" == "forge eval run" ]]; then
    printf '%s\\n%s\\n' "$FORGE_DATA_DIR" "$FORGE_RUNS_DIR" > "$FAKE_LOG.env"
    while [[ "$1" != --run-id ]]; do
        shift
    done
    mkdir -p "$FORGE_RUNS_DIR/$2"
    printf '{"git_sha": "%s"}' "$FAKE_MADE_AT" > "$FORGE_RUNS_DIR/$2/manifest.json"
fi
if [[ "$tool $1 $2" == "forge report run" ]]; then
    mkdir -p "$FORGE_RUNS_DIR/$3/eval"
    printf '# Run %s: official EX 0.0%%\n' "$3" > "$FORGE_RUNS_DIR/$3/eval/val-report.md"
fi
"""


def _check_script(name: str) -> str:
    script = SMOKE / name
    assert os.access(script, os.X_OK), f"{script} is not executable"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text
    return text


def _in_order(text: str, steps: list[str]) -> None:
    positions = [text.find(step) for step in steps]
    assert -1 not in positions, [s for s, p in zip(steps, positions, strict=True) if p == -1]
    assert positions == sorted(positions)


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return done.stdout.strip()


@pytest.mark.phase(1)
def test_phase_1_smoke_runs_the_checkpoint_steps() -> None:
    text = _check_script("phase_1.sh")
    # Integration checkpoint 1 in docs/PLAN.md, in order.
    _in_order(
        text,
        [
            'uv run forge data fetch --from-zip "$zip"',
            "uv run forge data stats",
            "uv run forge eval gold --split val",
            'uv run forge sql run --db concert_singer "DELETE FROM singer"',
            "uv run forge eval agreement --mutations 500 --out",
        ],
    )


@pytest.mark.phase(2)
def test_phase_2_smoke_runs_the_checkpoint_steps() -> None:
    text = _check_script("phase_2.sh")
    # Integration checkpoint 2 in docs/PLAN.md, in order: gold scores 100%, the tiny model
    # answers the same questions, the runs are compared, and the reports rebuild clean.
    _in_order(
        text,
        [
            "uv run forge eval run --backend fake --model gold --split val --limit 20",
            '"# Run gold: official EX 100.0% [100.0%, 100.0%] on val"',
            'uv run forge eval run --backend hf --model "$model" --split val --limit 20',
            "uv run forge report compare gold tiny",
            "uv run forge report rebuild tiny",
            "uv run forge report rebuild gold",
        ],
    )


@pytest.mark.phase(2)
def test_make_smoke_marks_only_a_clean_commit_that_passed(tmp_path: Path) -> None:
    _check_script("pipeline.sh")
    dry_run = subprocess.run(["make", "-n", "smoke"], cwd=ROOT, capture_output=True, text=True)
    assert "scripts/smoke/pipeline.sh" in dry_run.stdout, dry_run.stdout

    # A repo of one commit with the pipeline, what it reads, and the real ignore rules.
    repo = tmp_path / "repo"
    for rel in ("scripts/smoke/pipeline.sh", "configs/models.yaml", ".gitignore"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, repo / rel)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    sha = _git(repo, "rev-parse", "HEAD")
    fake = tmp_path / "bin" / "uv"
    fake.parent.mkdir()
    fake.write_text(FAKE_UV, encoding="utf-8")
    fake.chmod(0o755)
    log = tmp_path / "uv.log"

    def smoke(
        made_at: str, fail: str = "", phase: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        log.unlink(missing_ok=True)
        # make gate PHASE=N exports PHASE to the tests, so only a case that sets it has it.
        env = {key: value for key, value in os.environ.items() if key != "PHASE"}
        env.update(
            PATH=f"{fake.parent}{os.pathsep}{os.environ['PATH']}",
            FAKE_PYTHON=sys.executable,
            FAKE_LOG=str(log),
            FAKE_MADE_AT=made_at,
            FAKE_FAIL=fail,
        )
        if phase is not None:
            env["PHASE"] = phase
        script = repo / "scripts" / "smoke" / "pipeline.sh"
        return subprocess.run([str(script)], env=env, capture_output=True, text=True)

    earlier = repo / f".smoke-ok-{'0' * 40}"
    earlier.write_text("a pass on an earlier commit\n", encoding="utf-8")
    passed = smoke(sha)
    assert passed.returncode == 0, passed.stderr
    tiny = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))["tiny"]
    run_id = f"smoke-{sha[:7]}"
    assert log.read_text(encoding="utf-8").splitlines() == [
        "pytest --phase 8 tests/smoke",
        f"forge eval run --backend hf --model {tiny['repo']}@{tiny['revision']} --split val "
        f"--limit 20 --run-id {run_id} --workers 2 --allow-dirty",
        f"forge report run {run_id}",
    ]
    data_dir, runs_dir = Path(f"{log}.env").read_text(encoding="utf-8").splitlines()
    assert Path(data_dir).resolve() == (repo / "tests" / "fixtures" / "spider_mini").resolve()
    # The report's path and the report itself are printed, then the scratch folder goes.
    report = Path(runs_dir) / run_id / "eval" / "val-report.md"
    assert f"\n== {report}\n# Run {run_id}: official EX 0.0%\n" in passed.stdout
    assert not Path(runs_dir).exists()
    marker = repo / f".smoke-ok-{sha}"
    assert marker.read_text(encoding="utf-8") == f"make smoke passed on {sha} with run {run_id}\n"
    assert not earlier.exists()
    assert _git(repo, "status", "--porcelain", "--untracked-files=all") == ""
    assert smoke(sha, phase="8").returncode == 0
    assert marker.is_file()

    # Uncommitted changes pass, but vouch for no commit, so the marker goes.
    dirty = smoke(f"{sha}-dirty")
    assert dirty.returncode == 0, dirty.stderr
    assert f"smoke passed on {sha}-dirty, not on commit {sha}, so no marker" in dirty.stdout
    assert not marker.exists()

    # So does a smoke that skipped some phases' tests, as make gate PHASE=2 runs it.
    marker.write_text("an earlier pass\n", encoding="utf-8")
    reduced = smoke(sha, phase="2")
    assert reduced.returncode == 0, reduced.stderr
    assert log.read_text(encoding="utf-8").splitlines()[0] == "pytest --phase 2 tests/smoke"
    assert f"# Run {run_id}: official EX 0.0%" in reduced.stdout
    assert "smoke tests of phases 0 to 2 only, so no marker" in reduced.stdout
    assert not marker.exists()

    # A failed step stops the pipeline and takes the commit's earlier marker with it.
    marker.write_text("an earlier pass\n", encoding="utf-8")
    failed = smoke(sha, fail="pytest")
    assert failed.returncode != 0
    assert log.read_text(encoding="utf-8").splitlines() == ["pytest --phase 8 tests/smoke"]
    assert not marker.exists()
