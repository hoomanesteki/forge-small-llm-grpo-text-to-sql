"""The FG-301 throughput spike (kaggle/spike/throughput.py), checked off the GPU.

Nothing here needs a GPU, vLLM or the network: the stages that would are run with fakes,
so these check the wiring and the report, not a measurement.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
import types
from argparse import Namespace
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.spike import THROUGHPUT, throughput

ROOT = Path(__file__).resolve().parents[2]
spike = throughput()


def _run_args(tmp_path: Path, *extra: str) -> Namespace:
    return spike.parse_args(
        ["run", "--out-dir", str(tmp_path / "out"), "--work-dir", str(tmp_path / "work"), *extra]
    )


@pytest.mark.phase(3)
def test_import_runs_nothing(tmp_path: Path) -> None:
    # Loading the script must not import a heavy library, start a stage or write a file.
    code = textwrap.dedent(
        f"""
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("spike", {str(THROUGHPUT)!r})
        module = importlib.util.module_from_spec(spec)
        sys.modules["spike"] = module
        spec.loader.exec_module(module)
        heavy = {{"torch", "vllm", "trl", "transformers", "unsloth", "datasets", "huggingface_hub"}}
        print(sorted(name for name in sys.modules if name.split(".")[0] in heavy))
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "[]"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.phase(3)
@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["limit-probe", "--max-beats", "0"], "must be at least 1, got 0"),
        (["run", "--budget-minutes", "soon"], "invalid float value"),
        (["secrets", "--labels", "HF_TOKEN"], "--out"),
        ([], "required"),
    ],
)
def test_parse_rejects_bad_settings(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        spike.parse_args(argv)
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.phase(3)
def test_kernel_argv_is_used_when_kaggle_passes_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Kaggle starts a script kernel with no arguments; bundling writes them into the script.
    monkeypatch.setattr(sys, "argv", ["throughput.py"])
    argv = ["limit-probe", "--out-dir", str(tmp_path), "--interval-seconds", "0"]
    monkeypatch.setattr(spike, "KERNEL_ARGV", [*argv, "--max-beats", "1"])
    assert spike.main() == 0
    assert json.loads((tmp_path / spike.PROBE_NAME).read_text())["beats"] == 1


@pytest.mark.phase(3)
def test_limit_probe_rewrites_its_heartbeat(tmp_path: Path) -> None:
    argv = ["limit-probe", "--out-dir", str(tmp_path), "--interval-seconds", "0"]
    assert spike.main([*argv, "--max-beats", "3"]) == 0
    beat = json.loads((tmp_path / spike.PROBE_NAME).read_text())
    assert beat["schema"] == spike.REPORT_SCHEMA
    assert beat["beats"] == 3
    assert beat["started_at"] <= beat["last_beat_at"]
    # Only the one file, rewritten in place: no temporary file is left beside it.
    assert [p.name for p in tmp_path.iterdir()] == [spike.PROBE_NAME]


@pytest.mark.phase(3)
def test_plan_probes_the_session(tmp_path: Path) -> None:
    args = _run_args(tmp_path, "--secret-labels", "HF_TOKEN", "WANDB_API_KEY")
    stages = spike.plan_stages(args, python="/usr/bin/python3")
    assert [(s.name, s.python) for s in stages] == [
        ("env", "/usr/bin/python3"),
        ("secrets", "/usr/bin/python3"),
    ]
    # Each stage's arguments parse as its child command.
    secrets = spike.parse_args([*stages[1].argv, "--out", "x.json"])
    assert (secrets.command, secrets.labels) == ("secrets", ["HF_TOKEN", "WANDB_API_KEY"])


def _fake_runner(outcomes: dict[str, Any], calls: list[str]) -> Any:
    def runner(stage: Any, result_file: Path, timeout: float) -> Any:
        assert timeout > 0
        assert result_file.suffix == ".json"
        calls.append(stage.name)
        return outcomes.get(stage.name, spike.Outcome("ok", result={"stage": stage.name}))

    return runner


@pytest.mark.phase(3)
def test_run_report_records_every_stage(tmp_path: Path) -> None:
    args = _run_args(tmp_path)
    calls: list[str] = []
    outcomes = {"secrets": spike.Outcome("timeout", error="killed after 60 s, the budget left")}
    report = spike.run(args, runner=_fake_runner(outcomes, calls))

    stored = json.loads((tmp_path / "out" / spike.REPORT_NAME).read_text())
    assert stored == report
    assert set(report) == {
        "schema",
        "story",
        "started_at",
        "updated_at",
        "finished_at",
        "launch",
        "args",
        "stages",
    }
    assert (report["schema"], report["story"]) == (spike.REPORT_SCHEMA, "FG-301")
    assert report["finished_at"] is not None
    assert report["args"]["secret_labels"] == ["HF_TOKEN"]
    statuses = {s["name"]: (s["status"], s["error"]) for s in report["stages"]}
    assert statuses == {
        "env": ("ok", None),
        "secrets": ("timeout", "killed after 60 s, the budget left"),
    }
    assert calls == ["env", "secrets"]
    for stage in report["stages"]:
        assert set(stage) >= {"name", "status", "started_at", "seconds", "result", "error"}
    assert report["stages"][0]["result"] == {"stage": "env"}


@pytest.mark.phase(3)
def test_run_skips_stages_once_out_of_budget(tmp_path: Path) -> None:
    args = _run_args(tmp_path, "--budget-minutes", "1")
    ticks: Iterator[int] = iter(range(0, 10_000, 25))  # every reading is 25 s later
    calls: list[str] = []
    report = spike.run(args, runner=_fake_runner({}, calls), clock=lambda: float(next(ticks)))
    statuses = [(s["name"], s["status"]) for s in report["stages"]]
    assert statuses[:1] == [("env", "ok")]
    assert all(status == "skipped" for _, status in statuses[1:])
    assert calls == ["env"]
    assert report["stages"][1]["error"] == "out of budget (1 min)"


@pytest.fixture
def fakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A folder on a child's PYTHONPATH, where a test writes the modules it imports."""
    folder = tmp_path / "fakes"
    folder.mkdir()
    monkeypatch.setenv("PYTHONPATH", str(folder))
    return folder


@pytest.mark.phase(3)
@pytest.mark.parametrize(
    ("source", "timeout", "status", "error"),
    [
        ("raise RuntimeError('no client here')", 60.0, "error", "RuntimeError: no client here"),
        ("import os; print('dying'); os._exit(3)", 60.0, "error", "exit code 3 and no result"),
        ("import time; time.sleep(60)", 1.0, "timeout", "killed after 1 s"),
    ],
)
def test_run_child_reports_a_failed_stage(
    fakes: Path, tmp_path: Path, source: str, timeout: float, status: str, error: str
) -> None:
    # The secrets stage imports kaggle_secrets, so a stand-in for it decides how it ends.
    (fakes / "kaggle_secrets.py").write_text(source + "\n")
    stage = spike.Stage("secrets", sys.executable, ("secrets",))
    started = time.monotonic()
    outcome = spike.run_child(stage, tmp_path / "results" / "secrets.json", timeout)
    assert (outcome.status, outcome.result) == (status, None)
    assert error in outcome.error
    if "exit code" in error:
        assert "dying" in outcome.error  # the end of its output comes with it
    assert time.monotonic() - started < 30


@pytest.mark.phase(3)
def test_run_child_returns_the_stage_result(fakes: Path, tmp_path: Path) -> None:
    (fakes / "kaggle_secrets.py").write_text(
        "class UserSecretsClient:\n    def get_secret(self, label):\n        return 'x'\n"
    )
    stage = spike.Stage("secrets", sys.executable, ("secrets", "--labels", "HF_TOKEN"))
    outcome = spike.run_child(stage, tmp_path / "secrets.json", 60.0)
    assert outcome.status == "ok", outcome.error
    assert outcome.result["labels"] == {"HF_TOKEN": {"readable": True}}


@pytest.mark.phase(3)
def test_secrets_stage_records_whether_a_read_worked_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    canary = "canary-" + os.urandom(8).hex()

    class CredentialError(Exception):
        pass

    class UserSecretsClient:
        def get_secret(self, label: str) -> str:
            if label == "HF_TOKEN":
                return canary
            raise CredentialError(f"no secret {label}, value {canary}")

    fake = types.ModuleType("kaggle_secrets")
    fake.UserSecretsClient = UserSecretsClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kaggle_secrets", fake)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    out = tmp_path / "secrets.json"
    argv = ["secrets", "--out", str(out), "--labels", "HF_TOKEN", "WANDB_API_KEY"]
    assert spike.main(argv) == 0

    written = out.read_text()
    assert canary not in written
    assert json.loads(written)["result"] == {
        "run_type": "Batch",
        "client": "available",
        "labels": {
            "HF_TOKEN": {"readable": True},
            "WANDB_API_KEY": {"readable": False, "error": "CredentialError"},
        },
    }


@pytest.mark.phase(3)
def test_env_stage_reports_off_kaggle_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(spike, "_internet", lambda: {"url": "offline in tests", "status": 0})
    info = spike.env_stage(Namespace())
    assert info["python"] == sys.version
    assert info["torch"]["version"]
    assert set(info["image"]) == {"git_commit", "build_date"}
    assert info["internet"] == {"url": "offline in tests", "status": 0}
    assert info["packages"]["trl"] is not None
    # Off Kaggle some probes fail; each says why instead of stopping the stage.
    for probe in (info["image"]["git_commit"], info["nvidia_smi"]):
        assert isinstance(probe, list | str) or set(probe) == {"error"}
    json.dumps(info)
