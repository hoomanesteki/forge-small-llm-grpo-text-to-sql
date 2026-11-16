"""kaggle/entry.py, the script a Kaggle job kernel runs, checked on the laptop.

pip and the job are stood in for, and the input folder is built the way the code and data
datasets mount, so these check the stages and what status.json says, not a GPU run.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import types
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from forge.data.spider import raw_files
from forge.eval import run_files
from forge.eval.config import HOSTS
from forge.ops import kaggle, manifest, sessions
from tests.fakes import FakeHub
from tests.kernel import KERNEL_ENV, SHA, WHEEL, argv, build_inputs, kernel_folder
from tests.spike import ENTRY, entry

script = entry()
TOKEN = "hf_" + "".join(chr(97 + (i * 7) % 26) for i in range(30))


@pytest.fixture
def kernel_env(tmp_path: Path, spider_mini: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return kernel_folder(tmp_path, monkeypatch)


@pytest.fixture
def kernel_root(kernel_env: Path, spider_mini: Path) -> Path:
    """That folder with both datasets mounted under input/."""
    build_inputs(kernel_env, spider_mini)
    return kernel_env


def status_of(root: Path) -> dict[str, Any]:
    path = root / "working" / "runs" / "run-1" / "status.json"
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


class Job:
    """Stands in for eval_job: records each call and what it saw."""

    def __init__(self, fail: BaseException | None = None) -> None:
        self.calls: list[bool] = []
        self.fail = fail
        self.env: dict[str, str | None] = {}

    def __call__(self, kernel: Any, preflight: bool) -> str:
        self.calls.append(preflight)
        self.env = {name: os.environ.get(name) for name in KERNEL_ENV}
        if self.fail is not None:
            # As a run that dies after writing its first manifest would leave it.
            kernel.run_dir.mkdir(parents=True, exist_ok=True)
            (kernel.run_dir / "manifest.json").write_text(json.dumps({"status": "running"}))
            raise self.fail
        return "finished"


@pytest.mark.phase(3)
def test_import_runs_nothing(tmp_path: Path) -> None:
    # The script starts on an image without Forge: loading it must import nothing beyond
    # the standard library, and run nothing.
    code = textwrap.dedent(
        f"""
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("entry", {str(ENTRY)!r})
        module = importlib.util.module_from_spec(spec)
        sys.modules["entry"] = module
        spec.loader.exec_module(module)
        heavy = {{"forge", "torch", "transformers", "huggingface_hub", "packaging", "yaml"}}
        print(sorted(name for name in sys.modules if name.split(".")[0] in heavy))
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "[]"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.phase(3)
def test_names_match_forges() -> None:
    # The script can't import these before the wheel is in, so it keeps copies.
    assert script.GIT_SHA_FILE == kaggle.GIT_SHA_FILE
    assert script.CHECKSUM_FILE == kaggle.CHECKSUM_FILE
    assert kaggle.REQUIREMENTS.name == script.REQUIREMENTS_FILE
    assert script.CODE_SHA_ENV == manifest.CODE_SHA_ENV
    assert script.MANIFEST_FILE == run_files.MANIFEST_FILE
    assert script.SESSIONS_FILE == sessions.SESSIONS_FILE
    assert script.HOSTS == HOSTS
    assert set(script.STAGE_FUNCTIONS) == set(script.STAGES)


@pytest.mark.phase(3)
def test_a_finished_job_runs_every_stage_in_order(
    kernel_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = Job()
    monkeypatch.setitem(script.JOBS, "eval", job)
    pip: list[list[str]] = []

    assert script.main(argv(kernel_root), run=lambda c: pip.append(list(c))) == 0

    status = status_of(kernel_root)
    assert (status["status"], status["stage"], status["error"]) == ("finished", "upload", None)
    assert list(status["stages"]) == list(script.STAGES)
    assert status["code_sha"] == SHA and status["options"]["model"] == "gold"
    code = kernel_root / "input" / "forge-code"
    assert pip == [
        [*script.PIP, "--no-deps", str(code / WHEEL)],
        [*script.PIP, "-r", str(code / "requirements-gpu.txt")],
    ]
    # The preflight runs first, through the same job, and the job sees the kernel's paths.
    assert job.calls == [True, False]
    assert job.env["FORGE_CODE_SHA"] == SHA
    assert job.env["FORGE_DATA_DIR"] == str(kernel_root / "input" / "forge-spider")
    assert job.env["FORGE_RUNS_DIR"] == str(kernel_root / "working" / "runs")
    # A dev job's look is logged in its output: a kernel has no checkout to log it in.
    assert job.env["FORGE_DEV_USAGE"] == str(kernel_root / "working" / "dev-usage.md")
    secrets = status["stages"]["secrets"]["result"]
    assert set(secrets) == {"HF_TOKEN", "WANDB_API_KEY"}
    assert status["stages"]["upload"]["result"].startswith("skipped: no HF_TOKEN")


@pytest.mark.phase(3)
def test_a_failed_job_still_says_so_and_closes_its_manifest(
    kernel_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setitem(script.JOBS, "eval", Job(fail=ValueError(f"bad reply {TOKEN}")))

    assert script.main(argv(kernel_root, "--no-preflight"), run=lambda c: None) == 0

    status = status_of(kernel_root)
    assert (status["status"], status["stage"]) == ("failed", "job")
    assert status["error"] == "ValueError: bad reply ***"
    assert status["stages"]["preflight"]["result"] == "skipped (--no-preflight)"
    run = kernel_root / "working" / "runs" / "run-1"
    closed = json.loads((run / "manifest.json").read_text())
    assert closed["status"] == "failed" and closed["ended_at"]
    # The token is in neither the log nor any file the kernel keeps.
    printed = capsys.readouterr()
    assert TOKEN not in printed.out + printed.err
    assert all(TOKEN not in path.read_text() for path in run.iterdir())


@pytest.mark.phase(3)
def test_a_stopped_session_ends_the_job_as_failed(
    kernel_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def stopped(kernel: Any, preflight: bool) -> str:
        os.kill(os.getpid(), signal.SIGTERM)
        return "finished"

    monkeypatch.setitem(script.JOBS, "eval", stopped)
    before = signal.getsignal(signal.SIGTERM)

    script.main(argv(kernel_root), run=lambda c: None)

    status = status_of(kernel_root)
    assert (status["status"], status["stage"]) == ("failed", "preflight")
    assert status["error"] == f"Stopped: stopped by signal {int(signal.SIGTERM)}"
    assert signal.getsignal(signal.SIGTERM) is before


@pytest.mark.phase(3)
def test_code_of_another_commit_is_refused_before_installing(
    kernel_env: Path, spider_mini: Path
) -> None:
    build_inputs(kernel_env, spider_mini, sha=SHA[::-1])
    pip: list[Sequence[str]] = []

    script.main(argv(kernel_env), run=pip.append)

    status = status_of(kernel_env)
    assert (status["status"], status["stage"]) == ("failed", "code")
    assert f"holds commit {SHA[::-1]}, but the job was pushed at {SHA}" in status["error"]
    assert pip == []


@pytest.mark.phase(3)
def test_install_names_what_the_image_lacks(
    kernel_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = script._installed
    monkeypatch.setattr(
        script, "_installed", lambda name: None if name == "sqlglot" else real(name)
    )

    script.main(argv(kernel_root), run=lambda c: None)

    status = status_of(kernel_root)
    assert (status["status"], status["stage"]) == ("failed", "install")
    assert "sqlglot>=25.0 is not installed" in status["error"]
    assert "Pin each in kaggle/requirements-gpu.txt" in status["error"]


@pytest.mark.phase(3)
def test_unmet_requirements_skip_extras_and_name_versions() -> None:
    installed = {"numpy": "1.25.0", "sqlglot": "30.31.0"}.get
    requires = ["numpy>=1.26", "sqlglot>=25.0", "torch>=2.4; extra == 'cpu'", "scipy>=1.11"]
    assert script.unmet_requirements(requires, installed) == [
        "numpy>=1.26 but 1.25.0 is installed",
        "scipy>=1.11 is not installed",
    ]


@pytest.mark.phase(3)
def test_zipped_databases_are_unpacked_and_every_file_checked(
    kernel_env: Path, spider_mini: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path = kernel_env
    inputs = build_inputs(tmp_path, spider_mini, zipped=True)
    job = Job()
    monkeypatch.setitem(script.JOBS, "eval", job)

    script.main(argv(tmp_path), run=lambda c: None)

    unpacked = tmp_path / "temp" / "forge-spider"
    assert status_of(tmp_path)["status"] == "finished"
    assert job.env["FORGE_DATA_DIR"] == str(unpacked)
    assert sorted(raw_files(unpacked)) == sorted(raw_files(spider_mini))

    # A file that differs from spider.sha256 stops the job before it starts.
    tables = inputs / "forge-spider" / "tables.json"
    tables.write_text(tables.read_text() + " ", encoding="utf-8")
    shutil.rmtree(tmp_path / "working")
    job.calls.clear()
    script.main(argv(tmp_path), run=lambda c: None)
    status = status_of(tmp_path)
    assert (status["status"], status["stage"]) == ("failed", "data")
    assert "1 Spider files don't match spider.sha256: tables.json (" in status["error"]
    assert job.calls == []


@pytest.mark.phase(3)
def test_secrets_are_recorded_by_where_they_came_from_never_by_value(
    kernel_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class CredentialError(Exception):
        pass

    class UserSecretsClient:
        def get_secret(self, label: str) -> str:
            if label == "HF_TOKEN":
                return TOKEN
            raise CredentialError(f"no secret {label}")

    monkeypatch.setitem(
        sys.modules, "kaggle_secrets", types.SimpleNamespace(UserSecretsClient=UserSecretsClient)
    )
    hub = FakeHub()

    def makes_its_run(kernel: Any, preflight: bool) -> str:
        if not preflight:
            # As eval_job leaves the run's folder, its lock among its files.
            kernel.run_dir.mkdir(parents=True)
            (kernel.run_dir / "manifest.json").write_text(json.dumps({"status": "finished"}))
            (kernel.run_dir / ".lock").write_text("")
        return "finished"

    monkeypatch.setattr("huggingface_hub.HfApi", hub.client)
    monkeypatch.setitem(script.JOBS, "eval", makes_its_run)

    script.main(argv(kernel_root), run=lambda c: None)

    status = status_of(kernel_root)
    assert status["stages"]["secrets"]["result"] == {
        "HF_TOKEN": "kaggle secrets",
        "WANDB_API_KEY": "kaggle secrets: CredentialError",
    }
    assert status["stages"]["upload"]["result"] == "uploaded to forge-owner/forge-runs"
    run = kernel_root / "working" / "runs" / "run-1"
    assert hub.tokens == [TOKEN]
    assert hub.calls == [
        (
            "upload_folder",
            {
                "repo_id": "forge-owner/forge-runs",
                "repo_type": "model",
                "folder_path": str(run),
                "path_in_repo": "runs/run-1",
                "ignore_patterns": [".lock"],
                # The kernel's copy may be partial, so it never deletes what the store has.
                "delete_patterns": None,
                "commit_message": "Add run run-1",
            },
        )
    ]
    assert set(hub.files) == {"runs/run-1/manifest.json", "runs/run-1/sessions.json"}
    assert hub.files["runs/run-1/manifest.json"] == b'{"status": "finished"}'
    # The store's copy holds the kernel's session up to its upload; the end comes after.
    (uploaded,) = json.loads(hub.files["runs/run-1/sessions.json"])
    assert uploaded["started_at"] == status["started_at"]
    assert uploaded["ended_at"] <= status["ended_at"]
    assert TOKEN not in capsys.readouterr().out
    assert TOKEN not in (run / "status.json").read_text()


@pytest.mark.phase(3)
def test_a_resume_starts_from_the_last_attempts_output(
    kernel_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    last = kernel_root / "input" / "forge-run-1" / "runs" / "run-1"
    (last / "eval").mkdir(parents=True)
    (last / "manifest.json").write_text(json.dumps({"status": "failed"}))
    (last / "eval" / "val-scores.jsonl").write_text("{}\n")
    seen: list[list[str]] = []

    def resumed(kernel: Any, preflight: bool) -> str:
        files = sorted(p.relative_to(kernel.run_dir).as_posix() for p in kernel.run_dir.rglob("*"))
        seen.append(files if kernel.run_dir.exists() else [])
        return "finished"

    monkeypatch.setitem(script.JOBS, "eval", resumed)

    script.main(argv(kernel_root, "--resume"), run=lambda c: None)

    assert status_of(kernel_root)["status"] == "finished"
    # The preflight sees a fresh folder; the job, the last attempt's files.
    assert seen == [[], ["eval", "eval/val-scores.jsonl", "manifest.json"]]


def _sessions_of(root: Path) -> list[dict[str, Any]]:
    path = root / "working" / "runs" / "run-1" / "sessions.json"
    data: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return data


@pytest.mark.phase(3)
def test_the_run_session_spans_the_whole_kernel(
    kernel_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def job(kernel: Any, preflight: bool) -> str:
        if not preflight:
            # As run_eval keeps it: the session spans the job alone.
            now = datetime.now(UTC)
            sessions.begin(kernel.run_dir, now)
            sessions.end(kernel.run_dir, now, "finished")
            ended = {"status": "finished", "ended_at": now.isoformat()}
            (kernel.run_dir / "manifest.json").write_text(json.dumps(ended))
        return "finished"

    monkeypatch.setitem(script.JOBS, "eval", job)

    script.main(argv(kernel_root), run=lambda c: None)

    status = status_of(kernel_root)
    # Setup, preflight and upload hold the GPU too, so the session is the kernel's.
    whole = {"started_at": status["started_at"], "ended_at": status["ended_at"]}
    assert _sessions_of(kernel_root) == [{**whole, "status": "finished"}]
    assert status["session"] == 1
    run = kernel_root / "working" / "runs" / "run-1"
    (session,) = sessions.read_sessions(run) or []
    assert (session.started_at, session.ended_at) == tuple(
        datetime.fromisoformat(whole[name]) for name in ("started_at", "ended_at")
    )


@pytest.mark.phase(3)
def test_a_resume_that_fails_before_its_run_starts_is_a_failed_session(
    kernel_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    last = kernel_root / "input" / "forge-run-1" / "runs" / "run-1"
    last.mkdir(parents=True)
    first = {"started_at": "2026-11-15T09:00:00+00:00", "ended_at": "2026-11-15T20:15:00+00:00"}
    stopped = {**first, "status": "stopped_time_guard"}
    (last / "manifest.json").write_text(json.dumps(stopped))
    (last / "sessions.json").write_text(json.dumps([stopped]))

    def refused(kernel: Any, preflight: bool) -> str:
        if not preflight:
            raise script.JobError("the run can't resume on other data")
        return "finished"

    monkeypatch.setitem(script.JOBS, "eval", refused)

    script.main(argv(kernel_root, "--resume"), run=lambda c: None)

    status = status_of(kernel_root)
    assert (status["status"], status["session"]) == ("failed", 2)
    # The first session stays as it was; the kernel's hours are a session of their own.
    whole = {"started_at": status["started_at"], "ended_at": status["ended_at"]}
    assert _sessions_of(kernel_root) == [stopped, {"status": "failed", **whole}]

    # A kernel that fails before its job holds no run, so it has no session.
    early = kernel_root / "early"
    early.mkdir()
    script.main(argv(early), run=lambda c: None)
    assert status_of(early)["session"] is None
    assert not (early / "working" / "runs" / "run-1" / "sessions.json").exists()
