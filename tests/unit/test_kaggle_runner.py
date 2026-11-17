"""``forge kaggle run`` and the commands that follow its job, with uv and kaggle stood in
for: the checks before a push, the kernel it renders, and what it records."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import types
from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime, timedelta, tzinfo
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from forge import cli
from forge.core.types import RunManifest
from forge.eval.config import EvalConfig, read_config_file
from forge.ops import clock, kaggle, kaggle_job
from forge.ops.dev_usage import log_dev_use, read_uses
from forge.ops.hub import FINAL, manifest_bytes
from forge.ops.kaggle import (
    MAX_KERNEL_NAME,
    KaggleClient,
    KaggleError,
    check_kernel_metadata,
    title_slug,
    valid_slug,
)
from forge.ops.kaggle_job import (
    COLAB_DIRS,
    KERNEL_BUILD,
    MAIN_GUARD,
    SPIKE_SLUG,
    SPIKE_TITLE,
    AdapterSource,
    adapter_kernel_sources,
    entry_argv,
    kernel_name,
    plan_job,
    write_kernel,
)
from forge.ops.ledger import read_rows, summary
from forge.ops.sessions import Session
from forge.train.config import load_train_config, parse_train_config
from forge.train.sft import promote_best
from tests.fakes import sft_run_folder
from tests.kernel import WHEEL
from tests.spike import entry, load_script

ROOT = Path(__file__).resolve().parents[2]
OWNER = "forge-owner"
CONFIG = "model: gold\nbackend: fake\nlimit: 5\nops:\n  est_gpu_hours: 0.5\n"
NOT_FOUND = "404 Client Error: Not Found\n\nThe resource was not found.\n"
QUOTA = "resource,used,remaining,total,refreshAt\r\nGPU,3.50h,26.50h,30.00h,2026-11-21T00:00\r\n"
# What kaggle 2.2.5 prints for an error it has no handler for, as a push whose answer
# never came back: the error, then its advice.
LOST = (
    "ReadTimeout: HTTPSConnectionPool(host='api.kaggle.com', port=443): Read timed out.\n\n"
    "If this is unexpected, you can re-run with `kaggle --debug [command]` to see the full\n"
    "traceback.\n"
)
# kernels status for a kernel Kaggle doesn't have: kaggle 2.2.5's message for a 403, and
# a 404 as it prints one.
DENIED = (
    "Cannot access kernel '{kernel}' (Permission 'kernels.get' was denied). The most likely "
    "cause is a wrong kernel slug.\n"
)
NO_KERNEL = (
    "404 Client Error: Not Found for url: "
    "https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus\n\n"
    "The resource was not found. Check the spelling of the reference you passed.\n"
)
script = entry()


def _done(command: list[str], code: int = 0, stdout: str = "", stderr: str = "") -> Any:
    return subprocess.CompletedProcess(command, code, stdout=stdout, stderr=stderr)


class FakeKaggle:
    """Stands in for uv and kaggle: a code dataset whose versions each hold a GIT_SHA, the
    data dataset, the GPU quota, and kernels that go through ``states``. With
    ``push_failure`` a push exits 1 with it, having saved the kernel or not as ``saves``
    says, and status answers for a kernel Kaggle doesn't have as kaggle 2.2.5 does."""

    def __init__(self, *code_versions: str, data: bool = True) -> None:
        self.code_versions = list(code_versions)
        self.data = data
        self.pending = 0  # status calls that still show the last upload in processing
        self.quota = QUOTA
        self.push_output: str | None = None
        self.push_failure: str | None = None
        self.saves = True
        self.missing: set[str] = set()  # kernels whose push failed before Kaggle saved them
        self.states = ["COMPLETE"]
        self.outputs: dict[str, str] = {}
        self.pushed: list[dict[str, str]] = []
        self.commands: list[list[str]] = []

    def __call__(self, args: Sequence[str]) -> Any:
        command = list(args)
        self.commands.append(command)
        verb = command[1:3]
        if command[:2] == ["uv", "build"]:
            out = Path(command[command.index("--out-dir") + 1])
            (out / WHEEL).write_bytes(b"PK\x03\x04 a wheel")
            return _done(command)
        if verb == ["datasets", "status"]:
            return self._dataset_status(command)
        if verb in (["datasets", "create"], ["datasets", "version"]):
            folder = Path(command[command.index("-p") + 1])
            self.code_versions.append((folder / "GIT_SHA").read_text(encoding="utf-8").strip())
            self.pending = 1
            return _done(command, stdout="Dataset version is being created.\n")
        if verb == ["datasets", "download"]:
            number = int(command[3].rsplit("/", 1)[1])
            folder = Path(command[command.index("-p") + 1])
            (folder / "GIT_SHA").write_text(f"{self.code_versions[number - 1]}\n")
            return _done(command)
        if command[1:] == ["quota", "-v"]:
            return _done(command, stdout=self.quota)
        if verb == ["kernels", "push"]:
            folder = Path(command[command.index("-p") + 1])
            kernel = json.loads((folder / "kernel-metadata.json").read_text())["id"]
            if self.push_failure is not None and not self.saves:
                self.missing.add(kernel)
                return _done(command, 1, stderr=self.push_failure)
            self.pushed.append({p.name: p.read_text(encoding="utf-8") for p in folder.iterdir()})
            if self.push_failure is not None:
                return _done(command, 1, stderr=self.push_failure)
            return _done(command, stdout=self.push_output or _pushed(kernel))
        if verb == ["kernels", "status"]:
            if "/".join(command[3].split("/")[:2]) in self.missing:
                return _done(command, 1, stderr=DENIED.format(kernel=command[3]))
            state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            return _done(command, stdout=f'{command[3]} has status "KernelWorkerStatus.{state}"\n')
        if verb == ["kernels", "logs"]:
            return _done(command, stdout="[forge] job eval run-1: finished\n")
        if verb == ["kernels", "output"]:
            folder = Path(command[command.index("-p") + 1])
            pattern = command[-1] if command[-2] == "--file-pattern" else ""
            for name, text in self.outputs.items():
                if re.search(pattern, name):
                    (folder / name).parent.mkdir(parents=True, exist_ok=True)
                    (folder / name).write_text(text, encoding="utf-8")
            (folder / f"{command[3].split('/')[1]}.log").write_text("log\n", encoding="utf-8")
            return _done(command)
        raise AssertionError(f"unexpected command {command}")

    def _dataset_status(self, command: list[str]) -> Any:
        slug = command[3].split("/")[1]
        exists = bool(self.code_versions) if slug == "forge-code" else self.data
        if not exists:
            return _done(command, 1, stderr=NOT_FOUND)
        if "--format" not in command:
            return _done(command, stdout="ready")
        number = len(self.code_versions) if slug == "forge-code" else 1
        status = "ready"
        if slug == "forge-code" and self.pending:
            self.pending -= 1
            status, number = "blobs_received", number - 1
        state = {"status": status, "current_version_number": number}
        return _done(command, stdout=json.dumps(state))

    def ran(self, *words: str) -> list[list[str]]:
        return [c for c in self.commands if c[1 : 1 + len(words)] == list(words)]


def _pushed(kernel: str) -> str:
    return (
        "Kernel version 1 successfully pushed.  Please check progress at "
        f"https://www.kaggle.com/code/{kernel}\n"
    )


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return out.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A committed checkout that passed make smoke, with Forge's kaggle/ files and a job
    config, a ledger outside it and no developer credentials."""
    root = tmp_path / "repo"
    for rel in (
        "kaggle/requirements-gpu.txt",
        "kaggle/requirements-train-gpu.txt",
        "kaggle/kernel-metadata.template.json",
        "kaggle/entry.py",
        "kaggle/spike/throughput.py",
        "configs/spike/throughput.yaml",
    ):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, root / rel)
    (root / "configs" / "eval").mkdir(parents=True)
    (root / "configs" / "eval" / "job.yaml").write_text(CONFIG, encoding="utf-8")
    (root / ".gitignore").write_text("build/\n/runs\n.smoke-ok-*\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "init")
    (root / f".smoke-ok-{_git(root, 'rev-parse', 'HEAD')}").write_text("passed\n")
    for name in ("KAGGLE_USERNAME", "KAGGLE_CONFIG_DIR", "FORGE_RUNS_DIR", "FORGE_RUN_STORE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FORGE_ROOT", str(root))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("FORGE_KAGGLE_USER", OWNER)
    monkeypatch.setenv("FORGE_LEDGER", str(tmp_path / "ledger.md"))
    monkeypatch.setenv("FORGE_WEEKLY_GPU_HOURS", "30")
    return root


def _use(fake: FakeKaggle, monkeypatch: pytest.MonkeyPatch) -> FakeKaggle:
    monkeypatch.setattr(kaggle, "KaggleClient", lambda: KaggleClient(run=fake, executable="kaggle"))
    # No real waiting between status checks.
    monkeypatch.setattr(kaggle_job, "time", types.SimpleNamespace(sleep=lambda seconds: None))
    return fake


def _run(*args: str) -> Any:
    return CliRunner().invoke(cli.app, ["kaggle", "run", *args])


def _eval(repo: Path, *extra: str) -> Any:
    return _run("--job", "eval", "--config", str(repo / "configs/eval/job.yaml"), *extra)


def _today() -> date:
    """Today's UTC date, which run ids and log rows take, whatever the machine's zone."""
    return datetime.now(UTC).date()


def _spent(repo: Path, hours: float, on: date | None = None) -> None:
    ledger = Path(repo.parent / "ledger.md")
    day = (on or _today()).isoformat()
    row = f"| {day} | kaggle | T4 | {hours:.2f} | earlier-run | test | 0.00 |"
    ledger.write_text(f"# Compute ledger\n\n| a |\n|---|\n{row}\n", encoding="utf-8")


def _record(repo: Path, run_id: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((repo / "runs" / run_id / "kernels.json").read_text())
    return data


MAIN = ROOT / "configs" / "sft" / "qwen3-4b-main.yaml"


def _train(run_id: str, config: Path = MAIN) -> None:
    """Push the SFT job of ``config`` as run ``run_id``, so this machine has its
    kernels.json, which names the kernel an eval of its checkpoints mounts."""
    result = _run("--job", "sft", "--config", str(config), "--run-id", run_id, "--yes")
    assert result.exit_code == 0, result.output


@pytest.mark.phase(3)
def test_refuses_without_smoke_marker(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _use(FakeKaggle(_git(repo, "rev-parse", "HEAD")), monkeypatch)
    head = _git(repo, "rev-parse", "HEAD")
    # A marker for another commit doesn't count.
    (repo / f".smoke-ok-{head}").rename(repo / f".smoke-ok-{head[::-1]}")

    result = _eval(repo, "--yes")

    assert result.exit_code == 1, result.output
    assert f"error: make smoke has not passed on {head[:7]}" in result.stderr
    assert fake.commands == []


@pytest.mark.phase(3)
def test_refuses_on_a_dirty_tree(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _use(FakeKaggle(_git(repo, "rev-parse", "HEAD")), monkeypatch)
    (repo / "configs" / "eval" / "job.yaml").write_text(CONFIG + "n: 2\n", encoding="utf-8")

    result = _eval(repo, "--yes")

    assert result.exit_code == 1, result.output
    assert "error: the working tree has uncommitted changes" in result.stderr
    assert fake.commands == []


@pytest.mark.phase(3)
def test_refuses_over_budget(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    fake = _use(FakeKaggle(head), monkeypatch)
    _spent(repo, 29.8)  # 0.2 h left this week by the ledger; the job needs 0.5

    result = _eval(repo, "--yes")

    assert result.exit_code == 1, result.output
    assert "ledger: 0.20 of 30 h left this week" in result.stdout
    assert "kaggle quota: 26.50 h left" in result.stdout
    assert "error: the estimate is over what is left this week" in result.stderr
    assert fake.ran("kernels") == []

    # Kaggle's own quota counts too, when the ledger has room.
    _spent(repo, 0.0)
    fake.quota = QUOTA.replace("26.50h", "0.40h")
    assert _eval(repo, "--yes").exit_code == 1
    assert fake.ran("kernels") == []

    # Allowed, the job goes, and its run's config says it was over budget.
    result = _eval(repo, "--yes", "--allow-over-budget")
    assert result.exit_code == 0, result.output
    run_id = f"{_today():%Y-%m-%d}-gold-eval-{head[:7]}"
    argv = _baked_argv(fake.pushed[0]["entry.py"])
    options = json.loads(argv[argv.index("--options") + 1])
    assert options == {"model": "gold", "backend": "fake", "limit": 5, "allow_over_budget": True}
    assert _record(repo, run_id)["kernels"][0]["over_budget"] is True


@pytest.mark.phase(3)
def test_run_pushes_the_job_and_records_its_kernel(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    fake = _use(FakeKaggle(head), monkeypatch)
    # Asked, and told no, it pushes nothing.
    declined = CliRunner().invoke(
        cli.app,
        ["kaggle", "run", "--job", "eval", "--config", str(repo / "configs/eval/job.yaml")],
        input="n\n",
    )
    assert declined.exit_code == 1 and "Push it? [y/N]: n\nnothing pushed" in declined.stdout
    assert fake.ran("kernels") == [] and not (repo / "runs").exists()

    result = _eval(repo, "--yes")

    assert result.exit_code == 0, result.output
    run_id = f"{_today():%Y-%m-%d}-gold-eval-{head[:7]}"
    kernel = f"{OWNER}/forge-{run_id}"
    assert f"code {OWNER}/forge-code/1 at {head[:7]}, data {OWNER}/forge-spider" in result.stdout
    assert "estimate 0.50 GPU h" in result.stdout
    assert f"run {run_id}: kernel {kernel}, version 1" in result.stdout
    # Its GIT_SHA was HEAD, so nothing was published; the push asks for a T4 and stops
    # the job at its estimate and half again.
    assert fake.ran("datasets", "version") == [] and fake.ran("datasets", "create") == []
    folder = repo / KERNEL_BUILD / run_id
    push = ["kaggle", "kernels", "push", "-p", str(folder), "--accelerator", "NvidiaTeslaT4"]
    assert fake.ran("kernels", "push") == [[*push, "-t", "2700"]]
    record = _record(repo, run_id)
    assert (record["job"], record["code_sha"]) == ("eval", head)
    assert record["kernels"][0]["kernel"] == kernel
    assert record["kernels"][0]["code"] == f"{OWNER}/forge-code/1"
    assert _git(repo, "status", "--porcelain") == ""

    # The same run again is refused: it has a kernel already.
    again = _eval(repo, "--yes")
    assert again.exit_code == 1 and "pass --resume to carry it on" in again.stderr


@pytest.mark.phase(3)
def test_run_publishes_head_when_the_code_dataset_holds_another_commit(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    fake = _use(FakeKaggle(head[::-1]), monkeypatch)

    result = _eval(repo, "--yes")

    assert result.exit_code == 0, result.output
    assert f"{OWNER}/forge-code doesn't hold {head[:7]} yet: publishing it" in result.stdout
    assert len(fake.ran("datasets", "version")) == 1
    # The kernel pins the new version, which status said was processing at first.
    meta = json.loads(fake.pushed[0]["kernel-metadata.json"])
    assert meta["dataset_sources"] == [f"{OWNER}/forge-code/2", f"{OWNER}/forge-spider"]
    downloads = fake.ran("datasets", "download")
    assert [c[3] for c in downloads] == [f"{OWNER}/forge-code/1", f"{OWNER}/forge-code/2"]


@pytest.mark.phase(3)
def test_run_needs_the_data_dataset_and_a_push_kaggle_accepts(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    fake = _use(FakeKaggle(head, data=False), monkeypatch)
    result = _eval(repo, "--yes")
    assert result.exit_code == 1
    assert "forge-owner/forge-spider doesn't exist; run forge kaggle publish-data" in result.stderr

    # A push error is refused, and nothing is recorded.
    fake.data = True
    fake.push_output = "Kernel push error: Notebook not found\n"
    result = _eval(repo, "--yes")
    assert result.exit_code == 1
    assert "was refused: Kernel push error: Notebook not found" in result.stderr
    assert not (repo / "runs").exists()

    # kaggle 2.2.5 exits 0 when it drops a dataset source, and Kaggle has saved and started
    # that version without it: it is recorded, so status, logs and wait find it, with a
    # warning rather than a refusal.
    head_id = f"{_today():%Y-%m-%d}-gold-eval-{head[:7]}"
    dropped = (
        "The following are not valid dataset sources and could not be added to the kernel: "
        "['forge-owner/forge-spider']"
    )
    fake.push_output = f"{dropped}\n{_pushed(f'{OWNER}/forge-{head_id}')}"
    result = _eval(repo, "--yes")
    assert result.exit_code == 0, result.output
    assert f"warning: {dropped}" in result.stderr
    assert f"Kaggle started version 1 without them; forge kaggle wait {head_id}" in result.stderr
    assert _record(repo, head_id)["kernels"][0]["kernel"] == f"{OWNER}/forge-{head_id}"
    status = CliRunner().invoke(cli.app, ["kaggle", "status", head_id])
    assert status.exit_code == 0 and "COMPLETE" in status.stdout


@pytest.mark.phase(3)
def test_entry_renders_job_args(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    config = repo / "configs" / "eval" / "job.yaml"
    plan = plan_job("eval", config, owner=OWNER, sha=head, runs_root=repo / "runs")
    code, data = f"{OWNER}/forge-code/7", f"{OWNER}/forge-spider"

    folder = write_kernel(repo, plan, code=code, data=data, run_store="forge-owner/forge-runs")

    meta = json.loads((folder / "kernel-metadata.json").read_text())
    assert check_kernel_metadata(meta) == []
    assert (meta["id"], meta["code_file"]) == (plan.kernel, "entry.py")
    assert (meta["is_private"], meta["enable_gpu"], meta["enable_internet"]) == (True, True, True)
    assert (meta["dataset_sources"], meta["kernel_sources"]) == ([code, data], [])
    # The script holds the job's arguments, and reads them back as the job's.
    rendered = load_script(folder / "entry.py", "forge_rendered_entry")
    assert entry_argv(plan, "forge-owner/forge-runs") == rendered.KERNEL_ARGV
    args = script.parse_args(rendered.KERNEL_ARGV)
    assert (args.job, args.run_id, args.code_sha) == ("eval", plan.run_id, head)
    assert args.options == {"model": "gold", "backend": "fake", "limit": 5}
    assert (args.preflight, args.resume, args.run_store) == (True, False, "forge-owner/forge-runs")
    # The job's deadline counts to the push's timeout, the estimate and half again.
    assert args.timeout_s == plan.timeout_s == 2700
    assert EvalConfig(**args.options, host="kaggle").platform == "kaggle"
    # Everything else in the script is as committed.
    baked = f"KERNEL_ARGV: list[str] = {json.dumps(rendered.KERNEL_ARGV)}"
    unbaked = (folder / "entry.py").read_text().replace(baked, "KERNEL_ARGV: list[str] = []")
    assert unbaked == (repo / "kaggle" / "entry.py").read_text()


@pytest.mark.phase(3)
def test_resume_mounts_the_last_attempts_kernel(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    fake = _use(FakeKaggle(head), monkeypatch)
    assert _eval(repo, "--yes", "--run-id", "first").exit_code == 0
    assert "has no" in _eval(repo, "--yes", "--run-id", "other", "--resume").stderr

    result = _eval(repo, "--yes", "--run-id", "first", "--resume", "--no-preflight")

    assert result.exit_code == 0, result.output
    meta = json.loads(fake.pushed[1]["kernel-metadata.json"])
    assert (meta["id"], meta["title"]) == (f"{OWNER}/forge-first-2", "Forge first 2")
    assert meta["kernel_sources"] == [f"{OWNER}/forge-first/1"]
    argv = _baked_argv(fake.pushed[1]["entry.py"])
    assert argv[-2:] == ["--no-preflight", "--resume"]
    assert [k["kernel"] for k in _record(repo, "first")["kernels"]] == [
        f"{OWNER}/forge-first",
        f"{OWNER}/forge-first-2",
    ]
    # A resume keeps the run's options, the over-budget flag among them.
    over = _eval(repo, "--yes", "--run-id", "first", "--resume", "--allow-over-budget")
    assert over.exit_code == 1 and "was pushed with other options" in over.stderr


@pytest.mark.phase(3)
def test_a_resume_skips_an_attempt_whose_kernel_source_kaggle_dropped(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    fake = _use(FakeKaggle(head), monkeypatch)
    assert _eval(repo, "--yes", "--run-id", "r1").exit_code == 0
    # Kaggle starts the second attempt without the first one's output, so its restore
    # fails and its own output holds no run.
    lost = (
        "The following are not valid kernel sources and could not be added to the kernel: "
        f"['{OWNER}/forge-r1/1']"
    )
    fake.push_output = f"{lost}\n{_pushed(f'{OWNER}/forge-r1-2')}"
    second = _eval(repo, "--yes", "--run-id", "r1", "--resume")
    assert second.exit_code == 0, second.output
    assert (
        f"warning: this attempt can't restore the run, so a --resume mounts {OWNER}/forge-r1/1 "
        "again" in second.stderr
    )
    fake.push_output = None

    third = _eval(repo, "--yes", "--run-id", "r1", "--resume")
    fourth = _eval(repo, "--yes", "--run-id", "r1", "--resume")

    assert third.exit_code == fourth.exit_code == 0, third.output + fourth.output
    metas = [json.loads(pushed["kernel-metadata.json"]) for pushed in fake.pushed]
    assert [(m["id"], m["kernel_sources"]) for m in metas] == [
        (f"{OWNER}/forge-r1", []),
        (f"{OWNER}/forge-r1-2", [f"{OWNER}/forge-r1/1"]),
        # The attempt numbers go on; the run is carried on from the last attempt holding it.
        (f"{OWNER}/forge-r1-3", [f"{OWNER}/forge-r1/1"]),
        (f"{OWNER}/forge-r1-4", [f"{OWNER}/forge-r1-3/1"]),
    ]
    kernels = _record(repo, "r1")["kernels"]
    assert [k.get("kernel_source_dropped", False) for k in kernels] == [False, True, False, False]
    # The kernel to follow is still the latest pushed, which is the one running.
    status = CliRunner().invoke(cli.app, ["kaggle", "status", "r1"])
    assert status.stdout.startswith(f"{OWNER}/forge-r1-4/1: COMPLETE")


@pytest.mark.phase(3)
def test_kernel_names_fit_kaggles_limit_and_follow_the_run_id() -> None:
    # A name that fits is the run id's own, as it always was.
    assert kernel_name("first") == ("forge-first", "Forge first")
    assert kernel_name("first", 2) == ("forge-first-2", "Forge first 2")
    fits = "-".join(["ab"] * 14) + "-cd"  # 44 characters, so a title of 50
    assert len(fits) == 44
    assert kernel_name(fits) == (f"forge-{fits}", f"Forge {fits}")
    # A longer one keeps the run id up to a dash, then a hash of all of it.
    default = "2026-11-16-qwen2-5-coder-1-5b-instruct-eval-c8f4b41"
    assert kernel_name(default) == (
        f"forge-2026-11-16-qwen2-5-coder-1-5b-{_digest(default)}",
        f"Forge 2026-11-16-qwen2-5-coder-1-5b-{_digest(default)}",
    )
    one_word = "x" * 80
    assert kernel_name(one_word)[0] == f"forge-{'x' * 35}-{_digest(one_word)}"
    # A resume's suffix can take a name that fit past the limit.
    assert kernel_name(fits, 2)[0] == f"forge-{fits[:32]}-{_digest(fits)}-2"
    for run_id in (fits, default, one_word, f"{fits}-d", "-".join(["a1"] * 30)):
        for attempt in (1, 2, 10, 1000):
            slug, title = kernel_name(run_id, attempt)
            assert len(slug) <= MAX_KERNEL_NAME and len(title) <= MAX_KERNEL_NAME, (slug, title)
            assert valid_slug(slug) and title_slug(title) == slug, (slug, title)
            assert slug.endswith(f"-{attempt}") == (attempt > 1)
    # Run ids that begin alike, cut to the same start, still name kernels of their own.
    assert len({kernel_name(f"{default}-{i}")[0] for i in range(100)}) == 100


def _digest(run_id: str) -> str:
    return hashlib.sha256(run_id.encode()).hexdigest()[:8]


@pytest.mark.phase(3)
def test_a_run_id_past_kaggles_limit_is_pushed_and_resumed_under_a_shorter_kernel(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    fake = _use(FakeKaggle(head), monkeypatch)
    config = repo.parent / "smoke.yaml"
    shutil.copyfile(ROOT / "configs" / "eval" / "smoke-kaggle-50-val.yaml", config)

    # FG-303's 50-question job: its default run id made a 57-character title, which Kaggle
    # refused with a 400 once the code dataset had been checked.
    first = _run("--job", "eval", "--config", str(config), "--yes")

    assert first.exit_code == 0, first.output
    run_id = f"{_today():%Y-%m-%d}-qwen2-5-coder-1-5b-instruct-eval-{head[:7]}"
    assert len(f"Forge {run_id}") == 57
    stem = f"{_today():%Y-%m-%d}-qwen2-5-coder-1-5b-{_digest(run_id)}"
    kernel = f"{OWNER}/forge-{stem}"
    meta = json.loads(fake.pushed[0]["kernel-metadata.json"])
    assert (meta["id"], meta["title"]) == (kernel, f"Forge {stem}")
    assert check_kernel_metadata(meta) == []
    # The run keeps its id: the kernel runs it, and kernels.json maps it to the kernel.
    argv = _baked_argv(fake.pushed[0]["entry.py"])
    assert argv[argv.index("--run-id") + 1] == run_id
    assert f"run {run_id}: kernel {kernel}, version 1" in first.stdout
    assert f"next: forge kaggle wait {run_id}" in first.stdout
    status = CliRunner().invoke(cli.app, ["kaggle", "status", run_id])
    assert status.stdout.startswith(f"{kernel}/1: COMPLETE")

    resumed = _run("--job", "eval", "--config", str(config), "--yes", "--resume")

    assert resumed.exit_code == 0, resumed.output
    meta = json.loads(fake.pushed[1]["kernel-metadata.json"])
    assert (meta["id"], meta["title"]) == (f"{kernel}-2", f"Forge {stem} 2")
    assert meta["kernel_sources"] == [f"{kernel}/1"]
    assert check_kernel_metadata(meta) == []
    argv = _baked_argv(fake.pushed[1]["entry.py"])
    assert (argv[argv.index("--run-id") + 1], argv[-1]) == (run_id, "--resume")
    assert [k["kernel"] for k in _record(repo, run_id)["kernels"]] == [kernel, f"{kernel}-2"]
    # Every command that follows the run finds its latest kernel by the run id.
    for command in ("status", "logs", "output"):
        followed = CliRunner().invoke(cli.app, ["kaggle", command, run_id])
        assert followed.exit_code == 0, followed.output
    assert [c[3] for c in fake.ran("kernels", "status")] == [f"{kernel}/1", f"{kernel}-2/1"]
    assert [c[3] for c in fake.ran("kernels", "logs")] == [f"{kernel}-2/1"]
    assert fake.ran("kernels", "output")[-1][3] == f"{kernel}-2/1"


@pytest.mark.phase(3)
def test_jobs_a_later_story_builds_and_bad_options_are_refused_before_any_upload(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _use(FakeKaggle(_git(repo, "rev-parse", "HEAD")), monkeypatch)
    config = str(repo / "configs/eval/job.yaml")
    result = _run("--job", "grpo", "--config", config, "--yes")
    assert result.exit_code == 1 and "error: --job grpo comes with FG-504" in result.stderr

    for text, message in (
        ("model: org/m\nbackend: llamacpp\nops: {est_gpu_hours: 1}\n", "comes with FG-702"),
        ("model: org/m\nbackend: hf\nops: {est_gpu_hours: 1}\n", "backend hf runs on the CPU"),
        ("model: gold\nbackend: fake\n", "sets no ops.est_gpu_hours"),
        ("model: gold\nbackend: fake\nops: {est_gpu_hours: 13}\n", "won't fit a 12 h session"),
    ):
        path = repo.parent / "job.yaml"
        path.write_text(text, encoding="utf-8")
        result = _run("--job", "eval", "--config", str(path), "--yes")
        assert result.exit_code == 1 and message in result.stderr, result.stderr
    assert fake.commands == []


@pytest.mark.phase(4)
def test_sft_job_pushes_the_config_it_checked(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    fake = _use(FakeKaggle(head), monkeypatch)
    config = ROOT / "configs" / "sft" / "qwen3-4b-main.yaml"

    result = _run("--job", "sft", "--config", str(config), "--yes")

    assert result.exit_code == 0, result.output
    run_id = f"{_today():%Y-%m-%d}-qwen3-4b-sft-{head[:7]}"
    assert f"run {run_id}: kernel {OWNER}/forge-{run_id}, version 1" in result.stdout
    assert "estimate 2.50 GPU h" in result.stdout
    # The code and data datasets mounted, a T4, and the plan's 2.5 h and half again.
    meta = json.loads(fake.pushed[0]["kernel-metadata.json"])
    assert meta["dataset_sources"] == [f"{OWNER}/forge-code/1", f"{OWNER}/forge-spider"]
    assert fake.ran("kernels", "push")[0][-4:] == ["--accelerator", "NvidiaTeslaT4", "-t", "13500"]
    args = script.parse_args(_baked_argv(fake.pushed[0]["entry.py"]))
    assert (args.job, args.run_id, args.code_sha, args.preflight) == ("sft", run_id, head, True)
    # The kernel's time guard counts to the same timeout.
    assert args.timeout_s == 13500
    # The whole config as it was checked here, defaults filled in, which the kernel reads
    # back as the same config.
    checked = load_train_config(config)
    assert args.options == {"config": checked.model_dump(mode="json")}
    assert parse_train_config(args.options["config"]) == checked
    assert _record(repo, run_id)["job"] == "sft"


@pytest.mark.phase(4)
def test_eval_job_scores_a_checkpoint_with_its_base_models_config(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    fake = _use(FakeKaggle(head), monkeypatch)
    base = ROOT / "configs" / "eval" / "qwen3-4b-val.yaml"
    checkpoint = "runs/2026-11-20-qwen3-4b-sft-abc1234/checkpoints/step-000242"
    _train("2026-11-20-qwen3-4b-sft-abc1234")

    result = _run("--job", "eval", "--config", str(base), "--adapter", checkpoint, "--yes")

    assert result.exit_code == 0, result.output
    args = script.parse_args(_baked_argv(fake.pushed[-1]["entry.py"]))
    # The base run's options and the adapter alone, so the two runs differ in it alone.
    assert args.options == {**read_config_file(base), "adapter": checkpoint}
    assert EvalConfig(**args.options, host="kaggle").manifest_config()["adapter"] == checkpoint

    named = repo.parent / "named.yaml"
    named.write_text(CONFIG + f"adapter: {checkpoint}\n", encoding="utf-8")
    for job, config, message in (
        ("eval", named, "names an adapter already; leave out --adapter"),
        ("sft", ROOT / "configs" / "sft" / "qwen3-4b-main.yaml", "--adapter is an eval job's"),
    ):
        refused = _run("--job", job, "--config", str(config), "--adapter", checkpoint, "--yes")
        assert refused.exit_code == 1 and message in refused.stderr, refused.stderr
    assert len(fake.pushed) == 2


@pytest.mark.phase(4)
def test_eval_job_adapter_is_checked_before_the_push(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _use(FakeKaggle(_git(repo, "rev-parse", "HEAD")), monkeypatch)
    base = ROOT / "configs" / "eval" / "qwen3-4b-val.yaml"
    small = ROOT / "configs" / "eval" / "qwen3-1-7b-val.yaml"
    run = "2026-11-20-qwen3-4b-sft-abc1234"
    checkpoint = f"runs/{run}/checkpoints/final"
    _train(run)

    def push(config: Path, adapter: str) -> Any:
        return _run(
            *("--job", "eval", "--config", str(config), "--adapter", adapter),
            *("--run-id", f"adapter-{len(fake.pushed)}", "--yes"),
        )

    # A shell's trailing slash and a leading ./ go into the kernel spelled the one way it
    # looks for, and a Hub repo as it was given.
    for adapter, pushed in (
        (f"{checkpoint}/", checkpoint),
        (f"./{checkpoint}", checkpoint),
        ("org/adapter@abc123", "org/adapter@abc123"),
    ):
        result = push(base, adapter)
        assert result.exit_code == 0, result.output
        options = script.parse_args(_baked_argv(fake.pushed[-1]["entry.py"])).options
        assert options["adapter"] == pushed, adapter

    # This machine holds the run, pulled, and the record of another run's push: each must
    # have trained the model the config evaluates.
    sft_run_folder(repo / "runs", run, read_config_file(base)["model"], ["checkpoints/final"])
    other = "2026-11-20-qwen3-1-7b-sft-abc1234"
    (repo / "runs" / other).mkdir()
    record = {"job": "sft", "options": {"config": {"run": {"model_ref": "org/m"}}}}
    (repo / "runs" / other / "kernels.json").write_text(
        json.dumps({**record, "kernels": [{"kernel": f"{OWNER}/forge-{other}"}]})
    )
    evaluated = "2026-11-20-qwen3-4b-eval-abc1234"
    (repo / "runs" / evaluated).mkdir()
    (repo / "runs" / evaluated / "kernels.json").write_text(
        json.dumps({"job": "eval", "options": {}, "kernels": [{"kernel": "x"}]})
    )
    local = repo.parent / "adapter"
    local.mkdir()
    pushes = len(fake.pushed)
    for config, adapter, message in (
        (base, f"runs/{run}/checkpoints/latest", "a run's checkpoints are checkpoints/final"),
        (base, f"/kaggle/working/{checkpoint}", "but not in runs/: the kernel looks for it"),
        (base, f"{run}/checkpoints/final", f"kernel looks for it as {checkpoint}, so name it so"),
        (base, str(local), "is a folder on this machine, which the kernel can't read"),
        (base, "no such repo", "is neither a directory nor a Hub repo id"),
        (small, checkpoint, f"run {run} trained Qwen/Qwen3-4B@"),
        (base, f"runs/{other}/checkpoints/final", f"run {other} trained org/m, and the job"),
        (base, f"runs/{evaluated}/checkpoints/final", "is of kind eval, which trains no adapter"),
    ):
        refused = push(config, adapter)
        assert refused.exit_code == 1 and message in refused.stderr, (adapter, refused.stderr)
    # A run's folder, or a path in runs/ with no run's checkpoint in it, from a folder that
    # holds no runs/ and from the checkout whose runs/ holds the run: the kernel would fetch
    # nothing, so each is refused, however this machine's folders lie.
    hint = "names no checkpoint; name a checkpoint, runs/<run_id>/checkpoints/<name>, which"
    near_misses = [f"runs/{run}", f"./runs/{run}/", "runs/checkpoints", f"runs/{run}/final"]
    for cwd in (repo.parent, repo):
        monkeypatch.chdir(cwd)
        for adapter in near_misses:
            refused = push(base, adapter)
            assert refused.exit_code == 1 and hint in refused.stderr, (cwd, adapter)
    # Refused before anything was pushed.
    assert len(fake.pushed) == pushes
    assert push(base, checkpoint).exit_code == 0


@pytest.mark.phase(4)
def test_an_eval_job_mounts_the_output_of_the_kernel_that_trained_its_adapter(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _use(FakeKaggle(_git(repo, "rev-parse", "HEAD")), monkeypatch)
    base = ROOT / "configs" / "eval" / "qwen3-4b-val.yaml"
    run, unversioned = "2026-11-20-qwen3-4b-sft-abc1234", "2026-11-21-qwen3-4b-sft-abc1234"
    _train(run)
    # kaggle 2.2.5's line when the server sends no version: Kaggle then mounts the kernel's
    # latest output, its only one, since each attempt at a run is a kernel of its own.
    url = f"https://www.kaggle.com/code/{OWNER}/forge-{unversioned}"
    fake.push_output = f"Kernel version successfully pushed.  Please check progress at {url}\n"
    _train(unversioned)
    fake.push_output = None

    def push(adapter: str, run_id: str, *extra: str) -> tuple[str, list[str], list[str]]:
        result = _run(
            *("--job", "eval", "--config", str(base), "--adapter", adapter),
            *("--run-id", run_id, "--yes", *extra),
        )
        assert result.exit_code == 0, result.output
        meta = json.loads(fake.pushed[-1]["kernel-metadata.json"])
        printed = result.stdout + result.stderr
        return printed, meta["kernel_sources"], _baked_argv(fake.pushed[-1]["entry.py"])

    # The training kernel's output beside the datasets, at the version its push recorded.
    printed, sources, argv = push(f"runs/{run}/checkpoints/final", "final-val")
    assert sources == [f"{OWNER}/forge-{run}/1"]
    assert f"its adapter from {OWNER}/forge-{run}/1" in printed
    assert json.loads(fake.pushed[-1]["kernel-metadata.json"])["dataset_sources"] == [
        f"{OWNER}/forge-code/1",
        f"{OWNER}/forge-spider",
    ]
    # No manifest of the run is here yet, so no digest goes for the kernel to check.
    assert "--adapter-sha256" not in argv
    assert push(f"runs/{unversioned}/checkpoints/final", "other-val")[1] == [
        f"{OWNER}/forge-{unversioned}"
    ]

    # Pulled, the run's manifest records the digest of each checkpoint it kept, which the
    # kernel holds the folder it copies to.
    model = read_config_file(base)["model"]
    pulled = sft_run_folder(repo / "runs", run, model, ["checkpoints/step-000242", FINAL])
    half = f"runs/{run}/checkpoints/step-000242"
    _, sources, argv = push(half, "half-val")
    args = script.parse_args(argv)
    assert (args.options["adapter"], sources) == (half, [f"{OWNER}/forge-{run}/1"])
    assert args.adapter_sha256 == pulled.outputs["adapter_sha256:checkpoints/step-000242"]
    # A resume mounts the eval's last attempt, to carry the run on, and the adapter's kernel.
    # When Kaggle drops a kernel source, either may be gone, and the next resume skips it.
    lost = "The following are not valid kernel sources and could not be added to the kernel: []"
    fake.push_output = f"{lost}\n{_pushed(f'{OWNER}/forge-half-val-2')}"
    printed, sources, argv = push(half, "half-val", "--resume")
    fake.push_output = None
    assert sources == [f"{OWNER}/forge-half-val/1", f"{OWNER}/forge-{run}/1"]
    assert argv[-1] == "--resume" and script.parse_args(argv).adapter_sha256 == args.adapter_sha256
    assert f"carrying on {OWNER}/forge-half-val/1, its adapter from {OWNER}/forge-{run}/1" in (
        printed
    )
    warning = "warning: this attempt may lack the run or its adapter, so a --resume mounts"
    assert f"{warning} {OWNER}/forge-half-val/1 again" in printed
    assert push(half, "half-val", "--resume")[1][0] == f"{OWNER}/forge-half-val/1"

    # A run resumed past an attempt whose own kernel source Kaggle dropped, as a GRPO run
    # may be, keeps its checkpoints in the last attempt that holds it, as a resume finds.
    grpo = "2026-11-22-qwen3-4b-grpo-abc1234"
    kernels = [
        {"kernel": f"{OWNER}/forge-{grpo}", "version": 1},
        {"kernel": f"{OWNER}/forge-{grpo}-2", "version": 1},
        {"kernel": f"{OWNER}/forge-{grpo}-3", "version": 1, "kernel_source_dropped": True},
    ]
    (repo / "runs" / grpo).mkdir()
    (repo / "runs" / grpo / "kernels.json").write_text(
        json.dumps({"job": "grpo", "options": {}, "kernels": kernels})
    )
    assert adapter_kernel_sources(f"runs/{grpo}/checkpoints/final", repo / "runs") == (
        AdapterSource(f"runs/{grpo}/checkpoints/final", f"{OWNER}/forge-{grpo}-2/1", None)
    )


@pytest.mark.phase(4)
def test_an_eval_job_refuses_a_checkpoint_no_kernel_output_holds(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _use(FakeKaggle(_git(repo, "rev-parse", "HEAD")), monkeypatch)
    base = ROOT / "configs" / "eval" / "qwen3-4b-val.yaml"
    model, runs = read_config_file(base)["model"], repo / "runs"
    pilot, unknown, pulled, kept = (f"2026-11-{day}-qwen3-4b-sft-abc1234" for day in range(16, 20))
    # Trained on the laptop, as the local pilot was; pulled with no record of its push; and
    # pushed from here and pulled, keeping its final adapter alone.
    sft_run_folder(runs, pilot, model, [FINAL], platform="local-mps", gpu=None)
    sft_run_folder(runs, pulled, model, [FINAL])
    _train(kept)
    sft_run_folder(runs, kept, model, [FINAL])
    pushes = len(fake.pushed)

    for adapter, message in (
        (pilot, f"run {pilot} was made on local-mps, so its checkpoints are in no kernel output"),
        (unknown, f"run {unknown} is not known here, and with no {runs / unknown}/kernels.json"),
        (pulled, f"run {pulled} was made on kaggle, and with no {runs / pulled}/kernels.json"),
    ):
        refused = _run(
            *("--job", "eval", "--config", str(base)),
            *("--adapter", f"runs/{adapter}/checkpoints/final", "--yes"),
        )
        assert refused.exit_code == 1 and message in refused.stderr, (adapter, refused.stderr)
    step = f"runs/{kept}/checkpoints/step-000100"
    refused = _run("--job", "eval", "--config", str(base), "--adapter", step, "--yes")
    assert refused.exit_code == 1
    assert f"run {kept}'s manifest records no digest of checkpoints/step-000100" in refused.stderr
    assert len(fake.pushed) == pushes

    # A notebook mounts no kernel output: it reads the checkpoint from the run store with
    # its Colab HF_TOKEN, spelled as given.
    book = repo.parent / "unknown.ipynb"
    adapter = f"runs/{unknown}/checkpoints/final"
    result = _run(
        *("--job", "eval", "--config", str(base), "--adapter", adapter),
        *("--emit-notebook", str(book)),
    )
    assert result.exit_code == 0, result.output
    code = "".join(json.loads(book.read_text())["cells"][2]["source"])
    args = script.parse_args(_baked_argv(code))
    assert (args.options["adapter"], args.adapter_sha256) == (adapter, None)


@pytest.mark.phase(4)
def test_an_eval_job_of_best_pushes_the_checkpoint_best_json_names(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _use(FakeKaggle(_git(repo, "rev-parse", "HEAD")), monkeypatch)
    base = ROOT / "configs" / "eval" / "qwen3-4b-val.yaml"
    run, runs = "2026-11-20-qwen3-4b-sft-abc1234", repo / "runs"
    half = "checkpoints/step-000242"
    _train(run)
    trained = sft_run_folder(runs, run, read_config_file(base)["model"], [half, FINAL])

    def push(run_id: str) -> Any:
        adapter = f"runs/{run}/checkpoints/best"
        return _run(
            *("--job", "eval", "--config", str(base), "--adapter", adapter),
            *("--run-id", run_id, "--yes"),
        )

    # Not promoted yet: no kernel output holds best/, and no best.json says what it copies.
    refused = push("too-early")
    assert refused.exit_code == 1, refused.output
    assert f"no kernel output holds checkpoints/best of run {run}" in refused.stderr

    promote_best(run, {half: 0.5, FINAL: 0.4}, runs_root=runs)
    result = push("best-val")

    assert result.exit_code == 0, result.output
    # The step's own folder, which the training kernel's output holds, and the digest the
    # run's manifest records for it.
    args = script.parse_args(_baked_argv(fake.pushed[-1]["entry.py"]))
    assert args.options["adapter"] == f"runs/{run}/{half}"
    assert args.adapter_sha256 == trained.outputs[f"adapter_sha256:{half}"]
    meta = json.loads(fake.pushed[-1]["kernel-metadata.json"])
    assert meta["kernel_sources"] == [f"{OWNER}/forge-{run}/1"]

    # A best.json naming another checkpoint than the one best/ holds, or none, is refused.
    record = runs / run / "best.json"
    promoted = json.loads(record.read_text())
    pushes = len(fake.pushed)
    for checkpoint, message in (
        (FINAL, f"names {FINAL}, but run {run}'s manifest records sha256"),
        ("checkpoints/best", f"names no checkpoint of run {run}: 'checkpoints/best'"),
    ):
        record.write_text(json.dumps({**promoted, "checkpoint": checkpoint}))
        refused = push("refused-val")
        assert refused.exit_code == 1 and message in refused.stderr, refused.stderr
    assert len(fake.pushed) == pushes


@pytest.mark.phase(4)
def test_an_sft_run_resumes_from_its_last_kernels_output(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The SFT pilot's second session (FG-496): the same run, config and commit, in a kernel
    # of its own that mounts the first one's output, where the run's checkpoint is.
    fake = _use(FakeKaggle(_git(repo, "rev-parse", "HEAD")), monkeypatch)
    config, other = repo.parent / "pilot.yaml", repo.parent / "other.yaml"
    text = "run: {kind: sft, model_ref: org/m}\nsft: {}\nops: {est_gpu_hours: 1, %s}\n"
    config.write_text(text % "stop_after_step: 50", encoding="utf-8")
    other.write_text(text % "stop_after_step: 60", encoding="utf-8")
    run = "2026-11-20-pilot-abc1234"

    def push(path: Path, *extra: str) -> Any:
        return _run("--job", "sft", "--config", str(path), "--run-id", run, "--yes", *extra)

    assert push(config).exit_code == 0
    assert "--resume to carry it on" in push(config).stderr
    result = push(config, "--resume")

    assert result.exit_code == 0, result.output
    meta = json.loads(fake.pushed[1]["kernel-metadata.json"])
    assert (meta["id"], meta["kernel_sources"]) == (
        f"{OWNER}/forge-{run}-2",
        [f"{OWNER}/forge-{run}/1"],
    )
    first = script.parse_args(_baked_argv(fake.pushed[0]["entry.py"]))
    args = script.parse_args(_baked_argv(fake.pushed[1]["entry.py"]))
    assert (args.job, args.run_id, args.resume, first.resume) == ("sft", run, True, False)
    # The run's options as its first push made them, so the kernel trains the same run, and
    # the same timeout, which its guard counts to.
    assert args.options == first.options and args.timeout_s == first.timeout_s == 5400
    assert args.options["config"]["ops"]["stop_after_step"] == 50
    refused = push(other, "--resume")
    assert refused.exit_code == 1 and "was pushed with other options" in refused.stderr
    # The run's checkpoints are now in the second kernel's output, which holds the whole run.
    final = f"runs/{run}/checkpoints/final"
    source = adapter_kernel_sources(final, repo / "runs")
    assert source.kernel == f"{OWNER}/forge-{run}-2/1"
    assert len(fake.pushed) == 2


@pytest.mark.phase(4)
def test_sft_job_refuses_what_a_kernel_cant_train(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _use(FakeKaggle(_git(repo, "rev-parse", "HEAD")), monkeypatch)
    main = str(ROOT / "configs" / "sft" / "qwen3-4b-main.yaml")
    configs = {
        "grpo": "run: {kind: grpo, model_ref: org/m}\ngrpo: {}\nops: {est_gpu_hours: 1}\n",
        "unbudgeted": "run: {kind: sft, model_ref: org/m}\nsft: {}\n",
        "eval": CONFIG,
    }
    for name, text in configs.items():
        (repo.parent / f"{name}.yaml").write_text(text, encoding="utf-8")
    refused = [
        ((main, "--reason", "baseline"), "an SFT job trains on the train split only"),
        ((str(ROOT / "configs/sft/tiny-smoke.yaml"),), "trains on cpu, and a kernel on its GPU"),
        ((str(ROOT / "configs/sft/pilot-qwen25coder05b-mps.yaml"),), "trains on mps, and a"),
        ((str(repo.parent / "grpo.yaml"),), "trains a grpo run, and --job sft an sft one"),
        ((str(repo.parent / "unbudgeted.yaml"),), "sets no ops.est_gpu_hours"),
        ((str(repo.parent / "eval.yaml"),), "model: Extra inputs are not permitted"),
    ]
    for args, message in refused:
        result = _run("--job", "sft", "--config", *args, "--yes")
        assert result.exit_code == 1 and message in result.stderr, result.stderr
    assert fake.commands == []


@pytest.mark.phase(3)
def test_dev_job_needs_a_reason_and_its_push_logs_the_look(
    repo: Path, dev_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    fake = _use(FakeKaggle(head), monkeypatch)
    dev = repo.parent / "dev.yaml"
    dev.write_text("model: gold\nbackend: fake\nsplit: dev\nops: {est_gpu_hours: 0.5}\n")
    spike = str(repo / "configs/spike/throughput.yaml")
    for args, message in (
        (["--job", "eval", "--config", str(dev)], "--split dev needs --reason"),
        (
            ["--job", "eval", "--config", str(repo / "configs/eval/job.yaml"), "--reason", "x"],
            "--reason goes with --split dev",
        ),
        (["--job", "spike", "--config", spike, "--reason", "x"], "takes no --reason"),
        # A notebook could spend any number of looks, and log none of them.
        (
            [
                "--job",
                "eval",
                "--config",
                str(dev),
                "--reason",
                "baseline",
                "--emit-notebook",
                str(repo.parent / "dev.ipynb"),
            ],
            "can't log its dev look",
        ),
    ):
        result = _run(*args, "--yes")
        assert result.exit_code == 1 and message in result.stderr, result.output
    assert fake.commands == [] and not dev_log.exists()

    result = _run("--job", "eval", "--config", str(dev), "--reason", "baseline", "--yes")

    assert result.exit_code == 0, result.output
    run_id = f"{_today():%Y-%m-%d}-gold-eval-{head[:7]}"
    # The kernel gets the reason it needs to load dev, and the push logs the look here.
    argv = _baked_argv(fake.pushed[0]["entry.py"])
    options = json.loads(argv[argv.index("--options") + 1])
    assert options == {"model": "gold", "backend": "fake", "split": "dev", "reason": "baseline"}
    assert read_uses(dev_log) == [(_today().isoformat(), run_id, "baseline")]
    assert "dev look 1 of 6 logged in docs/dev-usage.md" in result.stdout

    # Once the six looks are spent, a dev job is refused before anything is uploaded.
    for i in range(2, 7):
        log_dev_use(f"run-{i}", "earlier look", path=dev_log)
    fake.commands.clear()
    spent = _run("--job", "eval", "--config", str(dev), "--reason", "one more", "--run-id", "x")
    assert spent.exit_code == 1 and "had its 6 looks" in spent.stderr
    assert fake.commands == [] and len(read_uses(dev_log)) == 6


@pytest.mark.phase(3)
def test_a_dev_look_is_logged_once_kaggle_accepts_the_push(
    repo: Path, dev_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    fake = _use(FakeKaggle(head), monkeypatch)
    dev = repo.parent / "dev.yaml"
    dev.write_text("model: gold\nbackend: fake\nsplit: dev\nops: {est_gpu_hours: 0.5}\n")

    def push(*extra: str) -> Any:
        return _run("--job", "eval", "--config", str(dev), "--reason", "baseline", "--yes", *extra)

    # A push Kaggle refuses starts no kernel, so it spends nothing.
    fake.push_output = "Kernel push error: Notebook not found\n"
    refused = push()
    assert refused.exit_code == 1 and "was refused" in refused.stderr
    assert not dev_log.exists()

    # Once Kaggle has the kernel, it may read dev: the look is logged before the record,
    # whose failure no longer hides it, and the error says not to push again.
    fake.push_output = None
    blocked = repo.parent / "runs-is-a-file"
    blocked.write_text("")
    monkeypatch.setenv("FORGE_RUNS_DIR", str(blocked))
    unrecorded = push()
    assert unrecorded.exit_code == 1, unrecorded.output
    run_id = f"{_today():%Y-%m-%d}-gold-eval-{head[:7]}"
    assert f"Kaggle runs {OWNER}/forge-{run_id} version 1 all the same" in unrecorded.stderr
    assert read_uses(dev_log) == [(_today().isoformat(), run_id, "baseline")]
    monkeypatch.delenv("FORGE_RUNS_DIR")

    # A version Kaggle started without a source counts the same.
    fake.push_output = (
        "The following are not valid dataset sources and could not be added to the kernel: "
        f"['{OWNER}/forge-spider']\n{_pushed(f'{OWNER}/forge-d1')}"
    )
    started = push("--run-id", "d1")
    assert started.exit_code == 0, started.output
    assert "dev look 2 of 6 logged in docs/dev-usage.md" in started.stdout
    fake.push_output = None

    # A resume carries on the same run, from the same commit with the same options: it is
    # the same look, so it adds no row, and goes once the looks are spent.
    resumed = push("--run-id", "d1", "--resume")
    assert resumed.exit_code == 0, resumed.output
    assert "dev look: run d1's own, logged when it was first pushed" in resumed.stdout
    for i in range(3, 7):
        log_dev_use(f"run-{i}", "earlier look", path=dev_log)
    again = push("--run-id", "d1", "--resume")
    assert again.exit_code == 0, again.output
    assert [logged for _, logged, _ in read_uses(dev_log)].count("d1") == 1
    assert len(read_uses(dev_log)) == 6
    spent = push("--run-id", "d2")
    assert spent.exit_code == 1 and "had its 6 looks" in spent.stderr


@pytest.mark.phase(3)
def test_a_push_kaggle_started_logs_its_look_and_is_recorded_whatever_fails(
    repo: Path, dev_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    fake = _use(FakeKaggle(head), monkeypatch)
    dev = repo.parent / "dev.yaml"
    dev.write_text("model: gold\nbackend: fake\nsplit: dev\nops: {est_gpu_hours: 0.5}\n")

    def push(run_id: str) -> Any:
        args = ("--job", "eval", "--config", str(dev), "--reason", "baseline", "--yes")
        return _run(*args, "--run-id", run_id)

    # kaggle 2.2.5's line for a push the server gave no number: the kernel runs, and its
    # latest version is this push.
    fake.push_output = (
        "Kernel version successfully pushed.  Please check progress at "
        f"https://www.kaggle.com/code/{OWNER}/forge-v1\n"
    )
    versionless = push("v1")
    assert versionless.exit_code == 0, versionless.output
    assert "dev look 1 of 6 logged in docs/dev-usage.md" in versionless.stdout
    assert f"run v1: kernel {OWNER}/forge-v1, its latest version" in versionless.stdout
    assert _record(repo, "v1")["kernels"][0]["version"] is None
    status = CliRunner().invoke(cli.app, ["kaggle", "status", "v1"])
    assert status.exit_code == 0 and status.stdout.startswith(f"{OWNER}/forge-v1: COMPLETE")
    fake.push_output = None

    # The dev-usage write fails: the push is recorded all the same, so status, logs, wait
    # and --resume find the kernel, and the error says what is missing.
    monkeypatch.setenv("FORGE_DEV_USAGE", str(repo.parent))
    unlogged = push("v2")
    assert unlogged.exit_code == 1, unlogged.output
    assert "its dev look is not logged" in unlogged.stderr
    assert f"Kaggle runs {OWNER}/forge-v2 version 1 all the same, so don't push it again" in (
        unlogged.stderr
    )
    assert _record(repo, "v2")["kernels"][0]["kernel"] == f"{OWNER}/forge-v2"
    monkeypatch.setenv("FORGE_DEV_USAGE", str(dev_log))

    # Another push took the last look while this one went up: Kaggle runs it anyway.
    def taken(run_id: str, reason: str) -> int:
        raise kaggle_job.DevBudgetExceeded("dev split already used 6 of 6 times")

    monkeypatch.setattr(kaggle_job, "log_dev_use", taken)
    raced = push("v3")
    assert raced.exit_code == 1, raced.output
    assert "its dev look is not logged (dev split already used 6 of 6 times)" in raced.stderr
    assert _record(repo, "v3")["kernels"][0]["kernel"] == f"{OWNER}/forge-v3"

    # Both steps fail, and the error names both.
    blocked = repo.parent / "runs-is-a-file"
    blocked.write_text("")
    monkeypatch.setenv("FORGE_RUNS_DIR", str(blocked))
    neither = push("v4")
    assert neither.exit_code == 1, neither.output
    assert "its dev look is not logged" in neither.stderr
    assert f"the push is not in {blocked / 'v4' / 'kernels.json'}" in neither.stderr
    assert len(fake.ran("kernels", "push")) == 4


@pytest.mark.phase(3)
def test_a_dev_push_whose_answer_was_lost_spends_its_look_once_kaggle_has_the_kernel(
    repo: Path, dev_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    fake = _use(FakeKaggle(head), monkeypatch)
    dev = repo.parent / "dev.yaml"
    dev.write_text("model: gold\nbackend: fake\nsplit: dev\nops: {est_gpu_hours: 0.5}\n")

    def push(run_id: str) -> Any:
        args = ("--job", "eval", "--config", str(dev), "--reason", "baseline", "--yes")
        return _run(*args, "--run-id", run_id)

    # The CLI fails after Kaggle saved the kernel: Kaggle has it, so it runs and may read
    # dev. The push is recorded, its look logged, and the lost answer is a warning.
    fake.push_failure = LOST
    made = push("lost1")

    assert made.exit_code == 0, made.output
    kernel = f"{OWNER}/forge-lost1"
    assert "warning: kaggle kernels push -p " in made.stderr
    assert "exited 1: ReadTimeout: HTTPSConnectionPool" in made.stderr
    assert f"but Kaggle has {kernel}, so the push was made and only its answer was lost" in (
        made.stderr
    )
    assert "dev look 1 of 6 logged in docs/dev-usage.md" in made.stdout
    assert f"run lost1: kernel {kernel}, its latest version" in made.stdout
    assert read_uses(dev_log) == [(_today().isoformat(), "lost1", "baseline")]
    entry = _record(repo, "lost1")["kernels"][0]
    assert (entry["kernel"], entry["version"]) == (kernel, None)
    status = CliRunner().invoke(cli.app, ["kaggle", "status", "lost1"])
    assert status.exit_code == 0 and status.stdout.startswith(f"{kernel}: COMPLETE")

    # Kaggle has no such kernel: the push never started, so it fails and spends nothing.
    fake.saves = False
    failed = push("lost2")

    assert failed.exit_code == 1, failed.output
    assert "exited 1: ReadTimeout: HTTPSConnectionPool" in failed.stderr
    assert "warning" not in failed.stderr and "unknown" not in failed.stderr
    assert not (repo / "runs" / "lost2").exists()
    assert read_uses(dev_log) == [(_today().isoformat(), "lost1", "baseline")]
    # Each failed push asked about its own new kernel once.
    assert [c[3] for c in fake.ran("kernels", "status")] == [kernel, kernel, f"{kernel[:-1]}2"]

    # The spike's one kernel was made by an earlier push, so a status can't tell its new
    # version from the last: a failed spike push fails as before, without asking.
    spike = str(repo / "configs/spike/throughput.yaml")
    refused = _run("--job", "spike", "--config", spike, "--yes")
    assert refused.exit_code == 1 and "exited 1: ReadTimeout" in refused.stderr
    assert len(fake.ran("kernels", "status")) == 3


@pytest.mark.phase(3)
def test_bad_kernel_slugs_and_notebook_resumes_are_refused_before_any_upload(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The code dataset holds another commit, so anything that got past the checks would
    # publish one.
    fake = _use(FakeKaggle(_git(repo, "rev-parse", "HEAD")[::-1]), monkeypatch)
    # Good run ids, but no kernel slug: Kaggle's has single hyphens between letters and digits.
    for run_id in ("a--b", "run-"):
        result = _eval(repo, "--yes", "--run-id", run_id)
        assert result.exit_code == 1, result.output
        assert f"run id {run_id} makes the kernel slug forge-{run_id}, which Kaggle refuses" in (
            result.stderr
        )
    # A notebook fetches the datasets but no kernel's output, so it has nothing to resume.
    book = repo.parent / "resume.ipynb"
    result = _eval(repo, "--resume", "--emit-notebook", str(book))
    assert result.exit_code == 1
    assert "a notebook can't fetch the last attempt's kernel output" in result.stderr
    assert fake.commands == [] and not book.exists() and not (repo / "runs").exists()


@pytest.mark.phase(3)
def test_spike_job_pushes_the_throughput_script(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    fake = _use(FakeKaggle(), monkeypatch)
    config = str(repo / "configs/spike/throughput.yaml")
    preflight = _run("--job", "spike", "--config", config, "--preflight", "--yes")
    assert preflight.exit_code == 1 and "has no preflight" in preflight.stderr

    result = _run("--job", "spike", "--config", config, "--yes")

    assert result.exit_code == 0, result.output
    # The spike runs without Forge: no dataset is checked or mounted.
    assert fake.ran("datasets") == []
    meta = json.loads(fake.pushed[0]["kernel-metadata.json"])
    assert (meta["id"], meta["title"]) == (f"{OWNER}/{SPIKE_SLUG}", SPIKE_TITLE)
    assert (meta["code_file"], meta["dataset_sources"]) == ("throughput.py", [])
    assert _baked_argv(fake.pushed[0]["throughput.py"]) == ["run"]
    assert fake.ran("kernels", "push")[0][-2:] == ["-t", "8100"]  # 1.5 h and half again
    run_id = f"{_today():%Y-%m-%d}-throughput-spike-{head[:7]}"
    assert _record(repo, run_id)["job"] == "spike"
    # The same kernel scripts/kaggle_spike.py bundles, so FG-301's steps still meet.
    bundler = load_script(ROOT / "scripts" / "kaggle_spike.py", "forge_spike_bundler")
    assert (bundler.KERNELS["gpu"].slug, bundler.KERNELS["gpu"].title) == (SPIKE_SLUG, SPIKE_TITLE)


@pytest.mark.phase(3)
def test_emit_notebook_valid(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    fake = _use(FakeKaggle(head), monkeypatch)
    canary = "hf_" + "".join(chr(97 + (i * 5) % 26) for i in range(30))
    monkeypatch.setenv("HF_TOKEN", canary)
    path = repo.parent / "job.ipynb"

    result = _eval(repo, "--emit-notebook", str(path))

    assert result.exit_code == 0, result.output
    assert "nothing pushed" in result.stdout
    assert fake.ran("kernels") == [] and fake.ran("quota") == []
    text = path.read_text(encoding="utf-8")
    assert canary not in text
    book = json.loads(text)
    assert (book["nbformat"], book["nbformat_minor"]) == (4, 5)
    assert book["metadata"]["kernelspec"]["language"] == "python"
    ids = [cell["id"] for cell in book["cells"]]
    assert len(set(ids)) == len(ids)
    for cell in book["cells"]:
        assert cell["cell_type"] in ("markdown", "code") and cell["metadata"] == {}
        assert isinstance(cell["source"], list) and all(isinstance(s, str) for s in cell["source"])
        if cell["cell_type"] == "code":
            assert (cell["execution_count"], cell["outputs"]) == (None, [])
            ast.parse("".join(cell["source"]))
    intro, fetch, code, run = ("".join(cell["source"]) for cell in book["cells"])
    # The same stages as a kernel, named in order, from the same script with the same job.
    assert "The job's stages, in order: " + ", ".join(script.STAGES) + "." in intro
    assert f"{OWNER}/forge-code/1" in fetch and f"{OWNER}/forge-spider" in fetch
    assert run == "main()"
    notebook = load_script_text(code, repo.parent / "notebook_entry.py")
    run_id = f"{_today():%Y-%m-%d}-gold-eval-{head[:7]}"
    config = repo / "configs" / "eval" / "job.yaml"
    plan = plan_job("eval", config, owner=OWNER, sha=head, runs_root=repo / "runs")
    assert plan.run_id == run_id
    expected = [*entry_argv(plan, "hoomanesteki/forge-runs"), "--host", "colab", *COLAB_DIRS]
    assert expected == notebook.KERNEL_ARGV
    assert notebook.STAGES == script.STAGES
    assert "__main__" not in code
    assert (ROOT / "kaggle" / "entry.py").read_text(encoding="utf-8").endswith(MAIN_GUARD)


@pytest.mark.phase(3)
def test_wait_ends_on_the_kernel_state_and_the_jobs_own_status(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    fake = _use(FakeKaggle(head), monkeypatch)
    assert _eval(repo, "--yes", "--run-id", "run-1").exit_code == 0
    kernel = f"{OWNER}/forge-run-1/1"
    status = {"status": "finished", "stage": "upload", "stages": {"upload": {"result": "skipped"}}}
    fake.outputs = {"runs/run-1/status.json": json.dumps(status)}
    fake.states = ["QUEUED", "RUNNING", "RUNNING", "COMPLETE"]

    result = CliRunner().invoke(cli.app, ["kaggle", "wait", "run-1", "--poll", "5"])

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        f"{kernel}: QUEUED",
        f"{kernel}: RUNNING",
        f"{kernel}: COMPLETE",
        "job finished at stage upload; upload: skipped",
        # This output holds no run's manifest, so nothing says what to log.
        "ledger: no row, the output has no manifest; use forge ledger add",
    ]
    assert [c[3] for c in fake.ran("kernels", "status")] == [kernel] * 4
    assert script.STATUS_FILE == kaggle_job.STATUS_FILE

    # A job that failed inside a kernel that completed is still a failure.
    failed = {"status": "failed", "stage": "data", "error": "JobError: no data", "stages": {}}
    fake.outputs = {"runs/run-1/status.json": json.dumps(failed)}
    result = CliRunner().invoke(cli.app, ["kaggle", "wait", "run-1"])
    assert result.exit_code == 1
    assert "job failed at stage data; upload: not reached" in result.stdout
    assert "error: JobError: no data" in result.stderr
    # It held its GPU all the same, for the span its status.json gives, which the note
    # quotes for forge ledger add.
    failed.update(started_at="2026-11-16T23:40:00+00:00", ended_at="2026-11-17T00:10:00+00:00")
    fake.outputs = {"runs/run-1/status.json": json.dumps(failed)}
    result = CliRunner().invoke(cli.app, ["kaggle", "wait", "run-1"])
    assert result.exit_code == 1
    assert (
        "ledger: no row, the output has no manifest; use forge ledger add: the kernel held its "
        "GPU 0.50 h, from 2026-11-16T23:40:00+00:00 to 2026-11-17T00:10:00+00:00, so --hours "
        "0.50 --on 2026-11-16\n" in result.stdout
    )
    assert read_rows() == []
    failed["ended_at"] = "not a time"
    fake.outputs = {"runs/run-1/status.json": json.dumps(failed)}
    result = CliRunner().invoke(cli.app, ["kaggle", "wait", "run-1"])
    assert "its status.json gives no span it can count" in result.stdout

    fake.states = ["ERROR"]
    result = CliRunner().invoke(cli.app, ["kaggle", "wait", "run-1"])
    assert result.exit_code == 1
    assert "the kernel ended ERROR; read forge kaggle logs run-1" in result.stderr

    # status maps the worker state; a run pushed from nowhere has no kernel to ask about.
    fake.states = ["CANCEL_ACKNOWLEDGED"]
    result = CliRunner().invoke(cli.app, ["kaggle", "status", "run-1"])
    assert (result.exit_code, result.stdout) == (0, f"{kernel}: CANCEL_ACKNOWLEDGED\n")
    result = CliRunner().invoke(cli.app, ["kaggle", "logs", "missing"])
    assert result.exit_code == 1 and "no kernel was pushed for run missing" in result.stderr


@pytest.mark.phase(3)
def test_output_brings_the_run_into_runs(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _use(FakeKaggle(_git(repo, "rev-parse", "HEAD")), monkeypatch)
    assert _eval(repo, "--yes", "--run-id", "run-1").exit_code == 0
    fake.outputs = {
        "runs/run-1/manifest.json": _kernel_manifest("run-1").decode(),
        "runs/run-1/eval/val-scores.jsonl": "{}\n",
        "preflight/run-1-preflight/manifest.json": "{}",
    }
    stale = repo / "runs" / "run-1" / "eval" / "old.jsonl"
    stale.parent.mkdir(parents=True)
    stale.write_text("from an earlier pull\n")

    result = CliRunner().invoke(cli.app, ["kaggle", "output", "run-1"])

    assert result.exit_code == 0, result.output
    run = repo / "runs" / "run-1"
    assert sorted(p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file()) == [
        "eval/val-scores.jsonl",
        "kernel/forge-run-1.log",
        "kernel/preflight/run-1-preflight/manifest.json",
        "kernels.json",
        "manifest.json",
    ]
    # The run it holds ended, so its session gets its ledger row.
    logged = "eval run, finished\nledger: added 0.50 h on kaggle for run run-1\n"
    assert result.stdout.endswith(logged)


@pytest.mark.phase(3)
def test_output_leaves_no_file_of_an_earlier_attempt(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _use(FakeKaggle(_git(repo, "rev-parse", "HEAD")), monkeypatch)
    assert _eval(repo, "--yes", "--run-id", "run-1").exit_code == 0
    runs = repo / "runs"
    (runs / "run-2").mkdir()
    (runs / "run-2" / "error.txt").write_text("another run's\n")
    fake.outputs = {
        "runs/run-1/manifest.json": _kernel_manifest("run-1", status="failed").decode(),
        "runs/run-1/error.txt": "Traceback: the session ended\n",
        "runs/run-1/eval/val-scores.jsonl": "{}\n",
        "preflight/run-1-preflight/manifest.json": "{}",
    }
    first = CliRunner().invoke(cli.app, ["kaggle", "output", "run-1"])
    assert first.exit_code == 0, first.output
    assert (runs / "run-1" / "error.txt").is_file()

    # The attempt that resumed it finished, so its output has no error.txt, nor a preflight.
    fake.outputs = {
        "runs/run-1/manifest.json": _kernel_manifest("run-1").decode(),
        "runs/run-1/eval/val-scores.jsonl": "{}\n{}\n",
    }
    second = CliRunner().invoke(cli.app, ["kaggle", "output", "run-1"])

    assert second.exit_code == 0, second.output
    run = runs / "run-1"
    assert sorted(p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file()) == [
        "eval/val-scores.jsonl",
        "kernel/forge-run-1.log",
        "kernels.json",
        "manifest.json",
    ]
    assert (run / "eval" / "val-scores.jsonl").read_text() == "{}\n{}\n"
    assert (runs / "run-2" / "error.txt").read_text() == "another run's\n"
    # Nothing of the download is left beside the runs.
    assert sorted(p.name for p in runs.iterdir()) == ["run-1", "run-2"]


START = datetime(2026, 11, 16, 1, 0, tzinfo=UTC)


def _kernel_manifest(run_id: str, **changes: Any) -> bytes:
    """The manifest of an eval job that ran in a kernel, finished half an hour in."""
    fields: dict[str, Any] = {
        "run_id": run_id,
        "kind": "eval",
        "git_sha": "c" * 40,
        "config": {"split": "val"},
        "config_hash": "4" * 64,
        "data_manifest_hash": "5" * 64,
        "model_ref": "gold",
        "platform": "kaggle",
        "gpu": "Tesla T4",
        "seed": 0,
        "started_at": START,
        "ended_at": START + timedelta(minutes=30),
        "status": "finished",
        **changes,
    }
    return manifest_bytes(RunManifest(**fields))


def _ended(run_id: str, status: str, sessions: list[Session]) -> dict[str, str]:
    """A kernel's output for a job that ended with ``status``: status.json, the manifest
    and the run's sessions, the last one the kernel's own."""
    last = sessions[-1]
    assert last.ended_at is not None
    record = {"status": status, "stage": "upload", "stages": {}}
    record |= {"started_at": last.started_at.isoformat(), "ended_at": last.ended_at.isoformat()}
    manifest = _kernel_manifest(run_id, status=status, ended_at=last.ended_at)
    return {
        f"runs/{run_id}/status.json": json.dumps({**record, "session": len(sessions)}),
        f"runs/{run_id}/manifest.json": manifest.decode(),
        f"runs/{run_id}/sessions.json": json.dumps([s.to_json() for s in sessions]),
    }


@pytest.mark.phase(3)
def test_wait_and_output_log_each_kernel_session_once(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _use(FakeKaggle(_git(repo, "rev-parse", "HEAD")), monkeypatch)
    assert _eval(repo, "--yes", "--run-id", "run-1").exit_code == 0
    ledger = repo.parent / "ledger.md"
    # The kernel's whole session, setup and upload included, which the time guard ended.
    first = Session(START, START + timedelta(hours=11, minutes=15), "stopped_time_guard")
    fake.outputs = _ended("run-1", "stopped_time_guard", [first])

    waited = CliRunner().invoke(cli.app, ["kaggle", "wait", "run-1"])

    assert waited.exit_code == 1, waited.output
    assert waited.stdout.endswith(
        "eval run, stopped_time_guard\nledger: added 11.25 h on kaggle for run run-1\n"
    )
    assert "carry it on with forge kaggle run --resume --run-id run-1" in waited.stderr
    # Fetching the output after, or pulling it, logs the session no second time.
    for args in (["kaggle", "output", "run-1"], ["runs", "pull", "run-1", "--store", "kaggle"]):
        again = CliRunner().invoke(cli.app, args)
        assert again.exit_code == 0, again.output
        assert again.stdout.endswith("ledger: run run-1 has its row already\n")

    # Resumed in a second kernel two days later: its output holds both sessions, and only
    # the second is new, with its own hour.
    assert _eval(repo, "--yes", "--run-id", "run-1", "--resume").exit_code == 0
    later = START + timedelta(days=2)
    second = Session(later, later + timedelta(hours=1), "finished")
    fake.outputs = _ended("run-1", "finished", [first, second])

    waited = CliRunner().invoke(cli.app, ["kaggle", "wait", "run-1"])

    assert waited.exit_code == 0, waited.output
    assert waited.stdout.endswith(
        "ledger: session 1 of run run-1 has its row already\n"
        "ledger: added 1.00 h on kaggle for session 2 of run run-1\n"
    )
    assert [(row.hours, row.session) for row in read_rows(ledger)] == [(11.25, 1), (1.0, 2)]
    # The output wait read went to a scratch folder: runs/ holds what output fetched.
    assert json.loads((repo / "runs" / "run-1" / "sessions.json").read_text()) == [first.to_json()]


@pytest.mark.phase(3)
def test_client_reads_what_the_kaggle_cli_prints() -> None:
    def answer(stdout: str, code: int = 0) -> KaggleClient:
        return KaggleClient(run=lambda args: _done(list(args), code, stdout), executable="kaggle")

    status = answer(
        'o/k has status "KernelWorkerStatus.ERROR"\nFailure message: "out of memory"\n'
    ).kernel_status("o/k")
    assert (status.state, status.failure, status.done) == ("ERROR", "out of memory", True)
    assert not answer('o/k has status "KernelWorkerStatus.QUEUED"').kernel_status("o/k").done
    with pytest.raises(KaggleError, match="printed"):
        answer('o/k has status "KernelWorkerStatus.PAUSED"').kernel_status("o/k")
    assert answer(QUOTA).gpu_hours_left() == 26.5
    with pytest.raises(KaggleError, match="shows no GPU quota"):
        answer("No quota information available\n").gpu_hours_left()
    with pytest.raises(KaggleError, match="was refused: Kernel push error: bad title"):
        answer("Kernel push error: bad title\n").push_kernel(Path("k"), timeout_s=60, gpu=True)
    pushed = answer(_pushed("o/forge-x")).push_kernel(Path("k"), timeout_s=60, gpu=False)
    assert (pushed.kernel, pushed.version, pushed.dropped) == ("o/forge-x", 1, ())
    dropped = "The following are not valid kernel sources and could not be added to the kernel: []"
    pushed = answer(f"{dropped}\n{_pushed('o/forge-x')}").push_kernel(
        Path("k"), timeout_s=60, gpu=True
    )
    assert (pushed.version, pushed.dropped, pushed.kernel_source_dropped) == (1, (dropped,), True)
    datasets = dropped.replace("kernel sources", "dataset sources")
    pushed = answer(f"{datasets}\n{_pushed('o/forge-x')}").push_kernel(
        Path("k"), timeout_s=60, gpu=True
    )
    assert (pushed.dropped, pushed.kernel_source_dropped) == ((datasets,), False)
    with pytest.raises(KaggleError, match="printed"):
        answer("ready").dataset_version("o/forge-code")


@pytest.mark.phase(3)
@pytest.mark.parametrize(
    ("stdout", "code", "version"),
    [
        # kaggle 2.2.5's line when the server sends no version number.
        ("Kernel version successfully pushed.  Please check progress at {url}\n", 0, None),
        # Another wording, with no push error: Kaggle has the kernel.
        ("Kernel version 3 pushed. Please check progress at {url}\n", 0, 3),
        # The CLI failed after it said the push went through.
        ("Kernel version 2 successfully pushed.  Please check progress at {url}\n", 1, 2),
    ],
)
def test_a_push_kaggle_started_is_never_refused(
    stdout: str, code: int, version: int | None
) -> None:
    output = stdout.format(url="https://www.kaggle.com/code/o/forge-x")
    client = KaggleClient(run=lambda args: _done(list(args), code, output), executable="kaggle")

    pushed = client.push_kernel(Path("k"), timeout_s=60, gpu=True)

    assert (pushed.kernel, pushed.version, pushed.output) == ("o/forge-x", version, output.strip())


@pytest.mark.phase(3)
def test_a_push_kaggle_never_started_says_how_it_failed() -> None:
    def answer(stdout: str, code: int, stderr: str = "") -> KaggleClient:
        return KaggleClient(
            run=lambda args: _done(list(args), code, stdout, stderr), executable="kaggle"
        )

    with pytest.raises(KaggleError, match="was refused: Kernel push error: bad title"):
        answer("Kernel push error: bad title\n", 0).push_kernel(Path("k"), timeout_s=60, gpu=True)
    # A CLI that fails before Kaggle answers started nothing, and says how it failed.
    with pytest.raises(KaggleError, match="exited 1: Connection reset") as raised:
        answer("", 1, "Connection reset\n").push_kernel(Path("k"), timeout_s=60, gpu=True)
    assert "refused" not in str(raised.value)


@pytest.mark.phase(3)
def test_a_push_whose_answer_was_lost_is_made_only_if_kaggle_has_its_kernel(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "kernel"
    folder.mkdir()
    (folder / "kernel-metadata.json").write_text(json.dumps({"id": "o/forge-x"}))

    def client(code: int, stdout: str = "", stderr: str = "") -> tuple[KaggleClient, list[Any]]:
        commands: list[list[str]] = []

        def run(args: Sequence[str]) -> Any:
            commands.append(list(args))
            if list(args)[1:3] == ["kernels", "push"]:
                return _done(list(args), 1, stderr=LOST)
            return _done(list(args), code, stdout, stderr)

        return KaggleClient(run=run, executable="kaggle"), commands

    def push(kaggle: KaggleClient) -> Any:
        return kaggle.push_kernel(folder, timeout_s=60, gpu=True, new_kernel=True)

    # Kaggle has the kernel: the push was made, and only its answer was lost.
    saved, commands = client(0, 'o/forge-x has status "KernelWorkerStatus.QUEUED"\n')
    pushed = push(saved)
    assert (pushed.kernel, pushed.version, pushed.dropped) == ("o/forge-x", None, ())
    assert pushed.lost is not None
    assert pushed.lost.startswith(f"kaggle kernels push -p {folder} --accelerator NvidiaTeslaT4")
    assert "exited 1: " in pushed.lost and "Read timed out." in pushed.lost
    assert commands[1] == ["kaggle", "kernels", "status", "o/forge-x"]
    # Kaggle has no such kernel, whichever way kaggle 2.2.5 says so: it fails as before.
    for answer in (DENIED.format(kernel="o/forge-x"), NO_KERNEL):
        with pytest.raises(KaggleError, match="exited 1: ReadTimeout") as raised:
            push(client(1, stderr=answer)[0])
        assert "unknown" not in str(raised.value) and "refused" not in str(raised.value)
    # Kaggle can't be asked: the push fails, saying it may have been saved after all.
    with pytest.raises(KaggleError) as unknown:
        push(client(1, stderr="ConnectionError: Connection reset by peer\n")[0])
    message = str(unknown.value)
    assert "exited 1: ReadTimeout" in message
    assert "whether Kaggle saved o/forge-x all the same is unknown" in message
    assert "Connection reset by peer" in message
    assert message.endswith("check kaggle kernels status o/forge-x before pushing again")
    # A kernel an earlier push made isn't asked about.
    earlier, commands = client(0, 'o/forge-x has status "KernelWorkerStatus.QUEUED"\n')
    with pytest.raises(KaggleError, match="exited 1: ReadTimeout"):
        earlier.push_kernel(folder, timeout_s=60, gpu=True)
    assert [command[1:3] for command in commands] == [["kernels", "push"]]
    # find_kernel tells a missing kernel from any other failure.
    assert client(1, stderr=NO_KERNEL)[0].find_kernel("o/forge-x") is None
    with pytest.raises(KaggleError, match="exited 1: 500 Server Error"):
        client(1, stderr="500 Server Error: Internal Server Error\n")[0].find_kernel("o/k")


def _baked_argv(source: str) -> list[str]:
    line = next(line for line in source.splitlines() if line.startswith("KERNEL_ARGV"))
    argv: list[str] = json.loads(line.split("=", 1)[1])
    return argv


def load_script_text(code: str, path: Path) -> Any:
    path.write_text(code, encoding="utf-8")
    return load_script(path, "forge_notebook_entry")


@pytest.mark.phase(3)
def test_run_outside_a_checkout_is_one_error_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = tmp_path / "plain"
    (plain / "configs" / "eval").mkdir(parents=True)
    (plain / "configs" / "eval" / "job.yaml").write_text(
        "model: m\nbackend: fake\nsplit: val\nops:\n  est_gpu_hours: 0.5\n", encoding="utf-8"
    )
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    monkeypatch.setenv("FORGE_ROOT", str(plain))
    monkeypatch.setenv("FORGE_KAGGLE_USER", OWNER)
    result = _run("--job", "eval", "--config", str(plain / "configs/eval/job.yaml"))
    assert result.exit_code == 1, result.output
    assert f"error: {plain} is not in a git checkout" in result.stderr
    assert isinstance(result.exception, SystemExit)


# 19:30 on a Sunday evening in Honolulu, ten hours west of UTC, is already Monday in UTC.
EVENING = datetime(2026, 11, 16, 5, 30, tzinfo=UTC)
LOCAL_DAY, UTC_DAY = date(2026, 11, 15), date(2026, 11, 16)


class _Evening(datetime):
    """A clock stopped at ``EVENING``."""

    @classmethod
    def now(cls, tz: tzinfo | None = None) -> datetime:  # type: ignore[override]
        local = EVENING.astimezone().replace(tzinfo=None)
        return EVENING.astimezone(tz) if tz is not None else local


@pytest.fixture
def far_west_evening(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The machine ten hours west of UTC, its clocks stopped at ``EVENING``: the local date
    is the day before the UTC one. The zone is put back afterwards."""
    before = os.environ.get("TZ")
    os.environ["TZ"] = "HST10"  # POSIX: needs no zone database
    time.tzset()
    # Both clocks a date can come from: the UTC one and, through time.time, the local one.
    monkeypatch.setattr(clock, "datetime", _Evening)
    monkeypatch.setattr(time, "time", EVENING.timestamp)
    try:
        yield
    finally:
        if before is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = before
        time.tzset()


@pytest.mark.phase(3)
def test_dates_follow_utc_far_west_of_it(
    repo: Path, dev_log: Path, far_west_evening: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (date.today(), clock.utc_today()) == (LOCAL_DAY, UTC_DAY)
    head = _git(repo, "rev-parse", "HEAD")
    fake = _use(FakeKaggle(head), monkeypatch)
    dev = repo.parent / "dev.yaml"
    dev.write_text("model: gold\nbackend: fake\nsplit: dev\nops: {est_gpu_hours: 0.5}\n")
    push = ["--job", "eval", "--config", str(dev), "--reason", "baseline", "--yes"]
    # Hours logged on today's UTC date count in this week's window, so the push is checked
    # against them.
    _spent(repo, 29.8, on=UTC_DAY)

    refused = _run(*push)

    assert refused.exit_code == 1, refused.output
    assert "ledger: 0.20 of 30 h left this week" in refused.stdout
    assert fake.ran("kernels") == []

    pushed = _run(*push, "--allow-over-budget")

    assert pushed.exit_code == 0, pushed.output
    run_id = f"{UTC_DAY:%Y-%m-%d}-gold-eval-{head[:7]}"
    assert _record(repo, run_id)["kernels"][0]["over_budget"] is True
    # The dev look the push spends is dated as its run id is.
    assert read_uses(dev_log) == [(UTC_DAY.isoformat(), run_id, "baseline")]
    # So is a row added by hand, and the window counts it with the others.
    by_hand = ["--platform", "kaggle", "--hours", "0.1", "--purpose", "the push, by hand"]
    added = CliRunner().invoke(cli.app, ["ledger", "add", *by_hand, "--run-id", run_id])
    assert added.exit_code == 0, added.output
    assert [row.date for row in read_rows(repo.parent / "ledger.md")] == [UTC_DAY, UTC_DAY]
    week = summary()
    assert week.quota_hours_window == pytest.approx(29.9)
    assert "of 30 (remaining 0.10)" in week.render()
