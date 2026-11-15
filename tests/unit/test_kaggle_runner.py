"""``forge kaggle run`` and the commands that follow its job, with uv and kaggle stood in
for: the checks before a push, the kernel it renders, and what it records."""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import types
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from forge import cli
from forge.eval.config import EvalConfig
from forge.ops import kaggle, kaggle_job
from forge.ops.dev_usage import log_dev_use, read_uses
from forge.ops.kaggle import KaggleClient, KaggleError, check_kernel_metadata
from forge.ops.kaggle_job import (
    COLAB_DIRS,
    KERNEL_BUILD,
    MAIN_GUARD,
    SPIKE_SLUG,
    SPIKE_TITLE,
    entry_argv,
    plan_job,
    write_kernel,
)
from tests.kernel import WHEEL
from tests.spike import entry, load_script

ROOT = Path(__file__).resolve().parents[2]
OWNER = "forge-owner"
CONFIG = "model: gold\nbackend: fake\nlimit: 5\nops:\n  est_gpu_hours: 0.5\n"
NOT_FOUND = "404 Client Error: Not Found\n\nThe resource was not found.\n"
QUOTA = "resource,used,remaining,total,refreshAt\r\nGPU,3.50h,26.50h,30.00h,2026-11-21T00:00\r\n"
script = entry()


def _done(command: list[str], code: int = 0, stdout: str = "", stderr: str = "") -> Any:
    return subprocess.CompletedProcess(command, code, stdout=stdout, stderr=stderr)


class FakeKaggle:
    """Stands in for uv and kaggle: a code dataset whose versions each hold a GIT_SHA, the
    data dataset, the GPU quota, and kernels that go through ``states``."""

    def __init__(self, *code_versions: str, data: bool = True) -> None:
        self.code_versions = list(code_versions)
        self.data = data
        self.pending = 0  # status calls that still show the last upload in processing
        self.quota = QUOTA
        self.push_output: str | None = None
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
            self.pushed.append({p.name: p.read_text(encoding="utf-8") for p in folder.iterdir()})
            kernel = json.loads((folder / "kernel-metadata.json").read_text())["id"]
            return _done(command, stdout=self.push_output or _pushed(kernel))
        if verb == ["kernels", "status"]:
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


def _spent(repo: Path, hours: float) -> None:
    ledger = Path(repo.parent / "ledger.md")
    row = f"| {date.today().isoformat()} | kaggle | T4 | {hours:.2f} | earlier-run | test | 0.00 |"
    ledger.write_text(f"# Compute ledger\n\n| a |\n|---|\n{row}\n", encoding="utf-8")


def _record(repo: Path, run_id: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((repo / "runs" / run_id / "kernels.json").read_text())
    return data


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
    run_id = f"{date.today():%Y-%m-%d}-gold-eval-{head[:7]}"
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
    run_id = f"{date.today():%Y-%m-%d}-gold-eval-{head[:7]}"
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

    # kaggle 2.2.5 exits 0 when it drops a dataset source; the job would start without it.
    fake.data = True
    fake.push_output = (
        "The following are not valid dataset sources and could not be added to the kernel: "
        "['forge-owner/forge-spider']\nKernel version 1 successfully pushed.\n"
    )
    result = _eval(repo, "--yes")
    assert result.exit_code == 1
    assert "was refused: The following are not valid dataset sources" in result.stderr
    assert not (repo / "runs").exists()


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
def test_jobs_a_later_story_builds_and_bad_options_are_refused_before_any_upload(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _use(FakeKaggle(_git(repo, "rev-parse", "HEAD")), monkeypatch)
    config = str(repo / "configs/eval/job.yaml")
    result = _run("--job", "sft", "--config", config, "--yes")
    assert result.exit_code == 1 and "error: --job sft comes with FG-403" in result.stderr

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
    run_id = f"{date.today():%Y-%m-%d}-gold-eval-{head[:7]}"
    # The kernel gets the reason it needs to load dev, and the push logs the look here.
    argv = _baked_argv(fake.pushed[0]["entry.py"])
    options = json.loads(argv[argv.index("--options") + 1])
    assert options == {"model": "gold", "backend": "fake", "split": "dev", "reason": "baseline"}
    assert read_uses(dev_log) == [(date.today().isoformat(), run_id, "baseline")]
    assert "dev look 1 of 6 logged in docs/dev-usage.md" in result.stdout

    # Once the six looks are spent, a dev job is refused before anything is uploaded.
    for i in range(2, 7):
        log_dev_use(f"run-{i}", "earlier look", path=dev_log)
    fake.commands.clear()
    spent = _run("--job", "eval", "--config", str(dev), "--reason", "one more", "--run-id", "x")
    assert spent.exit_code == 1 and "had its 6 looks" in spent.stderr
    assert fake.commands == [] and len(read_uses(dev_log)) == 6


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
    run_id = f"{date.today():%Y-%m-%d}-throughput-spike-{head[:7]}"
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
    run_id = f"{date.today():%Y-%m-%d}-gold-eval-{head[:7]}"
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
        "runs/run-1/manifest.json": "{}",
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
    assert (pushed.kernel, pushed.version) == ("o/forge-x", 1)
    with pytest.raises(KaggleError, match="printed"):
        answer("ready").dataset_version("o/forge-code")


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
