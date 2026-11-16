"""kaggle/entry.py running a real eval job on the fixture data, with pip stood in for.

The job is what a kernel runs: ``run_eval`` with the job's options, a preflight on five
examples first, and the report at the end, outside any git checkout, so the manifest's
commit comes from the code dataset. A vLLM job runs with vLLM faked.
"""

from __future__ import annotations

import functools
import json
import logging
import subprocess
import sys
import types
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

import httpx2
import pytest
from huggingface_hub.errors import HfHubHTTPError

from forge.core.types import RunManifest, Score
from forge.data.splits import load_val
from forge.eval import runner
from forge.eval.config import EvalConfig
from forge.ops.dev_usage import read_uses
from forge.ops.hub import HubRunStore
from forge.ops.sessions import read_sessions
from forge.train import sft
from forge.train.config import OpsConfig, TrainConfig, load_train_config
from tests.fakes import REPO_URL, FakeHub, FakeVllm, gold_reply
from tests.kernel import SHA, argv, build_inputs, kernel_folder, mount_output
from tests.spike import entry

ROOT = Path(__file__).resolve().parents[2]
script = entry()
# Built at run time, so no literal here looks like a real token.
TOKEN = "hf_" + "".join(chr(97 + (i * 11) % 26) for i in range(30))


def _scores(run_dir: Path, split: str = "val") -> list[Score]:
    lines = (run_dir / "eval" / f"{split}-scores.jsonl").read_text().splitlines()
    return [Score.model_validate_json(line) for line in lines]


@pytest.mark.phase(3)
def test_entry_runs_an_eval_job_end_to_end(
    tmp_path: Path, spider_mini: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel_env = kernel_folder(tmp_path, monkeypatch)
    build_inputs(kernel_env, spider_mini, zipped=True)
    options = {"model": "gold", "backend": "fake", "limit": 6, "allow_over_budget": True}

    assert script.main(argv(kernel_env, options=options), run=lambda command: None) == 0

    work = kernel_env / "working"
    status = json.loads((work / "runs" / "run-1" / "status.json").read_text())
    assert (status["status"], status["error"]) == ("finished", None), status["error"]
    run = RunManifest.model_validate_json((work / "runs" / "run-1" / "manifest.json").read_text())
    assert (run.status, run.git_sha, run.platform) == ("finished", SHA, "kaggle")
    laptop = EvalConfig(model="gold", backend="fake", limit=6, host="kaggle")
    assert run.config == {**laptop.manifest_config(), "allow_over_budget": True}
    assert (work / "runs" / "run-1" / "eval" / "val-report.md").is_file()
    scores = (work / "runs" / "run-1" / "eval" / "val-scores.jsonl").read_text().splitlines()
    assert len(scores) == 6
    # The run's one session is the whole kernel's, which held the GPU from start to end.
    (session,) = read_sessions(work / "runs" / "run-1") or []
    assert (session.status, status["session"]) == ("finished", 1)
    assert session.started_at == datetime.fromisoformat(status["started_at"])
    assert session.ended_at == datetime.fromisoformat(status["ended_at"])
    assert run.started_at >= session.started_at and run.gpu_hours > 0

    # The preflight ran the same job on five examples, in a run of its own.
    preflight = work / "preflight" / "run-1-preflight"
    first = RunManifest.model_validate_json((preflight / "manifest.json").read_text())
    assert (first.status, first.config["limit"], first.git_sha) == ("finished", 5, SHA)
    assert len((preflight / "eval" / "val-scores.jsonl").read_text().splitlines()) == 5


@pytest.mark.phase(3)
def test_entry_runs_a_vllm_job_on_one_engine(
    tmp_path: Path, spider_mini: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # vLLM faked. A second engine might not fit on the T4 beside the preflight's, so the
    # preflight and the job share the one the preflight loaded.
    vllm = FakeVllm(gold_reply(load_val(spider_mini))).install(monkeypatch)
    kernel_env = kernel_folder(tmp_path, monkeypatch)
    build_inputs(kernel_env, spider_mini)
    options = {"model": "org/model@abc123", "backend": "vllm", "limit": 6}

    assert script.main(argv(kernel_env, options=options), run=lambda command: None) == 0

    work = kernel_env / "working"
    status = json.loads((work / "runs" / "run-1" / "status.json").read_text())
    assert (status["status"], status["error"]) == ("finished", None), status["error"]
    assert len(vllm.engines) == 1
    # Five preflight prompts, then the job's six, one request a batch.
    assert [len(call["sampling_params"]) for _, call in vllm.calls] == [5, 6]
    run = RunManifest.model_validate_json((work / "runs" / "run-1" / "manifest.json").read_text())
    assert (run.status, run.platform, run.config["backend"]) == ("finished", "kaggle", "vllm")
    scores = (work / "runs" / "run-1" / "eval" / "val-scores.jsonl").read_text().splitlines()
    assert len(scores) == 6
    preflight = work / "preflight" / "run-1-preflight" / "manifest.json"
    assert RunManifest.model_validate_json(preflight.read_text()).status == "finished"


@pytest.mark.phase(3)
def test_eval_job_stops_at_its_deadline_then_uploads_and_resumes(
    tmp_path: Path, spider_mini: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [0.0]
    gold = gold_reply(load_val(spider_mini))

    def slow(text: str, sample_idx: int) -> str:
        now[0] += 10.0  # each reply takes 10 s of the kernel's time
        return gold(text, sample_idx)

    FakeVllm(slow).install(monkeypatch)
    hub = FakeHub()
    monkeypatch.setattr("huggingface_hub.HfApi", hub.client)
    # Two questions a batch, so the guard has batches to stop between.
    monkeypatch.setattr(runner, "run_eval", functools.partial(runner.run_eval, batch_size=2))
    options = {"model": "org/model@abc123", "backend": "vllm", "limit": 6}
    first = kernel_folder(tmp_path / "first", monkeypatch)
    build_inputs(first, spider_mini)
    monkeypatch.setenv("HF_TOKEN", TOKEN)

    # Pushed with a two-minute timeout. The preflight's five replies take 50 s, then each
    # batch 20 s: after two, 30 s are left, within a batch and the 12 s kept back.
    kernel_argv = argv(first, "--timeout-s", "120", options=options)
    assert script.main(kernel_argv, run=lambda c: None, clock=lambda: now[0]) == 0

    run_dir = first / "working" / "runs" / "run-1"
    status = json.loads((run_dir / "status.json").read_text())
    assert (status["status"], status["stage"], status["error"]) == (
        "stopped_time_guard",
        "upload",
        None,
    )
    assert status["stages"]["job"]["result"] == {"status": "stopped_time_guard"}
    assert status["stages"]["upload"]["result"] == "uploaded to forge-owner/forge-runs"
    stopped = RunManifest.model_validate_json((run_dir / "manifest.json").read_text())
    assert stopped.status == "stopped_time_guard"
    assert [s.ex_official for s in _scores(run_dir)] == [None] * 4
    # A stopped run gets its report once it finishes; the store has it, ready to resume.
    assert not (run_dir / "eval" / "val-report.md").exists()
    uploaded = RunManifest.model_validate_json(hub.files["runs/run-1/manifest.json"])
    assert uploaded.status == "stopped_time_guard"
    assert len(hub.files["runs/run-1/eval/val-scores.jsonl"].splitlines()) == 4

    second = kernel_folder(tmp_path / "second", monkeypatch)
    build_inputs(second, spider_mini)
    mount_output(first, second, "forge-run-1")
    resume = argv(second, "--timeout-s", "3600", "--resume", options=options)
    assert script.main(resume, run=lambda c: None, clock=lambda: now[0]) == 0

    run_dir = second / "working" / "runs" / "run-1"
    status = json.loads((run_dir / "status.json").read_text())
    assert (status["status"], status["error"], status["session"]) == ("finished", None, 2)
    scores = _scores(run_dir)
    assert len(scores) == 6 and all(s.ex_fast and s.ex_official for s in scores)
    assert (run_dir / "eval" / "val-report.md").is_file()
    sessions = read_sessions(run_dir) or []
    assert [s.status for s in sessions] == ["stopped_time_guard", "finished"]


@pytest.mark.phase(3)
def test_entry_loads_dev_once_in_a_dev_job(
    tmp_path: Path, with_dev: Path, dev_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel_env = kernel_folder(tmp_path, monkeypatch)
    build_inputs(kernel_env, with_dev)
    options = {"model": "gold", "backend": "fake", "split": "dev", "reason": "baseline"}

    assert script.main(argv(kernel_env, options=options), run=lambda command: None) == 0

    work = kernel_env / "working"
    status = json.loads((work / "runs" / "run-1" / "status.json").read_text())
    assert (status["status"], status["error"]) == ("finished", None), status["error"]
    run = RunManifest.model_validate_json((work / "runs" / "run-1" / "manifest.json").read_text())
    assert (run.status, run.config["split"]) == ("finished", "dev")
    assert (work / "runs" / "run-1" / "eval" / "dev-report.md").is_file()
    # The preflight answered val, so the job's load was the one look, logged in the
    # kernel's output; the repo's log is forge kaggle run's to write, at the push.
    preflight = work / "preflight" / "run-1-preflight" / "manifest.json"
    first = RunManifest.model_validate_json(preflight.read_text())
    assert (first.status, first.config["split"], first.config["limit"]) == ("finished", "val", 5)
    assert [(run_id, reason) for _, run_id, reason in read_uses(work / "dev-usage.md")] == [
        ("run-1", "baseline")
    ]
    assert not dev_log.exists()


def _commit_failed(text: str) -> HfHubHTTPError:
    """The Hub answering 503 to a commit, which huggingface_hub sends once."""
    request = httpx2.Request("POST", f"{REPO_URL}/commit/main")
    return HfHubHTTPError(text, response=httpx2.Response(503, request=request))


@pytest.mark.phase(3)
@pytest.mark.parametrize(
    ("make_error", "retries"),
    [
        # Refused at once, so only status.json quotes it.
        (ValueError, 0),
        # Network failures the run store tries again, warning on stderr each time.
        (ConnectionError, 3),
        (_commit_failed, 3),
    ],
)
def test_nothing_the_kernel_writes_or_prints_holds_a_secret(
    tmp_path: Path,
    spider_mini: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    make_error: Callable[[str], Exception],
    retries: int,
) -> None:
    kernel_env = kernel_folder(tmp_path, monkeypatch)
    build_inputs(kernel_env, spider_mini)

    class UserSecretsClient:
        def get_secret(self, label: str) -> str:
            if label != "HF_TOKEN":
                raise LookupError(f"no secret {label}")
            # Pasted with its line break, which huggingface_hub drops before it sends it.
            return TOKEN + "\n"

    monkeypatch.setitem(
        sys.modules, "kaggle_secrets", types.SimpleNamespace(UserSecretsClient=UserSecretsClient)
    )
    hub = FakeHub()
    error = make_error(f"401 Unauthorized, Authorization: Bearer {TOKEN}")
    hub.fail["upload_folder"] = [error] * (retries + 1)
    monkeypatch.setattr("huggingface_hub.HfApi", hub.client)
    waits: list[float] = []
    monkeypatch.setitem(HubRunStore.__init__.__kwdefaults__, "sleep", waits.append)
    # As in a kernel, where nothing sets up logging, Python's last resort prints forge's
    # warnings on stderr, which Kaggle keeps as the kernel's log.
    monkeypatch.setattr(logging.getLogger("forge"), "propagate", False)
    options = {"model": "gold", "backend": "fake", "limit": 3}
    factory = logging.getLogRecordFactory()

    assert script.main(argv(kernel_env, "--no-preflight", options=options), run=lambda c: None) == 0

    run_dir = kernel_env / "working" / "runs" / "run-1"
    status = json.loads((run_dir / "status.json").read_text())
    assert (status["status"], status["stage"]) == ("finished", "upload")
    assert status["stages"]["upload"]["result"].startswith("failed: ")
    assert status["stages"]["upload"]["result"].endswith(
        f"{type(error).__name__}: 401 Unauthorized, Authorization: Bearer ***"
    )
    assert hub.tokens == [TOKEN]
    assert waits == [2.0, 4.0, 8.0][:retries]
    manifest = RunManifest.model_validate_json((run_dir / "manifest.json").read_text())
    assert manifest.status == "finished"
    printed = capsys.readouterr()
    assert TOKEN not in printed.out + printed.err
    warned = [line for line in printed.err.splitlines() if "trying again in" in line]
    assert len(warned) == retries
    assert all(line.endswith(" s") and "Bearer ***" in line for line in warned)
    files = [path for path in (kernel_env / "working").rglob("*") if path.is_file()]
    assert run_dir / "manifest.json" in files and run_dir / "status.json" in files
    assert [path for path in files if TOKEN.encode() in path.read_bytes()] == []
    # The kernel's own records only: the factory it replaced is back.
    assert logging.getLogRecordFactory() is factory


@pytest.mark.phase(3)
def test_a_resume_carries_on_after_attempts_that_failed_before_their_job(
    tmp_path: Path, spider_mini: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gold = gold_reply(load_val(spider_mini))
    replies = [0]

    def flaky(text: str, sample_idx: int) -> str:
        replies[0] += 1
        if replies[0] == 8:  # five for the preflight and two for the first batch, then down
            raise RuntimeError("the GPU fell over")
        return gold(text, sample_idx)

    def pip_fails(command: Sequence[str]) -> None:
        raise subprocess.CalledProcessError(1, list(command))

    FakeVllm(flaky).install(monkeypatch)
    monkeypatch.setattr(runner, "run_eval", functools.partial(runner.run_eval, batch_size=2))
    options = {"model": "org/model@abc123", "backend": "vllm", "limit": 6}

    def attempt(name: str, *, resume: bool = True, pip: script.Runner = lambda c: None) -> Path:
        root = kernel_folder(tmp_path / name, monkeypatch)
        build_inputs(root, spider_mini)
        if resume:
            mount_output(previous, root)
        if name == "third":
            (root / "input" / "forge-spider" / "tables.json").write_text("[]\n")
        args = argv(root, *(["--resume"] if resume else []), options=options)
        assert script.main(args, run=pip) == 0
        return root

    previous = attempt("first", resume=False)
    status = json.loads((previous / "working" / "runs" / "run-1" / "status.json").read_text())
    assert (status["status"], status["stage"]) == ("failed", "job")
    assert len(_scores(previous / "working" / "runs" / "run-1")) == 2
    # Two resumes that fail before their job, at the install and at the data: each output
    # still holds the run, so the next resume carries on from it.
    for name, stage, pip in (("second", "install", pip_fails), ("third", "data", lambda c: None)):
        previous = attempt(name, pip=pip)
        run_dir = previous / "working" / "runs" / "run-1"
        status = json.loads((run_dir / "status.json").read_text())
        assert (status["status"], status["stage"]) == ("failed", stage)
        assert len(_scores(run_dir)) == 2

    last = attempt("fourth")

    run_dir = last / "working" / "runs" / "run-1"
    status = json.loads((run_dir / "status.json").read_text())
    assert (status["status"], status["error"], status["session"]) == ("finished", None, 4)
    scores = _scores(run_dir)
    assert len(scores) == 6 and all(s.ex_fast and s.ex_official for s in scores)
    # Every attempt is a session of the run, the early failures too.
    sessions = read_sessions(run_dir) or []
    assert [s.status for s in sessions] == ["failed", "failed", "failed", "finished"]


@pytest.mark.phase(4)
def test_entry_runs_an_sft_job_end_to_end(
    tmp_path: Path, spider_mini: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The tiny model on the CPU, standing in for a T4: the job trains the config's 3 steps
    # after a 2-step preflight, each a run of its own, outside any git checkout.
    kernel_env = kernel_folder(tmp_path, monkeypatch)
    build_inputs(kernel_env, spider_mini)
    smoke = load_train_config(ROOT / "configs" / "sft" / "tiny-smoke.yaml")
    config = smoke.model_copy(
        update={"ops": OpsConfig(checkpoint_every=100, keep_checkpoints_at=(0.5,))}
    )
    options = {"config": config.model_dump(mode="json")}

    job = argv(kernel_env, "--timeout-s", "3600", options=options)
    job[job.index("eval")] = "sft"
    pip: list[list[str]] = []
    assert script.main(job, run=lambda command: pip.append(list(command))) == 0

    work = kernel_env / "working"
    run_dir = work / "runs" / "run-1"
    status = json.loads((run_dir / "status.json").read_text())
    assert (status["status"], status["error"]) == ("finished", None), status["error"]
    # A training job installs the trl stack, never vLLM's (ADR-0006).
    trl_stack = kernel_env / "input" / "forge-code" / "requirements-train-gpu.txt"
    assert pip[-1] == [*script.PIP, "-r", str(trl_stack)]
    assert status["stages"]["install"]["result"]["requirements"] == trl_stack.name
    run = RunManifest.model_validate_json((run_dir / "manifest.json").read_text())
    assert (run.kind, run.status, run.git_sha, run.platform) == ("sft", "finished", SHA, "kaggle")
    assert run.config["sft"]["max_steps"] == 3 and run.outputs["adapter"] == "checkpoints/final"
    # The checkpoint half way, step 1 of 3, kept beside the final adapter for the upload.
    assert run.outputs["kept_at_0.5"] == "checkpoints/step-000001"
    for rel in ("checkpoints/final", "checkpoints/step-000001"):
        assert (run_dir / rel / "adapter_model.safetensors").is_file(), rel
    lines = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [line["step"] for line in lines if "loss" in line] == [1, 2, 3]
    # A training run writes no sessions.json, so its one session is the kernel's, added
    # with the run's status and spanning the whole kernel, as an eval job's is.
    (session,) = read_sessions(run_dir) or []
    assert (session.status, status["session"]) == ("finished", 1)
    assert session.started_at == datetime.fromisoformat(status["started_at"])
    assert session.ended_at == datetime.fromisoformat(status["ended_at"])

    # The preflight trained 2 steps of the same config, saving as the job does.
    preflight = work / "preflight" / "run-1-preflight"
    first = RunManifest.model_validate_json((preflight / "manifest.json").read_text())
    assert (first.status, first.config["sft"]["max_steps"]) == ("finished", 2)
    assert (preflight / "checkpoints" / "final" / "adapter_config.json").is_file()
    # The rows were built in the temp folder, which the output leaves out.
    assert sorted(p.name for p in work.iterdir()) == ["preflight", "runs"]
    assert len(list((kernel_env / "temp" / "sft").iterdir())) == 1


@pytest.mark.phase(4)
@pytest.mark.parametrize(
    ("timeout_s", "guard", "expected"),
    [
        # The push's hour less the tenth kept back for the guard's save and the upload is
        # 54 minutes from the kernel's start, and the setup and preflight took 10.
        (3600, 675.0, 44.0),
        # A config's own guard that comes sooner stays.
        (3600, 30.0, 30.0),
        # A timeout the setup and preflight used up leaves nothing to train in.
        (600, 675.0, None),
    ],
)
def test_sft_jobs_time_guard_stops_before_the_pushs_timeout(
    tmp_path: Path,
    spider_mini: Path,
    monkeypatch: pytest.MonkeyPatch,
    timeout_s: int,
    guard: float,
    expected: float | None,
) -> None:
    now = [0.0]

    def clock() -> float:
        return now[0]

    trained: list[tuple[TrainConfig, object]] = []

    def run_sft(config: TrainConfig, *, clock: object, **kwargs: object) -> object:
        trained.append((config, clock))
        now[0] += 600.0  # each run takes ten minutes of the kernel's time
        return types.SimpleNamespace(manifest=types.SimpleNamespace(status="finished"))

    monkeypatch.setattr(sft, "run_sft", run_sft)
    kernel_env = kernel_folder(tmp_path, monkeypatch)
    build_inputs(kernel_env, spider_mini)
    smoke = load_train_config(ROOT / "configs" / "sft" / "tiny-smoke.yaml")
    config = smoke.model_copy(update={"ops": OpsConfig(time_guard_minutes=guard)})
    job = argv(kernel_env, "--timeout-s", str(timeout_s), options={"config": config.model_dump()})
    job[job.index("eval")] = "sft"

    assert script.main(job, run=lambda command: None, clock=clock) == 0

    status = json.loads((kernel_env / "working" / "runs" / "run-1" / "status.json").read_text())
    # The preflight's 2 steps keep the config's guard, on the kernel's clock.
    assert trained[0][0].ops.time_guard_minutes == guard and trained[0][1] is clock
    if expected is None:
        assert (status["status"], status["stage"]) == ("failed", "job")
        assert "leaves the job no time to train" in status["error"]
        assert len(trained) == 1
        return
    assert (status["status"], status["error"]) == ("finished", None), status["error"]
    (job_config, job_clock) = trained[1]
    assert job_config.ops.time_guard_minutes == pytest.approx(expected) and job_clock is clock
    # Only the guard: the run's config, and so its hash, is the one forge kaggle run checked.
    assert job_config.model_dump(exclude={"ops"}) == config.model_dump(exclude={"ops"})


@pytest.mark.phase(4)
def test_entry_fetches_a_checkpoint_adapter_from_the_run_store(
    tmp_path: Path, spider_mini: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vllm = FakeVllm(gold_reply(load_val(spider_mini))).install(monkeypatch)
    kernel_env = kernel_folder(tmp_path, monkeypatch)
    build_inputs(kernel_env, spider_mini)
    hub = FakeHub()
    monkeypatch.setattr("huggingface_hub.HfApi", hub.client)
    sft_run = "2026-11-20-qwen3-4b-sft-abc1234"
    adapter = f"runs/{sft_run}/checkpoints/step-000242"
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        hub.files[f"{adapter}/{name}"] = f"{name} at step 242".encode()
    hub.files[f"runs/{sft_run}/checkpoints/final/adapter_model.safetensors"] = b"final"
    options = {"model": "org/model@abc123", "adapter": adapter, "backend": "vllm", "limit": 3}

    # Without a token the store can't be read, and the job says so before a model loads.
    assert script.main(argv(kernel_env, options=options), run=lambda command: None) == 0
    work = kernel_env / "working"
    status = json.loads((work / "runs" / "run-1" / "status.json").read_text())
    assert status["status"] == "failed" and "without an HF_TOKEN secret" in status["error"]
    assert vllm.engines == []

    monkeypatch.setenv("HF_TOKEN", "hf_" + "".join(chr(97 + i % 26) for i in range(30)))
    again = argv(kernel_env, options=options)
    again[again.index("run-1")] = "run-2"
    assert script.main(again, run=lambda command: None) == 0

    status = json.loads((work / "runs" / "run-2" / "status.json").read_text())
    assert (status["status"], status["error"]) == ("finished", None), status["error"]
    # That checkpoint alone, at the path the config names, as on the laptop.
    fetched = work / adapter
    assert sorted(p.name for p in fetched.iterdir()) == [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]
    assert not (work / "runs" / sft_run / "checkpoints" / "final").exists()
    (call,) = [kwargs for name, kwargs in hub.calls if name == "snapshot_download"]
    assert call["allow_patterns"] == [f"{adapter}/*"]
    assert vllm.engines[0]["enable_lora"] is True
    run = RunManifest.model_validate_json((work / "runs" / "run-2" / "manifest.json").read_text())
    assert (run.status, run.adapter_ref, run.config["adapter"]) == ("finished", adapter, adapter)
