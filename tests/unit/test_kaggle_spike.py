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

from forge.data import prompts as forge_prompts
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
def test_defaults_measure_the_planned_spike() -> None:
    args = spike.parse_args(["run"])
    assert args.out_dir == "/kaggle/working"
    assert args.work_dir.startswith("/kaggle/temp/")
    assert args.models == ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"]
    # Story FG-301, question 4: 100 prompts, here with replies of up to 256 tokens.
    assert (args.prompts, args.max_tokens) == (100, 256)
    assert args.vllm is True
    assert args.secret_labels == ["HF_TOKEN"]
    # Inside the story's 1.5 GPU hours, with room for the report to get out.
    assert args.budget_minutes < 90


@pytest.mark.phase(3)
@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["run", "--prompts", "0"], "must be at least 1, got 0"),
        (["install", "--out", "o", "--venv", "v", "--stack", "pytorch"], "invalid choice"),
        (["vllm-bench", "--model", "m"], "--out"),
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
def test_plan_runs_each_stack_after_its_install(tmp_path: Path) -> None:
    args = _run_args(tmp_path)
    stages = spike.plan_stages(args, python="/usr/bin/python3")
    names = [stage.name for stage in stages]
    small, large = "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"
    assert names == [
        "env",
        "secrets",
        f"download:{small}",
        f"download:{large}",
        "install:vllm",
        f"vllm:{small}",
        f"vllm:{large}",
    ]
    by_name = {stage.name: stage for stage in stages}
    venv = tmp_path / "work" / "venvs" / "vllm"
    bench = by_name[f"vllm:{large}"]
    # Benches run in their stack's venv, need its install and the model, and the last
    # one of each stack deletes the venv.
    assert bench.python == str(venv / "bin" / "python")
    assert bench.needs == ("install:vllm", f"download:{large}")
    assert bench.removes == str(venv)
    assert by_name[f"vllm:{small}"].removes is None
    assert by_name["install:vllm"].python == "/usr/bin/python3"
    # Each bench's arguments parse as its child command, with the run's settings.
    vllm = spike.parse_args([*bench.argv, "--out", "x.json"])
    assert (vllm.command, vllm.model, vllm.prompts, vllm.max_tokens) == (
        "vllm-bench",
        large,
        100,
        256,
    )

    only = spike.plan_stages(_run_args(tmp_path, "--no-vllm"), python="py")
    assert [s.name for s in only][-2:] == [f"download:{small}", f"download:{large}"]


def _fake_runner(outcomes: dict[str, Any], calls: list[str]) -> Any:
    def runner(stage: Any, result_file: Path, timeout: float) -> Any:
        assert timeout > 0
        assert result_file.suffix == ".json"
        calls.append(stage.name)
        return outcomes.get(stage.name, spike.Outcome("ok", result={"stage": stage.name}))

    return runner


@pytest.mark.phase(3)
def test_run_report_records_every_stage(tmp_path: Path) -> None:
    args = _run_args(tmp_path, "--models", "m/small", "m/large")
    venvs = tmp_path / "work" / "venvs"
    (venvs / "vllm").mkdir(parents=True)
    calls: list[str] = []
    outcomes = {
        "download:m/large": spike.Outcome("error", error="RuntimeError: no such repo"),
        "vllm:m/small": spike.Outcome("timeout", error="killed after 60 s, the budget left"),
    }
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
    assert report["args"]["models"] == ["m/small", "m/large"]
    statuses = {s["name"]: (s["status"], s["error"]) for s in report["stages"]}
    assert statuses == {
        "env": ("ok", None),
        "secrets": ("ok", None),
        "download:m/small": ("ok", None),
        "download:m/large": ("error", "RuntimeError: no such repo"),
        "install:vllm": ("ok", None),
        "vllm:m/small": ("timeout", "killed after 60 s, the budget left"),
        # Never started: its model didn't download.
        "vllm:m/large": ("skipped", "needs download:m/large"),
    }
    assert "vllm:m/large" not in calls
    for stage in report["stages"]:
        assert set(stage) >= {"name", "status", "started_at", "seconds", "result", "error"}
    assert report["stages"][0]["result"] == {"stage": "env"}
    # Each stack's venv is gone once its last bench is over.
    assert not (venvs / "vllm").exists()


@pytest.mark.phase(3)
def test_run_skips_stages_once_out_of_budget(tmp_path: Path) -> None:
    args = _run_args(tmp_path, "--models", "m/small", "--budget-minutes", "1")
    ticks: Iterator[int] = iter(range(0, 10_000, 25))  # every reading is 25 s later
    calls: list[str] = []
    report = spike.run(args, runner=_fake_runner({}, calls), clock=lambda: float(next(ticks)))
    statuses = [(s["name"], s["status"]) for s in report["stages"]]
    assert statuses[:1] == [("env", "ok")]
    assert all(status == "skipped" for _, status in statuses[1:])
    assert calls == ["env"]
    errors = {s["name"]: s["error"] for s in report["stages"]}
    assert errors["secrets"] == errors["install:vllm"] == "out of budget (1 min)"
    assert errors["vllm:m/small"] == "needs install:vllm, download:m/small"


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


FAKE_VLLM = textwrap.dedent(
    """
    import time
    from types import SimpleNamespace

    CALLS = []


    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs


    class LLM:
        def __init__(self, **kwargs):
            CALLS.append(("LLM", kwargs))

        def chat(self, messages, params, **kwargs):
            CALLS.append(("chat", len(messages), params.kwargs, kwargs))
            time.sleep(0.05)
            reply = SimpleNamespace(
                text="```sql\\nSELECT 1\\n```", token_ids=list(range(7)), finish_reason="stop"
            )
            return [SimpleNamespace(prompt_token_ids=[0] * 50, outputs=[reply]) for _ in messages]
    """
)


@pytest.mark.phase(3)
def test_vllm_stage_generates_greedy_fp16(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("vllm")
    exec(FAKE_VLLM, fake.__dict__)
    monkeypatch.setitem(sys.modules, "vllm", fake)
    args = spike.parse_args(["vllm-bench", "--out", "x.json", "--model", "Qwen/Qwen3-1.7B"])
    result = spike.vllm_stage(args)

    (_, llm_kwargs), (_, prompts, sampling, chat_kwargs) = fake.CALLS
    # A T4 has no bf16, so the weights load in fp16.
    assert llm_kwargs["dtype"] == "half"
    assert llm_kwargs["model"] == "Qwen/Qwen3-1.7B"
    assert prompts == 100
    assert sampling["temperature"] == 0.0 and sampling["max_tokens"] == 256
    assert chat_kwargs["chat_template_kwargs"] == {"enable_thinking": False}
    assert (result["prompts"], result["prompt_tokens"], result["completion_tokens"]) == (
        100,
        5000,
        700,
    )
    assert result["cut_at_max_tokens"] == 0
    per_second = result["completion_tokens"] / result["generate_seconds"]
    assert result["completion_tokens_per_second"] == pytest.approx(per_second, rel=0.05)
    assert result["memory"] is None  # no CUDA here
    assert result["samples"][0].startswith("```sql")


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


@pytest.mark.phase(3)
def test_install_stage_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A stand-in uv that logs its arguments, so the commands are checked without installing.
    bin_dir, log = tmp_path / "bin", tmp_path / "uv.log"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import sys
            from pathlib import Path
            args = sys.argv[1:]
            with open({str(log)!r}, "a") as log:
                log.write(" ".join(args) + "\\n")
            if args[0] == "venv":
                Path(args[-1], "bin").mkdir(parents=True)
            elif args == ["--version"]:
                print("uv 0.12.3")
            elif args[:2] == ["pip", "freeze"]:
                print("vllm==0.33.0")
            """
        )
    )
    uv.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    venv = tmp_path / "venvs" / "vllm"
    args = spike.parse_args(["install", "--out", "x.json", "--stack", "vllm", "--venv", str(venv)])
    result = spike.install_stage(args)

    python = str(venv / "bin" / "python")
    requirements = " ".join(spike.STACKS["vllm"].requirements)
    assert log.read_text().splitlines() == [
        # vLLM brings its own torch, so the venv doesn't see the image's.
        f"venv --python {sys.executable} {venv}",
        f"pip install --python {python} {requirements}",
        "--version",
        f"pip freeze --python {python}",
    ]
    assert result["uv"] == "uv 0.12.3"
    assert result["uv_bootstrap_seconds"] is None  # uv was already there
    assert result["freeze"] == ["vllm==0.33.0"]
    assert result["install_seconds"] >= 0


@pytest.mark.phase(3)
def test_prompts_follow_forges_format() -> None:
    # The kernel can't import Forge, so the script keeps a copy of the p1 format.
    assert spike.SYSTEM_PROMPT == forge_prompts.SYSTEM_PROMPT
    assert spike.USER_TEMPLATE == forge_prompts.USER_TEMPLATE
    prompts = spike.spike_prompts(100)
    assert len(prompts) == 100
    assert len({p[1]["content"] for p in prompts}) == 100  # no question repeats
    assert spike.spike_prompts(100) == prompts
    assert [m["role"] for m in prompts[0]] == ["system", "user"]
