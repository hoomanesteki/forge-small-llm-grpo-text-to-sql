"""The FG-301 throughput spike (kaggle/spike/throughput.py), checked off the GPU.

Nothing here needs a GPU, vLLM or the network: the stages that would are run with fakes,
so these check the wiring and the report, not a measurement.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
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
import yaml

from forge.data import prompts as forge_prompts
from forge.data.splits import read_split_ids
from forge.ops.kaggle import METADATA_FILE, bake_argv, check_kernel_metadata
from forge.ops.ledger import read_rows
from tests.spike import THROUGHPUT, load_script, throughput
from tests.tiny_hf import CHATML, QWEN3, byte_tokenizer

ROOT = Path(__file__).resolve().parents[2]
BENCH = ["grpo-bench", "--out", "o", "--work-dir", "w", "--model", "m"]
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
    assert (args.prompts, args.max_tokens) == (100, 256)
    # Story FG-301, question 5: 10 steps, 6 generations, 256-token completions, 4-bit.
    assert (args.steps, args.num_generations, args.max_completion_length) == (10, 6, 256)
    assert args.quant == "4bit"
    # Plan section 3.10's batch: 6 completions a device, 4 accumulation steps.
    assert (args.batch_size, args.grad_accum) == (6, 4)
    assert args.stacks == ["trl", "unsloth", "trl-vllm"]
    assert args.vllm is True
    assert args.bnb_models == ["Qwen/Qwen3-8B"]
    assert args.secret_labels == ["HF_TOKEN"]
    # Inside the story's 1.5 GPU hours, with room for the report to get out.
    assert args.budget_minutes < 90


@pytest.mark.phase(3)
@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["run", "--num-generations", "5"], "must be a multiple of --num-generations 5"),
        (["run", "--num-generations", "1", "--batch-size", "1"], "must be 2 or more"),
        (["run", "--prompts", "0"], "must be at least 1, got 0"),
        (["run", "--stacks", "pytorch"], "invalid choice"),
        (["vllm-bench", "--model", "m"], "--out"),
        (
            [*BENCH, "--stack", "trl", "--device", "cpu"],
            "--device cpu runs the trl stack without --quant 4bit",
        ),
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
    monkeypatch.delenv(spike.CHILD_ENV, raising=False)
    monkeypatch.setattr(sys, "argv", ["throughput.py"])
    argv = ["limit-probe", "--out-dir", str(tmp_path), "--interval-seconds", "0"]
    monkeypatch.setattr(spike, "KERNEL_ARGV", [*argv, "--max-beats", "1"])
    assert spike.main() == 0
    assert json.loads((tmp_path / spike.PROBE_NAME).read_text())["beats"] == 1


@pytest.mark.phase(3)
def test_launch_report_is_written_before_the_arguments_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An IPython kernel's argv, and baked arguments this script can't take: the report
    # still says how the kernel was started.
    launcher = str(tmp_path / "ipykernel_launcher.py")
    monkeypatch.delenv(spike.CHILD_ENV, raising=False)
    monkeypatch.setattr(sys, "argv", [launcher, "-f", "/tmp/kernel-4242.json"])
    baked = ["run", "--out-dir", str(tmp_path / "out"), "--no-such-option"]
    monkeypatch.setattr(spike, "KERNEL_ARGV", baked)
    with pytest.raises(SystemExit) as exit_info:
        spike.main()
    assert exit_info.value.code == 2
    assert "--no-such-option" in capsys.readouterr().err
    report = json.loads((tmp_path / "out" / spike.REPORT_NAME).read_text())
    assert report["launch"] == {
        "argv0": launcher,
        "argv": [launcher, "-f", "/tmp/kernel-4242.json"],
        "kernel_argv": baked,
        "file": str(THROUGHPUT),
        "cwd": os.getcwd(),
    }
    assert (report["args"], report["stages"], report["finished_at"]) == (None, [], None)


@pytest.mark.phase(3)
def test_a_child_takes_its_command_line_over_the_baked_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # run_child marks each child, which runs the same baked script with its own arguments.
    monkeypatch.setenv(spike.CHILD_ENV, "1")
    argv = ["limit-probe", "--out-dir", str(tmp_path), "--interval-seconds", "0"]
    monkeypatch.setattr(sys, "argv", ["throughput.py", *argv, "--max-beats", "2"])
    monkeypatch.setattr(spike, "KERNEL_ARGV", ["run", "--out-dir", str(tmp_path / "run")])
    assert spike.main() == 0
    assert json.loads((tmp_path / spike.PROBE_NAME).read_text())["beats"] == 2
    # A child writes no launch report: the kernel's is the one that counts.
    assert [p.name for p in tmp_path.iterdir()] == [spike.PROBE_NAME]


# Runs the baked script as ipykernel runs a cell: argv is the kernel's own, the code is
# compiled under a cell's file name that is no file on disk, its source is in linecache,
# and the namespace is a __main__ module with no __file__.
IPYKERNEL_DRIVER = """\
import json, linecache, sys, types
from pathlib import Path

script, launcher, cell, work = sys.argv[1:]
source = Path(script).read_text(encoding="utf-8")
sys.argv = [launcher, "-f", "/tmp/kernel-4242.json"]
linecache.cache[cell] = (len(source), None, source.splitlines(keepends=True), cell)
main = types.ModuleType("__main__")
sys.modules["__main__"] = main
exec(compile(source, cell, "exec"), main.__dict__)
ns = main.__dict__
path = ns["_script_path"](Path(work))
stage = ns["Stage"]("secrets", sys.executable, ("secrets", "--labels", "HF_TOKEN"))
outcome = ns["run_child"](stage, Path(work) / "secrets.json", 60.0, script=path)
print(json.dumps({"script": path, "status": outcome.status, "result": outcome.result}))
"""


@pytest.mark.phase(3)
def test_ipykernel_launch_runs_the_baked_argv(fakes: Path, tmp_path: Path) -> None:
    (fakes / "kaggle_secrets.py").write_text(
        "class UserSecretsClient:\n    def get_secret(self, label):\n        return 'x'\n"
    )
    out, work = tmp_path / "out", tmp_path / "work"
    argv = ["limit-probe", "--out-dir", str(out), "--interval-seconds", "0", "--max-beats", "1"]
    baked = tmp_path / "kernel" / "throughput.py"
    baked.parent.mkdir()
    baked.write_text(bake_argv(THROUGHPUT.read_text(encoding="utf-8"), argv), encoding="utf-8")
    # A real file named like a script, which must not be taken for this one.
    launcher = tmp_path / "ipykernel_launcher.py"
    launcher.write_text("from ipykernel import kernelapp as app\napp.launch_new_instance()\n")
    cell = tmp_path / "ipykernel_4242" / "2381940622.py"  # never written, as in a kernel
    driver = [sys.executable, "-c", IPYKERNEL_DRIVER, str(baked), str(launcher), str(cell)]
    done = subprocess.run(
        [*driver, str(work)], cwd=tmp_path, capture_output=True, text=True, timeout=120
    )
    # The cell ends without raising SystemExit, which a kernel would show as an error.
    assert done.returncode == 0, done.stderr

    # The baked arguments ran, not the kernel's -f <connection file>.
    assert json.loads((out / spike.PROBE_NAME).read_text())["beats"] == 1
    launch = json.loads((out / spike.REPORT_NAME).read_text())["launch"]
    assert (launch["argv0"], launch["file"], launch["kernel_argv"]) == (str(launcher), None, argv)
    # With no file to run, the children run the cell's source, written out once.
    found = json.loads(done.stdout.splitlines()[-1])
    assert found["script"] == str(work / "throughput.py")
    assert Path(found["script"]).read_text(encoding="utf-8") == baked.read_text(encoding="utf-8")
    # The child, marked as one, ran the stage it was given and not the baked probe.
    assert found["status"] == "ok"
    assert found["result"]["labels"] == {"HF_TOKEN": {"readable": True}}


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
    small, large, reference = "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B"
    # The GRPO benches come first, so a vLLM stage that hangs can't take their time, and
    # the reference is fetched only when vLLM's stack is next; it loads in 4-bit right
    # after vLLM's install, before the fp16 benches.
    assert names == [
        "env",
        "secrets",
        f"download:{small}",
        f"download:{large}",
        "install:trl",
        f"grpo:trl:{small}",
        f"grpo:trl:{large}",
        "install:unsloth",
        f"grpo:unsloth:{small}",
        f"grpo:unsloth:{large}",
        "install:trl-vllm",
        f"grpo:trl-vllm:{small}",
        f"grpo:trl-vllm:{large}",
        f"download:{reference}",
        "install:vllm",
        f"vllm:{reference}:bnb",
        f"vllm:{small}",
        f"vllm:{large}",
    ]
    # Those three may run in the minutes the budget keeps for them, and nothing else may.
    assert [stage.name for stage in stages if stage.reserved] == names[-5:-2]
    by_name = {stage.name: stage for stage in stages}
    venv = tmp_path / "work" / "venvs" / "trl"
    bench = by_name[f"grpo:trl:{large}"]
    # Benches run in their stack's venv, need its install and the model, and the last
    # one of each stack deletes the venv.
    assert bench.python == str(venv / "bin" / "python")
    assert bench.needs == ("install:trl", f"download:{large}")
    assert bench.removes == str(venv)
    assert by_name[f"grpo:trl:{small}"].removes is None
    # Each bench sees one GPU of a T4 x2 session, and the env stage both.
    for name in (f"grpo:trl:{large}", f"grpo:unsloth:{small}", f"vllm:{small}"):
        assert by_name[name].env == (("CUDA_VISIBLE_DEVICES", "0"),), name
    assert by_name["env"].env == by_name["install:trl"].env == ()
    assert by_name["install:trl"].python == "/usr/bin/python3"
    # Each bench's arguments parse as its child command, with the run's settings.
    child = spike.parse_args([*bench.argv, "--out", "x.json"])
    assert (child.command, child.stack, child.model, child.device) == (
        "grpo-bench",
        "trl",
        large,
        "cuda",
    )
    assert (child.steps, child.num_generations, child.max_completion_length) == (10, 6, 256)
    vllm = spike.parse_args([*by_name[f"vllm:{small}"].argv, "--out", "x.json"])
    assert (vllm.command, vllm.prompts, vllm.max_tokens) == ("vllm-bench", 100, 256)
    assert vllm.quantization is None
    # The reference loads in 4-bit with bitsandbytes in vLLM's venv, on a few prompts, under
    # a cap of its own; the last fp16 bench deletes the venv.
    bnb = by_name[f"vllm:{reference}:bnb"]
    assert bnb.needs == ("install:vllm", f"download:{reference}")
    assert (bnb.kind, by_name[f"vllm:{small}"].kind) == ("bnb", "vllm")
    assert bnb.removes is None
    assert by_name[f"vllm:{large}"].removes == str(tmp_path / "work" / "venvs" / "vllm")
    assert bnb.env == spike.ONE_GPU
    child = spike.parse_args([*bnb.argv, "--out", "x.json"])
    assert (child.model, child.quantization) == (reference, "bitsandbytes")
    assert child.prompts == spike.BNB_PROMPTS < args.prompts

    only = spike.plan_stages(_run_args(tmp_path, "--no-vllm", "--stacks", "trl"), python="py")
    assert [s.name for s in only][-3:] == ["install:trl", f"grpo:trl:{small}", f"grpo:trl:{large}"]
    assert not any(s.name.startswith("vllm:") or reference in s.name for s in only)
    no_bnb = spike.plan_stages(_run_args(tmp_path, "--bnb-models"), python="py")
    assert [s.name for s in no_bnb][-1] == f"vllm:{large}"
    assert f"download:{reference}" not in [s.name for s in no_bnb]
    assert not any(stage.reserved for stage in no_bnb)
    # A reference among the models is fetched with them, so only its install and load
    # are kept for.
    early = spike.plan_stages(_run_args(tmp_path, "--models", small, reference), python="py")
    assert [s.name for s in early if s.reserved] == ["install:vllm", f"vllm:{reference}:bnb"]


@pytest.mark.phase(3)
def test_bnb_stage_loads_the_reference_model() -> None:
    # The kernel can't read configs/models.yaml, so the script names FG-307's 4-bit
    # reference itself, and vLLM's stack carries the bitsandbytes it needs.
    models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    four_bit = [
        entry["repo"]
        for entry in models["reference"].values()
        if entry.get("quantization") == "bitsandbytes"
    ]
    assert spike.parse_args(["run"]).bnb_models == four_bit == [spike.REFERENCE_MODEL]
    bnb_pin = [r for r in spike.STACKS["trl-vllm"].requirements if r.startswith("bitsandbytes==")]
    assert bnb_pin and set(bnb_pin) <= set(spike.STACKS["vllm"].requirements)


def _fake_runner(outcomes: dict[str, Any], calls: list[str]) -> Any:
    def runner(stage: Any, result_file: Path, timeout: float) -> Any:
        assert timeout > 0
        assert result_file.suffix == ".json"
        calls.append(stage.name)
        return outcomes.get(stage.name, spike.Outcome("ok", result={"stage": stage.name}))

    return runner


@pytest.mark.phase(3)
def test_run_report_records_every_stage(tmp_path: Path) -> None:
    args = _run_args(tmp_path, "--models", "m/small", "--stacks", "trl", "unsloth")
    venvs = tmp_path / "work" / "venvs"
    (venvs / "unsloth").mkdir(parents=True)
    calls: list[str] = []
    outcomes = {
        "install:trl": spike.Outcome("error", error="RuntimeError: uv pip install exited 1"),
        "vllm:m/small": spike.Outcome("timeout", error="killed after 60 s"),
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
        "stage_caps_minutes",
        "bnb_reserve_minutes",
    }
    assert (report["schema"], report["story"]) == (spike.REPORT_SCHEMA, "FG-301")
    assert report["finished_at"] is not None
    assert report["launch"]["script"] == str(THROUGHPUT)  # what each child runs
    assert report["args"]["models"] == ["m/small"]
    statuses = {s["name"]: (s["status"], s["error"]) for s in report["stages"]}
    assert statuses == {
        "env": ("ok", None),
        "secrets": ("ok", None),
        "download:m/small": ("ok", None),
        "install:trl": ("error", "RuntimeError: uv pip install exited 1"),
        # Never started: its install failed.
        "grpo:trl:m/small": ("skipped", "needs install:trl"),
        "install:unsloth": ("ok", None),
        "grpo:unsloth:m/small": ("ok", None),
        "download:Qwen/Qwen3-8B": ("ok", None),
        "install:vllm": ("ok", None),
        "vllm:Qwen/Qwen3-8B:bnb": ("ok", None),
        "vllm:m/small": ("timeout", "killed after 60 s"),
    }
    assert "grpo:trl:m/small" not in calls
    fields = {"name", "status", "started_at", "seconds", "timeout_seconds", "env", "result"}
    for stage in report["stages"]:
        assert set(stage) >= {*fields, "error"}
    assert report["stages"][0]["result"] == {"stage": "env"}
    envs = {s["name"]: s["env"] for s in report["stages"]}
    assert envs["env"] == {}
    assert envs["grpo:unsloth:m/small"] == {"CUDA_VISIBLE_DEVICES": "0"}
    # Each stack's venv is gone once its last bench is over.
    assert not (venvs / "unsloth").exists()


@pytest.mark.phase(3)
def test_run_without_a_script_to_start_stops_with_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def nowhere(work: Path) -> str:
        raise RuntimeError("can't find this script's file or its source")

    monkeypatch.setattr(spike, "_script_path", nowhere)
    calls: list[str] = []
    report = spike.run(_run_args(tmp_path), runner=_fake_runner({}, calls))
    assert calls == []
    assert report["error"] == "can't find this script's file or its source"
    assert report["stages"] == [] and report["finished_at"] is not None
    assert json.loads((tmp_path / "out" / spike.REPORT_NAME).read_text()) == report


@pytest.mark.phase(3)
def test_run_skips_stages_once_out_of_budget(tmp_path: Path) -> None:
    # No 4-bit reference, so no minutes are kept for one.
    args = _run_args(tmp_path, "--models", "m/small", "--budget-minutes", "1", "--bnb-models")
    ticks: Iterator[int] = iter(range(0, 10_000, 25))  # every reading is 25 s later
    calls: list[str] = []
    report = spike.run(args, runner=_fake_runner({}, calls), clock=lambda: float(next(ticks)))
    statuses = [(s["name"], s["status"]) for s in report["stages"]]
    assert statuses[:1] == [("env", "ok")]
    assert all(status == "skipped" for _, status in statuses[1:])
    assert calls == ["env"]
    errors = {s["name"]: s["error"] for s in report["stages"]}
    assert errors["secrets"] == errors["install:trl"] == "out of budget (1 min)"
    assert errors["grpo:trl:m/small"] == "needs install:trl, download:m/small"


@pytest.mark.phase(3)
def test_a_hung_stage_costs_only_its_cap(tmp_path: Path) -> None:
    args = _run_args(tmp_path, "--models", "m/small")
    now = [0.0]
    timeouts: dict[str, float] = {}

    def runner(stage: Any, result_file: Path, timeout: float) -> Any:
        timeouts[stage.name] = timeout
        if stage.name == "grpo:trl:m/small":
            now[0] += timeout  # hangs until it is killed
            return spike.Outcome("timeout", error=f"killed after {timeout:.0f} s")
        now[0] += 1.0
        return spike.Outcome("ok", result={})

    report = spike.run(args, runner=runner, clock=lambda: now[0])
    caps = spike.STAGE_CAPS_MINUTES
    assert report["stage_caps_minutes"] == caps
    kinds = {stage.name: stage.kind for stage in spike.plan_stages(args)}
    assert set(caps) == set(kinds.values())
    # Each stage may run for its kind's cap, all well inside the budget.
    assert max(caps.values()) < args.budget_minutes
    for stage in report["stages"]:
        cap = caps[kinds[stage["name"]]] * 60
        assert timeouts[stage["name"]] == stage["timeout_seconds"] == cap, stage["name"]
    # The hang cost its cap, not the time left, and every stage after it still ran.
    statuses = {s["name"]: s["status"] for s in report["stages"]}
    assert statuses.pop("grpo:trl:m/small") == "timeout"
    assert set(statuses.values()) == {"ok"}


@pytest.mark.phase(3)
def test_the_grpo_benches_leave_the_4bit_reference_its_minutes(tmp_path: Path) -> None:
    args = _run_args(tmp_path, "--models", "m/small", "m/large")
    now = [0.0]
    timeouts: dict[str, float] = {}

    def runner(stage: Any, result_file: Path, timeout: float) -> Any:
        timeouts[stage.name] = timeout
        if stage.kind == "grpo":
            now[0] += timeout  # every GRPO bench runs until it is killed
            return spike.Outcome("timeout", error=f"killed after {timeout:.0f} s")
        now[0] += 60.0
        return spike.Outcome("ok", result={})

    report = spike.run(args, runner=runner, clock=lambda: now[0])

    assert report["bnb_reserve_minutes"] == spike.BNB_RESERVE_MINUTES == 20
    kept = "out of budget (80 min, the last 20 kept for the 4-bit reference)"
    statuses = {s["name"]: (s["status"], s["error"]) for s in report["stages"]}
    assert statuses == {
        "env": ("ok", None),
        "secrets": ("ok", None),
        "download:m/small": ("ok", None),
        "download:m/large": ("ok", None),
        "install:trl": ("ok", None),
        "grpo:trl:m/small": ("timeout", "killed after 1500 s"),
        "grpo:trl:m/large": ("timeout", "killed after 1500 s"),
        "install:unsloth": ("ok", None),
        # Cut where the kept minutes start, 60 minutes in.
        "grpo:unsloth:m/small": ("timeout", "killed after 240 s"),
        "grpo:unsloth:m/large": ("skipped", kept),
        "install:trl-vllm": ("skipped", kept),
        "grpo:trl-vllm:m/small": ("skipped", "needs install:trl-vllm"),
        "grpo:trl-vllm:m/large": ("skipped", "needs install:trl-vllm"),
        # The answer FG-306 and FG-307 wait on still comes, under its own cap, and the
        # fp16 benches take what is left.
        "download:Qwen/Qwen3-8B": ("ok", None),
        "install:vllm": ("ok", None),
        "vllm:Qwen/Qwen3-8B:bnb": ("ok", None),
        "vllm:m/small": ("ok", None),
        "vllm:m/large": ("ok", None),
    }
    assert timeouts["vllm:Qwen/Qwen3-8B:bnb"] == spike.STAGE_CAPS_MINUTES["bnb"] * 60
    assert timeouts["download:Qwen/Qwen3-8B"] == spike.STAGE_CAPS_MINUTES["download"] * 60


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
    outcome = spike.run_child(
        stage, tmp_path / "results" / "secrets.json", timeout, script=str(THROUGHPUT)
    )
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
    outcome = spike.run_child(stage, tmp_path / "secrets.json", 60.0, script=str(THROUGHPUT))
    assert outcome.status == "ok", outcome.error
    assert outcome.result["labels"] == {"HF_TOKEN": {"readable": True}}


@pytest.mark.phase(3)
def test_run_child_gives_the_stage_its_environment(
    fakes: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The stand-in for kaggle_secrets records the environment the child sees.
    seen = tmp_path / "seen.json"
    names = ("CUDA_VISIBLE_DEVICES", "TOKENIZERS_PARALLELISM", spike.CHILD_ENV)
    (fakes / "kaggle_secrets.py").write_text(
        textwrap.dedent(
            f"""\
            import json, os
            with open({str(seen)!r}, "w") as out:
                json.dump({{name: os.environ.get(name) for name in {names!r}}}, out)
            class UserSecretsClient:
                def get_secret(self, label):
                    return "x"
            """
        )
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")  # a T4 x2 session's two GPUs
    stage = spike.Stage("secrets", sys.executable, ("secrets",), env=spike.ONE_GPU)
    outcome = spike.run_child(stage, tmp_path / "secrets.json", 60.0, script=str(THROUGHPUT))
    assert outcome.status == "ok", outcome.error
    assert json.loads(seen.read_text()) == {
        "CUDA_VISIBLE_DEVICES": "0",
        "TOKENIZERS_PARALLELISM": "false",
        spike.CHILD_ENV: "1",
    }


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
    assert llm_kwargs["quantization"] is None
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
def test_vllm_stage_loads_bitsandbytes_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    # FG-306 loads a 4-bit reference only if the spike shows vLLM can, as the backend
    # asks for it: bitsandbytes over fp16 weights.
    fake = types.ModuleType("vllm")
    exec(FAKE_VLLM, fake.__dict__)
    monkeypatch.setitem(sys.modules, "vllm", fake)
    argv = ["vllm-bench", "--out", "x.json", "--model", "Qwen/Qwen3-8B", "--prompts", "10"]
    result = spike.vllm_stage(spike.parse_args([*argv, "--quantization", "bitsandbytes"]))

    (_, llm_kwargs), (_, prompts, _, _) = fake.CALLS
    assert (llm_kwargs["quantization"], llm_kwargs["dtype"]) == ("bitsandbytes", "half")
    assert prompts == result["prompts"] == 10
    assert result["settings"]["quantization"] == "bitsandbytes"
    assert len(result["samples"]) == 3


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
def test_secret_probe_never_writes_a_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Shaped like a Hugging Face token, so the probe's scrub has to catch it in error text.
    canary = "hf_" + os.urandom(12).hex()

    class UserSecretsClient:
        def get_secret(self, label: str) -> str:
            if label == "HF_TOKEN":
                return canary
            raise ConnectionError(f"service said no, {canary}") from OSError(canary)

    fake = types.ModuleType("kaggle_secrets")
    fake.UserSecretsClient = UserSecretsClient  # type: ignore[attr-defined]
    fake.__file__ = "kaggle_secrets.py"
    monkeypatch.setitem(sys.modules, "kaggle_secrets", fake)
    monkeypatch.setenv("KAGGLE_USER_SECRETS_TOKEN", canary)
    # Loading it runs nothing: off Kaggle its output folder doesn't exist.
    secret_probe = load_script(ROOT / "kaggle" / "spike" / "secret_probe.py", "secret_probe")
    out = tmp_path / "secret-probe.json"
    secret_probe.main(out)

    written = out.read_text()
    assert canary not in written and canary not in capsys.readouterr().out
    found = json.loads(written)
    assert found["has_user_secrets_token"] is True
    assert "KAGGLE_USER_SECRETS_TOKEN" in found["kaggle_env_names"]
    assert found["HF_TOKEN"] == {"readable": True, "length": len(canary)}
    refused = found["FORGE_NO_SUCH_LABEL"]
    assert (refused["error"], refused["module"]) == ("ConnectionError", "builtins")
    assert refused["message"] == "service said no, hf_<redacted>"
    assert "hf_<redacted>" in refused["cause"]


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


# The venv's python, which the stand-in uv writes: it logs its arguments too, and its pip
# puts a torch into the venv's own site-packages when FAKE_PIP_TORCH is set, as uv's own
# install does too, whatever system site packages hold.
FAKE_VENV_PYTHON = """\
#!{python}
import os, sys
from pathlib import Path
args = sys.argv[1:]
with open({log!r}, "a") as log:
    log.write("python " + " ".join(args) + "\\n")
if args[:3] == ["-m", "pip", "install"] and os.environ.get("FAKE_PIP_TORCH"):
    site = next(Path(sys.argv[0]).parents[1].glob("lib/python*/site-packages"))
    info = site / "torch-2.14.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\\nName: torch\\nVersion: 2.14.0\\n")
"""


@pytest.fixture
def fake_uv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stand-in uv on PATH that logs its arguments, so commands are checked without
    installing; returns the log, which the venv's python writes to as well."""
    bin_dir, log = tmp_path / "bin", tmp_path / "uv.log"
    bin_dir.mkdir()
    python = FAKE_VENV_PYTHON.format(python=sys.executable, log=str(log))
    uv = bin_dir / "uv"
    uv.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import os, sys
            from pathlib import Path
            args = sys.argv[1:]
            with open({str(log)!r}, "a") as log:
                log.write(" ".join(args) + "\\n")
            if args[0] == "venv":
                venv = Path(args[-1])
                (venv / "lib" / "python3.13" / "site-packages").mkdir(parents=True)
                (venv / "bin").mkdir()
                (venv / "bin" / "python").write_text({python!r})
                (venv / "bin" / "python").chmod(0o755)
            elif args == ["--version"]:
                print("uv 0.12.3")
            elif args[:2] == ["pip", "freeze"]:
                print("trl==1.18.0")
            elif args[:2] == ["pip", "install"] and os.environ.get("FAKE_PIP_TORCH"):
                venv = Path(args[args.index("--python") + 1]).parents[1]
                site = next(venv.glob("lib/python*/site-packages"))
                info = site / "torch-2.9.0.dist-info"
                info.mkdir()
                info.joinpath("METADATA").write_text(
                    "Metadata-Version: 2.1\\nName: torch\\nVersion: 2.9.0\\n"
                )
            """
        )
    )
    uv.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("FAKE_PIP_TORCH", raising=False)
    return log


def _install(tmp_path: Path, stack: str) -> tuple[Path, dict[str, Any]]:
    venv = tmp_path / "venvs" / stack
    args = spike.parse_args(["install", "--out", "x.json", "--stack", stack, "--venv", str(venv)])
    return venv, spike.install_stage(args)


@pytest.mark.phase(3)
def test_install_stage_commands(tmp_path: Path, fake_uv: Path) -> None:
    venv, result = _install(tmp_path, "trl")

    python = str(venv / "bin" / "python")
    requirements = " ".join(spike.STACKS["trl"].requirements)
    # uv ignores system site packages when it installs, so the stack that runs on the
    # image's torch installs with the venv's own pip, which counts them.
    assert fake_uv.read_text().splitlines() == [
        f"venv --python {sys.executable} --seed --system-site-packages {venv}",
        f"python -m pip install --no-input --progress-bar off {requirements}",
        "--version",
        f"pip freeze --python {python}",
    ]
    assert result["installer"] == "pip"
    assert result["system_site_packages"] is True
    assert result["uv"] == "uv 0.12.3"
    assert result["uv_bootstrap_seconds"] is None  # uv was already there
    assert result["freeze"] == ["trl==1.18.0"]
    assert result["install_seconds"] >= 0


@pytest.mark.phase(3)
def test_install_stage_uses_uv_for_a_stack_with_its_own_torch(
    tmp_path: Path, fake_uv: Path
) -> None:
    venv, result = _install(tmp_path, "trl-vllm")

    python = str(venv / "bin" / "python")
    requirements = " ".join(spike.STACKS["trl-vllm"].requirements)
    assert fake_uv.read_text().splitlines()[:2] == [
        f"venv --python {sys.executable} {venv}",
        f"pip install --python {python} {requirements}",
    ]
    assert (result["installer"], result["system_site_packages"]) == ("uv", False)


@pytest.mark.phase(3)
@pytest.mark.parametrize("own_torch", ["unsloth", "trl-vllm"])
def test_install_stage_refuses_a_torch_inside_a_venv_meant_to_reuse_the_images(
    tmp_path: Path, fake_uv: Path, monkeypatch: pytest.MonkeyPatch, own_torch: str
) -> None:
    monkeypatch.setenv("FAKE_PIP_TORCH", "1")
    with pytest.raises(RuntimeError, match=r"torch 2\.14\.0 .*image's torch"):
        _install(tmp_path, "trl")
    # A stack that brings its own torch keeps the one it installs.
    venv, result = _install(tmp_path, own_torch)
    assert spike._venv_packages(venv)["torch"] == "2.9.0"
    assert result["installer"] == "uv"


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


@pytest.mark.phase(3)
def test_grpo_prompts_are_rendered_with_thinking_off() -> None:
    # Every stack trains on the same text: TRL 0.22, Unsloth's, has no
    # chat_template_kwargs, so it can't be left to the trainer.
    chats = spike.spike_prompts(3)
    prompts = spike.render_prompts(byte_tokenizer(QWEN3), chats)
    assert len(prompts) == 3
    for prompt, chat in zip(prompts, chats, strict=True):
        assert prompt.startswith("<|im_start|>system\n" + spike.SYSTEM_PROMPT)
        assert chat[1]["content"] in prompt
        assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    # A template that ignores the switch fails the stage instead of benching with
    # thinking on.
    with pytest.raises(RuntimeError, match="ignores enable_thinking"):
        spike.render_prompts(byte_tokenizer(CHATML), chats)


def _requirement_lines(path: Path) -> list[str]:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def _pins(requirements: tuple[str, ...]) -> dict[str, str]:
    return dict(requirement.split("==") for requirement in requirements)


@pytest.mark.phase(3)
def test_requirements_file_is_the_trl_vllm_stack() -> None:
    # FG-302 ships kaggle/requirements-gpu.txt and FG-306 adds vLLM to it; the spike
    # installs and times the same pins.
    gpu = spike.STACKS["trl-vllm"].requirements
    path = ROOT / "kaggle" / "requirements-gpu.txt"
    assert tuple(_requirement_lines(path)) == gpu
    # FG-301 ran these pins on a T4, and the file says which stages measured them.
    text = path.read_text(encoding="utf-8")
    assert "unverified" not in text
    for stage in ("install:trl-vllm", "vllm:<model>", "grpo:trl-vllm:<model>"):
        assert stage in text, stage
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")

    def locked(name: str, version: str) -> bool:
        return f'\nname = "{name}"\nversion = "{version}"\n' in lock

    # vllm 0.33.0 pins torch 2.14.0 and needs transformers>=5.16.1,<5.21, below the lock's.
    pins = _pins(gpu)
    assert (pins["vllm"], pins["torch"], pins["transformers"]) == ("0.33.0", "2.14.0", "5.20.2")
    # Every other pin is the version uv.lock gives CI, where the lock has the package.
    for name, version in pins.items():
        if name not in ("torch", "transformers") and f'\nname = "{name}"\n' in lock:
            assert locked(name, version), (name, version)
    # The trl stack, the training stack without vLLM, keeps the lock's versions throughout.
    for name, version in _pins(spike.STACKS["trl"].requirements).items():
        if f'\nname = "{name}"\n' in lock:
            assert locked(name, version), (name, version)


@pytest.mark.phase(3)
def test_sql_block_reward() -> None:
    completions = [
        [{"role": "assistant", "content": "```sql\nSELECT 1\n```"}],
        [{"role": "assistant", "content": "SELECT 1"}],
        [{"role": "assistant", "content": "```sql\nSELECT 1"}],
    ]
    assert spike.sql_block_reward(prompts=[None] * 3, completions=completions) == [1.0, 0.0, 0.0]


@pytest.mark.phase(3)
def test_bundle_writes_both_kernels(tmp_path: Path) -> None:
    from kaggle_spike import KERNELS, bundle

    commands = bundle("forge-owner", tmp_path)
    gpu = json.loads((tmp_path / "gpu" / METADATA_FILE).read_text())
    limit = json.loads((tmp_path / "limit" / METADATA_FILE).read_text())
    for meta in (gpu, limit):
        assert check_kernel_metadata(meta) == []
        assert meta["is_private"] is True and meta["enable_internet"] is True
        assert meta["code_file"] == "throughput.py"
    assert (gpu["id"], gpu["enable_gpu"], gpu["machine_shape"]) == (
        "forge-owner/forge-throughput-spike",
        True,
        "NvidiaTeslaT4",
    )
    # Waiting for the session limit spends no GPU quota.
    assert limit["enable_gpu"] is False and "machine_shape" not in limit

    # Each kernel's script carries the arguments it will run with.
    for name, kernel in KERNELS.items():
        spec = importlib.util.spec_from_file_location(
            f"baked_{name}", tmp_path / name / "throughput.py"
        )
        assert spec is not None and spec.loader is not None
        baked = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baked
        spec.loader.exec_module(baked)
        argv = baked.KERNEL_ARGV
        assert argv == list(kernel.argv)
        assert baked.parse_args(argv).command == kernel.argv[0]

    # The push timeout keeps to the story's 1.5 GPU hours and outlasts the run's budget.
    budget = spike.parse_args(["run"]).budget_minutes * 60
    assert budget < KERNELS["gpu"].timeout_s <= 90 * 60
    assert commands[:3] == [
        f"kaggle kernels push -p {tmp_path / 'gpu'} --accelerator NvidiaTeslaT4 -t 5400",
        "kaggle kernels status forge-owner/forge-throughput-spike",
        f"kaggle kernels output forge-owner/forge-throughput-spike -p {tmp_path / 'gpu'}-output -o",
    ]
    assert commands[3] == f"kaggle kernels push -p {tmp_path / 'limit'} -t 900"


@pytest.mark.phase(3)
def test_spike_doc_keeps_research_apart_from_measurements() -> None:
    doc = (ROOT / "docs" / "spikes" / "kaggle-gpu.md").read_text(encoding="utf-8")
    questions = [part for part in doc.split("\n## ") if part[:1].isdigit()]
    assert [part.split(".", 1)[0] for part in questions] == ["1", "2", "3", "4", "5", "6"]
    for part in questions:
        research, found, measured = part.partition("**Measured on 2026-11-16.**")
        title = part.splitlines()[0]
        assert found, f"{title}: no measurements section"
        assert "**What research says today.**" in research, title
        assert "](https://" in research, f"{title}: research without a source"
        # Each measurement names the report field it comes from.
        assert re.search(r"`[a-z_-]+[.:]", measured), f"{title}: no report field named"
    assert "Not measured yet" not in doc and "To be measured" not in doc

    # The reports it quotes are committed as the kernels wrote them, each in the README.
    reports = ROOT / "docs" / "spikes" / "kaggle-gpu"
    readme = (reports / "README.md").read_text(encoding="utf-8")
    listed = re.findall(r"^\| `([a-z-]+\.json)` \|", readme, flags=re.MULTILINE)
    assert sorted(listed) == sorted(path.name for path in reports.glob("*.json"))
    for name in ("spike-report.json", "limit-probe.json", "secret-probe.json"):
        assert f"`{name}`" in doc, name
    report = json.loads((reports / "spike-report.json").read_text(encoding="utf-8"))
    assert (report["schema"], report["story"]) == (spike.REPORT_SCHEMA, "FG-301")
    assert report["finished_at"] is not None
    run = json.loads((reports / "kernels.json").read_text(encoding="utf-8"))
    assert run["job"] == "spike" and run["kernels"][0]["kernel"].endswith("/forge-throughput-spike")
    # Both records the spike informs are decided, each with what it measured.
    for name in ("0005-kaggle-compute", "0006-train-backend"):
        assert f"(../adr/{name}.md)" in doc
        adr = (ROOT / "docs" / "adr" / f"{name}.md").read_text(encoding="utf-8")
        assert adr.startswith(f"# ADR-{name[:4]}:")
        assert "\n- Status: accepted\n" in adr, name
        assert "\n## Measured\n" in adr, name


@pytest.mark.phase(3)
def test_pending_runs_say_what_blocks_them() -> None:
    pending = yaml.safe_load((ROOT / "docs" / "runs" / "pending.yaml").read_text())["pending"]
    for entry in pending:
        assert entry["status"] == "blocked", entry["run"]
        assert entry["blocked_on"], entry["run"]
        assert (ROOT / entry["how"].split(",")[0]).is_file(), entry["how"]
    # FG-301's runs stay inside the story's 1.5 GPU hours (plan section 3.8): any still
    # pending by its estimate, the spike that ran by its ledger row.
    spike_run = json.loads((ROOT / "docs" / "spikes" / "kaggle-gpu" / "kernels.json").read_text())
    spent = sum(
        row.hours
        for row in read_rows(ROOT / "docs" / "compute-ledger.md")
        if row.run_id == spike_run["run_id"]
    )
    assert spent > 0
    assert spent + sum(e["est_gpu_hours"] for e in pending if e["story"] == "FG-301") <= 1.5


def _budget_rows() -> dict[str, float]:
    """Plan section 3.8's GPU hour budget, as {the job's name before any colon or
    parenthesis: its hours}, the total row as ``Total``."""
    plan = (ROOT / "docs" / "PLAN.md").read_text(encoding="utf-8")
    section = plan.split("\n### 3.8 ", 1)[1].split("\n### ", 1)[0]
    table = section.split("**GPU hour budget**", 1)[1].split("\n\n")[1]
    rows = {}
    for line in table.splitlines()[2:]:
        job, _, hours, _ = (cell.strip() for cell in line.strip("|").split("|"))
        name = re.split(r"[:(]", job.strip("*"), maxsplit=1)[0].strip()
        number = re.search(r"[\d.]+", hours)
        assert number, line
        rows[name] = float(number.group())
    return rows


@pytest.mark.phase(3)
def test_budget_rows_follow_the_measurements() -> None:
    # Each worked-out row of plan section 3.8, from the files its note names and by its
    # formulas, so a row can't drift from them or be typed by hand.
    report = json.loads((ROOT / "docs" / "spikes" / "kaggle-gpu" / "spike-report.json").read_text())
    stages = {stage["name"]: stage["result"] for stage in report["stages"]}
    job = ROOT / "docs" / "runs" / "2026-11-16-smoke-kaggle-50-val-c8f4b41" / "status.json"
    kernel = json.loads(job.read_text())["stages"]
    fixed = sum(kernel[name]["seconds"] for name in ("code", "install", "data", "secrets"))
    models = {"1.7B": "Qwen/Qwen3-1.7B", "4B": "Qwen/Qwen3-4B"}

    def up(hours: float) -> float:
        return math.ceil(round(hours * 20, 6)) / 20

    def grpo(steps: int, size: str, sessions: int = 1) -> float:
        step = stages[f"grpo:trl:{models[size]}"]["median_step_seconds"]
        return up(steps * step * 1.5 / 3600 + 0.25 * sessions)

    def evaluate(questions: int, size: str) -> float:
        bench = stages[f"vllm:{models[size]}"]
        setup = fixed + stages[f"download:{models[size]}"]["seconds"] + bench["load_seconds"]
        generate = questions * 256 / bench["completion_tokens_per_second"]
        return up((setup + generate) * 1.5 / 3600)

    # Spider's dev split is 1,034 questions (ADR-0004); only its size is used here.
    val, dev = len(read_split_ids("val")), 1034
    # The reference's own config estimates: no run has loaded an 8B model on the T4.
    reference = 0.45 + 0.50
    candidates = 2 * evaluate(val, "1.7B") + evaluate(val, "4B")
    expected = {
        "Baselines": candidates
        + evaluate(val, "4B")
        + evaluate(100, "1.7B")
        + evaluate(dev, "4B")
        + reference,
        "SFT checkpoint evals": 2 * evaluate(val, "4B"),
        "GRPO pilot": grpo(100, "1.7B", sessions=2),
        "GRPO main": grpo(600, "4B", sessions=2),
        "Checkpoint selection": 3 * evaluate(val, "4B") + 2 * evaluate(dev, "4B"),
        "Hacking fix re-run": grpo(200, "4B") + evaluate(val, "4B"),
    }
    rows = _budget_rows()
    for name, hours in expected.items():
        assert rows[name] == pytest.approx(hours), name

    # The spike's row is its ledger row, and the total adds the rows up within the 45 h
    # the Phase 3 gate allows.
    spike_run = json.loads((ROOT / "docs" / "spikes" / "kaggle-gpu" / "kernels.json").read_text())
    ledger = read_rows(ROOT / "docs" / "compute-ledger.md")
    assert [row.hours for row in ledger if row.run_id == spike_run["run_id"]] == [
        rows["Throughput spike"]
    ]
    total = rows.pop("Total")
    assert total == pytest.approx(sum(rows.values())) and total <= 45
