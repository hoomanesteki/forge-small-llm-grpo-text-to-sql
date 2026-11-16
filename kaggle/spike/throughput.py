"""Throughput spike for FG-301: what one Kaggle T4 session can do, measured.

Pushed as a private script kernel, it writes one JSON report to /kaggle/working, which
`kaggle kernels output` brings back. docs/spikes/kaggle-gpu.md says how to push it and
which question each stage answers:

- env: Python, CUDA, torch, the GPUs, the image build, internet and disk.
- secrets: whether Kaggle Secrets can be read in a pushed run. The report says only
  whether a read worked, never what it returned.
- download:<model>: how long a model takes to fetch, and the commit fetched.
- install:<stack>: how long a candidate GPU stack takes to install into a venv of its own.
- vllm:<model>: fp16 generation throughput with vLLM on 100 prompts.
- grpo:<stack>:<model>: seconds per GRPO step and peak GPU memory over 10 steps.
- limit-probe, pushed as its own CPU kernel: a heartbeat file rewritten until the session
  is killed, so the output of a timed-out session shows what survives the kill.

Each stage runs in a child process, so a crash or an out-of-memory kill in one leaves the
others and the report intact; the report is rewritten after every stage. The stacks pin
versions that conflict with each other, so each gets its own venv, deleted after use.

The script needs only the standard library to start: heavy imports happen inside the
stages, and nothing runs on import. Kaggle runs a script kernel without arguments, so
scripts/kaggle_spike.py writes them into KERNEL_ARGV when it bundles the kernel.
"""

from __future__ import annotations

import argparse
import functools
import importlib.metadata
import itertools
import json
import os
import platform
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

KERNEL_ARGV: list[str] = []

REPORT_SCHEMA = "forge-kaggle-spike/1"
REPORT_NAME = "spike-report.json"
PROBE_NAME = "limit-probe.json"
DEFAULT_MODELS = ("Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B")
MODEL_FILES = ["*.json", "*.safetensors", "*.txt", "*.jinja"]
PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "trl",
    "datasets",
    "huggingface-hub",
    "bitsandbytes",
    "vllm",
    "unsloth",
    "kaggle",
)
INTERNET_PROBE_URL = "https://huggingface.co"
UV_PIN = "uv==0.12.3"
# Plan section 3.10: prompts are fitted to 1,536 tokens, LoRA covers every linear layer.
MAX_PROMPT_TOKENS = 1536
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# The prompt format of forge.data.prompts (p1), copied because the kernel runs without
# Forge installed. A unit test keeps the copy in step.
SYSTEM_PROMPT = (
    "You write SQLite queries. Reply with exactly one SQL query in a ```sql code block. "
    "No explanation."
)
USER_TEMPLATE = "Database schema:\n{schema}\n\nQuestion: {question}"
# A made-up schema of about a Spider database's size, so the prompts carry no dataset text.
SCHEMA = "\n".join(
    [
        "CREATE TABLE author (author_id INTEGER PRIMARY KEY, name TEXT, country TEXT, "
        "birth_year INTEGER)",
        "CREATE TABLE book (book_id INTEGER PRIMARY KEY, title TEXT, author_id INTEGER, "
        "genre TEXT, pages INTEGER, published_year INTEGER, "
        "FOREIGN KEY (author_id) REFERENCES author(author_id))",
        "CREATE TABLE library (library_id INTEGER PRIMARY KEY, name TEXT, city TEXT, "
        "opened_year INTEGER)",
        "CREATE TABLE copy (copy_id INTEGER PRIMARY KEY, book_id INTEGER, library_id INTEGER, "
        "condition TEXT, FOREIGN KEY (book_id) REFERENCES book(book_id), "
        "FOREIGN KEY (library_id) REFERENCES library(library_id))",
        "CREATE TABLE member (member_id INTEGER PRIMARY KEY, name TEXT, library_id INTEGER, "
        "joined_year INTEGER, FOREIGN KEY (library_id) REFERENCES library(library_id))",
        "CREATE TABLE loan (loan_id INTEGER PRIMARY KEY, copy_id INTEGER, member_id INTEGER, "
        "loan_date TEXT, return_date TEXT, FOREIGN KEY (copy_id) REFERENCES copy(copy_id), "
        "FOREIGN KEY (member_id) REFERENCES member(member_id))",
    ]
)
QUESTIONS = (
    "How many books are in the {genre} genre?",
    "List the titles of {genre} books with more than {pages} pages.",
    "Which library in {city} holds the most copies of {genre} books?",
    "What is the average page count of books published after {year}?",
    "Name the authors from {country} who wrote at least {count} books.",
    "How many loans did members of libraries in {city} take out in {year}?",
    "Which members joined after {year} and have never borrowed a {genre} book?",
    "For each genre, how many copies are in poor condition in {city}?",
    "What is the title of the longest book by an author born before {year}?",
    "List the libraries opened before {year} that have more than {count} members.",
)
FILLS = (
    ("fantasy", 300, "Lyon", 1990, "Chile", 3),
    ("history", 450, "Osaka", 2005, "Kenya", 2),
    ("poetry", 120, "Quito", 1975, "Norway", 5),
    ("science", 600, "Perth", 2012, "India", 4),
    ("travel", 250, "Tunis", 1999, "Peru", 6),
    ("drama", 180, "Leeds", 1984, "Egypt", 2),
    ("crime", 380, "Turin", 2018, "Ghana", 3),
    ("biography", 520, "Hanoi", 1968, "Spain", 7),
    ("comics", 90, "Accra", 2001, "Japan", 2),
    ("cooking", 210, "Basel", 1993, "Mexico", 4),
)


def spike_prompts(count: int) -> list[list[dict[str, str]]]:
    """``count`` chat prompts in Forge's format, the same ones on every run."""
    prompts = []
    for i in range(count):
        genre, pages, city, year, country, many = FILLS[(i // len(QUESTIONS)) % len(FILLS)]
        question = QUESTIONS[i % len(QUESTIONS)].format(
            genre=genre, pages=pages, city=city, year=year, country=country, count=many
        )
        user = USER_TEMPLATE.format(schema=SCHEMA, question=question)
        prompts.append(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        )
    return prompts


@dataclass(frozen=True)
class Stack:
    """A candidate set of GPU packages, installed together into one venv."""

    requirements: tuple[str, ...]
    # Reuse the image's torch and its CUDA build instead of downloading another: the
    # install stage fails if a torch lands in the venv all the same.
    system_site_packages: bool


# Candidate pins, unverified until FG-301 runs on a T4. They conflict, hence one venv each:
# vLLM 0.33.0 pins torch 2.14.0 and transformers < 5.21, and Unsloth's T4 recipe pins
# vLLM 0.11.2 (torch 2.9.0), transformers 4.57.6 and TRL 0.22.2.
STACKS = {
    "vllm": Stack(("vllm==0.33.0",), system_site_packages=False),
    # Forge's training stack as uv.lock has it, on the image's torch: ADR-0006's training
    # stack if vLLM can't run beside it.
    "trl": Stack(
        (
            "trl==1.18.0",
            "transformers==5.23.0",
            "peft==0.21.3",
            "accelerate==1.15.0",
            "datasets==5.2.0",
            "huggingface-hub==2.4.0",
            "bitsandbytes==0.50.2",
        ),
        system_site_packages=True,
    ),
    # Forge's GPU stack, as kaggle/requirements-gpu.txt pins it: the newest vLLM TRL 1.18
    # accepts, with the torch and transformers it needs, so TRL can generate through it.
    "trl-vllm": Stack(
        (
            "vllm==0.33.0",
            "torch==2.14.0",
            "transformers==5.20.2",
            "trl==1.18.0",
            "peft==0.21.3",
            "accelerate==1.15.0",
            "datasets==5.2.0",
            "huggingface-hub==2.4.0",
            "bitsandbytes==0.50.2",
        ),
        system_site_packages=False,
    ),
    # Unsloth's own Kaggle T4 GRPO notebook, with its newest release.
    "unsloth": Stack(
        (
            "unsloth==2026.11.4",
            "unsloth-zoo==2026.11.4",
            "vllm==0.11.2",
            "transformers==4.57.6",
            "trl==0.22.2",
            "datasets==4.3.0",
            "bitsandbytes==0.50.2",
            "torchao>=0.16.0,<0.18.0",
        ),
        system_site_packages=False,
    ),
}
GRPO_STACKS = ("trl", "unsloth", "trl-vllm")  # in the order run measures them


@dataclass(frozen=True)
class Stage:
    name: str
    python: str  # the interpreter the child runs under
    argv: tuple[str, ...]  # the child's subcommand and arguments
    needs: tuple[str, ...] = ()  # stages that must have passed first
    removes: str | None = None  # a venv to delete once this stage is over


@dataclass(frozen=True)
class Outcome:
    status: str  # ok, error or timeout
    result: Any = None
    error: str | None = None


Runner = Callable[[Stage, Path, float], Outcome]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="throughput.py", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--prompts", type=_positive, default=100)
    shared.add_argument("--seed", type=int, default=0)
    grpo = argparse.ArgumentParser(add_help=False)
    grpo.add_argument("--steps", type=_positive, default=10, help="GRPO optimizer steps")
    grpo.add_argument("--num-generations", type=_positive, default=6)
    grpo.add_argument("--max-completion-length", type=_positive, default=256)
    grpo.add_argument("--batch-size", type=_positive, default=6, help="completions per device")
    grpo.add_argument("--grad-accum", type=_positive, default=4)
    grpo.add_argument("--lora-rank", type=_positive, default=16)
    grpo.add_argument("--quant", choices=["4bit", "none"], default="4bit")

    run = commands.add_parser("run", parents=[shared, grpo], help="every stage, one child each")
    run.add_argument("--out-dir", default="/kaggle/working", help="kept as the kernel's output")
    run.add_argument("--work-dir", default="/kaggle/temp/forge-spike", help="discarded scratch")
    run.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    run.add_argument("--vllm", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--stacks", nargs="*", choices=GRPO_STACKS, default=list(GRPO_STACKS))
    run.add_argument("--max-tokens", type=_positive, default=256, help="vLLM tokens per reply")
    run.add_argument("--secret-labels", nargs="*", default=["HF_TOKEN"])
    run.add_argument(
        "--budget-minutes",
        type=float,
        default=80.0,
        help="start no stage after this; push with -t a little longer so the report gets out",
    )

    child = argparse.ArgumentParser(add_help=False)
    child.add_argument("--out", required=True, help="where the child writes its result")
    commands.add_parser("env", parents=[child])
    secrets = commands.add_parser("secrets", parents=[child])
    secrets.add_argument("--labels", nargs="*", default=["HF_TOKEN"])
    download = commands.add_parser("download", parents=[child])
    download.add_argument("--model", required=True)
    download.add_argument("--revision", default=None)
    install = commands.add_parser("install", parents=[child])
    install.add_argument("--stack", choices=sorted(STACKS), required=True)
    install.add_argument("--venv", required=True)
    vllm = commands.add_parser("vllm-bench", parents=[child, shared])
    vllm.add_argument("--model", required=True)
    vllm.add_argument("--revision", default=None)
    vllm.add_argument("--max-tokens", type=_positive, default=256)
    vllm.add_argument("--max-model-len", type=_positive, default=2048)
    vllm.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    bench = commands.add_parser("grpo-bench", parents=[child, shared, grpo])
    bench.add_argument("--model", required=True)
    bench.add_argument("--revision", default=None)
    bench.add_argument("--stack", choices=GRPO_STACKS, required=True)
    bench.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    bench.add_argument("--work-dir", required=True)

    probe = commands.add_parser("limit-probe", help="rewrite a heartbeat until killed")
    probe.add_argument("--out-dir", default="/kaggle/working")
    probe.add_argument("--interval-seconds", type=float, default=30.0)
    probe.add_argument("--max-beats", type=_positive, default=None, help="stop after this many")

    args = parser.parse_args(list(argv))
    # TRL generates batch_size * grad_accum completions at a time, whole groups only.
    whole = args.command not in ("run", "grpo-bench") or (
        args.num_generations >= 2
        and (args.batch_size * args.grad_accum) % args.num_generations == 0
    )
    if not whole:
        parser.error(
            f"--batch-size {args.batch_size} times --grad-accum {args.grad_accum} must be "
            f"a multiple of --num-generations {args.num_generations}, which must be 2 or more"
        )
    cpu = args.command == "grpo-bench" and args.device == "cpu"
    if cpu and (args.quant == "4bit" or args.stack != "trl"):
        parser.error("--device cpu runs the trl stack without --quant 4bit")
    return args


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def plan_stages(args: argparse.Namespace, python: str = sys.executable) -> list[Stage]:
    """What ``run`` does, in order: probes, downloads, then each stack's install and benches."""
    work = Path(args.work_dir)
    stages = [
        Stage("env", python, ("env",)),
        Stage("secrets", python, ("secrets", "--labels", *args.secret_labels)),
    ]
    stages += [Stage(f"download:{m}", python, ("download", "--model", m)) for m in args.models]
    if args.vllm:
        stages += _stack_stages("vllm", "vllm", work, python, args, _vllm_argv)
    for stack in args.stacks:
        bench = functools.partial(_grpo_argv, stack, work)
        stages += _stack_stages(stack, f"grpo:{stack}", work, python, args, bench)
    return stages


def _stack_stages(
    stack: str,
    label: str,
    work: Path,
    python: str,
    args: argparse.Namespace,
    bench_argv: Callable[[str, argparse.Namespace], tuple[str, ...]],
) -> list[Stage]:
    """Install ``stack`` into its own venv, bench each model in it, then delete it."""
    venv = work / "venvs" / stack
    install = Stage(f"install:{stack}", python, ("install", "--stack", stack, "--venv", str(venv)))
    benches = [
        Stage(
            f"{label}:{model}",
            str(venv / "bin" / "python"),
            bench_argv(model, args),
            needs=(install.name, f"download:{model}"),
        )
        for model in args.models
    ]
    benches[-1] = replace(benches[-1], removes=str(venv))
    return [install, *benches]


def _vllm_argv(model: str, args: argparse.Namespace) -> tuple[str, ...]:
    options = _options(prompts=args.prompts, max_tokens=args.max_tokens, seed=args.seed)
    return ("vllm-bench", "--model", model, *options)


def _grpo_argv(stack: str, work: Path, model: str, args: argparse.Namespace) -> tuple[str, ...]:
    options = _options(
        prompts=args.prompts,
        seed=args.seed,
        steps=args.steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lora_rank=args.lora_rank,
        quant=args.quant,
        work_dir=work / "grpo",
    )
    return ("grpo-bench", "--stack", stack, "--model", model, *options)


def _options(**values: object) -> tuple[str, ...]:
    """``--name value`` pairs, underscores in names written as hyphens."""
    return tuple(
        part
        for name, value in values.items()
        for part in (f"--{name.replace('_', '-')}", str(value))
    )


def new_report(args: argparse.Namespace) -> dict[str, Any]:
    now = _now()
    return {
        "schema": REPORT_SCHEMA,
        "story": "FG-301",
        "started_at": now,
        "updated_at": now,
        "finished_at": None,
        # How Kaggle started the script, which FG-303's entry script needs to know.
        "launch": {"argv0": sys.argv[0], "file": _script_file(), "cwd": os.getcwd()},
        "args": {k: v for k, v in vars(args).items() if k != "command"},
        "stages": [],
    }


def run(
    args: argparse.Namespace,
    runner: Runner | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Run every planned stage and keep the report at ``out_dir`` current after each."""
    out_dir, work = Path(args.out_dir), Path(args.work_dir)
    # Children share one model cache, kept out of the output folder.
    runner = runner or functools.partial(run_child, hf_home=work / "hf")
    report = new_report(args)
    report_path = out_dir / REPORT_NAME
    write_json(report_path, report)
    deadline = clock() + args.budget_minutes * 60
    passed: set[str] = set()
    for stage in plan_stages(args):
        entry: dict[str, Any] = {"name": stage.name, "status": "skipped", "started_at": _now()}
        entry.update(seconds=None, result=None, error=None)
        missing = [name for name in stage.needs if name not in passed]
        left = deadline - clock()
        if missing:
            entry["error"] = "needs " + ", ".join(missing)
        elif left <= 0:
            entry["error"] = f"out of budget ({args.budget_minutes:g} min)"
        else:
            start = clock()
            outcome = runner(stage, work / "results" / f"{_slug(stage.name)}.json", left)
            entry.update(status=outcome.status, result=outcome.result, error=outcome.error)
            entry["seconds"] = round(clock() - start, 3)
            if outcome.status == "ok":
                passed.add(stage.name)
        if stage.removes and Path(stage.removes).exists():
            try:
                shutil.rmtree(stage.removes)
            except OSError as exc:
                entry["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        report["stages"].append(entry)
        report["updated_at"] = _now()
        write_json(report_path, report)
        print(f"[spike] {stage.name}: {entry['status']}", flush=True)
    report["finished_at"] = _now()
    write_json(report_path, report)
    return report


def run_child(
    stage: Stage, result_file: Path, timeout: float, hf_home: Path | None = None
) -> Outcome:
    """Run one stage as ``python throughput.py <argv> --out <file>``, its output in the log."""
    result_file.unlink(missing_ok=True)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    command = [stage.python, _script_path(), *stage.argv, "--out", str(result_file)]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
    if hf_home is not None:
        env["HF_HOME"] = str(hf_home)
    # A session of its own, so a kill also reaches every process the stage started.
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        env=env,
        start_new_session=True,
    )
    killed = threading.Event()

    def kill() -> None:
        killed.set()
        os.killpg(proc.pid, signal.SIGKILL)

    timer = threading.Timer(timeout, kill)
    timer.start()
    tail: deque[str] = deque(maxlen=30)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            tail.append(line)
        code = proc.wait()
    finally:
        timer.cancel()
    if killed.is_set():
        return Outcome("timeout", error=f"killed after {timeout:.0f} s, the budget left")
    if result_file.is_file():
        data = json.loads(result_file.read_text(encoding="utf-8"))
        if data["ok"]:
            return Outcome("ok", result=data["result"])
        return Outcome("error", error=data["error"])
    return Outcome("error", error=f"exit code {code} and no result; last output:\n{''.join(tail)}")


def child(stage: Callable[[argparse.Namespace], Any], args: argparse.Namespace) -> int:
    """Run a stage in this process and write its result, or its error, to ``--out``."""
    out = Path(args.out)
    try:
        result = stage(args)
    except Exception as exc:
        traceback.print_exc()
        write_json(out, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    write_json(out, {"ok": True, "result": result})
    return 0


def env_stage(args: argparse.Namespace) -> dict[str, Any]:
    """What the session runs on. Each probe that fails records why and the rest go on."""
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": _probe(lambda: os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")),
        "kaggle_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        # The image's own build stamp, so a later run can tell whether the image moved.
        "image": {
            name: _probe(functools.partial(_read_etc, name))
            for name in ("git_commit", "build_date")
        },
        "packages": versions(),
        "nvidia_smi": _probe(_nvidia_smi),
        "torch": _probe(_torch_info),
        "internet": _probe(_internet),
        "disk_free_bytes": {
            path: shutil.disk_usage(path).free
            for path in ("/kaggle/working", "/kaggle/temp", "/tmp")
            if Path(path).exists()
        },
    }


def secrets_stage(args: argparse.Namespace) -> dict[str, Any]:
    """Whether each Kaggle Secret label can be read. Never records or prints a value."""
    found: dict[str, Any] = {"run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE")}
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError as exc:
        found["client"] = f"unavailable: {type(exc).__name__}"
        return found
    client = UserSecretsClient()
    labels: dict[str, Any] = {}
    for label in args.labels:
        try:
            value = client.get_secret(label)
        except Exception as exc:  # the client raises its own CredentialError and BackendError
            # Only the type: a message is not worth the risk of echoing anything secret.
            labels[label] = {"readable": False, "error": type(exc).__name__}
        else:
            labels[label] = {"readable": bool(value)}
            del value
    found["client"] = "available"
    found["labels"] = labels
    return found


def download_stage(args: argparse.Namespace) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    start = time.perf_counter()
    path = Path(snapshot_download(args.model, revision=args.revision, allow_patterns=MODEL_FILES))
    return {
        "model": args.model,
        "commit": path.name,  # a snapshot folder is named after its commit
        "seconds": round(time.perf_counter() - start, 3),
        "bytes": _tree_bytes(path),
    }


def install_stage(args: argparse.Namespace) -> dict[str, Any]:
    """Create the stack's venv and time how long its packages take to install."""
    stack = STACKS[args.stack]
    venv = Path(args.venv)
    uv = [found] if (found := shutil.which("uv")) else None
    bootstrap = None
    if uv is None:
        start = time.perf_counter()
        _check([sys.executable, "-m", "pip", "install", "--quiet", UV_PIN])
        bootstrap = round(time.perf_counter() - start, 3)
        uv = [sys.executable, "-m", "uv"]
    start = time.perf_counter()
    python = str(venv / "bin" / "python")
    if stack.system_site_packages:
        # uv ignores system site packages when it installs, so it would put a PyPI torch
        # in the venv for trl and peft; the venv's own pip counts the image's as there.
        flags = ["--seed", "--system-site-packages"]
        _check([*uv, "venv", "--python", sys.executable, *flags, str(venv)])
        pip = [python, "-m", "pip", "install", "--no-input", "--progress-bar", "off"]
        _check([*pip, *stack.requirements])
        installer = "pip"
    else:
        _check([*uv, "venv", "--python", sys.executable, str(venv)])
        _check([*uv, "pip", "install", "--python", python, *stack.requirements])
        installer = "uv"
    seconds = round(time.perf_counter() - start, 3)
    if stack.system_site_packages and (own := _venv_packages(venv).get("torch")):
        raise RuntimeError(
            f"the install put torch {own} into {venv}, which would shadow the "
            "image's torch the stack is meant to run on"
        )
    return {
        "stack": args.stack,
        "requirements": list(stack.requirements),
        "system_site_packages": stack.system_site_packages,
        "installer": installer,
        "uv": _check([*uv, "--version"]).strip(),
        "uv_bootstrap_seconds": bootstrap,
        "install_seconds": seconds,
        "venv_bytes": _tree_bytes(venv),
        "freeze": _check([*uv, "pip", "freeze", "--python", python]).splitlines(),
    }


def vllm_stage(args: argparse.Namespace) -> dict[str, Any]:
    """Greedy fp16 generation on the spike prompts: a T4 has no bf16."""
    from vllm import LLM, SamplingParams

    start = time.perf_counter()
    llm = LLM(
        model=args.model,
        revision=args.revision,
        dtype="half",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    load_seconds = time.perf_counter() - start
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=args.seed)
    start = time.perf_counter()
    outputs = llm.chat(
        spike_prompts(args.prompts),
        params,
        use_tqdm=False,
        chat_template_kwargs={"enable_thinking": False},
    )
    seconds = time.perf_counter() - start
    completion_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
    return {
        "model": args.model,
        "settings": {
            "dtype": "half",
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
        },
        "versions": versions(),
        "load_seconds": round(load_seconds, 3),
        "generate_seconds": round(seconds, 3),
        "prompts": len(outputs),
        "prompt_tokens": sum(len(out.prompt_token_ids) for out in outputs),
        "completion_tokens": completion_tokens,
        "completion_tokens_per_second": round(completion_tokens / seconds, 1),
        "cut_at_max_tokens": sum(out.outputs[0].finish_reason == "length" for out in outputs),
        "memory": cuda_memory(),
        # A few replies, to check they are SQL and that thinking stayed off.
        "samples": [out.outputs[0].text[:400] for out in outputs[:3]],
    }


def sql_block_reward(
    prompts: list[Any], completions: list[Any], **kwargs: Any
) -> list[float | None]:
    """1 for a reply with a closed sql block, else 0: a cheap stand-in for Forge's rewards,
    so the timing is the trainer's and not the SQL executor's."""
    texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
    return [1.0 if re.search(r"```sql\s.*?```", text, re.DOTALL) else 0.0 for text in texts]


def grpo_stage(args: argparse.Namespace) -> dict[str, Any]:
    """Time ``--steps`` GRPO steps, LoRA in fp16 on the GPU (plain fp32 with ``--device cpu``)."""
    on_gpu = args.device == "cuda"
    if args.stack == "unsloth":
        # Unsloth patches TRL when it is imported, so it has to come first.
        os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")
        import unsloth  # noqa: F401

    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    marks: list[float] = []

    class StepTimer(TrainerCallback):
        def on_train_begin(self, *args: Any, **kwargs: Any) -> None:
            marks.append(time.perf_counter())

        def on_step_end(self, *args: Any, **kwargs: Any) -> None:
            marks.append(time.perf_counter())

    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model, tokenizer, peft_config = (_unsloth_model if args.stack == "unsloth" else _hf_model)(args)
    load_seconds = time.perf_counter() - start

    fields = GRPOConfig.__dataclass_fields__
    config: dict[str, Any] = {
        "output_dir": str(Path(args.work_dir) / args.stack),
        "max_steps": args.steps,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "learning_rate": 5e-6,  # plan section 3.10
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": "none",
        "seed": args.seed,
        # A T4 has no bf16, and TRL's bf16 default crashes there (TRL issue 7889).
        "fp16": on_gpu,
        "bf16": False,
        "use_cpu": not on_gpu,
        "optim": "adamw_8bit" if on_gpu else "adamw_torch",
    }
    if "chat_template_kwargs" in fields:
        config["chat_template_kwargs"] = {"enable_thinking": False}
    if "max_prompt_length" in fields:
        # TRL before 0.28 cuts prompts to 512 tokens unless told otherwise.
        config["max_prompt_length"] = MAX_PROMPT_TOKENS
    if args.stack == "trl-vllm":
        config.update(use_vllm=True, vllm_mode="colocate")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[sql_block_reward],
        args=GRPOConfig(**config),
        train_dataset=Dataset.from_list([{"prompt": p} for p in spike_prompts(args.prompts)]),
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[StepTimer()],
    )
    if args.quant == "4bit" and args.stack != "unsloth":
        # TRL casts QLoRA adapters to bf16, which fp16's grad scaler can't unscale (TRL
        # issue 7776); fp32 adapters train fine under fp16 autocast.
        trained = trainer.model
        assert trained is not None  # the constructor sets it
        for param in trained.parameters():
            if param.requires_grad:
                param.data = param.data.float()

    start = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - start
    steps = [round(b - a, 3) for a, b in itertools.pairwise(marks)]
    settings = {key: value for key, value in config.items() if key not in ("output_dir", "seed")}
    settings["lora_rank"] = args.lora_rank
    # Unsloth's fast_inference generates with its own colocated vLLM.
    settings["generates_with_vllm"] = args.stack in ("trl-vllm", "unsloth")
    return {
        "stack": args.stack,
        "model": args.model,
        "device": args.device,
        "quant": args.quant,
        "settings": settings,
        "completions_per_step": args.batch_size * args.grad_accum,
        "versions": versions(),
        "load_seconds": round(load_seconds, 3),
        "train_seconds": round(train_seconds, 3),
        # The first step also builds the optimizer and warms the kernels up.
        "step_seconds": steps,
        "median_step_seconds": statistics.median(steps[1:]) if len(steps) > 1 else None,
        "memory": cuda_memory() if on_gpu else None,
        "log_history": trainer.state.log_history,
    }


def _hf_model(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    """The model through transformers, 4-bit NF4 if asked, and the LoRA config TRL adds."""
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    on_gpu = args.device == "cuda"
    quant = None
    if args.quant == "4bit":
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.float16 if on_gpu else torch.float32,
        quantization_config=quant,
        # No FlashAttention 2 below compute capability 8.0.
        attn_implementation="sdpa",
        device_map={"": 0} if on_gpu else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, padding_side="left"
    )
    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
        lora_dropout=0.0,
        target_modules=list(LORA_TARGETS),
        task_type="CAUSAL_LM",
    )
    return model, tokenizer, lora


def _unsloth_model(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    """The model through Unsloth with its colocated vLLM, LoRA already applied."""
    import torch
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        revision=args.revision,
        max_seq_length=MAX_PROMPT_TOKENS + args.max_completion_length,
        load_in_4bit=args.quant == "4bit",
        fast_inference=True,
        max_lora_rank=args.lora_rank,
        dtype=torch.float16,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
        target_modules=list(LORA_TARGETS),
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    return model, tokenizer, None


def limit_probe(args: argparse.Namespace) -> int:
    """Rewrite a heartbeat file every interval until the session is killed."""
    path = Path(args.out_dir) / PROBE_NAME
    started, beats = _now(), 0
    while True:
        beats += 1
        write_json(
            path,
            {
                "schema": REPORT_SCHEMA,
                "started_at": started,
                "last_beat_at": _now(),
                "beats": beats,
                "interval_seconds": args.interval_seconds,
                "run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
            },
        )
        print(f"[limit-probe] beat {beats}", flush=True)
        if args.max_beats is not None and beats >= args.max_beats:
            return 0
        time.sleep(args.interval_seconds)


def versions() -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return found


def cuda_memory() -> dict[str, int] | None:
    """Peak memory of this process's allocator, and what the whole device has in use."""
    import torch

    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        # vLLM's engine runs in a process of its own, which only this number sees.
        "device_used_bytes": total - free,
        "device_total_bytes": total,
    }


def write_json(path: Path, data: Any) -> None:
    """Replace ``path`` in one step, so a kill mid-write never leaves half a report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    # default=str: a value JSON can't hold is kept as text rather than losing the report.
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _probe(read: Callable[[], Any]) -> Any:
    try:
        return read()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _read_etc(name: str) -> str:
    return Path("/etc", name).read_text(encoding="utf-8").strip()


def _nvidia_smi() -> list[str]:
    query = "--query-gpu=name,driver_version,memory.total,compute_cap"
    return _check(["nvidia-smi", query, "--format=csv,noheader"]).splitlines()


def _torch_info() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "version": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "devices": [],
    }
    if torch.cuda.is_available():
        # Without emulation: the T4 (sm_75) runs bf16 only in software.
        info["bf16_supported"] = torch.cuda.is_bf16_supported(including_emulation=False)
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append(
                {
                    "name": props.name,
                    "capability": f"{props.major}.{props.minor}",
                    "total_memory_bytes": props.total_memory,
                }
            )
    return info


def _internet() -> dict[str, Any]:
    request = urllib.request.Request(INTERNET_PROBE_URL, method="HEAD")
    with urllib.request.urlopen(request, timeout=10) as response:
        return {"url": INTERNET_PROBE_URL, "status": response.status}


def _check(command: list[str]) -> str:
    """Run ``command`` and return its output; on failure, raise with the end of its errors."""
    done = subprocess.run(command, capture_output=True, text=True)
    sys.stderr.write(done.stderr)
    if done.returncode != 0:
        tail = done.stderr[-2000:]
        raise RuntimeError(f"{shlex.join(command)} exited {done.returncode}: {tail}")
    return done.stdout


def _venv_packages(venv: Path) -> dict[str, str]:
    """The distributions installed in ``venv`` itself, by lowercase name: not those it sees
    through system site packages, which ``importlib.metadata`` would list too."""
    folders = [str(path) for path in sorted(venv.glob("lib/python*/site-packages"))]
    if not folders:
        raise RuntimeError(f"no lib/python*/site-packages under {venv} to check")
    found: dict[str, str] = {}
    for dist in importlib.metadata.distributions(path=folders):
        name = dist.metadata["Name"]
        if name:
            found[re.sub(r"[-_.]+", "-", name).lower()] = dist.version
    return found


def _tree_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", name).strip("-")


def _script_file() -> str | None:
    file = globals().get("__file__")
    return str(Path(file).resolve()) if file else None


def _script_path() -> str:
    """This script's file, which every child runs: Kaggle may not set ``__file__``."""
    for candidate in (_script_file(), sys.argv[0]):
        if candidate and candidate.endswith(".py") and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise RuntimeError("can't find this script's file to start a stage; run it as a .py file")


CHILDREN: dict[str, Callable[[argparse.Namespace], Any]] = {
    "env": env_stage,
    "secrets": secrets_stage,
    "download": download_stage,
    "install": install_stage,
    "vllm-bench": vllm_stage,
    "grpo-bench": grpo_stage,
}


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:] or KERNEL_ARGV or ["run"]
    args = parse_args(argv)
    if args.command == "run":
        # Exit 0 even when stages fail: the report says which, and a kernel that ends in
        # an error might not keep its output.
        run(args)
        return 0
    if args.command == "limit-probe":
        return limit_probe(args)
    return child(CHILDREN[args.command], args)


if __name__ == "__main__":
    raise SystemExit(main())
