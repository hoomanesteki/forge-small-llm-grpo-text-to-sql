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
- limit-probe, pushed as its own CPU kernel: a heartbeat file rewritten until the session
  is killed, so the output of a timed-out session shows what survives the kill.

Each stage runs in a child process, so a crash or an out-of-memory kill in one leaves the
others and the report intact; the report is rewritten after every stage. A GPU stack
installs into a venv of its own, deleted after use, so it can't disturb the image.

The script needs only the standard library to start: heavy imports happen inside the
stages, and nothing runs on import. Kaggle runs a script kernel without arguments, so
scripts/kaggle_spike.py writes them into KERNEL_ARGV when it bundles the kernel.
"""

from __future__ import annotations

import argparse
import functools
import importlib.metadata
import json
import os
import platform
import re
import shlex
import shutil
import signal
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
    # Reuse the image's torch and its CUDA build instead of downloading another.
    system_site_packages: bool


# Candidate pins, unverified until FG-301 runs on a T4. vLLM 0.33.0 pins torch 2.14.0 and
# transformers < 5.21, so it gets a venv of its own.
STACKS = {
    "vllm": Stack(("vllm==0.33.0",), system_site_packages=False),
}


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

    run = commands.add_parser("run", parents=[shared], help="every stage, one child each")
    run.add_argument("--out-dir", default="/kaggle/working", help="kept as the kernel's output")
    run.add_argument("--work-dir", default="/kaggle/temp/forge-spike", help="discarded scratch")
    run.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    run.add_argument("--vllm", action=argparse.BooleanOptionalAction, default=True)
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

    probe = commands.add_parser("limit-probe", help="rewrite a heartbeat until killed")
    probe.add_argument("--out-dir", default="/kaggle/working")
    probe.add_argument("--interval-seconds", type=float, default=30.0)
    probe.add_argument("--max-beats", type=_positive, default=None, help="stop after this many")

    return parser.parse_args(list(argv))


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
    """Create the stack's venv and time how long its packages take to install with uv."""
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
    flags = ["--system-site-packages"] if stack.system_site_packages else []
    _check([*uv, "venv", "--python", sys.executable, *flags, str(venv)])
    python = str(venv / "bin" / "python")
    _check([*uv, "pip", "install", "--python", python, *stack.requirements])
    seconds = round(time.perf_counter() - start, 3)
    return {
        "stack": args.stack,
        "requirements": list(stack.requirements),
        "system_site_packages": stack.system_site_packages,
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
