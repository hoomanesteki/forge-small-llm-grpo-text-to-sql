"""Throughput spike for FG-301: what one Kaggle T4 session can do, measured.

Pushed as a private script kernel, it writes one JSON report to /kaggle/working, which
`kaggle kernels output` brings back. docs/spikes/kaggle-gpu.md says how to push it and
which question each stage answers:

- env: Python, CUDA, torch, the GPUs, the image build, internet and disk.
- secrets: whether Kaggle Secrets can be read in a pushed run. The report says only
  whether a read worked, never what it returned.
- limit-probe, pushed as its own CPU kernel: a heartbeat file rewritten until the session
  is killed, so the output of a timed-out session shows what survives the kill.

Each stage runs in a child process, so a crash or an out-of-memory kill in one leaves the
others and the report intact; the report is rewritten after every stage.

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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

KERNEL_ARGV: list[str] = []

REPORT_SCHEMA = "forge-kaggle-spike/1"
REPORT_NAME = "spike-report.json"
PROBE_NAME = "limit-probe.json"
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


@dataclass(frozen=True)
class Stage:
    name: str
    python: str  # the interpreter the child runs under
    argv: tuple[str, ...]  # the child's subcommand and arguments


@dataclass(frozen=True)
class Outcome:
    status: str  # ok, error or timeout
    result: Any = None
    error: str | None = None


Runner = Callable[[Stage, Path, float], Outcome]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="throughput.py", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="every stage, one child each")
    run.add_argument("--out-dir", default="/kaggle/working", help="kept as the kernel's output")
    run.add_argument("--work-dir", default="/kaggle/temp/forge-spike", help="discarded scratch")
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
    """What ``run`` does, in order."""
    return [
        Stage("env", python, ("env",)),
        Stage("secrets", python, ("secrets", "--labels", *args.secret_labels)),
    ]


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
    runner = runner or run_child
    out_dir, work = Path(args.out_dir), Path(args.work_dir)
    report = new_report(args)
    report_path = out_dir / REPORT_NAME
    write_json(report_path, report)
    deadline = clock() + args.budget_minutes * 60
    for stage in plan_stages(args):
        entry: dict[str, Any] = {"name": stage.name, "status": "skipped", "started_at": _now()}
        entry.update(seconds=None, result=None, error=None)
        left = deadline - clock()
        if left <= 0:
            entry["error"] = f"out of budget ({args.budget_minutes:g} min)"
        else:
            start = clock()
            outcome = runner(stage, work / "results" / f"{_slug(stage.name)}.json", left)
            entry.update(status=outcome.status, result=outcome.result, error=outcome.error)
            entry["seconds"] = round(clock() - start, 3)
        report["stages"].append(entry)
        report["updated_at"] = _now()
        write_json(report_path, report)
        print(f"[spike] {stage.name}: {entry['status']}", flush=True)
    report["finished_at"] = _now()
    write_json(report_path, report)
    return report


def run_child(stage: Stage, result_file: Path, timeout: float) -> Outcome:
    """Run one stage as ``python throughput.py <argv> --out <file>``, its output in the log."""
    result_file.unlink(missing_ok=True)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    command = [stage.python, _script_path(), *stage.argv, "--out", str(result_file)]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
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
