"""Forge's job in a Kaggle kernel: install the pushed commit's code, check the data, run it.

``forge kaggle run`` writes this script into the kernel it pushes, with the job's arguments
in ``KERNEL_ARGV``, since Kaggle starts a script kernel with none. Until the code dataset's
wheel is installed it needs only the standard library, and the job modules
(``forge.eval.runner``, later ``forge.train``) are imported only when their stage runs, so
the script can be tested on a laptop with fakes. The stages, in order (``STAGES``):

- env: the interpreter, the platform and the GPUs, for the log.
- code: the code dataset under the input folder, whose ``GIT_SHA`` must be the commit the
  job was pushed at. That commit goes to the job in ``FORGE_CODE_SHA``, which a manifest
  takes as its commit where there is no git checkout (``forge.ops.manifest.code_sha``).
- install: ``pip install --no-deps`` the wheel, then its ``requirements-gpu.txt``; every
  package the wheel needs must then be there, or the stage says which is not.
- data: the Spider dataset, found by its ``spider.sha256``, with ``database.zip`` unpacked
  if Kaggle left it packed, and every file checked against that list before any job runs.
- secrets: ``HF_TOKEN`` and ``WANDB_API_KEY`` from Kaggle Secrets (or Colab's) when they
  are attached. No value is ever printed or written, and the job goes on without them.
- preflight: the job on ``PREFLIGHT_EXAMPLES`` examples through the same path, in a run
  of its own under ``preflight/``, so a broken job fails in minutes. The job then runs on
  the model the preflight loaded: a vLLM engine holds most of the GPU's memory while it
  lives, so a second one might not fit beside it, and loading once saves GPU time. A dev
  job's preflight answers val questions, so dev loads once, in the job.
- job: the job itself, its files in ``<work>/runs/<run_id>/``. On a resume, the earlier
  attempt's files come from its kernel's output, mounted under the input folder.
- upload: that folder to the Hub run store (``forge.ops.hub``, which tries again on
  network errors), only when there is an ``HF_TOKEN``.

A dev job's ``load_dev`` logs its look in ``dev-usage.md`` in the output folder, since a
kernel has no checkout; the repo's ``docs/dev-usage.md`` got the row when ``forge kaggle
run`` pushed the job.

Whatever happens, a ``finally`` marks a manifest still ``running`` as failed and writes
``runs/<run_id>/status.json``: the stage reached, how the job ended (finished, failed or
stopped_time_guard) and why. The script exits 0 either way, because a kernel that ends in
an error might not keep its output; ``forge kaggle wait`` reads status.json instead.

The GPU is held from the kernel's start to its end, setup and upload included, so that is
the session the ledger counts. status.json gives its ``started_at`` and ``ended_at``, and
the same span goes into the run's ``sessions.json`` (``forge.ops.sessions``): the session
the job began in this kernel is stretched to it, or, when none began, as when a resumed
job fails before its run starts again, the kernel's session is added. ``session`` in
status.json is its number. The upload stage stretches it up to the upload first, so the
run store's copy holds nearly all of it.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

KERNEL_ARGV: list[str] = []

STATUS_SCHEMA = "forge-kaggle-job/1"
STATUS_FILE = "status.json"
STAGES = ("env", "code", "install", "data", "secrets", "preflight", "job", "upload")
PREFLIGHT_EXAMPLES = 5
SECRET_LABELS = ("HF_TOKEN", "WANDB_API_KEY")
# The names forge.ops.kaggle and forge.ops.manifest use; this script can't import them
# before the wheel is installed, so a unit test keeps each copy in step.
GIT_SHA_FILE = "GIT_SHA"
REQUIREMENTS_FILE = "requirements-gpu.txt"
CHECKSUM_FILE = "spider.sha256"
CODE_SHA_ENV = "FORGE_CODE_SHA"
MANIFEST_FILE = "manifest.json"
SESSIONS_FILE = "sessions.json"
DEV_USAGE_ENV = "FORGE_DEV_USAGE"
DEV_USAGE_FILE = "dev-usage.md"
HOSTS = ("kaggle", "colab")
PIP = (sys.executable, "-m", "pip", "install", "--no-input", "--progress-bar", "off")

# Runs one command, raising CalledProcessError when it fails; tests stand in for pip.
Runner = Callable[[Sequence[str]], None]


class JobError(RuntimeError):
    """A stage can't go on: the message says what is missing or wrong."""


class Stopped(BaseException):
    """The session was told to stop: Kaggle's timeout, or a cancel."""


@dataclass
class Kernel:
    """One run of this script: its arguments, its folders and what it has learned."""

    args: argparse.Namespace
    run: Runner
    status: dict[str, Any]
    secrets: list[str] = field(default_factory=list)  # values, only to keep them out of text
    wheel: Path | None = None
    requirements: Path | None = None
    # The eval job's model backend and token counter, loaded once for the preflight and the job.
    model: tuple[Any, Any] | None = None

    @property
    def work(self) -> Path:
        return Path(self.args.work_dir)

    @property
    def run_dir(self) -> Path:
        return self.work / "runs" / self.args.run_id


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="entry.py", description=__doc__.splitlines()[0])
    parser.add_argument("--job", choices=sorted(JOBS), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--options", type=_json_object, required=True, help="the job's options")
    parser.add_argument("--code-sha", required=True, help="the commit the job was pushed at")
    parser.add_argument("--run-store", required=True, help="the Hub repo the run goes to")
    parser.add_argument("--host", choices=HOSTS, default="kaggle")
    parser.add_argument("--input-dir", default="/kaggle/input")
    parser.add_argument("--work-dir", default="/kaggle/working", help="kept as the output")
    parser.add_argument("--temp-dir", default="/kaggle/temp", help="discarded scratch")
    parser.add_argument("--preflight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true", help="continue the run's last attempt")
    return parser.parse_args(list(argv))


def _json_object(text: str) -> dict[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError(f"must be a JSON object, got {text!r}")
    return value


def env_stage(kernel: Kernel) -> dict[str, Any]:
    print(f"[forge] python {platform.python_version()} at {sys.executable}", flush=True)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        "gpus": _gpus(),
    }


def code_stage(kernel: Kernel) -> dict[str, Any]:
    """Find the wheel and its requirements, and check they are the pushed commit's."""
    folder = _one(Path(kernel.args.input_dir), GIT_SHA_FILE, "the code dataset").parent
    found = (folder / GIT_SHA_FILE).read_text(encoding="utf-8").strip()
    if found != kernel.args.code_sha:
        raise JobError(
            f"the code dataset in {folder} holds commit {found}, but the job was pushed at "
            f"{kernel.args.code_sha}; push it again so it mounts that commit's version"
        )
    wheels = sorted(folder.glob("forge-*.whl"))
    if len(wheels) != 1:
        raise JobError(f"{folder} must hold one forge wheel, not {[w.name for w in wheels]}")
    kernel.wheel, kernel.requirements = wheels[0], folder / REQUIREMENTS_FILE
    if not kernel.requirements.is_file():
        raise JobError(f"{folder} has no {REQUIREMENTS_FILE}")
    os.environ[CODE_SHA_ENV] = found
    return {"folder": str(folder), "wheel": wheels[0].name, "sha": found}


def install_stage(kernel: Kernel) -> dict[str, Any]:
    assert kernel.wheel is not None and kernel.requirements is not None
    # No dependencies from the wheel: pip would resolve them against the image's torch.
    kernel.run([*PIP, "--no-deps", str(kernel.wheel)])
    kernel.run([*PIP, "-r", str(kernel.requirements)])
    importlib.invalidate_caches()
    unmet = unmet_requirements(importlib.metadata.requires("forge") or [], _installed)
    if unmet:
        raise JobError(
            "the image lacks what forge needs: "
            + "; ".join(unmet)
            + f". Pin each in kaggle/{REQUIREMENTS_FILE} at its uv.lock version"
        )
    return {"forge": importlib.metadata.version("forge")}


def unmet_requirements(
    requires: Iterable[str], installed: Callable[[str], str | None]
) -> list[str]:
    """Each of a wheel's requirements, extras aside, that ``installed`` doesn't meet."""
    from packaging.requirements import Requirement

    unmet = []
    for line in requires:
        requirement = Requirement(line)
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue
        version = installed(requirement.name)
        if version is None:
            unmet.append(f"{requirement} is not installed")
        elif not requirement.specifier.contains(version, prereleases=True):
            unmet.append(f"{requirement} but {version} is installed")
    return unmet


def data_stage(kernel: Kernel) -> dict[str, Any]:
    """Find Spider, unpack its databases if needed, and check every file it lists."""
    from forge.data.spider import check_files, load_checksums

    folder = _one(Path(kernel.args.input_dir), CHECKSUM_FILE, "the Spider dataset").parent
    if not (folder / "database").is_dir():
        folder = _unpacked(folder, Path(kernel.args.temp_dir) / "forge-spider")
    expected = load_checksums(folder / CHECKSUM_FILE)
    problems = check_files(folder, expected)
    if problems:
        shown = ", ".join(f"{rel} ({why})" for rel, why in sorted(problems.items())[:5])
        raise JobError(f"{len(problems)} Spider files don't match {CHECKSUM_FILE}: {shown}")
    os.environ["FORGE_DATA_DIR"] = str(folder)
    return {"folder": str(folder), "files": len(expected)}


def _unpacked(folder: Path, target: Path) -> Path:
    """A copy of ``folder`` with its ``database.zip`` unpacked into ``database/``."""
    packed = folder / "database.zip"
    if not packed.is_file():
        raise JobError(f"{folder} has neither database/ nor database.zip")
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    for item in folder.iterdir():
        if item.is_file() and item != packed:
            shutil.copyfile(item, target / item.name)
    with zipfile.ZipFile(packed) as archive:
        # Whether Kaggle zips the folder itself or only what is in it is not documented.
        inside = all(name.startswith("database/") for name in archive.namelist())
        archive.extractall(target if inside else target / "database")
    return target


def secrets_stage(kernel: Kernel) -> dict[str, str]:
    """Where each secret came from, or why it is missing. Never what it is."""
    found = {}
    for label in SECRET_LABELS:
        value, where = os.environ.get(label), "environment"
        if not value:
            value, where = _read_secret(label)
        if value:
            # The Hub and W&B clients read these variables.
            os.environ[label] = value
            kernel.secrets.append(value)
        found[label] = where
        print(f"[forge] {label}: {where if value else 'not available, ' + where}", flush=True)
    return found


def _read_secret(label: str) -> tuple[str | None, str]:
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        pass
    else:
        try:
            return UserSecretsClient().get_secret(label), "kaggle secrets"
        # The client raises its own CredentialError and BackendError, and ConnectionError.
        # Only the type is kept: a message is not worth the risk of echoing a secret.
        except Exception as exc:
            return None, f"kaggle secrets: {type(exc).__name__}"
    try:
        from google.colab import userdata
    except ImportError:
        return None, "no secrets client on this host"
    try:
        return userdata.get(label), "colab secrets"
    except Exception as exc:  # SecretNotFoundError, NotebookAccessError
        return None, f"colab secrets: {type(exc).__name__}"


def preflight_stage(kernel: Kernel) -> dict[str, Any] | str:
    if not kernel.args.preflight:
        return "skipped (--no-preflight)"
    status = JOBS[kernel.args.job](kernel, True)
    if status != "finished":
        raise JobError(f"the preflight ended {status}; see {kernel.work / 'preflight'}")
    return {"status": status, "examples": PREFLIGHT_EXAMPLES}


def job_stage(kernel: Kernel) -> dict[str, Any]:
    if kernel.args.resume:
        _restore(kernel)
    status = JOBS[kernel.args.job](kernel, False)
    kernel.status["status"] = status
    return {"status": status}


def _restore(kernel: Kernel) -> None:
    """Copy the run's files from its last attempt's output, so the job carries on."""
    where = f"runs/{kernel.args.run_id}/manifest.json"
    manifest = _one(Path(kernel.args.input_dir), where, "the last attempt's output")
    if kernel.run_dir.exists():
        raise JobError(f"{kernel.run_dir} exists already, so the last attempt can't go there")
    shutil.copytree(manifest.parent, kernel.run_dir)


def eval_job(kernel: Kernel, preflight: bool) -> str:
    """``forge eval run`` with the job's options, as the runner runs it on a laptop, on the
    model the kernel loaded for an earlier run, if any."""
    from forge.eval.config import EvalConfig, check_config
    from forge.eval.report import write_run_report
    from forge.eval.runner import MODEL_BACKENDS, model_backend, run_eval

    options = dict(kernel.args.options)
    runs_root = kernel.work / ("preflight" if preflight else "runs")
    if preflight:
        options["limit"] = min(options.get("limit") or PREFLIGHT_EXAMPLES, PREFLIGHT_EXAMPLES)
        options["run_id"] = f"{kernel.args.run_id}-preflight"
        if options.get("split") == "dev":
            # Every load of dev is a look, and the push paid for one: the job's.
            options["split"] = "val"
            del options["reason"]
    else:
        options["run_id"] = kernel.args.run_id
    config = EvalConfig(**options, host=kernel.args.host)
    if config.backend in MODEL_BACKENDS and kernel.model is None:
        check_config(config)  # before a model loads for options no run takes
        kernel.model = model_backend(config)
    backend, count_tokens = kernel.model or (None, None)
    run = run_eval(
        config,
        runs_root=runs_root,
        progress=_progress,
        backend=backend,
        count_tokens=count_tokens,
    )
    if not preflight:
        write_run_report(kernel.args.run_id, runs_root=runs_root)
    return run.manifest.status


def upload_stage(kernel: Kernel) -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        return "skipped: no HF_TOKEN, so the kernel output is the only copy"
    from forge.ops.hub import HubRunStore

    repo = kernel.args.run_store
    _record_session(kernel.run_dir, kernel.status["started_at"], _now())
    try:
        HubRunStore(repo, token=token).put_folder(kernel.args.run_id, kernel.run_dir)
    # The run is complete in the kernel output whatever the Hub says, so a failed upload
    # is recorded rather than turned into a failed job.
    except Exception as exc:
        return f"failed: {type(exc).__name__}: {exc}"
    return f"uploaded to {repo}"


JOBS: Mapping[str, Callable[[Kernel, bool], str]] = {"eval": eval_job}
STAGE_FUNCTIONS: Mapping[str, Callable[[Kernel], Any]] = {
    "env": env_stage,
    "code": code_stage,
    "install": install_stage,
    "data": data_stage,
    "secrets": secrets_stage,
    "preflight": preflight_stage,
    "job": job_stage,
    "upload": upload_stage,
}


def main(argv: Sequence[str] | None = None, run: Runner | None = None) -> int:
    args = parse_args(KERNEL_ARGV if argv is None else argv)
    status: dict[str, Any] = {
        "schema": STATUS_SCHEMA,
        "run_id": args.run_id,
        "job": args.job,
        "host": args.host,
        "code_sha": args.code_sha,
        "options": args.options,
        "preflight": args.preflight,
        "resume": args.resume,
        "status": "failed",
        "stage": None,
        "error": None,
        "started_at": _now(),
        "ended_at": None,
        "session": None,
        "stages": {},
    }
    kernel = Kernel(args, run or _run, status)
    os.environ["FORGE_RUNS_DIR"] = str(kernel.work / "runs")
    os.environ["FORGE_CACHE_DIR"] = str(Path(args.temp_dir) / "cache")
    os.environ[DEV_USAGE_ENV] = str(kernel.work / DEV_USAGE_FILE)
    # Outside any checkout, so a manifest's commit is FORGE_CODE_SHA's: the wheel's own.
    kernel.work.mkdir(parents=True, exist_ok=True)
    os.chdir(kernel.work)
    previous = signal.signal(signal.SIGTERM, _stop)
    try:
        for name in STAGES:
            status["stage"] = name
            started = time.monotonic()
            result = STAGE_FUNCTIONS[name](kernel)
            status["stages"][name] = {"seconds": round(time.monotonic() - started, 1)}
            status["stages"][name]["result"] = result
    except BaseException as exc:
        trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        print(_redact(trace, kernel.secrets), file=sys.stderr, flush=True)
        status["status"] = "failed"
        status["error"] = _redact(f"{type(exc).__name__}: {exc}", kernel.secrets)
    finally:
        signal.signal(signal.SIGTERM, previous)
        _close_manifest(kernel.run_dir)
        status["ended_at"] = _now()
        status["session"] = _record_session(
            kernel.run_dir, status["started_at"], status["ended_at"]
        )
        _write_json(kernel.run_dir / STATUS_FILE, status)
        print(f"[forge] {args.job} {args.run_id}: {status['status']}", flush=True)
    return 0


def _close_manifest(run_dir: Path) -> None:
    """Mark failed a manifest that a crash or a kill left ``running``."""
    path = run_dir / MANIFEST_FILE
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") == "running":
            manifest.update(status="failed", ended_at=_now())
            _write_json(path, manifest)


def _record_session(run_dir: Path, started_at: str, ended_at: str) -> int | None:
    """Make the run's session in this kernel span ``started_at`` to ``ended_at``, the
    kernel's, and return its number; None when the kernel holds no run.

    The session the job began here is the run's last, begun after the kernel started. A
    session still running then failed. When the job began none, the kernel's is added, with
    the run's status if the run ended in this kernel, and failed if not.
    """
    manifest_path = run_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        return None
    path = run_dir / SESSIONS_FILE
    sessions = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    start = datetime.fromisoformat(started_at)
    if sessions and datetime.fromisoformat(sessions[-1]["started_at"]) >= start:
        session = sessions[-1]
        if session["status"] == "running":
            session["status"] = "failed"
    else:
        run = json.loads(manifest_path.read_text(encoding="utf-8"))
        ended = run.get("ended_at")
        here = ended is not None and datetime.fromisoformat(ended) >= start
        session = {"status": run["status"] if here else "failed"}
        sessions.append(session)
    session.update(started_at=started_at, ended_at=ended_at)
    _write_json(path, sessions)
    return len(sessions)


def _stop(signum: int, frame: object) -> None:
    raise Stopped(f"stopped by signal {signum}")


def _run(command: Sequence[str]) -> None:
    print(f"[forge] {' '.join(command)}", flush=True)
    subprocess.run(list(command), check=True)


def _progress(done: int, total: int) -> None:
    print(f"[forge] scored {done} of {total}", flush=True)


def _installed(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _one(root: Path, pattern: str, what: str) -> Path:
    found = sorted(root.glob(f"**/{pattern}"))
    if len(found) != 1:
        raise JobError(f"{what} should hold one {pattern} under {root}, found {len(found)}")
    return found[0]


def _gpus() -> list[str] | str:
    command = ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
    try:
        done = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {type(exc).__name__}"
    return [line.strip() for line in done.stdout.splitlines() if line.strip()]


def _redact(text: str, secrets: Iterable[str]) -> str:
    for secret in secrets:
        text = text.replace(secret, "***")
    return text


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
