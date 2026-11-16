"""Forge's job in a Kaggle kernel: install the pushed commit's code, check the data, run it.

``forge kaggle run`` writes this script into the kernel it pushes, with the job's arguments
in ``KERNEL_ARGV``, since Kaggle starts a script kernel with none. Until the code dataset's
wheel is installed it needs only the standard library, and the job modules
(``forge.eval.runner``, ``forge.train.sft``) are imported only when their stage runs, so
the script can be tested on a laptop with fakes. The stages, in order (``STAGES``):

- restore: on a resume, the run's files so far, copied from the last attempt's kernel
  output, mounted under the input folder. It comes first, so that whatever fails after,
  this attempt's output holds the run too and the next resume carries on from it.
- env: the interpreter, the platform and the GPUs, for the log.
- code: the code dataset under the input folder, whose ``GIT_SHA`` must be the commit the
  job was pushed at, with the wheel and both GPU package lists. That commit goes to the
  job in ``FORGE_CODE_SHA``, which a manifest takes as its commit where there is no git
  checkout (``forge.ops.manifest.code_sha``).
- install: ``pip install --no-deps`` the wheel, then the GPU packages its job needs
  (``REQUIREMENTS_FILES``, ADR-0006): an eval job ``requirements-gpu.txt``, vLLM's stack,
  and a training job ``requirements-train-gpu.txt``, the trl stack FG-301 ran GRPO on,
  which has no vLLM. Every package the wheel needs must then be there, or the stage says
  which is not. Its result names the file it installed.
- data: the Spider dataset, found by its ``spider.sha256``, with ``database.zip`` unpacked
  if Kaggle left it packed, and every file checked before any job runs against the list
  the installed wheel pins (``forge.data.spider.CHECKSUMS``), not the dataset's own copy.
- secrets: ``HF_TOKEN`` and ``WANDB_API_KEY`` from Kaggle Secrets (or Colab's) when they
  are attached, without the line breaks and outer spaces huggingface_hub drops. No value
  is ever printed or written: every stage result in status.json, the error, the
  traceback, a failed run's ``error.txt`` and every log record, such as the run store's
  warning on a Hub error it tries again, go through ``_redact``. The job goes on without
  them.
- preflight: the job on ``PREFLIGHT_EXAMPLES`` examples through the same path, in a run
  of its own under ``preflight/``, so a broken job fails in minutes. The job then runs on
  the model the preflight loaded: a vLLM engine holds most of the GPU's memory while it
  lives, so a second one might not fit beside it, and loading once saves GPU time. A dev
  job's preflight answers val questions, so dev loads once, in the job. An SFT job's
  preflight trains ``PREFLIGHT_STEPS`` steps of its config, saving and all, and frees its
  model's GPU memory before the job loads the model again for a fresh adapter.
- job: the job itself, its files in ``<work>/runs/<run_id>/``, where a resume carries on
  from the files the restore stage brought back. With
  ``--timeout-s``, the push's timeout, an eval job stops between batches before Kaggle
  would end the session (``forge.ops.timeguard.eval_deadline``, counted from the kernel's
  start), as ``stopped_time_guard``, and its run is uploaded and resumed like any other.
  An SFT job's time guard (``TimeGuardCallback``) counts its config's
  ``ops.time_guard_minutes`` from the job's start, or only up to that deadline, less the
  share it keeps back, when that comes first. An eval job whose adapter is a training run's
  checkpoint, ``runs/<run_id>/checkpoints/<name>`` as on a laptop, fetches that folder
  from the Hub run store to the same path under the work folder first, which needs
  ``HF_TOKEN``. An SFT job builds its rows in the temp folder, and keeps its checkpoints
  in its run's folder for the upload: an SFT run can't be resumed yet, so a checkpoint in
  the store before the end would serve nothing.
- upload: that folder to the Hub run store (``forge.ops.hub``, which tries again on
  network errors), only when there is an ``HF_TOKEN``, and only when the store's copy
  holds no session this one lacks (``forge.ops.run_copies.put_run``).

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
the job began in this kernel, one past those the restore stage brought back, is stretched
to it, or, when none began, as when a resumed job fails before its run starts again, the
kernel's session is added. ``session`` in
status.json is its number. The upload stage stretches it up to the upload first, so the
run store's copy holds nearly all of it.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import logging
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
STAGES = ("restore", "env", "code", "install", "data", "secrets", "preflight", "job", "upload")
PREFLIGHT_EXAMPLES = 5
PREFLIGHT_STEPS = 2
SECRET_LABELS = ("HF_TOKEN", "WANDB_API_KEY")
# The names forge.ops.kaggle and forge.ops.manifest use; this script can't import them
# before the wheel is installed, so a unit test keeps each copy in step.
GIT_SHA_FILE = "GIT_SHA"
EVAL_REQUIREMENTS = "requirements-gpu.txt"
TRAIN_REQUIREMENTS = "requirements-train-gpu.txt"
# The GPU packages each job installs: vLLM generates for evaluation, and training runs on
# the trl stack, since vLLM 0.33.0 can't load a 4-bit policy (ADR-0006). A unit test keeps
# this in step with the jobs forge kaggle run pushes this script for.
REQUIREMENTS_FILES = {
    "eval": EVAL_REQUIREMENTS,
    "sft": TRAIN_REQUIREMENTS,
    "grpo": TRAIN_REQUIREMENTS,
}
CHECKSUM_FILE = "spider.sha256"
CODE_SHA_ENV = "FORGE_CODE_SHA"
MANIFEST_FILE = "manifest.json"
SESSIONS_FILE = "sessions.json"
ERROR_FILE = "error.txt"
DEV_USAGE_ENV = "FORGE_DEV_USAGE"
DEV_USAGE_FILE = "dev-usage.md"
HOSTS = ("kaggle", "colab")
PIP = (sys.executable, "-m", "pip", "install", "--no-input", "--progress-bar", "off")

# Runs one command, raising CalledProcessError when it fails; tests stand in for pip.
Runner = Callable[[Sequence[str]], None]
Clock = Callable[[], float]  # seconds that never go back, as forge.ops.timeguard counts them


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
    clock: Clock = time.monotonic
    started: float = field(init=False)  # on clock, when the kernel's time began
    sessions_before: int = 0  # the run's sessions the restore stage brought back

    def __post_init__(self) -> None:
        self.started = self.clock()

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
    parser.add_argument(
        "--timeout-s", type=int, help="the push's timeout, which the job's deadline counts to"
    )
    return parser.parse_args(list(argv))


def _json_object(text: str) -> dict[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError(f"must be a JSON object, got {text!r}")
    return value


def restore_stage(kernel: Kernel) -> dict[str, Any] | str:
    """Copy the run's files from its last attempt's output, so the job carries on."""
    if not kernel.args.resume:
        return "skipped: a new run"
    where = f"runs/{kernel.args.run_id}/{MANIFEST_FILE}"
    manifest = _one(Path(kernel.args.input_dir), where, "the last attempt's output")
    if kernel.run_dir.exists():
        raise JobError(f"{kernel.run_dir} exists already, so the last attempt can't go there")
    # status.json said how that attempt ended; this one writes its own.
    shutil.copytree(manifest.parent, kernel.run_dir, ignore=shutil.ignore_patterns(STATUS_FILE))
    files = sum(path.is_file() for path in kernel.run_dir.rglob("*"))
    sessions = kernel.run_dir / SESSIONS_FILE
    if sessions.is_file():
        kernel.sessions_before = len(json.loads(sessions.read_text(encoding="utf-8")))
    return {"from": str(manifest.parent), "files": files}


def env_stage(kernel: Kernel) -> dict[str, Any]:
    print(f"[forge] python {platform.python_version()} at {sys.executable}", flush=True)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        "gpus": _gpus(),
    }


def code_stage(kernel: Kernel) -> dict[str, Any]:
    """Find the wheel and the job's GPU package list, and check they are the pushed
    commit's."""
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
    # forge kaggle publish-code puts both lists in every version, so a version without one
    # wasn't made by it.
    missing = [
        name for name in sorted(set(REQUIREMENTS_FILES.values())) if not (folder / name).is_file()
    ]
    if missing:
        raise JobError(f"{folder} has no {' or '.join(missing)}")
    kernel.wheel = wheels[0]
    kernel.requirements = folder / REQUIREMENTS_FILES[kernel.args.job]
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
            + f". Pin each in kaggle/{kernel.requirements.name} at its uv.lock version"
        )
    return {"forge": importlib.metadata.version("forge"), "requirements": kernel.requirements.name}


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
    """Find Spider, unpack its databases if needed, and check every file the wheel pins."""
    from forge.data.spider import check_files, load_checksums

    folder = _one(Path(kernel.args.input_dir), CHECKSUM_FILE, "the Spider dataset").parent
    if not (folder / "database").is_dir():
        folder = _unpacked(folder, Path(kernel.args.temp_dir) / "forge-spider")
    # The wheel's list, which publish-data checked the upload against: the dataset's own
    # copy would only show that the dataset agrees with itself.
    expected = load_checksums()
    problems = check_files(folder, expected)
    if problems:
        shown = ", ".join(f"{rel} ({why})" for rel, why in sorted(problems.items())[:5])
        raise JobError(
            f"{len(problems)} Spider files don't match the {CHECKSUM_FILE} forge pins: {shown}"
        )
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
        # The token huggingface_hub sends, and so the text an error might quote.
        value = (value or "").replace("\r", "").replace("\n", "").strip()
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
    status = JOBS[kernel.args.job](kernel, False)
    kernel.status["status"] = status
    return {"status": status}


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
        _fetch_adapter(kernel, config.adapter)
        kernel.model = model_backend(config)
    backend, count_tokens = kernel.model or (None, None)
    deadline = None
    if not preflight and kernel.args.timeout_s is not None:
        from forge.ops.timeguard import eval_deadline

        timeout_min = kernel.args.timeout_s / 60
        deadline = eval_deadline(timeout_min, clock=kernel.clock, start=kernel.started)
    run = run_eval(
        config,
        runs_root=runs_root,
        progress=_progress,
        backend=backend,
        count_tokens=count_tokens,
        deadline=deadline,
    )
    # A run the deadline stopped gets its report from the attempt that finishes it.
    if not preflight and run.manifest.status == "finished":
        write_run_report(kernel.args.run_id, runs_root=runs_root)
    return run.manifest.status


def _fetch_adapter(kernel: Kernel, adapter: str | None) -> None:
    """Fetch the adapter of an eval job from the run store when it names a training run's
    checkpoint, ``runs/<run_id>/checkpoints/<name>``, that the work folder lacks."""
    parts = (adapter or "").split("/")
    if len(parts) != 4 or parts[0] != "runs" or parts[2] != "checkpoints":
        return
    target = kernel.work / str(adapter)
    if target.is_dir():
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise JobError(
            f"the adapter {adapter} is a checkpoint in the run store {kernel.args.run_store}, "
            "and without an HF_TOKEN secret the kernel can't read it; attach one"
        )
    from forge.ops.hub import HubRunStore

    HubRunStore(kernel.args.run_store, token=token).get_folder(
        parts[1], "/".join(parts[2:]), target
    )
    print(f"[forge] adapter {adapter} fetched from {kernel.args.run_store}", flush=True)


def sft_job(kernel: Kernel, preflight: bool) -> str:
    """``forge train sft`` with the job's config, the run's platform the kernel's host; a
    preflight trains ``PREFLIGHT_STEPS`` steps of it, in a run of its own. With the push's
    timeout, the job's time guard stops it in time for the upload (``_guard_minutes``)."""
    from forge.train.config import parse_train_config
    from forge.train.sft import run_sft

    config = parse_train_config(kernel.args.options["config"], where="the job's config")
    if config.sft is None:
        raise JobError(f"the job's config trains a {config.run.kind} run, not an sft one")
    run_id, runs_root = kernel.args.run_id, kernel.work / "runs"
    if preflight:
        steps = config.sft.model_copy(update={"max_steps": PREFLIGHT_STEPS})
        config = config.model_copy(update={"sft": steps})
        run_id, runs_root = f"{run_id}-preflight", kernel.work / "preflight"
    elif kernel.args.timeout_s is not None:
        minutes = min(config.ops.time_guard_minutes, _guard_minutes(kernel))
        # The ops section changes no weight, so the run's config and hash stay the same.
        config = config.model_copy(
            update={"ops": config.ops.model_copy(update={"time_guard_minutes": minutes})}
        )
    run = run_sft(
        config,
        run_id=run_id,
        runs_root=runs_root,
        sft_root=Path(kernel.args.temp_dir) / "sft",
        host=kernel.args.host,
        clock=kernel.clock,
    )
    if preflight:
        _free_gpu()
    return run.manifest.status


def _guard_minutes(kernel: Kernel) -> float:
    """The minutes a training job's time guard may count from now: up to an eval job's
    deadline for the push's timeout, counted from the kernel's start, less the share that
    deadline keeps back, which leaves the guard's last save and the upload their time."""
    from forge.ops.timeguard import eval_deadline

    deadline = eval_deadline(kernel.args.timeout_s / 60, clock=kernel.clock, start=kernel.started)
    minutes = (deadline.remaining_s() - deadline.margin_s) / 60
    if minutes <= 0:
        raise JobError(
            f"the push's {kernel.args.timeout_s} s timeout leaves the job no time to train "
            "after the kernel's setup and preflight; push it with a larger ops.est_gpu_hours"
        )
    return minutes


def _free_gpu() -> None:
    """Give back the GPU memory of a model nothing holds any more, cycles and all."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def upload_stage(kernel: Kernel) -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        return "skipped: no HF_TOKEN, so the kernel output is the only copy"
    from forge.ops.hub import HubRunStore
    from forge.ops.run_copies import put_run

    repo = kernel.args.run_store
    _record_session(kernel, _now())
    try:
        # Refused when the store's copy holds a session this one lacks, such as a laptop's
        # attempt since the output this kernel resumed from.
        put_run(HubRunStore(repo, token=token), kernel.run_dir)
    # The run is complete in the kernel output whatever the Hub says, so a failed upload
    # is recorded rather than turned into a failed job.
    except Exception as exc:
        return f"failed: {type(exc).__name__}: {exc}"
    return f"uploaded to {repo}"


JOBS: Mapping[str, Callable[[Kernel, bool], str]] = {"eval": eval_job, "sft": sft_job}
STAGE_FUNCTIONS: Mapping[str, Callable[[Kernel], Any]] = {
    "restore": restore_stage,
    "env": env_stage,
    "code": code_stage,
    "install": install_stage,
    "data": data_stage,
    "secrets": secrets_stage,
    "preflight": preflight_stage,
    "job": job_stage,
    "upload": upload_stage,
}


def main(
    argv: Sequence[str] | None = None, run: Runner | None = None, clock: Clock = time.monotonic
) -> int:
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
    kernel = Kernel(args, run or _run, status, clock=clock)
    os.environ["FORGE_RUNS_DIR"] = str(kernel.work / "runs")
    os.environ["FORGE_CACHE_DIR"] = str(Path(args.temp_dir) / "cache")
    os.environ[DEV_USAGE_ENV] = str(kernel.work / DEV_USAGE_FILE)
    # Outside any checkout, so a manifest's commit is FORGE_CODE_SHA's: the wheel's own.
    kernel.work.mkdir(parents=True, exist_ok=True)
    os.chdir(kernel.work)
    # Before any stage, so no log record a stage makes escapes it; the secrets stage adds
    # what it redacts.
    record_factory = _redact_logs(kernel.secrets)
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
        status["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        signal.signal(signal.SIGTERM, previous)
        logging.setLogRecordFactory(record_factory)
        _close_manifest(kernel.run_dir)
        # A failed run keeps its traceback beside its manifest, the job's or the preflight's.
        for error in kernel.work.glob(f"*/*/{ERROR_FILE}"):
            _redact_file(error, kernel.secrets)
        status["ended_at"] = _now()
        status["session"] = _record_session(kernel, status["ended_at"])
        # Any stage's result may quote an exception, whose text may hold a secret.
        _write_json(kernel.run_dir / STATUS_FILE, _redacted(status, kernel.secrets))
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


def _record_session(kernel: Kernel, ended_at: str) -> int | None:
    """Make the run's session in this kernel span the kernel's start to ``ended_at``, and
    return its number; None when the kernel holds no run.

    The session the job began here is the run's last, past those the restore brought back;
    counted rather than told by its start, since times here are whole seconds. A session
    still running then failed. When the job began none, the kernel's is added, with the
    run's status if the run ended in this kernel, and failed if not.
    """
    run_dir, started_at = kernel.run_dir, kernel.status["started_at"]
    manifest_path = run_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        return None
    path = run_dir / SESSIONS_FILE
    sessions = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    if len(sessions) > kernel.sessions_before:
        session = sessions[-1]
        if session["status"] == "running":
            session["status"] = "failed"
    else:
        run = json.loads(manifest_path.read_text(encoding="utf-8"))
        ended, start = run.get("ended_at"), datetime.fromisoformat(started_at)
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


def _redact_logs(secrets: Sequence[str]) -> Callable[..., logging.LogRecord]:
    """Have every log record made from now on carry no secret in ``secrets``, whichever
    logger makes it and whichever handler prints it, Python's last resort on stderr
    included; returns the record factory it replaces, for ``main`` to put back.

    The factory every logger makes its records with formats each one, redacts its text and
    keeps its traceback only as redacted text, so no handler formats the raw arguments.
    """
    make = logging.getLogRecordFactory()
    formatter = logging.Formatter()

    def redacted(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = make(*args, **kwargs)
        try:
            text = record.getMessage()
        # A record whose arguments don't fit its format, which a handler would report
        # with the arguments, any secret among them; the format alone is kept.
        except Exception as exc:
            text = f"{record.msg} (not formatted: {type(exc).__name__})"
        record.msg, record.args = _redact(text, secrets), None
        if record.exc_info:
            record.exc_text = _redact(formatter.formatException(record.exc_info), secrets)
            record.exc_info = None
        if record.stack_info:
            record.stack_info = _redact(record.stack_info, secrets)
        return record

    logging.setLogRecordFactory(redacted)
    return make


def _redact_file(path: Path, secrets: Sequence[str]) -> None:
    text = path.read_text(encoding="utf-8")
    if (clean := _redact(text, secrets)) != text:
        path.write_text(clean, encoding="utf-8")


def _redacted(value: Any, secrets: Sequence[str]) -> Any:
    """``value``, a JSON value, with ``_redact`` applied to every string in it."""
    if isinstance(value, str):
        return _redact(value, secrets)
    if isinstance(value, dict):
        return {key: _redacted(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_redacted(item, secrets) for item in value]
    return value


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
