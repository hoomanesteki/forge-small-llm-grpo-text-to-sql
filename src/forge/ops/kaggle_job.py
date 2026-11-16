"""A GPU job on Kaggle: what ``forge kaggle run`` checks before it pushes, the kernel it
pushes, and how the job is followed after.

Before a push, in the plan's order (FG-303):

1. ``head_commit``: HEAD passed ``make smoke`` (``.smoke-ok-<sha>``), on a clean tree.
2. ``code_version``: the code dataset's current version holds HEAD's wheel, by its
   ``GIT_SHA``, or HEAD is published as a new version. The kernel mounts that version by
   number, so an upload made while it is queued can't change the code it runs.
3. ``check_budget``: the config's ``ops.est_gpu_hours`` against the hours the ledger
   leaves this week, and against Kaggle's own quota when ``kaggle quota`` says.
   ``--allow-over-budget`` lets an over-budget job through, and an eval job records it
   in its manifest's config.
4. The command asks to go ahead.
5. ``write_kernel``: ``kaggle/entry.py`` with the job's arguments written in, and private
   GPU metadata with internet that mounts the code and data datasets and, on a resume,
   the last attempt's kernel, whose output holds the run so far.
6. ``KaggleClient.push_kernel`` with a T4 and a timeout of the estimate and half again, at
   most a session's 12 hours: plan section 5.3 stops a job 50% over its estimate. The
   entry script gets the timeout too, and its eval job stops at the time guard before
   it. Each push is recorded in ``runs/<run_id>/kernels.json``, which status, logs, wait
   and output read to find the run's kernel.

A dev eval job takes ``--reason``, as ``forge eval run --split dev`` does, and is refused
once the six dev looks are spent (ADR-0004). The push spends its look: ``log_dev_look``
writes the row in ``docs/dev-usage.md`` here, where the repo is, since the kernel's own
log stays in its output, as soon as Kaggle has accepted the push and before anything else
that could fail. A ``--resume`` of the same run is the same look, so it adds no row and
goes even once the looks are spent. A notebook can run any number of times, so a dev job
has none, and it can't fetch a kernel's output, so it resumes nothing.

The kernel's slug, ``forge-<run_id>`` with ``-<attempt>`` on a resume, must be one Kaggle
takes; ``plan_job`` checks it with the run id, before any dataset is published.

The ``spike`` job pushes FG-301's throughput script instead of the entry script. It runs
without Forge, so it mounts no dataset and has no preflight.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

from forge.core.ids import make_run_id
from forge.eval.config import (
    EvalConfig,
    Ops,
    check_config,
    check_run_id,
    read_config_sections,
    read_ops,
)
from forge.eval.run_files import MANIFEST_FILE
from forge.ops.clock import utc_today
from forge.ops.dev_usage import MAX_DEV_USES, count_dev_uses, log_dev_use, read_uses
from forge.ops.hub import KERNEL_DIR, KERNELS_FILE, replace_run
from forge.ops.kaggle import (
    CODE_SLUG,
    DATA_SLUG,
    GIT_SHA_FILE,
    KaggleClient,
    KaggleError,
    bake_argv,
    load_template,
    publish_code,
    render_kernel_metadata,
    valid_slug,
    write_kernel_dir,
)
from forge.ops.ledger import Summary
from forge.ops.manifest import DirtyTreeError, git_sha
from forge.ops.sessions import SESSIONS_FILE

JOBS = ("eval", "sft", "grpo", "spike")
# Jobs whose trainers later stories build, with the story that brings each.
LATER_JOBS: Mapping[str, str] = {"sft": "FG-403", "grpo": "FG-504"}
ENTRY = Path("kaggle") / "entry.py"  # relative to the repo root, like the spike
SPIKE = Path("kaggle") / "spike" / "throughput.py"
# The FG-301 GPU kernel's name, the one scripts/kaggle_spike.py bundles.
SPIKE_SLUG = "forge-throughput-spike"
SPIKE_TITLE = "Forge throughput spike"
KERNEL_BUILD = Path("build") / "kaggle-kernel"
STATUS_FILE = "status.json"  # what kaggle/entry.py writes in the run's folder
SESSION_HOURS = 12.0
OVERRUN = 1.5
# kagglesdk's DatabundleVersionStatus names, lowercased as the CLI prints them: every
# other one is a step on the way to ready.
DATASET_BROKEN = frozenset({"failed", "deleted"})
DATASET_POLL_S = 15.0
# The paths a Colab runtime gives the job in place of /kaggle/...
COLAB_DIRS = ["--input-dir", "/content/input", "--work-dir", "/content/working"]
COLAB_DIRS += ["--temp-dir", "/content/temp"]
# The line that ends the entry script; a notebook calls main() in a cell of its own.
MAIN_GUARD = '\nif __name__ == "__main__":\n    raise SystemExit(main())\n'
_SMOKE_MARKER = ".smoke-ok-{sha}"


class JobRefused(RuntimeError):
    """A check before a push says no. The message says why, and what to do instead."""


@dataclass(frozen=True)
class JobPlan:
    """One push of one job, before any dataset or kernel is touched."""

    job: str
    run_id: str
    code_sha: str
    est_gpu_hours: float
    # The entry job's options (an eval job's are EvalConfig's), or the spike's {"argv"}.
    options: dict[str, Any]
    kernel: str  # owner/slug
    title: str
    preflight: bool
    resume: bool = False
    kernel_sources: tuple[str, ...] = ()

    @property
    def timeout_s(self) -> int:
        return math.ceil(min(SESSION_HOURS, self.est_gpu_hours * OVERRUN) * 3600)

    @property
    def mounts_datasets(self) -> bool:
        return self.job != "spike"

    @property
    def allow_over_budget(self) -> bool:
        return bool(self.options.get("allow_over_budget"))

    @property
    def reads_dev(self) -> bool:
        return self.options.get("split") == "dev"


@dataclass(frozen=True)
class Budget:
    est_gpu_hours: float
    ledger_left: float  # this week's Kaggle hours the ledger leaves
    weekly_quota: float
    kaggle_left: float | None  # kaggle quota's GPU hours left, when it said
    kaggle_note: str = ""  # why kaggle_left is unknown

    @property
    def over(self) -> bool:
        known = [left for left in (self.ledger_left, self.kaggle_left) if left is not None]
        return any(self.est_gpu_hours > left for left in known)

    def lines(self) -> list[str]:
        kaggle = (
            f"{self.kaggle_left:.2f} h left" if self.kaggle_left is not None else self.kaggle_note
        )
        return [
            f"estimate {self.est_gpu_hours:.2f} GPU h",
            f"ledger: {self.ledger_left:.2f} of {self.weekly_quota:.0f} h left this week",
            f"kaggle quota: {kaggle}",
        ]


def head_commit(root: Path) -> str:
    """HEAD's commit, once it is clean and has passed ``make smoke``."""
    try:
        sha = git_sha(root)
    except DirtyTreeError as exc:
        raise DirtyTreeError(
            "the working tree has uncommitted changes; a GPU job runs a commit, so commit "
            "them and run make smoke on it first"
        ) from exc
    if not (root / _SMOKE_MARKER.format(sha=sha)).is_file():
        raise JobRefused(
            f"make smoke has not passed on {sha[:7]}: run it on this commit before a GPU job "
            "(CONTRIBUTING rule 8)"
        )
    return sha


def plan_job(
    job: str,
    config: Path,
    *,
    owner: str,
    sha: str,
    runs_root: Path,
    run_id: str | None = None,
    resume: bool = False,
    preflight: bool | None = None,
    allow_over_budget: bool = False,
    reason: str | None = None,
    notebook: bool = False,
    today: date | None = None,
) -> JobPlan:
    """What pushing ``job`` with ``config`` at ``sha`` would push, checked, or what its
    notebook would run, with ``notebook``.

    ``preflight`` is on by default for entry jobs. ``reason`` goes with a dev eval job,
    and only with one. Raises ``JobRefused`` for a job that can't go yet, a run that
    clashes with an earlier push, a run id that makes no kernel slug, a resume or a dev
    job as a notebook, a dev job once the dev looks are spent, ``ValueError`` for a bad
    config, and ``NotAvailable`` for options a later story builds.
    """
    if job in LATER_JOBS:
        raise JobRefused(f"--job {job} comes with {LATER_JOBS[job]}")
    if job not in JOBS:
        raise ValueError(f"job must be one of {', '.join(JOBS)}, got {job!r}")
    if resume and notebook:
        raise JobRefused(
            "a notebook can't fetch the last attempt's kernel output, so it can't resume a "
            "run; push the resume with forge kaggle run --resume, or give the notebook a new "
            "--run-id"
        )
    day = today or utc_today()
    options: dict[str, Any]
    if job == "spike":
        if preflight:
            raise JobRefused("the spike runs without Forge, so it has no preflight")
        if reason is not None:
            raise JobRefused("the spike reads no split, so it takes no --reason")
        argv, ops = read_spike_config(config)
        options = {"argv": argv, **({"allow_over_budget": True} if allow_over_budget else {})}
        default_id = f"{day:%Y-%m-%d}-throughput-spike-{sha[:7]}"
    else:
        file_options, ops = read_config_sections(config)
        options = {
            **file_options,
            **({"reason": reason} if reason is not None else {}),
            **({"allow_over_budget": True} if allow_over_budget else {}),
        }
        # Checked here as the kernel will check it, so a bad option costs no GPU time.
        eval_config = EvalConfig(**options, host="kaggle")
        check_config(eval_config)
        if eval_config.split == "dev" and notebook:
            raise JobRefused(
                "a notebook can run any number of times, so it can't log its dev look; "
                "push a dev job with forge kaggle run, which logs the one look it spends"
            )
        default_id = make_run_id("eval", eval_config.model, sha, on=day)
    est = ops.est_gpu_hours
    if est is None:
        raise JobRefused(f"{config} sets no ops.est_gpu_hours, which the budget check needs")
    if est > SESSION_HOURS:
        raise JobRefused(f"{est:g} GPU h won't fit a {SESSION_HOURS:g} h session; split the job")
    run_id = run_id or default_id
    check_run_id(run_id)
    record = read_record(runs_root, run_id)
    kernel_sources: tuple[str, ...] = ()
    if job == "spike":
        slug, title = SPIKE_SLUG, SPIKE_TITLE
    else:
        slug, title = f"forge-{run_id}", f"Forge {run_id}"
    if record is not None and not resume:
        carry_on = "" if job == "spike" else "--resume to carry it on, or "
        raise JobRefused(
            f"run {run_id} was pushed already as {record['kernels'][-1]['kernel']}; pass "
            f"{carry_on}--run-id to name a new run"
        )
    if resume:
        if job == "spike":
            raise JobRefused("a spike has no run to resume; it starts from nothing each time")
        if record is None:
            raise JobRefused(f"run {run_id} has no {_record_path(runs_root, run_id)} to resume")
        if record["code_sha"] != sha:
            raise JobRefused(
                f"run {run_id} was made at {record['code_sha'][:7]}, not {sha[:7]}; resume it "
                "from that commit, or pass --run-id to start a new run"
            )
        if record["options"] != options:
            raise JobRefused(
                f"run {run_id} was pushed with other options (its config's, --reason or "
                "--allow-over-budget); resume it with the same ones"
            )
        last = record["kernels"][-1]
        kernel_sources = (f"{last['kernel']}/{last['version']}",)
        attempt = len(record["kernels"]) + 1
        slug, title = f"{slug}-{attempt}", f"{title} {attempt}"
    if not valid_slug(slug):
        raise JobRefused(
            f"run id {run_id} makes the kernel slug {slug}, which Kaggle refuses: lowercase "
            "letters and digits with single hyphens between them; pass another --run-id"
        )
    reads_dev = options.get("split") == "dev"
    same_look = resume and _look_logged(run_id)
    if reads_dev and not same_look and count_dev_uses() >= MAX_DEV_USES:
        raise JobRefused(
            f"the dev split has had its {MAX_DEV_USES} looks (docs/dev-usage.md, "
            "ADR-0004), so no dev job can go"
        )
    return JobPlan(
        job=job,
        run_id=run_id,
        code_sha=sha,
        est_gpu_hours=est,
        options=options,
        kernel=f"{owner}/{slug}",
        title=title,
        preflight=preflight is not False and job != "spike",
        resume=resume,
        kernel_sources=kernel_sources,
    )


def read_spike_config(path: Path) -> tuple[list[str], Ops]:
    """The spike script's arguments and the ops section of the spike config at ``path``."""
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(loaded, dict) or set(loaded) - {"argv", "ops"}:
        raise ValueError(f"{path} must be a mapping with argv and ops, such as 'argv: [run]'")
    argv = loaded.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
        raise ValueError(f"{path}: argv must be a list of the script's arguments, got {argv!r}")
    return argv, read_ops(path, loaded.get("ops", {}))


def code_version(
    root: Path,
    owner: str,
    client: KaggleClient,
    sha: str,
    *,
    say: Callable[[str], None] = print,
    wait_s: float = 900.0,
) -> str:
    """``<owner>/forge-code/<version>`` of a processed version holding ``sha``'s wheel.

    The current version is used when its ``GIT_SHA`` is ``sha``. Otherwise ``sha`` is
    published, and its version is waited for: ``datasets version`` returns before Kaggle
    has processed the upload.
    """
    ref = f"{owner}/{CODE_SLUG}"
    current = None
    if client.dataset_exists(ref):
        current = _ready_version(client, ref, None, wait_s)
        if _dataset_sha(client, ref, current) == sha:
            return f"{ref}/{current}"
    say(f"{ref} doesn't hold {sha[:7]} yet: publishing it")
    publish_code(root, owner, client)
    version = _ready_version(client, ref, current, wait_s)
    found = _dataset_sha(client, ref, version)
    if found != sha:
        raise KaggleError(f"{ref} version {version} holds {found}, not the published {sha}")
    return f"{ref}/{version}"


def data_source(
    owner: str,
    client: KaggleClient,
    *,
    wait_s: float = 900.0,
) -> str:
    """``<owner>/forge-spider``, once it exists and is processed."""
    ref = f"{owner}/{DATA_SLUG}"
    if not client.dataset_exists(ref):
        raise JobRefused(f"{ref} doesn't exist; run forge kaggle publish-data once first")
    _ready_version(client, ref, None, wait_s)
    return ref


def _ready_version(
    client: KaggleClient,
    ref: str,
    after: int | None,
    wait_s: float,
) -> int:
    """The dataset's version number once it is ready, and newer than ``after`` if given."""
    waited = 0.0
    while True:
        state = client.dataset_version(ref)
        if state.status == "ready" and (after is None or state.number > after):
            return state.number
        if state.status in DATASET_BROKEN:
            raise KaggleError(f"{ref} is {state.status}, so a kernel can't mount it")
        if waited >= wait_s:
            raise KaggleError(f"{ref} was not ready after {wait_s:g} s; try again later")
        time.sleep(DATASET_POLL_S)
        waited += DATASET_POLL_S


def _dataset_sha(client: KaggleClient, ref: str, version: int) -> str:
    with tempfile.TemporaryDirectory(prefix="forge-code-") as folder:
        path = client.dataset_file(f"{ref}/{version}", GIT_SHA_FILE, Path(folder))
        return path.read_text(encoding="utf-8").strip()


def check_budget(est_gpu_hours: float, ledger: Summary, client: KaggleClient) -> Budget:
    """The job's estimate against the ledger's week and Kaggle's quota, when it says."""
    try:
        left: float | None = client.gpu_hours_left()
        note = ""
    except KaggleError as exc:
        left, note = None, f"unavailable ({exc})"
    return Budget(est_gpu_hours, ledger.remaining_week, ledger.weekly_quota, left, note)


def log_dev_look(plan: JobPlan) -> int | None:
    """Log the dev look a pushed dev job spends, and return how many are spent; ``None``
    for a push that spends none: a job that reads no dev, or a resume of a run whose look
    is logged already, which is the same look (one row a run id)."""
    if not plan.reads_dev or (plan.resume and _look_logged(plan.run_id)):
        return None
    return log_dev_use(plan.run_id, str(plan.options["reason"]))


def _look_logged(run_id: str) -> bool:
    return any(logged == run_id for _, logged, _ in read_uses())


def entry_argv(plan: JobPlan, run_store: str) -> list[str]:
    """The arguments kaggle/entry.py runs ``plan`` with, the push's timeout among them, so
    the job stops at its time guard before Kaggle ends the session."""
    argv = ["--job", plan.job, "--run-id", plan.run_id, "--code-sha", plan.code_sha]
    argv += ["--options", json.dumps(plan.options, sort_keys=True), "--run-store", run_store]
    argv += ["--timeout-s", str(plan.timeout_s)]
    argv.append("--preflight" if plan.preflight else "--no-preflight")
    return [*argv, "--resume"] if plan.resume else argv


def write_kernel(
    root: Path, plan: JobPlan, *, code: str | None, data: str | None, run_store: str
) -> Path:
    """Write the kernel folder for ``plan`` under build/kaggle-kernel, ready to push.

    ``code`` and ``data`` are the dataset sources an entry job mounts.
    """
    owner, slug = plan.kernel.split("/")
    if plan.mounts_datasets:
        if code is None or data is None:
            raise ValueError(f"a {plan.job} job mounts the code and data datasets")
        script, argv, datasets = ENTRY, entry_argv(plan, run_store), [code, data]
    else:
        script, argv, datasets = SPIKE, list(plan.options["argv"]), []
    values = {"owner": owner, "slug": slug, "title": plan.title, "code_file": script.name}
    meta = render_kernel_metadata(
        load_template(root), values, dataset_sources=datasets, kernel_sources=plan.kernel_sources
    )
    source = bake_argv((root / script).read_text(encoding="utf-8"), argv)
    folder = root / KERNEL_BUILD / plan.run_id
    shutil.rmtree(folder, ignore_errors=True)
    return write_kernel_dir(folder, meta, source)


def notebook(
    root: Path, plan: JobPlan, *, code: str | None, data: str | None, run_store: str
) -> dict[str, Any]:
    """The job as an nbformat 4 notebook for Colab: fetch the two datasets, then run the
    entry script's stages, the same ones a kernel runs."""
    if code is None or data is None:
        raise JobRefused("the spike mounts no dataset and runs on Kaggle; it has no notebook")
    source = (root / ENTRY).read_text(encoding="utf-8")
    if not source.endswith(MAIN_GUARD):
        raise ValueError(f"{ENTRY} must end with its main guard")
    argv = [*entry_argv(plan, run_store), "--host", "colab", *COLAB_DIRS]
    script = bake_argv(source.removesuffix(MAIN_GUARD), argv)
    stages = re.search(r"^STAGES = \((.*)\)$", source, re.MULTILINE)
    if stages is None:
        raise ValueError(f"{ENTRY} must name its STAGES")
    names = [name.strip(' "') for name in stages.group(1).split(",")]
    intro = [
        f"# Forge {plan.job} job {plan.run_id}",
        "",
        f"The Colab fallback of the Kaggle kernel `forge kaggle run` would push: commit "
        f"`{plan.code_sha}`, code dataset `{code}`, data `{data}`, about "
        f"{plan.est_gpu_hours:g} GPU hours.",
        "",
        "1. Runtime, Change runtime type: a T4 GPU.",
        "2. Under Secrets, add `KAGGLE_API_TOKEN` (to fetch the private datasets), and "
        "`HF_TOKEN` and `WANDB_API_KEY` if you have them. No value is ever printed.",
        "3. Run every cell. The job's stages, in order: " + ", ".join(names) + ".",
        "4. Afterwards add the hours to the ledger: `forge ledger add --platform colab ...`.",
    ]
    fetch = [
        "import os, subprocess, sys",
        "try:",
        "    from google.colab import userdata",
        '    os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")',
        "except Exception as exc:  # outside Colab, or the secret isn't there",
        '    print(f"no KAGGLE_API_TOKEN secret ({type(exc).__name__}); using kaggle\'s own")',
        'subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle==2.2.5"], '
        "check=True)",
        f"for ref, folder in [({code!r}, 'forge-code'), ({data!r}, 'forge-spider')]:",
        '    target = f"/content/input/{folder}"',
        '    command = ["kaggle", "datasets", "download", ref, "-p", target, "--unzip", "-q"]',
        "    subprocess.run(command, check=True)",
    ]
    cells = [
        _cell("markdown", intro, 0),
        _cell("code", fetch, 1),
        _cell("code", script.splitlines(), 2),
        _cell("code", ["main()"], 3),
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": []},
        },
        "cells": cells,
    }


def _cell(kind: str, lines: Sequence[str], index: int) -> dict[str, Any]:
    text = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    cell: dict[str, Any] = {"cell_type": kind, "id": f"forge-{index}", "metadata": {}}
    cell["source"] = text
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


@dataclass(frozen=True)
class KernelRecord:
    kernel: str  # owner/slug/version
    job: str


def record_push(
    runs_root: Path,
    plan: JobPlan,
    *,
    kernel: str,
    version: int,
    code: str | None,
    budget: Budget,
    now: datetime | None = None,
) -> Path:
    """Add the push to ``runs/<run_id>/kernels.json``, starting the file on a first push."""
    path = _record_path(runs_root, plan.run_id)
    record = read_record(runs_root, plan.run_id) or {
        "run_id": plan.run_id,
        "job": plan.job,
        "code_sha": plan.code_sha,
        "options": plan.options,
        "kernels": [],
    }
    record["kernels"].append(
        {
            "kernel": kernel,
            "version": version,
            "pushed_at": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
            "code": code,
            "est_gpu_hours": plan.est_gpu_hours,
            "timeout_s": plan.timeout_s,
            "allow_over_budget": plan.allow_over_budget,
            "over_budget": budget.over,
            "kaggle_gpu_hours_left": budget.kaggle_left,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def read_record(runs_root: Path, run_id: str) -> dict[str, Any] | None:
    path = _record_path(runs_root, run_id)
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or not record.get("kernels"):
        raise ValueError(f"{path} lists no kernel")
    return record


def last_kernel(runs_root: Path, run_id: str) -> KernelRecord:
    """The kernel of the run's latest push, ``owner/slug/version``."""
    record = read_record(runs_root, run_id)
    if record is None:
        raise JobRefused(
            f"no kernel was pushed for run {run_id} from here: "
            f"{_record_path(runs_root, run_id)} is missing"
        )
    last = record["kernels"][-1]
    return KernelRecord(f"{last['kernel']}/{last['version']}", str(record["job"]))


def wait_for(
    client: KaggleClient,
    kernel: str,
    *,
    poll_s: float = 60.0,
    timeout_s: float = (SESSION_HOURS + 1) * 3600,
    say: Callable[[str], None] = print,
) -> str:
    """Poll ``kernel`` until it ends: COMPLETE, ERROR or CANCEL_*. Returns that state."""
    waited, last = 0.0, None
    while True:
        status = client.kernel_status(kernel)
        if status.state != last:
            say(f"{kernel}: {status.state}" + (f" ({status.failure})" if status.failure else ""))
            last = status.state
        if status.done:
            return status.state
        if waited >= timeout_s:
            raise KaggleError(f"{kernel} was still {status.state} after {timeout_s:g} s")
        time.sleep(poll_s)
        waited += poll_s


def job_record(client: KaggleClient, kernel: str, run_id: str, folder: Path) -> Path:
    """Download what ``forge kaggle wait`` reads from the kernel output into ``folder``: the
    ``status.json`` the entry script wrote for ``run_id``, and the run's manifest and
    sessions, which its ledger rows come from. Returns the run's folder there."""
    names = "|".join(re.escape(name) for name in (STATUS_FILE, MANIFEST_FILE, SESSIONS_FILE))
    client.kernel_output(kernel, folder, pattern=f"runs/{re.escape(run_id)}/({names})$")
    found = sorted(folder.glob(f"**/runs/{run_id}/{STATUS_FILE}"))
    if not found:
        raise KaggleError(f"{kernel}'s output has no runs/{run_id}/{STATUS_FILE}")
    return found[0].parent


def read_status(run_dir: Path) -> dict[str, Any]:
    """The job's ``status.json`` in ``run_dir``."""
    data: dict[str, Any] = json.loads((run_dir / STATUS_FILE).read_text(encoding="utf-8"))
    return data


def fetch_output(client: KaggleClient, kernel: str, run_id: str, runs_root: Path) -> Path:
    """Download the kernel's output: the run's files into ``runs/<run_id>/``, and the rest
    (its log, the preflight run, a spike's report) into ``runs/<run_id>/kernel/``.

    The run's folder then holds the kernel's copy of the run and no file an earlier copy
    left (``forge.ops.hub.replace_run``); output that holds no run leaves it as it was.
    """
    run_dir = runs_root / run_id
    runs_root.mkdir(parents=True, exist_ok=True)
    # Staged beside the run, so a download that fails changes nothing in it.
    with tempfile.TemporaryDirectory(prefix=".output-", dir=runs_root) as folder:
        staging = Path(folder)
        client.kernel_output(kernel, staging)
        produced = staging / "runs" / run_id
        if produced.is_dir():
            replace_run(produced, run_dir, move=True)
            shutil.rmtree(staging / "runs")
        rest = run_dir / KERNEL_DIR
        if rest.exists():
            shutil.rmtree(rest)
        rest.mkdir(parents=True)
        for item in staging.iterdir():
            shutil.move(item, rest / item.name)
    return run_dir


def _record_path(runs_root: Path, run_id: str) -> Path:
    return runs_root / run_id / KERNELS_FILE
