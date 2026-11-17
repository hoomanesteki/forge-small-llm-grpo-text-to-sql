"""``forge train grpo``: GRPO on the train split, from a training config to a LoRA adapter
and its run manifest, across as many sessions as it takes (FG-504).

One run, in order (the starting values are ADR-0011's):

1. The backend and the platform, as ``forge train sft`` finds them (``forge.train.sft``):
   CUDA on the laptop has no platform to name (ADR-0016), so a GPU run is a kernel job.
2. The model and its tokenizer. With ``grpo.init_adapter`` the LoRA is that adapter,
   trainable (``TrainBackend.load``); without it, a fresh one. A training run's
   checkpoint, such as ``runs/<run_id>/checkpoints/best``, must sit in its run's folder
   beside the run's manifest, of the same model, and hold the files the run saved there
   by the digest the manifest records, when it records one (``start_adapter``).
3. The rows: the train questions ``forge.train.sft_data`` keeps, from ``load_train()``
   alone, each a ``prompt`` with its ``example_id``, ``db_id``, ``gold_sql`` and
   ``hardness``, the columns the rewards read (``grpo_rows``). TRL 1.18 has no prompt
   limit, so a prompt still over ``grpo.max_prompt_length`` tokens after the overflow
   policy is left out here, as SFT leaves it out.
4. The manifest. Its config is what can change the adapter: the run, data, LoRA, GRPO and
   rewards sections, the rows' hash and counts (``grpo_data``), the starting adapter's
   digest, the backend, device and dtype (``trainer``), and what TRL takes that the config
   leaves to it, with TRL's version (``trl``: beta and the loss among
   ``LIBRARY_SETTINGS``), all under the config hash. ``adapter_ref`` is the starting
   adapter and ``parent_run`` its run. The ops section changes no weight, so it stays
   out, as SFT's does, and a later session may change it.
5. One ``SandboxPool`` for the run, which the rewards (``forge.train.rewards``, on the
   train split's gold results) and the in-training eval share. With ``grpo.eval_every``
   the eval (``forge.eval.callbacks``) comes last among the callbacks, after those every
   job has (``forge.train.callbacks``), and its lines go to the same ``metrics.jsonl``.
6. Training, then the adapter in ``checkpoints/final`` and the manifest ``finished``; or
   the time guard's checkpoint and ``stopped_time_guard``; or ``failed``, with the
   traceback in ``error.txt``. Its outputs hold the digest of each checkpoint kept.

``resume`` carries on the run ``run_id`` names from its latest stored checkpoint, which
holds the optimizer, scheduler, RNG state and step count beside the adapter, so the steps
go on from where it stopped (``TrainBackend.grpo(resume_from=...)``). The run must be a
GRPO run that hasn't finished, made at the same commit, on the same data and to the same
config hash. Each attempt is a session in the run's ``sessions.json``
(``forge.ops.sessions``), so the manifest shows each boundary the run crossed and its GPU
hours add up its sessions.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

import yaml
from datasets import Dataset

from forge.core.hashing import canonical_json, sha256_hex
from forge.core.ids import make_run_id
from forge.core.types import Hardness, RunManifest
from forge.data.prompts import ChatTokenizer, chat_token_counter
from forge.data.spider import resolve_data_dir
from forge.data.splits import load_train
from forge.eval.config import Host, check_run_id
from forge.eval.gold_cache import split_gold
from forge.eval.run_files import RunConflict, write_atomic
from forge.ops.adapters import adapter_folder, adapter_sha256, checkpoint_ref, digest_output
from forge.ops.hub import FINAL, METRICS_FILE, LocalRunStore, RunNotFound
from forge.ops.manifest import EndStatus, accelerator, build_manifest, close, code_sha
from forge.ops.paths import runs_dir
from forge.ops.sessions import SESSIONS_FILE, begin, end
from forge.ops.timeguard import Clock
from forge.train.backends import TrainBackend, grpo_library_values, select_backend
from forge.train.config import TrainConfig
from forge.train.rewards import Rewards, reward_pool
from forge.train.sft import CONFIG_FILE, ERROR_FILE, RunExists, platform_for, run_outputs
from forge.train.sft_data import build_sft_dataset

log = logging.getLogger(__name__)


class GrpoRow(TypedDict):
    prompt: list[dict[str, str]]  # system, then user
    example_id: str
    db_id: str
    gold_sql: str
    hardness: Hardness


@dataclass(frozen=True)
class StartAdapter:
    """The adapter a run starts from, as its manifest records it."""

    ref: str  # as the config names it
    folder: Path  # its files
    sha256: str  # their digest (forge.ops.adapters.adapter_sha256)
    parent_run: str | None  # the training run it is a checkpoint of, if any


@dataclass(frozen=True)
class GrpoRun:
    """A GRPO session as it ended: the run's manifest and where its files are."""

    manifest: RunManifest
    root: Path  # the run's folder
    rows: int  # the rows it trains on
    adapter: Path | None  # checkpoints/final; None when the time guard stopped the run
    stopped_at: int | None  # the step the time guard stopped after
    resumed_from: str | None  # the checkpoint this session went on from


def start_adapter(ref: str, model_ref: str) -> StartAdapter:
    """The adapter ``ref`` names, checked as the module docstring's step 2 says.

    Raises ``FileNotFoundError`` for a checkpoint this machine lacks or one whose run has
    no manifest beside it, and ``ValueError`` for one of another model or of other files
    than its run saved there.
    """
    folder = adapter_folder(ref)
    digest = adapter_sha256(folder)
    found = checkpoint_ref(ref)
    if found is None:
        return StartAdapter(ref, folder, digest, None)
    try:
        trained = LocalRunStore(Path(found.folder or ".")).get_manifest(found.run_id)
    except RunNotFound as exc:
        raise RunNotFound(
            f"{ref} is {found.rel} of run {found.run_id}, which needs the run's manifest "
            f"beside it: {exc}; forge runs pull {found.run_id}"
        ) from exc
    if trained.model_ref != model_ref:
        raise ValueError(
            f"{ref} is {found.rel} of run {found.run_id}, which trained {trained.model_ref}, "
            f"and this run trains {model_ref}"
        )
    saved = trained.outputs.get(digest_output(found.rel))
    if saved is None:
        # Runs made before FG-492 recorded no digests, the local SFT pilot's among them.
        log.warning(
            "run %s records no digest of %s; GRPO starts from the files there, sha256 %s",
            found.run_id,
            found.rel,
            digest,
        )
    elif saved != digest:
        raise ValueError(
            f"{folder} holds other files than run {found.run_id} saved as {found.rel} "
            f"(sha256 {digest}, not {saved}); forge runs pull {found.run_id} again"
        )
    return StartAdapter(ref, folder, digest, found.run_id)


def grpo_rows(
    config: TrainConfig, tokenizer: ChatTokenizer, *, data_dir: Path
) -> tuple[Dataset, dict[str, Any]]:
    """The rows a run trains on, and what its manifest records of them, as the module
    docstring's step 3 says."""
    if config.grpo is None:
        raise ValueError(f"run.kind is {config.run.kind}, and GRPO rows are for grpo runs")
    built = build_sft_dataset(
        config.data.prompt_version,
        config.run.model_ref,
        chat_token_counter(tokenizer),
        max_tokens=config.grpo.max_prompt_length,
        data_dir=data_dir,
    )
    examples = {example.id: example for example in load_train(data_dir)}
    rows: list[GrpoRow] = []
    for row in built.rows:
        example = examples[row["example_id"]]
        rows.append(
            GrpoRow(
                prompt=row["prompt"],
                example_id=example.id,
                db_id=example.db_id,
                gold_sql=example.gold_sql,
                hardness=example.hardness,
            )
        )
    digest = sha256_hex(canonical_json([config.data.prompt_version, config.run.model_ref, rows]))
    meta = {"hash": digest, **built.meta.counts.model_dump()}
    return Dataset.from_list(rows), meta


def manifest_config(
    config: TrainConfig,
    data: Mapping[str, Any],
    start: StartAdapter | None,
    backend: TrainBackend,
) -> dict[str, Any]:
    """What a GRPO run's manifest records and hashes, as the module docstring lists it."""
    if config.grpo is None:
        raise ValueError(f"run.kind is {config.run.kind}, and this is a grpo run's manifest")
    sections: dict[str, Any] = config.model_dump(
        mode="json", include={"run", "data", "lora", "grpo", "rewards"}
    )
    return {
        **sections,
        "grpo_data": dict(data),
        "init_adapter_sha256": None if start is None else start.sha256,
        "trainer": {
            "backend": backend.name,
            "device": backend.device,
            "dtype": backend.precision.dtype,
        },
        "trl": grpo_library_values(config.grpo),
    }


def run_grpo(
    config: TrainConfig,
    *,
    run_id: str | None = None,
    resume: bool = False,
    allow_dirty: bool = False,
    data_dir: Path | None = None,
    runs_root: Path | None = None,
    cache_root: Path | None = None,
    backend: TrainBackend | None = None,
    host: Host | None = None,
    env: Mapping[str, str] | None = None,
    clock: Clock = time.monotonic,
    workers: int | None = None,
) -> GrpoRun:
    """Train, or with ``resume`` carry on, the GRPO run ``config`` describes, as the
    module docstring says.

    ``backend`` replaces the one the config names, ``host`` is the kernel's machine a job
    runs on, ``env`` where the callbacks look for a W&B key, ``clock`` the time guard's,
    ``cache_root`` where the gold results are kept (``forge.ops.paths.cache_dir()``) and
    ``workers`` the size of the SQL pool (``reward_workers()``). Raises ``DirtyTreeError``
    on uncommitted changes unless ``allow_dirty``, ``RunExists`` for a new run's id already
    used, ``RunNotFound`` for a resume of no run, ``RunConflict`` for one this config can't
    carry on, and ``ValueError`` for a config, a device or an adapter no run can take.
    """
    grpo = config.grpo
    if grpo is None:
        raise ValueError(f"run.kind is {config.run.kind}, and forge train grpo trains grpo runs")
    if resume and run_id is None:
        raise ValueError("name the run to resume: its run id")
    root = resolve_data_dir(data_dir).resolve()
    runs_root = runs_dir() if runs_root is None else runs_root
    # Asked once, before a model loads, as forge train sft does.
    sha = code_sha(allow_dirty=allow_dirty)
    run_id = run_id or make_run_id("grpo", config.run.model_ref, sha)
    check_run_id(run_id)
    run_dir = runs_root / run_id
    store = LocalRunStore(runs_root)
    previous: RunManifest | None = None
    if resume:
        # As much as can be checked before a model loads; the config hash needs the rows.
        previous = store.get_manifest(run_id)
        _check_resumable(previous, git_sha=sha)
    elif run_dir.exists():
        raise RunExists(f"run {run_id} is in {run_dir} already; pass --resume or a new --run-id")
    start = (
        None
        if grpo.init_adapter is None
        else start_adapter(grpo.init_adapter, config.run.model_ref)
    )
    if backend is None:
        backend = select_backend(
            config.run.backend,
            device=config.run.device,
            max_seq_length=config.max_seq_length,
            gpu_memory_utilization=grpo.gpu_memory_utilization,
        )
    platform = host if host is not None else platform_for(backend.device)
    # Imported once the backend is made: Unsloth patches transformers as it is imported.
    from forge.eval.callbacks import val_eval_callback
    from forge.train.callbacks import (
        CheckpointCallback,
        MetricsCallback,
        TimeGuardCallback,
        training_callbacks,
    )

    # Made now, so the time guard counts the model's loading too.
    callbacks = training_callbacks(
        config.ops, store, run_id, metrics_path=run_dir / METRICS_FILE, env=env, clock=clock
    )
    guard = next(c for c in callbacks if isinstance(c, TimeGuardCallback))
    checkpoints = next(c for c in callbacks if isinstance(c, CheckpointCallback))
    metrics = next(c for c in callbacks if isinstance(c, MetricsCallback))
    model, tokenizer = backend.load(
        config.run.model_ref,
        config.lora,
        load_in_4bit=config.run.load_in_4bit,
        seed=config.run.seed,
        adapter=None if start is None else start.folder,
    )
    rows, data = grpo_rows(config, tokenizer, data_dir=root)
    gold = split_gold("train", data_dir=root, cache_root=cache_root).results
    fresh = build_manifest(
        "grpo",
        manifest_config(config, data, start, backend),
        model_ref=config.run.model_ref,
        seed=config.run.seed,
        run_id=run_id,
        platform=platform,
        adapter_ref=None if start is None else start.ref,
        parent_run=None if start is None else start.parent_run,
        gpu=accelerator(platform),
        data_dir=root,
        git_sha=sha,
    )
    resumed_from: str | None = None
    if previous is not None:
        _check_resumable(
            previous,
            config_hash=fresh.config_hash,
            data_manifest_hash=fresh.data_manifest_hash,
        )
        resumed_from = store.latest_checkpoint(run_id)
        if resumed_from is None:
            raise RunConflict(
                f"run {run_id} stored no checkpoint to resume from; pass a new --run-id"
            )
    with reward_pool(root, workers=workers) as pool:
        rewards = Rewards(gold, pool, config.rewards)
        if grpo.eval_every is not None:
            callbacks.append(
                val_eval_callback(
                    pool,
                    tokenizer,
                    every=grpo.eval_every,
                    examples=grpo.eval_examples,
                    prompt_version=config.data.prompt_version,
                    max_new_tokens=grpo.max_completion_length,
                    model_ref=config.run.model_ref,
                    record=metrics.write,
                    data_dir=root,
                    cache_root=cache_root,
                )
            )
        manifest = _start(run_dir, fresh, previous, store)
        log.info(
            "grpo %s: %d rows, on %s%s",
            run_id,
            len(rows),
            platform,
            "" if resumed_from is None else f", from {resumed_from}",
        )
        adapter: Path | None = None
        try:
            with tempfile.TemporaryDirectory(prefix=f"forge-grpo-{run_id}-") as work:
                trained = backend.grpo(
                    model,
                    tokenizer,
                    rows,
                    rewards.functions,
                    grpo,
                    callbacks,
                    output_dir=work,
                    seed=config.run.seed,
                    reward_weights=rewards.weights,
                    resume_from=None if resumed_from is None else run_dir / resumed_from,
                )
                if guard.stopped_at is None:
                    adapter = Path(shutil.copytree(trained, run_dir / FINAL))
        except BaseException as exc:
            # KeyboardInterrupt too: a run stopped by hand is failed, not still running.
            _end(run_dir, manifest, "failed", store, checkpoints.kept, error=exc)
            raise
    status: EndStatus = "finished" if guard.stopped_at is None else "stopped_time_guard"
    final = _end(run_dir, manifest, status, store, checkpoints.kept)
    return GrpoRun(final, run_dir, len(rows), adapter, guard.stopped_at, resumed_from)


def _check_resumable(previous: RunManifest, **session: str) -> None:
    """Raise ``RunConflict`` unless the stored run ``previous`` is an unfinished GRPO run
    whose manifest has the fields ``session`` names as this session has them:
    ``git_sha``, ``config_hash`` or ``data_manifest_hash``."""
    run = previous.run_id
    if previous.kind != "grpo":
        raise RunConflict(f"run {run} is a {previous.kind} run; forge train grpo resumes grpo runs")
    if previous.status == "finished":
        raise RunConflict(f"run {run} has finished; pass a new --run-id to train again")
    for field, now in session.items():
        before = getattr(previous, field)
        if before != now:
            what = field.replace("_", " ").replace("git sha", "commit")
            raise RunConflict(
                f"run {run} was made with {what} {before}, and this session has {now}; resume "
                "it as it was made, or pass a new --run-id"
            )


def _outputs(run_dir: Path, kept: Mapping[float, int] | None = None) -> dict[str, str]:
    """A training run's outputs (``forge.train.sft.run_outputs``) and its sessions file."""
    return {**run_outputs(run_dir, kept), "sessions": SESSIONS_FILE}


def _start(
    run_dir: Path, fresh: RunManifest, previous: RunManifest | None, store: LocalRunStore
) -> RunManifest:
    """Write the config and the running manifest, ``fresh`` or the resumed run's, and
    begin a session; a resumed run keeps its first start."""
    if previous is None:
        try:
            # Not exist_ok: of two processes given one run id, the second stops here.
            run_dir.mkdir(parents=True)
        except FileExistsError as exc:
            raise RunExists(f"run {fresh.run_id} was started in {run_dir} meanwhile") from exc
        manifest = fresh
    else:
        manifest = previous.model_copy(update={"status": "running", "ended_at": None})
    write_atomic(run_dir / CONFIG_FILE, yaml.safe_dump(manifest.config, sort_keys=False).encode())
    begin(run_dir, fresh.started_at, resumes=previous)
    manifest = manifest.model_copy(update={"outputs": _outputs(run_dir)})
    store.put_manifest(manifest)
    return manifest


def _end(
    run_dir: Path,
    manifest: RunManifest,
    status: EndStatus,
    store: LocalRunStore,
    kept: Mapping[float, int],
    *,
    error: BaseException | None = None,
) -> RunManifest:
    """End the session and write the run's manifest as it ended, with the traceback
    beside it when it failed; an earlier session's traceback goes once one doesn't."""
    if error is None:
        (run_dir / ERROR_FILE).unlink(missing_ok=True)
    else:
        write_atomic(run_dir / ERROR_FILE, "".join(traceback.format_exception(error)).encode())
    now = datetime.now(UTC)
    final = close(manifest, status, now=now, sessions=end(run_dir, now, status))
    final = final.model_copy(update={"outputs": _outputs(run_dir, kept)})
    store.put_manifest(final)
    return final
