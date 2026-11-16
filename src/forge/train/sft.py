"""``forge train sft``: from a training config to a LoRA adapter and its run manifest.

One run, in order (plan FG-403; the starting values are ADR-0010's):

1. The backend the config names (``forge.train.backends.select_backend``), and the
   platform its manifest gives: ``local-cpu`` or ``local-mps``, or in a kernel job the
   host ``kaggle/entry.py`` names. CUDA on the laptop has no platform to name (ADR-0016),
   so such a run is refused before anything loads; a GPU run goes through a kernel job.
2. The model with LoRA, and its tokenizer.
3. The SFT rows of the train split, measured with that tokenizer (``forge.train.sft_data``)
   and saved to ``<sft_root>/<hash>/``, which keeps a folder already built with that hash,
   as ``forge data build-sft`` builds it. The trainer reads the saved copy once its rows
   hash to its name, opened in memory so the trainer's tokenizing writes no cache files
   beside it. A row longer than ``max_length`` tokens, prompt and reply as TRL encodes
   them, is left out (``fit_rows``): TRL would cut its reply, and the model would learn a
   query without its end. With ``subset_rows`` the run trains on that many of the rest,
   those whose ids hash lowest with the run's seed (``subset``), so the same seed draws
   the same rows on any machine.
4. The manifest, ``running``. Its config is what can change the adapter: the run, data,
   LoRA and SFT sections, the rows' hash and count, how many were over ``max_length`` and
   how many it trains on (``sft_data``), and the backend, device and dtype it trains with
   (``trainer``), all under the config hash. Its data manifest hash is the Spider files'.
   The ops section changes no weight, so it stays out, as an eval run's does.
5. Training, with the callbacks every job has (``forge.train.callbacks``): a checkpoint
   in the run every ``ops.checkpoint_every`` steps, the time guard, counted from when the
   run starts, and the trainer's logs in ``metrics.jsonl``. The checkpoint at each share
   of the run ``ops.keep_checkpoints_at`` names is kept from retention, and the manifest's
   outputs name it ``kept_at_<share>``, such as ``kept_at_0.5``. The callbacks' run store
   is ``runs_root`` itself, so a push is a copy into the run's folder that never fails on
   the network, and no kept checkpoint waits out of the trainer's rotation as
   ``unpushed-checkpoint-<step>``, which would be in the temp folder the trainer saves in
   and this function deletes. So the run's folder holds every checkpoint the run keeps:
   the latest ``KEEP_CHECKPOINTS`` steps, the kept shares and ``final``. In a kernel job
   that folder is in the kernel's output, which is how an eval job gets a checkpoint.
6. The adapter in ``checkpoints/final``, and the manifest ``finished``, its outputs
   holding the digest of each checkpoint the run keeps (``forge.ops.adapters``), which an
   eval run of that checkpoint records too. When the time
   guard stopped training, its checkpoint is the run's last, there is no ``final``, and
   the manifest says ``stopped_time_guard``. A run that fails says ``failed``, with the
   traceback in ``error.txt``.

The run's folder is ``runs/<run_id>/`` (``FORGE_RUNS_DIR``), in plan section 3.9's layout:
``manifest.json``, ``config.yaml``, ``metrics.jsonl``, ``checkpoints/step-NNNNNN/`` and
``checkpoints/final/``. The trainer has no eval set, since training reads train only, so
no run writes ``checkpoints/best/``: that is the checkpoint val picks once each is
evaluated, which ``promote_best`` copies there (``forge runs promote``, FG-404). An SFT run
can't be resumed yet, so a run id names one run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from datasets import Dataset, load_from_disk

from forge.core.ids import make_run_id
from forge.core.types import RunManifest
from forge.data.prompts import ChatTokenizer, chat_token_counter
from forge.data.spider import resolve_data_dir
from forge.eval.config import EvalConfig, Host, check_run_id
from forge.eval.metrics import summarize
from forge.eval.run_files import ADAPTER_DIGEST, read_run, write_atomic
from forge.ops.adapters import (
    ADAPTER_FILES,
    adapter_sha256,
    checkpoint_digests,
    checkpoint_ref,
    digest_output,
)
from forge.ops.hub import (
    BEST,
    FINAL,
    METRICS_FILE,
    PROMOTION_FILE,
    LocalRunStore,
    RunStore,
    checkpoint_dir,
    checkpoint_step,
)
from forge.ops.manifest import EndStatus, Platform, accelerator, build_manifest, close, code_sha
from forge.ops.paths import runs_dir
from forge.ops.timeguard import Clock
from forge.train.backends import TrainBackend, select_backend
from forge.train.config import SftConfig, TrainConfig
from forge.train.sft_data import SftMeta, SftRow, build_sft_dataset, save_sft, sft_hash

log = logging.getLogger(__name__)

CONFIG_FILE = "config.yaml"
ERROR_FILE = "error.txt"
# The files a training run's manifest lists in its outputs, when they exist.
OUTPUTS: Mapping[str, str] = {
    "config": CONFIG_FILE,
    "metrics": METRICS_FILE,
    "adapter": FINAL,
    "error": ERROR_FILE,
}
# The manifest's name for a checkpoint ops.keep_checkpoints_at keeps.
KEPT_OUTPUT = "kept_at_{share:g}"
# What promote_best's scores are; it writes them in PROMOTION_FILE.
PROMOTION_METRIC = "official EX on val"
# Settings an eval run's config may hold that change no score.
_SCORELESS = ("adapter", "allow_over_budget")
LOCAL_PLATFORMS: Mapping[str, Platform] = {"cpu": "local-cpu", "mps": "local-mps"}


class RunExists(FileExistsError):
    """The run id names a run that is already there."""


@dataclass(frozen=True)
class SftRun:
    """A training run as it ended: its final manifest and where its files are."""

    manifest: RunManifest
    root: Path  # the run's folder
    sft_data: Path  # the saved rows it trained on
    adapter: Path | None  # checkpoints/final; None when the time guard stopped the run
    stopped_at: int | None  # the step the time guard stopped after


def platform_for(device: str) -> Platform:
    """What a run's manifest calls the machine it trains on, on ``device``."""
    if device in LOCAL_PLATFORMS:
        return LOCAL_PLATFORMS[device]
    raise ValueError(
        f"a run on {device} here has no platform its manifest can name (ADR-0016); train "
        "on cpu or mps here, and on a GPU through a Kaggle job"
    )


@dataclass(frozen=True)
class FitRows:
    """The rows a run trains on, and how many of the saved ones were too long."""

    rows: Dataset
    over_max_length: int


def manifest_config(
    config: TrainConfig, meta: SftMeta, fit: FitRows, backend: TrainBackend
) -> dict[str, Any]:
    """What an SFT run's manifest records and hashes, as the module docstring lists it."""
    sections: dict[str, Any] = config.model_dump(
        mode="json", include={"run", "data", "lora", "sft"}
    )
    return {
        **sections,
        "sft_data": {
            "hash": meta.hash,
            "rows": meta.counts.rows,
            "over_max_length": fit.over_max_length,
            "trained": len(fit.rows),
        },
        "trainer": {
            "backend": backend.name,
            "device": backend.device,
            "dtype": backend.precision.dtype,
        },
    }


def run_sft(
    config: TrainConfig,
    *,
    run_id: str | None = None,
    allow_dirty: bool = False,
    data_dir: Path | None = None,
    runs_root: Path | None = None,
    sft_root: Path | None = None,
    backend: TrainBackend | None = None,
    host: Host | None = None,
    env: Mapping[str, str] | None = None,
    clock: Clock = time.monotonic,
) -> SftRun:
    """Train the SFT run ``config`` describes, as the module docstring says.

    ``backend`` replaces the one the config names, such as a stand-in in a test.
    ``sft_root`` is where the rows' folder goes (``data/sft`` by default), ``host`` the
    kernel's machine a job runs on, ``env`` where the callbacks look for a W&B key, and
    ``clock`` the time guard's clock. Raises
    ``DirtyTreeError`` on uncommitted changes unless ``allow_dirty``, ``RunExists`` for a
    run id already used, and ``ValueError`` for a config or a device no SFT run can take.
    """
    if config.sft is None:
        raise ValueError(f"run.kind is {config.run.kind}, and forge train sft trains sft runs")
    root = resolve_data_dir(data_dir).resolve()
    runs_root = runs_dir() if runs_root is None else runs_root
    # Asked before a model loads, so uncommitted changes cost no time, and only here: the
    # manifest names this commit, whatever the tree does while the model and rows load.
    sha = code_sha(allow_dirty=allow_dirty)
    run_id = run_id or make_run_id("sft", config.run.model_ref, sha)
    check_run_id(run_id)
    run_dir = runs_root / run_id
    if run_dir.exists():
        raise RunExists(f"run {run_id} is in {run_dir} already; pass a new --run-id")
    if backend is None:
        backend = select_backend(
            config.run.backend, device=config.run.device, max_seq_length=config.max_seq_length
        )
    platform = host if host is not None else platform_for(backend.device)
    # Imported once the backend is made: Unsloth patches transformers as it is imported,
    # so nothing may import transformers before it.
    from forge.train.callbacks import CheckpointCallback, TimeGuardCallback, training_callbacks

    store = LocalRunStore(runs_root)
    # Made now, so the time guard counts the model's loading too.
    callbacks = training_callbacks(
        config.ops, store, run_id, metrics_path=run_dir / METRICS_FILE, env=env, clock=clock
    )
    guard = next(c for c in callbacks if isinstance(c, TimeGuardCallback))
    checkpoints = next(c for c in callbacks if isinstance(c, CheckpointCallback))
    model, tokenizer = backend.load(
        config.run.model_ref,
        config.lora,
        load_in_4bit=config.run.load_in_4bit,
        seed=config.run.seed,
    )
    folder, meta = _build_rows(config, tokenizer, root, sft_root)
    fit = fit_rows(open_rows(folder, meta), tokenizer, config.sft, seed=config.run.seed)
    manifest = build_manifest(
        "sft",
        manifest_config(config, meta, fit, backend),
        model_ref=config.run.model_ref,
        seed=config.run.seed,
        run_id=run_id,
        platform=platform,
        gpu=accelerator(platform),
        data_dir=root,
        git_sha=sha,
    )
    try:
        # Not exist_ok: of two processes given one run id, the second stops here.
        run_dir.mkdir(parents=True)
    except FileExistsError as exc:
        raise RunExists(f"run {run_id} was started in {run_dir} meanwhile") from exc
    manifest = _start(run_dir, manifest, store)
    log.info("sft %s: %d rows from %s, on %s", run_id, len(fit.rows), folder, platform)
    adapter: Path | None = None
    try:
        with tempfile.TemporaryDirectory(prefix=f"forge-sft-{run_id}-") as work:
            trained = backend.sft(
                model,
                tokenizer,
                fit.rows,
                config.sft,
                callbacks,
                output_dir=work,
                seed=config.run.seed,
            )
            if guard.stopped_at is None:
                adapter = Path(shutil.copytree(trained, run_dir / FINAL))
    except BaseException as exc:
        # KeyboardInterrupt too: a run stopped by hand is failed, not still running.
        _end(run_dir, manifest, "failed", store, checkpoints.kept, error=exc)
        raise
    status: EndStatus = "finished" if guard.stopped_at is None else "stopped_time_guard"
    final = _end(run_dir, manifest, status, store, checkpoints.kept)
    return SftRun(final, run_dir, folder, adapter, guard.stopped_at)


def open_rows(folder: Path, meta: SftMeta) -> Dataset:
    """The saved SFT rows in ``folder``, in memory, once they hash to ``meta``'s hash.

    In memory, because ``SFTTrainer`` tokenizes with ``datasets.map``, which would write
    its cache files into the folder of a dataset opened from disk.
    """
    loaded = load_from_disk(str(folder), keep_in_memory=True)
    if not isinstance(loaded, Dataset):
        raise ValueError(f"{folder} holds a {type(loaded).__name__}, not one dataset of rows")
    found = sft_hash(cast(list[SftRow], loaded.to_list()), meta.prompt_version, meta.tokenizer)
    if found != meta.hash:
        raise ValueError(
            f"the rows in {folder} hash to {found}, not to its name; remove the folder, and "
            "the next run builds it again"
        )
    return loaded


def row_tokens(tokenizer: ChatTokenizer, row: Mapping[str, Any]) -> int:
    """A row's tokens, prompt and reply, as TRL's ``SFTTrainer`` encodes them to train on."""
    encoded = tokenizer.apply_chat_template(
        [*row["prompt"], *row["completion"]], tokenize=True, return_dict=True
    )
    return len(encoded["input_ids"])


def subset(example_ids: Sequence[str], size: int, seed: int) -> list[int]:
    """The positions of the ``size`` ids that hash lowest with ``seed``, in their order.

    A hash, not a random draw, so the rows depend neither on a library's generator nor on
    the order of the rows.
    """

    def rank(position: int) -> str:
        return hashlib.sha256(f"{seed}:{example_ids[position]}".encode()).hexdigest()

    return sorted(sorted(range(len(example_ids)), key=rank)[:size])


def fit_rows(rows: Dataset, tokenizer: ChatTokenizer, cfg: SftConfig, *, seed: int) -> FitRows:
    """The rows a run trains on: those that fit in ``cfg.max_length`` tokens, or
    ``cfg.subset_rows`` of them, as the module docstring says.

    Raises ``ValueError`` when no row fits, or when the subset is larger than what does.
    """
    lengths = [row_tokens(tokenizer, row) for row in rows]
    kept = [index for index, length in enumerate(lengths) if length <= cfg.max_length]
    if not kept:
        raise ValueError(
            f"all {len(lengths)} SFT rows are over max_length {cfg.max_length} tokens; the "
            f"shortest has {min(lengths, default=0)}"
        )
    over = len(lengths) - len(kept)
    if over:
        log.warning(
            "sft: %d of %d rows are over max_length %d tokens and left out",
            over,
            len(lengths),
            cfg.max_length,
        )
    if cfg.subset_rows is not None:
        if cfg.subset_rows > len(kept):
            raise ValueError(
                f"subset_rows {cfg.subset_rows} is more than the {len(kept)} rows that fit "
                f"in max_length {cfg.max_length}"
            )
        ids = rows["example_id"]
        drawn = subset([ids[index] for index in kept], cfg.subset_rows, seed)
        kept = [kept[position] for position in drawn]
    if len(kept) == len(lengths):
        return FitRows(rows, over)
    # In memory, as open_rows reads them, so selecting writes no index file beside them.
    return FitRows(rows.select(kept, keep_in_memory=True), over)


@dataclass(frozen=True)
class Promotion:
    """A run's ``checkpoints/best``: which checkpoint was copied there, and why."""

    checkpoint: str  # the promoted checkpoint's path in the run
    best: Path  # checkpoints/best in the run's folder
    record: dict[str, Any]  # what best.json holds
    manifest: RunManifest  # the run's manifest, its outputs naming best and best.json


def promote_best(
    run_id: str,
    scores_by_ckpt: Mapping[str, float],
    *,
    sources: Mapping[str, str] | None = None,
    runs_root: Path | None = None,
    store: RunStore | None = None,
) -> Promotion:
    """Copy the checkpoint of SFT run ``run_id`` that val scores highest to
    ``checkpoints/best``, where GRPO starts from (plan section 3.10).

    ``scores_by_ckpt`` maps checkpoints of the run, as paths in it such as
    ``checkpoints/step-000242`` and ``checkpoints/final``, to their official EX on val, and
    ``sources`` maps each to the eval run its score comes from. A tie goes to the later
    checkpoint: an earlier one is promoted only when val scores it higher.

    Each checkpoint's folder must hold the files the run saved there, by the digest its
    manifest records. The run's folder in ``runs_root`` (``runs/``) gets
    ``checkpoints/best/``, ``best.json`` with the checkpoint, the scores and their sources,
    and a manifest whose outputs name both, and best/'s digest. With ``store``, which must
    hold the same manifest as the folder, they go there too, the manifest last, so neither
    copy names a best it lacks. Promoting again replaces all three. Raises ``RunNotFound``
    for no such run, and ``ValueError`` for a run that is not a finished SFT run, or for a
    checkpoint, a score or a source it can't take.
    """
    if not scores_by_ckpt:
        raise ValueError(f"no checkpoint of run {run_id} has a score, so none can be promoted")
    local = LocalRunStore(runs_dir() if runs_root is None else runs_root)
    manifest = local.get_manifest(run_id)
    if (manifest.kind, manifest.status) != ("sft", "finished"):
        raise ValueError(
            f"run {run_id} is a {manifest.status} {manifest.kind} run; a checkpoint is "
            "promoted from a finished SFT run"
        )
    run_dir = local.root / run_id
    order = {rel: _checkpoint_order(rel, run_id) for rel in scores_by_ckpt}
    for rel, score in scores_by_ckpt.items():
        # Not 0 <= score <= 1 is true of NaN too.
        if isinstance(score, bool) or not isinstance(score, int | float) or not 0 <= score <= 1:
            raise ValueError(f"{rel} scores {score!r}; an EX is a share from 0 to 1")
        missing = [name for name in ADAPTER_FILES if not (run_dir / rel / name).is_file()]
        if missing:
            raise ValueError(
                f"{run_dir / rel} has no {' or '.join(missing)}, so it holds no adapter to "
                "promote; forge runs pull the run first"
            )
        _check_saved(manifest, rel, run_dir / rel)
    if sources is not None and set(sources) != set(scores_by_ckpt):
        raise ValueError(
            f"the sources name {sorted(sources)}, and the scores {sorted(scores_by_ckpt)}"
        )
    if store is not None and store.get_manifest(run_id) != manifest:
        raise ValueError(
            f"{store} holds another manifest of run {run_id} than {run_dir} does; forge "
            "runs pull the run, then promote"
        )
    chosen = max(scores_by_ckpt, key=lambda rel: (scores_by_ckpt[rel], order[rel]))
    best = run_dir / BEST
    with tempfile.TemporaryDirectory(prefix=".best-", dir=run_dir) as work:
        # Copied whole beside the run first, so a failed copy leaves the old best/ as it was.
        staged = Path(shutil.copytree(run_dir / chosen, Path(work) / "best"))
        if best.exists():
            shutil.rmtree(best)
        staged.rename(best)
    ranked = sorted(scores_by_ckpt, key=order.__getitem__)
    record: dict[str, Any] = {
        "run_id": run_id,
        "checkpoint": chosen,
        "metric": PROMOTION_METRIC,
        "scores": {rel: scores_by_ckpt[rel] for rel in ranked},
    }
    if sources is not None:
        record["eval_runs"] = {rel: sources[rel] for rel in ranked}
    write_atomic(run_dir / PROMOTION_FILE, (json.dumps(record, indent=2) + "\n").encode())
    outputs = {
        **manifest.outputs,
        "best": BEST,
        "promotion": PROMOTION_FILE,
        digest_output(BEST): manifest.outputs[digest_output(chosen)],
    }
    promoted = manifest.model_copy(update={"outputs": outputs})
    if store is not None:
        store.put_folder(run_id, best, BEST, replace=True)
        store.put_file(run_id, run_dir / PROMOTION_FILE, PROMOTION_FILE)
        store.put_manifest(promoted)
    # Last, so a failed push leaves the folder's manifest as the store's, and promoting
    # again goes through.
    local.put_manifest(promoted)
    log.info("promoted %s of run %s to %s", chosen, run_id, best)
    return Promotion(chosen, best, record, promoted)


def checkpoint_scores(
    run_id: str, eval_run_ids: Sequence[str], *, base: EvalConfig, runs_root: Path | None = None
) -> tuple[dict[str, float], dict[str, str]]:
    """The official EX on val of each checkpoint of SFT run ``run_id`` that one of the eval
    runs ``eval_run_ids`` in ``runs_root`` scored, and that eval run's id, by checkpoint.

    ``base`` is the base model's full val config, such as ``configs/eval/qwen3-4b-val.yaml``:
    the SFT run's model on every val question, with no adapter, at the prompt version the
    run trained on (its config's ``data.prompt_version``). Each eval run must be
    finished and made with that config and one of the run's checkpoints as the adapter
    (``<...>/<run_id>/checkpoints/<name>``, wherever its folder was, as
    ``forge.ops.adapters.checkpoint_ref`` reads it), on the files the run saved there, by
    their digest, with an official verdict on every score. The scores then differ by
    checkpoint alone, and each from the base model's by its adapter alone. Raises
    ``ValueError`` when one isn't, and as ``read_run`` does for a run that can't be read.
    """
    if not eval_run_ids:
        raise ValueError(f"name the eval runs of run {run_id}'s checkpoints, one at least")
    root = runs_dir() if runs_root is None else runs_root
    trained = LocalRunStore(root).get_manifest(run_id)
    _check_base(base, trained)
    expected = _scored(base.manifest_config())
    scores: dict[str, float] = {}
    sources: dict[str, str] = {}
    for eval_id in eval_run_ids:
        made = read_run(root / eval_id)
        manifest = made.manifest
        split = manifest.config["split"]
        if split != "val":
            raise ValueError(
                f"run {eval_id} evaluated {split}; checkpoints are chosen on val, and dev is "
                "for reported numbers (ADR-0004)"
            )
        found = checkpoint_ref(manifest.adapter_ref or "")
        if found is None or found.run_id != run_id:
            raise ValueError(
                f"run {eval_id}'s adapter is {manifest.adapter_ref}, no checkpoint of run {run_id}"
            )
        rel = found.rel
        settings = _scored(manifest.config)
        if settings != expected:
            changed = sorted(
                name
                for name in settings.keys() | expected.keys()
                if settings.get(name) != expected.get(name)
            )
            raise ValueError(
                f"run {eval_id} differs from the base config in {', '.join(changed)}, not in "
                "its adapter alone"
            )
        scored = manifest.outputs.get(ADAPTER_DIGEST)
        saved = trained.outputs.get(digest_output(rel))
        if scored is None or scored != saved:
            raise ValueError(
                f"run {eval_id} scored adapter files with sha256 {scored}, and run {run_id} "
                f"saved {rel} with sha256 {saved}, so the score isn't that checkpoint's"
            )
        if rel in sources:
            raise ValueError(f"runs {sources[rel]} and {eval_id} both scored {rel}; name one")
        ex = summarize(made.scores).ex_official
        if ex is None:
            raise ValueError(
                f"run {eval_id} lacks an official verdict on some scores; resume it so its "
                "official batch runs"
            )
        scores[rel], sources[rel] = ex.value, eval_id
    return scores, sources


def _check_base(base: EvalConfig, trained: RunManifest) -> None:
    """Raise ``ValueError`` unless ``base`` scores the training run's model alone on every
    val question, with the prompt version it trained on, as a checkpoint's eval runs must
    too."""
    if base.split != "val" or base.limit is not None:
        asked = f"the first {base.limit} {base.split} questions" if base.limit else base.split
        raise ValueError(
            f"the base config scores {asked}; checkpoints are chosen on the whole val split"
        )
    if base.adapter is not None:
        raise ValueError(f"the base config names the adapter {base.adapter}; it scores no adapter")
    if base.model != trained.model_ref:
        raise ValueError(
            f"the base config scores {base.model}, and run {trained.run_id} trained "
            f"{trained.model_ref}"
        )
    # The eval runs agree with the base config, so a base at another prompt version would
    # pick best/ from scores at a prompt the adapter never trained on.
    prompt = trained.config.get("data", {}).get("prompt_version")
    if base.prompt_version != prompt:
        trained_on = prompt or "no prompt version its config names"
        raise ValueError(
            f"the base config prompts with {base.prompt_version}, and run {trained.run_id} "
            f"trained on {trained_on}"
        )


def _scored(config: Mapping[str, Any]) -> dict[str, Any]:
    """An eval config's settings but those that change no score."""
    return {name: value for name, value in config.items() if name not in _SCORELESS}


def _check_saved(trained: RunManifest, rel: str, folder: Path) -> None:
    """Raise ``ValueError`` unless ``folder`` holds the adapter files the training run
    saved as its checkpoint ``rel``, by the digest its manifest records."""
    saved = trained.outputs.get(digest_output(rel))
    if saved is None:
        raise ValueError(
            f"run {trained.run_id}'s manifest records no digest of {rel}, so nothing shows "
            "which weights it saved there"
        )
    found = adapter_sha256(folder)
    if found != saved:
        raise ValueError(
            f"{folder} holds other files than run {trained.run_id} saved as {rel} (sha256 "
            f"{found}, not {saved}); forge runs pull the run again"
        )


def _checkpoint_order(rel: str, run_id: str) -> tuple[int, int]:
    """Where checkpoint ``rel`` comes in its run: by step, and the final adapter last."""
    if rel == FINAL:
        return (1, 0)
    step = checkpoint_step(rel)
    if step is None:
        raise ValueError(
            f"{rel!r} is no checkpoint of run {run_id} to promote: it takes {FINAL} or a "
            f"checkpoints/step-NNNNNN, never {BEST}"
        )
    return (0, step)


def _build_rows(
    config: TrainConfig, tokenizer: ChatTokenizer, data_dir: Path, sft_root: Path | None
) -> tuple[Path, SftMeta]:
    """The train split's SFT rows, measured with the model's tokenizer, saved."""
    built = build_sft_dataset(
        config.data.prompt_version,
        config.run.model_ref,
        chat_token_counter(tokenizer),
        data_dir=data_dir,
    )
    return save_sft(built, sft_root), built.meta


def _outputs(run_dir: Path, kept: Mapping[float, int] | None = None) -> dict[str, str]:
    """The run's ``OUTPUTS`` that exist, the kept checkpoints of ``kept``, a share of the
    run to its step, and the digest of each checkpoint in the folder."""
    found = {name: rel for name, rel in OUTPUTS.items() if (run_dir / rel).exists()}
    for share, step in (kept or {}).items():
        rel = checkpoint_dir(step)
        if (run_dir / rel).is_dir():
            found[KEPT_OUTPUT.format(share=share)] = rel
    return {**found, **checkpoint_digests(run_dir)}


def _start(run_dir: Path, manifest: RunManifest, store: LocalRunStore) -> RunManifest:
    """Write the config and the running manifest, which the time guard reads back."""
    write_atomic(run_dir / CONFIG_FILE, yaml.safe_dump(manifest.config, sort_keys=False).encode())
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
    """Write the run's final manifest, with the traceback beside it when it failed."""
    if error is not None:
        write_atomic(run_dir / ERROR_FILE, "".join(traceback.format_exception(error)).encode())
    final = close(manifest, status).model_copy(update={"outputs": _outputs(run_dir, kept)})
    store.put_manifest(final)
    return final
