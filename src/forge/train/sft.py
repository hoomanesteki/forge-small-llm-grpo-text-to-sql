"""``forge train sft``: from a training config to a LoRA adapter and its run manifest.

One run, in order (plan FG-403; the starting values are ADR-0010's):

1. The backend the config names (``forge.train.backends.select_backend``), and the
   platform its manifest gives: ``local-cpu`` or ``local-mps``. CUDA here has no
   platform to name (ADR-0016), so such a run is refused before anything loads; a GPU run
   goes through a kernel job.
2. The model with LoRA, and its tokenizer.
3. The SFT rows of the train split, measured with that tokenizer (``forge.train.sft_data``)
   and saved to ``<sft_root>/<hash>/``, which keeps a folder already built with that hash,
   as ``forge data build-sft`` builds it. The trainer reads the saved copy once its rows
   hash to its name, opened in memory so the trainer's tokenizing writes no cache files
   beside it.
4. The manifest, ``running``. Its config is what can change the adapter: the run, data,
   LoRA and SFT sections, the rows' hash and count (``sft_data``), and the backend, device
   and dtype it trains with (``trainer``), all under the config hash. Its data manifest
   hash is the Spider files'. The ops section changes no weight, so it stays out, as an
   eval run's does.
5. Training, with the callbacks every job has (``forge.train.callbacks``): a checkpoint
   in the run every ``ops.checkpoint_every`` steps, the time guard, counted from when the
   run starts, and the trainer's logs in ``metrics.jsonl``.
6. The adapter in ``checkpoints/final``, and the manifest ``finished``. When the time
   guard stopped training, its checkpoint is the run's last, there is no ``final``, and
   the manifest says ``stopped_time_guard``. A run that fails says ``failed``, with the
   traceback in ``error.txt``.

The run's folder is ``runs/<run_id>/`` (``FORGE_RUNS_DIR``), in plan section 3.9's layout:
``manifest.json``, ``config.yaml``, ``metrics.jsonl``, ``checkpoints/step-NNNNNN/`` and
``checkpoints/final/``. The trainer has no eval set, since training reads train only, so
no run writes ``checkpoints/best/``: that is the checkpoint val picks once each is
evaluated (FG-404). An SFT run can't be resumed yet, so a run id names one run.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from datasets import Dataset, load_from_disk

from forge.core.ids import make_run_id
from forge.core.types import RunManifest
from forge.data.prompts import ChatTokenizer, chat_token_counter
from forge.data.spider import resolve_data_dir
from forge.eval.config import check_run_id
from forge.eval.run_files import write_atomic
from forge.ops.hub import FINAL, METRICS_FILE, LocalRunStore
from forge.ops.manifest import EndStatus, Platform, accelerator, build_manifest, close, code_sha
from forge.ops.paths import runs_dir
from forge.ops.timeguard import Clock
from forge.train.backends import TrainBackend, select_backend
from forge.train.config import TrainConfig
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


def manifest_config(config: TrainConfig, meta: SftMeta, backend: TrainBackend) -> dict[str, Any]:
    """What an SFT run's manifest records and hashes, as the module docstring lists it."""
    sections: dict[str, Any] = config.model_dump(
        mode="json", include={"run", "data", "lora", "sft"}
    )
    return {
        **sections,
        "sft_data": {"hash": meta.hash, "rows": meta.counts.rows},
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
    env: Mapping[str, str] | None = None,
    clock: Clock = time.monotonic,
) -> SftRun:
    """Train the SFT run ``config`` describes, as the module docstring says.

    ``backend`` replaces the one the config names, such as a stand-in in a test.
    ``sft_root`` is where the rows' folder goes (``data/sft`` by default), ``env`` where
    the callbacks look for a W&B key, and ``clock`` the time guard's clock. Raises
    ``DirtyTreeError`` on uncommitted changes unless ``allow_dirty``, ``RunExists`` for a
    run id already used, and ``ValueError`` for a config or a device no SFT run can take.
    """
    if config.sft is None:
        raise ValueError(f"run.kind is {config.run.kind}, and forge train sft trains sft runs")
    root = resolve_data_dir(data_dir).resolve()
    runs_root = runs_dir() if runs_root is None else runs_root
    # Asked before a model loads, so uncommitted changes cost no time.
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
    platform = platform_for(backend.device)
    # Imported once the backend is made: Unsloth patches transformers as it is imported,
    # so nothing may import transformers before it.
    from forge.train.callbacks import TimeGuardCallback, training_callbacks

    store = LocalRunStore(runs_root)
    # Made now, so the time guard counts the model's loading too.
    callbacks = training_callbacks(
        config.ops, store, run_id, metrics_path=run_dir / METRICS_FILE, env=env, clock=clock
    )
    guard = next(c for c in callbacks if isinstance(c, TimeGuardCallback))
    model, tokenizer = backend.load(
        config.run.model_ref,
        config.lora,
        load_in_4bit=config.run.load_in_4bit,
        seed=config.run.seed,
    )
    folder, meta = _build_rows(config, tokenizer, root, sft_root)
    rows = open_rows(folder, meta)
    manifest = build_manifest(
        "sft",
        manifest_config(config, meta, backend),
        model_ref=config.run.model_ref,
        seed=config.run.seed,
        run_id=run_id,
        platform=platform,
        gpu=accelerator(platform),
        allow_dirty=allow_dirty,
        data_dir=root,
    )
    try:
        # Not exist_ok: of two processes given one run id, the second stops here.
        run_dir.mkdir(parents=True)
    except FileExistsError as exc:
        raise RunExists(f"run {run_id} was started in {run_dir} meanwhile") from exc
    manifest = _start(run_dir, manifest, store)
    log.info("sft %s: %d rows from %s, on %s", run_id, meta.counts.rows, folder, platform)
    adapter: Path | None = None
    try:
        with tempfile.TemporaryDirectory(prefix=f"forge-sft-{run_id}-") as work:
            trained = backend.sft(
                model, tokenizer, rows, config.sft, callbacks, output_dir=work, seed=config.run.seed
            )
            if guard.stopped_at is None:
                adapter = Path(shutil.copytree(trained, run_dir / FINAL))
    except BaseException as exc:
        # KeyboardInterrupt too: a run stopped by hand is failed, not still running.
        _end(run_dir, manifest, "failed", store, error=exc)
        raise
    status: EndStatus = "finished" if guard.stopped_at is None else "stopped_time_guard"
    final = _end(run_dir, manifest, status, store)
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


def _outputs(run_dir: Path) -> dict[str, str]:
    return {name: rel for name, rel in OUTPUTS.items() if (run_dir / rel).exists()}


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
    *,
    error: BaseException | None = None,
) -> RunManifest:
    """Write the run's final manifest, with the traceback beside it when it failed."""
    if error is not None:
        write_atomic(run_dir / ERROR_FILE, "".join(traceback.format_exception(error)).encode())
    final = close(manifest, status).model_copy(update={"outputs": _outputs(run_dir)})
    store.put_manifest(final)
    return final
