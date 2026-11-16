"""What an eval run's adapter names, and the digest that ties a score to its weights.

An adapter is a local folder or a Hub repo id with an optional ``@revision``
(``forge.infer.base.split_ref``). A folder that is a training run's checkpoint is
``<folder>/<run_id>/checkpoints/<name>``, the name ``final``, ``best`` or
``step-NNNNNN`` (plan section 3.9): ``runs/<run_id>/checkpoints/final`` from the repo
root on a laptop, and the same path in a kernel's work folder, where an eval job fetches
it from the run store first. ``checkpoint_ref`` is the one parser of that form, for
``forge kaggle run --adapter`` before a push, for the kernel's fetch and for
``forge runs promote``, so the three never disagree on a spelling. A trailing ``/``, as
a shell's completion adds, and a leading ``./`` name the same checkpoint.

``adapter_sha256`` is the digest of an adapter's files, ``ADAPTER_FILES``: the weights
and the settings PEFT loads them with. An eval run with an adapter records it in its
manifest's outputs as ``adapter_sha256`` (``forge.eval.run_files``), and a training run
records the digest of each checkpoint it keeps as ``adapter_sha256:<checkpoint>``, such
as ``adapter_sha256:checkpoints/final`` (``checkpoint_digests``). A val score is then
tied to the exact weights that made it, wherever the files went in between, which a path
alone can't show: the run store may hold other files under the same name.
``check_adapter`` holds a folder to a digest recorded elsewhere, as ``forge runs push
--adapter-sha256`` does for a run whose manifest records none.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from forge.eval.config import check_run_id
from forge.eval.run_files import ADAPTER_DIGEST
from forge.infer.base import split_ref
from forge.ops.hub import BEST, CHECKPOINTS, FINAL, RUNS, checkpoint_step

# A LoRA adapter as PEFT saves one, which forge eval run --adapter loads.
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


@dataclass(frozen=True)
class CheckpointRef:
    """A training run's checkpoint, as an adapter names it."""

    folder: str  # the folder that holds the run's folder, as the adapter writes it; "" for none
    run_id: str
    rel: str  # the checkpoint's path in its run, such as checkpoints/final

    @property
    def in_runs(self) -> bool:
        """Whether the adapter names it in ``runs/``, from where the command runs: the form
        a kernel fetches from the run store."""
        return self.folder == RUNS

    @property
    def path(self) -> str:
        """The checkpoint's folder, spelled the one way: ``runs/<run_id>/checkpoints/final``."""
        return str(PurePosixPath(self.folder or ".", self.run_id, self.rel))


def checkpoint_ref(adapter: str) -> CheckpointRef | None:
    """The training run's checkpoint ``adapter`` names, or None when it names none, as a
    Hub repo id or another folder does.

    Raises ``ValueError`` for a path whose last folders are ``<run_id>/checkpoints/<name>``
    with a name no checkpoint has, or a run id no run has.
    """
    parts = PurePosixPath(adapter).parts
    if len(parts) < 3 or parts[-2] != CHECKPOINTS:
        return None
    run_id, rel = parts[-3], f"{CHECKPOINTS}/{parts[-1]}"
    if rel not in (FINAL, BEST) and checkpoint_step(rel) is None:
        raise ValueError(
            f"{adapter} names {rel} of run {run_id}, and a run's checkpoints are {FINAL}, "
            f"{BEST} and {CHECKPOINTS}/step-NNNNNN"
        )
    try:
        check_run_id(run_id)
    except ValueError as exc:
        raise ValueError(f"{adapter} names a checkpoint of no run: {exc}") from None
    folder = PurePosixPath(*parts[:-3]).as_posix() if len(parts) > 3 else ""
    return CheckpointRef(folder, run_id, rel)


def adapter_sha256(folder: Path) -> str:
    """The digest of the adapter in ``folder``: the sha256 of one line a file of
    ``ADAPTER_FILES``, its name and its own sha256, so the same files hash alike in any
    folder. Raises ``FileNotFoundError`` when the folder lacks one of them."""
    missing = [name for name in ADAPTER_FILES if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{folder} has no {' or '.join(missing)}, so it holds no adapter")
    lines = []
    for name in ADAPTER_FILES:
        with (folder / name).open("rb") as handle:
            lines.append(f"{name} {hashlib.file_digest(handle, 'sha256').hexdigest()}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def check_adapter(folder: Path, expected: str) -> None:
    """Raise ``ValueError`` unless the adapter in ``folder`` has the digest ``expected``, and
    as ``adapter_sha256`` does for a folder that holds none."""
    found = adapter_sha256(folder)
    if found != expected:
        raise ValueError(f"{folder} holds adapter files with sha256 {found}, not {expected}")


def adapter_folder(adapter: str) -> Path:
    """The folder of the adapter ``adapter`` names, as a backend loads it: a local folder,
    or the files of a Hub repo at its revision, from the Hub's cache once fetched.

    Raises ``FileNotFoundError`` for a training run's checkpoint this machine lacks, which
    is never a Hub repo, and as ``split_ref`` does for anything else that is neither.
    """
    if Path(adapter).is_dir():
        return Path(adapter)
    ref = checkpoint_ref(adapter)
    if ref is not None:
        raise FileNotFoundError(
            f"{adapter} is {ref.rel} of run {ref.run_id}, and there is no such folder here; "
            f"forge runs pull {ref.run_id} first"
        )
    repo, revision = split_ref(adapter)
    import huggingface_hub

    return Path(
        huggingface_hub.snapshot_download(
            repo, revision=revision, allow_patterns=list(ADAPTER_FILES)
        )
    )


def digest_output(rel: str) -> str:
    """The manifest output that names the digest of a training run's checkpoint ``rel``."""
    return f"{ADAPTER_DIGEST}:{rel}"


def checkpoint_digests(run_dir: Path) -> dict[str, str]:
    """The digest of each checkpoint in the training run's folder ``run_dir`` that holds an
    adapter, by the manifest output that names it (``digest_output``)."""
    folder = run_dir / CHECKPOINTS
    if not folder.is_dir():
        return {}
    return {
        digest_output(f"{CHECKPOINTS}/{path.name}"): adapter_sha256(path)
        for path in sorted(folder.iterdir())
        if all((path / name).is_file() for name in ADAPTER_FILES)
    }
