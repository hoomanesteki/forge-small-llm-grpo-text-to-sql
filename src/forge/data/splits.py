"""Train, val and dev, split by database so no schema shows up in two of them.

``dev`` is Spider's official dev set. ``val`` is 10% of the training databases, rounded
up, with all of their questions: the first ones when the databases are sorted by
``sha256(db_id + VAL_SEED)``, so it is the same on every run and every machine.
Databases without rows are passed over (ADR-0004). ``train`` is the rest.

The split of the full download is committed under ``splits/``: one ``id<TAB>db_id``
line per question in ``<split>.ids.txt``, and ``splits.json`` with the seed and each
file's sha256. ``forge data splits --write`` regenerates them. CI checks invariant I1
against them without the data, and training checks its examples against them.

Training code calls ``load_train()`` and ``assert_train_only()``, and nothing else here.
``load_dev()`` needs a reason and spends one of the project's six looks at dev.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import get_args

from pydantic import BaseModel

from forge.core.hashing import sha256_hex
from forge.core.types import Example, Split
from forge.data.spider import ROWLESS_DBS, load_raw
from forge.ops.dev_usage import log_dev_use

SPLITS: tuple[Split, ...] = get_args(Split)

# Changing either one moves databases between train and val, which changes every val
# number and the data manifest hash: bump the seed's version rather than reuse a seed.
VAL_SEED = "forge-val-v1"
VAL_PERCENT = 10

MANIFEST_DIR: Traversable = files("forge.data") / "splits"
MANIFEST_FILE = "splits.json"


class SplitLeak(ValueError):
    """A database or an example is in a split it must not be in."""


class SplitManifest(BaseModel, frozen=True):
    """How the committed split files were made, and the sha256 of each one."""

    seed: str
    val_percent: int
    files: dict[str, str]  # "<split>.ids.txt" -> sha256 of its bytes


def val_databases(train_db_ids: Iterable[str], *, seed: str = VAL_SEED) -> frozenset[str]:
    """The val databases: ``VAL_PERCENT`` of these training databases, rounded up.

    They are the first by ``sha256(db_id + seed)`` hex, skipping databases without rows,
    whose gold results are all empty and would make val accuracy meaningless.
    """
    databases = set(train_db_ids)
    if len(databases) < 2:
        raise ValueError(
            f"splitting off val needs 2 or more training databases, got {len(databases)}"
        )
    count = math.ceil(len(databases) * VAL_PERCENT / 100)
    candidates = sorted(databases - ROWLESS_DBS, key=lambda db: sha256_hex(f"{db}{seed}".encode()))
    if len(candidates) < count:
        raise ValueError(
            f"val needs {count} databases with rows, but only {len(candidates)} of the "
            f"{len(databases)} training databases have any"
        )
    return frozenset(candidates[:count])


def split_examples(examples: Sequence[Example]) -> dict[Split, list[Example]]:
    """Spider's examples, labeled train or dev as ``load_raw()`` gives them, as three splits.

    Each split keeps the input order, and val's examples are relabeled ``val``.
    """
    databases: dict[Split, set[str]] = {name: set() for name in SPLITS}
    for example in examples:
        databases[example.split].add(example.db_id)
    if databases["val"]:
        raise ValueError("these examples are already split: some are labeled val")
    shared = databases["train"] & databases["dev"]
    if shared:
        raise SplitLeak(f"databases in both Spider train and dev: {', '.join(sorted(shared))}")

    val_dbs = val_databases(databases["train"])
    splits: dict[Split, list[Example]] = {name: [] for name in SPLITS}
    for example in examples:
        if example.db_id in val_dbs:
            splits["val"].append(example.model_copy(update={"split": "val"}))
        else:
            splits[example.split].append(example)
    return splits


def load_train(data_dir: Path | None = None) -> list[Example]:
    """The train split: what training code may load, and all it may load.

    It also passes ``assert_train_only``, so a changed seed or rule can't move a
    committed val or dev database into training before the manifests are rewritten.
    """
    train = split_examples(load_raw(data_dir))["train"]
    assert_train_only(train)
    return train


def load_val(data_dir: Path | None = None) -> list[Example]:
    """The val split, for checkpoint selection and tuning, never for training."""
    return split_examples(load_raw(data_dir))["val"]


def load_dev(*, reason: str, run_id: str, data_dir: Path | None = None) -> list[Example]:
    """Spider dev, for reported numbers only.

    Each call is one of the project's six dev evaluations. It is logged with its reason
    in ``docs/dev-usage.md`` (``FORGE_DEV_USAGE`` overrides the path), and the seventh
    raises ``DevBudgetExceeded``. The data loads before the row is written, so a missing
    download doesn't use up a look.
    """
    if not reason or not reason.strip():
        raise ValueError("load_dev needs a reason: every dev evaluation is logged, six at most")
    dev = split_examples(load_raw(data_dir))["dev"]
    log_dev_use(run_id, reason)
    return dev


def assert_train_only(examples: Iterable[Example]) -> None:
    """Raise ``SplitLeak`` unless every example is a train example from a train database.

    Training code calls this on what it is about to train on. The database check, against
    the committed manifests, also catches val and dev questions that never went through
    the split, such as ``load_raw()``'s, which labels val's questions train.
    """
    held_out = _held_out_databases()
    leaks = [e.id for e in examples if e.split != "train" or e.db_id in held_out]
    if leaks:
        shown = ", ".join(leaks[:5]) + (", ..." if len(leaks) > 5 else "")
        raise SplitLeak(f"{len(leaks)} val or dev example(s) passed to training: {shown}")


@cache
def _held_out_databases() -> frozenset[str]:
    return frozenset(db_id for name in ("val", "dev") for _, db_id in read_split_ids(name))


def ids_file(name: Split) -> str:
    return f"{name}.ids.txt"


def render_manifests(splits: Mapping[Split, Sequence[Example]]) -> dict[str, str]:
    """The manifest files for these splits, by file name: ids in split order, then the hashes."""
    texts = {
        ids_file(name): "".join(f"{e.id}\t{e.db_id}\n" for e in splits[name]) for name in SPLITS
    }
    manifest = SplitManifest(
        seed=VAL_SEED,
        val_percent=VAL_PERCENT,
        files={name: sha256_hex(text.encode()) for name, text in texts.items()},
    )
    return {**texts, MANIFEST_FILE: manifest.model_dump_json(indent=2) + "\n"}


def write_manifests(splits: Mapping[Split, Sequence[Example]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in render_manifests(splits).items():
        path = out_dir / name
        # Bytes, not text mode, so the files hash the same on every platform.
        path.write_bytes(text.encode())
        written.append(path)
    return written


def read_manifest(directory: Traversable = MANIFEST_DIR) -> SplitManifest:
    return SplitManifest.model_validate_json((directory / MANIFEST_FILE).read_bytes())


def read_split_ids(name: Split, directory: Traversable = MANIFEST_DIR) -> list[tuple[str, str]]:
    """A committed split's ``(example id, db_id)`` pairs, once its sha256 checks out."""
    filename = ids_file(name)
    data = (directory / filename).read_bytes()
    if sha256_hex(data) != read_manifest(directory).files.get(filename):
        raise ValueError(
            f"{filename} does not match its sha256 in {MANIFEST_FILE}; "
            "regenerate both with `forge data splits --write`"
        )
    pairs = []
    for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        example_id, _, db_id = line.partition("\t")
        if not example_id or not db_id or "\t" in db_id:
            raise ValueError(f"{filename} line {number} is not '<id><TAB><db_id>': {line!r}")
        pairs.append((example_id, db_id))
    return pairs
