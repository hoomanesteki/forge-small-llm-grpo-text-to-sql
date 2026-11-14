"""Train, val and dev, split by database so no schema shows up in two of them.

``dev`` is Spider's official dev set. ``val`` is 10% of the training databases, rounded
up, with all of their questions: the first ones when the databases are sorted by
``sha256(db_id + VAL_SEED)``, so it is the same on every run and every machine.
Databases without rows are passed over (ADR-0004). ``train`` is the rest.

Training code calls ``load_train()`` and nothing else here. ``load_dev()`` needs a
reason and spends one of the project's six looks at dev.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import get_args

from forge.core.hashing import sha256_hex
from forge.core.types import Example, Split
from forge.data.spider import ROWLESS_DBS, load_raw
from forge.ops.dev_usage import log_dev_use

SPLITS: tuple[Split, ...] = get_args(Split)

# Changing either one moves databases between train and val, which changes every val
# number and the data manifest hash: bump the seed's version rather than reuse a seed.
VAL_SEED = "forge-val-v1"
VAL_PERCENT = 10


class SplitLeak(ValueError):
    """A database or an example is in a split it must not be in."""


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
    """The train split: what training code may load, and all it may load."""
    return split_examples(load_raw(data_dir))["train"]


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
