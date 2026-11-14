"""The integrity invariants of plan section 4, run at every gate.

I1: no ``db_id`` is in more than one split, and training code raises on val or dev.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pytest

from forge.core.types import Example
from forge.data.spider import load_raw
from forge.data.splits import (
    MANIFEST_DIR,
    SPLITS,
    VAL_PERCENT,
    VAL_SEED,
    SplitLeak,
    assert_train_only,
    load_train,
    load_val,
    read_manifest,
    read_split_ids,
    render_manifests,
    split_examples,
    val_databases,
)


@pytest.mark.phase(1)
def test_i1_splits_disjoint_by_db(spider_mini: Path) -> None:
    # The committed manifests describe the full download, so CI checks them without it.
    manifest = read_manifest()
    assert (manifest.seed, manifest.val_percent) == (VAL_SEED, VAL_PERCENT)
    rows = {name: read_split_ids(name) for name in SPLITS}  # each file's sha256 is checked
    dbs = {name: {db_id for _, db_id in rows[name]} for name in SPLITS}
    for a, b in combinations(SPLITS, 2):
        assert not dbs[a] & dbs[b], f"{a} and {b} share databases"
    ids = [example_id for name in SPLITS for example_id, _ in rows[name]]
    assert len(ids) == len(set(ids)) == 9693
    sizes = {name: (len(rows[name]), len(dbs[name])) for name in SPLITS}
    assert sizes == {"train": (7787, 131), "val": (872, 15), "dev": (1034, 20)}
    # Val is exactly what the rule picks from all of Spider's training databases.
    assert val_databases(dbs["train"] | dbs["val"]) == dbs["val"]

    # The fixture, split fresh: the same holds, and every question is in one split.
    raw = load_raw()
    splits = split_examples(raw)
    fixture_dbs = {name: {e.db_id for e in splits[name]} for name in SPLITS}
    for a, b in combinations(SPLITS, 2):
        assert not fixture_dbs[a] & fixture_dbs[b], f"{a} and {b} share databases"
    assert sorted(e.id for split in splits.values() for e in split) == sorted(e.id for e in raw)
    assert load_train() == splits["train"]
    assert load_val() == splits["val"]
    assert fixture_dbs["val"] == {"flight_1"}
    assert len(splits["val"]) >= 20

    # Training refuses val and dev, by label and, for questions that skipped the split,
    # by database: load_raw() labels val's questions train.
    assert_train_only(load_train())
    dev_example = Example(
        id="spider-dev-00000",
        db_id="concert_singer",
        question="How many singers are there?",
        gold_sql="SELECT count(*) FROM singer",
        hardness="easy",
        split="dev",
    )
    relabeled = dev_example.model_copy(update={"split": "train"})
    for leaked in (load_val(), [dev_example], [relabeled], raw):
        with pytest.raises(SplitLeak, match="val or dev example"):
            assert_train_only(leaked)


@pytest.mark.phase(1)
@pytest.mark.requires_full_data
def test_i1_committed_manifests_match_the_full_data() -> None:
    fresh = render_manifests(split_examples(load_raw()))
    committed = {name: (MANIFEST_DIR / name).read_bytes().decode() for name in fresh}
    assert fresh == committed
