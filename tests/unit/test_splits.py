"""The val rule, splitting by database, the manifests, and what training may load."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from forge.core.types import Example, Split
from forge.data import spider, splits
from forge.data.spider import FAILING_GOLD_IDS, ROWLESS_DBS, DataNotFound
from forge.data.splits import (
    MANIFEST_FILE,
    SPLITS,
    VAL_SEED,
    SplitLeak,
    load_dev,
    load_train,
    load_val,
    read_manifest,
    read_split_ids,
    split_examples,
    val_databases,
    write_manifests,
)
from forge.ops.dev_usage import MAX_DEV_USES, DevBudgetExceeded, read_uses

TRAIN_PACKAGE = Path(__file__).resolve().parents[2] / "src" / "forge" / "train"

# Everything that hands out val or dev questions: load_raw and split_examples count,
# since their output holds both.
FORBIDDEN_IN_TRAIN = frozenset({"load_dev", "load_raw", "load_val", "split_examples"})


def _example(i: int, db_id: str, split: Split = "train") -> Example:
    return Example(
        id=f"x-{i:03d}",
        db_id=db_id,
        question=f"q{i}",
        gold_sql="SELECT 1",
        hardness="easy",
        split=split,
    )


@pytest.fixture
def dev_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch dev-usage log, so no test spends the project's real dev budget."""
    path = tmp_path / "dev-usage.md"
    monkeypatch.setenv("FORGE_DEV_USAGE", str(path))
    return path


@pytest.fixture
def with_dev(spider_mini: Path, tmp_path: Path) -> Path:
    """The fixture with dorm_1 moved from train to dev, so dev isn't empty."""
    root = Path(shutil.copytree(spider_mini, tmp_path / "with_dev"))
    items = json.loads((root / "train_spider.json").read_text(encoding="utf-8"))
    (root / "train_spider.json").write_text(
        json.dumps([item for item in items if item["db_id"] != "dorm_1"]), encoding="utf-8"
    )
    (root / "dev.json").write_text(
        json.dumps([item for item in items if item["db_id"] == "dorm_1"]), encoding="utf-8"
    )
    return root


@pytest.mark.phase(1)
def test_val_split_deterministic(spider_mini: Path) -> None:
    dbs = [f"db_{i:03d}" for i in range(146)]
    picked = val_databases(dbs)

    # The rule spelled out: the first 10%, rounded up, by sha256(db_id + seed) hex.
    ranked = sorted(dbs, key=lambda db: hashlib.sha256(f"{db}{VAL_SEED}".encode()).hexdigest())
    assert picked == frozenset(ranked[:15])
    # Input order and repeats don't change it, and the seed is the only other input.
    assert val_databases(reversed(dbs + dbs)) == picked
    assert val_databases(dbs, seed="forge-val-v2") != picked
    # Always rounded up, so even the smallest set of databases gets a val database.
    sizes = {n: len(val_databases(dbs[:n])) for n in (2, 10, 11, 140, 146)}
    assert sizes == {2: 1, 10: 1, 11: 2, 140: 14, 146: 15}

    # A fresh interpreter with other hash seeds gets the same val split: nothing
    # depends on set order or PYTHONHASHSEED.
    val = load_val()
    code = "from forge.data.splits import load_val; print(' '.join(e.id for e in load_val()))"
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for hash_seed in ("1", "2")
    }
    assert outputs == {" ".join(e.id for e in val) + "\n"}
    assert val == load_val()
    assert {e.db_id for e in val} == {"flight_1"}
    assert len(val) == 20


@pytest.mark.phase(1)
def test_val_skips_databases_without_rows() -> None:
    # geo and imdb rank first or not, val takes the database that has rows.
    assert val_databases(["geo", "imdb", "dorm_1"]) == {"dorm_1"}
    with pytest.raises(ValueError, match="databases with rows"):
        val_databases(["geo", "imdb", "yelp"])
    with pytest.raises(ValueError, match="2 or more"):
        val_databases(["dorm_1"])


@pytest.mark.phase(1)
def test_split_examples_by_database() -> None:
    train_dbs = [f"db_{i:02d}" for i in range(11)]
    examples = [_example(i, train_dbs[i % 11]) for i in range(33)] + [
        _example(33 + i, "zoo", "dev") for i in range(3)
    ]
    splits = split_examples(examples)

    val_dbs = val_databases(train_dbs)
    assert len(val_dbs) == 2
    assert {e.db_id for e in splits["val"]} == val_dbs
    assert {e.db_id for e in splits["train"]} == set(train_dbs) - val_dbs
    assert [e.db_id for e in splits["dev"]] == ["zoo"] * 3
    assert {e.split for e in splits["val"]} == {"val"}
    # Every example is in exactly one split, each split in input order, and only the
    # split label of val's examples changed.
    ids = [e.id for e in examples]
    for split in splits.values():
        assert [e.id for e in split] == sorted((e.id for e in split), key=ids.index)
    assert sorted(e.id for split in splits.values() for e in split) == ids
    by_id = {e.id: e for e in examples}
    assert all(e.model_copy(update={"split": "train"}) == by_id[e.id] for e in splits["val"])


@pytest.mark.phase(1)
def test_split_refuses_a_database_in_train_and_dev() -> None:
    examples = [_example(0, "a"), _example(1, "b"), _example(2, "b", "dev")]
    with pytest.raises(SplitLeak, match="both Spider train and dev: b"):
        split_examples(examples)
    with pytest.raises(ValueError, match="already split"):
        split_examples([_example(0, "a"), _example(1, "b", "val")])


@pytest.mark.phase(1)
def test_load_train_refuses_databases_the_manifests_hold_out(
    spider_mini: Path, tmp_path: Path
) -> None:
    # A data directory whose train has concert_singer, a dev database in the committed
    # manifests: the fresh split calls it train, and load_train still refuses it.
    root = Path(shutil.copytree(spider_mini, tmp_path / "relabeled"))
    items = json.loads((root / "train_spider.json").read_text(encoding="utf-8"))
    for item in items:
        if item["db_id"] == "dorm_1":
            item["db_id"] = "concert_singer"
    (root / "train_spider.json").write_text(json.dumps(items), encoding="utf-8")
    (root / "database" / "concert_singer").mkdir()
    (root / "database" / "concert_singer" / "concert_singer.sqlite").touch()

    assert "concert_singer" in {e.db_id for e in split_examples(spider.load_raw(root))["train"]}
    with pytest.raises(SplitLeak, match="10 val or dev example"):
        load_train(root)
    assert {e.db_id for e in load_train()} == {
        "college_1",
        "department_management",
        "dorm_1",
        "geo",
    }


@pytest.mark.phase(1)
def test_load_dev_requires_reason(with_dev: Path, dev_log: Path) -> None:
    for reason in ("", "   ", "\n\t"):
        with pytest.raises(ValueError, match="needs a reason"):
            load_dev(reason=reason, run_id="run-0", data_dir=with_dev)
    with pytest.raises(ValueError, match="run id"):
        load_dev(reason="baseline", run_id=" ", data_dir=with_dev)
    with pytest.raises(DataNotFound):
        load_dev(reason="baseline", run_id="run-0", data_dir=with_dev / "missing")
    # Refused or failed calls spend nothing.
    assert not dev_log.exists()

    dev = load_dev(reason="FG-104 check", run_id="run-1", data_dir=with_dev)
    assert [e.id for e in dev] == [f"spider-dev-{i:05d}" for i in range(10)]
    assert {(e.db_id, e.split) for e in dev} == {("dorm_1", "dev")}
    assert not {e.db_id for e in load_train(with_dev) + load_val(with_dev)} & {"dorm_1"}
    assert [use[1:] for use in read_uses(dev_log)] == [("run-1", "FG-104 check")]

    # Every call is a look at dev, and the seventh is refused without a new row.
    for i in range(2, MAX_DEV_USES + 1):
        load_dev(reason="budget", run_id=f"run-{i}", data_dir=with_dev)
    with pytest.raises(DevBudgetExceeded):
        load_dev(reason="one more look", run_id="run-7", data_dir=with_dev)
    assert len(read_uses(dev_log)) == MAX_DEV_USES


@pytest.mark.phase(1)
def test_val_databases_have_rows_and_working_golds() -> None:
    val = read_split_ids("val")
    # Gold-vs-gold on val must reach 100% EX (the phase 1 demo), so no empty databases
    # and none of the golds that fail on SQLite.
    assert not {db_id for _, db_id in val} & ROWLESS_DBS
    assert not {example_id for example_id, _ in val} & FAILING_GOLD_IDS


@pytest.mark.phase(1)
def test_manifests_round_trip_and_catch_edits(spider_mini: Path, tmp_path: Path) -> None:
    fresh = split_examples(spider.load_raw())
    written = write_manifests(fresh, tmp_path)
    assert sorted(path.name for path in written) == sorted(
        [MANIFEST_FILE, *(f"{name}.ids.txt" for name in SPLITS)]
    )
    assert read_manifest(tmp_path).seed == VAL_SEED
    for name in SPLITS:
        assert read_split_ids(name, tmp_path) == [(e.id, e.db_id) for e in fresh[name]]
    assert (tmp_path / "val.ids.txt").read_bytes().startswith(b"spider-train-00010\tflight_1\n")

    # Moving one question to another database is caught by the file's hash.
    path = tmp_path / "val.ids.txt"
    path.write_bytes(path.read_bytes().replace(b"\tflight_1\n", b"\tdorm_1\n", 1))
    with pytest.raises(ValueError, match=r"val\.ids\.txt does not match its sha256"):
        read_split_ids("val", tmp_path)


def _forbidden_references(package: Path) -> list[str]:
    """``<file>:<line>: <name>`` for each reference to a forbidden loader under ``package``."""
    found: list[tuple[str, int, str]] = []
    for path in package.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.Attribute):
                name = node.attr
            elif isinstance(node, ast.alias):
                name = node.name.rpartition(".")[2]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                name = node.value  # getattr(module, "load_dev") and the like
            else:
                continue
            # A star import could bring any of them in unseen.
            is_star = isinstance(node, ast.alias) and name == "*"
            if name in FORBIDDEN_IN_TRAIN or is_star:
                found.append((path.relative_to(package).as_posix(), node.lineno, name))
    return [f"{rel}:{line}: {name}" for rel, line, name in sorted(found)]


@pytest.mark.phase(1)
def test_train_package_never_loads_val_or_dev(tmp_path: Path) -> None:
    # Each name exists, so a rename can't leave the scan looking for nothing.
    assert all(hasattr(splits, name) or hasattr(spider, name) for name in FORBIDDEN_IN_TRAIN)

    package = tmp_path / "train"
    (package / "rewards").mkdir(parents=True)
    sources = {
        "clean.py": (
            "from forge.data.splits import assert_train_only, load_train\n\n"
            "rows = load_train()\nassert_train_only(rows)\nshards = sorted(glob('*'))\n"
        ),
        "imported.py": "from forge.data.splits import load_dev as dev\n",
        "star.py": "from forge.data.splits import *\n",
        "called.py": "def f(examples):\n    return split_examples(examples)\n",
        "rewards/attribute.py": "import forge.data.splits\n\nrows = forge.data.splits.load_val()\n",
        "rewards/dynamic.py": "from forge.data import spider\n\nf = getattr(spider, 'load_raw')\n",
    }
    for rel, source in sources.items():
        (package / rel).write_text(source, encoding="utf-8")
    assert _forbidden_references(package) == [
        "called.py:2: split_examples",
        "imported.py:1: load_dev",
        "rewards/attribute.py:3: load_val",
        "rewards/dynamic.py:3: load_raw",
        "star.py:1: *",
    ]

    # The real package arrives in phase 4; until then there is nothing to scan, and
    # from then on this keeps val and dev out of it.
    assert _forbidden_references(TRAIN_PACKAGE) == []
