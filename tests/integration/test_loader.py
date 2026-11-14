"""The loader on the committed fixture subset, and on the full download where it exists."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from forge.data.manifest import data_manifest_hash
from forge.data.spider import (
    FAILING_GOLD_IDS,
    db_path,
    load_checksums,
    load_raw,
    raw_files,
    resolve_data_dir,
    verify_files,
)

FIXTURE_DBS = {"college_1", "department_management", "dorm_1", "flight_1", "geo"}
LEVELS = ("easy", "medium", "hard", "extra")

# data_manifest_hash of the full download. It changes only if the files, the ids or
# the hash recipe change, and any of those means runs before and after aren't comparable.
FULL_DATA_MANIFEST_HASH = "e10958d3b7a9490c15623285275d2840fbce551c32432fc62c0ad440cb06651e"


@pytest.mark.phase(1)
def test_fixture_subset_loads(spider_mini: Path) -> None:
    examples = load_raw()  # through FORGE_DATA_DIR
    assert examples == load_raw(spider_mini)

    assert [e.id for e in examples] == [f"spider-train-{i:05d}" for i in range(50)] + [
        f"spider-others-{i:05d}" for i in range(10)
    ]
    assert {e.split for e in examples} == {"train"}
    # flight_1, the fixture's val database, has 20 so smoke runs on val get 20 questions.
    assert Counter(e.db_id for e in examples) == {**dict.fromkeys(FIXTURE_DBS, 10), "flight_1": 20}
    assert {e.db_id for e in examples if e.id.startswith("spider-others-")} == {"geo"}

    # Question and gold SQL come from Spider's JSON unchanged, in file order.
    raw = json.loads((spider_mini / "train_spider.json").read_text(encoding="utf-8"))
    assert [(e.db_id, e.question, e.gold_sql) for e in examples[:50]] == [
        (item["db_id"], item["question"], item["query"]) for item in raw
    ]

    tables = json.loads((spider_mini / "tables.json").read_text(encoding="utf-8"))
    assert {table["db_id"] for table in tables} == FIXTURE_DBS
    assert all(db_path(db).is_file() for db in FIXTURE_DBS)


@pytest.mark.phase(1)
def test_hardness_assigned(spider_mini: Path) -> None:
    examples = load_raw()
    assert all(e.hardness in LEVELS for e in examples)

    # One of each level, readable from the SQL: a single filter is easy, ORDER BY over
    # three columns medium, GROUP BY with ORDER BY and LIMIT hard, INTERSECT extra.
    assert [examples[i].hardness for i in (0, 1, 7, 9)] == ["easy", "medium", "hard", "extra"]
    assert "INTERSECT" in examples[9].gold_sql

    # The fixture took a round of each level at a time (see its README): 5 of each for
    # flight_1, and 3, 3, 2 and 2 for the databases with 10 questions.
    by_db = {db: Counter(e.hardness for e in examples if e.db_id == db) for db in FIXTURE_DBS}
    assert by_db.pop("flight_1") == dict.fromkeys(LEVELS, 5)
    assert all(
        counts == {"easy": 3, "medium": 3, "hard": 2, "extra": 2} for counts in by_db.values()
    )


@pytest.mark.phase(1)
@pytest.mark.requires_full_data
def test_full_counts_match_manifest() -> None:
    root = resolve_data_dir()
    checksums = load_checksums()
    verify_files(root, checksums)
    # The data hash covers exactly the files that were verified.
    assert raw_files(root) == sorted(checksums)

    examples = load_raw(root)
    files = Counter(e.id.rsplit("-", 1)[0] for e in examples)
    assert files == {"spider-train": 7000, "spider-others": 1659, "spider-dev": 1034}
    assert Counter(e.split for e in examples) == {"train": 8659, "dev": 1034}

    dbs = {name: {e.db_id for e in examples if e.id.startswith(f"{name}-")} for name in files}
    assert len(dbs["spider-train"]) == 140
    assert dbs["spider-others"] == {"academic", "geo", "imdb", "restaurants", "scholar", "yelp"}
    train_dbs = dbs["spider-train"] | dbs["spider-others"]
    assert len(train_dbs) == 146
    assert len(dbs["spider-dev"]) == 20
    assert not train_dbs & dbs["spider-dev"]
    tables = json.loads((root / "tables.json").read_text(encoding="utf-8"))
    assert {table["db_id"] for table in tables} == train_dbs | dbs["spider-dev"]

    # The FG-101 spike's hardness mix; dev's is the split published for Spider dev.
    hardness = {s: Counter(e.hardness for e in examples if e.split == s) for s in ("train", "dev")}
    assert hardness["train"] == {"easy": 1983, "medium": 2999, "hard": 1922, "extra": 1755}
    assert hardness["dev"] == {"easy": 248, "medium": 446, "hard": 174, "extra": 166}

    failing = {e.id: e for e in examples if e.id in FAILING_GOLD_IDS}
    assert {i: e.db_id for i, e in failing.items()} == {
        "spider-train-03153": "assets_maintenance",
        "spider-train-04513": "document_management",
        "spider-train-04514": "document_management",
    }
    assert "Ref_Company_Types" in failing["spider-train-03153"].gold_sql
    assert all(
        failing[i].gold_sql.index("ORDER BY") < failing[i].gold_sql.index("INTERSECT")
        for i in ("spider-train-04513", "spider-train-04514")
    )

    assert data_manifest_hash(root) == FULL_DATA_MANIFEST_HASH
