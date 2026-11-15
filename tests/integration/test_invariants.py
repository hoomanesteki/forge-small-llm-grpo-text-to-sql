"""The integrity invariants of plan section 4, run at every gate.

I1: no ``db_id`` is in more than one split, and training code raises on val or dev.
I2: running any SQL leaves every database file byte-identical, killed workers included, and
non-SELECT statements are denied.
I5: the fast comparator agrees with the official evaluator on at least 98% of predictions.
"""

from __future__ import annotations

import hashlib
import math
import shutil
from collections import Counter
from itertools import combinations
from pathlib import Path

import pytest
from test_suite_eval import exec_eval

from forge.core.types import ErrorKind, Example
from forge.data.spider import db_path, load_raw
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
from forge.eval.agreement import AGREEMENT_TARGET, agreement_mutations, agreement_rate, score
from forge.eval.mutations import CHANGES
from forge.eval.testsuite import official_exec_match
from forge.sql import sandbox
from forge.sql.executor import SqliteExecutor
from forge.sql.sandbox import SandboxPool
from tests.sandbox_jobs import review_query, run_with_fg191_limits


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


# I2 query templates: {t} is a table, {c} its first column, {scratch} the working
# directory, where anything ATTACH or VACUUM INTO managed to write would appear.
I2_SELECTS = (
    "SELECT * FROM {t}",
    "SELECT count(*) FROM {t}",
    "SELECT {c}, count(*) FROM {t} GROUP BY {c} ORDER BY 2 DESC LIMIT 5",
    "SELECT DISTINCT {c} FROM {t} WHERE {c} IS NOT NULL;",
    "WITH s AS (SELECT {c} FROM {t}) SELECT count(*) FROM s",
    "WITH replace AS (SELECT {c} FROM {t}) SELECT count(*) FROM replace",
    "SELECT a.{c} FROM {t} AS a JOIN {t} AS b ON a.rowid = b.rowid LIMIT 3",
)
I2_DENIED = (
    "INSERT INTO {t} DEFAULT VALUES",
    "INSERT INTO {t} SELECT * FROM {t}",
    "REPLACE INTO {t} SELECT * FROM {t}",
    "UPDATE {t} SET {c} = NULL",
    "UPDATE {t} SET forge_missing = 1",
    "DELETE FROM {t}",
    "DELETE FROM forge_missing",
    "INSERT INTO forge_missing VALUES (1)",
    "DELETE FROM {t} WHERE {c} = 'unclosed",
    "WITH s AS (SELECT 1) DELETE FROM {t}",
    "WITH s AS (SELECT 1) DELETE FROM forge_missing",
    # Keywords only SQLite's own reading finds (see statement_keyword).
    "\ufeffDELETE FROM forge_missing",
    "\ufeffEXPLAIN SELECT * FROM {t}",
    "WITH \u017felect AS (SELECT 1) DELETE FROM forge_missing",
    "WITH s AS (SELECT $a(() ) DELETE FROM forge_missing",
    "WITH s AS (SELECT @a(') ) UPDATE {t} SET forge_missing = 1 -- '",
    "DROP TABLE {t}",
    "DROP TABLE IF EXISTS forge_missing",
    "DROP VIEW IF EXISTS forge_missing",
    "ALTER TABLE {t} RENAME TO forge_gone",
    "ALTER TABLE {t} ADD COLUMN forge_extra TEXT",
    "ALTER TABLE {t} DROP COLUMN forge_missing",
    "DROP INDEX forge_missing",
    "CREATE TABLE forge_new AS SELECT * FROM {t}",
    "CREATE TEMP TABLE forge_new AS SELECT * FROM {t}",
    "CREATE INDEX forge_index ON {t} ({c})",
    "CREATE VIEW forge_view AS SELECT * FROM {t}",
    "CREATE TRIGGER forge_trigger AFTER DELETE ON {t} BEGIN SELECT 1; END",
    "ATTACH DATABASE '{scratch}/attached.sqlite' AS evil",
    "ATTACH DATABASE 'relative.sqlite' AS evil",
    "ATTACH DATABASE 'file:{scratch}/uri.sqlite?mode=rwc' AS evil",
    "ATTACH DATABASE CASE WHEN 1 > = 1 THEN '{scratch}/case.sqlite' END AS evil",
    "DETACH DATABASE main",
    "PRAGMA writable_schema = 1",
    "PRAGMA journal_mode = WAL",
    "PRAGMA user_version = 42",
    "PRAGMA query_only = 0",
    "PRAGMA table_info({t})",
    "VACUUM",
    "VACUUM INTO '{scratch}/copy.sqlite'",
    "REINDEX {t}",
    "ANALYZE {t}",
    "BEGIN IMMEDIATE",
    "SAVEPOINT forge",
    "EXPLAIN SELECT * FROM {t}",
    "SELECT * FROM {t}; DELETE FROM {t}",
    "SELECT 1; DROP TABLE {t}",
)
I2_JUNK: tuple[tuple[str, ErrorKind], ...] = (
    ("", "empty_sql"),
    ("-- DELETE FROM {t}", "empty_sql"),
    ("SELEC * FORM {t}", "syntax"),
    (" \v", "empty_sql"),
    ("SELECT * FROM {t} WHERE {c} = 'unclosed", "syntax"),
    ("SELECT * FROM forge_missing", "no_such_table"),
    ("SELECT forge_missing FROM {t}", "no_such_column"),
    ("SELECT * FROM {t} WHERE {c} = ?", "other"),
    ("\x00DROP TABLE {t}", "other"),
)


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@pytest.mark.phase(1)
def test_i2_db_files_unchanged(
    spider_mini: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    before = _file_hashes(spider_mini / "database")
    assert len(before) == 5

    with SqliteExecutor(spider_mini) as executor:
        # Every table of every fixture database, with its first column.
        targets: list[tuple[str, str, str]] = []
        for rel in before:
            db_id = rel.split("/")[0]
            listing = executor.run(db_id, "SELECT name FROM sqlite_master WHERE type = 'table'")
            for (table,) in listing.rows or ():
                columns = executor.run(db_id, f"SELECT * FROM {_quote(table)} LIMIT 0").columns
                assert columns, (db_id, table)
                targets.append((db_id, _quote(table), _quote(columns[0])))
        assert len(targets) == 26

        templates: list[tuple[str, ErrorKind | None]] = [
            *((sql, None) for sql in I2_SELECTS),
            *((sql, "denied") for sql in I2_DENIED),
            *I2_JUNK,
        ]
        items: list[tuple[str, str]] = []
        expected: list[ErrorKind | None] = []
        # The counts share no factor, so each template meets every table in turn.
        assert math.gcd(len(templates), len(targets)) == 1
        for i in range(1000):
            template, kind = templates[i % len(templates)]
            db_id, table, column = targets[i % len(targets)]
            items.append((db_id, template.format(t=table, c=column, scratch=tmp_path)))
            expected.append(kind)

        results = executor.run_many(items, workers=2)

    got = [None if result.ok else result.error_kind for result in results]
    mismatches = [
        (items[i], expected[i], results[i]) for i in range(len(items)) if got[i] != expected[i]
    ]
    assert not mismatches, mismatches[:5]
    counts = Counter(got)
    assert counts[None] > 100 and counts["denied"] > 600, counts

    # Workers killed in the middle of a query, one on each database, leave no trace either.
    databases = sorted({db_id for db_id, _, _ in targets})
    monkeypatch.setattr(sandbox, "_run_job", run_with_fg191_limits)
    with SandboxPool(len(databases), data_dir=spider_mini) as pool:
        killed = pool.run_many([(db_id, review_query(100)) for db_id in databases], timeout_ms=50)
        assert all((r.error or "").startswith("killed after") for r in killed), killed
        assert pool.restarts == len(databases)

    assert _file_hashes(spider_mini / "database") == before
    assert list(tmp_path.iterdir()) == []


# Predictions that write files unless every layer holds. {folder} is the database's own
# folder, where the evaluator runs gold on every .sqlite file, and {scratch} the working
# directory. Some are writes only after the evaluator's rewrites: postprocess turns '> ='
# into '>=', replace_cur_year turns YEAR(CURDATE()) into 2020, and remove_distinct drops
# DISTINCT and keeps only the first statement sqlparse finds.
I2_OFFICIAL = (
    "ATTACH DATABASE CASE WHEN 1 > = 1 THEN '{folder}/planted.sqlite' END AS e",
    "ATTACH DATABASE '{scratch}/' || YEAR(CURDATE()) || '.sqlite' AS e",
    "ATTACH DATABASE '{folder}/attached.sqlite' AS e",
    "ATTACH DATABASE 'relative.sqlite' AS e",
    "DISTINCT ATTACH DATABASE '{folder}/hidden.sqlite' AS e",
    "DISTINCT ATTACH DATABASE 'file:{scratch}/uri.sqlite?mode=rwc' AS e",
    "VACUUM INTO '{folder}/copy.sqlite'",
    "DISTINCT VACUUM INTO '{scratch}/vacuum.sqlite'",
    "; VACUUM INTO '{scratch}/leading.sqlite'",
    "DISTINCT DELETE FROM {t}",
    "DISTINCT DROP TABLE {t}",
    "DISTINCT CREATE TABLE forge_new AS SELECT 1",
    "DELETE FROM forge_missing",
)
EMPTY_GOLD = "SELECT 1 WHERE 0"  # empty on every database, so a no-op would match


@pytest.mark.phase(1)
def test_i2_official_eval_leaves_files_unchanged(
    spider_mini: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A copy, so a regression can't plant files in the committed fixture.
    data = tmp_path / "spider"
    shutil.copytree(spider_mini, data)
    monkeypatch.setenv("FORGE_DATA_DIR", str(data))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.chdir(scratch)
    before = _file_hashes(data)

    items: list[tuple[str, str, str]] = []
    with SqliteExecutor(data) as executor:
        for folder in sorted((data / "database").iterdir()):
            listing = executor.run(
                folder.name, "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
            table = _quote((listing.rows or [("missing",)])[0][0])
            for template in I2_OFFICIAL:
                pred = template.format(folder=folder, scratch=scratch, t=table)
                items.append((folder.name, pred, EMPTY_GOLD))
    assert len(items) == 5 * len(I2_OFFICIAL)

    assert official_exec_match(items) == [False] * len(items)
    # The evaluator's own connections hold too, without Forge's checks in front of them.
    # Its verdicts are upstream's: it runs only the ';' of '; VACUUM INTO ...', which
    # matches the empty gold result. Only the files are checked here.
    for db_id, pred, gold in items:
        exec_eval.eval_exec_match(
            db=str(db_path(db_id, data)),
            p_str=pred,
            g_str=gold,
            plug_value=False,
            keep_distinct=False,
            progress_bar_for_each_datapoint=False,
        )

    assert _file_hashes(data) == before
    assert list(scratch.iterdir()) == []
    # Nothing was planted beside the databases, so gold still runs on every one of them.
    databases = sorted({db_id for db_id, _, _ in items})
    assert official_exec_match([(db, EMPTY_GOLD, EMPTY_GOLD) for db in databases]) == [True] * 5


@pytest.mark.phase(1)
def test_i5_fast_vs_official_agreement(spider_mini: Path) -> None:
    # On mutations of every fixture gold query for now; FG-307 adds stored model outputs.
    scores = score(agreement_mutations(load_raw(), 300, seed=0))

    disagreements = [(s.mutation.example_id, s.mutation.pred_sql) for s in scores if not s.agree]
    assert agreement_rate(scores) >= AGREEMENT_TARGET, disagreements
    assert {s.mutation.kind for s in scores} == {"unchanged", *CHANGES}
    # Both verdicts are common, so a checker that always says no couldn't pass.
    assert 0.25 < sum(s.official for s in scores) / len(scores) < 0.75
    assert all(s.fast and s.official for s in scores if s.mutation.kind == "unchanged")


@pytest.mark.phase(1)
@pytest.mark.requires_full_data
def test_i5_agreement_on_full_val() -> None:
    # The measurement behind docs/audit/agreement-*.md and ADR-0008.
    scores = score(agreement_mutations(load_val(), 2000, seed=0))

    assert agreement_rate(scores) >= AGREEMENT_TARGET
