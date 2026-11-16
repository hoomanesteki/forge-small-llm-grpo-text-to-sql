"""The integrity invariants of plan section 4, run at every gate.

I1: no ``db_id`` is in more than one split, and training code raises on val or dev.
I2: running any SQL leaves every database file byte-identical, killed workers included, and
non-SELECT statements are denied.
I4 (a run's report, for now): every number in it traces to the run's stored scores and
manifest.
I5: the fast comparator agrees with the official evaluator on at least 98% of predictions.
I6: the same greedy eval run twice agrees within 0.5 points of official EX.
I7: every GPU run kept here has a ledger row, and the ledger's cash stays within $50.
"""

from __future__ import annotations

import hashlib
import math
import re
import shutil
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from itertools import combinations
from pathlib import Path

import pytest
from test_suite_eval import exec_eval

from forge.core.types import ErrorKind, Example, GenParams, Prompt, RunManifest
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
from forge.eval.agreement import (
    AGREEMENT_TARGET,
    agreement_mutations,
    agreement_rate,
    score,
    score_agreement,
)
from forge.eval.config import EvalConfig
from forge.eval.mutations import CHANGES
from forge.eval.report import write_run_report
from forge.eval.reproducibility import Reproducibility, compare_runs
from forge.eval.run_files import read_run
from forge.eval.runner import run_eval
from forge.eval.testsuite import official_exec_match
from forge.infer.base import FakeBackend
from forge.ops.hub import LocalRunStore, open_store
from forge.ops.ledger import LedgerRow, append, audit, read_rows, record_run
from forge.ops.paths import ledger_path
from forge.sql import sandbox
from forge.sql.executor import SqliteExecutor
from forge.sql.sandbox import SandboxPool
from tests.report_trace import untraced
from tests.sandbox_jobs import review_query, run_with_fg191_limits

ROOT = Path(__file__).resolve().parents[2]


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


NO_SQL = "Let me think about it."
# SQLite can't parse it, but official EX drops DISTINCT, runs count(*) and matches.
COUNT_DISTINCT = "SELECT count(DISTINCT *) FROM Aircraft"
MISSPELT = "SELEC name, distance FROM Aircraft"
# I4's run: what the fake model answers first, and again when that query failed to run;
# None is the gold query, and so is a question not listed. The replies reach every kind of
# number a report holds: retries that fix a query, fail again or give up, one reply only
# official EX matches, and more failures of one class than a report shows.
I4_REPLIES: dict[str, tuple[str | None, str | None]] = {
    "spider-train-00010": (COUNT_DISTINCT, COUNT_DISTINCT),
    "spider-train-00012": (NO_SQL, None),
    "spider-train-00013": (MISSPELT, MISSPELT),
    "spider-train-00014": ("SELECT aid FROM Aircraft WHERE distance > 100", None),
    "spider-train-00016": ("SELECT count(*) FROM Aircraft WHERE distance > 1000", None),
    "spider-train-00018": ("SELECT name FROM Aircraft WHERE aid = 12", None),
    "spider-train-00023": ("SELECT name FROM Employees", NO_SQL),
    **dict.fromkeys([f"spider-train-{n:05d}" for n in (19, 20, 21, 22, 25, 26)], (NO_SQL, NO_SQL)),
}


@pytest.mark.phase(2)
def test_i4_report_numbers_trace_to_manifest(spider_mini: Path, tmp_path: Path) -> None:
    gold = {example.id: example.gold_sql for example in load_val()}

    def reply(prompt: Prompt, params: GenParams, sample_idx: int) -> str:
        first, again = I4_REPLIES.get(prompt.example_id, (None, None))
        # A retry turn adds the first reply and the error to the prompt's two messages.
        text = again if len(prompt.messages) > 2 else first
        if text is None:
            text = gold[prompt.example_id]
        return text if text == NO_SQL else f"```sql\n{text}\n```"

    runs = tmp_path / "runs"
    run = run_eval(
        EvalConfig(model="gold", backend="fake", retry=1, run_id="i4"),
        backend=FakeBackend(reply, model_ref="gold"),
        allow_dirty=True,
        runs_root=runs,
        cache_root=tmp_path / "cache",
        workers=1,
    )
    path = write_run_report("i4", runs_root=runs, now=datetime(2026, 11, 15, tzinfo=UTC)).report
    text = path.read_text(encoding="utf-8")
    # Every section that holds numbers is there to trace.
    for part in (
        "\n## Retry\n",
        "\nThe checkers disagree on 1 sample: ",
        "\n### no_sql: 5 of 7 shown\n",
        "\nThe first reply's query failed to run:\n",
    ):
        assert part in text, part
    assert untraced(text, run.paths.root, data_dir=spider_mini) == []

    # A number edited by hand, or one that comes from nowhere, is found wherever it is.
    title = text.split("\n", 1)[0]
    # The low end of the title's interval, a tenth of a point higher.
    raised = re.sub(r"\[(\d+\.\d)%", lambda m: f"[{float(m.group(1)) + 0.1:.1f}%", title, count=1)
    retried = "| retried: the first query failed to run |"
    for old, new in (
        (title, raised),
        ("| official EX | 9 | 20 |", "| official EX | 10 | 20 |"),
        ("| +5.0 [", "| +6.0 ["),
        (f"{retried} 10 |", f"{retried} 9 |"),
        ("\n| no_sql | 7 |", "\n| no_sql | 6 |"),
        ("### no_sql: 5 of 7 shown", "### no_sql: 5 of 8 shown"),
        ("\n## By hardness\n", "\n## By hardness\n\nEasy questions gained 12 points.\n"),
        ("\n| value |", "\n| best guess | 3 |\n| value |"),
        # The prose lines that hold numbers from the stored files.
        ("only fast EX matches 0, and only official EX 1.", "only fast EX matches 0, and 2."),
        ("20 questions in 12 clusters.", "20 questions in 11 clusters."),
        # The Generated line holds the time and nothing more.
        ("00:00:00 UTC\n", "00:00:00 UTC, official EX 87.5% on dev\n"),
        ("has an id of 12?", "has an id of 21?"),
        ("WHERE aid  =  12", "WHERE aid  =  21"),
    ):
        assert old in text, old
        assert untraced(text.replace(old, new), run.paths.root, data_dir=spider_mini), new

    # A run of several samples a question states how it sampled, and which sample failed.
    def sampled(prompt: Prompt, params: GenParams, sample_idx: int) -> str:
        return NO_SQL if sample_idx else f"```sql\n{gold[prompt.example_id]}\n```"

    several = run_eval(
        EvalConfig(model="gold", backend="fake", n=2, temperature=0.7, limit=6, run_id="i4-n2"),
        backend=FakeBackend(sampled, model_ref="gold"),
        allow_dirty=True,
        runs_root=runs,
        cache_root=tmp_path / "cache",
        workers=1,
    )
    text = write_run_report("i4-n2", runs_root=runs).report.read_text(encoding="utf-8")
    assert "\n| Generation | sampled at temperature 0.7, top-p 1, seed 0, at most 256" in text
    assert "\n### no_sql: 5 of 6 shown\n" in text and ", sample 1)\n" in text
    assert untraced(text, several.paths.root, data_dir=spider_mini) == []


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


@pytest.mark.phase(3)
def test_i5_agreement_on_committed_runs() -> None:
    # Real model replies, each with both verdicts stored: the FG-291 pilot for now, and
    # FG-307's baselines once they are committed beside it.
    store = LocalRunStore(ROOT / "docs" / "runs")
    rates = {run: score_agreement(read_run(store.root / run).scores) for run in store.runs()}
    assert rates, "docs/runs holds the FG-291 pilot"
    assert all(rate >= AGREEMENT_TARGET for rate in rates.values()), rates


def _repeat(
    config: EvalConfig, runs: Path, cache: Path, reply: Callable[[Prompt, GenParams, int], str]
) -> Path:
    run = run_eval(
        config,
        backend=FakeBackend(reply, model_ref=config.model),
        allow_dirty=True,
        runs_root=runs,
        cache_root=cache,
        workers=1,
    )
    return run.paths.root


@pytest.mark.phase(3)
def test_i6_greedy_eval_reproducible(
    spider_mini: Path, tiny_model_ref: str, tmp_path: Path
) -> None:
    runs, cache = tmp_path / "runs", tmp_path / "cache"
    val = sorted(load_val(), key=lambda example: example.id)
    gold = {example.id: example.gold_sql for example in val}
    wrong = {example.id for example in val[::3]}  # a third answered wrong, the same each run

    def reply(prompt: Prompt, params: GenParams, sample_idx: int) -> str:
        sql = "SELECT 1" if prompt.example_id in wrong else gold[prompt.example_id]
        return f"```sql\n{sql}\n```"

    # Two fixture runs of the fake backend: the same scores, and the first ten questions
    # alone repeat the start of the whole split.
    config = EvalConfig(model="gold", backend="fake")
    a = read_run(_repeat(replace(config, run_id="a"), runs, cache, reply))
    b = read_run(_repeat(replace(config, run_id="b"), runs, cache, reply))
    first = read_run(_repeat(replace(config, run_id="first", limit=10), runs, cache, reply))
    same = compare_runs(a, b)
    assert same.ok, same.describe()
    assert (same.samples, same.verdicts_differ, same.replies_differ) == (len(val), (), ())
    assert same.hits == (len(val) - len(wrong),) * 2
    start = compare_runs(first, a)
    assert (start.ok, start.samples) == (True, 10)

    # A run whose reply to one question changed is caught: one sample of 20 is 5 points.
    flipped = val[1].id
    assert flipped not in wrong

    def drifted(prompt: Prompt, params: GenParams, sample_idx: int) -> str:
        return (
            "```sql\nSELECT 2\n```"
            if prompt.example_id == flipped
            else reply(prompt, params, sample_idx)
        )

    c = read_run(_repeat(replace(config, run_id="c"), runs, cache, drifted))
    drift = compare_runs(a, c)
    assert not drift.ok and drift.gap_points == 100 / len(val)
    assert drift.verdicts_differ == drift.replies_differ == ((flipped, 0),)
    assert "5.0 points apart (I6 allows 0.5)" in drift.describe()
    # Only a repeat of the same eval compares: another prompt is another eval.
    p2 = read_run(_repeat(replace(config, run_id="p2", prompt_version="p2"), runs, cache, reply))
    with pytest.raises(ValueError, match="differ in prompt_version: a repeat changes only"):
        compare_runs(a, p2)
    # The tolerance holds at its edge: half a point is allowed, a whole one not.
    assert Reproducibility(("x", "y"), 200, (100, 101), (), ()).ok
    assert not Reproducibility(("x", "y"), 200, (100, 102), (), ()).ok

    # Two real CPU runs of the tiny model on the fixture's val split, greedy: the same
    # reply to every question, token for token.
    tiny = EvalConfig(model=tiny_model_ref, backend="hf")
    real = [
        read_run(
            run_eval(
                replace(tiny, run_id=run_id),
                allow_dirty=True,
                runs_root=runs,
                cache_root=cache,
                workers=1,
            ).paths.root
        )
        for run_id in ("tiny-a", "tiny-b")
    ]
    checked = compare_runs(*real)
    assert checked.ok, checked.describe()
    assert (checked.samples, checked.replies_differ) == (len(val), ())


def _gpu_run(run_id: str, platform: str, gpu: str | None) -> RunManifest:
    started = datetime(2026, 11, 20, 9, 0, tzinfo=UTC)
    return RunManifest(
        run_id=run_id,
        kind="eval",
        git_sha="c" * 40,
        config={"split": "val"},
        config_hash="4" * 64,
        data_manifest_hash="5" * 64,
        model_ref="Qwen/Qwen3-4B",
        platform=platform,  # type: ignore[arg-type]
        gpu=gpu,
        seed=0,
        started_at=started,
        ended_at=started + timedelta(minutes=30),
        status="finished",
    )


@pytest.mark.phase(3)
def test_i7_gpu_runs_in_ledger(tmp_path: Path) -> None:
    # The gate's check: every run committed under docs/runs or kept in the local runs
    # store, against the real ledger. No Kaggle run exists yet, so only the FG-291 pilot,
    # whose local-mps row must name its platform, is checked for now.
    stores = [LocalRunStore(ROOT / "docs" / "runs"), open_store("local")]
    kept = [store.get_manifest(run) for store in stores for run in store.runs()]
    assert kept, "docs/runs holds the FG-291 pilot"
    assert audit(kept, read_rows(ledger_path())) == []

    # On fixture runs, the same check catches what a gate must refuse.
    store, ledger = LocalRunStore(tmp_path / "runs"), tmp_path / "compute-ledger.md"
    kaggle = _gpu_run("2026-11-20-qwen3-4b-eval-abc1234", "kaggle", "Tesla T4")
    pilot = _gpu_run("2026-11-20-qwen2-5-coder-eval-abc1234", "local-mps", "Apple M4")
    laptop = _gpu_run("2026-11-20-tiny-eval-abc1234", "local-cpu", None)
    for manifest in (kaggle, pilot, laptop):
        store.put_manifest(manifest)
    found = [store.get_manifest(run) for run in store.runs()]
    # Only a run on a budget platform needs a row: a pilot's is welcome, not required.
    assert audit(found, read_rows(ledger)) == [
        f"run {kaggle.run_id} ran on kaggle, and the ledger has no row for it"
    ]
    # Pulling the run writes the row that settles it.
    row, _ = record_run(kaggle, ledger)
    assert row is not None and row.hours == 0.5
    assert audit(found, read_rows(ledger)) == []

    day = date(2026, 11, 20)
    append(LedgerRow(day, "kaggle", "Apple M4", 0.5, pilot.run_id, "mislabeled pilot"), ledger)
    append(LedgerRow(day, "modal", "L4", 2.0, "boost", "boost run", cost_usd=50.01), ledger)
    assert audit(found, read_rows(ledger)) == [
        f"run {pilot.run_id} ran on local-mps, but its ledger row of 2026-11-20 says kaggle",
        "the ledger spends $50.01, over the $50 limit",
    ]
