# Progress

One table per phase. Status is one of: todo, in progress, done, blocked.

## Phase 0: Foundation (v0.0.1)

| ID | Title | Pts | Status | PR | Run ids | Notes |
|---|---|---|---|---|---|---|
| FG-001 | Repo scaffold and toolchain | 3 | done |  |  |  |
| FG-002 | CI and branch protection | 3 | done |  |  |  |
| FG-003 | Docs skeleton and contributor rules | 2 | done |  |  |  |
| FG-004 | Backlog import and traceability scripts | 5 | done |  |  |  |
| FG-005 | Accounts runbook and `forge doctor` | 3 | done |  |  |  |
| FG-006 | Compute ledger, dev-usage log and manifest builder | 2 | done |  |  |  |

## Phase 1: Data and SQL environment (v0.1.0)

| ID | Title | Pts | Status | PR | Run ids | Notes |
|---|---|---|---|---|---|---|
| FG-101 | Spike: data sources, licenses and official evaluator | 2 | done |  |  | Pinned HF zip, CC BY-SA 4.0, evaluator vendored read-only with a timeout, 5-db fixture |
| FG-102 | Core types and hashing | 2 | done |  |  | Contracts as in 3.5, UTC-dated run ids, `build_manifest` returns a `RunManifest` |
| FG-103 | Spider loader, checksums and fixture subset | 3 | done |  |  | Fetch from a pinned zip, 170 files checked against `spider.sha256`, 9,693 examples, 60-question fixture, real data hash |
| FG-104 | Splits by database and dev guard | 3 | done |  |  | val is 15 of 146 train dbs (872 questions) by hashed `db_id`, committed id manifests, `assert_train_only`, `load_dev` logs each use, ADR-0004 |
| FG-105 | Schema serializer and prompt builder | 5 | done |  |  | One `CREATE TABLE` line per table from `sqlite_master`, p2 sample rows, `fit_prompt` overflow policy, `forge data prompt-stats` |
| FG-106 | Safe SQL executor | 5 | done |  |  | Read-only immutable URI, default-deny authorizer, deadline every 1,000 VM steps, row cap, statements counted before execution, spawn pool, `forge sql run`; every train and val gold query runs except the 3 known failures |
| FG-107 | SQL extraction and result comparator | 3 | done |  |  | Last sql block, else last SELECT/WITH line, think blocks dropped; multiset or list rows, 6-decimal floats, column orders up to 6 columns; top-level ORDER BY by sqlglot with a regex fallback |
| FG-108 | Official evaluator integration and agreement check | 5 | done |  |  | `official_exec_match` calls `eval_exec_match` in a worker thread, refusing empty, comment-only and denied SQL; sqlglot mutations; gold 100% on val with both checkers; agreement 99.15% on 2,000 val mutations (`docs/audit/agreement-2026-11-14.md`), all 17 misses from DISTINCT removal; ADR-0008, `scripts/smoke/phase_1.sh` |

## Phase 2: Evaluation harness (v0.2.0)

| ID | Title | Pts | Status | PR | Run ids | Notes |
|---|---|---|---|---|---|---|
| FG-201 | Inference backend protocol and CPU backend | 3 | todo |  |  |  |
| FG-202 | Eval runner | 5 | todo |  |  |  |
| FG-203 | Metrics with confidence intervals and paired tests | 5 | todo |  |  |  |
| FG-204 | Error taxonomy | 3 | todo |  |  |  |
| FG-205 | Report generator | 3 | todo |  |  |  |
| FG-206 | Execution-guided retry | 3 | todo |  |  |  |
| FG-207 | CPU smoke pipeline in CI | 3 | todo |  |  |  |

## Phase 3: Free-GPU ops and baselines (v0.3.0)

| ID | Title | Pts | Status | PR | Run ids | Notes |
|---|---|---|---|---|---|---|
| FG-301 | Spike: Kaggle, Unsloth and vLLM on a T4 | 3 | todo |  |  |  |
| FG-302 | Code and data packaging as Kaggle datasets | 3 | todo |  |  |  |
| FG-303 | Kaggle job runner CLI | 5 | todo |  |  |  |
| FG-304 | Hub run store and resume | 5 | todo |  |  |  |
| FG-305 | Time guard and automatic ledger rows | 3 | todo |  |  |  |
| FG-306 | vLLM inference backend | 3 | todo |  |  |  |
| FG-307 | GPU baselines and model choice | 3 | todo |  |  |  |

## Phase 4: Supervised fine-tuning (v0.4.0)

| ID | Title | Pts | Status | PR | Run ids | Notes |
|---|---|---|---|---|---|---|
| FG-401 | SFT dataset builder | 3 | todo |  |  |  |
| FG-402 | Training backend abstraction and configs | 5 | todo |  |  |  |
| FG-403 | SFT trainer with CPU smoke | 3 | todo |  |  |  |
| FG-404 | SFT runs on Kaggle | 3 | todo |  |  |  |
| FG-405 | SFT results and comparison | 2 | todo |  |  |  |

## Phase 5: Rewards and GRPO (v0.5.0)

| ID | Title | Pts | Status | PR | Run ids | Notes |
|---|---|---|---|---|---|---|
| FG-501 | Reward functions | 5 | todo |  |  |  |
| FG-502 | Adversarial reward tests | 3 | todo |  |  |  |
| FG-503 | Parallel reward computation and gold cache | 3 | todo |  |  |  |
| FG-504 | GRPO trainer with CPU smoke | 5 | todo |  |  |  |
| FG-505 | GRPO dataset, filtering and curriculum | 3 | todo |  |  |  |
| FG-506 | Training telemetry and alerts | 3 | todo |  |  |  |
| FG-507 | GRPO runs on Kaggle | 5 | todo |  |  |  |
| FG-508 | Checkpoint selection and dev evaluation | 3 | todo |  |  |  |

## Phase 6: Integrity and ablations (v0.6.0)

| ID | Title | Pts | Status | PR | Run ids | Notes |
|---|---|---|---|---|---|---|
| FG-601 | Reward-hacking audit tooling | 5 | todo |  |  |  |
| FG-602 | Fix confirmed hacks, regression tests and a short re-run | 3 | todo |  |  |  |
| FG-603 | Ablations on the small model | 5 | todo |  |  |  |
| FG-604 | Seed variance | 2 | todo |  |  |  |
| FG-605 | Out-of-domain check on BIRD mini-dev (Could) | 3 | todo |  |  |  |
| FG-606 | Optional boost run with Modal free credits (Could) | 3 | todo |  |  |  |

## Phase 7: Packaging and demo (v0.7.0)

| ID | Title | Pts | Status | PR | Run ids | Notes |
|---|---|---|---|---|---|---|
| FG-701 | Merge and publish with a model card | 3 | todo |  |  |  |
| FG-702 | GGUF export and quantization check | 5 | todo |  |  |  |
| FG-703 | Local CLI and Ollama Modelfile | 3 | todo |  |  |  |
| FG-704 | Hugging Face Space demo on free CPU | 3 | todo |  |  |  |
| FG-705 | Cost and latency table | 2 | todo |  |  |  |

## Phase 8: Results and release (v1.0.0)

| ID | Title | Pts | Status | PR | Run ids | Notes |
|---|---|---|---|---|---|---|
| FG-801 | Results page generator | 3 | todo |  |  |  |
| FG-802 | README, docs and prose lint | 3 | todo |  |  |  |
| FG-803 | Fresh-clone reproducibility check | 3 | todo |  |  |  |
| FG-804 | Final review and release v1.0.0 | 2 | todo |  |  |  |

## Found

Work discovered along the way that is not a plan story.

| ID | Title | Status | Notes |
|---|---|---|---|
| FG-091 | Gate 0 fixes: run log hardening | done | Locked appends, strict ledger fields, strict repo root, hashing refuses non-JSON |
| FG-092 | Gate 0 fixes: doctor and .env | done | `six` for the Kaggle CLI, .env reaches the CLI, file tokens masked |
| FG-093 | Gate 0 fixes: backlog labels and test checks | done | GPU spike labeled GPU (11), phase and skip marks checked, gitignore |

- GPU packages are pinned outside `uv.lock`, in `kaggle/requirements-gpu.txt` (ADR-0002, FG-301).
- `core/hashing.py` landed with FG-006 because the manifest builder needs it; FG-102 adds its tests.
- `Prompt` is frozen but not hashable, and its message dicts can still be edited in place,
  because the contract types `messages` as a tuple of dicts. Key prompt caches by
  `(example_id, prompt_version)` and treat messages as read-only (FG-105).
- `RunKind` (`core/ids.py`) and `Platform` (`ops/manifest.py`) repeat the `RunManifest`
  literals so callers get type checking; unit tests fail if they drift apart.
- `build_manifest` refuses a naive `now`, and run ids take their date from the UTC start
  time, so a Kaggle run and a laptop run started together get the same date.
- CI runs on every pushed branch as well as on PRs, so a branch is checked before its PR opens.
- The six `train_others` databases and `music_2` have no rows: their gold results are empty or a
  single 0 or NULL. Keep them out of val (FG-104) and out of GRPO (FG-505); SFT can use them.
- Training golds 3153, 4513 and 4514 fail on SQLite, and the official evaluator asserts on a
  failing gold. FG-103 flags them; FG-108 skips them.
- The evaluator's `asyncio.run` fails inside a running event loop (notebook cells), so FG-108
  calls it from a script or a worker thread.
- Headline test-suite EX needs `testsuitedatabases.zip` (about 1.2 GB, Google Drive) in FG-508.
- Only `wta_1` (dev) has text that is not valid UTF-8, so FG-106 builds its own test database.
- `data_manifest_hash()` now loads and hashes the data, so `build_manifest` needs
  `FORGE_DATA_DIR` or `data/spider`; tests use the `spider_mini` fixture in `tests/conftest.py`.
  The full and fixture hashes are pinned in tests, and FG-104 updates both when it adds val.
- The three failing training golds are `FAILING_GOLD_IDS` in `forge.data.spider`, for FG-108.
- The Drive source is checked against the HAL-9001 mirror's sha256 (same size and central
  directory). Nobody has downloaded the Drive file in full to confirm it.
- Fixture ids are positions in the fixture files, not Spider's: `spider-train-00010` there is
  Spider's item 343. Its README lists the original indices.
- The val rule picks `flight_1` from the fixture as well as from the full data, so the
  fixture's val is a real val database. The fixture now has 20 `flight_1` questions and 10
  for each other database, still 60 in all.
- Under `forge-val-v1` no row-less database ranks in the first 15, so skipping them changes
  nothing today; it only matters if the seed changes. No val database holds a failing gold.
- Most val questions come in pairs that share a gold query: 872 questions, 456 distinct
  gold queries. Confidence intervals (FG-203) should allow for the pairs.
- `load_train()` checks its examples against the committed manifests, so a seed or rule
  change that moves a val or dev database into train needs `forge data splits --write`
  (full data) before training will load.
- Training code should call `assert_train_only()` on the dataset it builds (FG-401,
  FG-505). The static check also bans `load_raw` and `split_examples` under `forge/train/`.
- Seven Spider tables (in bike_1, college_2, flight_4, icfp_1 and sakila_1) have `--` comments
  in their DDL. Schema text drops comments, since on one line they would swallow the rest of the
  statement; flight_4's column notes go with them. All 873 normalized statements rebuild their
  tables with the same columns, keys and foreign keys (`tests/integration/test_schema.py`).
- `forge.data.schema` opens databases read-only through its own small helper. FG-106 moves it into
  `forge.sql.executor`.
- `forge data prompt-stats` with the Qwen2.5 tokenizer (from `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`):
  at p1 only baseball_1's 82 train questions pass 1,536 tokens, and dropping columns brings them
  to 530 or fewer. At p2, 1,215 train, 246 val and 160 dev prompts pass 1,536 and lose their
  sample rows, so for those p2 is p1. FG-307 should keep that in mind when comparing the two.
  Nothing is excluded from training at either version.
- An excluded example keeps its shortest prompt in `FittedPrompt.prompt` and stays in the eval
  denominator. FG-202 decides whether eval sends that prompt or the full one.
- `prompt-stats` reads dev questions and schemas without logging a dev use: it runs no model.
- SQLite runs `DROP TABLE|INDEX|VIEW|TRIGGER IF EXISTS` on a missing object as a no-op without
  asking the authorizer, so it would come back ok with no rows and match an empty gold result.
  The executor reports any statement without result columns as `denied`.
- A write that names a missing table fails with `no_such_table` before the authorizer is asked,
  so `denied` is not the only error a non-SELECT statement can get. FG-501's `non_select` guard
  should look at the SQL (sqlglot), not only at `error_kind`.
- Eight gold queries return more than the 10,000-row cap: baseball_1 train 03661-03664 (18,228
  and 16,049 rows), flight_4 val 06877-06878 (10,507) and wta_1 dev 00455-00456 (20,662). Their
  results come back truncated, so FG-107 has to decide how a truncated result compares, or
  FG-202/FG-503 run gold with a higher cap. Found by running gold SQL only, so not a dev use.
- The slowest gold query (baseball_1, `spider-train-03657`) takes 1.2 s on the laptop against the
  2 s default timeout. On a slower Kaggle CPU with busy workers it could time out, so the FG-503
  gold cache should run gold with a longer timeout.
- `run_many` starts a spawn pool per call (about a second of start-up), and a script that calls it
  needs `if __name__ == "__main__":`. FG-503 may want one long-lived pool; FG-303's Kaggle entry
  needs the guard.
- `SQLITE_FUNCTION` allows `randomblob` and `zeroblob`, which can allocate up to SQLite's 1 GB
  length limit in one VM step, before the deadline is checked. If reward workers run short of
  memory, lower `SQLITE_LIMIT_LENGTH` in `connect_readonly` (FG-503).
- The vendored evaluator still opens its own connections (read-only in `exec_eval`, read-write in
  `get_schema` and `isValidSQL`). The one-module rule test covers `src/forge` only; FG-108 decides
  how official EX opens databases.
- A truncated result never matches an untruncated one (with the same cap their row counts differ),
  and two truncated results compare on the rows kept, so the six train and val golds over the cap
  still match themselves. That is exact only for the same query: FG-202 and FG-503 should run gold,
  and predictions against those golds, with a cap above 20,662 rows.
- `order_matters` counts only a top-level ORDER BY. 34 of the 8,659 train and val golds have one only
  in a subquery, where the official evaluator's substring test says order matters. With rows shuffled
  and columns reordered, the two comparators disagreed on 21 of 8,620 golds, all of them among those
  34; on 8,650 random gold pairs from the same database they agreed every time. FG-108 measures this
  properly. Every train and val gold parses with sqlglot 30.31, so the regex fallback never runs on them.
- `extract_sql` takes one line when there is no sql block, as section 3.7 says, so an unfenced query
  split over several lines loses its tail. An unclosed `<think>` (a reply cut off while thinking) is
  not dropped either. FG-204 can count both if they show up.
- The official evaluator removes every `DISTINCT`, inside `COUNT(DISTINCT x)` too, where it becomes
  `count(x)`. All 17 disagreements in the 2,000-item val check come from that: 14 `COUNT(DISTINCT ...)`
  golds the evaluator matches to another column's distinct count, and 3 `SELECT DISTINCT` golds whose
  prediction has the same distinct rows over other duplicates. The fast comparator stays as it is
  (ADR-0008). The mutations never add or drop a `DISTINCT`, so FG-307 should watch that case on real
  outputs, and FG-601 should look for reward gains the official metric doesn't show.
- No disagreement in the 2,000 items came from `ORDER BY` detection. The mutations never reorder the
  rows of a correct answer on their own, though, so the 34 golds with an `ORDER BY` only in a subquery
  (FG-107 note above) are still the known gap there: on those, the same rows in another order match
  for the fast comparator and not for the evaluator. FG-307's check on real outputs will show it if
  models do that.
- Official EX opens databases through the vendored evaluator's own read-only connections, after
  `forge.sql.executor` has vetted the prediction for 100 ms (empty, comment-only and denied SQL is false
  without running). The one-module rule test still covers `src/forge` only.
- `eval_exec_match` compares on every file whose name contains `.sqlite` in the database's folder, so
  unpacking the test-suite databases beside Spider's turns classical EX into test-suite EX (FG-508).
- The evaluator's per-query timeout stays upstream's 60 s: it is a default argument bound when the module
  loads, so changing it means editing vendored code. `official_exec_match` adds an overall deadline
  (1 h by default) and stops its worker after the item it is on.
- sqlglot reads Spider's double-quoted strings (`"France"`) as names, so `change_literal` only changes
  numbers and single-quoted strings.
- Timings on the laptop: `forge eval gold --split val` about 10 s, `forge eval agreement --mutations 2000`
  about 20 s, `scripts/smoke/phase_1.sh` about 20 s. `make smoke` stays FG-207's.
- `forge eval gold --split dev` runs gold queries only, no model, so like `prompt-stats` it is not a dev use.
