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
| FG-108 | Official evaluator integration and agreement check | 5 | done |  |  | `official_exec_match` calls `eval_exec_match` in a worker thread, refusing empty, comment-only and denied SQL; sqlglot mutations; gold 100% on val with both checkers; agreement 99.15% on 2,000 val mutations and 98.96% on the 1,445 distinct ones (`docs/audit/agreement-2026-11-15.md`), all 17 misses (15 distinct) from DISTINCT removal; ADR-0008, `scripts/smoke/phase_1.sh` |

## Phase 2: Evaluation harness (v0.2.0)

| ID | Title | Pts | Status | PR | Run ids | Notes |
|---|---|---|---|---|---|---|
| FG-201 | Inference backend protocol and CPU backend | 3 | done |  |  | `InferenceBackend`, `FakeBackend` and `render_chat` (thinking off where the template reads the flag); `HfBackend` decodes from `GenParams` alone in left-padded batches, seeds each sample by example and index, loads PEFT adapters; tiny model pinned in `configs/models.yaml` |
| FG-202 | Eval runner | 5 | done |  |  | `forge eval run` with the fake (gold, select-1) and hf backends; prompts fitted as in training; predictions in one `SandboxPool` a run; gold cached per split as JSON lines; official EX in one batch at the end, 10 s an item, a 2 s timeout no match (ADR-0008); a batch's generations then scores appended and flushed; a killed run resumes without duplicates; manifest running, then finished or failed with `error.txt` |
| FG-203 | Metrics with confidence intervals and paired tests | 5 | done |  |  | `summarize`, `compare` and `pass_at_k` in `forge.eval.metrics`; 95% percentile bootstrap, 1,000 resamples, `default_rng(0)` for each interval, drawing whole questions or `gold_clusters`; exact McNemar from statsmodels, with no p-value for several samples a question; official EX when every score has a verdict, else fast |
| FG-204 | Error taxonomy | 3 | done |  |  | `classify` in `forge.eval.taxonomy`: no_sql, syntax and schema_link from the error kind, then sqlglot shapes compare tables, joins, select lists, aggregates, WHERE, GROUP BY to LIMIT and last literals; names resolve to tables against each database's columns, a double-quoted name no table has is a string, and a nested filter query counts in its condition; `forge eval run` stores the class of each wrong prediction |
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
| FG-191 | Gate 1 fixes: sandbox safety | done | Statements denied by keyword before SQLite runs them, SQLite limits on every connection (the evaluator's too), a watchdog at the deadline, a 64 MiB result budget, `\v` counted as SQLite does; official EX vets what the evaluator runs and strips leading `;` |
| FG-192 | Gate 1 fixes: data and prompts | done | `load_dev` refuses an empty dev, `assert_train_only` refuses iterators, fetch won't overwrite other data and reports network errors, p2 samples keep their spaces, `[name]` quoted, fallback lines must parse; ADR-0004 says what counts as a dev use |
| FG-193 | Gate 1 fixes: hashing and reports | done | A model hashes like its `model_dump()`, manifests keep a JSON copy of the config and need a real UTC offset, `@` stays in local paths; the agreement report credits Spider (CC BY-SA 4.0) and counts distinct items, regenerated as `docs/audit/agreement-2026-11-15.md` |
| FG-194 | Gate 1 fixes: hard deadline | done | Untrusted SQL runs in `SandboxPool` workers that are killed 250 ms past their timeout, and each official EX item runs in one too; SQLite limits set from Spider's measured needs (4 kB values and rows, 8 kB SQL, 100-character LIKE patterns, 200 columns); result budgets count real memory, and the evaluator's fetch is bounded; ADR-0014 |
| FG-195 | Gate 1 fixes: review follow-ups | done | `forge data fetch` reports hf_xet's bare `RuntimeError` with the retry hint, `canonical_json` sorts keys as the strings JSON writes so a manifest's copy of an int-keyed config hashes like it, and a `/` after the `@` outside `refs/` makes a model ref a path; the FG-192 note on `extract_sql` no longer claims short sentences that parse as a SELECT are passed over |
| FG-196 | Gate 1 fixes: sandbox polish | done | A watchdog test that fails if the watchdog is removed (a query the progress handler never checks); a worker that dies under an official EX item counts as no match instead of aborting the run, and both eval CLIs catch `SandboxError`; a job whose result can't be pickled reports the type instead of killing its worker; the executor, ADR-0014 and progress narrowed to say the in-process few-hundredths bound was measured for four functions only and the item timeout covers the whole item (FG-508) |
| FG-197 | Gate 1 fixes: final polish | done | Official EX counts only a worker's death as a miss, so a pool that can't start raises again; a reply the parent can't unpickle keeps its worker; `statement_keyword` reads byte order marks, Tcl-style parameters, ASCII keywords and the `WITH` grammar as SQLite does; a forced Hub download that fails shows its cause, the bad zip and the hint; the timing tests assert the kill and use the worst rows the limits allow, gold is measured as the evaluator runs it, and ADR-0014 lists the known limits |

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
  calls it from a script or a worker thread. Since FG-194 it runs in a worker process.
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
  Since FG-192 `normalize_sql` also reads SQLite's `[name]` as quoted; no Spider table uses one.
- `forge.data.schema` opens databases read-only through its own small helper. FG-106 moves it into
  `forge.sql.executor`.
- `forge data prompt-stats` with the Qwen2.5 tokenizer (from `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`):
  at p1 only baseball_1's 82 train questions pass 1,536 tokens, and dropping columns brings them
  to 530 or fewer. At p2, 1,215 train, 246 val and 160 dev prompts pass 1,536 and lose their
  sample rows, so for those p2 is p1. FG-307 should keep that in mind when comparing the two.
  Nothing is excluded from training at either version. Since FG-192 the numbers are the same.
- 23 of those 82 fitted baseball_1 train prompts (ids between `spider-train-03629` and `03710`,
  at p1 and p2) show a table their gold SQL uses as a stub without columns, because the question
  doesn't name it: home_game in 10, player_college and salary in 4 each, park, player,
  player_award and team in 2 each. Found by comparing the tables sqlglot finds in each gold query
  with `FittedPrompt.dropped_columns`. The model can't see the columns it needs there, so FG-401
  and FG-505 should drop those examples or flag them.
- An excluded example keeps its shortest prompt in `FittedPrompt.prompt` and stays in the eval
  denominator. FG-202 decides whether eval sends that prompt or the full one. It sends that
  prompt: eval fits prompts with the backend's tokenizer, as training does.
- `prompt-stats` reads dev questions and schemas without logging a dev use: it runs no model.
  ADR-0004 now says what counts as a dev use, and `prompt-stats` and `forge eval gold` print a
  note saying so with `--split dev` (FG-192).
- SQLite runs `DROP TABLE|INDEX|VIEW|TRIGGER IF EXISTS` on a missing object as a no-op without
  asking the authorizer, so it would come back ok with no rows and match an empty gold result.
  The executor reports any statement without result columns as `denied`.
- A write that names a missing table or column fails on the name before SQLite asks the authorizer.
  Since FG-191 the executor denies by statement keyword first (`statement_keyword`), so every
  statement that isn't `SELECT`, `VALUES` or `WITH ... SELECT` gets `denied`, `EXPLAIN` included, and
  FG-501's `non_select` guard can rely on `error_kind`. Since FG-197 the keyword is read the way
  SQLite's tokenizer reads it: a byte order mark is whitespace where a token starts, a Tcl-style
  parameter such as `$a(...)` is one token, keywords are ASCII (Python upper-cases a long s to S),
  and a `WITH` clause follows its grammar. Writes behind any of those came back as `no_such_table`,
  `no_such_column` or `other`, a BOM even let `EXPLAIN` run, and a table named `replace` made a
  query `denied`.
- Eight gold queries return more than the 10,000-row cap: baseball_1 train 03661-03664 (18,228
  and 16,049 rows), flight_4 val 06877-06878 (10,507) and wta_1 dev 00455-00456 (20,662). Their
  results come back truncated, so FG-107 has to decide how a truncated result compares, or
  FG-202/FG-503 run gold with a higher cap. Found by running gold SQL only, so not a dev use.
  FG-202 runs gold, and predictions against it, with a 100,000-row cap, the evaluator's own.
- The slowest gold query (baseball_1, `spider-train-03657`) takes 1.2 s on the laptop against the
  2 s default timeout. On a slower Kaggle CPU with busy workers it could time out, so the FG-503
  gold cache should run gold with a longer timeout.
- `run_many` starts a spawn pool per call (about a second of start-up), and a script that calls it
  needs `if __name__ == "__main__":`. FG-503 may want one long-lived pool; FG-303's Kaggle entry
  needs the guard. Since FG-194 the pool is a `SandboxPool`, which starts in about 0.2 s.
- `SQLITE_FUNCTION` allows `randomblob` and `zeroblob`, which could allocate up to SQLite's 1 GB
  length limit in one VM step. FG-191 sets `SQLITE_LIMIT_LENGTH` to 100 kB in `connect_readonly`,
  and FG-194 to 4 kB.
- The vendored evaluator still opens its own connections (read-only in `exec_eval`, read-write in
  `get_schema` and `isValidSQL`). The one-module rule test covers `src/forge` only; FG-108 decides
  how official EX opens databases. Since FG-191 `exec_eval`'s connections have the executor's
  authorizer and limits; `isValidSQL` commits any DDL it is given and must never see a prediction.
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
  not dropped either. FG-204 can count both if they show up. Since FG-192 the line must also parse
  as a query with sqlglot, so most sentences opening with "With" or "Select" after the query no
  longer win; short ones that parse as a SELECT ("Select it", "Select this one", "select distinct
  values") still do, and FG-204 can count them too (FG-195). An unfenced query line that sqlglot
  can't parse gives `None`, not SQL that fails to run, but sqlglot is laxer than SQLite in places:
  it takes `SELECT name, FROM singer`, which the executor then reports as `syntax`. All 9,693 gold
  queries still come through on one line (`tests/integration/test_extract.py`).
- The official evaluator removes every `DISTINCT`, inside `COUNT(DISTINCT x)` too, where it becomes
  `count(x)`. All 17 disagreeing items in the 2,000-item val check (15 distinct) come from that: 14
  (13 distinct) with a `COUNT(DISTINCT ...)` gold the evaluator matches to another column's distinct
  count, and 3 (2 distinct) with a `SELECT DISTINCT` gold whose prediction has the same distinct rows
  over other duplicates. The fast comparator stays as it is
  (ADR-0008). The mutations never add or drop a `DISTINCT`, so FG-307 should watch that case on real
  outputs, and FG-601 should look for reward gains the official metric doesn't show.
- No disagreement in the 2,000 items came from `ORDER BY` detection. The mutations never reorder the
  rows of a correct answer on their own, though, so the 34 golds with an `ORDER BY` only in a subquery
  (FG-107 note above) are still the known gap there: on those, the same rows in another order match
  for the fast comparator and not for the evaluator. FG-307's check on real outputs will show it if
  models do that.
- Official EX opens databases through the vendored evaluator's own read-only connections, after
  `forge.sql.executor` has vetted the prediction for 100 ms (empty, comment-only and denied SQL is false
  without running). The one-module rule test still covers `src/forge` only. Since FG-191 it also vets
  the string the evaluator will run after its rewrites (`DISTINCT ATTACH ...` becomes an `ATTACH`).
- `eval_exec_match` compares on every file whose name contains `.sqlite` in the database's folder, so
  unpacking the test-suite databases beside Spider's turns classical EX into test-suite EX (FG-508).
- The evaluator's per-query timeout stays upstream's 60 s: it is a default argument bound when the module
  loads, so changing it means editing vendored code. `official_exec_match` adds an overall deadline
  (1 h by default) and stops its worker after the item it is on. Since FG-194 each item runs in a worker
  process that is killed after `item_timeout_s` (60 s by default), and the overall deadline kills the
  item it is on.
- sqlglot reads Spider's double-quoted strings (`"France"`) as names, so `change_literal` only changes
  numbers and single-quoted strings.
- Timings on the laptop: `forge eval gold --split val` about 10 s, `forge eval agreement --mutations 2000`
  about 20 s, `scripts/smoke/phase_1.sh` about 20 s. `make smoke` stays FG-207's.
- `forge eval gold --split dev` runs gold queries only, no model, so like `prompt-stats` it is not a dev use.
- A read-only connection still runs `ATTACH` and `VACUUM INTO`, which create the files they name.
  Before FG-191 a prediction that SQLite rejected for a syntax error, such as an `ATTACH` with `> =`
  that the evaluator then rewrote, reached the evaluator and could plant an empty `.sqlite` beside a
  database; every later official evaluation of it then raised `GoldQueryFailed`.
  `SQLITE_LIMIT_ATTACHED` 0 stops both on any connection.
- SQLite checks the progress handler and interrupts only between rows, at loop jumps, so the work
  inside one row can't be stopped: a row of 20 `instr` calls over 100 kB strings ran 1.6 s past a
  100 ms watchdog. The length limit bounds each call (the slowest measured, a `LIKE` with a 1,000-character
  pattern on 100 kB, takes about 0.2 s; at 1 MB one `instr` takes seconds), so only a statement that
  packs many such calls into one row still overruns. If reward workers ever see that, run them in
  processes that can be killed (FG-503). The watchdog thread costs about 40 µs per `run` (8 µs to
  52 µs for `SELECT 1` on the laptop). The FG-191 review found such rows: 100 `instr` calls on a 99 kB
  string ran 8.7 s against 300 ms, 200 calls 18.5 s against 2,000 ms. FG-194 runs untrusted SQL in
  workers that are killed at the deadline. Its limits keep each call cheap but don't bound a row:
  FG-197 measured 8 kB of compact calls reading their arguments from a column at 0.07 s of `instr`,
  0.34 to 0.38 s of `LIKE` or `GLOB`, 2.2 to 2.8 s of `ltrim` and 4.3 to 5.5 s of `unhex` on the
  laptop (Apple M4), and nested CTEs multiply calls past the text: 1,000 `unhex` calls in 379 bytes
  ran 6.5 s against 100 ms. The in-process deadline has no bound; only the pool kill is a hard one
  (ADR-0014). The FG-194 and FG-196 figures (0.06 s, then a second or two) came from rows with
  fewer or cheaper calls than the limits allow.
- SQLite reads `\v` as whitespace only inside a run that starts with one of space, tab, newline, form
  feed or return, and Python's check after the first statement never skips it. `statement_count`
  follows both rules; sqlite3 on 3.11.15 and 3.12.13 agrees on the cases in the unit tests.
- Re-running `forge eval agreement --mutations 2000 --split val --seed 0` on the FG-191 code gave the
  same report as the 2026-11-14 one apart from the date and commit. FG-193 regenerated it with the
  Spider attribution and distinct counts; the UTC date had moved on, so
  `docs/audit/agreement-2026-11-15.md` replaces it, with the same counts over all items.
- sqlglot 30.31 raises `RecursionError`, not a `SqlglotError`, on about 47 nested parentheses
  (`SELECT ((((...1))))`). Gold queries never nest that deep, but model output can: `extract_sql`
  treats it as no query, and any later code that parses model output with sqlglot (FG-204, FG-501,
  FG-601) must catch it too.
- p2 sample values had every run of whitespace collapsed, so 37 stored values in 11 databases
  showed as values that don't exist, among them flight_2's (dev) airport codes, stored with a
  leading space (`' APG'`) in `flights` and without one in `airports`. Since FG-192 only line breaks
  become spaces.
- huggingface-hub 2.4 raises its HTTP errors as `OSError`s, but a connection that still fails after
  its retries raises a bare `httpx2` error, which is not one. `forge data fetch` catches both, so
  `httpx2` is now a declared dependency. Both pinned Hub zips are stored on Xet (their resolve URLs
  answer with an `x-xet-hash`), so their bytes come through hf_xet 1.8, which reports a failed
  transfer as a bare `RuntimeError` (`Task error: File reconstruction error: ...`). Since FG-195
  fetch catches that too.
- `forge data fetch` keeps the zip in `data/download` in the repo (git-ignored), not next to
  `FORGE_DATA_DIR`, and fills in missing files but replaces files that differ only with `--force`,
  so pointed at `tests/fixtures/spider_mini` it stops instead of downloading and extracting there.
- `load_dev()` raises `DataNotFound` when the data has no dev questions, before it writes the row,
  so the fixture or a partial download never spends one of the six looks.
- The 2,000 agreement items hold 1,445 distinct checks (database, gold and prediction). Unchanged
  gold queries repeat the most (626 items, 370 distinct), since the mutations cycle through val's
  872 questions, which share 456 gold queries; a few predictions also recur against a different
  gold, so the key includes the gold. I5 still reads the rate over all items, as the plan states
  it, and the report gives the distinct rate beside it (FG-193).
- `make_run_id` strips `@<revision>` only when the part before the `@` is shaped like a hub repo id
  (FG-193) and the revision has no `/` unless it starts with `refs/` (FG-195), so `runs/x@y/ckpt`
  names `ckpt` and `Qwen/Qwen3-4B@refs/pr/1` names `qwen3-4b`. A relative path whose `@` is in its
  last name (`runs/ckpt@v2`) still reads as repo `runs/ckpt` at revision `v2`, so pass local
  checkpoints as `./runs/...` or absolute paths.
- Since FG-193 `canonical_json` dumps models in Python mode, so `bytes` inside one (a BLOB in
  `ExecResult.rows`) raises `TypeError` instead of hashing like the same text as a string.
- `build_manifest` stores the config as the JSON it hashed, so a `Path` comes back as a string and
  a tuple as a list (FG-193), and an int key as a string. `canonical_json` sorted int keys as
  numbers, so `{2: a, 10: b}` and its copy hashed differently; since FG-195 it sorts keys as the
  strings JSON writes, and the copy hashes like the config. Mixed int and string keys now hash
  instead of raising, and two keys that JSON writes alike (`1` and `"1"`) raise `TypeError`. Only
  configs with non-string keys hash differently from before; the data manifest's keys are strings,
  so the pinned data hashes are unchanged. Writing, reading back and sorting takes about 2.6 times
  the old single pass: 10 ms against 4 ms for a 20,662-row result on the laptop.
- SQLite reads each `CREATE` statement in `sqlite_master` through `SQLITE_LIMIT_LENGTH` whenever it
  prepares a query, so that limit can't go under the longest one, soccer_1's 1,191 bytes: one byte
  less and every query on soccer_1 fails. Spider's largest stored value is 213 bytes. Records (sorts,
  `DISTINCT`, materialized CTEs) count against it too; a sorted self-join of baseball_1's `team`
  needs between 400 and 600 bytes. Measured for FG-194 (ADR-0014).
- SQLite names an unaliased result column by its expression's text, and that name counts against
  the length limit, so a single unaliased expression over 4 kB fails with `string or blob too big`
  while the same one with `AS v` runs. A comment over the limit fails the same way, and `printf()`
  past the limit returns NULL instead of failing.
- A sandbox job crosses to its worker by module and name, and a spawned worker imports modules
  afresh, so a lambda can't be a job and a monkeypatch in the test process doesn't reach a worker.
  Tests that patch what a worker runs either pass `tests/fakes.py::InlineSandbox` as the pool or
  put the job in an importable module (`tests/sandbox_jobs.py`).
- The sandbox pool costs well under a millisecond a query and about 0.16 s a restart on the laptop
  (`scripts/bench_sandbox.py`, ADR-0014). FG-202 and FG-501/FG-503 run predictions through a
  `SandboxPool` and can hand it to `official_exec_match` as `pool`; `SqliteExecutor.run` stays for
  gold queries, schema reads and tests.
- The limits are Spider's. FG-508 should check the test-suite databases against them, and FG-605
  BIRD, which stores longer text, the way `test_spider_data_fits_the_limits` checks Spider.
- With FG-194's pool and limits, `forge eval agreement --mutations 2000 --split val --seed 0` gives
  the same report as `docs/audit/agreement-2026-11-15.md` apart from the date and commit, so it is
  not regenerated.
- `GoldQueryFailed` now pickles from its fields, since it comes back from a sandbox worker; an
  exception that can't be pickled comes back as a `SandboxError` naming it. A job's *return value*
  that can't be pickled is now caught in `_serve` and reported the same way, naming its type,
  instead of crashing the worker (FG-196). One that pickles in the worker but can't be rebuilt in
  the parent comes back the same way and keeps its worker (FG-197).
- The watchdog is for steps so slow that the progress handler's next check comes late or never, and
  how soon the handler alone stops a query depends on what its steps cost. FG-196 noted that the
  handler alone stops any query within 0.65 s; that was wrong (FG-197). A 50-row query of 100
  `unhex` calls a row runs about 420 VM steps and 0.6 to 0.7 s a row, and against a 300 ms timeout
  it stopped at 0.61 s with the watchdog and 1.8 to 2.0 s without.
  `test_watchdog_stops_a_query_the_handler_never_checks` covers the far end, a 10-row `VALUES`
  query under `PROGRESS_STEPS` steps that the handler never checks: it times out with the
  watchdog and completes (`ok=True`) without it.
  `test_timeout_holds_for_expensive_rows`'s `SLOW_ROWS` are stopped by the handler, so that test
  passes with or without the watchdog (the FG-194 review's point).
- Official EX worker failures: a worker killed at `item_timeout_s` already counted as no match; since
  FG-196 a worker that *crashes* under an item counts as no match too (logged), rather than raising
  `SandboxError` out of the whole call. `forge eval gold` and `forge eval agreement` now catch
  `SandboxError`, so a pool that can't start prints a clean error. FG-196 caught every `SandboxError`
  in `official_exec_match`, so a pool whose workers couldn't start scored each item as a miss;
  since FG-197 only `SandboxWorkerDied` counts, and a start-up failure raises again.
- `DEFAULT_ITEM_TIMEOUT_S` (60 s) bounds the whole official-EX item: two vetting runs plus the gold
  and the prediction on every `.sqlite` in the folder. With Spider's one database per folder that is
  about the evaluator's 60 s per query, but once the test-suite databases land (FG-508) an item runs
  2N queries under the one deadline, so it must scale with the file count then (FG-196, ADR-0014).
- Known limits after the Phase 1 re-verification (FG-197), each in ADR-0014's "Known limits"
  with its measurements (Apple M4 laptop):
  - `SqliteExecutor.run`'s deadline is soft and its overrun has no bound; it is for trusted SQL.
  - A statement whose slow work all falls in its final step (a slow `WHERE` that matches no
    row) can come back ok after its deadline: 0.34 s against 100 ms in process in one of two
    runs, at most `KILL_GRACE_MS` late through a pool.
  - The official-EX vet refuses only empty and denied SQL, so one prediction can hold an item
    for `item_timeout_s` (60 s); the evaluator's own 60 s is a progress handler.
  - No limit bounds what a query allocates inside SQLite: JSON aggregates grew 328 to 655 MiB in
    a 100 ms run, and an official EX item reached 707 MiB in 0.5 s. FG-503 should cap the memory
    of each worker before reward workers run long batches.
  - Results take a constant factor more than `row_size` counts: 1.5 times in process for a
    62 MiB result, whose pool worker peaked at 262 MiB.
  - Without its `DISTINCT`, `spider-train-01297` (soccer_1) returns 93,475 rows, within 7% of
    the evaluator's `MAX_ROWS`; val's largest is 10,507 rows and dev's 20,662.
  - FG-508 must scale `item_timeout_s` with the number of `.sqlite` files in a folder; the FG-508
    story doesn't say so yet.
  - `make_run_id` reads `runs/ckpt@v2` as repo `runs/ckpt` at `v2` (slug `ckpt`),
    `org/model@feature/x` as a path (slug `x`), and keeps a trailing slash's revision
    (`qwen3-4b-main`); only slugs change.
- Tests that time a batch call `SandboxPool.start()` first, which waits until every worker is
  ready. A two-query warm-up could leave one worker still starting (1 of 20 rounds), and the
  rest of its start-up then landed in the timed batch (FG-197).
- transformers 5 fills every generation setting a call leaves unset from the checkpoint's
  `generation_config.json`. The tiny model ships temperature 0.7, top_k 20, top_p 0.8 and a 1.05
  repetition penalty, so `HfBackend` keeps only the special token ids and greedy is a plain
  argmax. FG-306 should check what vLLM takes from the same file (FG-201).
- transformers 5 loads a checkpoint in the dtype it was saved in (bfloat16 for the tiny model);
  `HfBackend` defaults to float32 (FG-201).
- Every row of an `HfBackend` batch gets the batch's wall time as `latency_ms`, so a sum over
  generations counts each batch once per row; FG-202 and FG-301 should time each `generate` call
  for total generation time (FG-201). `forge eval run` prints that total for each attempt
  (`EvalRun.generate_s`, FG-202).
- `check_params` refuses `n > 1` at temperature 0, which would give n copies of one greedy reply,
  so FG-202's `--n` needs a `--temperature` above 0 (FG-201).
- The tiny model's logits are nearly flat (std about 0.01), so its greedy replies can change with
  batch padding through float error. Tests that compare batchings use the offline model in
  `tests/tiny_hf.py`, whose smallest top-1 lead over the tests' 72 greedy steps is 0.0025; FG-207's
  smoke checks should compare runs with the same batches only (FG-201).
- For a `qwen2` checkpoint, transformers 5 loads `Qwen2Tokenizer` whatever `tokenizer_config.json`
  names and rebuilds any other kind of tokenizer wrongly, so the offline test model has a
  byte-level Qwen2 tokenizer (FG-201).
- `chat_token_counter` (FG-105) passes `enable_thinking=False` to every template and `render_chat`
  only to one that reads it; other templates ignore the flag, so both render the same prompt
  and `tokens_in` matches the overflow policy's count (FG-201).
- `split_ref` reads `org/model@feature/x` as branch `feature/x` of `org/model` unless a directory
  by that name exists, where `make_run_id` slugs it as a path (FG-201).
- transformers 5 types `generate`'s `self` with a protocol that mypy can't match to any model
  class, so `HfBackend` holds its model through a one-method `CausalLM` protocol; FG-402's train
  backends will meet the same (FG-201).
- Eval prompts go through `fit_prompt` measured by the backend's tokenizer (words for the fake
  backend), so a long schema is cut as in training, and a prompt still over 1,536 tokens is sent
  as cut and stays in the denominator. At p1 no val or dev prompt passes the limit (FG-202).
- The gold cache, `cache/gold/<split>-<data hash[:12]>.jsonl`, took 0.9 s to build for val, 1.3 s
  for dev and 3.6 s for train on the laptop, and 0.1 s or less to read back; the files are 1.8, 1.8
  and 6.0 MB. No Spider gold result holds a blob or an infinite float, which pydantic's JSON would
  turn into text and `null`; the cache tags both anyway. FG-503 can reuse `forge.eval.gold_cache`
  for rewards (FG-202).
- On full val, `forge eval run --backend fake --model gold` scores 872 of 872 with both checkers in
  about 6 s, and `--model select-1` scores 2 of 872 with both: spider-train-01784 and 01785
  (small_bank_1) have the answer 1. FG-501's constant-query guard can't assume that `SELECT 1`
  never scores (FG-202).
- Resuming a dev run calls `load_dev` again, which logs a second row with the same run id, since
  every call is a look (ADR-0004). FG-304 could let a resumed run keep its first look (FG-202).
- Without `--run-id`, `make_run_id` gives two evals of one model on the same commit and day the same
  id, whatever their splits, so the second is refused as a run with other options; pass `--run-id`.
  A finished run is never resumed or overwritten (FG-202).
- A run folder holds the `.lock` file that keeps a second process off it; FG-304's Hub mirror should
  leave it out (FG-202).
- The scores and generations files hold the contract types only, so the executor's error message
  for a failed prediction is not stored. FG-204 classifies from the live result in the runner;
  FG-205's failure examples can show the class, or a later story can store the message (FG-202).
- `forge eval run` records `platform: local-cpu`, `build_manifest`'s default, whatever the device;
  FG-303 should set `kaggle` for runs there (FG-202).
- Spider asks most questions twice in other words with one gold query, so `summarize` and
  `compare` take `clusters`, and `gold_clusters` puts each such pair in one (val: 872 questions
  in 456 clusters). FG-205 should pass them for val and dev, or the intervals come out too
  narrow. McNemar's p-value still takes each question as an independent pair (FG-203).
- `compare` gives McNemar's table but no p-value when a question has several samples, which
  are not independent pairs; the change in EX keeps its interval (FG-203).
- Each interval reseeds `default_rng(0)`, so it depends only on the scores it covers, and
  FG-207's I4 can recompute any interval in a report by summarizing the same scores. A test
  pins one interval, so a numpy release that changes the stream fails it before reports
  drift (FG-203).
- statsmodels takes about a second to import, since it brings in pandas and scipy, so
  `forge.eval.metrics` imports it inside `compare` only (FG-203).
- `classify` takes the database's columns (`table_columns`) where the plan's signature has the
  gold result: resolving a bare or a double-quoted name needs the schema, and the gold result
  decides no class. `forge eval run` reads each database's columns once, in process (FG-204).
- No train or val gold query double-quotes a column or table name, though 2,848 of the 8,659
  double-quote text, such as `"France"`. So the taxonomy reads a double-quoted name that no table
  in scope has as a string, as SQLite does, and `"France"` and `'France'` are one value (FG-204).
- Each of the 8,656 train and val gold queries that run classifies as `other` against itself: the
  canonical form finds no difference where there is none. On 4,000 agreement mutations of the same
  questions (seed 0), every change landed in a class its rule allows, which
  `tests/integration/test_taxonomy.py` checks on the fixture. A classification takes about 1 ms on
  the laptop (FG-204).
- A query nested in a condition counts in that condition, so a filter whose subquery is missing or
  picks another column or aggregate is `where`. Dropping the condition that held a table's only use
  is `wrong_tables`, since tables come first (FG-204).
- `error_class` follows fast EX, since a score is written before the official batch runs, so a
  prediction that only the official evaluator accepts keeps a class. FG-205's taxonomy table should
  count the failures of the metric it reports (FG-204).
