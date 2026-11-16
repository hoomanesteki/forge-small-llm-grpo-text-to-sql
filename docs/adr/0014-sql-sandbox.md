# ADR-0014: Untrusted SQL runs in three layers, the last a worker that can be killed

- Status: accepted
- Date: 2026-11-15

## Context

Rewards and evaluation run SQL a model wrote, hundreds of thousands of times, so one query
must not be able to write, fill memory or hold a worker for long. FG-106 and FG-191 gave
every connection a read-only URI, a default-deny authorizer and SQLite limits, and gave
`SqliteExecutor.run` a deadline: a progress handler every 1,000 VM steps and a watchdog
thread that calls `interrupt()`. SQLite looks at both only at VM loop jumps, so it can't
stop work packed into one row. The FG-191 review showed it with a one-row `SELECT` of many
`instr` calls on a 99 kB string (`WITH t(x) AS MATERIALIZED (SELECT hex(zeroblob(49500)))
SELECT instr(x, hex(zeroblob(25000)) || '1') + ...`). Reproduced on the laptop: 100 calls
ran 8.7 s against a 300 ms timeout and 200 calls 18.5 s against 2,000 ms (4.4 s and 8.5 to
8.9 s when FG-197 ran them again on an Apple M4). Official EX had the same gap, and worse: the
vendored evaluator's only deadline is a 60 s progress handler, and it ran in a thread,
which can't be stopped at all.

## Decision

Three layers, each of which holds on its own.

1. **Authorizer and read-only URI** (FG-106, FG-191). `connect_readonly` opens
   `mode=ro&immutable=1`, and the authorizer passes only `SELECT`, `READ`, `FUNCTION` and
   `RECURSIVE`, so writes, DDL, `PRAGMA` and transactions fail as they are prepared. The
   executor denies anything but one `SELECT`, `VALUES` or `WITH ... SELECT` by keyword
   before SQLite sees it, reading the keyword as SQLite's tokenizer does (FG-197): a byte
   order mark is whitespace, a parameter's Tcl-style suffix is part of the parameter,
   keywords are ASCII, and a `WITH` clause is read by its grammar, so a table named
   `replace` is a name.
2. **SQLite limits and the in-process deadline**, set from what Spider needs rather than
   guessed. Measured over the full data (all 166 databases and 9,693 gold queries, dev
   included, which is not a dev use under ADR-0004), pinned by
   `tests/integration/test_executor.py::test_every_gold_query_runs_under_the_limits` and
   `::test_spider_data_fits_the_limits`:

   | Need | Spider's largest | Limit |
   |---|---|---|
   | Gold query | 577 bytes | `SQLITE_LIMIT_SQL_LENGTH` 8,192 (completions are at most 256 tokens) |
   | LIKE pattern in gold | 42 characters | `SQLITE_LIMIT_LIKE_PATTERN_LENGTH` 100 |
   | Stored value | 213 bytes (cre_Doc_Tracking_DB) | `SQLITE_LIMIT_LENGTH` 4,096 |
   | `CREATE` statement in `sqlite_master` | 1,191 bytes (soccer_1) | the same 4,096 |
   | Row | 355 bytes of values (baseball_1 `team`, numbers counted as 8) | the same 4,096 |
   | Columns | 48 in a table (baseball_1 `team`), 19 in a gold result | `SQLITE_LIMIT_COLUMN` 200 |
   | Gold result | 20,662 rows, 3.3 MB as Python objects (wta_1, dev) | 64 MiB of rows |
   | Gold result as the evaluator runs it, `DISTINCT` removed | 93,475 rows, 10.3 MB (soccer_1, train); val 10,507 rows, dev 20,662 | the evaluator's 100,000 rows and 64 MiB |

   SQLite reads each `CREATE` statement through the length limit whenever it prepares a
   query, so the schema text, not the data, sets the floor: every gold query runs at
   1,191 bytes, and soccer_1's fail one byte under it. 4,096 leaves more than three times
   that. The cost of `instr`, `LIKE`, `GLOB` or `replace` grows with the product of their
   inputs' lengths, so the length and pattern limits keep each call cheap. They don't
   bound the work in one row. A row holds as many calls as 8 kB of SQL, each reading its
   longest inputs from a column, and a common table expression that is referenced several
   times is copied into every reference, so nesting multiplies the calls past what the text
   holds. On the laptop (Apple M4, SQLite 3.53.1), one row of compact calls filling 8 kB,
   the long arguments in one-row materialized CTEs, ran 0.07 s with `instr`, 0.05 s with
   `replace`, 0.34 to 0.38 s with `LIKE` or `GLOB`, 2.2 to 2.8 s with `ltrim` (the length
   limit holds its character set to 341 characters) and 4.3 to 5.5 s with `unhex` (731 calls
   joined with `+`, the densest packing; 670 calls joined with `||` ran 4.7 to 4.9 s). Nested
   CTEs turned 397 bytes of SQL into 10,000 `instr` calls that ran 0.9 to 1.0 s, and 379
   bytes into 1,000 `unhex` calls that ran 6.5 s, each against a 100 ms timeout. So the
   in-process deadline is soft, with no bound on its overrun, and only layer 3 bounds it
   (`test_executor.py::test_limits_bound_the_work_in_one_row` and
   `::test_in_process_deadline_is_soft_for_one_row`). `SQLITE_LIMIT_ATTACHED` 0 stays,
   which stops `ATTACH` and `VACUUM INTO` on any connection. The result budget counts what
   rows take in memory (`row_size`: the tuple, `sys.getsizeof` of each value and the list
   slot), where FG-191 counted 8 bytes a number and a character a byte, so a wide row of
   numbers took about four times its count. The vendored evaluator's connections get the
   same limits, and its `exec_on_db_` reads results in batches with the same budget and a
   100,000-row ceiling instead of `fetchall` (`third_party/test_suite_eval/CHANGES.md`).
3. **Killable worker processes.** `forge.sql.sandbox.SandboxPool` keeps up to N spawned
   workers, each with its own `SqliteExecutor`, and sends each one job at a time over a
   pipe. A query still running `KILL_GRACE_MS` (250 ms) past its `timeout_ms` gets its
   worker killed with SIGKILL and comes back as `ExecResult(ok=False, error_kind="timeout",
   error="killed after ... ms (timeout ... ms)")`; a fresh worker starts the next time the
   slot has work. A worker that dies under a query gives an `other` error. `restarts`
   counts the replacements, and each one is logged at debug level with the count so far.
   `start()` brings every worker up before a batch that is timed. `SandboxPool.call` runs
   any module-level function in a worker under the same kind of deadline, raising
   `SandboxTimeout` for a kill and `SandboxWorkerDied` for a death; any other
   `SandboxError` means no worker could start or a reply couldn't cross the pipe. Official
   EX runs each item, the vetting and the evaluator's call together, that way, with
   `item_timeout_s` (60 s) and the overall deadline on top (ADR-0008); a killed item is
   not a match, and neither is one whose worker dies (FG-196), but a worker that can't
   start raises, which both eval CLIs report as an error (FG-197). The 60 s bounds the
   whole item (two vetting runs, then the gold and the prediction on every `.sqlite` in
   the folder), which with Spider's one database per folder is about the evaluator's own
   60 s per query (`exec_eval.TIMEOUT`). Once the test-suite databases sit beside it
   (FG-508) an item runs 2N queries under the one deadline, so it must be revisited then
   (see "Revisit if").

Which path for which SQL:

- **Untrusted SQL**, anything a model wrote or a reward or evaluation scores, goes through
  a `SandboxPool` (`run`, `run_many`) or `SqliteExecutor.run_many`, which makes one for the
  call. `forge.eval.agreement.score` runs predictions and official items on one pool.
  The eval runner (FG-202) runs predictions and official items on one pool per run, and
  the reward functions (FG-501, FG-503) must do the same.
- **Trusted SQL**, Spider's gold queries, schema reads and tests, may use
  `SqliteExecutor.run` in process; its deadline is soft, with no bound on the overrun.
  `forge sql run` is in process too: it runs what the person at the terminal typed.

## Measured cost

`scripts/bench_sandbox.py` on the laptop (Apple silicon, Python 3.12), pool of 4:

| | Fixture (60 gold queries x 5), two runs | Full data (9,690 gold queries) |
|---|---|---|
| In process, one at a time | 0.08 to 0.10 ms a query | 1.14 ms |
| `pool.run`, one at a time | 0.15 to 0.18 ms (0.05 to 0.10 ms more) | 1.20 ms (0.06 ms more) |
| `pool.run_many`, wall time a query | 0.21 to 0.34 ms | 0.51 ms |
| First worker up and answering | 154 to 169 ms | 205 ms |
| Restart, kill to the next answer | 155 to 158 ms median, 214 ms worst of 20 | 163 ms median, 180 ms worst of 10 |

So a query pays well under a millisecond for the pool. On the fixture's tiny queries the
parent's bookkeeping is the bottleneck and four workers are no faster than one; on real
gold queries they take less than half the time.

## Consequences

- A runaway query costs its timeout plus at most the grace, and a kill costs a restart
  of about 0.2 s, so a batch full of pathological predictions stays bounded in time.
- Layer 2 stops loops at the in-process deadline, but it is soft: one row can run for
  seconds past it, and the evaluator's own deadline is 60 s and its column-permutation
  search on wide results runs in Python (upstream issue #12). The kill bounds all of that,
  and a crash, at the timeout plus the grace.
- `SqliteExecutor.run_many` starts a pool for every call (about 0.2 s). A training run's
  rewards share one long-lived pool, `forge.train.rewards.reward_pool()`, started before
  the first batch (FG-503); `official_exec_match` accepts one as `pool`.
- Each worker imports what its jobs need at start (`imports`), so module loading doesn't
  count against a job's deadline.
- Limits are set for Spider. The test-suite databases (FG-508) and BIRD (FG-605) must be
  checked against them the way `test_spider_data_fits_the_limits` checks Spider; BIRD
  stores longer text.
- An unnamed result column is named by its expression's text, which the length limit also
  bounds, so a single unaliased expression over 4 kB fails with `string or blob too big`.
  Spider's longest gold query is 577 bytes.

## Known limits

Accepted rather than fixed after the Phase 1 re-verification (FG-197); `docs/progress.md`
lists them too. Timings are from the laptop (Apple M4, 16 GB, Python 3.12.13, SQLite
3.53.1).

- **In-process overrun.** `SqliteExecutor.run`'s deadline is soft and its overrun has no
  bound (layer 2). That matters only if untrusted SQL reaches `run` instead of a pool. The
  FG-191 recipe now fails at once with `string or blob too big`; under FG-191's limits a
  pool kills it 552 ms into a 300 ms timeout.
- **Late but ok.** A statement whose slow work all falls in its final step can end after its
  deadline and still come back ok, because nothing checks the deadline once that step has
  produced its last row. A `SELECT` whose `WHERE` runs 60 `unhex` calls and matches no row
  came back ok after 0.34 s against a 100 ms timeout in one of two runs (the other timed
  out). The same 60 calls in a one-row `SELECT` with no `FROM` time out. How late has no
  measured bound. Through a pool such a result is at most `KILL_GRACE_MS` late.
- **The official-EX vet.** `VET_TIMEOUT_MS` (100 ms) refuses only empty and denied SQL. A
  prediction that times out there still goes to the evaluator, so one adversarial
  prediction can hold an item for `item_timeout_s` (60 s by default) before it is killed
  and scored as no match. `forge eval run` closes this for reported numbers (FG-202,
  ADR-0008): a prediction whose sandboxed run timed out at 2,000 ms never reaches the
  evaluator, and each item gets 10 s. Direct callers such as `forge eval agreement`
  still have the 60 s default.
- **The evaluator's timeout.** Its 60 s is a progress handler, as soft as layer 2; the item
  kill bounds it.
- **Memory.** No limit bounds what a query allocates inside SQLite. Aggregating
  4,000-character strings over an endless recursive CTE, `json_group_array`,
  `json_group_object` and `jsonb_group_array` grew 328 to 655 MiB in a 100 ms run in
  process (the `jsonb` one also ran on to 212 ms), and `median` 11 MiB. A pool worker
  (39 MiB idle) peaked at 386 MiB with a 100 ms timeout and 603 MiB with 300 ms, so the
  default 2 s plus the grace allows more than was measured here. An official EX item
  killed at 0.5 s peaked at 707 MiB; at the default 60 s only the host's memory bounds it,
  since the vet doesn't refuse a timeout. Since FG-503 each worker on Linux, where CI and
  Kaggle run, caps its address space with `RLIMIT_AS` at what it maps once ready plus
  `WORKER_MEMORY_LIMIT` (1 GiB, about four times the largest result's peak); a query past
  it gets `MemoryError`, comes back as an `other` error (`SandboxOutOfMemory` from
  `call`, which official EX counts as no match), and its worker is replaced. SQLite stops
  growing such an aggregate at the cap but runs on to the deadline, so it still costs its
  timeout. macOS accepts `RLIMIT_AS` without enforcing it, so on the laptop no cap is set
  and the deadline's kill is the only bound.
- **Result memory.** `row_size` counts `sys.getsizeof`, which misses the allocator's
  rounding, and `ExecResult` and pickling for the pipe copy the rows. A result counted at
  62 MiB (10,000 rows of 150 two-byte blobs) raised the in-process peak by 95 MiB (1.5
  times the count); through a pool the worker peaked at 262 MiB and the parent grew
  113 MiB. These are constant factors on a counted budget.
- **The evaluator's row ceiling.** Run as `eval_exec_match` runs it, with `DISTINCT`
  removed, soccer_1's training gold `spider-train-01297` returns 93,475 rows, within 7% of
  `exec_eval.MAX_ROWS`; val and dev keep more than four times the room. A lower ceiling
  would make that gold raise `GoldQueryFailed`, which
  `test_every_gold_query_runs_under_the_limits` would catch.
- **One item, many databases.** `item_timeout_s` covers the whole official-EX item. With
  the test-suite databases (FG-508) a correct but slow prediction could be killed where
  upstream's 60 s per query would pass it, so FG-508 must scale the item timeout with the
  number of `.sqlite` files (see "Revisit if"). The requirement is recorded here and in
  `docs/progress.md`, not yet in the FG-508 story.
- **Run-id slugs** (not the sandbox's; recorded here with the rest of that review).
  `make_run_id` reads `runs/ckpt@v2`, `ckpt@v2` and `models/qwen3@base` as a hub repo at a
  revision, so their slugs are `ckpt`, `ckpt` and `qwen3`; pass local checkpoints as
  `./runs/...` or absolute paths. `org/model@feature/x` reads as a path (slug `x`) unless
  the branch is written `feature%2Fx`, as `hf://` paths write it, and a trailing slash in
  `Qwen/Qwen3-4B@main/` gives `qwen3-4b-main`. Only the slug changes; the date, kind and
  commit stay.

## Revisit if

A legitimate query (gold, or a correct prediction) fails on a limit; restarts show up in
a reward or eval run often enough to cost real time; a worker's memory, which only its
deadline bounds, shows up in a long run; or a later SQLite changes where it checks for
interrupts. Also when the test-suite databases land (FG-508): `item_timeout_s` then covers
the gold and the prediction on every database, so a correct but slow prediction could be
killed where upstream's 60 s per query would pass it, and the item timeout should scale
with the number of `.sqlite` files in the folder (FG-196).
