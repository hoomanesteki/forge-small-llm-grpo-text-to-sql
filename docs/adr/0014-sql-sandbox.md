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
ran 8.7 s against a 300 ms timeout and 200 calls 18.5 s against 2,000 ms. Official EX had
the same gap, and worse: the vendored evaluator's only deadline is a 60 s progress
handler, and it ran in a thread, which can't be stopped at all.

## Decision

Three layers, each of which holds on its own.

1. **Authorizer and read-only URI** (FG-106, FG-191). `connect_readonly` opens
   `mode=ro&immutable=1`, and the authorizer passes only `SELECT`, `READ`, `FUNCTION` and
   `RECURSIVE`, so writes, DDL, `PRAGMA` and transactions fail as they are prepared. The
   executor denies anything but one `SELECT`, `VALUES` or `WITH ... SELECT` by keyword
   before SQLite sees it.
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
   | Gold result | 20,662 rows, 3.3 MB as Python objects (wta_1, dev) | 64 MiB of rows; the evaluator also 100,000 rows |

   SQLite reads each `CREATE` statement through the length limit whenever it prepares a
   query, so the schema text, not the data, sets the floor: every gold query runs at
   1,191 bytes, and soccer_1's fail one byte under it. 4,096 leaves more than three times
   that. The cost of `instr`, `LIKE`, `GLOB` or `replace` grows with the product of their
   inputs' lengths, so for those the length and statement limits together bound the work in
   one row: the slowest rows that fit, each call on the longest values and as many calls as
   8 kB holds, take 37 ms (`instr`) to 58 ms (`LIKE` with a 100-character pattern) on the
   laptop, against 18.5 s under FG-191's 100 kB
   (`test_executor.py::test_limits_bound_the_work_in_one_row`). This was measured for those
   four functions only (FG-196). Others are slower: a row of `ltrim` calls with a large
   character set, or `unhex` with a large ignore set, runs a second or two, so the
   in-process deadline is soft and only layer 3 bounds it
   (`test_executor.py::test_in_process_deadline_is_soft_for_one_row`).
   `SQLITE_LIMIT_ATTACHED` 0
   stays, which stops `ATTACH` and `VACUUM INTO` on any connection. The result budget
   counts what rows take in memory (`row_size`: the tuple, `sys.getsizeof` of each value
   and the list slot), where FG-191 counted 8 bytes a number and a character a byte, so a
   wide row of numbers took about four times its count. The vendored evaluator's
   connections get the same limits, and its `exec_on_db_` reads results in batches with
   the same budget and a 100,000-row ceiling instead of `fetchall`
   (`third_party/test_suite_eval/CHANGES.md`).
3. **Killable worker processes.** `forge.sql.sandbox.SandboxPool` keeps up to N spawned
   workers, each with its own `SqliteExecutor`, and sends each one job at a time over a
   pipe. A query still running `KILL_GRACE_MS` (250 ms) past its `timeout_ms` gets its
   worker killed with SIGKILL and comes back as `ExecResult(ok=False, error_kind="timeout",
   error="killed after ... ms (timeout ... ms)")`; a fresh worker starts the next time the
   slot has work. A worker that dies under a query gives an `other` error. `restarts`
   counts the replacements, and each one is logged at debug level with the count so far.
   `SandboxPool.call` runs any module-level function in a worker under the same kind of
   deadline. Official EX runs each item, the vetting and the evaluator's call together,
   that way, with `item_timeout_s` (60 s) and the overall deadline on top (ADR-0008); a
   killed item is not a match, and neither is one whose worker crashes (FG-196). The 60 s
   bounds the whole item — two vetting runs and the gold and the prediction on every
   `.sqlite` in the folder — which with Spider's one database per folder is about the
   evaluator's own 60 s per query (`exec_eval.TIMEOUT`). Once the test-suite databases sit
   beside it (FG-508) an item runs 2N queries under the one deadline, so it must be
   revisited then (see "Revisit if").

Which path for which SQL:

- **Untrusted SQL**, anything a model wrote or a reward or evaluation scores, goes through
  a `SandboxPool` (`run`, `run_many`) or `SqliteExecutor.run_many`, which makes one for the
  call. `forge.eval.agreement.score` runs predictions and official items on one pool.
  The eval runner (FG-202) and the reward functions (FG-501, FG-503) must do the same.
- **Trusted SQL**, Spider's gold queries, schema reads and tests, may use
  `SqliteExecutor.run` in process; its deadline is soft. Layer 2 holds the overrun to a
  few hundredths of a second for `instr`, `replace`, `LIKE` and `GLOB`, the functions
  measured, but not in general (FG-196). `forge sql run` is in process too: it runs what
  the person at the terminal typed.

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
  of about 0.2 s, so a batch full of pathological predictions stays bounded.
- Layer 2 stops loops at the in-process deadline, but it is soft: a single row of the
  slower string functions (FG-196) runs a second or two past it, and the evaluator's own
  deadline is 60 s and its column-permutation search on wide results runs in Python
  (upstream issue #12). The kill bounds all of that, and a crash, at the timeout plus the
  grace.
- `SqliteExecutor.run_many` starts a pool for every call (about 0.2 s). FG-503 keeps one
  long-lived pool for a training run; `official_exec_match` accepts it as `pool`.
- Each worker imports what its jobs need at start (`imports`), so module loading doesn't
  count against a job's deadline.
- Limits are set for Spider. The test-suite databases (FG-508) and BIRD (FG-605) must be
  checked against them the way `test_spider_data_fits_the_limits` checks Spider; BIRD
  stores longer text.
- An unnamed result column is named by its expression's text, which the length limit also
  bounds, so a single unaliased expression over 4 kB fails with `string or blob too big`.
  Spider's longest gold query is 577 bytes.

## Revisit if

A legitimate query (gold, or a correct prediction) fails on a limit; restarts show up in
a reward or eval run often enough to cost real time; or a later SQLite changes where it
checks for interrupts. Also when the test-suite databases land (FG-508): `item_timeout_s`
then covers the gold and the prediction on every database, so a correct but slow
prediction could be killed where upstream's 60 s per query would pass it, and the item
timeout should scale with the number of `.sqlite` files in the folder (FG-196).
