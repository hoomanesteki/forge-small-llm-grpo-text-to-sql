# ADR-0015: A score says how many replies it took and how the first one did

- Status: accepted
- Date: 2026-11-15

## Context

FG-206 adds execution-guided retry. With `forge eval run --retry 1`, a sample whose query
fails to run gets one more turn: the prompt's messages, the model's reply as the
assistant's turn, then `That query failed with: <error>. Reply with a corrected query.`
The sample is scored on the second reply. Reports must show EX with and without retry,
and the `Score` contract (plan section 3.5, FG-102) has no field that says a score took
two replies, or how the first one did. Only a story that says so changes
`src/forge/core`; FG-206 does (ADR-0001).

EX without retry can't be inferred from the first query having failed. Fast EX never
matches a query that fails to run, but official EX can: the evaluator rewrites a
prediction before running it (`> =` becomes `>=`, `YEAR(CURDATE())` becomes `2020`, and
every `DISTINCT` goes). On the fixture's flight_1, `SELECT count(DISTINCT *) FROM
Aircraft` is a syntax error to SQLite, and official EX matches it to the gold query
`SELECT count(*) FROM Aircraft`; a query with `> =` for `>=` fails and matches the same
way. So the first reply's official verdict has to be measured.

## Decision

`Score` gets two fields, both with defaults, so a score stored before FG-206 reads as it
did:

- `attempts: int = 1`, the replies the sample took. It is 2 when the first reply's query
  failed to run and the run retried it: `ExecResult.ok` is false, which takes in a reply
  with no SQL, a denied statement, an error and a timeout. A query that ran is never
  retried, even when its rows are wrong, since there is no error to show.
- `first_try_ex: bool | None = None`, official EX of the first reply when `attempts` is 2,
  and `None` with one reply, whose verdict is `ex_official`.

The other fields are the second reply's when there are two: `valid`, `ex_fast`,
`ex_official`, `error_class`, and the sample's line in `eval/<split>-generations.jsonl`,
with that turn's tokens and latency. The first reply goes to
`eval/<split>-first-tries.jsonl`, so every reply the model gave is stored, and official EX
judges first tries in the run's one batch at the end, beside the scored replies. A batch
appends its first tries, then its generations, then its scores, so a kill leaves nothing
the next attempt can't drop, and the files must pair: each retried score has one first
try, and no other score has any.

A first query that timed out is no official match without the evaluator, as in scoring
(ADR-0008). Its retry turn names the timeout (`timed out after 2000 ms`), not the time
the kill measured, which varies, so a resumed run asks the same question again. Each
retry is drawn with a seed of its own, derived from the run's seed and the sample index:
the first reply's seed would steer the second toward the choices that made the first.

Without retry, a retried sample scores as its first reply: not valid, no fast match, and
`first_try_ex` for official EX (`forge.eval.retry.without_retry`). A run's report shows
both when any score has 2 attempts, with the change between them and its paired
interval, and a compare report adds official EX without retry when either run retried.

## Consequences

- Score lines gain the two fields, `"attempts":1,"first_try_ex":null` with one reply.
  Older lines still parse, and so do runs without a first tries file.
- Reports refuse a finished run whose scores contradict its options: 2 attempts in a run
  without retry, a retried score with no `first_try_ex`, or a `first_try_ex` with one
  reply. `read_run` refuses first tries that don't pair one to one with retried scores.
- Every report header names the run's retry option, so a number made with retry can't
  pass for one made without.
- The Hub run store (FG-304) must carry `eval/<split>-first-tries.jsonl` with the other
  outputs; the manifest lists it as `first_tries` when it exists.
- A sample's tokens and latency describe the reply it is scored on. A retried sample's
  cost is its first try's and its generation's together.

## Revisit if

A run retries more than once (`attempts` above 2 would need each try's verdict), a reward
learns from the retry turn, or FG-508 retries a stored run's failures without generating
its first replies again.
