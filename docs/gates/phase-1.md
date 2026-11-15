# Phase 1 gate: Data and SQL environment

Decision: PASS
Date: 2026-11-15
Commit: 5a47c69 (top of the Phase 1 stack, after fix stories FG-191 to FG-197)

## Tests (passed / failed / skipped for later phases)

Fresh clone at d00d6c5 (the last code change; 5a47c69 after it only corrects docs and a docstring, and `make check` passes on it):

| Run | Result |
|---|---|
| `make gate PHASE=1` with the full Spider data | 464 passed, 0 failed, 145 s |
| `make check` without the full data (what CI sees) | 385 unit and 67 integration passed; the 12 skips are exactly the `requires_full_data` tests |
| `scripts/check_story_tests.py --phase 1` | all 46 planned tests exist across 14 stories |
| `gitleaks` over every commit since Phase 0 | 63 commits scanned, no leaks |

CI on GitHub ran lint-type, unit and integration on Python 3.11 and 3.12, and security, for every story branch.

## Invariants I1..I7 (pass / fail / n.a.)

| Invariant | Result |
|---|---|
| I1 splits disjoint by database, training code can't load val or dev | pass |
| I2 database files unchanged, non-SELECT denied | pass: 1,000 mixed queries, ATTACH and VACUUM INTO through the official path, and workers killed mid-query on every fixture database leave every file byte-identical |
| I5 fast comparator agrees with the official evaluator | pass: 99.15% on 2,000 mutations (98.96% on the 1,445 distinct ones); the report rebuilds identically from the final code |
| I3, I4, I6, I7 | n.a. (rewards, reports and GPU runs come in later phases) |

## Runs this phase

None. No GPU work in Phase 1.

## Ledger: GPU h this phase / total / remaining weekly quota; cash $ total

0.00 h / 0.00 h / 30.00 h remaining; $0.00.

## Dev split uses so far: 0 of 6

## Demo output

```text
$ forge data fetch --from-zip data/download/data/spider.zip
data/spider: all 170 files match spider.sha256, nothing to do

$ forge data stats
train       7787        131  ...
val          872         15  ...
dev         1034         20  ...
total       9693        166  ...

$ forge eval gold --split val
split val: 872 gold queries over 15 databases
fast EX        872/872  100.00%
official EX    872/872  100.00%

$ forge sql run --db concert_singer "DELETE FROM singer"
denied: DELETE is not a query; only SELECT and VALUES statements may run (0.0 ms)

$ forge eval agreement --mutations 2000 --split val --seed 0
agreement 99.15% (1983 of 2000)
```

`scripts/smoke/phase_1.sh` passes on the full data and writes nothing into `docs/` or the data folder.

## Findings and fix stories

The independent review (spec, correctness, integrity and a fresh-clone run, plus the eight per-story code reviews) raised 32 findings. Skeptics confirmed 22 and refuted 10. The 22 were 3 high, 5 medium and 14 low. Every fix story was reviewed in turn, and what those reviews found went into the next one:

| Story | What it fixed |
|---|---|
| FG-191 | Writes are denied by keyword, whatever error SQLite would raise first. The official evaluator's connection gets the same default-deny authorizer and limits, so a disguised ATTACH can no longer plant files. Forge vets the exact string the evaluator runs. |
| FG-192 | An empty dev split no longer costs a dev use. `forge data fetch` can't overwrite the fixture and reports network failures with a hint. p2 sample values keep their real whitespace. Bracketed identifiers parse. Prose lines no longer win SQL extraction. ADR-0004 says what counts as a dev use. |
| FG-193 | Canonical JSON encodes a model like its dump. Manifests snapshot their config. Local paths with `@` keep their name. The naive-time guard checks the UTC offset. The agreement report carries the Spider attribution and counts distinct items. |
| FG-194 | Model SQL runs in worker processes that are killed at a hard deadline. SQLite limits come from Spider's measured needs. Result memory is counted realistically. The evaluator's fetch is capped. See ADR-0014. |
| FG-195 | Xet transfer failures get the fetch hint. Int-keyed configs hash the same way twice. Refs like `runs/x@y/ckpt` keep their name. |
| FG-196 | The watchdog test fails when the watchdog is removed. Official EX survives a worker death. An unsendable job result gets a clear error. Doc claims are narrowed to what was measured. |
| FG-197 | Only a worker's death counts as a miss in official EX (pool start-up failures raise). Statement keywords are read the way SQLite's tokenizer does. Forced Hub download failures show their real cause. Sandbox tests are checked at the worst case the limits allow. The known limits are recorded in ADR-0014. |

A final re-verification gave all 36 findings (the gate's 22 plus 14 from the fix reviews) to skeptics, who tried to reproduce each one at the fixed commit. None was still high or medium. FG-197 closed the low remainders or recorded them below.

FG-197's own review found two documentation details: the shape of query that comes back ok after its deadline, and the densest `unhex` row. Both were measured again on the laptop and corrected in FG-197's last commit. No code changed.

### Known limits (ADR-0014)

The ADR lists them with measured numbers. In short: the in-process executor's deadline is soft and is used only for trusted SQL (gold queries, schema reads, the `forge sql run` command); one adversarial official-EX item can cost up to its 60 s item timeout before it is killed and scored as no match; the item timeout must scale with the number of databases once the test-suite databases arrive (FG-508); and an `@` in the last segment of a relative local path is read as a revision.

## Budget table update (section 3.8), if estimates changed

No change.
