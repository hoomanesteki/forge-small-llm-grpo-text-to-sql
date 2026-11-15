# Phase 2 gate: Evaluation harness

Decision: PASS
Date: 2026-11-15
Commit: the top of the Phase 2 stack, after fix stories FG-292 to FG-295 (tagged v0.2.0 on main)

## Tests (passed / failed / skipped for later phases)

Fresh clone of the final stack (`2bb9eb2`):

| Run | Result |
|---|---|
| `make gate PHASE=2` with the full Spider data | 657 passed, 0 failed, 2 min |
| `make check` without the full data (what CI sees) | 553 unit and 91 integration passed; the 13 skips are exactly the `requires_full_data` tests |
| `scripts/check_story_tests.py --phase 2` | all 79 planned tests exist across 22 stories |
| `gitleaks` over every commit since Phase 1 | 53 commits scanned, no leaks |

CI on GitHub runs lint-type, unit and integration on Python 3.11 and 3.12, security, and since FG-207 the CPU smoke pipeline, on every pushed branch and PR. Every story branch was green before landing.

## Invariants I1..I7 (pass / fail / n.a.)

| Invariant | Result |
|---|---|
| I1 splits disjoint by database, training code can't load val or dev | pass |
| I2 database files unchanged, non-SELECT denied | pass: eval runs leave every database byte-identical |
| I4 report numbers trace to the run's stored files | pass: `forge report rebuild` renders each committed report again from its scores and manifest and matches it line for line, its chart byte for byte |
| I5 fast comparator agrees with the official evaluator | pass: `test_i5_agreement_on_full_val` on the full val split |
| I3, I6, I7 | n.a. (rewards, GPU reproducibility and the GPU ledger check come in later phases) |

## Runs this phase

| Run | Kind | Platform | GPU h | Status | Key metric | Manifest |
|---|---|---|---|---|---|---|
| `2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7` | eval, local pilot (ADR-0016) | `local-mps` (Apple M4) | 0.17 | finished | official EX 47.2% [43.1%, 51.1%] on val | [manifest.json](../runs/2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7/manifest.json) |

The pilot is Qwen2.5-Coder-0.5B-Instruct, greedy, prompt p1, on the 872 val questions. Its number comes from the generated [report](../runs/2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7/eval/val-report.md). It is a pilot, not a project result: the results come from the Kaggle T4 runs.

## Ledger: GPU h this phase / total / remaining weekly quota; cash $ total

0.00 h / 0.00 h / 30.00 h remaining; $0.00. The pilot's 0.17 h on `local-mps` is in the ledger but counts toward neither the Kaggle quota nor the project's GPU hours (ADR-0016).

## Dev split uses so far: 0 of 6

## Demo output

Paths shortened.

```text
$ make smoke
scored 20 of 20
official EX      0/20  0.00%
# Run smoke-2bb9eb2: official EX 0.0% [0.0%, 0.0%] on val
smoke passed: wrote .smoke-ok-2bb9eb2400b6c0fe382882b8788d75a09f56867f

$ forge eval run --backend fake --model gold --split val
valid          872/872  100.00%
fast EX        872/872  100.00%
official EX    872/872  100.00%

$ bash scripts/smoke/phase_2.sh
# Run gold: official EX 100.0% [100.0%, 100.0%] on val
# Official EX on val, run b against run a: -100.0 points [-100.0, -100.0], McNemar p = 0.0020 on clusters
runs/tiny/eval/val-report.md: matches its rebuild
runs/tiny/eval/val-hardness.png: matches its rebuild
runs/tiny/eval/compare-gold.md: matches its rebuild
runs/gold/eval/val-report.md: matches its rebuild
runs/gold/eval/val-hardness.png: matches its rebuild
phase 2 smoke passed

$ FORGE_RUNS_DIR=<copy of docs/runs> forge report rebuild 2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7
.../eval/val-report.md: matches its rebuild
.../eval/val-hardness.png: matches its rebuild

$ forge eval run --backend fake --model gold --split dev --limit 2
error: --split dev needs --reason: every dev evaluation is logged in docs/dev-usage.md, and there are six at most (ADR-0004)
```

`make smoke` takes 14 s on the laptop. A random tiny model scoring 0% is expected: the smoke run checks the pipeline, not the model.

## Findings and fix stories

The independent review (spec, correctness, integrity and a fresh-clone run, plus the eight per-story code reviews) raised 33 findings. Skeptics confirmed 22 and refuted 11. The 22 were 0 high, 3 medium and 19 low; two of the mediums were the same provenance bug. The pilot's own review added one more for this phase, and three that belong to Phase 3 stories (below).

| Story | What it fixed |
|---|---|
| FG-292 | A run's commit and dirty state come from the whole checkout `FORGE_ROOT` names, whatever the working directory; a run started from `docs/` could record a clean commit for a dirty tree. Outside a checkout it is one clear error. Report locks never dirty the tree. The smoke marker is written only by a full-scope smoke, which prints its report. A limited train run starts with `train_spider` instead of the databases with no rows. Bad adapter or dtype names are refused before any weights load. |
| FG-293 | McNemar's exact test pairs gold-query clusters, the same units the intervals resample, and clusters ignore whitespace. The taxonomy reads ORDER BY aliases and derived-table names as SQLite does. `forge report rebuild` masks only the exact timestamp, compares resolved compare paths and checks the chart byte for byte where the same renderer drew it. A test proves retried SQL runs in the sandbox pool. The pilot report was regenerated by the tool. |
| FG-294 | The pilot FG-291 has a story block with acceptance criteria and named tests. Section 5.5 says how gate fix stories are numbered when a phase has a pilot story. The pilot's unfenced replies are counted. |
| FG-295 | Wrapped ORDER BY aliases and derived-table columns read as SQLite reads them. A chart is skipped only when its text names another renderer. A symlinked or case-variant `--out` writes the default compare file. The dirty check's log excludes follow git's own spelling of the paths, and a Forge root its checkout doesn't track is refused instead of recording another repository's commit. The pilot extraction note was recounted by a committed script. |

A first re-verification gave 31 findings to skeptics at the fixed commit: the gate's 23 and 8 from the fix stories' reviews. 21 were fixed and 10 lows remained, which FG-295 took. A second round confirmed 9 of them, and the last, two provenance edge cases, was closed with tests that fail without the fix. The fresh-clone gate run passed all 16 of its checks.

### What the pilot showed

763 of the pilot's 872 replies had no `sql` code block despite prompt p1, so extraction rule 3 read most of them. Rule 3 keeps one line, and it cut 12 multi-line queries to their first line. Run whole, 5 of those 12 change class and 1 matches gold (`scripts/count_pilot_extraction.py`). The extraction rules are unchanged. FG-307 weighs this when it chooses the prompt format (ADR-0007).

### Moved to Phase 3

- A config file's `ops` section for the Kaggle runner: FG-303.
- The accelerator's name in new manifests (the pilot's says `gpu: null`): FG-305.
- The dev-config test, narrowed to pilot configs so dev baseline configs are valid: FG-307.

### Known limits

- Error taxonomy: `ORDER BY (alias) COLLATE x` where the alias names an expression keeps its parentheses in the canonical form, and a derived table's unaliased expression column named in double quotes from outside is read as a string. Both change only an error-class label, never EX, and no Spider gold query or pilot reply hits them.
- Compare reports: an `--out` in other letter case, written before the default compare file exists, is treated as a separate file. `forge report rebuild` then flags the difference.
- Charts: rebuild compares a chart's bytes only where the same renderer draws it (matplotlib, FreeType, numpy, Pillow, zlib and the platform, all named in the PNG). Elsewhere it says the chart wasn't compared. The pilot's chart was drawn on the Mac and is compared there.
- Squash merges leave a run's commit off `dev`, so each committed run has a tag `run/<run_id>` on the commit its manifest names.

## Budget table update (section 3.8), if estimates changed

No change.
