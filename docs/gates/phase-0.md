# Phase 0 gate: Foundation

Decision: PASS
Date: 2026-11-14
Commit: aaa87a9 (top of the Phase 0 stack, after fix stories FG-091 to FG-093)

## Tests (passed / failed / skipped for later phases)

Fresh clone, `make setup && make gate PHASE=0`: 53 passed, 0 failed, 0 skipped, in 28 s.
`scripts/check_story_tests.py --phase 0`: all 13 planned tests exist, carry a phase 0 marker
and are not skipped.

CI on GitHub (push builds, every story branch): lint-type, unit (3.11, 3.12),
integration (3.11, 3.12) and security (gitleaks, pip-audit) green.

## Invariants I1..I7 (pass / fail / n.a.)

n.a. The first invariants (I1, I2, I5) arrive in Phase 1. No GPU runs in Phase 0.

## Runs this phase

None.

## Ledger: GPU h this phase / total / remaining weekly quota; cash $ total

0.00 h / 0.00 h / 30.00 h remaining; $0.00.

## Dev split uses so far: 0 of 6

## Demo output

```text
$ uv run forge version
forge 0.0.1

$ uv run forge ledger summary
Kaggle hours, last 7 days: 0.00 of 30 (remaining 30.00)
GPU hours, all time: 0.00 over 0 sessions
Cash spent: $0.00

$ uv run python scripts/import_backlog.py --dry-run | tail -1
56 stories, 190 points, 9 phases, 11 GPU; 56 new issues

$ uv run forge doctor
ok    python             3.12.13
ok    uv                 uv 0.12.3
FAIL  kaggle auth        no credentials found
FAIL  hugging face auth  no HF_TOKEN
FAIL  run store          hoomanesteki/forge-runs: needs Hugging Face auth first
FAIL  disk               5.4 GB free (need 10)
4 required check(s) failed
```

`forge doctor` fails as expected: the Kaggle and Hugging Face accounts are not created yet,
and the machine has less free disk than the 10 GB the plan asks for. Each line prints its fix.

## Findings and fix stories

An independent review over four areas (spec conformance, correctness, integrity, fresh-clone
run) raised 35 findings. After de-duplication, every confirmed one was fixed:

| Story | Fixed |
|---|---|
| FG-091 | Ledger and dev-usage appends take a file lock (200 parallel appends keep 200 rows; 16 racing dev uses give exactly 6). Every ledger field is validated (platform, finite hours, no pipes or line breaks). No silent fallback to cwd for the repo root. Edits to the two append-only logs no longer make the tree dirty. `canonical_json` refuses values without a stable encoding. |
| FG-092 | `six` added so the locked Kaggle CLI starts. `.env` is read from the repo root and passed to the Kaggle CLI. Tokens read from files are masked. Timeouts become failed checks. Inline comments in `.env` parse. Runbook Kaggle Secrets step restored. |
| FG-093 | GPU spikes get the `type:gpu` label (11 in total). The traceability check rejects planned tests that are skipped, xfail or marked for a later phase. `.env.*` ignored; the template's `/site` line no longer hides the Quarto site. |

## GitHub evidence

- Stories landed as PRs #60 to #69 into `dev`, each squash-merged after 12 green checks
  (push and pull_request runs of lint-type, unit and integration on 3.11 and 3.12, and security).
- Backlog imported: issues #1 to #56 for the plan stories (11 labeled `type:gpu`) and #57 to #59
  for the fix stories, with milestones Phase 0 to Phase 8. A second import created 0 issues.
- Branch protection (`scripts/protect_branches.sh`): `dev` requires lint-type, unit (3.11, 3.12),
  integration (3.11, 3.12) and security, with strict status checks, linear history and admins
  included. `main` requires the same checks without strict mode.
- FG-002 AC2: a throwaway branch with an unused import failed `lint-type` at `make lint`
  (run 43641864023), and the branch was deleted.
- FG-002 AC3: a direct push to `dev` was rejected with `GH006: Protected branch update failed`.

## Open items

- `forge doctor` needs the Kaggle and Hugging Face accounts before Phase 3.

## Budget table update (section 3.8), if estimates changed

No change.
