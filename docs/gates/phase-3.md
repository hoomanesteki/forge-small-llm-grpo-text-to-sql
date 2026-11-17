# Phase 3 gate: Free-GPU ops and baselines

Decision: PASS
Date: 2026-11-17
Commit: `52abed7`, the top of the Phase 3 stack (`dev` at `c8f4b41`, then FG-301's completion, FG-398 and FG-307), with this report, its doc corrections and the 0.3.0 bump on top; to be tagged v0.3.0 on main

`docs/progress.md` marks every Phase 3 story done, and no high or medium finding is open. One criterion still waits on a PR: FG-302's real publish ran, but its output goes in the PR that lands FG-301's completion, which is not open yet. That and the rest of what the release needs are under Before release. The lows still open are under Known limits.

## Tests (passed / failed / skipped for later phases)

Fresh clone of the final stack (`52abed7`):

| Run | Result |
|---|---|
| `make gate PHASE=3` with the full Spider data, in the laptop's own time zone (CST) | 993 passed, 0 failed, 0 skipped, in 173.66 s; smoke: 7 smoke tests passed, and the tiny-model eval scored all 20 questions (official EX 0 of 20, as expected for the tiny model) and wrote its report and chart |
| `make check` without the full data and with `TZ=UTC` (what CI sees) | 857 unit and 119 integration passed; the 17 skips are exactly the tests that need the full Spider data |
| `scripts/check_story_tests.py --phase 3` | all 97 planned tests exist across 29 stories |
| `gitleaks` over every commit since Phase 2 (`8651ef5..52abed7`), and over `docs/` | 36 commits scanned, no leaks; no leaks in `docs/` |
| Commit messages since Phase 2 | all 36 conventional, each with a `Refs: FG-xxx` line; 35 subjects at 72 characters or fewer, and `c036edc` at 73 (under Findings) |
| `uv build` | a 258,907-byte wheel of 72 entries with no tests, docs or env files; it says 0.2.0 at `52abed7`, and this gate's last commit bumps it to 0.3.0 |

No test was skipped for a later phase: `docs/runs/pending.yaml` is empty, and the same 993 tests are collected with or without `--phase 3`.

CI on GitHub runs lint-type, unit and integration on Python 3.11, 3.12 and 3.13, security, and the CPU smoke pipeline, on every pushed branch and PR. Python 3.13 joined with FG-303; the spike then read 3.13.16 on Kaggle's image. All 14 Phase 3 PRs on `dev` (#124 to #136 and #138) landed with every check green: 14 checks for #124 and #125, and 18 from #126 on. The three branches still to land were green at their tips: FG-301's completion (`0702396`), FG-398 (`2d86268`) and FG-307 (`52abed7`).

Stories: `docs/progress.md` at `52abed7` marks every Phase 3 story done: FG-301 to FG-307, and FG-391 to FG-398. Milestone: the commits on `dev` close #25 to #27, #118 to #123 and #137 when `dev` merges to main. The issues of FG-301, FG-302, FG-303, FG-307 and FG-398 (#22, #23, #24, #28 and #141) belong to the three branches still to land, and close with their PRs. So plan section 5.5's step 1, a milestone with no open issues, is not met at this commit: the Phase 3 milestone has 15 open issues and 0 closed. It is met once the three PRs land and `dev` merges to main, as Phase 2's 12 issues were.

Contract drift: `git diff v0.2.0..52abed7 -- src/forge/core` is empty.

Run audit: plan section 5.5's step 5 asks that each run's manifest exist in the run store, the private Hub repo `hoomanesteki/forge-runs` (section 3.9). A kernel has no `HF_TOKEN`, so a run reaches that store only through `forge runs push` from the laptop. Listed with `HfApi.list_repo_files` on 2026-11-17, the store holds the 50-question run and none of FG-307's eight. So this audit reads `docs/runs/` in place of the run store, a deviation from step 5 until those eight are pushed (Before release). Each eval run's manifest is committed under `docs/runs/`, each run has its ledger row, both dev evaluations are in `docs/dev-usage.md` with their reason, and every report rebuilds (I4 below). The spike has no manifest kind, so its `spike-report.json` and its ledger row stand in. The runs of 2026-11-16 were made from `c8f4b41`, which is on `dev`; the three of 2026-11-17 were made from `2f77281` and `d2d6a3a`, which need `run/` tags (Before release).

## Invariants I1..I7 (pass / fail / n.a.)

From the fresh-clone gate run at `52abed7`, where every test in `tests/integration/test_invariants.py` passed:

| Invariant | Result |
|---|---|
| I1 splits disjoint by database, training code can't load val or dev | pass |
| I2 database files unchanged, non-SELECT denied | pass |
| I4 report numbers trace to the run's stored files | pass: `forge report rebuild` on a copy of `docs/runs/` says each of the nine Kaggle runs' reports "matches its rebuild", and so do eight charts and the p2 run's compare report. The 50-question job's chart was drawn on Linux, so a Mac says it isn't compared |
| I5 fast comparator agrees with the official evaluator | pass, as decided at this gate (under Findings): over FG-307's eight baseline runs pooled, 95 of 6,528 samples disagree, 98.54%; each of the chosen model's four runs reaches 98% alone (98.97%, 98.97%, 100.00% and 98.36%). The reference's dev run is under it alone, 97.58% (25 of 1,034), and every disagreement goes away once the evaluator keeps `DISTINCT` (`scripts/check_i5_distinct.py`, ADR-0008's note of 2026-11-17). `test_i5_agreement_on_full_val` passes too |
| I6 greedy evaluation reproduces EX within 0.5 points | pass on the T4: `compare_runs` of the I6 repeat against the whole run says "official EX 67.0% and 67.0% on the 100 samples both scored, 0.0 points apart (I6 allows 0.5); 0 verdicts and 1 replies differ"; the reply is `spider-train-00134` |
| I7 every GPU run has its ledger rows, and cash stays at or under $50 | pass: `test_i7_gpu_runs_in_ledger` audits every session of every committed and locally kept run against the real ledger, the `local-mps` pilot and the Kaggle runs both |
| I3 | n.a. (reward determinism comes with Phase 5) |

## Runs this phase

| Run | Kind | Platform | GPU h | Status | Key metric | Manifest |
|---|---|---|---|---|---|---|
| `2026-11-16-throughput-spike-c8f4b41` | FG-301 throughput spike | `kaggle`, T4 x2, one GPU per bench | 0.59 | finished, 17:54 to 18:29 UTC: 15 of its 18 stages ok, and the 3 that failed met vLLM 0.33.0's missing bitsandbytes | vLLM fp16: 577.3 and 391.4 tokens a second for Qwen3 1.7B and 4B; GRPO on TRL in 4-bit: 18.43 and 32.92 s a step, the 4B at 4.94 GB allocated | none: [spike-report.json](../spikes/kaggle-gpu/spike-report.json) |
| `2026-11-16-smoke-kaggle-50-val-c8f4b41` | FG-303's eval job on 50 val questions, integration checkpoint 3 | `kaggle`, T4 | 0.18 | finished | official EX 76.0% [60.0%, 90.0%], a pipeline check, not a baseline | [manifest.json](../runs/2026-11-16-smoke-kaggle-50-val-c8f4b41/manifest.json) |
| `2026-11-16-qwen3-1-7b-val-c8f4b41` | FG-307 candidate, val, p1 | `kaggle`, T4 | 0.24 | finished | official EX 55.7% [51.7%, 59.9%] | [manifest.json](../runs/2026-11-16-qwen3-1-7b-val-c8f4b41/manifest.json) |
| `2026-11-16-qwen3-4b-val-c8f4b41` | FG-307 candidate, val, p1 | `kaggle`, T4 | 0.21 | finished | official EX 43.1% [39.2%, 47.2%] | [manifest.json](../runs/2026-11-16-qwen3-4b-val-c8f4b41/manifest.json) |
| `2026-11-16-qwen2-5-coder-1-5b-val-c8f4b41` | FG-307 candidate, val, p1 | `kaggle`, T4 | 0.21 | finished | official EX 65.3% [61.2%, 69.1%] | [manifest.json](../runs/2026-11-16-qwen2-5-coder-1-5b-val-c8f4b41/manifest.json) |
| `2026-11-16-qwen2-5-coder-1-5b-i6-c8f4b41` | FG-307 I6 repeat, the first 100 val questions, p1 | `kaggle`, T4 | 0.17 | finished | official EX 67.0% [54.0%, 79.0%], 0.0 points from the whole run's score on the same 100 questions | [manifest.json](../runs/2026-11-16-qwen2-5-coder-1-5b-i6-c8f4b41/manifest.json) |
| `2026-11-16-qwen2-5-coder-1-5b-p2-c8f4b41` | FG-307 chosen model, val, p2 | `kaggle`, T4 | 0.15 | finished | official EX 67.4% [63.3%, 71.3%] | [manifest.json](../runs/2026-11-16-qwen2-5-coder-1-5b-p2-c8f4b41/manifest.json) |
| `2026-11-17-qwen3-8b-awq-val-2f77281` | FG-307 reference (Qwen3-8B, AWQ 4-bit), val, p2 | `kaggle`, T4 | 0.21 | finished | official EX 71.1% [67.2%, 74.9%] | [manifest.json](../runs/2026-11-17-qwen3-8b-awq-val-2f77281/manifest.json) |
| `2026-11-17-qwen2-5-coder-1-5b-dev-2f77281` | FG-307 chosen model, dev, p2: dev look 1 | `kaggle`, T4 | 0.20 | finished | official EX 67.2% [63.6%, 70.7%] on dev | [manifest.json](../runs/2026-11-17-qwen2-5-coder-1-5b-dev-2f77281/manifest.json) |
| `2026-11-17-qwen3-8b-awq-dev-d2d6a3a` | FG-307 reference, dev, p2: dev look 2 | `kaggle`, T4 | 0.25 | finished | official EX 70.8% [67.2%, 74.0%] on dev | [manifest.json](../runs/2026-11-17-qwen3-8b-awq-dev-d2d6a3a/manifest.json) |

Each key metric is the header of the run's report, under `eval/` beside its manifest. The val runs are on all 872 val questions and the dev runs on all 1,034 dev questions, unless the row says otherwise.

Two CPU sessions of FG-301 spent no GPU quota and have no ledger row: its session-limit probe (`hoomanesteki/forge-session-limit-probe`, pushed with `-t 900`, 15:56 to 16:11 UTC) and its secret probe (version 3 of the spike's kernel). Their reports are in `docs/spikes/kaggle-gpu/`. A third CPU kernel that day was Phase 4's: FG-495's mount probe, which listed `/kaggle/input` with the code dataset and the 50-question job's output as sources. Its record, `docs/spikes/kaggle-gpu/mount-probe.json` with `probe.py`, is on the Phase 4 branch `fix/fg-495-adapters-from-kernel-output` (`2a47fb2`) and lands with Phase 4 (Handed on).

### What the Kaggle runs showed

From `docs/spikes/kaggle-gpu.md`, which quotes each number from its report, unless a bullet names another source:

- `--accelerator NvidiaTeslaT4` gives a T4 x2 session, and every bench used one of the two GPUs. Kaggle starts a script kernel as `/kaggle/src/script.py`, in `/kaggle/working`, with no arguments, so the arguments Forge bakes into the script are the ones that count. Private datasets mount at `/kaggle/input/datasets/hoomanesteki/<slug>`.
- Kaggle refuses a kernel title over 50 characters with a 400 from `SaveKernel`. The 50-question job's first push, with a 57-character title, failed so; FG-398 keeps names within the limit.
- No Kaggle Secret reaches a CLI-pushed run: the secrets service answers `HF_TOKEN` with the same 400 as a label no secret has. So every eval kernel ran without `HF_TOKEN` and skipped its Hub upload, and results come back with `forge runs pull --store kaggle` and go to the Hub from the laptop with `forge runs push` (ADR-0005). That push was first made for the 50-question run (`docs/runbook.md`, Run store).
- The image runs Python 3.13.16 and torch 2.11.0+cu130 with CUDA 13.0; trl, bitsandbytes, vLLM and Unsloth are not preinstalled. In the 50-question job's kernel, pip installed the vLLM list in 391.6 s, four times uv's 95.9 s in the spike. Over the nine eval jobs, pip took 247.7 s to 446.4 s (each job's `status.json`, as the Found list of `docs/progress.md` gives them). A GPU wheel dataset is not decided.
- vLLM 0.33.0 runs in fp16 on the T4 but has no bitsandbytes, so the 4-bit reference moved to Qwen's AWQ checkpoint (FG-398), which loaded and scored in both of its runs (Runs this phase).
- 4-bit GRPO on TRL with transformers generation runs both Qwen3 sizes; Unsloth was 37% and 36% faster a step, but on its own stack (trl 0.22.2), and TRL with colocated vLLM failed at the 4-bit load. ADR-0006 is accepted on these numbers: TRL on the GPU.
- A session killed at its push timeout keeps its output, and `kaggle kernels status` reports it as `CANCEL_ACKNOWLEDGED`. At most two GPU sessions run at once: a third push is refused. The weekly quota is 30 h and refreshes at 00:00 UTC on Saturdays.
- From the FG-397 and FG-302 rows of `docs/progress.md`: the first `forge kaggle publish-data` met a 403 for a dataset not yet made, where a 404 was expected; FG-397 reads both as "no dataset yet". The real publish then made `hoomanesteki/forge-spider` from the 170 Spider files, as five uploads plus `spider.sha256`, in 2 min 53 s, and `hoomanesteki/forge-code` version 1 holds `c8f4b41`'s wheel. Its output is not yet in a PR (Before release).
- A kernel eval job spends about 11 minutes before its job stage starts: 636.8 s in the 50-question job, 391.6 s of it pip's install and 168.1 s the preflight.

### The model and the prompt

ADR-0012 chooses `Qwen/Qwen2.5-Coder-1.5B-Instruct` from the three p1 val runs:

- Its official EX is 65.3% [61.2%, 69.1%], against Qwen3-1.7B's 55.7% and Qwen3-4B's 43.1%.
- All three pass FG-301's GRPO memory check (3.02 GB and 4.94 GB allocated of the T4's 15.6 GB for 1.7B and 4B) and leave headroom under 75%.
- `scripts/check_choice_extraction.py` applies the rule to the scores as stored: "+9.5 points [+5.0, +13.6], McNemar p < 0.0001 on clusters" ahead of Qwen3-1.7B.
- It applies the rule again with each query that extraction rule 3 cut read whole. Qwen3-4B then reaches 66.7%, with 389 queries read whole: "+1.5 points [-2.4, +5.2], McNemar p = 0.3755 on clusters". That is not significant, so the tie-break takes the model with fewer parameters.
- Both readings choose the same model, so the extraction rule stays as ADR-0007 decided.
- On dev with p2, the chosen model scores 67.2% [63.6%, 70.7%] and the reference 70.8% [67.2%, 74.0%]. Nothing is chosen on dev.

ADR-0007 fixes p2 for every later phase: on the chosen model, p2 scores 67.4% [63.3%, 71.3%] against p1's 65.3%, "+2.2 points [+0.6, +3.8], McNemar p = 0.0115 on clusters". Rule 3 cut no query in either run, so the gain is not the format's.

## Ledger: GPU h this phase / total / remaining weekly quota; cash $ total

2.41 h / 2.41 h / 27.59 h remaining of the 30 h weekly quota, which refreshes on 2026-11-21 at 00:00 UTC; $0.00.

That is ten Kaggle rows: the spike's 0.59 h, the 50-question job's 0.18 h and FG-307's eight runs' 1.64 h, against budgets of 1.5 h for FG-301 and 3 h for FG-307. `kaggle quota -v` on 2026-11-17 counts 2.44 h used and 27.56 h left; the ledger keeps each session to two decimals, and Kaggle keeps its own count. The CPU probes spent no GPU quota and have no row. The FG-291 pilot's 0.17 h on `local-mps` is still in the ledger and counts toward neither the Kaggle quota nor the project's GPU hours (ADR-0016).

## Dev split uses so far: 2 of 6

Both are FG-307's baselines with p2, logged with the reason `baseline` when Kaggle accepted each push:

1. `2026-11-17-qwen2-5-coder-1-5b-dev-2f77281`, the chosen model.
2. `2026-11-17-qwen3-8b-awq-dev-d2d6a3a`, the reference.

## Demo output

The plan's demo names `configs/eval/qwen3-4b-val.yaml`. That config ran on 2026-11-16 as `2026-11-16-qwen3-4b-val-c8f4b41`, pushed, waited for and pulled like the rest. The push recorded in full is the 50-question job's, made the same day from `c8f4b41` with the steps of `scripts/smoke/phase_3.sh` run one at a time (paths shortened):

```text
$ uv run forge kaggle run --job eval --config configs/eval/smoke-kaggle-50-val.yaml --run-id 2026-11-16-smoke-kaggle-50-val-c8f4b41 --yes
code hoomanesteki/forge-code/1 at c8f4b41, data hoomanesteki/forge-spider
estimate 0.25 GPU h
ledger: 30.00 of 30 h left this week
kaggle quota: 29.41 h left
job eval: run 2026-11-16-smoke-kaggle-50-val-c8f4b41 as kernel hoomanesteki/forge-2026-11-16-smoke-kaggle-50-val-c8f4b41, stopped after 0.38 h at most
Kernel version 1 successfully pushed.  Please check progress at https://www.kaggle.com/code/hoomanesteki/forge-2026-11-16-smoke-kaggle-50-val-c8f4b41
run 2026-11-16-smoke-kaggle-50-val-c8f4b41: kernel hoomanesteki/forge-2026-11-16-smoke-kaggle-50-val-c8f4b41, version 1
next: forge kaggle wait 2026-11-16-smoke-kaggle-50-val-c8f4b41

$ uv run forge kaggle wait 2026-11-16-smoke-kaggle-50-val-c8f4b41
hoomanesteki/forge-2026-11-16-smoke-kaggle-50-val-c8f4b41/1: RUNNING
hoomanesteki/forge-2026-11-16-smoke-kaggle-50-val-c8f4b41/1: COMPLETE
job finished at stage upload; upload: skipped: no HF_TOKEN, so the kernel output is the only copy
eval run, finished
ledger: added 0.18 h on kaggle for run 2026-11-16-smoke-kaggle-50-val-c8f4b41

$ uv run forge runs pull 2026-11-16-smoke-kaggle-50-val-c8f4b41 --store kaggle
run 2026-11-16-smoke-kaggle-50-val-c8f4b41: .../runs/2026-11-16-smoke-kaggle-50-val-c8f4b41; the kernel's log and the rest in .../runs/2026-11-16-smoke-kaggle-50-val-c8f4b41/kernel
eval run, finished
ledger: run 2026-11-16-smoke-kaggle-50-val-c8f4b41 has its row already

$ uv run forge ledger summary
Kaggle hours, last 7 days: 2.41 of 30 (remaining 27.59)
GPU hours, all time: 2.41 over 10 sessions
Cash spent: $0.00
```

The ledger summary is from the gate run at `52abed7`, which gave the same lines with `TZ=UTC`. The 50-question job's report rebuilds on the laptop, its chart not compared as drawn on Linux. [ADR-0012](../adr/0012-model-choice.md) names the chosen model, with a val table for every candidate and `scripts/check_choice_extraction.py`'s output, whose two "chosen" lines the gate run printed again word for word:

```text
- chosen: Qwen/Qwen2.5-Coder-1.5B-Instruct, the best, ahead of the next by p below 0.05
- chosen: Qwen/Qwen2.5-Coder-1.5B-Instruct, the one with fewer parameters, as the best two don't differ significantly (p of 0.05 or more)
```

At `52abed7` in the fresh clone, `make smoke` wrote `.smoke-ok-52abed71d36bdc1eb7ccf60ab5efd440603d790b`. Then, with the kaggle CLI replaced by a fake that logged every call and the network denied:

```text
$ forge kaggle run --job eval --config configs/eval/qwen2-5-coder-1-5b-val-p2.yaml --emit-notebook <dir>/eval.ipynb
code forgegate/forge-code/7 at 52abed7, data forgegate/forge-spider
notebook <dir>/eval.ipynb: run 2026-11-17-qwen2-5-coder-1-5b-instruct-eval-52abed7, nothing pushed
estimate 0.3 GPU h; log what it takes with forge ledger add --platform colab

# with no smoke marker for the commit
error: make smoke has not passed on 52abed7: run it on this commit before a GPU job (CONTRIBUTING rule 8)

# with README.md edited
error: the working tree has uncommitted changes; a GPU job runs a commit, so commit them and run make smoke on it first
```

The fake saw only `datasets status` and `datasets download` calls, never a push. The notebook passed nbformat's validation, every code cell parses, and it holds neither of the canary tokens set for the run. Each refusal exited 1 and called nothing. The over-budget refusal was recorded at `4afd4f6` during re-verification: with a 29.9 h kaggle row in a throwaway ledger, `ledger: 0.10 of 30 h left this week`, then `error: the estimate is over what is left this week; pass --allow-over-budget...`, a line the record cuts short.

## Findings and fix stories

### The first round: FG-391 to FG-397

The independent review (spec, correctness, integrity and a fresh-clone run) raised 26 findings, and the per-story code reviews of FG-301 to FG-306 had raised 33 more. The gate review read FG-307's commits as closely as the rest. Each of the 59 was then checked by an independent attempt to reproduce it: 49 were confirmed and 10 refuted. Several confirmed findings were one defect seen by different reviewers. Merged, they are 33: 3 high, 14 medium and 16 low. The three highs:

- `make gate PHASE=3` and `make check` failed whenever the local date differed from the UTC date. Five runner tests built the expected run id from the local date, while the code dates runs in UTC. CI runs in UTC and passed; the laptop, six hours west, failed every evening.
- The spike's `trl` stack, meant to reuse the image's torch, would have installed its own: uv ignores system site packages when it installs.
- On a T4 x2 session the GRPO bench's trainer would see both GPUs in one process and train on 48 completions per step, while the report would say 24.

FG-391 to FG-395 took the 33 (5, 9, 8, 6 and 5 of them). FG-396 took what re-verification left open. FG-397 came from the first live Kaggle use.

| Story | What it fixed |
|---|---|
| FG-391 | One date, UTC, wherever a date names or windows something: run ids, ledger rows, dev-usage rows and the ledger's weekly window. Each attempt at a run is a session with its own hours and its own ledger row, and I7 checks every session. A kernel's session covers the whole kernel, and `forge kaggle wait` and `output` add its row. |
| FG-392 | Kernel eval jobs get a time guard from the push's timeout, with a tenth kept back for official EX and the upload. Every stage result, the error and `error.txt` are redacted. The data stage checks the Spider files against the checksums in the installed wheel. A dev look is logged as soon as Kaggle accepts the push, one look per run id across resumes (ADR-0004). |
| FG-393 | Hub downloads retry real network failures, and `get_manifest` tells a missing run from an outage. A stale local copy can't overwrite a newer stored run. `publish-data` counts only a ready dataset as there and removes its build folder after the upload. |
| FG-394 | The throughput spike measures what it says: the `trl` venv counts the image's torch, every bench sees one GPU and records what its trainer counted, every stack gets the same prompts with Qwen3's thinking off, and a stage loads the 4-bit reference through vLLM. |
| FG-395 | `TimeGuardCallback` saves and pushes at its stop even under `save_strategy="best"`. The vLLM backend sends `LLM.generate` the token ids the hf backend uses. A temperature vLLM would silently raise is refused. The p2 run is for the model ADR-0012's rule chooses. |
| FG-396 | The 17 items re-verification left open, 3 medium and 14 low: a push the kaggle CLI reports as started is never reported as refused, a resume skips an attempt whose kernel source Kaggle dropped, the run store retries what huggingface_hub didn't, the kernel redacts every log record, and the spike's 4-bit load runs right after vLLM's install. |
| FG-397 | `kaggle datasets status` answering 403 for a dataset not yet made in the account's own namespace means "no dataset yet", as a 404 does, so the first `publish-data` can create it. |

Re-verification at `4afd4f6` re-checked 46 findings, the 33 merged ones and 13 from the reviews of FG-391 to FG-395: 29 fixed, 4 fixed with a smaller low left, and 13 still open, 3 medium and 10 low. FG-396 took those 17, the 13 and the 4 smaller lows. Its fresh-clone run found a medium: a canary token reached the kernel log through the run store's retry warnings. Two of the 17 lows already named that channel, so the medium added no item, and FG-396 fixed it by redacting every log record. FG-396's own review found three lows, and re-verification one more, a missing pending entry for the Hub store's first real use. All four were closed later: FG-301's completion corrected the retry docs and recorded the store's first use from the laptop, which needed no pending entry, and replaced the spike doc's second-push recipe, as no second push was needed; FG-398 asks Kaggle about a push whose answer was lost.

### Since the first round: FG-301's completion, FG-398 and FG-307

These ran once the Kaggle account worked, on three branches stacked on `c8f4b41`. Each branch had a code review. Later commits fixed most of what the reviews found, one finding was kept by decision, and the rest is under Known limits.

| Story | What it did |
|---|---|
| FG-301's completion, with FG-302's and FG-303's real runs (`519e2d7` to `0702396`) | Committed the spike's reports under `docs/spikes/kaggle-gpu/` and answered each question of the spike doc from them. Accepted ADR-0006. Worked section 3.8 out from the measurements, with a test that recomputes each row. Committed the 50-question job, FG-303's last criterion. Recorded FG-302's real publish and the run store's first real use, and set FG-301, FG-302 and FG-303 done. |
| FG-398, Kaggle live-run fixes (`9079b35` to `2d86268`) | Kernel titles and slugs over 50 characters are refused, and a long run id gets a shorter kernel name that its record still finds. A push whose CLI fails is checked with `kaggle kernels status`, and counts as made when Kaggle has the attempt's kernel. Training jobs install `kaggle/requirements-train-gpu.txt`, the `trl` stack, with no vLLM. The 4-bit reference is `Qwen/Qwen3-8B-AWQ` with `quantization: awq`, and bitsandbytes is refused in one line. |
| FG-307, GPU baselines and model choice (`9281288` to `52abed7`, with the dev-look logs `d2d6a3a` and `632a9d0`) | Part A committed the five val runs of 2026-11-16, accepted ADR-0012 and ADR-0007, and budgeted section 3.8 for the chosen model. Part B committed the reference's val run and the chosen model's dev run, and held the reference's dev run, which missed I5 run by run. Part C wrote `scripts/check_i5_distinct.py`, committed that run under the I5 reading decided at this gate, and set FG-307 done. |

The review of FG-301's completion had 12 findings, 1 medium and 11 low. FG-398's had 3 low. FG-307's three parts had 1 high, 1 medium and 7 low.

| Review | Finding | Severity | Where it stands |
|---|---|---|---|
| FG-301's completion | Section 3.8's note said 8 h held three ablations, while its own formulas put FG-603 and FG-604 at 14.45 h to 15.25 h and the total over 45 h | medium | fixed by FG-307 in `f70c483`, which works the ablation row out by the formulas |
| FG-301's completion | The secret probe cut a traceback before it scrubbed it | low | fixed in `2f77281`, tested with a token across the cut |
| FG-301's completion | The retry docs named fewer cases where both layers retry than the code has | low | fixed in `2f77281` |
| FG-301's completion | No artifact test covered FG-302 | low | fixed in `2f77281`: the 50-question job's mounts are checked for FG-302 |
| FG-301's completion | The reports README didn't name what each file ran from | low | fixed in `2f77281` |
| FG-301's completion | Stale text: two findings on `docs/progress.md`, one on the spike doc and one on ADR-0005 | 4 low | fixed in `2f77281` |
| FG-301's completion | ADR-0012's bullet on the reference said vLLM loads it with bitsandbytes, then that vLLM 0.33.0 doesn't | low | fixed by FG-398 in `2d86268` |
| FG-301's completion | Three pip freeze lines in `spike-report.json` (lines 954, 1892 and 2164) name a third-party package | low | kept: they list what Kaggle's image had installed, measured data, and the gate run passed them as known false positives |
| FG-301's completion | `0702396` holds three changes under one scope | low | open (Known limits) |
| FG-398 | The FG-307 row of `docs/progress.md` still named `quantization: bitsandbytes` | low | fixed by FG-307 in `67f3b35` |
| FG-398 | The lost-answer check trusts that the attempt's kernel is new | low | open (Known limits) |
| FG-398 | `find_kernel` reads a 401 as "no such kernel" | low | open (Known limits) |
| FG-307, part A | Section 3.8's 15.25 h for ablations is more than FG-603's and FG-604's own budgets | medium | fixed in `67f3b35`: the note says so, and Phase 6 sizes them before they run |
| FG-307, part A | The choice check took a second or partial run of a candidate | low | fixed in `c18f16c`, which refuses it |
| FG-307, part A | Numbers worked out by hand in `docs/progress.md`, and ADR-0012's I5 counts under no test | low | fixed in `67f3b35` and `d63640b`: the hand-made sums left, and ADR-0012 quotes each report's I5 line under a test |
| FG-307, part A | The FG-307 row still named `quantization: bitsandbytes`, the clause FG-398's review found | low | fixed with it in `67f3b35`: the row names the AWQ reference |
| FG-307, part B | The reference's dev run was held out of `docs/runs/`, so FG-307 was not finished | high | closed by part C (`a636fdc`, `3697712`, `52abed7`) |
| FG-307, part B | A sentence that no run had loaded the AWQ checkpoint on a T4 | low | fixed: `configs/models.yaml` in `52abed7`; the runbook, the spike doc and both reference configs in `439a828`; the Found list of `docs/progress.md` with this report's correction |
| FG-307, part C | ADR-0012 said FG-307's runs are listed in `pending.yaml` | low | fixed in `439a828` |
| FG-307, part C | A Found item counted eight kernel eval jobs, where nine are committed | low | fixed in `439a828` |
| FG-307, part C | ADR-0008's note had only the script's table | low | fixed in `439a828`: it gives the pilot's agreement (99.43%, 5 of 872) and the 50-question job's (100.00%, 0 of 50) beside the table |

Fresh-clone gate runs, by commit. The stack commits of the first round have the same trees as `dev` commits: `52e9dd4` as `9618973` (#130), `4afd4f6` as `9acec15` (#135) and `aa4f3d5` as `de8a3b5` (#136).

| Run (UTC) | Commit | `make gate PHASE=3` | What else |
|---|---|---|---|
| The review, 2026-11-16 00:46 | `52e9dd4` | failed, without the full Spider data: 5 failed, 853 passed, 15 skipped | All 5 failures in `test_kaggle_runner.py`, from the local and UTC dates |
| Re-verification, 2026-11-16 06:49 | `4afd4f6` | passed: 935 passed, 2 skipped (FG-307's pending runs) | `make check`, story tests, `gitleaks` and `make smoke` passed, and the tests again twelve hours west of UTC |
| After FG-396, 2026-11-16 | `aa4f3d5` | passed: 969 passed, 2 skipped (FG-307's pending runs) | 97 planned tests across 29 stories; smoke passed |
| This gate, 2026-11-17 | `52abed7` | passed: 993 passed, 0 skipped | `make check` in UTC, story tests, `gitleaks`, commit messages, the demo and `uv build`, as above |

The run at `52abed7` found one new low: the squash merge of #128, `c036edc`, has a 73-character subject, one over CONTRIBUTING's 72. The subject was set at merge as FG-305's 66-character squash subject with " (#128)" added. The PR's title, "FG-305 Time guard and automatic ledger rows", is 43 characters and is not the subject.

### Decisions at this gate

- I5 is read over FG-307's eight baseline runs pooled, as plan section 4 states it, "≥ 98% on stored baseline outputs": 98.54%. Each of the chosen model's four runs must also reach 98% alone, since GRPO's reward stands in for official EX on that model's replies, and all four do. Every committed run's agreement is still reported, in its report and when the test fails. ADR-0008's note of 2026-11-17 records this, and plan section 4's table is unchanged.
- The evidence for that reading: `scripts/check_i5_distinct.py` scored each of the 95 disagreements again with the official evaluator. Each got its stored verdict again, and none remained once the evaluator kept `DISTINCT`, which it drops by default. The reference writes such a `DISTINCT` more often than the candidates: 170 of 872 replies on val and 215 of 1,034 on dev, against 59 to 91 of 872. That is why its dev run, at 97.58%, is the one run under 98% alone. So the fast comparator has no fault there for GRPO's reward to inherit. Official EX keeps the evaluator's default, for comparability with published Spider numbers.
- This reading relaxed a test. Before `3697712`, `test_i5_agreement_on_committed_runs` asserted 98% for every committed run alone. Since `3697712` it asserts the eight runs pooled and the chosen model's four runs alone. The reference's dev run failed the old assertion, and the work stopped there, as CONTRIBUTING rule 3 asks: `67f3b35` marked FG-307 blocked. Hooman decided the reading on 2026-11-17 (ADR-0008's note), and `3697712` changed the test with it. Six committed runs are now reported but not asserted alone: Qwen3-1.7B's val run (98.05%), Qwen3-4B's (99.54%), the reference's val run (98.39%) and dev run (97.58%), the FG-291 pilot (99.43%) and the 50-question job (100.00%). The first four still count in the pool.
- Extraction rule 3 stays as ADR-0007 decided, since the model choice is the same under both readings (above).
- The dev looks and the reference ran with p2, the prompt ADR-0007 fixed, so the chosen model's dev baseline compares with the trained model's dev evaluation and the reference runs with the same prompt and harness.
- `c036edc` stays as merged, since fixing it would rewrite published history. Squash subjects are written at 65 characters or fewer, and checked before a merge, so the " (#NNN)" added at merge keeps them within 72.
- The ablations: Phase 6 sizes FG-603 and FG-604 to the 15.25 h section 3.8 gives them before they run, as its note says, cutting in section 10.2's order if the weeks need it.
- Plan sections 3.12 and 3.13 now name Python 3.11 to 3.13 and the TRL stack on the GPU, as ADR-0006 asked of this gate (`439a828`).

### Phase gate 3 checklist

- The budget table in section 3.8 is updated with measured throughput, and the total, 40.79 h, stays within 45 h (last section).
- `kaggle/entry.py` read again at `52abed7`: secrets are never printed, and the final status is always written. The secrets stage prints only where each secret came from or why it is missing. It keeps each value to redact it, and sets it in the environment for the Hub and W&B clients, where a child process inherits it; no secret reaches a pushed kernel today. Every log record is redacted from before the first stage. A failure's traceback, `error.txt` and every stage result are redacted. `status.json` is written in a `finally` that also closes a manifest left running. Since the first round, FG-398 changed only which package list a job installs. The nine committed kernel runs each have their `status.json`, and each shows `HF_TOKEN` and `WANDB_API_KEY` unreadable, as `kaggle secrets: ConnectionError`.
- Tag `v0.3.0`: once the release PR merges.

### Before release

- The PR that lands FG-301's completion carries FG-302's publish output and FG-303's push, wait and pull output in its description. FG-302's criterion "A real publish works once (paste the output in the PR)" is met only then.
- The three stack branches land, and their PRs close #22, #23, #24, #28 and #141. With `dev` merged to main, the Phase 3 milestone then has no open issue (plan section 5.5, step 1).
- `forge runs push` puts FG-307's eight runs in the Hub run store, which holds none of them today (Run audit, above).
- Squash merges leave a run's commit off `dev`, so each committed run whose manifest names a stack commit needs a tag `run/<run_id>` on that commit, pushed when its branch lands:
  - `run/2026-11-17-qwen3-8b-awq-val-2f77281` on `2f77281`
  - `run/2026-11-17-qwen2-5-coder-1-5b-dev-2f77281` on `2f77281`
  - `run/2026-11-17-qwen3-8b-awq-dev-d2d6a3a` on `d2d6a3a`

  The other committed runs were made from `c8f4b41`, on `dev`, and the pilot's tag exists since Phase 2.

### Handed on

- Phase 4: the training configs name `Qwen/Qwen2.5-Coder-1.5B-Instruct` at the commit `configs/models.yaml` pins, with prompt p2 (ADR-0012, ADR-0007), and section 3.8 budgets its training at Qwen3 1.7B's measured rates.
- FG-495's mount probe: a kernel's output listed as a source mounts at `/kaggle/input/notebooks/hoomanesteki/<slug>/`, as `mount-probe.json` on `fix/fg-495-adapters-from-kernel-output` (`2a47fb2`) shows for the 50-question job's kernel. That record lands with Phase 4.
- The Phase 4 replay: FG-491's `forge ledger record` must take `record_run`'s new return, the rows and one note per session, and pass the run's sessions (FG-391, FG-396, note in `docs/progress.md`).
- FG-403 and FG-504: training runs keep no `sessions.json` of their own, so a resumed training run on the laptop is one session until those jobs call `forge.ops.sessions.begin` and `end` (FG-391).
- FG-601's audit: rewards and official EX part on `DISTINCT`, so it should look for fast EX gains that official EX doesn't show (ADR-0008).
- Phase 6: FG-603 and FG-604, sized to section 3.8's 15.25 h (above).
- A later story: the kernel's pip install (247.7 s to 446.4 s a job) could move to uv or to a wheel dataset, and the unchosen candidates' `-val-p2` and `-dev` configs, which never run, can go.

### Known limits

Open lows, from the reviews and the gate run:

- The lost-answer check after a failed push trusts that the attempt's kernel is new. If its slug already exists on Kaggle, from the same run id pushed in another checkout or from two run ids that make one slug, a failed push is recorded as made, and a dev job's look is logged (FG-398's review).
- `find_kernel` reads kaggle 2.2.5's "Permission 'kernels.get' was denied", which the CLI prints for a 401 as well as a 403, as "no such kernel", so an authentication failure on that call reads as a push that never started (FG-398's review).
- `0702396` holds three changes, the runbook's, the run store's retry notes and the story closes, under one `docs(runbook)` subject. The squash merge makes the branch one commit on `dev`.
- `c036edc`'s subject is 73 characters (above).
- `uv build` puts any untracked file that `.gitignore` doesn't name into the sdist, so a release builds from a clean checkout (the gate run's `uv build`).

Documented in `docs/progress.md`, the runbook or the spike doc:

- `check_config` takes `quantization: awq` for any vllm model, though only the reference needs it. The manifest records it, so such a run is visible.
- What a kernel's child processes print, pip or vLLM's engine, is not a log record and isn't redacted. No secret reaches a pushed kernel today, so its Hub upload is skipped and results reach the Hub from the laptop.
- A run pulled from the Hub before `forge kaggle wait` or `output` logs its last session only up to the upload. The runbook says to wait first.
- A new run's kernel that fails before its job holds no run, so its minutes need `forge ledger add`; the CLI prints a note quoting the kernel's `status.json` span.
- A Kaggle run's ledger rows count each whole kernel session, so they exceed the manifest's `gpu_hours` by the setup, preflight and upload.
- The eval deadline keeps a tenth of the push's timeout for official EX and the upload. Each of the nine kernel eval jobs finished without reaching its guard, so the tenth is untested on Kaggle.
- The session-limit probe measured a push timeout on a CPU kernel; that the 12-hour cut behaves the same is an assumption. The time guard stops jobs at 11 h 15 min either way.
- `publish-data` uploads a new version over a failed or deleted `forge-spider`, which no real account has tried yet.
- On the laptop `load_dev` logs each load, a resume's included (ADR-0004); a dev look is one row per run id only for `forge kaggle run --resume`.
- Eval jobs run vLLM 0.33.0's transformers 5.20.2 and torch 2.14.0, and training jobs the lock's trl 1.18.0 and transformers 5.23.0 on the image's torch 2.11.0+cu130, while CI runs transformers 5.23.0 and torch 2.15.0.
- Rebuild compares a chart's bytes only where the same renderer drew it, so the 50-question job's chart, drawn on Linux, isn't compared on the Mac.

## Budget table update (section 3.8), if estimates changed

Section 3.8 is worked out again from the measurements: FG-301's spike report and the 50-question job's `status.json`, by formulas written under the table, with the chosen model at Qwen3 1.7B's rates since FG-307. A row with no measured basis is still an estimate, and says so. The total went from about 43 h to 40.79 h, within the 45 h this gate allows:

| Job | Model | Est. GPU h | Story |
|---|---|---|---|
| Throughput spike (measured: its ledger row) | 1.7B and 4B | 0.59 | FG-301 |
| Baselines: 3 candidates on val, the chosen one with p2, the I6 repeat, the chosen model on dev, and a larger reference model on val and dev | up to 8B in 4-bit | 3.60 | FG-307 |
| SFT pilot (estimate) | 1.7B | 1 | FG-404 |
| SFT main (estimate) | chosen (1.5B) | 2.5 | FG-404 |
| SFT checkpoint evals: 2 on val | chosen | 0.90 | FG-404 |
| GRPO pilot (100 steps, stopped at 40 minutes and resumed) | 1.7B | 1.30 | FG-507 |
| GRPO main (about 600 steps, 2 sessions with resume) | chosen | 5.15 | FG-507 |
| Checkpoint selection: 3 on val, then SFT and GRPO on dev | chosen | 2.25 | FG-508 |
| Hacking fix re-run: about 200 steps, then a val eval | chosen | 2.25 | FG-602 |
| Ablations: B to E and a second seed of B, 300 steps and a val eval each | 1.7B | 15.25 | FG-603, FG-604 |
| Reserve (estimate) | - | 6 | - |
| **Total** | | **40.79 h** (2 Kaggle weeks of 30 h) | |

- A GRPO step is the median of the spike's steps 2 to 10 on the `trl` stack, 18.426 s for 1.7B and 32.924 s for 4B, times 1.5 for Spider's longer prompts and real rewards, plus 0.25 h a session.
- An eval is the kernel's measured setup, the model's download and vLLM's load, plus 256 tokens a question at vLLM's measured rate, times the same 1.5. Without the margin it gives 627 s for the 50-question job, whose session lasted 657 s.
- The Baselines row keeps its estimates; FG-307's runs took 1.64 h by their ledger rows. The eval rows are on the high side, then: FG-307's runs took 0.15 h to 0.25 h each.
- The ablation row is what the formulas give for the planned runs, more than FG-603's own 6 h and FG-604's own 2 h budgets and section 10.2's 11 + 1 h for week 4; Phase 6 sizes them (Decisions, above).
- `test_budget_rows_follow_the_measurements` recomputes each worked-out row from the spike's report and the 50-question job's `status.json`, and checks the total is at most 45 h.
