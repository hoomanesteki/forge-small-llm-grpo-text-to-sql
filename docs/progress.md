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
| FG-205 | Report generator | 3 | done |  |  | `forge report run`, `compare` and `rebuild` (`forge.eval.report`) from a finished run's stored files (`read_run`) and its split's questions in data with the manifest's hash; official EX leads, with fast EX and the valid rate beside it, 95% intervals over gold clusters, a by-hardness table and PNG, the error classes of official EX failures, and 5 failures from each of the 3 most common classes; compare gives the paired change, McNemar's table and p, and changes by hardness and class; rebuild renders the run's reports again and diffs them, the `Generated:` line aside |
| FG-206 | Execution-guided retry | 3 | done |  |  | `--retry 1` asks once more, with the executor's error, for each query that fails to run (never one that ran), in the same `SandboxPool`, and scores the second reply; `Score` gains `attempts` and `first_try_ex` (ADR-0015); first replies kept in `eval/<split>-first-tries.jsonl` and judged in the one official batch; each retry drawn with its own seed; reports show EX with and without retry and the paired change |
| FG-207 | CPU smoke pipeline in CI | 3 | done |  |  | `make smoke` (`scripts/smoke/pipeline.sh`) runs `tests/smoke`, then `forge eval run` with the pinned tiny model on the fixture's 20 val questions and `forge report run`, in a scratch runs folder; a pass on a clean commit writes the git-ignored `.smoke-ok-<sha>`, and uncommitted work passes without one; CI `smoke` job on 3.12, Hugging Face cache keyed on the tiny revision, 10-minute bound, a required check; `forge eval run --workers`; I4 traces every number of a run report to its stored files (`tests/report_trace.py`); `scripts/smoke/phase_2.sh` |
| FG-291 | Local pilot baseline on Apple silicon | 3 | done |  | 2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7 | `local-mps` platform (ADR-0016), counted in neither the Kaggle quota nor GPU hours; `forge eval run --device` (`cpu` by default, `mps` refused on dev) and `--config` for `configs/eval/*.yaml`; Qwen2.5-Coder-0.5B-Instruct pinned as `pilot`; the full val run on the M4's GPU is in `docs/runs/` with its generated report, val only; a plan story since FG-294 |

## Phase 3: Free-GPU ops and baselines (v0.3.0)

| ID | Title | Pts | Status | PR | Run ids | Notes |
|---|---|---|---|---|---|---|
| FG-301 | Spike: Kaggle, Unsloth and vLLM on a T4 | 3 | done (2026-11-16) |  | 2026-11-16-throughput-spike-c8f4b41 | Built on CPU: the kernel metadata template and `check_kernel_metadata`, the self-contained spike `kaggle/spike/throughput.py` (env, secrets, downloads, installs, vLLM fp16 and GRPO benches, session-limit probe) and its bundler, the pins in `kaggle/requirements-gpu.txt`, ADR-0005. Ran on Kaggle on 2026-11-16: the GPU spike from c8f4b41 (kernel `forge-throughput-spike` version 2, 0.59 GPU h in the ledger), the session-limit probe (a CPU kernel pushed with `-t 900`) and a secret probe (`kaggle/spike/secret_probe.py`, a CPU version 3 of the spike's kernel); their reports are in `docs/spikes/kaggle-gpu/`, and every question of `docs/spikes/kaggle-gpu.md` is answered from them, field by field: a T4 x2 session on Python 3.13.16 and torch 2.11.0+cu130, vLLM 0.33.0 generating in fp16 but with no bitsandbytes, GRPO in 4-bit on the lock's TRL with the 4B at 4.94 GB peak, Unsloth faster but on trl 0.22.2, no Kaggle Secret in a pushed run, and a session killed at its timeout keeping its output. ADR-0006 accepted (TRL with transformers generation), ADR-0005 and ADR-0012 note what they waited on, and section 3.8's budget is worked out from the numbers (`test_budget_rows_follow_the_measurements`) |
| FG-302 | Code and data packaging as Kaggle datasets | 3 | done (2026-11-16) |  |  | Built and tested with the CLI mocked: `KaggleClient` runs kaggle as argument lists, `forge kaggle publish-code` (wheel at a clean HEAD, pinned `requirements-gpu.txt`, `GIT_SHA`; `version -m <sha> -r zip`, or `create` the first time) and `publish-data` (the 170 verified Spider files and `spider.sha256` as `<owner>/forge-spider`, CC BY-SA 4.0 attribution and citation, uploaded once). Both private, never `-u`. Published for real on 2026-11-16: `forge kaggle publish-data` created the private `hoomanesteki/forge-spider` from the 170 Spider files as five uploads (the databases as `database.zip`) plus `spider.sha256`, in 2 min 53 s, and the private `hoomanesteki/forge-code` version 1 holds c8f4b41's wheel, `requirements-gpu.txt` and `GIT_SHA` (`kaggle datasets files`); both outputs are in the PR, and the 50-question eval job mounted both. No `publish-wheels`: FG-301 measured the install (pip 392 s in a kernel, uv 96 s) and left a wheel dataset undecided |
| FG-303 | Kaggle job runner CLI | 5 | done (2026-11-16) |  | 2026-11-16-smoke-kaggle-50-val-c8f4b41 | Built and tested with the kaggle CLI mocked: `forge kaggle run` (smoke marker and clean tree, config checked as the kernel will, code dataset pinned by version with HEAD's `GIT_SHA` or published, ledger and `kaggle quota` budget with `--allow-over-budget` in the eval run's config, confirm, T4 push with a timeout of the estimate and half again, `runs/<run_id>/kernels.json`, `--resume`, `--job spike`, `--emit-notebook`), `status`, `logs`, `wait` and `output`; `kaggle/entry.py` (env, code, install, data, secrets, preflight, job, upload; `status.json` always); `ops` section in eval configs; `code_sha` from `FORGE_CODE_SHA` outside a checkout; Python 3.13 in CI. The real job ran on 2026-11-16 from c8f4b41: the 50-question eval (`configs/eval/smoke-kaggle-50-val.yaml`), pushed with `--run-id` since the default id's 57-character kernel title is over Kaggle's 50 (FG-398 fixes that), finished in a 0.18 h session, came back with `forge runs pull --store kaggle` and is in `docs/runs/` with its ledger row; its kernel-written report matches its rebuild on the laptop with the full data (`test_phase3_kaggle_eval_job_present`, `test_phase3_kaggle_eval_job_rebuilds`). Push, wait and output are in the PR |
| FG-304 | Hub run store and resume | 5 | done |  |  | `forge.ops.hub`: `RunStore` (plan 3.5's methods plus `put_folder`, `runs` and `prune_checkpoints`), `LocalRunStore` and `HubRunStore` on the private model repo through huggingface_hub 2.4 (`upload_file`, `upload_folder`, `list_repo_files`, `hf_hub_download`, `snapshot_download`, `delete_files`), each call tried 3 more times, 2, 4 and 8 s apart, on network errors; `latest_checkpoint` by step number; retention keeps the latest 2 `step-NNNNNN` and `best/`; `forge runs list|show|pull|push`, `pull --store kaggle` being `forge kaggle output`; `forge eval run --store hub` fetches an earlier attempt to resume and stores the run when it ends; the kernel's upload stage goes through the store. Tested with the Hub client faked. First real use on 2026-11-16, from the laptop, since a kernel gets no `HF_TOKEN` (FG-301): `forge runs push 2026-11-16-smoke-kaggle-50-val-c8f4b41` uploaded FG-303's pulled kernel run to `hoomanesteki/forge-runs` (`docs/runbook.md`, Run store) |
| FG-305 | Time guard and automatic ledger rows | 3 | done |  |  | `forge.ops.timeguard.Deadline` times steps on a monotonic clock and stops before one that might pass the deadline; `TimeGuardCallback` saves, stops, pushes `checkpoints/step-NNNNNN`, marks the stored manifest `stopped_time_guard` and prunes (tiny model on CPU, fake clock); `run_eval(deadline=)` stops between batches and resumes; manifests name their GPU (CUDA name, Apple chip) and count wall-clock GPU hours; `forge runs pull` adds one ledger row a run id under the append's lock; I7 on `docs/runs`, `runs/` and the ledger |
| FG-306 | vLLM inference backend | 3 | done |  |  | `forge.infer.vllm_backend.VllmBackend`: fp16 (`dtype="half"`), `max_model_len` 2048, 85% of GPU memory, LoRA (rank up to 64) only with an adapter, by one `LoRARequest` (a folder, or a Hub snapshot at its revision), 4-bit bitsandbytes allowed and nothing else; Qwen3 templates through `LLM.chat` with thinking off, others through `render_chat` as token ids; `sampling_kwargs` maps `GenParams` with a seed per example id; vLLM imported only when a backend is made. `forge eval run --backend vllm` in kernel jobs only; the runner's `model_backend` builds hf and vllm, and the kernel loads one model for its preflight and job. `kaggle/requirements-gpu.txt` is now vllm 0.33.0, torch 2.14.0, transformers 5.20.2 and the lock's trl, peft and the rest (the spike's `trl-vllm` stack). Tested with vLLM faked; FG-307 proves it on a T4 |
| FG-307 | GPU baselines and model choice | 3 | done (2026-11-17) |  | 2026-11-16-qwen3-1-7b-val-c8f4b41, 2026-11-16-qwen3-4b-val-c8f4b41, 2026-11-16-qwen2-5-coder-1-5b-val-c8f4b41, 2026-11-16-qwen2-5-coder-1-5b-p2-c8f4b41, 2026-11-16-qwen2-5-coder-1-5b-i6-c8f4b41, 2026-11-17-qwen3-8b-awq-val-2f77281, 2026-11-17-qwen2-5-coder-1-5b-dev-2f77281, 2026-11-17-qwen3-8b-awq-dev-d2d6a3a | Built and tested on CPU: candidates Qwen3-1.7B, Qwen3-4B and Qwen2.5-Coder-1.5B-Instruct and the 4-bit reference Qwen3-8B (`Qwen/Qwen3-8B-AWQ` with `quantization: awq` for vllm since FG-398), pinned with the Hub's license, gated flag and size (`scripts/check_models.py`), and a config per run with `ops.est_gpu_hours` (2.8 h in all); `forge kaggle run --reason` for dev jobs, logged at the push, with a val preflight in the kernel; I6 (`forge.eval.reproducibility.compare_runs`) on fake and tiny-model CPU runs, I5 on committed runs' stored verdicts; `gpu_story_runs` and `test_phase3_baseline_reports_present`; `scripts/smoke/phase_3.sh`. Ran on Kaggle, two kernels at a time, and committed in `docs/runs/` with their ledger rows and laptop-rendered reports: on 2026-11-16 from c8f4b41 the three candidates on val with p1, the coder with p2 and the I6 repeat (0.0 points apart); on 2026-11-17 from 2f77281 the reference on val with p2, the AWQ checkpoint's first run on a T4, and the chosen model on dev with p2 (dev look 1); from d2d6a3a the reference on dev with p2 (dev look 2), committed once I5 was settled: `scripts/check_i5_distinct.py` scored the 95 disagreements of FG-307's eight runs again with the evaluator's `DISTINCT` kept (`official_exec_match(keep_distinct=True)`) and none remained, so I5 is read over the baselines pooled, 98.54%, and for each of the chosen model's four runs alone, as ADR-0008's note of 2026-11-17 records. ADR-0012 accepted: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (official EX 65.3% on val), chosen under the stored scores and again with rule 3's cut queries read whole, where Qwen3-4B passes it but not significantly (`scripts/check_choice_extraction.py`, which now refuses a second or partial run of a candidate); its Measured section has the reference's val row and both dev EX (the chosen model 67.2%, the reference 70.8%), and nothing is chosen on dev. ADR-0007 accepted: p2 (+2.2 points, McNemar p = 0.0115 on clusters), fixed for every later phase, so the chosen model's dev config and both reference configs use p2. Section 3.8 redone at 1.7B's rates for the chosen model, with the ablations worked out by the formulas (40.79 h in all) and the runs' measured hours in its note; FG-301's and part A's review items fixed. `docs/runs/pending.yaml` is empty |

## Phase 4: Supervised fine-tuning (v0.4.0)

| ID | Title | Pts | Status | PR | Run ids | Notes |
|---|---|---|---|---|---|---|
| FG-401 | SFT dataset builder | 3 | done |  |  | `forge.train.sft_data`: TRL conversational prompt-completion rows (`prompt` from `fit_prompt`, `completion` the gold query in a sql block, `example_id`, `db_id`) from `load_train()` only, checked again by `assert_train_only`; repeats (casefolded, spacing collapsed), `FAILING_GOLD_IDS`, prompts over the limit and prompts that show a gold table without columns are left out and counted; `forge data build-sft --prompt-version --tokenizer [--out]` saves `data/sft/<sha256>/` with `meta.json`; `load_tokenizer` takes `<repo>@<revision>`. Masking checked through `SFTTrainer`'s own preparation with the tiny Qwen2.5 tokenizer |
| FG-402 | Training backend abstraction and configs | 5 | done |  |  | `forge.train.config`: strict pydantic `LoraConfig`, `SftConfig`, `GrpoConfig`, `OpsConfig` (and `run`, `data`) from one YAML file, refusing unknown or repeated keys, `max_completion_length` over 512, `num_generations` that doesn't divide the generation batch, any split but train, and Unsloth or 4-bit off CUDA; `forge.train.backends`: `TrainBackend`, `TrlBackend` (float32 on CPU; bf16 on MPS from macOS 14, else float32; bf16 on CUDA where the GPU has it, else fp16) and a lazily imported `UnslothBackend`, one trainer builder for both, `max_prompt_length` dropping GRPO rows before TRL, `select_backend`; `forge.train.callbacks`: `CheckpointCallback`, `MetricsCallback` (JSONL), optional W&B, `training_callbacks`. Unsloth itself untried: no training kernel installs it (ADR-0006, FG-398) |
| FG-403 | SFT trainer with CPU smoke | 3 | done |  |  | `forge train sft --config [--run-id] [--sft-dir] [--allow-dirty]` (`forge.train.sft.run_sft`): builds the rows with the model's tokenizer, keeps an existing `data/sft/<hash>/`, reads it in memory once it hashes to its name, trains with the job callbacks and saves `checkpoints/final`; the manifest's config holds the run, data, LoRA and SFT sections, `sft_data.hash` and the trainer's backend, device and dtype, under the config hash. Local runs are `local-cpu` or `local-mps` only. `configs/sft/`: tiny-smoke, pilot-qwen25coder05b-mps, qwen3-1p7b-pilot and qwen3-4b-main, at ADR-0010's values (written here, with `max_length` 2,048; FG-496 replaced the last two with qwen25coder-1p5b-pilot and qwen25coder-1p5b-main). `make smoke` trains 3 steps and evaluates the adapter on 10 val questions |
| FG-404 | SFT runs on Kaggle | 3 | blocked: nothing but the operator running its Kaggle runs in the runbook's order: the pilot's two sessions, its val run and promote, the main run, its checkpoints' val runs and promote (FG-496) |  |  | Built and tested on CPU: `ops.keep_checkpoints_at` saves the checkpoints at those shares of a run and keeps them from retention, named `kept_at_<share>` in the manifest (the main config keeps 0.5, beside `checkpoints/final`); `forge.train.sft.promote_best` and `forge runs promote <run_id> --eval ...` copy the checkpoint whose val run has the highest official EX to `checkpoints/best`, the later one on a tie, with `best.json`, in `runs/` and the run store, refusing dev and runs that differ in more than the adapter; `forge kaggle run --job sft` (the checked config in the options, a 2-step preflight, the host as platform, the time guard short of the push's timeout) and `--adapter` for eval jobs, whose kernel copies a `runs/<run_id>/checkpoints/<name>` adapter from the SFT kernel's output, mounted as a kernel source (FG-495); runbook "SFT runs"; `test_phase4_sft_runs_present` skips while FG-404 is in `docs/runs/pending.yaml`. Since FG-496 both configs train ADR-0012's model on p2 in fp16, `configs/sft/qwen25coder-1p5b-pilot.yaml` in two sessions (`ops.stop_after_step` 50, then `forge kaggle run --resume`) and `qwen25coder-1p5b-main.yaml` an epoch, and the test wants the pilot's two sessions. Pending: its 4 runs there, 4.4 GPU h |
| FG-405 | SFT results and comparison | 2 | blocked: the main-model comparison waits on FG-404's Kaggle SFT runs alone, to set beside FG-307's committed p2 val run of the chosen model, `2026-11-16-qwen2-5-coder-1-5b-p2-c8f4b41` |  |  | Done for the 0.5B local pilot on val (ADR-0016): FG-491's `forge report compare` of the base and SFT val runs, which a fresh compare of the committed runs matches but for its Generated line, is checked by `forge report rebuild` in `test_phase4_compare_report_rebuildable`, as is any later comparison in the folder of an SFT checkpoint's val run, each against a run of the same model and settings without the adapter; `docs/notes/sft.md` reads it in 5 bullets (overall, by hardness, the classes that fell and those that grew, hypotheses for GRPO), and `test_sft_notes_cite_only_report_numbers` finds each of its numbers in a report it links, 3 to 5 bullets a list; runbook "SFT runs" says how to make the main one. Pending: that comparison (`docs/runs/pending.yaml`) |
| FG-491 | Local SFT pilot on Apple silicon | 3 | done |  | 2026-11-16-qwen2-5-coder-0-5b-instruct-sft-db0d8d8, 2026-11-16-qwen2-5-coder-0-5b-instruct-eval-db0d8d8 | `configs/sft/pilot-qwen25coder05b-mps.yaml`: LoRA SFT of the pilot model on `mps` in bf16, `max_length` 1,536 with gradient checkpointing, a row at a time padded to 64 tokens and 16 a step, on 3,168 train rows drawn by hash (`sft.subset_rows`), as a 20-step probe put the epoch past 2.5 h; `forge train sft` leaves out rows over `max_length` rather than letting TRL cut them, and `sft_data` in the manifest counts them and the rows trained; `forge ledger record <run_id>` writes a local run's row from its manifest; the adapter's full val run (FG-291's config with `--adapter`), its report and its comparison with the base pilot are in `docs/runs/`, the training run without weights; val only (ADR-0016). The run store holds the SFT run, its adapter in `checkpoints/final` with it, since the push of 2026-11-17 (`docs/runs/README.md`) |

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
| FG-292 | Gate 2 fixes: run provenance | done | A run's commit and dirty check cover the whole checkout `FORGE_ROOT` names, wherever the run starts, and outside a checkout `forge eval run` prints one `error:` line; run locks are gitignored under `docs/runs`; `make smoke` prints its report and writes the marker only with every phase (PHASE unset or 8); `--split train --limit N` takes the first N in file order, train_spider before the rowless train_others, while val and dev stay in id order; `HfBackend` checks adapter and dtype before loading, and a test ends rows of one batch at different steps; `forge eval run --help` names the first-tries file |
| FG-293 | Gate 2 fixes: statistics and reports | done | McNemar's exact test pairs gold clusters, which key on gold queries with whitespace collapsed; the taxonomy reads a whole ORDER BY alias as SQLite does and drops derived-table and CTE qualifiers; `rebuild` masks only the Generated line's time, fails without the chart and compares its bytes where the renderer its PNG names is this machine's; `compare --out` that resolves to the default path is the default; a spy shows retried SQL runs in `SandboxPool`; the pilot's report regenerated |
| FG-294 | Gate 2 fixes: plan and pilot notes | done | FG-291 gets a Phase 2 story block whose Tests line names its 15 tests, so traceability covers 22 Phase 0 to 2 stories, and moves to the Phase 2 table; section 5.5 lets a pilot or found story hold `FG-N91`, with gate fixes from the next free number; the pilot's unfenced replies and rule-3 cuts counted; the Phase 2 demo says `make smoke` prints its report from a scratch folder |
| FG-295 | Gate 2 fixes: last items | done | The taxonomy reads a whole ORDER BY alias in parentheses or with COLLATE, drops parentheses around a whole ORDER BY term, and lets a derived table's or CTE's columns win over an alias; `rebuild` skips a chart only for a renderer-shaped Software text, and the full-data pilot test skips rather than pass when its chart isn't compared; `compare --out` naming the default file through a symlink or letter case writes that file; `git_sha` excludes the logs where `forge.ops.paths` puts them; the pilot's rule-3 note recounted by `scripts/count_pilot_extraction.py`; `GitError` hand-off for Phase 3 |
| FG-391 | Gate 3 fixes: UTC dates and ledger sessions | done | One UTC date for run ids, ledger rows, dev-usage rows and the ledger's week (`forge.ops.clock`), and runner tests that take the UTC date, with one pinned ten hours west of UTC; each attempt at a run is a session in its `sessions.json`, a manifest's GPU hours add up its sessions, and each session gets its own ledger row, marked `(session N)` after the first, which `append_once` keys on and I7 checks; a kernel's session spans the whole kernel (`status.json`, `sessions.json`); `forge kaggle wait` and `output` add the rows |
| FG-392 | Gate 3 fixes: kernel jobs | done | A kernel's eval job stops at a deadline counted from the kernel's start to the push's timeout (`--timeout-s`), a tenth kept back for official EX and the upload, then uploads and resumes; `forge eval run --deadline-minutes` does the same; the restore is the first stage, so a resume that fails early still carries the run; every stage result, the error and `error.txt` are redacted, with secrets cleaned as huggingface_hub sends them; the data is checked against the wheel's pinned list; a dev look is logged as soon as Kaggle accepts the push, one row a run id across resumes (ADR-0004 notes); a push that dropped a source is recorded with a warning; run ids that make no kernel slug and `--resume --emit-notebook` are refused before any upload; the runbook says when the code dataset is published |
| FG-393 | Gate 3 fixes: run store | done | Hub downloads retry network failures, and `get_manifest` tells a missing run (`RunNotFound`) from an outage (`StoreUnavailable`, with its cause); only refused or dropped connections, timeouts, 408, 429 and 5xx count, a `RuntimeError` only with such a cause; each failure is retried by huggingface_hub or by the store, measured with the real client on an in-process transport (at most 14 s of the store's waits, or 23 s of the client's unless the Hub's answer names a wait), and both retry only when a download's first metadata request is refused, times out or gets a status the client retries, and the client's retries of it drop their connections, 106 s of waits at most (`client_retried`); a retried pull starts from an empty folder; `--store hub`, `forge runs push` and the kernel's upload compare the two copies' sessions (`forge.ops.run_copies`), fetch the store's when it is newer, and refuse copies that went separate ways; a pull and `forge kaggle output` leave no file the new copy lacks but `kernels.json`, `kernel/` and the lock; `publish-data` counts only a ready dataset as there, uploads a new version over a failed or deleted one, refuses one Kaggle is still processing, and removes `build/kaggle-data` after the upload (`--keep-build`) |
| FG-394 | Gate 3 fixes: throughput spike | done | Fixed and tested on CPU with fakes, as FG-301 was; the spike itself still waits on the Kaggle account (it ran on 2026-11-16, FG-301). The `trl` stack's venv is seeded with pip, which counts the image's torch where uv does not, and the install fails if a torch lands in the venv; every bench runs with `CUDA_VISIBLE_DEVICES=0` while `env` sees all GPUs, and a GRPO bench records `n_gpu` and the completions TRL scored a step; GRPO prompts are rendered with thinking off for every stack (a template without the switch fails the stage), with the mean completion length; `vllm:Qwen/Qwen3-8B:bnb` loads the 4-bit reference as FG-306's backend asks; each stage kind has a time cap (`stage_caps_minutes`) and the GRPO stacks run before vLLM's; in an IPython kernel the baked argv wins, children are marked with `FORGE_SPIKE_CHILD`, `launch` is written before parsing, and the children run this script's file or the cell's source written out |
| FG-395 | Gate 3 fixes: time guard and vLLM backend | done | `TimeGuardCallback` turns evaluation off at its stop step and at that epoch's end, so `save_strategy="best"` can't drop the guard's save and no unbudgeted evaluation runs, and the epoch's end asks for the save again if something cleared it (tested on the tiny model); the vLLM backend renders every prompt with `render_chat` and sends `LLM.generate` the token ids the hf backend feeds its model, Qwen3's too, never `LLM.chat`, compared with the tiny model's real tokenizer on its own and a Qwen3-style template; `check_params` refuses a temperature between 0 and 0.01, which vLLM would raise to 0.01; `gpu_memory_utilization` is an eval option for vllm (config file or `--gpu-memory-utilization`), in the manifest's config only when set; the p2 run is for the model ADR-0012's rule chooses, as plan section 3.7 says (ADR-0007, `pending.yaml`, the runbook, `test_artifacts.py`) |
| FG-396 | Gate 3 fixes: last items | done | A push Kaggle started is never called refused (kaggle 2.2.5's versionless "successfully pushed" included, recorded with no version), and its dev look and its record are each written whatever the other raises; a resume whose kernel source Kaggle dropped is marked in `kernels.json` and the next resume mounts the attempt before it; a new run's kernel that failed before its job gets a note quoting its `status.json` span for `forge ledger add`; publish-data removes its bundle after a failed upload too; the run store retries what huggingface_hub didn't, told by the request's method, path and failure (a dropped download, a 5xx outside 500/502/503/504), and states the `Retry-After` worst case; a pull keeps this machine's `kernels.json` and `kernel/` unless the store's record lists more pushes; the kernel redacts every log record (stderr tested with a retried canary); the spike's 4-bit load runs right after vLLM's install under a 10-minute cap, and the budget's last 20 minutes are kept for its download, install and load; docstrings and comments corrected |
| FG-397 | Kaggle's answer for a missing dataset | done | The first real `forge kaggle publish-data` stopped at `kaggle datasets status`: for a slug in the owner's own namespace that holds no dataset, the live API answers 403, "You don't have access to this resource.", where the client expected the 404 "not found" the research notes describe, so a first publish could never create its dataset. Both answers now mean "no dataset yet", tested with the live API's exact text |
| FG-398 | Kaggle live-run fixes | done | What the first live Kaggle runs on 2026-11-16 showed, each fixed after a test that failed first, with the kaggle CLI and vLLM faked: a kernel title or slug over Kaggle's 50 characters is refused before any push, and a run whose `forge-<run_id>` would pass it gets the run id cut at a dash and 8 hex digits of its sha256 (`kernel_name`), found by the run id through `kernels.json` as before (FG-303's 57-character case and its resume tested); training jobs install `kaggle/requirements-train-gpu.txt`, the spike's `trl` stack with no vLLM, published beside the eval list, and the install stage names the file; the 4-bit reference is `Qwen/Qwen3-8B-AWQ` at 4da05a8 with `quantization: awq` (`scripts/check_models.py` ok on 2026-11-16), bitsandbytes refused in one line, its estimates kept as no run has measured it; a push whose CLI fails is checked with `kaggle kernels status`, and counts as made, recorded with its dev look and a warning, when Kaggle has the attempt's kernel |
| FG-492 | Gate 4 fixes: run checks | done | One parser of a checkpoint adapter, `forge.ops.adapters.checkpoint_ref` (`<folder>/<run_id>/checkpoints/<name>`, `final`, `best` or a step, a trailing `/` or a leading `./` alike), for `forge kaggle run --adapter`, which pushes it spelled the one way and refuses before any upload a checkpoint outside `runs/`, a folder on this machine, a bad name or a run this machine knows trained another model, for the kernel's fetch and for `forge runs promote`; an eval run with an adapter records its files' `adapter_sha256` in its outputs (a local folder, or the Hub repo's two files), a resume must find the same files, and an SFT run records `adapter_sha256:<checkpoint>` for each checkpoint it keeps, which promote checks; `forge runs promote --config <base val config>` and `checkpoint_scores(base=...)` take only eval runs of the base model's full val config with the checkpoint's files alone added, and `test_phase4_sft_runs_present` checks the same, with fixture runs in `test_sft_runs_are_scored_on_their_base_models_full_val`; `test_phase4_compare_report_rebuildable` and `test_sft_notes_cite_only_report_numbers` look for the main model's comparison (the val run of the main run's best/ against FG-307's run of its config) and skip only while FG-405 is in `docs/runs/pending.yaml`; a kept checkpoint whose push fails waits out of the trainer's rotation and is pushed again after the next push that goes through and at the end, tested on the tiny model; ADR-0017 records the `TrainBackend` signatures and `TrlBackend` on cpu, mps and cuda in plan section 3.5 and FG-402; `make smoke` fails at the pilots' commit db0d8d8, which their `run/` tags keep, on one stale assertion: there `tests/smoke/test_sft_smoke.py` expects `sft_data` to be `{"hash": meta.hash, "rows": meta.counts.rows}`, while that commit's `src/forge/train/sft.py` writes `over_max_length` and `trained` too, as the committed pilot manifest shows, and the test in this tree expects all four; `docs/runs/README.md` quotes both from the tag, and the runs stand, a known limit |
| FG-493 | Gate 4 fixes: ADR and notes | done | ADR-0010 corrected with a dated note: `tiny-smoke.yaml` takes the starting values by omission, the base trains in float32 on the CPU and on MPS in bfloat16 from macOS 14 (float32 before), as `precision_for` and the pilot's manifest say, and the pilot's length, batch split, padding and subset are quoted from its config and manifest, which `test_adr_0010_says_what_the_configs_and_the_pilot_did` checks; `test_sft_notes_cite_only_report_numbers` finds each number of a bullet on the report row or section named before it, in the report's order, against the report its list reads, and the notes no longer say every class past schema linking grew or that all where, projection, group_order and value failures run; `forge data build-sft`, `forge train sft` and `forge runs promote` without the cpu extra print one `error:` line naming it; `check_targets` takes PEFT's suffix targets such as `self_attn.q_proj`; a section with nothing under it is called empty; `run_sft` checks the tree once and its manifest names that commit; `wandb/` is ignored; `forge ledger record` says to run it again for a run still going, and the ledger header names it; the stale FG-403 note on `date.today()` is marked fixed by FG-391 |
| FG-494 | Gate 4 fixes: last items | done | `forge runs promote` and `checkpoint_scores` refuse a base config at another prompt version than the SFT run trained on; `forge kaggle run --adapter` refuses a path in `runs/` that names no checkpoint, such as `runs/<run_id>`, whatever this machine's folders hold; a module missing inside an installed cpu extra package is named as Python names it; `test_adr_0010_says_what_the_configs_and_the_pilot_did` compares each MPS branch with `precision_for` on both sides of macOS 14 and takes the note's numbers from the config's and manifest's values, and a second test shows swapped branches and wrong counts fail it; the notes test says what it leaves unchecked; the SFT pilot's adapter digest is recorded in `docs/runs/pending.yaml` and `docs/runs/README.md`, and `forge runs push --adapter-sha256` uploads nothing unless the run's adapter matches it; the README and FG-492's note cite only the `run/` tags and files in the tree for the stale smoke assertion at db0d8d8; the FG-404, FG-405 and FG-491 blockers say what blocks them now: the main SFT run trains ADR-0012's choice and the pilot Qwen3-1.7B, both on p2, as the runbook's "SFT runs", the main config's comment and a note in ADR-0012 say, and FG-405's comparison waits on FG-404's runs alone; the FG-404 budget test and the runbook tell the story's 4.5 h from plan section 3.8's 4.4 h; PLAN.md says "built with all tables" |
| FG-495 | Adapters from kernel output | done | A kernel pushed with the CLI can't read Kaggle Secrets, so a checkpoint adapter travels as the training kernel's output: `forge.ops.kaggle_job.adapter_kernel_sources`, the one lookup GRPO jobs will reuse, puts the run's last kernel that holds it, at the version its `kernels.json` records, in the eval job's `kernel_sources`, and refuses before the push a run made anywhere but on Kaggle (the local pilot), one this machine has no `kernels.json` for, a checkpoint the run's manifest says it didn't keep and a `best` with no `best.json`; `best` is pushed as the checkpoint `best.json` names, and the manifest's `adapter_sha256:<checkpoint>` goes as `--adapter-sha256`; the kernel copies the checkpoint's adapter files from a mounted output (identical copies as one), reads the run store only with an `HF_TOKEN`, fails naming both places otherwise and fails on another digest; a resume that mounts the adapter's kernel too warns that a dropped source may be either; a store-less kernel SFT job, traced end to end, already left every checkpoint the run keeps (the latest 2 steps, each kept share, `final`) at `runs/<run_id>/checkpoints/<name>`, now tested from the SFT kernel through `forge kaggle output` into an eval kernel; the mount probe is in `docs/spikes/kaggle-gpu/`; runbook "SFT runs" and `--adapter`, an ADR-0005 note and the runs README say so |
| FG-496 | Chosen-model SFT configs and an SFT resume | done | The Phase 4 gate's decisions of 2026-11-17: `configs/sft/qwen25coder-1p5b-pilot.yaml` and `qwen25coder-1p5b-main.yaml` train `Qwen/Qwen2.5-Coder-1.5B-Instruct` at its pinned commit on p2, in fp16 with no 4-bit, at ADR-0010's values, with plan section 3.8's estimates (1 h and 2.5 h); the pilot takes 1,600 rows, 100 steps, and its first session stops after step 50 (`ops.stop_after_step`, the time guard's stop at a set step) so a second, `forge kaggle run --resume`, rehearses the restore and finishes; the Qwen3 configs are gone. SFT runs resume: `forge train sft --resume` and `forge kaggle run --job sft --resume` carry a run on from its latest checkpoint (`TrainBackend.sft(resume_from=...)`, ADR-0017's note), refusing another commit, config hash, data or platform, a finished run and a checkpoint without trainer state; every SFT run keeps `sessions.json`, its GPU hours its sessions', and metrics past the checkpoint go; on the tiny model two sessions end with the adapter one session trains, bit for bit. `docs/runs/pending.yaml`, the runbook's "SFT runs" (each command in order), ADR-0012's consequence and a second note, ADR-0010's T4 row and note, plan section 3.8's pilot row, the runs README and the 1.7B p2 config's comment name the chosen-model pilot; the FG-404 budget test reads section 3.8's 4.4 h; `test_phase4_sft_runs_present` wants a ledger row a session and the pilot's two sessions |

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
- `--split train --limit N` takes the first N questions in the loader's file order. By id,
  train_others sorts first, so before FG-292 a limit up to 1,659 scored only its rowless
  databases. On the full data the first 4,511 train questions in file order are all on databases
  with rows, and music_2's 100 come next. Val and dev keep id order, so the committed pilot is
  unchanged. FG-505's curriculum starts from a limited train run (FG-292).
- Resuming a dev run calls `load_dev` again, which logs a second row with the same run id, since
  every call is a look (ADR-0004). FG-304 could let a resumed run keep its first look (FG-202).
- Without `--run-id`, `make_run_id` gives two evals of one model on the same commit and day the same
  id, whatever their splits, so the second is refused as a run with other options; pass `--run-id`.
  A finished run is never resumed or overwritten (FG-202).
- A run folder holds the `.lock` file that keeps a second process off it; FG-304's Hub mirror should
  leave it out (FG-202).
- The `.lock` stays after the run or report that held the folder, so `.gitignore` ignores it
  under `docs/runs` as well as `/runs`: a report written for a committed run leaves the tree
  clean (FG-292).
- Before FG-292 `git_sha` ran git in the current directory, so a run started in a subfolder
  checked only that folder for changes, and one started outside the repo named the commit of
  whatever checkout held the cwd, or crashed, not `FORGE_ROOT`'s. It now checks the whole
  checkout from its top, with the append-only logs excluded by top-anchored pathspecs, and
  raises `GitError` outside one (FG-292). Each exclude is the log's real path
  (`ledger_path`, `dev_usage_path`) relative to the top, so a Forge root below the top of a
  bigger checkout still runs clean after a run appends to its logs, and a `docs/dev-usage.md`
  at that top is no log (FG-295). git spells that path itself, so a `FORGE_ROOT` typed in
  other letter case on macOS keeps the excludes, and a Forge root whose `docs/PLAN.md` the
  checkout around it doesn't track (copied into another repository, untracked or ignored
  there) raises `GitError` instead of recording that repository's commit (FG-295).
- `git_sha` raises `GitError`, a `RuntimeError`, outside a checkout or where git can't run,
  and no longer `subprocess.CalledProcessError` or `OSError`, so every caller must catch
  `GitError`. `forge kaggle publish-code` and `forge kaggle run` caught `CalledProcessError`
  until the Phase 3 stack was replayed onto the gate 2 fixes; they catch `GitError` now and
  print one `error:` line outside a checkout (FG-292, FG-295, FG-303). The eval runner reads
  the commit at the Forge root, or, on a Kaggle kernel with no Forge root, from
  `FORGE_CODE_SHA` (FG-303).
- The scores and generations files hold the contract types only, so the executor's error message
  for a failed prediction is not stored. FG-204 classifies from the live result in the runner;
  FG-205's failure examples can show the class, or a later story can store the message (FG-202).
- `forge eval run` records `platform: local-cpu`, `build_manifest`'s default, whatever the device;
  FG-303 should set `kaggle` for runs there (FG-202).
- Spider asks most questions twice in other words with one gold query, so `summarize` and
  `compare` take `clusters`, and `gold_clusters` puts each such pair in one (val: 872 questions
  in 455 clusters). FG-205 should pass them for val and dev, or the intervals come out too
  narrow (FG-203).
- Gold queries of one pair can differ in spacing alone, so `gold_clusters` compares them with
  each run of whitespace as one space; on val that joins one pair the raw text split. With
  clusters, McNemar's exact test counts the clusters each run gets more right and drops ties,
  as the intervals draw clusters; reports keep the table of questions and say what the test
  paired (FG-293).
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
- The taxonomy reads a whole ORDER BY term that names a select alias as the alias, as SQLite
  does even when a column has that name, and drops the names of derived tables and CTEs from
  their columns. All 8,656 gold queries still classify as `other` against themselves, and of
  the pilot's 191 stored classes that the two queries decide, none changes with the fix, so its
  stored scores stand (FG-293).
- A whole ORDER BY term is also the alias in parentheses or with COLLATE, which SQLite skips
  before it looks for one, and parentheses around a whole ORDER BY term are dropped. A
  derived table or CTE has the columns its select list names (or its column list), and those
  win over an alias in WHERE, GROUP BY, HAVING and inside ORDER BY expressions, as a table's
  do; behind a `*` a name it may have is left as written. All 8,659 train and val gold
  queries still classify as `other` against themselves, the canonical forms of 2 change
  (spider-train-06103 and 06104 lose parentheses around an ORDER BY term), and none of the
  pilot's 191 classes the queries decide changes, so its stored scores and report stand
  (FG-295).
- A query nested in a condition counts in that condition, so a filter whose subquery is missing or
  picks another column or aggregate is `where`. Dropping the condition that held a table's only use
  is `wrong_tables`, since tables come first (FG-204).
- `error_class` follows fast EX, since a score is written before the official batch runs, so a
  prediction that only the official evaluator accepts keeps a class. FG-205's taxonomy table should
  count the failures of the metric it reports (FG-204).
- Reports read questions and gold queries from `FORGE_DATA_DIR`, which must hash to the run's
  data manifest hash, so a Kaggle run's report needs the same Spider files where it renders
  (FG-303, FG-304). On full val (872 questions) `forge report run` took 1.1 s on the laptop,
  `compare` 1.9 s and `rebuild` 1.6 s (FG-205).
- A dev run's report shows answers the run's own look already saw, so it needs the run's row in
  `docs/dev-usage.md`, refuses without it, and adds none; ADR-0004 now says so. FG-303 and FG-304
  have to bring a Kaggle dev run's row home before its report renders, and FG-508's dev compare
  reports spend no extra looks (FG-205).
- Example failures show the error class and whether the query ran, not the executor's message,
  which the scores and generations don't store (FG-202). Storing it means a contract change or
  a file of its own (FG-205).
- Error classes count the failures of official EX. A failure only official EX finds has no class,
  since a run classifies the misses of fast EX, so reports count it as `unclassified`; a compare
  report gives a paired change for official EX only (FG-205).
- `rebuild` diffs the markdown, masking only a `Generated:` line that holds the time as the
  report writes it, and fails when the chart is missing. The chart's bytes repeat on one machine,
  in matplotlib's default style whatever a matplotlibrc says, but CI's Linux draws other bytes
  than the Mac with the same releases. So the PNG's `Software` text names the releases and the
  platform that drew it, and `rebuild` compares the bytes only where that is the same and says
  so of a chart drawn elsewhere. Render a committed run's reports where its rebuild will run, as
  the pilot's were on the Mac (FG-205, FG-293). Only a Software text in `chart_renderer`'s form
  is skipped, so a chart an image editor saved is compared and fails, and the full-data pilot
  test skips, naming both renderers, where the chart isn't compared (FG-295).
- `rebuild <b>` checks the compare reports in run b's `eval` folder; one written with `--out`
  isn't checked and doesn't say it is, unless `--out` names that folder's file, through '..', a
  symlink or other letter case where the file system ignores it, and then that file is written.
  FG-307 and FG-405 commit reports under `docs/runs/`, so their checks should diff those copies
  against the run store's or rebuild in place (FG-205, FG-293, FG-295).
- For FG-207's I4: numbers from the stored files are in a report's title, its table rows and the
  headings over example failures, and in some prose lines: in a run report the checkers'
  disagreement and the questions and clusters in the line on intervals; in a compare report the
  questions and samples both runs answered, the questions and clusters on intervals, the
  questions they disagree on, the clusters each wins and McNemar's p-value. Table headers hold
  95%, the other prose lines only constants (1,000 resamples and seed 0, the 5 failures of 3
  classes, and Spider's citation: 1.0, 2018 and 4.0), and the `Generated:` line the time and
  nothing more. Question lines and code blocks quote the data and the generations, and cluster
  counts come from `gold_clusters` over the scored questions (FG-205, FG-293).
- Official EX can match a first query the executor rejects: the evaluator drops `DISTINCT`
  and turns `> =` into `>=` before running, so `SELECT count(DISTINCT *) FROM Aircraft`, a
  syntax error to SQLite, matches `SELECT count(*) FROM Aircraft`. EX without retry therefore
  judges each retried sample's first query in the official batch instead of counting it a miss,
  and retry can lose a sample official EX had (ADR-0015, FG-206).
- FG-508 plans a `--retry 1` dev run "using the same generations plus one retry turn (counted
  within the same dev use)". `forge eval run --retry 1` generates its first replies again and
  calls `load_dev`, which logs another look, so FG-508 needs a way to retry a stored run's
  failures from its generations without a second look (FG-206).
- A retried sample's tokens and latency are split between its first try and its generation, so
  FG-705's cost and latency table has to add both for a retry run. FG-304's Hub store must carry
  `eval/<split>-first-tries.jsonl` with the other outputs; the manifest lists it as
  `first_tries` (FG-206).
- A retry is drawn with a seed derived from the run's seed and the sample index, and a backend
  seeds each reply by its example id and its index among the prompt's replies, so FG-306's vLLM
  backend has to seed requests the same way (`sample_seed`) for retries to stay independent of
  first replies and of each other (FG-206).
- `make smoke` takes 40 to 50 s on the laptop, 25 to 33 s of it the smoke tests and 6 to 10 s the
  tiny model's 20 replies. The first CI smoke job took 57 s, setup and the tiny model's download
  included (7.5 MB once cached), and `make smoke` 38 s of it, the replies 1.9 s.
  `scripts/smoke/phase_2.sh` takes about 32 s on the full data (FG-207).
- The smoke run passes `--allow-dirty`, so it also runs on uncommitted work, but it writes the
  marker only when the run's manifest names HEAD's commit without `-dirty`. Each attempt first
  removes HEAD's marker and a pass removes the others, so a marker means the last smoke on that
  commit passed. FG-303 checks for `.smoke-ok-<HEAD sha>` (FG-207).
- A smoke with `PHASE` below 8, as `make gate PHASE=N` runs it, skips later phases' smoke tests,
  so it passes without a marker and says so; run `make smoke` for one. The smoke prints its
  report before the scratch folder goes, so a gate's demo can paste it (FG-292).
- The tiny model's 256-token replies hold no SQL, so on the smoke run every sample is `no_sql`,
  and with `--retry 1` each is asked again and fails the same way. The smoke proves the wiring of
  that path; I4's fake-backend run covers results of every other kind (FG-207).
- Two prose lines of a run report hold numbers from the scores, not constants: the checkers'
  disagreement, and the questions and clusters in the line on intervals. I4 traces both (FG-207).
- I4 covers a run report. A compare report's numbers (McNemar's table and p-value, the changes
  by hardness and class) aren't traced yet; FG-801's I4 for the results page can build on
  `tests/report_trace.py` (FG-207).
- `forge eval run --workers N` sizes the run's sandbox pool (by default one per CPU, up to 4).
  The manifest leaves it out, since it changes no score; `make smoke` uses 2 (FG-207).
- Before FG-291 `forge eval run` built `HfBackend` with `device="auto"`, so on a Mac the
  model ran on MPS while the manifest said `local-cpu`. `--device` now defaults to `cpu`, and
  `mps` is asked for by name, recorded in the config and filed as `local-mps` (FG-291).
- `forge eval run --config <yaml>` loads the file into click's default map: each value is
  checked by its option's type, and options on the command line override it. A file may set
  only the options that change a score; the run id, store and dev reason stay on the command
  line (FG-291). A file may also hold an `ops` section (`est_gpu_hours`), which
  `forge eval run` checks and ignores; `forge kaggle run --job eval --config` reads both
  with `read_config_sections` (FG-303).
- Report headers name the run's platform, so a pilot can't pass for a T4 run; I4's trace
  checks the row (FG-291).
- A pilot's numbers are a 0.5B model's in float32 on MPS through transformers, so they say
  roughly where a small model starts, not where the T4 baselines (float16, vLLM) will
  (ADR-0016, FG-291).
- The pilot ran on MPS with no fallback to the CPU; its ledger row has the hours, taken from
  the manifest's start and end (FG-291).
- In the pilot's report `schema_link` is the most common class of official EX failures, well
  ahead of `wrong_tables`: the 0.5B model mostly names tables and columns its database
  doesn't have. Worth watching in FG-307's baselines and in what SFT and GRPO change (FG-291).
- Counted by `scripts/count_pilot_extraction.py` from the pilot's committed generations and
  scores: 763 of its 872 replies have no ```` ```sql ```` block although p1 asks for one.
  `extract_sql` read 761 of them by rule 3, the last line that starts with SELECT or WITH and
  parses as a query, and 2 have no such line and are `no_sql`. In 12 the query ran over
  several lines and rule 3 kept only the first, the select list without its FROM, so all 12
  fail as `schema_link`. Run whole, 7 of them still name a column their database doesn't
  have, so extraction alone made 5 of that class's 266: 4 run and 1 fails on a misused
  aggregate, and 1 of the 5 (spider-train-01806) matches gold, the one official EX match
  rule 3 cost. A prompt or model that leaves the fence off loses multi-line queries this way,
  which FG-307's p1/p2 choice (ADR-0007) should count apart from schema errors. The
  extraction rules are unchanged (FG-294, FG-295).
- `docs/runs/<run_id>/` holds a run whole (manifest, config, generations, scores, report and
  chart), so `FORGE_RUNS_DIR=docs/runs forge report rebuild <run_id>` checks it, and
  `tests/integration/test_artifacts.py` checks the pilot offline and, with the full data, its
  rebuild and I4 trace. FG-307's `test_phase3_baseline_reports_present` can join that file
  (FG-291).
- Kaggle's newest GPU image (v173, released 2026-10-30) has Python 3.13 by its Dockerfile, and
  `pyproject.toml` requires Python below 3.13, so FG-302's wheel would not install there. The
  spike's `env` stage reads the live version; if it is 3.13, widen `requires-python` and add 3.13
  to CI, as ADR-0002's revisit condition says (FG-301). FG-303 did both, ahead of the spike: every
  package CI installs has a 3.13 wheel, and re-locking added only `audioop-lts` for the demo extra.
- The same image moved from CUDA 12.8 to CUDA 13 (torch 2.11.0+cu130), so wheels built for CUDA 12,
  bitsandbytes' among them, may not load. The spike's installs show it (FG-301).
- No one environment holds Forge's lock and vLLM: vllm 0.33.0 pins torch 2.14.0 and transformers
  below 5.21, while `uv.lock` has torch 2.15.0 and transformers 5.23.0. FG-306 either runs vLLM in
  an environment of its own or moves transformers back to 5.20; the spike measures both ways
  (`vllm` and `trl-vllm` stacks). Unsloth 2026.11.4 caps trl at 1.15.0, which can't train on CPU,
  so ADR-0006 (proposed) keeps TRL on the GPU and Unsloth optional (FG-301). ADR-0006 was
  accepted on 2026-11-16 with TRL on the GPU (FG-301).
- TRL 1.18 casts a 4-bit model's LoRA adapters to bf16 (issue 7776), which fp16 training on a T4
  can't unscale. The spike casts trainable parameters back to fp32 after building the trainer;
  FG-402's TRL backend needs the same on the T4, and `fp16=True` with `bf16=False` (issue 7889)
  (FG-301).
- `kaggle kernels push` sends only the code file, and Kaggle starts a script kernel without
  arguments. The spike is one self-contained file, and `forge.ops.kaggle.bake_argv` writes its
  arguments into it; FG-303's `kaggle/entry.py` can render its job arguments the same way. The
  spike's report records how Kaggle launched the script (`launch`), which the entry script needs
  (FG-301).
- Kaggle Secrets can only be attached in the web editor. The spike pushes its kernel once with
  `--no-run`, has the secret attached, then runs it, to learn whether a CLI-pushed version can read
  it (FG-301, ADR-0005).
- Integration checkpoint 3 asks every FG-301 run for a manifest, but the spike runs without Forge
  installed and `RunManifest.kind` has no spike kind. Its JSON report and ledger row stand in for a
  manifest unless a story adds the kind, which would change a contract (FG-301).
- Plan sections 3.12 and 3.13 still name Unsloth as the GPU trainer. ADR-0006 makes TRL the default
  once FG-301's numbers confirm it; update both rows then (FG-301). ADR-0006 was accepted on
  2026-11-16, and the rows change at Gate 3, as the note on sections 3.12 and 3.13 below says.
  Gate 3 changed them on 2026-11-17.
- The spike's session-limit probe is a CPU kernel killed by its own `-t 900` timeout, so it spends no
  GPU quota. That it behaves like the 12-hour cut is an assumption the run can't prove (FG-301).
- kaggle 2.2.5's `datasets create` and `version` print a refused upload, such as a slug already in
  use, as "Dataset creation error: ..." and exit 0, so `KaggleClient` reads their output as well as
  the exit code. `kernels push` does the same with "not valid dataset sources ... could not be
  added", which would start a job without its code or data; FG-303 should treat that line as a
  failure (FG-302).
- An API token, the runbook's setup, doesn't name its user: the kaggle CLI asks the server. The
  owner of Forge's datasets is `FORGE_KAGGLE_USER`, else `KAGGLE_USERNAME` or `kaggle.json`, and
  the runbook now says to set it (FG-302).
- `-r zip` uploads `database/` as one `database.zip`. Whether Kaggle mounts it unpacked as
  `database/<db>/<db>.sqlite` under `/kaggle/input` is not seen yet, so FG-303's entry script
  should find `spider.sha256` by glob, point `FORGE_DATA_DIR` at its folder and verify the files
  before a job starts (FG-302).
- `kaggle datasets version` returns before Kaggle has processed the upload. FG-303 should wait for
  `kaggle datasets status <owner>/forge-code` to say `ready` before pushing a kernel that mounts
  it (FG-302).
- The code dataset's wheel is installed with `--no-deps`, so the packages Forge itself needs
  (sqlglot, sqlparse, httpx2, statsmodels and the rest) have to be on Kaggle's image or in
  `requirements-gpu.txt`. Which of them the image lacks is not seen yet; the entry script's
  install stage names each one missing or out of range and stops before the job, and its fix is
  a pin at the `uv.lock` version (FG-303).
- `kernels push` prints the kernel's URL, and `forge kaggle run` records the kernel it names. How
  long a kernel title may be is not documented; a long model name makes a long run id and title,
  and the first real push will show whether Kaggle takes it (FG-303).
- An eval job on Kaggle refuses `backend: hf`, which runs on the CPU and would leave the T4 idle,
  so the story's real 50-question job waits for FG-306's vllm backend as well as the account
  (FG-303).
- `--allow-over-budget` is in the eval run's manifest config, as the plan asks, so it is part of
  the config hash: a resumed run has to be pushed with the same flag, and `--resume` refuses
  other options before anything is uploaded (FG-303).
- `RunManifest` has no field for where `git_sha` came from. A `kaggle` or `colab` platform means
  the code dataset's `GIT_SHA`, passed as `FORGE_CODE_SHA`; the runbook says so rather than
  changing the contract (FG-303).
- The entry script exits 0 when its job fails, as the spike does, because a kernel that ends in
  an error might not keep its output. `forge kaggle wait` therefore reads the run's `status.json`
  after `COMPLETE`, and exits 0 only for `finished` (FG-303).
- `upload_folder` only adds files, so a finished run's Hub copy kept its failed attempt's
  `error.txt`. `put_folder(..., replace=True)` passes `delete_patterns` and is used where the
  folder holds the whole run: the end of `forge eval run --store hub` and `forge runs push`. The
  kernel's upload stage only adds, since a kernel that failed early may hold part of a run (FG-304).
- With hf_xet installed, huggingface_hub 2.4's `upload_folder` commits a large folder in parts
  (256 files a commit to start, and a commit at least every 5 minutes while files go up), so a
  folder is one commit only while it is small, as runs and LoRA checkpoints are. Calling it again
  resumes (FG-304).
- huggingface_hub reports a failed LFS or Xet upload as a bare `RuntimeError` (it wraps the cause),
  so the run store retries `RuntimeError` along with transport errors, 408, 429 and 5xx (FG-304).
- `RunStore` has three methods beyond plan 3.5's five, and takes `str | Path` for local paths.
  FG-305's time guard and FG-504 should push a checkpoint with `put_folder(run_id, folder,
  checkpoint_dir(step))`, then call `prune_checkpoints(run_id)` (FG-304).
- The kernel writes `status.json` after its upload stage, so a run pulled from the Hub has none;
  `forge kaggle wait` reads it from the kernel output. FG-305's ledger row comes from the manifest,
  which the Hub copy has (FG-304).
- A resumed run keeps its first `started_at`, so its wall-clock GPU hours include the time
  between sessions, and a ledger row is written once a run id, so a row pulled after the
  first session misses the later ones. FG-507's two-session GRPO run is the first to hit
  this; it needs per-session hours or rows (FG-305). Since FG-391 each attempt is a session
  with its own hours and row.
- `kaggle/entry.py` gives eval jobs no `Deadline` yet, and would have to skip the report of
  a run the guard stopped, since `read_run` refuses unfinished runs. A kernel's own timeout
  (the estimate and half again) stops an eval job meanwhile (FG-305). Since FG-392 the push
  passes that timeout to the entry script, whose eval job stops at the time guard first and
  writes its report only once the run finishes.
- A manifest that `kaggle/entry.py` closes as failed after a crash keeps `gpu_hours` 0; its
  ledger row is still right, since a pulled row takes its hours from `started_at` and
  `ended_at` (FG-305).
- The FG-291 pilot's manifest predates GPU naming (`gpu` null, `gpu_hours` 0); its ledger
  row holds the chip and hours, and the committed manifest stays as made (FG-305).
- vllm 0.33.0 pins torch 2.14.0 and needs transformers below 5.21, so GPU jobs run
  transformers 5.20.2 and torch 2.14.0 while CI runs 5.23.0 and 2.15.0. The transformers code
  Forge calls in a kernel was read in the 5.20.2 source: `apply_chat_template` differs only in
  `sanitize_control_tokens` and in dropping `None` message keys, neither of which Forge uses,
  and the trainer callback module is the same file. CI never runs 5.20.2, so a later story
  that calls more of transformers in a kernel has to check it the same way (FG-306).
- Installing the GPU list replaces the image's torch 2.11.0 with 2.14.0 and its CUDA 13
  wheels, several GB a job; FG-301's install stage shows how long it takes, and whether
  `publish-wheels` pays (FG-306).
- When a call passes `SamplingParams`, vLLM takes only the end token ids from a checkpoint's
  `generation_config.json`, so no shipped temperature or top-k reaches a reply; FG-201's
  question is answered (FG-306).
- vLLM seeds a request's n samples with its seed, the seed plus 1 and so on, so one seed for
  every prompt would give every question's sample i the same draws. The backend seeds each
  prompt from the run's seed and its example id. Its random stream is vLLM's, so the same
  seed draws other samples on hf and vllm (FG-306).
- A vllm run's manifest config keeps `device: cpu`, the option's default, as a fake run's
  does: the option says where hf runs, and the platform and `gpu` say where a kernel job
  ran. Off a kernel host the backend is refused, as a GPU run there would be `local-cpu`
  (FG-306).
- `VllmBackend` takes `quantization="bitsandbytes"`, but no eval option sets it yet, so
  FG-307's 4-bit reference model needs one, in the manifest config since it changes a score
  (FG-306).
- The runner sends vLLM 32 questions a call, and a call waits for its longest reply, so the
  T4 idles at the end of each batch. A retry prompt near 1,536 tokens after a 256-token first
  reply can also leave the second reply less than 256 tokens before `max_model_len` 2048.
  FG-307 can measure both and raise the batch size or the length (FG-306).
- The plan's Qwen2.5-Coder-3B-Instruct has the Hub license `other`, named `qwen-research`:
  the Qwen Research License, non-commercial only. Qwen2.5-Coder-1.5B-Instruct (Apache-2.0)
  takes its place among the candidates, and `test_models_config.py` keeps the 3B out
  (FG-307).
- `forge kaggle run` couldn't push a dev eval: a config file can't carry the reason, and
  in a kernel `load_dev` would have failed to find `docs/dev-usage.md`, since `repo_root`
  needs a checkout. The push now logs the look, the kernel logs its own in its output
  (`FORGE_DEV_USAGE`), and a dev job's preflight answers val, so dev loads once a push. A
  dev job has no `--emit-notebook` (FG-307).
- `VllmBackend` took a `quantization` that nothing passed it, so the 8B reference couldn't
  load in 4-bit from a config. `quantization` is now an eval option, in the manifest's
  config only when set, so earlier runs keep their config hash (FG-307).
- One model's runs from one commit on one day share a default run id
  (`<date>-<model>-eval-<sha7>`), and the second push is refused. The runbook's
  "Baselines" names each run `<date>-<config name>-<sha7>`; config names are spelled as run
  ids, without dots, for that (FG-307).
- `docs/runs/pending.yaml` keeps FG-301's one entry per run, with `since` added to every
  entry. FG-307's entries list their `configs`, and those that run one of them (the chosen
  model's p2 and dev; the p2 was the top candidate's before FG-395) say `pick`, so their
  estimate is the dearest (FG-307). FG-301's entries left it on 2026-11-16, and FG-307's
  val runs and their picks once the runs were committed and the model chosen (FG-307).
- `test_every_eval_config_is_runnable` refused `split: dev` in every config; it now does
  for pilots (`device: mps`), and checks a vllm config with a kernel host and a dev config
  with a reason, as `forge kaggle run` does (FG-307).
- `scripts/count_extraction.py` counts what `scripts/count_pilot_extraction.py` (FG-295,
  on the Gate 2 branch) counts for the pilot, 763 unfenced, 761 by rule 3, 2 with no SQL
  and 12 cut, for any run, so ADR-0007 can compare p1 and p2; FG-295's also scores the cut
  queries whole. When the stacks meet, one script can absorb the other (FG-307).
- I6's CPU half runs the tiny model from `configs/models.yaml` in `tests/integration`,
  whose CI job had downloaded no model before and has no Hugging Face cache: about 4 MB per
  job (FG-307).
- The estimates in FG-307's configs (0.15 to 0.5 GPU h a job) assume about a quarter hour
  of install, download and engine start per kernel; FG-301's measurements should replace
  them before the first push, as the kernel's timeout is the estimate and half again
  (FG-307).
- The ledger keeps the seven columns plan 3.8 names, so a later session names itself at the
  end of its row's purpose, `(session 2)`, and a row that names none is its run's first:
  every committed row stays valid. A row added by hand for a session that never recorded
  its end has to name it the same way for I7 to count it (FG-391).
- A Kaggle run's ledger rows count each whole kernel session, while its manifest's
  `gpu_hours` adds up the job's own spans, so the ledger's hours are the larger by the
  setup, preflight and upload. The Hub copy is uploaded before the kernel ends, so a run
  pulled only from the Hub misses the upload's minutes; `forge kaggle wait` logs the whole
  session first (FG-391).
- A kernel that fails before its job (install, data, preflight) holds no run, so no
  manifest says what to log; its minutes still need `forge ledger add` (FG-391). Since
  FG-392 that holds for a new run's kernel only: a resume restores the run first, so its
  early failure is a session of the run, with its row.
- Training runs (FG-403, FG-504) don't go through `forge.eval.run_files`, so they keep no
  `sessions.json` themselves. In a kernel `entry.py` adds each kernel's session, but on the
  laptop a resumed training run is one session from its first start, and
  `TimeGuardCallback`'s `close` counts from there, until those jobs call
  `forge.ops.sessions.begin` and `end` (FG-391).
- Hand-off for the Phase 4 replay: since FG-391 `forge.ops.ledger.record_run(manifest,
  sessions=None, path=None)` returns `(rows, notes)`, the rows it appended and a note a
  session, where it returned `(row or None, note)`; a run gets a row for each session that
  has ended, and `append_once` keys a row on its run id and session number, the
  `(session N)` its purpose ends with after the first. FG-491's `forge ledger record`
  (exp/fg-491-local-pilot-sft) still does `_, note = record_run(manifest)`: on the replay
  it must pass `sessions_of(manifest, run_dir)` and echo each note, as `cli._log_run`
  does, or a resumed local-mps run is one row from its first start to its last end
  (FG-391, FG-396).
- Five runner tests read the local date, so west of UTC they failed every evening while
  CI, in UTC, passed. The far-west test stops the clock on a Sunday evening ten hours west
  (`TZ=HST10`, a POSIX zone that needs no zone database), where the local date is the day
  before the UTC one (FG-391).
- An eval deadline guards the batches, but official EX, the report and the upload come after
  the last one, so `eval_deadline` keeps a tenth of the time back for them: official EX
  grows with the run, as the estimate and so the timeout do. Kaggle's timeout may count
  from before the script starts, which the tenth covers too. FG-301's measurements show
  whether a tenth is enough (FG-392).
- `status.json` and `sessions.json` keep whole seconds, so two attempts started within one
  second (only in tests) looked like one; the entry script now tells the session its job
  began from those the restore brought back by counting them (FG-392).
- A resume whose kernel source Kaggle dropped fails at its restore stage, and its output
  holds no run, so the resume after it mounts nothing to restore either. Such a run needs
  a new `--run-id`, unless a later story has `plan_job` mount the last attempt whose
  output holds the run (FG-392). Since FG-396 `kernels.json` marks that attempt
  (`kernel_source_dropped`), and the next resume mounts the one before it.
- A dev look is one row a run id only for `forge kaggle run --resume`: the push can't tell
  whether the attempt before it reached dev. On the laptop `load_dev` still logs each load,
  a resume's included, as ADR-0004 decided (FG-392).
- huggingface_hub 2.4 retries most requests itself, 5 more times 1, 2, 4, 8 and 8 s apart, but
  sends a commit and the repo lookup that starts `snapshot_download` once, and its downloads
  turn an unreachable Hub into `LocalEntryNotFoundError` with the frames of the error cut off.
  So the store tells what the client retried by the path of the request that failed
  (`forge.ops.hub.client_retried`); an upgrade of huggingface_hub has to pass
  `test_store_and_client_retries_never_stack`, which runs the real client (FG-393). The
  path was not enough: a download's metadata request is retried only after a refused
  connection, a timeout or one of the client's statuses, and the download itself only on
  its own terms, so since FG-396 the method and the failure count too.
- `snapshot_download` that can't reach the Hub returns a `local_dir` that isn't empty as if it
  were complete, so a retried pull could have passed a half download for a whole run; each try
  now starts from an empty folder (FG-393).
- Which copy of a run is newer comes from its sessions' start times. A kernel resume restores
  from the last kernel's output, not from the store, so a laptop attempt made in between makes
  the two go separate ways; the kernel's upload is then refused, recorded in `status.json`, and
  a person keeps one copy (FG-393).
- A pull now removes a report made only on this machine, as it removes any file the store's
  copy lacks; `forge report run` makes it again (FG-393).
- publish-data uploads a new version over a failed or deleted `forge-spider`, which no real
  Kaggle account has tried yet; the first real publish (`docs/runs/pending.yaml`, FG-302) shows
  whether Kaggle takes a version there or wants the dataset deleted first (FG-393). That
  publish, on 2026-11-16, created the dataset, so the new-version path is still untried
  (FG-302).
- Under `save_strategy="best"` transformers 5.23 sets `should_save` to whether a step's
  evaluation is a new best after every callback's `on_step_end`, so a callback keeps its save
  only by turning that evaluation off. The guard can, since the Trainer runs its own
  `DefaultFlowCallback` before the callbacks it is given (FG-395).
- A temperature between 0 and 0.01 is now refused on hf too, where it sampled as asked: a
  config decodes alike on both backends, so 0.01 is the lowest sampling temperature a run
  takes (FG-395).
- `gpu_memory_utilization` is also a `forge eval run` option, since a config file's keys are
  that command's defaults; with hf it is refused, as vLLM's share of a GPU means nothing on
  the CPU or MPS (FG-395).
- The plan's FG-307 block still says "the top candidate with p2", against section 3.7's "for
  the chosen model"; ADR-0007 records why section 3.7 holds. The p2 run now waits for
  FG-301's memory check, as the chosen model's dev run does (FG-395).
- kaggle 2.2.5 prints "Kernel version successfully pushed" with no number when Kaggle
  sends none, and the kernel runs all the same. Such a push is recorded with no version,
  and `status`, `logs`, `output` and a resume take the kernel's latest, which is that
  push unless another came after it (FG-396).
- The kernel redacts log records through the record factory every logger uses, so
  libraries with handlers of their own (huggingface_hub's, transformers') are covered
  too. What a child process prints, pip or vLLM's engine, is not a record and goes to the
  log as it is (FG-396).
- The spike keeps its last 20 minutes for the 4-bit reference's download, install and
  load, a guess like the caps, so the GRPO benches get 60 of the 80. If FG-301's session
  still ends without the 4-bit answer, a second push with `argv: [run, --stacks]` runs
  vLLM's stages alone (docs/spikes/kaggle-gpu.md) (FG-396). The session got its answer,
  vLLM 0.33.0 has no bitsandbytes, so no second push was made (FG-301).
- `check_config` still takes `quantization` for any vllm model, though FG-306's block keeps
  4-bit loading for reference models above 4B; the manifest records it, so such a run is
  visible (G3-03, left as it was by FG-396).
- A run pulled from the Hub before `forge kaggle wait` or `output` logs its last session
  up to the upload only, and `append_once` then keeps that shorter row; the runbook says
  to wait first (G3-10, left as it was by FG-396).
- Kaggle refuses a kernel title over 50 characters with a bare `400 Client Error: Bad
  Request` from `SaveKernel`, and a kernel's title is `Forge <run_id>`, so a run id over 44
  characters can't be pushed: the default id of `Qwen/Qwen2.5-Coder-1.5B-Instruct`, and the
  runbook's `--run-id` pattern for the I6 repeat (45). The runbook says to shorten them until
  FG-398 checks the limit and shortens the kernel's name (FG-301, FG-303). Done in FG-398,
  and the runbook's advice to shorten them is gone.
- A Kaggle Secret attached in the editor never reaches a version pushed from the CLI: the
  secrets service answers 400 for it as for a label that doesn't exist, and `kaggle_secrets`
  raises that as the builtin `ConnectionError`. Kernel jobs run without `HF_TOKEN`, and the
  laptop uploads their runs (ADR-0005, FG-301).
- vLLM 0.33.0 has no bitsandbytes quantization, so FG-306's 4-bit allowance and FG-307's
  reference configs can't load, and TRL's colocated vLLM can't take a 4-bit policy. FG-398
  moves the reference to AWQ and gives training jobs the `trl` stack (ADR-0006, FG-301).
  Both done in FG-398.
- Kaggle runs at most two GPU sessions at a time ("Maximum batch GPU session count of 2
  reached."), so FG-307 pushes its baselines two at a time (FG-301).
- In a kernel, `pip install` of `requirements-gpu.txt` over the image took 392 s, where uv
  installs the same list into an empty venv in 96 s; with the preflight, an eval job spends
  about 11 minutes before its job starts. A faster install (uv, or wheels in a dataset) is
  left to FG-398 or a later story (FG-301). FG-398 left it: kernels still install with pip,
  so a later story takes it.
- `test_phase3_baseline_reports_rebuild` expects one check from each run's rebuild
  (`[()]`), but `rebuild` returns the report's and the chart's, so it fails once FG-307's
  runs land; it should check the report's diff and accept a chart that is compared equal or
  skipped, as `test_phase3_kaggle_eval_job_rebuilds` does (FG-303).
- Plan sections 3.12 and 3.13 still give Python 3.11 or 3.12 and Unsloth for GPU training,
  against the measured Python 3.13.16 image and ADR-0006's TRL stack; they change at Gate 3
  (FG-301). Gate 3 changed them on 2026-11-17 (`docs/gates/phase-3.md`).
- The Hub counts `Qwen/Qwen3-8B-AWQ`'s safetensors parameters as 8,190,735,360, the same
  as `Qwen/Qwen3-8B`'s, its packed 4-bit weights as 6,945,767,424 I32 values beside
  1,244,967,936 BF16 ones, so `params` in `configs/models.yaml` can't tell a checkpoint from
  its quantized copy; the repo and revision do (FG-398).
- The reference's estimates, 0.45 GPU h on val and 0.50 on dev, are below the 0.55 that plan
  section 3.8's formula gives the 4B candidate, and nothing measured backs them: no run has
  downloaded, loaded or timed the AWQ checkpoint on a T4. Its kernel's timeout is 1.5 times
  the estimate, so a slow load or slow AWQ kernels would stop its job at the time guard, to
  be resumed. Its own run should redo both estimates (FG-398).
- `kaggle kernels status` for a kernel Kaggle doesn't have is a 404 or, as kaggle 2.2.5
  turns a 403 into it, "Cannot access kernel ... (Permission 'kernels.get' was denied)";
  which of them the live API answers for a private slug that holds nothing is not measured,
  so `find_kernel` takes either (FG-398).
- When a push's CLI fails and `kaggle kernels status` fails too, `forge kaggle run` can't
  tell whether Kaggle saved the kernel: it records nothing and says to check before pushing
  again. If Kaggle has it, a dev job's kernel then reads dev without its row in
  `docs/dev-usage.md`, and Forge has no command to log a look by hand (FG-398).
- A failed push of the spike is not checked against Kaggle: its one kernel gets a new
  version each push, and `kaggle kernels status` names no version (FG-398).
- Phase 4: the training configs (from FG-402 on, plan section 3.10's `model_ref` and
  `prompt_version`) must name `Qwen/Qwen2.5-Coder-1.5B-Instruct` at the commit
  `configs/models.yaml` pins, with prompt `p2` (ADR-0012, ADR-0007). That is Phase 4's own
  change; FG-307 changes no training config (FG-307).
- Rule 3 read 726 of Qwen3-4B's 872 val replies with p1, which had no code block, against 7
  of Qwen3-1.7B's and 106 of the coder's. The 4B wrote most of them over several lines, so it cut
  328 of its queries by `scripts/count_extraction.py`'s reading (the joined lines parse as one
  query) and 389 by `scripts/count_pilot_extraction.py`'s (up to a blank line or a fence).
  Read whole, 206 of the 389 match gold, which puts the 4B at 66.7% against the coder's 65.3%;
  the choice holds either way (ADR-0012), and the extraction rule stays as ADR-0007 decided
  (FG-307).
- Section 3.8's eval rows are on the high side: by their ledger rows FG-307's runs took 0.15 h
  to 0.25 h each, where the figures that section's Baselines line lists, from its formula with
  the 1.5 margin and the reference's config estimates, run from 0.30 h to 0.55 h a run. Its
  note gives the runs' total beside the row's estimate, and
  `test_baseline_runs_wait_in_pending_within_the_story_budget` checks it against the ledger
  (FG-307).
- Qwen3-1.7B's val run is close to I5's 98% on its own: its report's checkers disagree on 17
  of its 872 samples (FG-307).
- The candidates ADR-0012 didn't choose keep their `-val-p2` and `-dev` configs, which never
  run; `test_models_config.py` counts them as such. A later story can delete them with the
  `pick` handling in `docs/runs/pending.yaml`'s header (FG-307).
- A kernel's pip install of `requirements-gpu.txt` took from 247.7 s (the p2 run) to 446.4 s
  (Qwen3-1.7B's) over the nine kernel eval jobs committed (each `status.json`), where section
  3.8's eval formula takes the 50-question job's 391.6 s (FG-307).
- The reference's dev run missed I5 as `test_i5_agreement_on_committed_runs` first read it,
  run by run: its report's checkers disagree on 25 of its 1,034 samples (only fast EX matches
  13, and only official EX 12). Plan section 4 states I5 as at least 98% "on stored baseline
  outputs", and the product owner settled the reading on 2026-11-17 (ADR-0008's note): FG-307's
  runs pooled, and each run of the chosen model alone, with every committed run's agreement
  still reported. The test now reads it so, and the plan's table is unchanged (FG-307).
- Every disagreement between the checkers on FG-307's eight runs, 95 of 6,528 samples, comes
  from the evaluator dropping `DISTINCT`: scored again with `keep_distinct` on, none remains,
  and each gets its stored verdict when scored again as the run scored it
  (`scripts/check_i5_distinct.py`, quoted in ADR-0008). The reference's replies hold a
  `DISTINCT` the evaluator drops more often than the candidates' (170 of 872 on val and 215
  of 1,034 on dev, against 59 to 91 of 872 for the candidates). So the fast comparator has no
  fault there for GRPO's reward to inherit, but rewards and official EX part on `DISTINCT`,
  which FG-601's audit should watch in what training changes (FG-307).
- `official_exec_match` takes `keep_distinct` for that check only; every reported number keeps
  the evaluator's default, which drops it, for comparability with published Spider numbers
  (FG-307).
- `test_i5_agreement_on_committed_runs` finds FG-307's runs with `test_artifacts.py`'s
  `gpu_story_runs`, by the configs they were made from, and the chosen model's by the chosen
  dev run's model (FG-307).
- The AWQ reference loads and scores on a T4 through vLLM 0.33.0: by their ledger rows its val
  run took 0.21 h and its dev run 0.25 h, against its configs' 0.45 and 0.50, which FG-398 left
  to its own run to check (FG-307).
- The reference's val run and the chosen model's dev run were made from `2f77281` and the
  reference's dev run from `d2d6a3a`, commits of FG-307's branch, not of `dev`, so each needs
  a `run/<run_id>` tag on its commit once the branch lands (`docs/runs/README.md`) (FG-307).
- `forge data build-sft` on the full train split with the pilot's tokenizer
  (Qwen2.5-Coder-0.5B-Instruct at its pinned commit), at p1 and at p2: 7,787 questions, 25 asked
  again on the same database (9 repeat it exactly, 16 differ only in case), 3 with gold that
  fails on SQLite, none over 1,536 tokens, and 23 whose fitted prompt shows a table the gold
  reads without its columns: the baseball_1 prompts the FG-105 note above found. That leaves
  7,736 rows. It took 7 s at p1 and 9 s at p2 on the laptop (FG-401).
- TRL 1.18's `SFTConfig.max_length` defaults to 1,024, below the 1,536-token prompt limit. With
  the pilot's tokenizer 333 p1 rows and 1,364 p2 rows are longer than 1,024 tokens (the longest
  is 1,615, the longest reply 205), and `truncation_mode="keep_start"` would cut their replies
  and the end-of-turn token, or drop the row when nothing of the reply is left. No row passes
  2,048, so FG-402 and FG-403 must set ADR-0010's 2,048 (FG-401).
- 2,046 of the 7,787 train gold queries end in `;`, and many keep Spider's spacing
  (`T1.a  =  T2.b`). The completions keep the gold as it is, so the model learns both; the
  executor runs a query with a trailing `;` (FG-401).
- The SFT dataset's hash is in its `meta.json`. `RunManifest.data_manifest_hash` holds the
  Spider files' hash, so FG-403 has to put the SFT hash in the run's config or outputs (FG-401).
- `test_assistant_only_labels` is a unit test, as the plan names it, and loads the tiny model.
  The unit CI jobs have no Hugging Face cache, unlike smoke, so each of the three downloads it
  (about 21 MB); a cache step like the smoke job's would save that (FG-401).
- `TrainBackend.sft` and `grpo` take `output_dir` and `seed` as keywords, and `load` takes
  `seed`, since LoRA draws its first weights before any trainer seeds; plan section 3.5's
  signatures have no place for them (FG-402). ADR-0017 puts them there (FG-492).
- The train config has no `ops.hub_run_repo`: the run store is the one `FORGE_RUN_STORE`
  names, as for eval, so one setting names it. `run.init_adapter`, `data.curriculum`,
  `data.filter_by_sft_pass_rate` and `rewards` are refused until FG-504, FG-505 and FG-501
  add them (FG-402).
- PyYAML reads `5e-6` as a string (YAML 1.1 wants `5.0e-6`). The train config loader reads it
  as a float; `forge.eval.config` still uses `yaml.safe_load`, so `temperature: 7e-1` is
  refused there as text (FG-402).
- unsloth 2026.11.4 caps `trl<=1.15.0` (which can't train on CPU or MPS), `transformers<=5.17`
  and `datasets<5`, so it can't share `uv.lock`. `UnslothBackend` is tested with a stand-in
  module only; its install and first real run were to come with the first GPU job (FG-404). TRL
  casts a 4-bit model's adapters to bf16, which fp16 can't unscale on a T4 (TRL issue 7776),
  so `TrlBackend` casts them back to fp32 (FG-402). Superseded by ADR-0006, as accepted,
  which keeps Unsloth out until a release on TRL 1.16 or later: a training kernel installs
  FG-398's `kaggle/requirements-train-gpu.txt`, which has no unsloth, so no GPU job installs
  it and `select_backend("auto")` picks `TrlBackend` on Kaggle (ADR-0017's note, FG-492).
- `SFTTrainer` tokenizes with `datasets.map`, which writes four `cache-*.arrow` files into
  the folder of a dataset opened with `load_from_disk`, so `data/sft/<hash>/` would gain
  files. FG-403 should open it with `keep_in_memory=True`, which writes none (FG-402).
- A periodic checkpoint whose push still fails on the network after the store's retries is
  logged, and training goes on; any other store error stops the job. The time guard's own
  push always raises, as before (FG-402).
- `forge kaggle run --job sft` is still refused, now naming FG-404: `kaggle/entry.py` runs
  eval jobs only, and `run_sft` makes local runs only. A kernel's SFT job needs the host as
  the manifest's platform and the Hub run store for its step checkpoints. Then the run's
  local folder lacks those checkpoints, so pushing it with `forge runs push`, which replaces
  the store's copy, would delete them (FG-403).
- PEFT writes the base model's repo into `adapter_config.json` without its revision
  (`revision: null`), so only the run manifest's `model_ref` pins the base an adapter was
  trained on. `forge eval run --adapter` loads the base from `--model`, so evaluation is
  unaffected; an adapter published on its own should carry the revision (FG-403).
- No SFT run writes `checkpoints/best/`: training reads train only, so the trainer has no
  eval set, and `best/` is the checkpoint val picks once each is evaluated, in FG-404 (FG-403).
- Five tests in `tests/unit/test_kaggle_runner.py` build the expected run id from
  `date.today()`, in local time, while `plan_job` dates run ids in UTC, so they fail on a
  machine west of UTC between UTC midnight and local midnight. CI runs in UTC; `TZ=UTC make
  check` passes locally (FG-403). Fixed by FG-391, which this note came after on the replay:
  they take the UTC date (`_today()`), so they pass in any time zone (FG-493).
- On the laptop (16 GB, about 8 GB of it held by a virtual machine), PyTorch's MPS memory
  grew past what was free when nothing capped it: the first SFT probes reached 9 to 11 GB
  and stalled in swap. `PYTORCH_MPS_HIGH_WATERMARK_RATIO` caps it as a share of the
  11.8 GiB Metal recommends. Under 0.6 (7.1 GiB), micro-batches of 2 ran out of memory at
  step 12 of a probe, in TRL's fused LM head backward; micro-batches of 1 fit, and trained
  faster. A micro-batch of one 1,536-token row ran out under 0.3 (3.55 GiB) and fit under
  0.4, holding 4.4 GiB, so the run takes 0.45 with `PYTORCH_MPS_LOW_WATERMARK_RATIO` 0.35,
  which frees the cache sooner (FG-491).
- Unpadded, micro-batches of 1 kept the MPS allocator under its cap, but the process still
  grew, to 11 GB by step 32 of a run, as PyTorch's MPS backend keeps what it builds for each
  new tensor shape, and nearly every row brought a new length. Padding each row to a multiple
  of 64 tokens (`sft.pad_to_multiple_of`) leaves at most 24 lengths; the committed run's
  process held between 5.7 and 6.3 GB from step 15 to step 170 (FG-491).
- The 20-step timing probe of the committed config: 44.1 s a step from step 2 to 20 (18.8 to
  97.8 s), about 7,600 tokens a step, so an epoch of the 7,683 rows that fit, 481 steps,
  would take about 5.9 h. The pilot trains 3,168 of them, 198 steps. The laptop was shared
  meanwhile (a virtual machine near 200% CPU, another checkout's tests), and an unpadded
  probe in a quieter hour took 35.9 s a step. The run's own pace is in its `metrics.jsonl`
  and manifest (FG-491).
- Three starts of the pilot run before the committed one were stopped and their folders
  removed: one beside `make check`, which slowed it to about 60 s a step, one under the 0.6
  cap while the virtual machine was busy, which swapped from its first steps, and one
  unpadded under the 0.45 cap, which grew as the note above says and took 80 to 90 s a
  step by step 30. Nothing else should run on the laptop during an MPS run (FG-491).
- 53 of the 7,736 p1 rows are longer than 1,536 tokens with the pilot's tokenizer, prompt and
  reply as `SFTTrainer` encodes them; TRL's `keep_start` would have cut their replies. They
  are left out and counted in the manifest. At FG-404's 2,048 none is (FG-401) (FG-491).
- transformers 5.23 renamed `gradient_checkpointing` to `activation_checkpointing` and warns
  on the old name, which TRL 1.18 still sets by default; the old name works as its alias, so
  Forge passes that one. FG-404 should check the name in a kernel's transformers 5.20.2
  (FG-491).
- The adapter trained on the bf16 base and is scored on the float32 base, as FG-291's base
  run was, so the two val runs differ in the adapter alone. Their comparison is
  `docs/runs/2026-11-16-qwen2-5-coder-0-5b-instruct-eval-db0d8d8/eval/compare-2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7.md`:
  a 0.5B model after 198 steps on 3,168 train rows, a pilot on val only, not FG-404's
  result (FG-491).
- The SFT pilot's adapter is only in `runs/` on the laptop that trained it; the committed
  run holds its manifest, config and metrics. `docs/runs/pending.yaml` keeps the push to the
  run store until the Hugging Face account exists (FG-491).
- `docs/runs` now holds a training run, so the tests that read each committed run as an eval
  run (I5 on committed runs, `gpu_story_runs`) take only eval runs (FG-491).
- Retention kept a run's latest 2 step checkpoints, so the main SFT run's checkpoint half way
  (step 242 of the 484 that 7,736 rows make at 16 a step) would have gone at step 300.
  `prune_checkpoints` now takes `keep_steps`, which both training callbacks pass for
  `ops.keep_checkpoints_at` (FG-404).
- A kernel's time guard counts `ops.time_guard_minutes` (11 h 15 min) from the job's start,
  while `forge kaggle run` stops the kernel at its estimate and half again, 3.75 h for the
  main SFT run. A job that slow is stopped on purpose (plan section 5.3), but Kaggle would
  stop it without the guard's checkpoint and manifest. The entry script has the push's
  timeout since FG-392, so a kernel's SFT job counts its guard only up to that timeout,
  from the kernel's start, less the tenth an eval job keeps back, here for the guard's last
  save and the upload, when that comes before `ops.time_guard_minutes` (FG-404).
- The kernel's SFT job keeps its step checkpoints in its own run folder and puts the whole
  run in the store at the end: an SFT run can't be resumed, so a checkpoint on the Hub before
  then would serve nothing. GRPO's resume (FG-504) needs the callbacks to push to the Hub
  store mid-run (FG-404).
- The run store is one Hub repo of many runs, and vLLM's `LoRARequest` takes a folder or a
  whole repo, so a kernel's eval job fetches a `runs/<run_id>/checkpoints/<name>` adapter
  alone with `HubRunStore.get_folder`; such a job needs the `HF_TOKEN` secret (FG-404).
- An SFT job's preflight trains its own adapter, so the job loads the model again for a fresh
  one, a minute or two of a 4-bit 4B's kernel time, after `gc.collect()` and
  `torch.cuda.empty_cache()` give the preflight's memory back (FG-404).
- `forge kaggle run --adapter` scores a checkpoint with its base model's FG-307 config, so the
  base and SFT val runs differ in the adapter alone, as `forge report compare` in FG-405 needs
  and as FG-491's local runs did (FG-404).
- FG-491 had already committed the comparison FG-405 names. `forge report compare` run
  again on the committed runs writes the same file but for its Generated line, so the
  committed file stays as it is rather than take a new time (FG-405).
- I4 (`tests/report_trace.py`) traces the numbers of a run report only. A compare report's
  are checked by `forge report rebuild`, which renders them again with the same code, but
  nothing recomputes them on its own as I4 does for a run report. FG-801's results page,
  which quotes comparisons, could use such a trace (FG-405).
- A compare report counts each run's failures by class, not how questions move from one
  class to another, so `docs/notes/sft.md` can't say whether the pilot's rise in the where,
  projection, group_order and value classes comes from questions whose schema linking SFT
  fixed. A table of each question's class in run a against run b would say; FG-508's
  comparison of base, SFT and GRPO could add one (FG-405).
- A run stores each failure's class when it is scored, and FG-293 and FG-295 changed the
  taxonomy after FG-291's base pilot ran, so the two pilot runs were classed by different
  code. `forge.eval.taxonomy.classify` at the SFT run's commit, which is today's, puts none
  of either run's failures in another class, so the comparison's class rows hold. Nothing
  checks this for a comparison of runs from different commits; `forge report compare`
  could (FG-405).
- FG-405's two named checks run the pilot's checks first and then skip while FG-405 is in
  `docs/runs/pending.yaml`, so a gate sees them skipped rather than passed until the main
  comparison is committed; a failing pilot check still fails them (FG-492).
- An eval of a Hub adapter fetches its two files with `snapshot_download` before the run
  starts, to hash them, into the cache PEFT and vLLM load it from, so a missing or private
  repo fails before any file is written (FG-492).
- FG-491's pilot runs come from before adapter digests and stay as they are:
  `test_sft_pilot_run_is_committed_without_weights` still pins the SFT pilot's outputs, and
  its eval run has no `adapter_sha256`, so nothing ties those scores to the laptop's
  `checkpoints/final` but the path until the run store holds it (FG-492).
- A kept checkpoint the store never took stays in `unpushed-checkpoint-<step>` in the
  trainer's output folder, which `forge train sft` makes temporary. Its store is local and
  never fails on the network, but GRPO (FG-504), which pushes to the Hub mid-run, has to
  keep or upload that folder when `CheckpointCallback.unpushed` isn't empty at the end
  (FG-492).
- `forge runs promote` takes `--config`, the base model's full val config, so the runbook's
  SFT steps name `configs/eval/qwen3-1-7b-val.yaml` and `configs/eval/qwen3-4b-val.yaml`;
  if ADR-0012 or ADR-0007 pick another model or prompt, the config changes with them
  (FG-492).
- `forge runs promote` imported `forge.train.sft`, and so datasets, outside its error
  handling too, so it gets the same one-line cpu extra error as `forge data build-sft` and
  `forge train sft` (FG-493).
- A training config's `model_dump`, which a kernel job's options carry, writes the trainer
  `run.kind` doesn't name as null, so only that section may be null; any other section
  written with nothing under it is refused as empty (FG-493).
- The notes check reads a number as cited for the report row or section the bullet named
  last before it, so a sentence names its row before its numbers. Class names that are
  English words (where, value, other, join) count as labels too: a stray one before a
  number fails the check rather than passing it (FG-493).
- Run b's committed scores show a few group_order and projection failures that never ran
  (`valid` false), which the compare report can't show, so the notes say nearly all of
  those classes' failures run; a valid-by-class count in `forge report compare` would let
  the notes cite it (FG-493).
- `forge.train.callbacks.wandb_callback` leaves `WANDB_DIR` unset, so W&B still writes
  `wandb/` where training starts; `.gitignore` takes it at any depth (FG-493).
- The SFT pilot's adapter digest, `e84967ce...9562`, was taken after both runs on the laptop's
  `checkpoints/final`, whose files were last written before the eval run started, so the push
  to the run store can check the folder (`forge runs push --adapter-sha256`); the manifests
  stay without it. The Hugging Face account, its token and the run store repo exist, so that
  push waits only for the operator (FG-494). It ran on 2026-11-17, as a note below says.
- A kernel pushed with the CLI gets no `HF_TOKEN`, so a kernel's results come back as its
  output, and an eval job can't fetch a `runs/<run_id>/checkpoints/<name>` adapter from the
  private run store as the runbook's "SFT runs" steps say. FG-404's val runs need another
  way in for the adapter, such as the SFT kernel's output mounted as a kernel source, and
  `docs/runs/pending.yaml` lists that beside their configs' change to ADR-0012's model
  (FG-494). FG-495 built that way in and took the blocker out.
- `forge runs promote` compares the base config's prompt version with the SFT run's
  `data.prompt_version`, so a p2 run of the chosen model is promoted with
  `configs/eval/<name>-val-p2.yaml`; `tests/fakes.sft_run_folder` now trains on p1 as a real
  run's config says (FG-494).
- ADR-0007 fixed p2 for every later phase, SFT data included, so both of FG-404's SFT runs
  train on p2 and are scored with their base model's `-val-p2` config: the pilot on
  Qwen3-1.7B, the small model plan section 3.8 budgets it and FG-507's GRPO pilot on, and
  the main run on ADR-0012's choice. FG-403's configs still train on p1, the main one
  Qwen3-4B, and `docs/runs/pending.yaml` lists each change. FG-307's p2 run of the chosen
  model, the base FG-405 compares the main run with, is committed, so that comparison waits
  on FG-404's runs alone (FG-494).
- Plan section 3.8, as FG-307 reworked it for the chosen model, gives FG-404's rows 4.4 h:
  the pilot, the main run and the main run's two checkpoints on val at 0.45 h each, with no
  row for the pilot's val run. The story's own budget is 4.5 h (its block in the plan), what
  the configs' estimates in `docs/runs/pending.yaml` add up to; the pending test and the
  runbook now tell the two apart (FG-494).
- The ADR-0010 check reads the note's numbers as numbers outside names and words, so
  `float32`, `bf16`, `T4` and `Phase 4` give none; it holds them against the table and the
  values the pilot's config and manifest parse to (FG-494).
- A kernel pushed with the CLI reads no Kaggle Secret: `KAGGLE_USER_SECRETS_TOKEN` is set,
  but the secrets service answers HTTP 400 for `HF_TOKEN` as for a label that doesn't exist,
  so ADR-0005's path 2 holds. An `owner/slug` kernel source mounts that kernel's latest
  output, the whole of its `/kaggle/working`, at `/kaggle/input/notebooks/<owner>/<slug>/`
  beside Kaggle's `__output__.json`, `__results__.html` and `__script__.py`, and a dataset
  at `/kaggle/input/datasets/<owner>/<slug>/` (`docs/spikes/kaggle-gpu/mount-probe.json`,
  2026-11-16) (FG-495).
- `run_sft`'s callbacks push to the run's own folder (`LocalRunStore`), so FG-492's
  `unpushed-checkpoint-<step>` never happens there. It would sit in the trainer's temp output
  folder, which `run_sft` deletes when training ends, so a job whose callbacks push to the
  Hub while it trains, as Phase 5's GRPO may, has to keep that folder first (FG-495).
- An eval job's kernel copies its adapter's two files into its own `/kaggle/working`, so
  they are in that kernel's output: a resume of the eval mounts them beside the training
  kernel's output, and the kernel takes copies with the same files as one. `forge kaggle
  output` downloads them with the rest and keeps only the eval's `runs/<run_id>/` (FG-495).
- A resume of an eval job with a checkpoint adapter mounts two kernels, and `forge kaggle
  run` doesn't read from Kaggle's "could not be added" line which one went, so such an
  attempt is marked as one a later resume skips either way, and the warning says it may
  lack the run or its adapter (FG-495).
- A Kaggle-made run whose manifest is here without its `kernels.json`, as a pull of a store
  copy that lacks the record would leave it, is refused rather than mounted as
  `<owner>/forge-<run_id>`: the slug can't say which attempt of a resumed run holds a
  checkpoint, or which version (FG-495).
- 2026-11-17: FG-491's pending push ran from the laptop that trained the SFT pilot, with the
  digest FG-494 recorded, and exited 0, so `docs/runs/pending.yaml` no longer lists it:
  `uv run forge runs push 2026-11-16-qwen2-5-coder-0-5b-instruct-sft-db0d8d8 --adapter-sha256
  e84967ce9d31e90d9f8dba217fa4b79e56e91d64fdf2919bfbd5b6be60739562` printed `run
  2026-11-16-qwen2-5-coder-0-5b-instruct-sft-db0d8d8 (sft, finished): uploaded to
  hoomanesteki/forge-runs as runs/2026-11-16-qwen2-5-coder-0-5b-instruct-sft-db0d8d8`. The
  run store now holds the run with its adapter in `checkpoints/final`, the files the eval run
  scored, and `docs/runs/README.md` quotes the command (FG-491).
- `forge kaggle run` refused `--resume` for an SFT run ("an SFT run can't be resumed yet"),
  so the gate's pilot, a first session stopped early and a `--resume` that finishes it,
  needed SFT resume first: `run_sft(resume=True)` takes the run's latest
  `checkpoints/step-NNNNNN`, which holds the trainer's optimizer, scheduler, scaler, RNG
  state and `trainer_state.json` because the callbacks copy the trainer's whole
  `checkpoint-<step>` folder, and the kernel's restore stage already brought the run back
  (FG-496).
- The pilot's stop is a step, `ops.stop_after_step: 50`, not a short `time_guard_minutes`:
  no SFT step has been timed on a T4, so a minute count could end the first session after
  all 100 steps or leave a third session to run, where a step gives one stop and one finish
  whatever the pace. The time guard still stops either session first if the push's timeout
  comes sooner (FG-496).
- The gate's order for FG-404's runs (pilot session 1, pilot resume, main, its checkpoint
  evals, promote, the comparison) names no val run of the pilot. Plan FG-404's step 1 evals
  the pilot on val before the main run, its acceptance wants val reports and `best/` for the
  SFT runs, and `test_phase4_sft_runs_present` scores the pilot's `checkpoints/final`, so
  `docs/runs/pending.yaml` keeps `sft-pilot-val` and its promote between the pilot's resume
  and the main run, 0.3 h by its config (FG-496).
- Plan section 3.5 still shows `TrainBackend.sft` without `resume_from`, and FG-404's story
  block still says the pilot trains "the small model"; this story changed only section
  3.8's pilot row in PLAN.md, and ADR-0017's and ADR-0012's notes record the changes
  (FG-496).
- Since FG-496 `run_sft` begins and ends a session in `sessions.json` as an eval run does,
  so a kernel's SFT job stretches its own session to the kernel's span rather than adding
  one, and the main run, if its time guard stops it, resumes as the pilot does. The FG-494
  notes above on the pilot staying on Qwen3-1.7B describe the plan before the gate (FG-496).
- Plan section 3.8's GRPO pilot row and FG-507's step 2 put the GRPO pilot on the small
  model, 1.7B, "from its SFT adapter", and the runbook said it starts from the SFT pilot's
  `checkpoints/best`. Since FG-496 no SFT run trains Qwen3-1.7B, so that GRPO pilot has no
  adapter to start from: Phase 5 has to move it to the chosen model, as the gate moved the
  SFT pilot, or train a 1.7B SFT adapter first (FG-496).
