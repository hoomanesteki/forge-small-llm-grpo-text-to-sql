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
| FG-101 | Spike: data sources, licenses and official evaluator | 2 | todo |  |  |  |
| FG-102 | Core types and hashing | 2 | todo |  |  |  |
| FG-103 | Spider loader, checksums and fixture subset | 3 | todo |  |  |  |
| FG-104 | Splits by database and dev guard | 3 | todo |  |  |  |
| FG-105 | Schema serializer and prompt builder | 5 | todo |  |  |  |
| FG-106 | Safe SQL executor | 5 | todo |  |  |  |
| FG-107 | SQL extraction and result comparator | 3 | todo |  |  |  |
| FG-108 | Official evaluator integration and agreement check | 5 | todo |  |  |  |

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
- CI runs on every pushed branch as well as on PRs, so a branch is checked before its PR opens.
