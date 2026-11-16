# Compute ledger

Append-only. Rows are written by `forge ledger add`, by `forge ledger record` for a run made
on this machine, and by `forge kaggle wait`, `forge kaggle output` and `forge runs pull`, one
a GPU session. Dates are UTC.

| date | platform | gpu | hours | run_id | purpose | cost_usd |
|---|---|---|---|---|---|---|
| 2026-11-15 | local-mps | Apple M4 | 0.17 | 2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7 | FG-291 pilot: Qwen2.5-Coder-0.5B-Instruct on val, greedy, p1 (ADR-0016) | 0.00 |
| 2026-11-16 | local-mps | Apple M4 | 1.78 | 2026-11-16-qwen2-5-coder-0-5b-instruct-sft-db0d8d8 | sft Qwen/Qwen2.5-Coder-0.5B-Instruct@ea3f2471cf1b1f0db85067f1ef93848e38e88c25, finished | 0.00 |
| 2026-11-16 | local-mps | Apple M4 | 0.26 | 2026-11-16-qwen2-5-coder-0-5b-instruct-eval-db0d8d8 | eval Qwen/Qwen2.5-Coder-0.5B-Instruct@ea3f2471cf1b1f0db85067f1ef93848e38e88c25 with runs/2026-11-16-qwen2-5-coder-0-5b-instruct-sft-db0d8d8/checkpoints/final on val, prompt p1, finished | 0.00 |
