# Compute ledger

Append-only. Rows are written by `forge ledger add`, and by `forge kaggle wait`, `forge kaggle
output` and `forge runs pull`, one a GPU session. Dates are UTC.

| date | platform | gpu | hours | run_id | purpose | cost_usd |
|---|---|---|---|---|---|---|
| 2026-11-15 | local-mps | Apple M4 | 0.17 | 2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7 | FG-291 pilot: Qwen2.5-Coder-0.5B-Instruct on val, greedy, p1 (ADR-0016) | 0.00 |
| 2026-11-16 | kaggle | T4 | 0.59 | 2026-11-16-throughput-spike-c8f4b41 | FG-301 throughput spike | 0.00 |
