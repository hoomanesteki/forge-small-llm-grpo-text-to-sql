# Compute ledger

Append-only. Rows are written by `forge ledger add`, and by `forge kaggle wait`, `forge kaggle
output` and `forge runs pull`, one a GPU session. Dates are UTC.

| date | platform | gpu | hours | run_id | purpose | cost_usd |
|---|---|---|---|---|---|---|
| 2026-11-15 | local-mps | Apple M4 | 0.17 | 2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7 | FG-291 pilot: Qwen2.5-Coder-0.5B-Instruct on val, greedy, p1 (ADR-0016) | 0.00 |
| 2026-11-16 | kaggle | T4 | 0.59 | 2026-11-16-throughput-spike-c8f4b41 | FG-301 throughput spike | 0.00 |
| 2026-11-16 | kaggle | Tesla T4 | 0.18 | 2026-11-16-smoke-kaggle-50-val-c8f4b41 | eval Qwen/Qwen2.5-Coder-1.5B-Instruct@2e1fd397ee46e1388853d2af2c993145b0f1098a on val, prompt p1, limit 50, finished | 0.00 |
| 2026-11-16 | kaggle | Tesla T4 | 0.24 | 2026-11-16-qwen3-1-7b-val-c8f4b41 | eval Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e on val, prompt p1, finished | 0.00 |
| 2026-11-16 | kaggle | Tesla T4 | 0.21 | 2026-11-16-qwen3-4b-val-c8f4b41 | eval Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c on val, prompt p1, finished | 0.00 |
| 2026-11-16 | kaggle | Tesla T4 | 0.21 | 2026-11-16-qwen2-5-coder-1-5b-val-c8f4b41 | eval Qwen/Qwen2.5-Coder-1.5B-Instruct@2e1fd397ee46e1388853d2af2c993145b0f1098a on val, prompt p1, finished | 0.00 |
| 2026-11-16 | kaggle | Tesla T4 | 0.17 | 2026-11-16-qwen2-5-coder-1-5b-i6-c8f4b41 | eval Qwen/Qwen2.5-Coder-1.5B-Instruct@2e1fd397ee46e1388853d2af2c993145b0f1098a on val, prompt p1, limit 100, finished | 0.00 |
| 2026-11-16 | kaggle | Tesla T4 | 0.15 | 2026-11-16-qwen2-5-coder-1-5b-p2-c8f4b41 | eval Qwen/Qwen2.5-Coder-1.5B-Instruct@2e1fd397ee46e1388853d2af2c993145b0f1098a on val, prompt p2, finished | 0.00 |
| 2026-11-17 | kaggle | Tesla T4 | 0.21 | 2026-11-17-qwen3-8b-awq-val-2f77281 | eval Qwen/Qwen3-8B-AWQ@4da05a8edb55c6046cce958586c33b61da07bb79 on val, prompt p2, finished | 0.00 |
| 2026-11-17 | kaggle | Tesla T4 | 0.20 | 2026-11-17-qwen2-5-coder-1-5b-dev-2f77281 | eval Qwen/Qwen2.5-Coder-1.5B-Instruct@2e1fd397ee46e1388853d2af2c993145b0f1098a on dev, prompt p2, finished | 0.00 |
| 2026-11-17 | kaggle | Tesla T4 | 0.25 | 2026-11-17-qwen3-8b-awq-dev-d2d6a3a | eval Qwen/Qwen3-8B-AWQ@4da05a8edb55c6046cce958586c33b61da07bb79 on dev, prompt p2, finished | 0.00 |
