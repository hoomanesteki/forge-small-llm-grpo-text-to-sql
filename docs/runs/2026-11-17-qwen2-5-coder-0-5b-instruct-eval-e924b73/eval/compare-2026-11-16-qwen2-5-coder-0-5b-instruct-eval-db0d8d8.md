# Official EX on val, run b against run a: +2.3 points [+0.0, +4.5], McNemar p = 0.0385 on clusters

Generated: 2026-11-17 11:27:07 UTC

Written by `forge report compare 2026-11-16-qwen2-5-coder-0-5b-instruct-eval-db0d8d8 2026-11-17-qwen2-5-coder-0-5b-instruct-eval-e924b73` from both runs' stored files and checked against them by `forge report rebuild 2026-11-17-qwen2-5-coder-0-5b-instruct-eval-e924b73`; don't edit it by hand.

| | run a | run b |
|---|---|---|
| Run | `2026-11-16-qwen2-5-coder-0-5b-instruct-eval-db0d8d8` | `2026-11-17-qwen2-5-coder-0-5b-instruct-eval-e924b73` |
| Model | `Qwen/Qwen2.5-Coder-0.5B-Instruct@ea3f2471cf1b1f0db85067f1ef93848e38e88c25` | `Qwen/Qwen2.5-Coder-0.5B-Instruct@ea3f2471cf1b1f0db85067f1ef93848e38e88c25` |
| Adapter | `runs/2026-11-16-qwen2-5-coder-0-5b-instruct-sft-db0d8d8/checkpoints/final` | `runs/2026-11-17-qwen2-5-coder-0-5b-instruct-grpo-e924b73/checkpoints/step-000250` |
| Backend | `hf` | `hf` |
| Platform | `local-mps` | `local-mps` |
| Prompt version | `p1` | `p1` |
| Generation | greedy, at most 256 new tokens | greedy, at most 256 new tokens |
| Retry | none | none |
| Git commit | `db0d8d855e16a92fece33f6416b11d3b2b1fb344` | `e924b73e2f96908016b56db6134ed7fef4b87cb9` |
| Manifest | [manifest.json](../../2026-11-16-qwen2-5-coder-0-5b-instruct-eval-db0d8d8/manifest.json) | [manifest.json](../manifest.json) |

Both runs answered the same 872 questions of val, 1 sample each, on data with manifest hash `e10958d3b7a9490c15623285275d2840fbce551c32432fc62c0ad440cb06651e`.

## Execution accuracy

| | run a [95% interval] | run b [95% interval] | change in points [95% interval] |
|---|---:|---:|---:|
| valid SQL | 75.8% [72.2%, 79.6%] | 78.1% [74.3%, 81.5%] | |
| fast EX | 51.8% [47.6%, 55.9%] | 54.4% [49.8%, 58.5%] | |
| official EX | 52.3% [47.9%, 56.4%] | 54.6% [50.0%, 58.9%] | +2.3 [+0.0, +4.5] |

Only official EX, the reported rate, gets a change: each sample of run b against
the same one of run a. Intervals are 95% percentile bootstraps over 1,000 resamples (seed 0) that draw whole questions, with the questions that share a gold query drawn together: 872 questions in 455 clusters.

## McNemar's test

| official EX | run b right | run b wrong |
|---|---:|---:|
| run a right | 427 | 29 |
| run a wrong | 49 | 367 |

The runs disagree on 78 of 872 questions: run b alone is right on 49, run a alone on 29.
The test pairs whole clusters, as the intervals draw them: run b gets more right in 43 clusters, run a in 25, and the rest tie.
Exact two-sided McNemar test on those clusters: p = 0.0385.

## By hardness

| hardness | questions | run a | run b | change in points [95% interval] |
|---|---:|---:|---:|---:|
| easy | 204 | 82.8% | 87.3% | +4.4 [+1.4, +7.9] |
| medium | 361 | 51.0% | 53.5% | +2.5 [-1.4, +6.6] |
| hard | 150 | 43.3% | 48.0% | +4.7 [-0.7, +10.1] |
| extra | 157 | 24.2% | 21.0% | -3.2 [-7.8, +1.3] |

## By error class

Failures of official EX in each run, by class.

| class | run a | run b | change |
|---|---:|---:|---:|
| no_sql | 0 | 0 | +0 |
| syntax | 2 | 0 | -2 |
| schema_link | 203 | 179 | -24 |
| wrong_tables | 71 | 94 | +23 |
| join | 20 | 21 | +1 |
| projection | 58 | 49 | -9 |
| aggregation | 0 | 0 | +0 |
| where | 22 | 19 | -3 |
| group_order | 19 | 14 | -5 |
| value | 17 | 16 | -1 |
| other | 3 | 2 | -1 |
| unclassified | 1 | 2 | +1 |

Unclassified failures are those only official EX finds: fast EX matched them, so the run gave them no class.
