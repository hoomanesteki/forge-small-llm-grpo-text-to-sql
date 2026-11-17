# Official EX on val, run b against run a: +7.3 points [+3.7, +10.8], McNemar p < 0.0001 on clusters

Generated: 2026-11-17 11:27:09 UTC

Written by `forge report compare 2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7 2026-11-17-qwen2-5-coder-0-5b-instruct-eval-e924b73` from both runs' stored files and checked against them by `forge report rebuild 2026-11-17-qwen2-5-coder-0-5b-instruct-eval-e924b73`; don't edit it by hand.

| | run a | run b |
|---|---|---|
| Run | `2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7` | `2026-11-17-qwen2-5-coder-0-5b-instruct-eval-e924b73` |
| Model | `Qwen/Qwen2.5-Coder-0.5B-Instruct@ea3f2471cf1b1f0db85067f1ef93848e38e88c25` | `Qwen/Qwen2.5-Coder-0.5B-Instruct@ea3f2471cf1b1f0db85067f1ef93848e38e88c25` |
| Adapter | none | `runs/2026-11-17-qwen2-5-coder-0-5b-instruct-grpo-e924b73/checkpoints/step-000250` |
| Backend | `hf` | `hf` |
| Platform | `local-mps` | `local-mps` |
| Prompt version | `p1` | `p1` |
| Generation | greedy, at most 256 new tokens | greedy, at most 256 new tokens |
| Retry | none | none |
| Git commit | `da3eff7afa2c101f55ba23a541a73001704cf0b9` | `e924b73e2f96908016b56db6134ed7fef4b87cb9` |
| Manifest | [manifest.json](../../2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7/manifest.json) | [manifest.json](../manifest.json) |

Both runs answered the same 872 questions of val, 1 sample each, on data with manifest hash `e10958d3b7a9490c15623285275d2840fbce551c32432fc62c0ad440cb06651e`.

## Execution accuracy

| | run a [95% interval] | run b [95% interval] | change in points [95% interval] |
|---|---:|---:|---:|
| valid SQL | 67.7% [63.8%, 71.4%] | 78.1% [74.3%, 81.5%] | |
| fast EX | 47.4% [43.1%, 51.3%] | 54.4% [49.8%, 58.5%] | |
| official EX | 47.2% [43.1%, 51.1%] | 54.6% [50.0%, 58.9%] | +7.3 [+3.7, +10.8] |

Only official EX, the reported rate, gets a change: each sample of run b against
the same one of run a. Intervals are 95% percentile bootstraps over 1,000 resamples (seed 0) that draw whole questions, with the questions that share a gold query drawn together: 872 questions in 455 clusters.

## McNemar's test

| official EX | run b right | run b wrong |
|---|---:|---:|
| run a right | 345 | 67 |
| run a wrong | 131 | 329 |

The runs disagree on 198 of 872 questions: run b alone is right on 131, run a alone on 67.
The test pairs whole clusters, as the intervals draw them: run b gets more right in 99 clusters, run a in 50, and the rest tie.
Exact two-sided McNemar test on those clusters: p < 0.0001.

## By hardness

| hardness | questions | run a | run b | change in points [95% interval] |
|---|---:|---:|---:|---:|
| easy | 204 | 77.9% | 87.3% | +9.3 [+4.9, +14.4] |
| medium | 361 | 44.9% | 53.5% | +8.6 [+2.3, +15.4] |
| hard | 150 | 37.3% | 48.0% | +10.7 [+0.7, +19.6] |
| extra | 157 | 22.3% | 21.0% | -1.3 [-8.3, +5.8] |

## By error class

Failures of official EX in each run, by class.

| class | run a | run b | change |
|---|---:|---:|---:|
| no_sql | 2 | 0 | -2 |
| syntax | 0 | 0 | +0 |
| schema_link | 266 | 179 | -87 |
| wrong_tables | 80 | 94 | +14 |
| join | 21 | 21 | +0 |
| projection | 51 | 49 | -2 |
| aggregation | 2 | 0 | -2 |
| where | 13 | 19 | +6 |
| group_order | 12 | 14 | +2 |
| value | 10 | 16 | +6 |
| other | 0 | 2 | +2 |
| unclassified | 3 | 2 | -1 |

Unclassified failures are those only official EX finds: fast EX matched them, so the run gave them no class.
