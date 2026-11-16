# Official EX on val, run b against run a: +5.0 points [+1.6, +8.6], McNemar p = 0.0086 on clusters

Generated: 2026-11-16 05:57:30 UTC

Written by `forge report compare 2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7 2026-11-16-qwen2-5-coder-0-5b-instruct-eval-db0d8d8` from both runs' stored files and checked against them by `forge report rebuild 2026-11-16-qwen2-5-coder-0-5b-instruct-eval-db0d8d8`; don't edit it by hand.

| | run a | run b |
|---|---|---|
| Run | `2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7` | `2026-11-16-qwen2-5-coder-0-5b-instruct-eval-db0d8d8` |
| Model | `Qwen/Qwen2.5-Coder-0.5B-Instruct@ea3f2471cf1b1f0db85067f1ef93848e38e88c25` | `Qwen/Qwen2.5-Coder-0.5B-Instruct@ea3f2471cf1b1f0db85067f1ef93848e38e88c25` |
| Adapter | none | `runs/2026-11-16-qwen2-5-coder-0-5b-instruct-sft-db0d8d8/checkpoints/final` |
| Backend | `hf` | `hf` |
| Platform | `local-mps` | `local-mps` |
| Prompt version | `p1` | `p1` |
| Generation | greedy, at most 256 new tokens | greedy, at most 256 new tokens |
| Retry | none | none |
| Git commit | `da3eff7afa2c101f55ba23a541a73001704cf0b9` | `db0d8d855e16a92fece33f6416b11d3b2b1fb344` |
| Manifest | [manifest.json](../../2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7/manifest.json) | [manifest.json](../manifest.json) |

Both runs answered the same 872 questions of val, 1 sample each, on data with manifest hash `e10958d3b7a9490c15623285275d2840fbce551c32432fc62c0ad440cb06651e`.

## Execution accuracy

| | run a [95% interval] | run b [95% interval] | change in points [95% interval] |
|---|---:|---:|---:|
| valid SQL | 67.7% [63.8%, 71.4%] | 75.8% [72.2%, 79.6%] | |
| fast EX | 47.4% [43.1%, 51.3%] | 51.8% [47.6%, 55.9%] | |
| official EX | 47.2% [43.1%, 51.1%] | 52.3% [47.9%, 56.4%] | +5.0 [+1.6, +8.6] |

Only official EX, the reported rate, gets a change: each sample of run b against
the same one of run a. Intervals are 95% percentile bootstraps over 1,000 resamples (seed 0) that draw whole questions, with the questions that share a gold query drawn together: 872 questions in 455 clusters.

## McNemar's test

| official EX | run b right | run b wrong |
|---|---:|---:|
| run a right | 341 | 71 |
| run a wrong | 115 | 345 |

The runs disagree on 186 of 872 questions: run b alone is right on 115, run a alone on 71.
The test pairs whole clusters, as the intervals draw them: run b gets more right in 86 clusters, run a in 54, and the rest tie.
Exact two-sided McNemar test on those clusters: p = 0.0086.

## By hardness

| hardness | questions | run a | run b | change in points [95% interval] |
|---|---:|---:|---:|---:|
| easy | 204 | 77.9% | 82.8% | +4.9 [+0.0, +9.9] |
| medium | 361 | 44.9% | 51.0% | +6.1 [+0.3, +12.4] |
| hard | 150 | 37.3% | 43.3% | +6.0 [-4.0, +14.9] |
| extra | 157 | 22.3% | 24.2% | +1.9 [-5.8, +9.6] |

## By error class

Failures of official EX in each run, by class.

| class | run a | run b | change |
|---|---:|---:|---:|
| no_sql | 2 | 0 | -2 |
| syntax | 0 | 2 | +2 |
| schema_link | 266 | 203 | -63 |
| wrong_tables | 80 | 71 | -9 |
| join | 21 | 20 | -1 |
| projection | 51 | 58 | +7 |
| aggregation | 2 | 0 | -2 |
| where | 13 | 22 | +9 |
| group_order | 12 | 19 | +7 |
| value | 10 | 17 | +7 |
| other | 0 | 3 | +3 |
| unclassified | 3 | 1 | -2 |

Unclassified failures are those only official EX finds: fast EX matched them, so the run gave them no class.
