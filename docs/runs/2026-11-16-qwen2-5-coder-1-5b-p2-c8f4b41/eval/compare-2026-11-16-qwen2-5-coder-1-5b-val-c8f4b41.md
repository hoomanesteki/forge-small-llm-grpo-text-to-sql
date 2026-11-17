# Official EX on val, run b against run a: +2.2 points [+0.6, +3.8], McNemar p = 0.0115 on clusters

Generated: 2026-11-17 00:08:46 UTC

Written by `forge report compare 2026-11-16-qwen2-5-coder-1-5b-val-c8f4b41 2026-11-16-qwen2-5-coder-1-5b-p2-c8f4b41` from both runs' stored files and checked against them by `forge report rebuild 2026-11-16-qwen2-5-coder-1-5b-p2-c8f4b41`; don't edit it by hand.

| | run a | run b |
|---|---|---|
| Run | `2026-11-16-qwen2-5-coder-1-5b-val-c8f4b41` | `2026-11-16-qwen2-5-coder-1-5b-p2-c8f4b41` |
| Model | `Qwen/Qwen2.5-Coder-1.5B-Instruct@2e1fd397ee46e1388853d2af2c993145b0f1098a` | `Qwen/Qwen2.5-Coder-1.5B-Instruct@2e1fd397ee46e1388853d2af2c993145b0f1098a` |
| Adapter | none | none |
| Backend | `vllm` | `vllm` |
| Platform | `kaggle` | `kaggle` |
| Prompt version | `p1` | `p2` |
| Generation | greedy, at most 256 new tokens | greedy, at most 256 new tokens |
| Retry | none | none |
| Git commit | `c8f4b419f1f2666984a48c49eeb7827fc0b7d1e4` | `c8f4b419f1f2666984a48c49eeb7827fc0b7d1e4` |
| Manifest | [manifest.json](../../2026-11-16-qwen2-5-coder-1-5b-val-c8f4b41/manifest.json) | [manifest.json](../manifest.json) |

Both runs answered the same 872 questions of val, 1 sample each, on data with manifest hash `e10958d3b7a9490c15623285275d2840fbce551c32432fc62c0ad440cb06651e`.

## Execution accuracy

| | run a [95% interval] | run b [95% interval] | change in points [95% interval] |
|---|---:|---:|---:|
| valid SQL | 85.2% [82.2%, 88.1%] | 86.6% [83.8%, 89.4%] | |
| fast EX | 65.1% [61.0%, 69.0%] | 67.5% [63.4%, 71.2%] | |
| official EX | 65.3% [61.2%, 69.1%] | 67.4% [63.3%, 71.3%] | +2.2 [+0.6, +3.8] |

Only official EX, the reported rate, gets a change: each sample of run b against
the same one of run a. Intervals are 95% percentile bootstraps over 1,000 resamples (seed 0) that draw whole questions, with the questions that share a gold query drawn together: 872 questions in 455 clusters.

## McNemar's test

| official EX | run b right | run b wrong |
|---|---:|---:|
| run a right | 553 | 16 |
| run a wrong | 35 | 268 |

The runs disagree on 51 of 872 questions: run b alone is right on 35, run a alone on 16.
The test pairs whole clusters, as the intervals draw them: run b gets more right in 29 clusters, run a in 12, and the rest tie.
Exact two-sided McNemar test on those clusters: p = 0.0115.

## By hardness

| hardness | questions | run a | run b | change in points [95% interval] |
|---|---:|---:|---:|---:|
| easy | 204 | 83.8% | 88.7% | +4.9 [+1.4, +8.7] |
| medium | 361 | 69.3% | 72.0% | +2.8 [+0.8, +4.7] |
| hard | 150 | 59.3% | 58.7% | -0.7 [-5.5, +3.8] |
| extra | 157 | 37.6% | 37.6% | +0.0 [-3.2, +3.8] |

## By error class

Failures of official EX in each run, by class.

| class | run a | run b | change |
|---|---:|---:|---:|
| no_sql | 2 | 0 | -2 |
| syntax | 1 | 0 | -1 |
| schema_link | 119 | 112 | -7 |
| wrong_tables | 49 | 44 | -5 |
| join | 48 | 52 | +4 |
| projection | 26 | 27 | +1 |
| aggregation | 4 | 2 | -2 |
| where | 14 | 15 | +1 |
| group_order | 16 | 13 | -3 |
| value | 20 | 14 | -6 |
| other | 0 | 0 | +0 |
| unclassified | 4 | 5 | +1 |

Unclassified failures are those only official EX finds: fast EX matched them, so the run gave them no class.
