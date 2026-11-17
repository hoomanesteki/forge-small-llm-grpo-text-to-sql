# GRPO run log

What each GRPO run did and what it found, from its committed files. FG-507 adds the Kaggle
runs; the local pilot came first. A number in a bullet is in the comparison its paragraph
links, on the row or under the heading named before it. A number from a run's own files is
quoted from them in a code span, as its JSON lines hold it, and a setting as its config
file writes it.

## Local pilot on Apple silicon (FG-591)

**What ran.** The pilot model, `Qwen/Qwen2.5-Coder-0.5B-Instruct`, trained with GRPO on the
laptop's GPU (`local-mps`, ADR-0016) from FG-491's SFT adapter: run
[`2026-11-17-qwen2-5-coder-0-5b-instruct-grpo-e924b73`](../runs/2026-11-17-qwen2-5-coder-0-5b-instruct-grpo-e924b73/),
from `configs/grpo/pilot-qwen25coder05b-mps.yaml`. It trained on the rows FG-505's filter
kept from that adapter's greedy replies to the first train questions (`--limit 3000`, run
[`2026-11-17-qwen2-5-coder-0-5b-instruct-eval-a27afde`](../runs/2026-11-17-qwen2-5-coder-0-5b-instruct-eval-a27afde/eval/train-report.md),
rows in [`grpo/`](../runs/grpo/2026-11-17-qwen2-5-coder-0-5b-instruct-eval-a27afde/summary.json)):
every question it got wrong and a share of those it got right. Each step took two prompts
with `num_generations: 4` replies each, trained on them one at a time
(`per_device_train_batch_size: 1`) for `max_steps: 350`, and every `eval_every: 50` steps
saved a checkpoint and scored fast EX on the same val questions. Training took
`"train_runtime": 8558.9254` seconds. A probe and the run's first start ran out of the GPU
memory the MPS watermark allows, and the config says how the batch came to one reply at a
time.

**What training showed.** A step here is eight replies to two prompts, so each reward's mean
swings across most of its range from one step to the next, and the
[charts](../runs/2026-11-17-qwen2-5-coder-0-5b-instruct-grpo-e924b73/plots/) show no trend in
the execution or validity reward through that noise. The format reward stayed near its top,
and the guard penalty fired on two steps (`"rewards/guard_penalty/mean": -0.0625`). Fast EX
on the fixed val questions stayed in a band a few questions wide:
`"step": 50, "val/ex_fast": 0.495` at the first checkpoint,
`"step": 200, "val/ex_fast": 0.51` and `"step": 250, "val/ex_fast": 0.51` at the best, and
`"step": 350, "val/ex_fast": 0.5` at the end. The best, the latest of the tie, is
`checkpoints/step-000250`, which the val run below scores on all of val.

**What the alerts said.** `"rule": "valid_drop"` and `"rule": "length_growth"` fired and
cleared all through the run, and neither meant much at this batch. One step's valid-SQL rate
moves by an eighth with each reply, so a step two replies short of the best step so far is
past the rule's drop. Length growth compares each step with the first one's mean,
`"completions/mean_length": 28.375`, a short step, and the completion-length chart holds
the same range from start to end. The zero-variance share of a step is none, half or all of
its two groups, and never stayed over the alert's line for long enough to fire it.

### Against the SFT pilot

The GRPO checkpoint's val run against FG-491's, both with FG-291's settings, so they differ in
the adapter alone; run a is the SFT adapter and run b `checkpoints/step-000250`:
[`compare-2026-11-16-qwen2-5-coder-0-5b-instruct-eval-db0d8d8.md`](../runs/2026-11-17-qwen2-5-coder-0-5b-instruct-eval-e924b73/eval/compare-2026-11-16-qwen2-5-coder-0-5b-instruct-eval-db0d8d8.md).

- **What moved overall.** Official EX went from 52.3% [47.9%, 56.4%] to 54.6% [50.0%, 58.9%],
  a change of +2.3 points [+0.0, +4.5], whose interval reaches down to zero; McNemar's test on
  clusters gives p = 0.0385. The runs disagree on 78 of 872 questions: GRPO alone is right on
  49, the SFT adapter alone on 29. Valid SQL rose from 75.8% to 78.1%.
- **By hardness.** Easy gained +4.4 [+1.4, +7.9], the only interval that clears zero. Medium,
  +2.5 [-1.4, +6.6], and hard, +4.7 [-0.7, +10.1], hold zero inside. Extra fell from 24.2% to
  21.0%, -3.2 [-7.8, +1.3].
- **By error class.** schema_link failures fell from 203 to 179 (-24) and projection from
  58 to 49 (-9), but wrong_tables grew from 71 to 94 (+23). The report counts failures by
  class, so it can't say whether the queries that left schema_link are the ones that joined
  wrong_tables. syntax went from 2 to 0 (-2) and group_order from 19 to 14 (-5).
- **How far to trust it.** `checkpoints/step-000250` was picked by its fast EX on a fixed
  subset of val during training, questions this comparison scores too, from checkpoints a few
  questions apart, so some of the gain may be that choice; the interval's lower end is at
  zero. It is a pilot on val alone: a 0.5B model, one seed, two prompts a step. FG-507's
  Kaggle runs, with larger groups and batches, are the result.

### Against the base model

The same GRPO val run against FG-291's base pilot, so SFT and GRPO together against neither;
run a is the base model:
[`compare-2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7.md`](../runs/2026-11-17-qwen2-5-coder-0-5b-instruct-eval-e924b73/eval/compare-2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7.md).

- **What moved overall.** Official EX went from 47.2% [43.1%, 51.1%] to 54.6% [50.0%, 58.9%],
  +7.3 points [+3.7, +10.8], and McNemar's test on clusters gives p < 0.0001. Valid SQL rose
  from 67.7% to 78.1%.
- **By hardness.** Easy, +9.3 [+4.9, +14.4], medium, +8.6 [+2.3, +15.4], and hard, +10.7
  [+0.7, +19.6], each clear zero. Extra went from 22.3% to 21.0%, -1.3 [-8.3, +5.8]: SFT and
  GRPO together left the hardest questions no better than the base did.
- **By error class.** schema_link failures fell from 266 to 179 (-87), the largest change,
  while wrong_tables grew from 80 to 94 (+14), where from 13 to 19 (+6) and value from 10 to
  16 (+6).
- **For FG-507.** The pilot's gains sit in the easier hardness levels and in schema linking;
  the hardest questions and the classes past schema linking did not improve. Those are what
  FG-507's runs, with longer replies, larger groups and more prompts a step, have most to
  show, and a valid-SQL alert read over a window of steps rather than one step would make
  its alerts worth reading at the pilot's batch.
