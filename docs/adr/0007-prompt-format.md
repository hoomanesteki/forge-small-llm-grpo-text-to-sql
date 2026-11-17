# ADR-0007: Non-thinking output with the SQL in a code block, and the prompt version

- Status: accepted
- Date: 2026-11-17 (proposed 2026-11-15)

## Context

Plan section 3.7 fixes the output format and leaves the prompt version to FG-307:

- `p1`: the system message "You write SQLite queries. Reply with exactly one SQL query in
  a ```` ```sql ```` code block. No explanation.", then the database's `CREATE TABLE`
  statements and the question (`forge.data.prompts`, `forge.data.schema`).
- `p2`: the same, with up to three rows of each table after it as `-- sample:` comments.
  The rows cost prompt tokens: past `MAX_PROMPT_TOKENS` (1,536) the overflow policy drops
  them first, so on a long schema p2 becomes p1.
- Qwen3 models answer in non-thinking mode (`enable_thinking=False`), with at most 256 new
  tokens, so a T4 doesn't spend its hours on reasoning traces.
- `extract_sql` takes the last ```` ```sql ```` block. Without one (rule 3) it takes the
  last line that starts with `SELECT` or `WITH` and parses, and only that line, so an
  unfenced query written over several lines loses every line after its first.

When this was proposed, the only real model replies were the FG-291 pilot's:
Qwen2.5-Coder-0.5B-Instruct on the whole val split with p1, on the laptop's GPU
(`local-mps`, ADR-0016). Counted from its stored generations and scores by
`scripts/count_extraction.py`:

| Run | Replies | sql block | Rule 3 | No SQL | Rule 3 cut a query | Cut, by class |
|---|---|---|---|---|---|---|
| 2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7 | 872 | 109 | 761 | 2 | 12 | schema_link 12 |

So 763 of the 872 replies ignored p1's code block: rule 3 read 761 of them, and 2 have no
query at all. In 12, rule 3 kept the first line of a query that went on over the next
lines, the select list without its `FROM`, and all 12 are scored `schema_link`, the class
for names the database doesn't have. Part of what a report calls a schema error can come
from the format, then, and a model or prompt that leaves the block off is marked down for
it. The pilot is a 0.5B model on another device, so this says what to count in FG-307, not
how the candidates will behave.

## Decision

- Output: non-thinking mode for Qwen3 and 256 new tokens at most, the SQL in a
  ```` ```sql ```` block, as plan section 3.7 says. This ADR changes no extraction rule.
- Prompt version: FG-307 runs the model ADR-0012's rule chooses from the p1 val runs on
  val with p2 too, from the same commit. p2 is fixed for every later phase only if its
  official EX is higher and `forge report compare` finds the paired difference significant
  (McNemar, p below 0.05). Otherwise p1 stays: it is the shorter prompt, and it is what the
  overflow policy falls back to anyway.
- That model is the eligible candidate with the best val official EX, with the rule's
  fewer-parameters tie-break, not the best p1 run alone. Plan section 3.7 compares p1 and
  p2 "for the chosen model", while the FG-307 story block says "the top candidate"; the two
  differ when the top candidate scores 75% or more on val, fails FG-301's memory check, or
  loses the tie-break to a smaller model. The prompt version is fixed for the model every
  later phase trains, so section 3.7's reading holds, and the p2 run waits for the choice.
- Each run in the comparison gets its row from `scripts/count_extraction.py` beside its EX,
  so a version that loses the code block more often is seen as a format problem, not read
  as a schema one.

## Measured

FG-307's runs of Qwen/Qwen2.5-Coder-1.5B-Instruct, the model ADR-0012's rule chose from the
p1 val runs (and the best of them, so section 3.7's reading and the story block's agree
here), on the whole val split, greedy, from `c8f4b41` on a Kaggle T4, 2026-11-16. From the
reports committed under `docs/runs/`:

| Prompt | Run | Official EX [95% interval] | Fast EX | Valid SQL |
|---|---|---|---|---|
| p1 | [2026-11-16-qwen2-5-coder-1-5b-val-c8f4b41](../runs/2026-11-16-qwen2-5-coder-1-5b-val-c8f4b41/eval/val-report.md) | 65.3% [61.2%, 69.1%] | 65.1% [61.0%, 69.0%] | 85.2% [82.2%, 88.1%] |
| p2 | [2026-11-16-qwen2-5-coder-1-5b-p2-c8f4b41](../runs/2026-11-16-qwen2-5-coder-1-5b-p2-c8f4b41/eval/val-report.md) | 67.4% [63.3%, 71.3%] | 67.5% [63.4%, 71.2%] | 86.6% [83.8%, 89.4%] |

`forge report compare 2026-11-16-qwen2-5-coder-1-5b-val-c8f4b41
2026-11-16-qwen2-5-coder-1-5b-p2-c8f4b41`, whose report is in the p2 run's folder
([compare-2026-11-16-qwen2-5-coder-1-5b-val-c8f4b41.md](../runs/2026-11-16-qwen2-5-coder-1-5b-p2-c8f4b41/eval/compare-2026-11-16-qwen2-5-coder-1-5b-val-c8f4b41.md)),
says, with run a the p1 run and run b the p2 run:

> Official EX on val, run b against run a: +2.2 points [+0.6, +3.8], McNemar p = 0.0115 on clusters
>
> The runs disagree on 51 of 872 questions: run b alone is right on 35, run a alone on 16.
>
> The test pairs whole clusters, as the intervals draw them: run b gets more right in 29 clusters, run a in 12, and the rest tie.

The extraction rows, from `uv run python scripts/count_extraction.py` on both runs:

| Run | Replies | sql block | Rule 3 | No SQL | Rule 3 cut a query | Cut, by class |
|---|---|---|---|---|---|---|
| 2026-11-16-qwen2-5-coder-1-5b-val-c8f4b41 | 872 | 764 | 106 | 2 | 0 | - |
| 2026-11-16-qwen2-5-coder-1-5b-p2-c8f4b41 | 872 | 724 | 148 | 0 | 0 | - |

With p2 the model left the code block off more often, but rule 3 cut no query in either
run, so the change in EX is not the format's.

The rule above, applied: p2's official EX is higher and McNemar's p is below 0.05, so **p2
is fixed for every later phase**. The chosen model's dev baseline and the reference's runs
use it too (`configs/eval/qwen2-5-coder-1-5b-dev.yaml`, `qwen3-8b-val.yaml` and
`qwen3-8b-dev.yaml`): the dev baseline has to compare with the trained model's dev
evaluation, which uses p2, and the reference shows what a larger model scores with the same
prompt and harness (ADR-0012).

## Consequences

- SFT data (FG-401), GRPO prompts (plan section 3.10's `prompt_version`) and every
  evaluation from here on use p2. Where a schema's sample rows would pass
  `MAX_PROMPT_TOKENS`, the overflow policy drops them, so for those prompts p2 is p1.
- SFT data (FG-401) puts every gold query in a ```` ```sql ```` block, so a model that
  ignores the block now is taught it before GRPO; the format reward (plan section 3.10)
  keeps it.
- If the chosen model still writes unfenced multi-line queries after SFT, a later story can
  let rule 3 read a whole query, as `scripts/count_extraction.py` already does to count
  them, and re-score the runs it affects.
