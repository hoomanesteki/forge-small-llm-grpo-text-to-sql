# ADR-0007: Non-thinking output with the SQL in a code block, and the prompt version

- Status: proposed, the p1 and p2 comparison pending FG-307's runs (`docs/runs/pending.yaml`)
- Date: 2026-11-15

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

The only real model replies so far are the FG-291 pilot's: Qwen2.5-Coder-0.5B-Instruct on
the whole val split with p1, on the laptop's GPU (`local-mps`, ADR-0016). Counted from its
stored generations and scores by `scripts/count_extraction.py`:

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

## Decision (proposed)

- Output: non-thinking mode for Qwen3 and 256 new tokens at most, the SQL in a
  ```` ```sql ```` block, as plan section 3.7 says. This ADR changes no extraction rule.
- Prompt version: FG-307 runs the top candidate of the p1 val runs (ADR-0012) on val with
  p2 too, from the same commit. p2 is fixed for every later phase only if its official EX
  is higher and `forge report compare` finds the paired difference significant (McNemar,
  p below 0.05). Otherwise p1 stays: it is the shorter prompt, and it is what the overflow
  policy falls back to anyway.
- Each run in the comparison gets its row from `scripts/count_extraction.py` beside its EX,
  so a version that loses the code block more often is seen as a format problem, not read
  as a schema one.

## Measured

Pending FG-307's runs. This section will hold the two val runs' official EX with their
intervals and the compare report's test, from the reports committed under `docs/runs/`,
and the extraction rows of both runs. Nothing here is typed by hand.

## Consequences

- SFT data (FG-401) puts every gold query in a ```` ```sql ```` block, so a model that
  ignores the block now is taught it before GRPO; the format reward (plan section 3.10)
  keeps it.
- If the chosen model still writes unfenced multi-line queries after SFT, a later story can
  let rule 3 read a whole query, as `scripts/count_extraction.py` already does to count
  them, and re-score the runs it affects.
