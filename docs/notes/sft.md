# SFT on val: the 0.5B local pilot

What supervised fine-tuning changed for the pilot model, `Qwen/Qwen2.5-Coder-0.5B-Instruct`,
on the val split: FG-491's LoRA SFT adapter against the same model without it. Both runs
were scored on the laptop's GPU (`local-mps`, ADR-0016) with FG-291's settings, so their
settings differ in the adapter alone. This is a pilot on val, not the project's result: a
0.5B model trained on part of the train split, never to be set beside a T4 number or a dev
number. Every number below is in the comparison that `forge report compare` wrote from
both runs' stored files, on the row or under the heading named before it, where run a is the
base model and run b the adapter:
[`compare-2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7.md`](../runs/2026-11-16-qwen2-5-coder-0-5b-instruct-eval-db0d8d8/eval/compare-2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7.md).

- **What moved overall.** Official EX went from 47.2% [43.1%, 51.1%] to 52.3% [47.9%, 56.4%],
  a change of +5.0 points [+1.6, +8.6], and McNemar's test on clusters gives p = 0.0086. The
  runs disagree on 186 of 872 questions: the adapter alone is right on 115, the base alone on
  71. Valid SQL rose from 67.7% to 75.8%.
- **By hardness.** Only medium's interval clears zero, at +6.1 [+0.3, +12.4]; easy's, +4.9
  [+0.0, +9.9], just reaches it. Hard, +6.0 [-4.0, +14.9], and extra, +1.9 [-5.8, +9.6],
  hold zero well inside: hard has 150 questions and extra 157, and at those sizes the pilot
  can't tell a gain there from none. Extra is still at 24.2%.
- **By error class, what moved.** The gain is mostly schema linking: schema_link failures,
  queries that name a table or column the database lacks, fell from 266 to 203 (-63), and
  wrong_tables from 80 to 71 (-9). Replies with no SQL (no_sql) went from 2 to 0, a class
  the base had already left nearly empty.
- **What didn't.** Most classes past schema linking grew: where from 13 to 22 (+9),
  projection from 51 to 58 (+7), group_order from 12 to 19 (+7), value from 10 to 17 (+7)
  and other from 0 to 3 (+3). Besides wrong_tables, the rest fell a little: join from 21 to
  20 (-1) and aggregation from 2 to 0 (-2). Of the classes before schema linking, syntax went
  from 0 to 2.
  The report counts each run's failures by class, so it can't say whether the classes that
  grew hold questions that now link the schema and then go wrong later in the query, which
  is the likely reading.
- **Hypotheses for GRPO.** schema_link is still the largest class, at 203. Those queries
  fail to run, so the validity reward marks down the model's own failing queries, which
  the SFT loss, on gold tokens, never sees; GRPO should cut them further. Nearly all the
  where, projection, group_order and value failures run and return the wrong rows, which
  only the execution reward against gold tells apart from right ones, so their counts show
  whether it teaches the logic. Extra, at +1.9 [-5.8, +9.6] after SFT, is where a gain
  will be hardest to show on val.

The main-model comparison, the SFT adapter of ADR-0012's chosen model against that model's
FG-307 val run on Kaggle T4s, is pending FG-404's runs ([pending.yaml](../runs/pending.yaml)).
It gets its own bullets here, from its own report, as the baseline GRPO has to beat; these
stay as the pilot's record.
