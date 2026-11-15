# ADR-0004: Val split by database from train, dev behind a six-use budget

- Status: accepted
- Date: 2026-11-14

## Context

Spider's dev set uses 20 databases that never appear in training, so its numbers measure
generalization to unseen schemas. Checkpoint selection and tuning need a held-out set
too. If it were cut by question, its schemas would also be in training and val would
overstate accuracy. Every look at dev is also a chance to tune against it. ADR-0003 and
the FG-101 spike add that seven training databases (the six `train_others` ones and
`music_2`) have no rows, so execution accuracy on them is meaningless.

## Decision

- **dev** is Spider's `dev.json`: 1,034 questions over 20 databases.
- **val** is 10% of the 146 training databases, rounded up (15), with all of their
  questions: the first ones when the databases are sorted by
  `sha256(db_id + "forge-val-v1")` hex, skipping databases without rows. On the full
  data that is 872 questions. The seed is a fixed constant (`VAL_SEED`), so val is the
  same on every run and machine.
- **train** is the other 131 databases: 7,787 questions.
- The split of the full download is committed in `src/forge/data/splits/`:
  `<split>.ids.txt` (`id<TAB>db_id`, one line per question) and `splits.json` (seed,
  percentage and each file's sha256), written by `forge data splits --write`. CI checks
  invariant I1 against them without the data, and a full-data test checks they match a
  fresh split.
- Training loads `load_train()` only. `assert_train_only()` refuses any val or dev
  example, by label and by database (checked against the committed manifests), and a
  static test fails if `src/forge/train/` names `load_val`, `load_dev`, `load_raw` or
  `split_examples`.
- `load_dev(reason=..., run_id=...)` raises without a reason, logs each use in
  `docs/dev-usage.md`, and refuses a seventh. It loads the data before it writes the row
  and raises `DataNotFound` when there are no dev questions (the fixture, a partial
  download), so a run that can't look at dev never spends a look.

## What counts as a dev use

A dev use is any run that shows model output on dev questions: generating answers for
them, or scoring or reading such answers. Each one needs `--reason`, loads dev through
`load_dev()`, and adds a row to `docs/dev-usage.md`. What makes it count is the model:
seeing how a model does on dev is what can steer later choices.

Reading dev without a model is not a dev use, needs no reason and writes no row:

- `forge eval gold --split dev`: each gold query scored against itself, which checks the
  checkers and the data.
- `forge data prompt-stats --split dev`: prompt token counts from questions and schemas.
- Counting questions and databases (`forge data stats`, `forge data splits`), and tests
  that run every gold query or rebuild every schema.

Both commands print a one-line note saying so when given `--split dev`.

## Consequences

- Changing the seed, the percentage or the row-less list moves databases between train
  and val, changes the data manifest hash, and needs `forge data splits --write`. Until
  the manifests are rewritten, `load_train()` refuses any database they hold out, so a
  stale manifest can't let val or dev into training.
- The six `train_others` databases and `music_2` stay in train, where SFT can use them;
  FG-505 filters them from GRPO.

## Revisit if

Val proves too small or too noisy to separate checkpoints. The change then takes a new
seed version and freshly written manifests, never hand edits.
