# ADR-0003: Spider 1.0 as the primary dataset, BIRD mini-dev as an optional check

- Status: accepted
- Date: 2026-11-14

## Context

Forge needs a text-to-SQL dataset with executable databases (rewards run the SQL), a
held-out set on unseen databases, a license that allows training and publishing, and
published numbers to compare against. The FG-101 spike (`docs/spikes/data.md`) checked
sources, license, counts and the official evaluator.

## Decision

- **Dataset:** Spider 1.0. Training uses `train_spider.json` plus `train_others.json`
  (8,659 questions over 146 databases), the official training set. `dev.json` (1,034
  questions over 20 databases, none shared with training) is used only for reported
  numbers (ADR-0004).
- **Source:** `data/spider.zip` from the Hugging Face dataset `xlangai/spider` at revision
  `6232cc3fad6d54c62b3ba23a364083a98ff36a17`, 99,736,136 bytes, sha256
  `5ddff97bb1d421282c593e8d30ce0ce107270f4dd4a21d60eba4bf287d5956b1`. It downloads with
  `hf_hub_download` and no token. Fallbacks: `HAL-9001/spider-databases` at revision
  `4a01bbac6520cd35b216db9e1724e5e1ada60aa4`, then the official Google Drive file
  `1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J`. A local zip is always accepted. Only the JSON
  files and `database/<db>/<db>.sqlite` are extracted, and their sha256 values are
  checked.
- **Evaluator:** the official test-suite evaluator, vendored from
  `taoyds/test-suite-sql-eval` at `e97acc546ecbee8fa27fa8dbf025ef61493a876c` as
  `test_suite_eval`, with read-only connections and a working timeout. Hardness comes
  from its `Evaluator.eval_hardness` on each example's parsed `sql` field. Classical EX
  on the original databases is used for val, CI and the agreement check. Test-suite EX
  on dev (it needs the 1.2 GB test-suite databases) is used for the headline numbers.
- **License:** CC BY-SA 4.0. The README, the model card and the fixture README carry the
  attribution and the Yu et al. (2018) citation. Any data Forge redistributes that is
  built from Spider stays CC BY-SA 4.0. Forge's code stays MIT, and the vendored
  evaluator keeps its Apache-2.0 license.
- **BIRD mini-dev** (CC BY-SA 4.0 per its card; FG-605 re-checks) is an optional
  out-of-domain check in FG-605, never in CI. Its databases are an 800 MB Google Drive
  download.

## Consequences

- The six `train_others` databases, and `music_2`, have no rows, so their gold results
  are empty or a single 0 or NULL. They are fine for SFT but give execution rewards no
  signal: FG-505 filters them from GRPO, and FG-104 keeps them out of val.
- Three training gold queries fail on SQLite. FG-103 flags them and FG-108 skips them.
- Spider is public and may be in a base model's pretraining data, so the main claim is
  the uplift from training (plan section 3.6).
- The headline dev evaluation (FG-508) needs the test-suite databases, downloaded once
  from Google Drive id `1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w`.

## Revisit if

The pinned revision disappears or its checksum changes, every fallback fails, or the
license terms change.
