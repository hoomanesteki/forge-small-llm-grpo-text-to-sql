# ADR-0012: The base model is chosen from measured val baselines

- Status: accepted
- Date: 2026-11-17 (proposed 2026-11-15)
- Chosen model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`

## Context

Training starts from one base model, and the uplift is measured on it (plan section 3.6),
so the choice has to come before SFT and from numbers rather than reputation. A coding
model can beat a larger general one on SQL, and a model that already scores very high
leaves little for training to show.

The candidates are open instruct models of about 1.5B to 4B, each pinned to a commit in
`configs/models.yaml`, whose license, gated flag and parameter count are the Hub's.
`scripts/check_models.py` checked them on 2026-11-15, and the reference's AWQ checkpoint
on 2026-11-16, reading metadata only, and checks them again whenever it is run:

| Name | Model | Parameters | License | Gated | Role |
|---|---|---|---|---|---|
| `qwen3-1-7b` | `Qwen/Qwen3-1.7B` | 2.03B | Apache-2.0 | no | candidate |
| `qwen3-4b` | `Qwen/Qwen3-4B` | 4.02B | Apache-2.0 | no | candidate |
| `qwen2-5-coder-1-5b` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 1.54B | Apache-2.0 | no | candidate |
| `qwen3-8b` | `Qwen/Qwen3-8B-AWQ` | 8.19B | Apache-2.0 | no | reference, 4-bit AWQ |

- The plan's example Qwen2.5-Coder-3B-Instruct is left out. Its Hub license is `other`,
  named `qwen-research`: the Qwen Research License, for non-commercial research and
  evaluation only, which would bind every model Forge publishes from it.
  Qwen2.5-Coder-1.5B-Instruct, Apache-2.0, keeps a coding model among the candidates.
- Qwen3-8B is about 16 GB of weights in 16 bits, more than a T4 holds, so the reference
  is Qwen's own 4-bit AWQ checkpoint of it, `Qwen/Qwen3-8B-AWQ`, which vLLM loads with
  `quantization: awq` (in its configs, recorded in its manifest). It shows what a larger
  model scores with the same prompt and harness, and is never a candidate. It was to load
  in 4-bit with bitsandbytes, but FG-301 found vLLM 0.33.0 has no such quantization (the
  spike's `vllm:Qwen/Qwen3-8B:bnb` stage, whose error lists `awq` among the methods it
  has), so FG-398 moved it to AWQ. vLLM's AWQ kernels need compute capability 7.5, which
  the T4 has; their Marlin form needs 8.0, so a T4 gets the plain ones. The reference's
  val run was the first to load the checkpoint on a T4, and it did (Measured).
- Every candidate's run is greedy on the whole val split with prompt p1, through vLLM in
  fp16 on a Kaggle T4 (FG-306), one kernel job per config in `configs/eval/`. Qwen3 runs in
  non-thinking mode (ADR-0007). The runs after the choice, the chosen model's dev run and
  the reference's two, use p2, the version ADR-0007 fixed from these runs.

## Decision: the selection rule, fixed before any number

1. Each candidate's val official EX, with its 95% interval, comes from its p1 run's report.
2. A candidate is eligible when:
   - FG-301's GRPO memory check passes for it on the T4: the spike's GRPO stage for its
     size (Qwen3 1.7B and 4B, 4-bit, 6 generations of up to 256 tokens, ADR-0006) ran
     without running out of memory. A model smaller than 1.7B counts as passing when 1.7B
     passes.
   - It leaves visible headroom: val official EX below 75%.
3. The eligible candidate with the best val official EX is chosen. When the best two don't
   differ significantly (`forge report compare`, McNemar p of 0.05 or more), the one with
   fewer parameters is, since each GRPO step on it costs less.
4. The chosen model and the reference are evaluated on dev once each, with
   `--reason baseline`: dev looks 1 and 2 of 6 (ADR-0004). Nothing is chosen on dev.
5. If no candidate is eligible, nothing is chosen: the rule has no answer, so the story
   stops and asks the product owner.

The runs and their configs are in `docs/runbook.md` ("Baselines"), and each run is
committed under `docs/runs/`, within FG-307's 3 GPU hours (plan section 3.8's note gives
their ledger hours). `docs/runs/pending.yaml` listed each run, with its estimate, until it
was committed.

## Measured

FG-301, 2026-11-16: the GRPO memory check of rule 2 passes for every candidate. The
spike's `grpo:trl:Qwen/Qwen3-1.7B` and `grpo:trl:Qwen/Qwen3-4B` stages, on the stack
ADR-0006 chose, ran 10 steps each without running out of memory, at a peak of 3.02 GB and
4.94 GB allocated of the T4's 15.6 GB (`docs/spikes/kaggle-gpu/spike-report.json`).
Qwen2.5-Coder-1.5B-Instruct, smaller than 1.7B, counts as passing, as rule 2 says.

FG-307, 2026-11-16: each candidate on the whole val split with p1, from `c8f4b41` on a
Kaggle T4, from the reports committed under `docs/runs/` (official EX by hardness on the
right):

| Candidate | Run | Official EX [95% interval] | Fast EX | Valid SQL | Easy | Medium | Hard | Extra |
|---|---|---|---|---|---:|---:|---:|---:|
| `Qwen/Qwen3-1.7B` | [2026-11-16-qwen3-1-7b-val-c8f4b41](../runs/2026-11-16-qwen3-1-7b-val-c8f4b41/eval/val-report.md) | 55.7% [51.7%, 59.9%] | 54.9% [50.9%, 59.2%] | 89.2% [86.9%, 91.6%] | 75.0% | 58.2% | 49.3% | 31.2% |
| `Qwen/Qwen3-4B` | [2026-11-16-qwen3-4b-val-c8f4b41](../runs/2026-11-16-qwen3-4b-val-c8f4b41/eval/val-report.md) | 43.1% [39.2%, 47.2%] | 42.9% [39.0%, 47.0%] | 54.9% [51.1%, 59.2%] | 72.5% | 43.2% | 30.0% | 17.2% |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | [2026-11-16-qwen2-5-coder-1-5b-val-c8f4b41](../runs/2026-11-16-qwen2-5-coder-1-5b-val-c8f4b41/eval/val-report.md) | 65.3% [61.2%, 69.1%] | 65.1% [61.0%, 69.0%] | 85.2% [82.2%, 88.1%] | 83.8% | 69.3% | 59.3% | 37.6% |

All three leave the headroom rule 2 asks for. How `extract_sql` read their replies, from
`uv run python scripts/count_extraction.py` on the three runs (ADR-0007):

| Run | Replies | sql block | Rule 3 | No SQL | Rule 3 cut a query | Cut, by class |
|---|---|---|---|---|---|---|
| 2026-11-16-qwen3-1-7b-val-c8f4b41 | 872 | 865 | 7 | 0 | 4 | schema_link 3, wrong_tables 1 |
| 2026-11-16-qwen3-4b-val-c8f4b41 | 872 | 145 | 726 | 1 | 328 | schema_link 322, syntax 2, wrong_tables 4 |
| 2026-11-16-qwen2-5-coder-1-5b-val-c8f4b41 | 872 | 764 | 106 | 2 | 0 | - |

Rule 3 read 726 of Qwen3-4B's replies, which had no code block, and the model wrote most
of those queries over several lines, so rule 3 kept their first line, often a select list
without its `FROM`, and 322 of the 328 queries it cut fail as `schema_link`. Its official EX
is partly the format's, then, and the choice could turn on the extraction rule, which stays
as ADR-0007 decided. So `scripts/check_choice_extraction.py` applies the rule twice: to the
stored scores, and with each query rule 3 cut read whole, up to a blank line or a code
fence, and scored as a run scores a query (`scripts/count_pilot_extraction.py`'s reading,
which finds more cut queries than `count_extraction.py`'s, as it doesn't ask the joined
lines to parse as one query). Its output, from
`FORGE_DATA_DIR=data/spider uv run python scripts/check_choice_extraction.py` on the three
runs:

```text
As stored:
- Qwen/Qwen3-1.7B (2.03B parameters): official EX 55.7% [51.7%, 59.9%], eligible
- Qwen/Qwen3-4B (4.02B parameters): official EX 43.1% [39.2%, 47.2%], eligible
- Qwen/Qwen2.5-Coder-1.5B-Instruct (1.54B parameters): official EX 65.3% [61.2%, 69.1%], eligible
- Qwen/Qwen2.5-Coder-1.5B-Instruct against Qwen/Qwen3-1.7B: +9.5 points [+5.0, +13.6], McNemar p < 0.0001 on clusters
- chosen: Qwen/Qwen2.5-Coder-1.5B-Instruct, the best, ahead of the next by p below 0.05
With the queries rule 3 cut read whole:
- Qwen/Qwen3-1.7B (2.03B parameters): official EX 55.8% [51.8%, 60.0%], eligible, 5 queries read whole
- Qwen/Qwen3-4B (4.02B parameters): official EX 66.7% [62.9%, 70.4%], eligible, 389 queries read whole
- Qwen/Qwen2.5-Coder-1.5B-Instruct (1.54B parameters): official EX 65.3% [61.2%, 69.1%], eligible, 0 queries read whole
- Qwen/Qwen3-4B against Qwen/Qwen2.5-Coder-1.5B-Instruct: +1.5 points [-2.4, +5.2], McNemar p = 0.3755 on clusters
- chosen: Qwen/Qwen2.5-Coder-1.5B-Instruct, the one with fewer parameters, as the best two don't differ significantly (p of 0.05 or more)
```

Both readings choose the same model. As stored, Qwen2.5-Coder-1.5B-Instruct leads the next
candidate by a significant margin; read whole, Qwen3-4B's official EX passes it, but not
significantly, and the tie-break of rule 3 takes the model with fewer parameters.

I6 holds on the T4: the first 100 val questions again (run
`2026-11-16-qwen2-5-coder-1-5b-i6-c8f4b41`), by `forge.eval.reproducibility.compare_runs`
against the whole run, "official EX 67.0% and 67.0% on the 100 samples both scored, 0.0
points apart (I6 allows 0.5); 0 verdicts and 1 replies differ"
(`test_phase3_baseline_reports_present`).

FG-307, 2026-11-17: the reference, Qwen3-8B in 4-bit as Qwen's AWQ checkpoint
`Qwen/Qwen3-8B-AWQ`, through vLLM 0.33.0 (`kaggle/requirements-gpu.txt`) on a Kaggle T4, on
the whole val split with p2, from `2f77281`. It is the first run of the checkpoint on a T4,
and it loaded and scored:

| Reference | Run | Official EX [95% interval] | Fast EX | Valid SQL | Easy | Medium | Hard | Extra |
|---|---|---|---|---|---:|---:|---:|---:|
| `Qwen/Qwen3-8B-AWQ` | [2026-11-17-qwen3-8b-awq-val-2f77281](../runs/2026-11-17-qwen3-8b-awq-val-2f77281/eval/val-report.md) | 71.1% [67.2%, 74.9%] | 71.1% [67.2%, 74.7%] | 97.1% [95.7%, 98.4%] | 82.4% | 78.1% | 56.7% | 54.1% |

It ran with p2, where the candidates above ran with p1, so the run to set beside it is the
chosen model's p2 run (ADR-0007). How `extract_sql` read its replies, from
`uv run python scripts/count_extraction.py` on its run:

| Run | Replies | sql block | Rule 3 | No SQL | Rule 3 cut a query | Cut, by class |
|---|---|---|---|---|---|---|
| 2026-11-17-qwen3-8b-awq-val-2f77281 | 872 | 872 | 0 | 0 | 0 | - |

Then rule 4, with p2: the chosen model and the reference on dev once each, with
`--reason baseline` (`docs/dev-usage.md`): the chosen model from `2f77281` as dev look 1,
and the reference from `d2d6a3a`, once its val run had loaded the AWQ checkpoint, as dev
look 2:

| Model | Run | Official EX [95% interval] |
|---|---|---|
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | [2026-11-17-qwen2-5-coder-1-5b-dev-2f77281](../runs/2026-11-17-qwen2-5-coder-1-5b-dev-2f77281/eval/dev-report.md) | 67.2% [63.6%, 70.7%] |
| `Qwen/Qwen3-8B-AWQ` | [2026-11-17-qwen3-8b-awq-dev-d2d6a3a](../runs/2026-11-17-qwen3-8b-awq-dev-d2d6a3a/eval/dev-report.md) | 70.8% [67.2%, 74.0%] |

Nothing is chosen on dev: the choice stands from val.

I5 holds as ADR-0008's note of 2026-11-17 reads it: over FG-307's runs pooled, and on each
run of the chosen model alone (`test_i5_agreement_on_committed_runs`). The reference's dev
run is the one run here under 98% on its own, and on every run the checkers part only
where the evaluator drops `DISTINCT` (`scripts/check_i5_distinct.py`). Each report counts
its disagreements:

- [2026-11-16-qwen3-1-7b-val-c8f4b41](../runs/2026-11-16-qwen3-1-7b-val-c8f4b41/eval/val-report.md): The checkers disagree on 17 samples: only fast EX matches 5, and only official EX 12.
- [2026-11-16-qwen3-4b-val-c8f4b41](../runs/2026-11-16-qwen3-4b-val-c8f4b41/eval/val-report.md): The checkers disagree on 4 samples: only fast EX matches 1, and only official EX 3.
- [2026-11-16-qwen2-5-coder-1-5b-val-c8f4b41](../runs/2026-11-16-qwen2-5-coder-1-5b-val-c8f4b41/eval/val-report.md): The checkers disagree on 9 samples: only fast EX matches 4, and only official EX 5.
- [2026-11-17-qwen3-8b-awq-val-2f77281](../runs/2026-11-17-qwen3-8b-awq-val-2f77281/eval/val-report.md): The checkers disagree on 14 samples: only fast EX matches 7, and only official EX 7.
- [2026-11-17-qwen2-5-coder-1-5b-dev-2f77281](../runs/2026-11-17-qwen2-5-coder-1-5b-dev-2f77281/eval/dev-report.md): The checkers disagree on 17 samples: only fast EX matches 6, and only official EX 11.
- [2026-11-17-qwen3-8b-awq-dev-d2d6a3a](../runs/2026-11-17-qwen3-8b-awq-dev-d2d6a3a/eval/dev-report.md): The checkers disagree on 25 samples: only fast EX matches 13, and only official EX 12.

Nothing here is typed by hand: `tests/integration/test_artifacts.py` checks the tables and
the I5 lines against the reports, and `tests/unit/test_count_extraction.py` and
`test_check_choice_extraction.py` check the scripts' output above.

## Consequences

- Training configs (plan section 3.10, `model_ref`) name `Qwen/Qwen2.5-Coder-1.5B-Instruct`
  at the commit `configs/models.yaml` pins, with prompt p2 (ADR-0007): Phase 4's configs,
  from FG-402 on. FG-496 made FG-404's SFT configs do so, the pilot's and the main run's,
  as the second note below says.
- The model is smaller than the Qwen3 1.7B that FG-301 benched for GRPO, so plan section
  3.8 budgets its training and evaluations at 1.7B's measured rates.
- Its val official EX, 65.3%, is below the 75% headroom line, and its hard and extra
  questions, at 59.3% and 37.6%, leave training the most room.
- Its dev run with p2 (Measured) is the baseline the trained model's dev evaluation is
  set against, with the same prompt (plan section 3.6).

## Notes

- 2026-11-17, Phase 4's configs (FG-494). The first consequence above holds for the configs
  of the runs that train the chosen model, and Phase 4's didn't meet it then: FG-403 wrote
  its SFT configs before this choice, and `configs/sft/qwen3-4b-main.yaml` trained Qwen3-4B
  on p1. The main run had to train `Qwen/Qwen2.5-Coder-1.5B-Instruct` on p2. The SFT pilot,
  `configs/sft/qwen3-1p7b-pilot.yaml`, was to stay on Qwen3-1.7B, the small model plan
  section 3.8 budgeted it on, but train on p2 too; the next note replaces that. FG-404's
  runs waited on those changes (`docs/runs/pending.yaml`).
- 2026-11-17, Phase 4 gate (FG-496). The gate changed the pilot: both of FG-404's Kaggle
  SFT runs train the chosen model, on p2, in fp16 on the T4 with no 4-bit, since the 1.5B
  base fits. The chosen model is now the smallest candidate, so a Qwen3-1.7B pilot would
  rehearse a bigger model than the main run, and the pilot is there to rehearse the main
  run's Kaggle path end to end. So `configs/sft/qwen25coder-1p5b-pilot.yaml` is a short run
  with the main run's settings: its first session stops after step 50
  (`ops.stop_after_step`), and a second, `forge kaggle run --resume`, restores the run from
  the first kernel's output and finishes it. `configs/sft/qwen25coder-1p5b-main.yaml` is
  the main run, an epoch, and the two Qwen3 configs are gone. Plan section 3.8's SFT pilot
  row names the chosen model, and its checkpoints are scored with
  `configs/eval/qwen2-5-coder-1-5b-val-p2.yaml`, as the main run's are.
