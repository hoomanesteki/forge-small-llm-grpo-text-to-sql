# ADR-0012: The base model is chosen from measured val baselines

- Status: proposed, the measured table pending FG-307's runs (`docs/runs/pending.yaml`)
- Date: 2026-11-15
- Chosen model: pending FG-307's runs

## Context

Training starts from one base model, and the uplift is measured on it (plan section 3.6),
so the choice has to come before SFT and from numbers rather than reputation. A coding
model can beat a larger general one on SQL, and a model that already scores very high
leaves little for training to show.

The candidates are open instruct models of about 1.5B to 4B, each pinned to a commit in
`configs/models.yaml`, whose license, gated flag and parameter count are the Hub's.
`scripts/check_models.py` checked them on 2026-11-15, reading metadata only, and checks
them again whenever it is run:

| Name | Model | Parameters | License | Gated | Role |
|---|---|---|---|---|---|
| `qwen3-1-7b` | `Qwen/Qwen3-1.7B` | 2.03B | Apache-2.0 | no | candidate |
| `qwen3-4b` | `Qwen/Qwen3-4B` | 4.02B | Apache-2.0 | no | candidate |
| `qwen2-5-coder-1-5b` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 1.54B | Apache-2.0 | no | candidate |
| `qwen3-8b` | `Qwen/Qwen3-8B` | 8.19B | Apache-2.0 | no | reference, 4-bit |

- The plan's example Qwen2.5-Coder-3B-Instruct is left out. Its Hub license is `other`,
  named `qwen-research`: the Qwen Research License, for non-commercial research and
  evaluation only, which would bind every model Forge publishes from it.
  Qwen2.5-Coder-1.5B-Instruct, Apache-2.0, keeps a coding model among the candidates.
- The reference is about 16 GB of weights in 16 bits, more than a T4 holds, so vLLM loads
  it in 4-bit with bitsandbytes (`quantization` in its configs, recorded in its manifest).
  It shows what a larger model scores with the same prompt and harness, and is never a
  candidate. FG-301 found that vLLM 0.33.0 doesn't load bitsandbytes weights: it has no
  such quantization (the spike's `vllm:Qwen/Qwen3-8B:bnb` stage), so FG-398 moves the
  reference to an AWQ checkpoint.
- Every run is greedy on the whole val split with prompt p1, through vLLM in fp16 on a
  Kaggle T4 (FG-306), one kernel job per config in `configs/eval/`. Qwen3 runs in
  non-thinking mode (ADR-0007).

## Decision (proposed): the selection rule, fixed before any number

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

The runs, their configs and their estimates are listed in `docs/runs/pending.yaml` and
`docs/runbook.md` ("Baselines"), within FG-307's 3 GPU hours.

## Measured

FG-301, 2026-11-16: the GRPO memory check of rule 2 passes for every candidate. The
spike's `grpo:trl:Qwen/Qwen3-1.7B` and `grpo:trl:Qwen/Qwen3-4B` stages, on the stack
ADR-0006 chose, ran 10 steps each without running out of memory, at a peak of 3.02 GB and
4.94 GB allocated of the T4's 15.6 GB (`docs/spikes/kaggle-gpu/spike-report.json`).
Qwen2.5-Coder-1.5B-Instruct, smaller than 1.7B, counts as passing, as rule 2 says.

The rest is pending FG-307's runs. This section will hold, for each candidate and the
reference, the val official EX with its interval, fast EX, valid SQL and EX by hardness,
and the dev EX of the chosen model and the reference, all from the reports committed under
`docs/runs/`.
Nothing here is typed by hand. `tests/integration/test_artifacts.py` fails once FG-307
leaves `pending.yaml` without those runs, and checks I5 on their replies and I6 on a
repeat of the first 100 val questions.

## Consequences

- Training configs (plan section 3.10, `model_ref`) name the chosen model once this ADR is
  accepted, and the SFT and GRPO estimates in section 3.8 are redone for its size.
- If FG-301 finds no GRPO stack that fits the 4B model, the rule leaves the 1.7B-class
  candidates, as ADR-0006's fallback already plans.
