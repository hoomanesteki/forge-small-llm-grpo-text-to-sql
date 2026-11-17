# ADR-0011: GRPO's starting hyperparameters

- Status: accepted
- Date: 2026-11-16

## Context

Plan section 3.10 sketches a GRPO config for the main run and leaves `beta` to the
library, asking that the value used be recorded. FG-504 builds the trainer on TRL 1.18.0
(the lock's), whose `GRPOConfig` has changed a good deal since the plan was written:
`max_prompt_length` is gone, `beta` defaults to 0, so no reference model is loaded, and
the default loss is `dapo` rather than the original GRPO one. No GRPO run has been made
yet, so these are starting values, not measured ones.

## Decision

`forge.train.config.GrpoConfig` starts from these values, and
`configs/grpo/qwen3-4b-main.yaml` writes them out:

| Setting | Value | Why |
|---|---|---|
| `num_generations` | 6 | The plan's group size |
| `per_device_train_batch_size`, `gradient_accumulation_steps` | 6 and 4 | 24 completions a step, 4 prompts with their whole groups (the config checks the group divides the batch) |
| `max_prompt_length` | 1,536 tokens | Forge's own limit (`forge.data.prompts`): a longer prompt is left out when the rows are built, since TRL no longer has the setting |
| `max_completion_length` | 256 tokens | A SQL reply needs far fewer, and each token costs T4 memory in every reply of a group |
| `learning_rate` | 5e-6 | The plan's |
| `max_steps` | 600 | The plan's, to be fitted to the step time FG-507 measures |
| `temperature` | 1.0 | TRL's; rollouts sample the policy as it stands |
| `beta` | `library_default`: 0.0 in TRL 1.18 | As the plan says, TRL's choice: no KL term and no reference pass. The run's manifest records the value |
| `num_iterations` | 1, with `steps_per_generation` left to TRL | Each optimizer step gets its own generation batch, so a checkpoint always falls between two and a resume replays the same prompts (TRL issue 7787) |
| rewards | execution 1.0, validity 0.1, format 0.05; guards -1.0, -0.5, -0.5, -0.2 | FG-501's; TRL weighs them (`reward_weights`), so each `rewards/<name>/mean` stays a share |
| in-training eval | every 100 steps, 200 val questions, fast EX | `forge.eval.callbacks`; each eval step has a checkpoint, kept for FG-508 to choose among |
| checkpoints | every 50 steps | `ops.checkpoint_every`, through the run store's callbacks rather than TRL's `save_steps` |

What TRL decides that the config leaves to it is recorded in each run's manifest under
`trl`, with TRL's version: `beta`, `loss_type` (`dapo`), `scale_rewards` (`group`),
`epsilon` (0.2), `top_p` (1.0) and `top_k` (0). It is part of the config hash, so a TRL
release that changes one of them makes another run rather than the same one.

## Consequences

- A change to any of these values is a change to a run's config, and shows in its
  manifest and config hash; the ops section is the exception, since it changes no weight.
- Without a reference model nothing holds the policy near the SFT adapter but the
  clipped ratio and the low learning rate. FG-506's telemetry watches completion length
  and the valid rate for drift.

## Revisit if

FG-507's pilot shows no learning signal (the zero-variance group fraction stays high), a
drift a KL term would have held back, or a step time that makes 600 steps miss the
budget.
