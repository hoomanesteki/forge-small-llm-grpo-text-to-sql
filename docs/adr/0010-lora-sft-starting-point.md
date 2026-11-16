# ADR-0010: LoRA r=16 on every linear layer, and where SFT starts

- Status: accepted
- Date: 2026-11-15

## Context

Plan section 3.13 fixes LoRA at r=16 on all linear layers over a 4-bit base, and FG-403
names SFT's starting hyperparameters: 1 epoch, learning rate 2e-4 with a cosine schedule
and 3% warmup, an effective batch of about 16 and sequences of up to 2,048 tokens. TRL
1.18's `SFTConfig` starts elsewhere: a learning rate of 2e-5, meant for full fine-tuning,
a linear schedule with no warmup, and `max_length` 1,024. FG-401 measured the SFT rows
with the pilot's tokenizer: 333 p1 rows and 1,364 p2 rows are longer than 1,024 tokens
and none passes 2,048, and TRL's `keep_start` truncation would cut those rows' replies.

## Decision

These are the defaults of `forge.train.config` (FG-402), so a config that leaves a field
out takes them. The Kaggle configs and the local pilot's state them anyway, with the
changes the table and the Consequences name. `configs/sft/tiny-smoke.yaml`, the tiny
model's run for `make smoke`, takes them by omission, as its comment says: it sets only
its steps, batch and logging, such as `max_steps: 3`.

| Setting | Value | Why |
|---|---|---|
| LoRA rank, alpha, dropout | 16, 32, 0.0 | The plan's rank, alpha twice it as usual, and one epoch is too short to need dropout |
| LoRA targets | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | Every linear layer of a Qwen block, attention and MLP |
| Epochs | 1 | A second is the product owner's call once val EX is seen (FG-404) |
| Learning rate | 2e-4, cosine, 3% warmup | The usual LoRA rate, ten times TRL's full fine-tuning default |
| Effective batch | 16 rows | 4 rows by 4 accumulation steps; 2 by 8 on the 4B, for its memory |
| Max length | 2,048 tokens | Fits every row whole: prompts are capped at 1,536 tokens and replies are short |
| Loss | The reply only | TRL's `completion_only_loss` on the prompt-completion rows (FG-401) |
| Base on a T4 | 4-bit NF4, fp16 compute, fp32 adapters | ADR-0006; the Kaggle pilot loads in 4-bit too, so it runs the main run's path |
| Base on the CPU | float32 | bitsandbytes runs on CUDA only |
| Base on MPS | bfloat16 where `torch.backends.mps` supports it (macOS 14 and later), float32 otherwise | `precision_for` in `forge.train.backends` (ADR-0017); bitsandbytes runs on CUDA only |

## Consequences

- An SFT run's manifest holds every one of these values in its config, under the config
  hash, so a change to any of them is a new config hash. Its `trainer` section records the
  dtype the base trained in.
- The local pilot (`configs/sft/pilot-qwen25coder05b-mps.yaml`) trains the 0.5B pilot
  model at these LoRA values, learning rate, schedule, warmup and epoch count, so its val
  numbers show what this starting point does on a small model before any GPU hour is spent
  (ADR-0016). For the laptop's memory and time it changes the length, the batch split and
  the rows it trains on, as the note below records.

## Revisit if

The pilots' val EX shows the adapter underfitting or overfitting, the 4B run doesn't fit
a T4's memory at these values, or FG-404 adds a second epoch.

## Notes

- 2026-11-16, Phase 4 gate (FG-493). This record first said that every file in
  `configs/sft/` states the starting values, that the base trains at the device's full
  precision on the CPU and MPS, and that the local pilot trains at the same values. None of
  the three held, and the Decision and Consequences above now say what the code and the
  committed pilot did. The pilot run, `docs/runs/2026-11-16-qwen2-5-coder-0-5b-instruct-sft-db0d8d8/`,
  trained as its config says:
  - `max_length: 1536` rather than the 2,048 tokens above, for the laptop's memory. The
    rows longer than that are left out, not cut, and its manifest counts them:
    `"over_max_length": 53`.
  - `per_device_train_batch_size: 1` with `gradient_accumulation_steps: 16`: the same 16
    rows an optimizer step, one at a time, since two long rows at once ran out of the GPU
    memory the run may take. `pad_to_multiple_of: 64` bounds how many row lengths the MPS
    backend builds graphs for.
  - `subset_rows: 3168`: a fixed subset of the rows that fit, drawn by hash with the run's
    seed, rather than an epoch of train, since a timing probe put the epoch past the time
    the pilot had (FG-491); its manifest has `"trained": 3168`.
  - `"dtype": "bfloat16"` in its manifest's `trainer`: bf16 on MPS, as the table now says.
    Its val run scored the adapter on the float32 base, as FG-291's base run was
    (ADR-0016), so the two val runs differ in the adapter alone.
