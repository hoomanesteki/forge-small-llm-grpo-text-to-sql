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
out takes them, and every file in `configs/sft/` states them anyway.

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
| Base on the CPU and MPS | The device's full precision | bitsandbytes runs on CUDA only |

## Consequences

- An SFT run's manifest holds every one of these values in its config, under the config
  hash, so a change to any of them is a new config hash.
- The local pilot (`configs/sft/pilot-qwen25coder05b-mps.yaml`) trains the 0.5B pilot
  model at the same values, so its val numbers show what this starting point does on a
  small model before any GPU hour is spent (ADR-0016).

## Revisit if

The pilots' val EX shows the adapter underfitting or overfitting, the 4B run doesn't fit
a T4's memory at these values, or FG-404 adds a second epoch.
