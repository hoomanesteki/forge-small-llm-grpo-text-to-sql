# ADR-0006: One TrainBackend, with TRL and PEFT on the GPU by default

- Status: accepted
- Date: 2026-11-16 (proposed 2026-11-15)

## Context

Plan sections 3.12 and 3.13 expected Unsloth on the GPU (4-bit LoRA, fast GRPO with vLLM
generation) and TRL with PEFT on the CPU. The library facts as of 2026-11-15, from the
research in [the FG-301 spike](../spikes/kaggle-gpu.md):

- Unsloth 2026.11.4 requires `trl<=1.15.0`, `transformers<=5.17.0`, `datasets<5`,
  `torch<2.15` and `typer<0.29`. Forge's lock has trl 1.18.0, transformers 5.23.0,
  datasets 5.2.0, torch 2.15.0 and typer 0.29. TRL 1.15.0 can't train on a CPU or on
  Apple's MPS (a Triton-only kernel, fixed in 1.16.0), so no Unsloth release today runs
  the same TRL that CI tests.
- Unsloth's official Kaggle T4 GRPO notebook pins vllm 0.11.2, transformers 4.57.6 and
  trl 0.22.2, an older TRL API than Forge's code. Unsloth's newest T4 notebook (October
  2026) installs no vLLM on a T4 at all and generates with transformers.
- A T4 is compute capability 7.5: no bf16 in hardware and no FlashAttention 2. Two open
  TRL issues bite there: 7889 (the default `bf16=True` crashes training on a T4 since
  TRL 1.15; set `fp16=True`) and 7776 (TRL casts QLoRA adapters to bf16, which crashes
  fp16 training at the first optimizer step).
- vLLM 0.33.0 runs on a T4 in fp16, but pins torch 2.14.0 and `transformers<5.21`. TRL
  1.18's vLLM extra accepts `vllm>=0.23,<0.34`.

## Measured

FG-301's GPU session on 2026-11-16 benched each stack on one T4 of a Kaggle T4 x2 session
(`n_gpu` 1 in every GRPO bench), with Qwen3 1.7B and 4B in 4-bit, LoRA rank 16, fp16, 6
generations of up to 256 tokens and 24 completions a step, for 10 steps
(`docs/spikes/kaggle-gpu/spike-report.json`, question 5 of the spike doc):

| Stage | Versions | Median s a step | Peak allocated | Load s | Install s |
|---|---|---:|---:|---:|---:|
| `grpo:trl:Qwen/Qwen3-1.7B` | trl 1.18.0, transformers 5.23.0, the image's torch 2.11.0 | 18.43 | 3.02 GB | 5.3 | 20.0 |
| `grpo:trl:Qwen/Qwen3-4B` | the same | 32.92 | 4.94 GB | 9.3 | 20.0 |
| `grpo:unsloth:Qwen/Qwen3-1.7B` | trl 0.22.2, transformers 4.57.6, vllm 0.11.2, torch 2.9.0 | 11.64 | 12.75 GB | 139.4 | 124.4 |
| `grpo:unsloth:Qwen/Qwen3-4B` | the same | 21.22 | 13.66 GB | 129.5 | 124.4 |
| `grpo:trl-vllm:Qwen/Qwen3-1.7B` | trl 1.18.0, vllm 0.33.0, transformers 5.20.2, torch 2.14.0 | error | | | 95.9 |
| `grpo:trl-vllm:Qwen/Qwen3-4B` | the same | error | | | 95.9 |

The median is of steps 2 to 10, and the device holds 15.64 GB. Unsloth's peaks are mostly
the KV cache its colocated vLLM reserves (its standby mode set vLLM's share to 0.76), not
what training needs. Every loss the four benches that ran logged is finite. Both
`grpo:trl-vllm:*` stages failed before training, when vLLM loaded the 4-bit policy:

```text
ValidationError: 1 validation error for ModelConfig
  Value error, Unknown quantization method: bitsandbytes. Must be one of ['awq', ...]
```

## Decision

- One `TrainBackend` interface (FG-402) for SFT and GRPO, with the same code path on the
  CPU and the GPU, so CI tests what the T4 runs.
- GRPO and SFT on the T4 use TRL 1.18 with transformers generation and the lock's
  versions: the spike's `trl` stack, trl 1.18.0, transformers 5.23.0, peft 0.21.3,
  accelerate 1.15.0, datasets 5.2.0, huggingface-hub 2.4.0 and bitsandbytes 0.50.2,
  installed over the image's torch. It ran Qwen3 1.7B and 4B in 4-bit NF4 with fp16
  (`fp16=True`, `bf16=False`) and a finite loss over 10 steps, the 4B at 4.94 GB of the
  T4's 15.64 GB. LoRA comes from PEFT as plan section 3.10 says (r=16 on every linear
  layer), attention is SDPA, and a 4-bit base's trainable parameters are cast back to fp32
  once the trainer is built (TRL issue 7776).
- TRL's colocated vLLM is out: vLLM 0.33.0 has no bitsandbytes, so it can't load the 4-bit
  policy (the error above).
- Unsloth stays out. It ran and was faster a step (11.6 s against 18.4 s for 1.7B, 21.2 s
  against 32.9 s for 4B, with about 130 s of load), but on trl 0.22.2 and transformers
  4.57.6, which this record's condition, an Unsloth release on TRL 1.16 or later, still
  rules out.
- Training jobs install the measured `trl` stack, `kaggle/requirements-train-gpu.txt`,
  and evaluation jobs keep the vLLM stack, `kaggle/requirements-gpu.txt` (the spike's
  `trl-vllm` list, whose vLLM 0.33.0 generated in fp16 on the T4). Since FG-398 the
  kernel installs the list its job needs, and no training job gets vLLM; unit tests keep
  each file the same as its spike stack.
- One GPU a job: a session has two T4s, and every bench used one.
- On the CPU (CI and the laptop) the same TRL backend trains in fp32 with `use_cpu=True`
  and a tiny model.
- The plan's fallback, GRPO on the 1.7B model alone, isn't needed: the 4B fits.

## Consequences

- Rollouts go through transformers, at the `trl` stack's step times above. Plan section
  3.8's GRPO rows are worked out from them: the steps, the median step for the model's
  size, and a margin for Spider's longer prompts and real rewards.
- fp16 training can overflow where bf16 would not, and 10 steps of a cheap reward show
  nothing about hundreds of steps on a real one. Unsloth patches some of those cases;
  plain TRL does not, so GRPO jobs watch for NaN and infinite losses.
- The CPU smoke and the T4 job share one backend, one TRL version and one set of configs.
- Training and evaluation need two package lists, which the code dataset holds side by
  side (FG-398).
- Plan sections 3.12 and 3.13 named Unsloth for GPU training until Gate 3, which changed
  them to match this record on 2026-11-17.

## Revisit if

An Unsloth release supports TRL 1.16 or later: it measured 37% less time a step for 1.7B
and 36% for 4B, for 120 to 135 s more load. Also if a vLLM release loads bitsandbytes
weights again or supports transformers 5.23 or later, TRL fixes issue 7889 or 7776, or the
real GRPO steps make the section 3.8 budget not fit.
