# ADR-0006: One TrainBackend, with TRL and PEFT on the GPU by default

- Status: proposed, to be confirmed by FG-301 measurements
- Date: 2026-11-15

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

## Decision (proposed)

- One `TrainBackend` interface (FG-402) for SFT and GRPO, with the same code path on the
  CPU and the GPU, so CI tests what the T4 runs.
- The GPU default is TRL 1.18 with PEFT LoRA (plan section 3.10: r=16 on every linear
  layer) in fp16 on the T4: `fp16=True`, `bf16=False`, SDPA attention. A base that
  doesn't fit in fp16 loads in 4-bit NF4 with fp16 compute, and its trainable parameters
  are cast back to fp32 once the trainer is built (TRL issue 7776). Rollouts use
  transformers generation, or TRL's colocated vLLM (`use_vllm=True`,
  `vllm_mode="colocate"`) if FG-301 finds a vLLM that installs beside this stack and
  runs on the T4.
- Unsloth is an optional accelerator behind the same interface. It is enabled only if
  FG-301 finds an Unsloth release that works with TRL 1.16 or later and runs on the T4.
  Today's release doesn't qualify.
- On the CPU (CI and the laptop) the same TRL backend trains in fp32 with `use_cpu=True`
  and a tiny model.
- The plan's fallback stands: if no vLLM path works on the T4, GRPO runs through TRL with
  transformers generation on the 1.7B model only, and the section 3.8 budget is redone
  for it.
- The GPU packages are `kaggle/requirements-gpu.txt`, the same versions as `uv.lock`
  where the lock has them (ADR-0002), except torch 2.14.0 and transformers 5.20.2: since
  FG-306 the file holds vllm 0.33.0 for evaluation, which needs both, so it is the
  spike's `trl-vllm` stack. If vLLM fails on the T4, the `trl` stack (the lock's
  versions on the image's torch) is the fallback.

## What FG-301 has to measure

Each is a stage of `kaggle/spike/throughput.py`:

- `grpo:trl:<model>`: seconds per step and peak memory for Qwen3 1.7B and 4B in 4-bit,
  6 generations of up to 256 tokens. This sets the GRPO budget.
- `grpo:trl-vllm:<model>`: whether TRL's colocated vLLM installs and runs on the T4, and
  what it saves per step.
- `grpo:unsloth:<model>`: the same steps through Unsloth's own T4 recipe, to show what
  Unsloth would gain if a compatible release appears.

Accept this record, or replace it, once those numbers are in. If it is accepted, the
GPU training rows of plan sections 3.12 and 3.13 change to match.

## Consequences

- Rollouts through transformers are slower than through vLLM, so GRPO steps cost more
  GPU time than the plan assumed until a vLLM path is confirmed. FG-301's numbers go into
  the section 3.8 budget at Gate 3.
- fp16 training can overflow where bf16 would not. Unsloth patches some of those cases;
  plain TRL does not, so GRPO jobs watch for NaN and infinite losses.
- The CPU smoke and the T4 job share one backend, one TRL version and one set of configs.

## Revisit if

An Unsloth release supports TRL 1.16 or later, a vLLM release supports transformers 5.23
or later, TRL fixes issues 7889 or 7776, or FG-301's measurements make the default too
slow for the budget.
