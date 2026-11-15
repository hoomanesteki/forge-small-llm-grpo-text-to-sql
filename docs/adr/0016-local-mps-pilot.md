# ADR-0016: Pilot evaluations on the laptop's GPU are labeled local-mps

- Status: accepted
- Date: 2026-11-15

## Context

Every number Forge reports is meant to come from Kaggle T4 runs (plan section 3.8), and
the Kaggle and Hugging Face accounts those runs need are still pending. Until they arrive,
the eval harness has only scored the fake backend and the tiny random-weight model, so
nothing has measured a real model on Spider. The laptop is an Apple M4, whose GPU PyTorch
reaches through its MPS (Metal) backend, and a 0.5B model fits in its memory in float32.
A pilot there gives real numbers now, on real questions, and finds problems in the harness
before any GPU hour is spent.

The `RunManifest` contract (plan section 3.5) knows the platforms `local-cpu`, `kaggle`,
`colab` and `modal`. A run on the M4's GPU is none of them, and filing it as `local-cpu`
would misstate how it was made. The device matters for a score: greedy decoding on
another device can pick another token where two are nearly tied.

## Decision

- `RunManifest.platform` gains `local-mps`: a run on Apple silicon's GPU through PyTorch's
  MPS backend. `forge.ops.manifest.Platform` and `forge.ops.ledger.PLATFORMS` take it too.
- `forge eval run --device cpu|mps` picks where the hf backend runs the model; `cpu` is
  the default. The manifest's config records the device, the manifest's platform follows
  it (`local-mps` for `mps`, `local-cpu` for `cpu`), and every report names the platform
  in its header. A device that torch can't use is refused; a run never moves to the CPU
  without being asked.
- A `local-mps` ledger row counts toward neither the Kaggle weekly quota nor the project's
  GPU hours. Both measure the free GPU budget, and the laptop spends neither.
- A `local-mps` run is a pilot. Pilot numbers are reported on val only, never on dev, so
  no pilot spends one of the six dev looks (ADR-0004); `forge eval run` refuses
  `--device mps` with `--split dev`. Their reports, manifests and scores are committed
  under `docs/runs/`, whose README says what the label means.
- The main results still come from Kaggle T4 runs, and FG-307 still chooses the model
  from them. A pilot number is never set beside a T4 number as the same measurement.

The first pilot (FG-291) evaluates Qwen/Qwen2.5-Coder-0.5B-Instruct (Apache-2.0, about
1 GB, not gated), pinned to a commit in `configs/models.yaml` under `pilot`, on the full
val split with greedy decoding and prompt p1, from `configs/eval/pilot-qwen25coder05b-val.yaml`.

## Consequences

- Pilot numbers are real but narrow: a 0.5B model, float32 weights, the hf backend on
  MPS. The T4 runs will use other models, float16 and vLLM (FG-306), so a pilot shows
  roughly where a small model starts, not a baseline for the uplift.
- Every eval config now records its device, so an unfinished run started before FG-291
  can't be resumed (its config hashes differently); start it again under a new id.
- `forge eval run` defaults to the CPU. Before FG-291 the hf backend took MPS whenever
  it was there, while the manifest still said `local-cpu`.
- I7 (FG-305) checks that every GPU run has a ledger row. Pilot runs get a row anyway,
  so I7 holds whichever way FG-305 counts them.

## Revisit if

The Kaggle and Hugging Face accounts arrive and T4 runs replace the pilots, a pilot is
wanted on dev, or a local CUDA machine joins the project (it would need its own platform).
