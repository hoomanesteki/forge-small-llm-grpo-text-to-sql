# Committed runs

Each folder here is a finished run copied whole from the run folder that made it
(`runs/<run_id>/`, plan section 3.9): its `manifest.json` and `config.yaml`, the
generations and scores in `eval/`, and the report and chart `forge report run` wrote from
them. No number here is typed by hand. To check a report against its files:

```bash
FORGE_RUNS_DIR=docs/runs uv run forge report rebuild <run_id>
```

That needs the data the run was made on, whose hash is in its manifest
(`FORGE_DATA_DIR=data/spider` for the full Spider download). It compares a chart byte for
byte only where the releases and platform its PNG names draw it, and elsewhere says it
didn't. Reports and generations quote Spider 1.0 (Yu et al., EMNLP 2018), licensed CC BY-SA
4.0.

## Pilot runs

A run whose manifest says `platform: local-mps` is a pilot (ADR-0016). It ran on the
laptop's GPU, an Apple M4 through PyTorch's MPS backend, while the Kaggle and Hugging Face
accounts the GPU plan needs were pending. Its report names the platform in its header.

- Pilot numbers are real: a real model on the real val split, scored by the same harness,
  official EX included.
- They are on val only, never on dev, so no pilot spends one of the six dev looks
  (ADR-0004). `forge eval run` refuses `--device mps` with `--split dev`.
- They are not the project's results. Those come from Kaggle T4 runs (plan section 3.8),
  with other models, float16 weights and vLLM. A pilot number is never set beside a T4
  number as the same measurement, and FG-307 chooses the model from the T4 runs.
- A pilot's ledger row (`docs/compute-ledger.md`) counts toward neither the Kaggle quota
  nor the project's GPU hours.

| Run | Model | Made with | Report |
|---|---|---|---|
| [2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7](2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7/) | `Qwen/Qwen2.5-Coder-0.5B-Instruct` (Apache-2.0), pinned in `configs/models.yaml` | `configs/eval/pilot-qwen25coder05b-val.yaml`: full val split, greedy, prompt p1, float32 on MPS | [val-report.md](2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7/eval/val-report.md) |

Story branches land on `dev` as squash merges, so the commit a manifest names is not on
`dev` itself. Each committed run has a tag `run/<run_id>` on that commit, which keeps it
on GitHub: `git switch --detach run/<run_id>`.

To make it again, from the commit in its manifest:

```bash
FORGE_DATA_DIR=data/spider uv run forge eval run --config configs/eval/pilot-qwen25coder05b-val.yaml
FORGE_DATA_DIR=data/spider uv run forge report run <run_id>
```

## Kaggle runs

A run whose manifest says `platform: kaggle` ran as a kernel job on a T4 (ADR-0005) and came
back with `forge runs pull <run_id> --store kaggle`. Its folder also holds what the kernel
kept beside the run: `kernels.json` (the pushes `forge kaggle run` made), `status.json` (each
stage of `kaggle/entry.py`, with its seconds and result) and the preflight's files in
`kernel/`. The kernel's log stays out of git. Its ledger row is the kernel's session, from
its start to its end.

| Run | Model | Made with | Report |
|---|---|---|---|
| [2026-11-16-smoke-kaggle-50-val-c8f4b41](2026-11-16-smoke-kaggle-50-val-c8f4b41/) | `Qwen/Qwen2.5-Coder-1.5B-Instruct` (Apache-2.0), pinned in `configs/models.yaml` | `configs/eval/smoke-kaggle-50-val.yaml`: the first 50 val questions, greedy, prompt p1, vLLM in fp16 on a T4. Integration checkpoint 3's job (`scripts/smoke/phase_3.sh`) and FG-303's real eval job: a check of the GPU path, not a baseline | [val-report.md](2026-11-16-smoke-kaggle-50-val-c8f4b41/eval/val-report.md) |

Its report was written in the kernel and matches its rebuild on the laptop with the full
data; its chart was drawn on Linux, which a Mac's rebuild says it can't compare. It was made
from `c8f4b41`, a commit on `dev`, so it needs no `run/` tag. To make it again from there:
`scripts/smoke/phase_3.sh <run_id>`.

## Pending runs

[pending.yaml](pending.yaml) lists the runs a story needs that haven't been made and
committed yet, what blocks each one, since when, and what it will fill in. FG-307's
baselines are there until they come back from Kaggle; FG-301's spike, FG-302's publish and
FG-303's eval job left it on 2026-11-16. `tests/integration/test_artifacts.py` finds a GPU
story's runs here by the configs they were made from, and skips its checks on them only
while the story has an entry in pending.yaml; once it has none, a missing run fails.
