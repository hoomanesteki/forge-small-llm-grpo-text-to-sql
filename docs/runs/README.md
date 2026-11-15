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

To make it again, from the commit in its manifest:

```bash
FORGE_DATA_DIR=data/spider uv run forge eval run --config configs/eval/pilot-qwen25coder05b-val.yaml
FORGE_DATA_DIR=data/spider uv run forge report run <run_id>
```
