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
| [2026-11-16-qwen2-5-coder-0-5b-instruct-sft-db0d8d8](2026-11-16-qwen2-5-coder-0-5b-instruct-sft-db0d8d8/) | The same model, LoRA-trained on train rows (FG-491) | `configs/sft/pilot-qwen25coder05b-mps.yaml`: the subset its `subset_rows` sets of the train rows that fit in 1,536 tokens, bf16 on MPS | [metrics.jsonl](2026-11-16-qwen2-5-coder-0-5b-instruct-sft-db0d8d8/metrics.jsonl) |
| [2026-11-16-qwen2-5-coder-0-5b-instruct-eval-db0d8d8](2026-11-16-qwen2-5-coder-0-5b-instruct-eval-db0d8d8/) | The same model with that run's adapter | `configs/eval/pilot-qwen25coder05b-val.yaml` with `--adapter runs/2026-11-16-qwen2-5-coder-0-5b-instruct-sft-db0d8d8/checkpoints/final`, the first pilot's settings | [val-report.md](2026-11-16-qwen2-5-coder-0-5b-instruct-eval-db0d8d8/eval/val-report.md), and against the first pilot [compare-2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7.md](2026-11-16-qwen2-5-coder-0-5b-instruct-eval-db0d8d8/eval/compare-2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7.md) |

[docs/notes/sft.md](../notes/sft.md) reads that comparison, by hardness and by error class,
citing no number it doesn't hold (FG-405).

A pilot that trains (FG-491) commits its training run without its weights: the manifest,
`config.yaml` and `metrics.jsonl`, one line a logging step. The adapter stays in
`runs/<run_id>/checkpoints/final/` on the laptop that trained it, where the eval run's
`--adapter` names it, until the run store exists to hold it ([pending.yaml](pending.yaml)).
The manifest's `sft_data` says which rows it trained on: the hash of the rows
`forge data build-sft` builds, how many were over `max_length` and left out, and how many
of the rest the subset took.

Story branches land on `dev` as squash merges, so the commit a manifest names is not on
`dev` itself. Each committed run has a tag `run/<run_id>` on that commit, which keeps it
on GitHub: `git switch --detach run/<run_id>`.

To make them again, from the commit in their manifests, the base pilot first:

```bash
FORGE_DATA_DIR=data/spider uv run forge eval run --config configs/eval/pilot-qwen25coder05b-val.yaml
FORGE_DATA_DIR=data/spider uv run forge report run <run_id>
```

Then the SFT pilot, its adapter's val run, its report and the comparison, each with a
ledger row from its manifest. Both runs took the MPS watermarks the SFT config names, for
the laptop's memory; they change no number:

```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.45 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.35
FORGE_DATA_DIR=data/spider uv run forge train sft --config configs/sft/pilot-qwen25coder05b-mps.yaml
FORGE_DATA_DIR=data/spider uv run forge eval run --config configs/eval/pilot-qwen25coder05b-val.yaml \
  --adapter runs/<sft_run_id>/checkpoints/final
FORGE_DATA_DIR=data/spider uv run forge report run <eval_run_id>
FORGE_DATA_DIR=data/spider uv run forge report compare 2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7 <eval_run_id>
uv run forge ledger record <sft_run_id> && uv run forge ledger record <eval_run_id>
```

Training again on MPS need not give the same weights bit for bit, so a second adapter's
scores can differ a little from these.

## Pending runs

[pending.yaml](pending.yaml) lists the runs a story needs that can't happen yet, what blocks
each one, since when, and what it will fill in. FG-301's Kaggle spike, FG-307's baselines and
FG-404's SFT runs are there until the Kaggle account exists, and FG-491's push of the SFT
pilot, adapter and all, until the Hugging Face run store does. FG-405's comparison of the
main model with and without SFT waits there on FG-307's and FG-404's runs. `tests/integration/test_artifacts.py`
finds a GPU story's runs here by the configs they were made from, and skips its checks on
them only while the story has an entry in pending.yaml; once it has none, a missing run fails.
A Kaggle SFT run is committed as FG-491's is, without its checkpoints, which the run store
keeps, and with the `best.json` that `forge runs promote` writes: which checkpoint is
`checkpoints/best`, and the val runs whose scores chose it (docs/runbook.md, "SFT runs").
