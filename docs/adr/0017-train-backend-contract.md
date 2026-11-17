# ADR-0017: The TrainBackend protocol takes the seed and the output folder

- Status: accepted
- Date: 2026-11-16

## Context

Plan section 3.5 lists `TrainBackend` (FG-402) as `load(model_ref, lora, *, load_in_4bit)`,
`sft(model, tok, dataset, cfg, callbacks)` and `grpo(model, tok, dataset, rewards, cfg,
callbacks)`, and FG-402's Build named a `TrlCpuBackend` in float32, with
`select_backend("auto")` falling back to the CPU. FG-402 built a different protocol, and
the plan was not updated with it (Phase 4 gate, G4-06 and G4-01). Plan section 5.4 lets a
story change a section 3.5 contract only with an ADR, and FG-504 builds GRPO against this
one, so it has to say what the code does.

Three things the plan's signatures leave out turned out to be needed:

- The seed reaches `load`. PEFT draws LoRA's first `A` weights in `get_peft_model`, inside
  `load`, before any trainer exists to seed them, so without the run's seed two runs of one
  config start from different adapters. Unsloth's `get_peft_model` takes it as
  `random_state`.
- The job names where the trainer writes. `forge train sft` gives the trainer a temporary
  folder and keeps the run's own files in `runs/<run_id>/`, where the checkpoint callbacks
  push each checkpoint (`forge.ops.hub`); a kernel job does the same in its work folder.
  The trainer's arguments need that folder and the seed (data order, dropout), so `sft` and
  `grpo` take both.
- A run's manifest records what it trained with (FG-403's `trainer` section: backend,
  device and dtype), since the dtype changes the weights. So a backend says its `name`,
  `device` and `precision`.

FG-491's pilot trains on the laptop's GPU through MPS (ADR-0016), and ADR-0006 trains on
the T4 with TRL and PEFT, keeping Unsloth out until an Unsloth release supports TRL 1.16 or
later. One TRL backend on every device does both, where a CPU-only one would need a second
backend for each.

## Decision

`forge.train.backends.TrainBackend` is the protocol plan section 3.5 now shows:

```python
class TrainBackend(Protocol):
    name: str                # "trl" or "unsloth"
    device: str              # where it trains: cpu, mps or cuda
    precision: Precision     # the dtype, and the trainer's fp16, bf16 and use_cpu flags
    def load(self, model_ref: str, lora: LoraConfig, *, load_in_4bit: bool,
             seed: int) -> tuple[Any, Any]: ...  # model, tokenizer
    def sft(self, model, tok, dataset, cfg: SftConfig, callbacks: Sequence[TrainerCallback],
            *, output_dir: str | Path, seed: int) -> str: ...  # the adapter's folder
    def grpo(self, model, tok, dataset, rewards: Sequence[RewardFunc], cfg: GrpoConfig,
             callbacks: Sequence[TrainerCallback], *, output_dir: str | Path,
             seed: int) -> str: ...
```

- `seed`, `output_dir` and `load_in_4bit` are keyword-only and required: a caller can't
  leave the seed out by accident.
- `TrlBackend(device)` replaces `TrlCpuBackend`: transformers, PEFT LoRA and TRL's trainers
  on `cpu`, `mps` or `cuda`, generating with transformers rather than vLLM. It trains in
  float32 on the CPU, in bf16 on MPS from macOS 14 (float32 before), and on CUDA in bf16
  where the GPU has it and in fp16 on a T4 (TRL issue 7889). 4-bit loading is CUDA only.
- `UnslothBackend` is unchanged: CUDA only, imported when it is made.
- `select_backend("auto")` picks Unsloth when torch sees a CUDA GPU and Unsloth is
  installed, and otherwise `TrlBackend` on the best device there is: CUDA, then MPS, then
  the CPU. CI has only the CPU, so FG-402's "auto-select picks CPU in CI" holds.

The code's contract doesn't change here; the plan now describes it. `forge.core` holds no
part of it, so the gate's diff of `src/forge/core` never showed it.

## Consequences

- FG-504 and every later backend call pass `seed=` and `output_dir=`; a backend written to
  the plan's old signatures fails at the first call, not quietly.
- A local run picks MPS on a Mac with `device: auto`. Its manifest says `local-mps`
  (ADR-0016), so a config meant for the CPU names `device: cpu`, as
  `configs/sft/tiny-smoke.yaml` does.
- The manifest's `trainer` section is part of the config hash, so the same config trained
  in another dtype is another run.

## Revisit if

Unsloth becomes the GPU default (ADR-0006), or a backend needs a setting these signatures
have no place for.

## Notes

- 2026-11-17, Unsloth on Kaggle (FG-492). This record first said ADR-0006 makes Unsloth an
  option on the GPU. ADR-0006, as accepted, keeps it out until an Unsloth release supports
  TRL 1.16 or later, and the Context above now says so. `UnslothBackend` stays in the code,
  and `select_backend("auto")` still picks it where torch sees a CUDA GPU and Unsloth is
  installed, but no Kaggle training kernel has it: the T4 image has none (the spike's `env`
  stage in `docs/spikes/kaggle-gpu/spike-report.json`), and the kernel installs
  `kaggle/requirements-train-gpu.txt` (FG-398), which adds none. So `auto` picks
  `TrlBackend` on the T4.
