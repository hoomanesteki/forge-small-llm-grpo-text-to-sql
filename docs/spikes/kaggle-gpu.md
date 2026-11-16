# Spike FG-301: Kaggle, Unsloth and vLLM on a T4

Blocked: the spike needs a phone-verified Kaggle account and its API token, which don't
exist yet. Everything that runs without them is built and tested on CPU. This page keeps
two things apart: what research says today, with sources, and what the first GPU session
has to measure. No number under "To be measured" has been measured; each one says so
until a run fills it in from its report.

- Date: 2026-11-15. Timebox: 1 day. GPU budget: 1.5 hours (plan section 3.8).
- Built: the kernel metadata template and its checks (`kaggle/kernel-metadata.template.json`,
  `forge.ops.kaggle`), the spike script `kaggle/spike/throughput.py`, the bundler
  `scripts/kaggle_spike.py`, and candidate GPU pins in `kaggle/requirements-gpu.txt`.
- Decided from research: [ADR-0005](../adr/0005-kaggle-compute.md) (accepted) and
  [ADR-0006](../adr/0006-train-backend.md) (proposed until this spike measures it).
- Pending on the first GPU session: every "To be measured" item below, the ledger row,
  and the section 3.8 budget update at Gate 3. `docs/runs/pending.yaml` lists the runs.

## Running the spike

Once `uv run forge doctor` shows Kaggle auth ok, on a clean commit that passed `make smoke`:

1. `kaggle quota -v`: at least 1.5 GPU hours left. Note the GPU hours used.
2. `uv run python scripts/kaggle_spike.py --owner <kaggle-username>`. It writes
   `build/kaggle-spike/gpu/` and `build/kaggle-spike/limit/`, each with a checked
   `kernel-metadata.json` and the spike script with its arguments written in, and prints
   the kaggle commands for each; step 4 pushes the GPU kernel through `forge kaggle run`
   instead.
3. For question 2, create the GPU kernel without running it, then attach a secret:
   `kaggle kernels push -p build/kaggle-spike/gpu --no-run`, open the kernel in the
   Kaggle editor, and under Add-ons, Secrets attach `HF_TOKEN`. A read-only token is
   enough: the stage only checks that it can be read, and never records its value.
4. Push the GPU kernel through the job runner, which checks the smoke marker, the clean
   tree and the week's budget first (FG-303, `docs/runbook.md` "Kaggle jobs"):
   `uv run forge kaggle run --job spike --config configs/spike/throughput.yaml`.
   It pushes a new version of the same `<owner>/forge-throughput-spike` kernel, on a T4,
   and prints the run id. Its timeout is the 1.5-hour estimate and half again, but the
   script starts no stage after 80 minutes and bounds each stage by its own cap and the
   time left, so the session ends long before the cut.
5. `uv run forge kaggle wait <run_id>` until it is `COMPLETE` or `ERROR`, then
   `uv run forge kaggle output <run_id>`. That brings back `spike-report.json` and the
   log, `forge-throughput-spike.log`, into `runs/<run_id>/kernel/`.
6. Push the session-limit probe as a CPU kernel, which spends no GPU quota:
   `kaggle kernels push -p build/kaggle-spike/limit -t 900`. After 15 minutes, check its
   status, fetch its output the same way, and compare `last_beat_at` in
   `limit-probe.json` with the 900-second timeout. Its `spike-report.json` holds only
   `launch`: how Kaggle started the CPU kernel.
7. `kaggle quota -v` again. The difference in GPU hours used goes in the ledger:
   `uv run forge ledger add --platform kaggle --gpu T4 --hours <h> --run-id <run_id> --purpose "FG-301 throughput spike"`,
   with the run id step 4 printed (`<YYYY-MM-DD>-throughput-spike-<sha7>`).
   The CPU probe gets no row.
8. Commit both JSON reports under `docs/spikes/kaggle-gpu/`, fill in every "To be
   measured" item below from them, and update ADR-0006 and the section 3.8 budget.

If the GPU kernel fails in its first 10 minutes, stop, read the log and reproduce on CPU
before pushing again (plan section 5.3). The report says which stage failed and why.

## What the spike script measures

`throughput.py run` runs each stage in a child process and rewrites the report after
every stage, so a crash, an out-of-memory kill or the timeout leaves the stages before it
in the report. Each GPU stack gets a venv of its own, deleted after its benches, because
their pins conflict (question 3). The GRPO stacks come before vLLM's, since ADR-0006
rests on their numbers, and each kind of stage is killed at its own cap
(`STAGE_CAPS_MINUTES` in the script, in the report as `stage_caps_minutes`: 5 minutes for
`env`, 2 for `secrets`, 15 for a download, 20 for an install, 25 for a GRPO bench and 15
for a vLLM one, guessed before any session), so a stage that hangs costs its cap and not
the rest of the 80 minutes. Each stage's `timeout_seconds` in the report is the smaller of
its cap and the time left. The stages, in order:

| Stage | Answers | Records |
|---|---|---|
| `env` | 1, 3 | Python, torch and CUDA versions, the GPUs and their compute capability, the image build (`/etc/git_commit`, `/etc/build_date`), internet, disk, preinstalled packages |
| `secrets` | 2 | Whether `HF_TOKEN` could be read in this pushed run; never its value |
| `download:<model>` | 4, 5 | Seconds and bytes to fetch Qwen3 1.7B and 4B, and the commit fetched; Qwen3-8B's just before vLLM's stack |
| `install:<stack>` | 3 | Seconds to install each stack, venv size, `uv pip freeze` of the venv's own packages, the installer used |
| `grpo:<stack>:<model>` | 5 | Seconds per step over 10 steps, peak memory, TRL's log, for 4-bit LoRA with 6 generations of up to 256 tokens; the GPUs the trainer saw (`n_gpu`), the completions TRL scored a step, the mean completion length |
| `vllm:<model>` | 4 | Load seconds, tokens per second on 100 prompts, greedy, fp16, up to 256 tokens a reply, device memory |
| `vllm:Qwen/Qwen3-8B:bnb` | 4 | Whether vLLM loads FG-307's reference in 4-bit with bitsandbytes (`quantization="bitsandbytes"` over fp16 weights, as FG-306's backend asks), then the same records on 10 prompts |
| `limit-probe` (own kernel) | 6 | A heartbeat rewritten every 30 s until the session is killed |

The prompts are 100 made-up text-to-SQL questions in Forge's p1 format over a six-table
schema, so they carry no Spider text; the report gives their token counts. The GRPO
reward is a cheap check for a closed sql block, so the timing is the trainer's and not
the SQL executor's. The GRPO batch is plan section 3.10's: 6 completions a device and 4
accumulation steps, so a step trains on 24 completions from 4 prompts.

A T4 session has two GPUs, and a Trainer that sees both in one process doubles its batch,
so every bench runs with `CUDA_VISIBLE_DEVICES=0`, as the jobs use the first GPU
(ADR-0005); the env stage still sees both. A GRPO bench records the GPUs its trainer
counted (`n_gpu`) and the completions TRL scored a step, counted by the reward, beside
the planned 24 (`planned_completions_per_step`). Each GRPO bench prompts with the same
text: the chats are rendered before training with Qwen3's thinking off, since Unsloth's
TRL 0.22 has no `chat_template_kwargs`, and a template that ignores the switch fails the
stage. The result says thinking was off and gives the mean completion length.

The script may run inside an IPython kernel rather than as `python throughput.py`:
then it has no `__file__`, its `argv` is the kernel's own (`ipykernel_launcher.py -f
<connection file>`), and its code may be a cell's with no file on disk. So the arguments
written into `KERNEL_ARGV` win over the interpreter's in the kernel's own process, while a
stage's child, which `run_child` marks with `FORGE_SPIKE_CHILD=1`, takes its command line.
Before it parses its arguments, every launch writes `spike-report.json` with `launch`
only (`argv0`, `argv`, `kernel_argv`, `file`, `cwd`), so a launch the arguments don't suit
still says how it happened. The children run this script's file, a file is taken for it
only if it holds the script's `REPORT_SCHEMA` line, and when there is none the cell's
source is written to the work folder and run from there (`launch.script`).

The stacks are candidates, each pinned in the script and none verified on a T4:

| Stack | Pins | Why |
|---|---|---|
| `vllm` | vllm 0.33.0 (brings torch 2.14.0), bitsandbytes 0.50.2 | Generation throughput for evaluation (FG-306), and the 4-bit reference (FG-307) |
| `trl` | Forge's lock: trl 1.18.0, transformers 5.23.0, peft 0.21.3, accelerate 1.15.0, datasets 5.2.0, huggingface-hub 2.4.0, plus bitsandbytes 0.50.2; on the image's torch (see below) | ADR-0006's training stack without vLLM, the fallback |
| `trl-vllm` | vllm 0.33.0, torch 2.14.0, transformers 5.20.2, then the lock's trl 1.18.0, peft 0.21.3, accelerate 1.15.0, datasets 5.2.0 and huggingface-hub 2.4.0, plus bitsandbytes 0.50.2 | The same list as `kaggle/requirements-gpu.txt` since FG-306: evaluation through vLLM, and TRL generating through colocated vLLM |
| `unsloth` | unsloth 2026.11.4, vllm 0.11.2, transformers 4.57.6, trl 0.22.2, datasets 4.3.0, bitsandbytes 0.50.2, torchao below 0.18 | Unsloth's own Kaggle T4 recipe, for comparison |

Each stack installs with uv into a venv of its own, except `trl`. Its venv is made with
system site packages so that it runs on the image's torch, but uv leaves those out when
it installs ("uv will _not_ take system site packages into account", `uv help venv`), so
it would download a PyPI torch for trl, peft and accelerate. That venv is seeded with pip
instead, which counts the image's packages as installed, and the stage fails if a torch
lands in the venv all the same.

## 1. Kernel metadata, the accelerator and private datasets

**What research says today.**

- `kernel-metadata.json` takes `id` (`owner/slug`, no version), `title` (at least 5
  characters, and it should slugify to the slug), `code_file`, `language` (only
  `python`), `kernel_type` (`script` or `notebook`), `is_private` (default true),
  `enable_gpu` (default false), `enable_internet`, `machine_shape`, and the lists
  `dataset_sources`, `kernel_sources`, `competition_sources` and `model_sources`. The
  code also reads `id_no`, `enable_tpu`, `docker_image`, `docker_image_pinning_type` and
  `keywords`. Sources: [kernels_metadata.md](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md),
  [kaggle_api_extended.py](https://raw.githubusercontent.com/Kaggle/kaggle-cli/main/src/kaggle/api/kaggle_api_extended.py).
- The docs say `enable_internet` defaults to false, but the code treats a missing key as
  true. Forge always sets it.
- There is no `accelerator` field. The accelerator is `machine_shape` or the push flag
  `--accelerator`, which overrides it, so it can be chosen from the CLI. `NvidiaTeslaT4`
  is the default GPU (a T4 x2 session). `NvidiaTeslaP100` is retired: the CLI accepts it
  with a warning and the server runs a T4. Sources: [kernels.md](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md),
  [Google Cloud deprecations](https://docs.cloud.google.com/compute/docs/deprecations).
- `kaggle kernels push -p DIR [-t SECONDS] [--accelerator ID] [--no-run]`;
  `kaggle kernels status` reports a `KernelWorkerStatus` of `QUEUED`, `RUNNING`,
  `COMPLETE`, `ERROR`, `CANCEL_REQUESTED`, `CANCEL_ACKNOWLEDGED` or `NEW_SCRIPT`;
  `kaggle kernels output -p DIR -o` writes the log as `<slug>.log`; `kaggle quota -v`
  shows GPU hours used, left and when they refresh. The CLI is kaggle 2.2.5 and needs
  Python 3.11 or later. Sources: [kernels.md](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md),
  [quota.md](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/quota.md),
  [PyPI kaggle](https://pypi.org/project/kaggle/).
- Datasets are private unless created with `-u`, and need `-r zip` (or `tar`) to upload
  folders; the default skips subfolders. `kaggle datasets metadata --update` without
  `"isPrivate": true` can make a dataset public. Source: [datasets.md](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md).
- Mounted inputs are under `/kaggle/input`, and the layout has changed, so jobs glob for
  their files instead of hard-coding paths.

`forge.ops.kaggle.check_kernel_metadata` turns each of those traps into an error before a
push: an unknown key, a missing `enable_internet`, a title that doesn't slugify to the
slug, a P100, a public kernel.

**To be measured on the first GPU session.**

- Does `--accelerator NvidiaTeslaT4` on a kernel whose metadata says the same give a T4
  session? (`env.nvidia_smi`, `env.torch.devices`.) Not measured yet.
- How many GPUs does the session show, and how much memory each? Not measured yet.
- How does Kaggle start a script kernel: what are `launch.argv0`, `launch.argv`,
  `launch.file` and `launch.cwd` in the report, and which file did the children run
  (`launch.script`)? FG-303's entry script depends on it. Not measured yet.

## 2. Kaggle Secrets in pushed runs

**What research says today.**

- A kernel reads a secret with `kaggle_secrets.UserSecretsClient().get_secret(label)`,
  and only secrets attached to that kernel. Attaching is only possible in the web editor
  (Add-ons, Secrets); the CLI and the API have no field for it, issue 582 asking for one
  is open, and Kaggle staff said they don't plan to add it. Sources:
  [kernels.md, using secrets](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md#using-secrets-in-kernels),
  [kaggle-cli issue 582](https://github.com/Kaggle/kaggle-cli/issues/582),
  [kaggle_secrets.py](https://github.com/Kaggle/docker-python/blob/main/patches/kaggle_secrets.py).
- Users report that a binding attached in the editor does not survive a later
  `kaggle kernels push`, so `get_secret` fails in the pushed version. This is reported,
  not confirmed by Kaggle.
- The client raises `CredentialError`, `ConnectionError` or `BackendError` when it can't
  read a secret. `KAGGLE_KERNEL_RUN_TYPE` is `Interactive` or `Batch`.

ADR-0005 plans for "no": results come back through `kaggle kernels output`, and a token
in a private dataset is the last resort, with its risk written down.

**To be measured on the first GPU session.**

- With `HF_TOKEN` attached in the editor after a `--no-run` push, can the next CLI push
  read it? (`secrets.labels.HF_TOKEN.readable`, and the error type if not.) Not measured
  yet.
- What is `KAGGLE_KERNEL_RUN_TYPE` in a pushed run? Not measured yet.

## 3. Python, CUDA and install times

**What research says today.**

- Kaggle's newest GPU image, v173 (release tag 2026-10-30), has Python 3.13, CUDA 13.0,
  torch 2.11.0+cu130, transformers 5.20.2, peft 0.21.2, accelerate 1.15.0 and
  huggingface_hub 1.33.0. The image before it (v172) had CUDA 12.8. Whether trl, vllm
  and bitsandbytes come preinstalled is unknown. Sources:
  [Dockerfile.tmpl](https://github.com/Kaggle/docker-python/blob/main/Dockerfile.tmpl),
  [docker-python releases](https://github.com/Kaggle/docker-python/releases).
- Forge's `pyproject.toml` required Python below 3.13, so on that image its wheel would
  not have installed. FG-303 widened it to below 3.14 and added 3.13 to CI (ADR-0002).
- The T4 is compute capability 7.5: fp16 only, no bf16 in hardware.
- The stacks conflict. vllm 0.33.0 pins torch 2.14.0 and transformers below 5.21;
  Unsloth 2026.11.4 caps trl at 1.15.0 and transformers at 5.17.0; Unsloth's T4 notebook
  pins vllm 0.11.2, which brings torch 2.9.0. No single environment holds Forge's lock,
  vLLM and Unsloth. Sources: [PyPI vllm 0.33.0](https://pypi.org/pypi/vllm/0.33.0/json),
  [PyPI unsloth 2026.11.4](https://pypi.org/pypi/unsloth/2026.11.4/json),
  [Kaggle-Qwen3_(4B)-GRPO.ipynb](https://github.com/unslothai/notebooks/blob/main/nb/Kaggle-Qwen3_(4B)-GRPO.ipynb).

**To be measured on the first GPU session.**

- Python version (`env.python`). Not measured yet.
- CUDA and torch versions, cuDNN, driver (`env.torch`, `env.nvidia_smi`), and the image
  build (`env.image`). Not measured yet.
- Install seconds and venv size for each stack (`install:<stack>`), and whether each
  installs at all. Not measured yet.
- Should GPU wheels be uploaded as a dataset to save install time? Decide from the
  install seconds against a session's length. Not decided yet.

## 4. vLLM in fp16 on the T4

**What research says today.**

- vLLM 0.33.0 supports compute capability 7.5 and lists the T4. On a T4 it falls back to
  its Triton attention backend (FlashAttention 2 needs 8.0), and it refuses bf16, so
  `dtype="half"`. A vLLM bug report from 2026-11-12 reproduces on 0.33.0 with a T4 and
  `dtype='half'`, so the release does run there. Sources:
  [vLLM GPU install](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html),
  [vllm/platforms/cuda.py](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/platforms/cuda.py),
  [vLLM issue 67147](https://github.com/vllm-project/vllm/issues/67147).
- No public benchmark gives Qwen3 1.7B or 4B throughput on one T4. The nearest reported
  figures are for Qwen2.5 0.5B and 1.5B on older vLLM, and are not a substitute for a
  measurement. Source: [vllm-serving-benchmark-t4](https://github.com/lyftium/vllm-serving-benchmark-t4).

**To be measured on the first GPU session.**

- Does vLLM 0.33.0 load and generate in fp16 on the T4? (`vllm:<model>` status.) Not
  measured yet.
- Tokens per second for Qwen3 1.7B on 100 prompts (`completion_tokens_per_second`). Not
  measured yet.
- Tokens per second for Qwen3 4B on 100 prompts. Not measured yet.
- Load seconds, device memory in use, and how many replies hit the 256-token cap. Not
  measured yet.
- Does vLLM 0.33.0 load Qwen3-8B in 4-bit with bitsandbytes and answer on the T4
  (`vllm:Qwen/Qwen3-8B:bnb` status, `load_seconds`, `memory`, `samples`)? FG-306 allows
  4-bit loading and FG-307 runs its reference only if it does. Not measured yet.

## 5. GRPO on the T4 with 4-bit models

**What research says today.**

- TRL 1.18's GRPO defaults to bf16, which crashes on a T4 since TRL 1.15 (issue 7889):
  set `fp16=True`. With a 4-bit (QLoRA) model, TRL casts the adapters to bf16, which
  crashes fp16 training (issue 7776); casting the trainable parameters back to fp32 after
  building the trainer avoids it. `max_prompt_length` is gone from `GRPOConfig` since
  TRL 0.28, so prompts are fitted when the data is built. Sources:
  [TRL issue 7889](https://github.com/huggingface/trl/issues/7889),
  [TRL issue 7776](https://github.com/huggingface/trl/issues/7776),
  [grpo_config.py](https://raw.githubusercontent.com/huggingface/trl/v1.18.0/trl/trainer/grpo_config.py).
- Unsloth's official T4 notebook runs Qwen3 4B GRPO with fast inference on the old stack
  of question 3. Unsloth reports Qwen3 4B on one T4 at about a step a minute with 2
  generations; that is a different batch from Forge's. Sources:
  [memory-efficient RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl),
  [Kaggle-Qwen3_(4B)-GRPO.ipynb](https://github.com/unslothai/notebooks/blob/main/nb/Kaggle-Qwen3_(4B)-GRPO.ipynb).
- Unsloth's standby mode overrides the vLLM memory share it is given, and conflicts with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Source:
  [unsloth_zoo vllm_utils.py](https://raw.githubusercontent.com/unslothai/unsloth-zoo/main/unsloth_zoo/vllm_utils.py).

The spike runs TRL's path first, since ADR-0006 makes it the default; the CPU path of the
same code runs in CI (`tests/integration/test_spike_grpo.py`).

**To be measured on the first GPU session.**

For Qwen3 1.7B and 4B, 4-bit, 6 generations, up to 256 completion tokens, 10 steps:

- TRL with transformers generation (`grpo:trl:<model>`): seconds per step (median after
  the first) and peak memory. Not measured yet.
- TRL with colocated vLLM (`grpo:trl-vllm:<model>`): whether it runs, then the same
  numbers. Not measured yet.
- Unsloth with fast inference (`grpo:unsloth:<model>`): whether it runs, then the same
  numbers. Not measured yet.
- Does fp16 training stay finite over the 10 steps (`log_history`)? Not measured yet.
- Did each bench train on one GPU and 24 completions a step (`grpo:<stack>:<model>`,
  `n_gpu` and `completions_per_step`), and how long were the replies with thinking off
  (`mean_completion_length`)? Not measured yet.

## 6. The 12-hour limit

**What research says today.**

- Sessions end after 12 hours (9 for TPUs); the CLI's own wait timeout is 12 hours. The
  weekly GPU quota is reported at about 30 hours and floats; `kaggle quota -v` gives the
  live value. Sources: [kaggle_api_extended.py](https://raw.githubusercontent.com/Kaggle/kaggle-cli/main/src/kaggle/api/kaggle_api_extended.py),
  [quota.md](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/quota.md).
- Only `/kaggle/working` is kept as a version's output, reportedly up to about 20 GB.
  Whether a session killed at its limit keeps that output is not documented, and how
  long outputs are kept is unknown. Treat Kaggle as a place to run jobs, not to store
  their results.

The plan already assumes "no": checkpoints go to the run store, and the time guard stops a
job at 11 h 15 min (FG-305).

**To be measured on the first GPU session.**

- When a session is killed at its timeout (`-t 900` on the CPU probe), is the output
  written before the kill kept? (`limit-probe.json` comes back, and its `last_beat_at`
  against the start plus 900 s.) Not measured yet.
- What does `kaggle kernels status` report for a session killed at its timeout? Not
  measured yet.
- The probe tests a timeout the push sets, which is the same cut as the 12-hour limit only
  if Kaggle ends both the same way. That is an assumption to confirm, not a result.

## Budget

The section 3.8 table keeps its estimates until this spike's numbers exist. At Gate 3 the
gatekeeper redoes the GRPO and evaluation rows from `grpo:*` and `vllm:*`, and the throughput
spike row from the ledger.
