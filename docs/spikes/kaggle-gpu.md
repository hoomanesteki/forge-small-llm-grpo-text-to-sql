# Spike FG-301: Kaggle, Unsloth and vLLM on a T4

Measured on 2026-11-16 (UTC), in one GPU session and two CPU sessions on Kaggle. This page
keeps two things apart: what research said before them, with sources, and what they
measured, each number with the report and field it comes from. The reports are committed
in [`kaggle-gpu/`](kaggle-gpu/), whose README names the kernel version and commit of each
and how it was fetched.

- Dates: research 2026-11-15, measurements 2026-11-16. Timebox: 1 day. GPU budget: 1.5
  hours (plan section 3.8); the GPU session spent 0.59 hours (`kaggle quota -v`, and its
  row in `docs/compute-ledger.md`).
- Built: the kernel metadata template and its checks (`kaggle/kernel-metadata.template.json`,
  `forge.ops.kaggle`), the spike script `kaggle/spike/throughput.py`, the bundler
  `scripts/kaggle_spike.py`, the GPU pins in `kaggle/requirements-gpu.txt`, and the secret
  probe `kaggle/spike/secret_probe.py` (question 2).
- Decided: [ADR-0005](../adr/0005-kaggle-compute.md) (accepted from research, with the
  measured answers to questions 2 and 6) and [ADR-0006](../adr/0006-train-backend.md)
  (accepted on question 5's table: TRL with transformers generation).
- The section 3.8 budget is redone from these numbers (see Budget).

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
   enough: the stage only checks that it can be read, and never records its value. The
   binding did not reach the version step 4 pushed (question 2).
4. Push the GPU kernel through the job runner, which checks the smoke marker, the clean
   tree and the week's budget first (FG-303, `docs/runbook.md` "Kaggle jobs"):
   `uv run forge kaggle run --job spike --config configs/spike/throughput.yaml`.
   It pushes a new version of the same `<owner>/forge-throughput-spike` kernel, on a T4,
   and prints the run id. Its timeout is the 1.5-hour estimate and half again, but the
   script starts no stage after 80 minutes and bounds each stage by its own cap and the
   time left, so the session ends long before the cut. The last 20 of those minutes are
   kept for the 4-bit reference (see below).
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
8. Commit both JSON reports under `docs/spikes/kaggle-gpu/`, answer each question below
   from them under its "Measured on" heading, and update ADR-0006 and the section 3.8
   budget. FG-301 did this for the 2026-11-16 session.

If the GPU kernel fails in its first 10 minutes, stop, read the log and reproduce on CPU
before pushing again (plan section 5.3). The report says which stage failed and why.

The 4-bit question is answered, so no second push is needed: `vllm:Qwen/Qwen3-8B:bnb`
failed at once, since vLLM 0.33.0 has no bitsandbytes (question 4). FG-398 moved FG-307's
reference to an AWQ checkpoint, which the reference's val run of 2026-11-17 was the first
to load on a T4, and it did (ADR-0012).

## What the spike script measures

`throughput.py run` runs each stage in a child process and rewrites the report after
every stage, so a crash, an out-of-memory kill or the timeout leaves the stages before it
in the report. Each GPU stack gets a venv of its own, deleted after its benches, because
their pins conflict (question 3). The GRPO stacks come before vLLM's, since ADR-0006
rests on their numbers, and each kind of stage is killed at its own cap
(`STAGE_CAPS_MINUTES` in the script, in the report as `stage_caps_minutes`: 5 minutes for
`env`, 2 for `secrets`, 15 for a download, 20 for an install, 25 for a GRPO bench, 15 for
a vLLM one and 10 for the 4-bit load, guessed before any session), so a stage that hangs
costs its cap and not the rest of the 80 minutes. FG-306 and FG-307 wait on the 4-bit
answer, and the GRPO benches could take every minute before it, so the last 20 minutes
(`bnb_reserve_minutes`) are kept for the reference's download, vLLM's install and the
4-bit load, which runs right after that install: no stage before them may start or run in
those minutes, and the fp16 benches after the load take what is left. Each stage's
`timeout_seconds` in the report is the smaller of its cap and the time it may use. The
stages, in order:

| Stage | Answers | Records |
|---|---|---|
| `env` | 1, 3 | Python, torch and CUDA versions, the GPUs and their compute capability, the image build (`/etc/git_commit`, `/etc/build_date`), internet, disk, preinstalled packages |
| `secrets` | 2 | Whether `HF_TOKEN` could be read in this pushed run; never its value |
| `download:<model>` | 4, 5 | Seconds and bytes to fetch Qwen3 1.7B and 4B, and the commit fetched; Qwen3-8B's just before vLLM's stack |
| `install:<stack>` | 3 | Seconds to install each stack, venv size, `uv pip freeze` of the venv's own packages, the installer used |
| `grpo:<stack>:<model>` | 5 | Seconds per step over 10 steps, peak memory, TRL's log, for 4-bit LoRA with 6 generations of up to 256 tokens; the GPUs the trainer saw (`n_gpu`), the completions TRL scored a step, the mean completion length |
| `vllm:Qwen/Qwen3-8B:bnb` | 4 | Whether vLLM loads FG-307's reference in 4-bit with bitsandbytes (`quantization="bitsandbytes"` over fp16 weights, as FG-306's backend asks), then the records of `vllm:<model>` on 10 prompts |
| `vllm:<model>` | 4 | Load seconds, tokens per second on 100 prompts, greedy, fp16, up to 256 tokens a reply, device memory |
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

The stacks, each pinned in the script, were candidates until this session; question 5 says
how each did:

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

**Measured on 2026-11-16.**

From the GPU spike's report (`spike-report.json`) unless another file is named:

- `--accelerator NvidiaTeslaT4`, on metadata whose `machine_shape` says the same, gave a T4
  x2 session: `env.nvidia_smi` lists two `Tesla T4, 580.178.04, 15360 MiB, 7.5`, and
  `env.torch.devices` two Tesla T4s of compute capability 7.5 with 15,636,037,632 bytes
  (15.6 GB) each.
- Every bench used one of the two, as planned: each `grpo:*` and `vllm:*` stage ran with
  `CUDA_VISIBLE_DEVICES=0` (its `env`), and each GRPO trainer counted one GPU (`n_gpu` is 1
  in all four `grpo:*` results that ran).
- Kaggle starts a script kernel as a file, not in IPython, and with no arguments:
  `launch.argv0`, `launch.file` and `launch.script` are `/kaggle/src/script.py`,
  `launch.argv` is that path alone, and `launch.cwd` is `/kaggle/working`. The arguments
  came from the script's baked ones (`launch.kernel_argv`, `["run"]`), and the children
  ran the same file. The CPU probe started the same way (`limit-spike-report.json`).
- Private datasets mount at `/kaggle/input/datasets/<owner>/<slug>`: the 50-question eval
  job found the code and the data there, the data with its 170 files
  (`stages.code.result.folder` and `stages.data.result` in its `status.json`, in
  `docs/runs/2026-11-16-smoke-kaggle-50-val-c8f4b41/`). Jobs find their files by glob, so
  the layout needed no change.
- Kaggle refuses a kernel title over 50 characters, which the research didn't turn up.
  `kaggle kernels push --no-run` of throwaway kernels (deleted afterwards) saved a
  50-character title and got `400 Client Error: Bad Request` from `SaveKernel` at 51, and
  nothing else in the metadata mattered. The first push of the 50-question job failed the
  same way: its default run id made the 57-character title `Forge
  2026-11-16-qwen2-5-coder-1-5b-instruct-eval-c8f4b41`. FG-398 checks the limit before a
  push and gives a long run id a kernel name that fits.

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

**Measured on 2026-11-16.**

- No. `HF_TOKEN` was attached to `hoomanesteki/forge-throughput-spike` in the editor
  after the `--no-run` push (step 3), and version 2, pushed from the CLI, couldn't read it:
  `secrets.labels.HF_TOKEN` is `{"readable": false, "error": "ConnectionError"}`. The
  50-question eval job got the same `ConnectionError` for `HF_TOKEN` and `WANDB_API_KEY`
  (`stages.secrets` in its `status.json`).
- Why, from the secret probe (`secret-probe.json`, version 3 of the same kernel, so an
  editor binding that had survived would apply): Kaggle sets `KAGGLE_USER_SECRETS_TOKEN`
  in the pushed run (`has_user_secrets_token` is true), and the secrets service answers
  `HTTP Error 400: Bad Request` for `HF_TOKEN` exactly as for `FORGE_NO_SUCH_LABEL`, a
  label no secret has (`HF_TOKEN.cause`, `FORGE_NO_SUCH_LABEL.cause`). So the binding
  attached in the editor did not survive the CLI push, as users reported. `kaggle_secrets`
  wraps the 400 in the builtin `ConnectionError` ("Connection error trying to communicate
  with service."), so here that error is the service refusing the label, not the network.
- `KAGGLE_KERNEL_RUN_TYPE` is `Batch` in a pushed run (`secrets.run_type`,
  `env.kaggle_run_type`, and the probe's `run_type`).

So ADR-0005's second way holds: no token on Kaggle. Results come back with `forge kaggle
output` (or `forge runs pull --store kaggle`) and go to the Hub from the laptop with `forge
runs push`, first done for the 50-question run (`docs/runbook.md`, Run store).

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

**Measured on 2026-11-16.**

- Python 3.13.16 (`env.python`), as the research said for v173.
- torch 2.11.0+cu130 with CUDA 13.0, cuDNN 92700 as torch reports it, no bf16
  (`env.torch`), and driver 580.178.04 (`env.nvidia_smi`). The image was built
  `20261030-181552` from commit `6a20e87c` (`env.image`), the day of v173's release tag.
- Preinstalled (`env.packages`): transformers 5.20.2, peft 0.21.2, accelerate 1.15.0,
  datasets 4.8.5, huggingface-hub 1.33.0 and kaggle 2.0.2. trl, bitsandbytes, vllm and
  unsloth are not, which the research couldn't tell.
- The machine has 4 CPUs and 33.7 GB of memory (`env.cpu_count`, `env.memory_bytes`), with
  20.9 GB free in `/kaggle/working` and 1.04 TB in `/kaggle/temp` (`env.disk_free_bytes`).
- Every stack installed, each into a venv of its own (`install:<stack>.result`:
  `install_seconds`, `venv_bytes`):

  | Stack | Installer | Seconds | Venv |
  |---|---|---:|---:|
  | `trl` | pip, over the image's torch | 20.0 | 0.47 GB |
  | `unsloth` | uv | 124.4 | 10.87 GB |
  | `trl-vllm` | uv | 95.9 | 9.67 GB |
  | `vllm` | uv, from uv's cache just after `trl-vllm` | 1.5 | 9.44 GB |

- In a kernel job the install is slower. `kaggle/entry.py` runs `pip install` for the
  wheel and then `requirements-gpu.txt`, the `trl-vllm` list, over the image, and in the
  50-question eval job that took 391.6 s (`stages.install.seconds` in its `status.json`),
  four times uv's 95.9 s for the same list. With the code (28.9 s), data (48.0 s), secrets
  (0.2 s) and preflight (168.1 s, which downloads and loads the model) stages, the job
  spent 636.8 s, about 11 minutes, before its job stage started.
- GPU wheels as a dataset: not decided. On these numbers, installing with uv instead of pip
  would save about 5 minutes a job, and prebuilt wheels might save more, but the venvs
  these lists make are 9 to 11 GB, a dataset to keep in step with every pin change.
  FG-398 or a later story may switch the kernel's install to uv or to wheels in a dataset;
  until one measures it, kernels install with pip, and FG-302's `publish-wheels` isn't
  built.

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

**Measured on 2026-11-16.**

- Yes. vLLM 0.33.0, with torch 2.14.0 and transformers 5.20.2 (`vllm:<model>.result.versions`),
  loaded Qwen3 1.7B and 4B in fp16 (`settings.dtype` `half`) and generated greedily on one
  T4 with the Triton attention backend, as the research expected. The log says `Using
  TRITON_ATTN attention backend out of potential backends: ['TRITON_ATTN',
  'FLEX_ATTENTION'].` For 100 prompts of up to 256 tokens a reply (`vllm:<model>.result`):

  | | Qwen3 1.7B | Qwen3 4B |
  |---|---:|---:|
  | Load (`load_seconds`) | 112.4 s | 126.7 s |
  | Generation (`generate_seconds`) | 9.9 s | 11.8 s |
  | Completion tokens (`completion_tokens`) | 5,728 | 4,601 |
  | Tokens a second (`completion_tokens_per_second`) | 577.3 | 391.4 |
  | Replies cut at 256 tokens (`cut_at_max_tokens`) | 0 | 0 |
  | Device memory in use (`memory.device_used_bytes`) | 13.1 GB | 13.2 GB |

  The 100 prompts hold 25,921 tokens (`prompt_tokens`), 259 a prompt. vLLM takes 85% of
  the device (`settings.gpu_memory_utilization`), so the memory in use is its share, not
  the model's need; torch's own counters read 0 (`memory.peak_allocated_bytes`), since the
  engine runs in a process of its own.
- 4-bit with bitsandbytes: no. `vllm:Qwen/Qwen3-8B:bnb` failed in 26.3 s, before any
  weight loaded, with `ValidationError: 1 validation error for ModelConfig` and `Value
  error, Unknown quantization method: bitsandbytes. Must be one of ['awq', 'auto_awq',
  'fp8', ...]`: vLLM 0.33.0 has no bitsandbytes quantization, though bitsandbytes 0.50.2
  was in its venv. The research behind FG-306 and ADR-0012 counted bitsandbytes among
  the quantizations vLLM runs on Turing. That holds for the vLLM 0.11.2 in Unsloth's
  stack, which loaded 4-bit weights in the same session (`'load_format': 'bitsandbytes'`
  in its log), and not for 0.33.0. The 8B's weights were downloaded (16.4 GB in 80.2 s,
  `download:Qwen/Qwen3-8B`) and never loaded. AWQ is on 0.33.0's list, and FG-398 moved
  FG-307's reference to the AWQ checkpoint `Qwen/Qwen3-8B-AWQ`, which the reference's runs
  of 2026-11-17 loaded and scored on a T4 (ADR-0012).

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

**Measured on 2026-11-16.**

Qwen3 1.7B and 4B in 4-bit, LoRA rank 16, fp16 (`settings.fp16` true, `bf16` false), 6
generations of up to 256 tokens, 6 completions a device and 4 accumulation steps, 10 steps
(`grpo:<stack>:<model>.result`):

| Stage | Median s a step | Peak allocated | Peak reserved | Load s | Install s |
|---|---:|---:|---:|---:|---:|
| `grpo:trl:Qwen/Qwen3-1.7B` | 18.43 | 3.02 GB | 4.02 GB | 5.3 | 20.0 |
| `grpo:trl:Qwen/Qwen3-4B` | 32.92 | 4.94 GB | 6.42 GB | 9.3 | 20.0 |
| `grpo:unsloth:Qwen/Qwen3-1.7B` | 11.64 | 12.75 GB | 13.25 GB | 139.4 | 124.4 |
| `grpo:unsloth:Qwen/Qwen3-4B` | 21.22 | 13.66 GB | 14.15 GB | 129.5 | 124.4 |
| `grpo:trl-vllm:Qwen/Qwen3-1.7B` | error | | | | 95.9 |
| `grpo:trl-vllm:Qwen/Qwen3-4B` | error | | | | 95.9 |

The median is of steps 2 to 10 (`median_step_seconds`), the peaks are of the device's
15.64 GB (`memory.peak_allocated_bytes`, `memory.peak_reserved_bytes`), the load is the
model's (`load_seconds`), and the install is the stack's (`install:<stack>`).

- TRL with transformers generation ran both models in 4-bit with fp16 and no out-of-memory,
  the 4B at 4.94 GB allocated at most, on the lock's versions (`versions`: trl 1.18.0,
  transformers 5.23.0, peft 0.21.3, on the image's torch 2.11.0+cu130).
- fp16 stayed finite: every loss and grad norm the four benches that ran logged is finite
  (`log_history`). Many are exactly 0: on a step where each prompt's 6 completions got the
  same reward (`frac_reward_zero_std` 1.0) GRPO has nothing to learn from, and 9 of the 4B
  TRL bench's 10 steps were such steps. The reward is a cheap check for a closed sql block,
  so these benches time the trainer; they say nothing about learning, nor about stability
  over hundreds of steps.
- Each bench trained on one GPU (`n_gpu` 1) and 24 completions a step
  (`completions_per_step` 24.0, the `planned_completions_per_step`), with thinking off
  (`thinking_off`) and short replies: 55.6 and 47.6 tokens on average for TRL's 1.7B and
  4B, 51.8 and 45.0 for Unsloth's (`mean_completion_length`).
- Unsloth with fast inference ran both, on its own stack (`versions`: trl 0.22.2,
  transformers 4.57.6, vLLM 0.11.2, torch 2.9.0), 37% and 36% faster a step than TRL,
  after about 130 s of loading its colocated vLLM engine. Its peaks are that engine's
  reservation, not the training's need: standby mode set vLLM's share to 0.76 (`Unsloth:
  Standby mode is enabled. Changing gpu_memory_utilization to 0.76.` in the log), as the
  research said it would. The research's report of about a step a minute for the 4B was
  on another batch.
- TRL with colocated vLLM failed on both models before training, at vLLM's load of the
  4-bit policy, with question 4's error (`grpo:trl-vllm:<model>.error`): vLLM 0.33.0
  can't take the `bitsandbytes` quantization of the policy TRL hands it.

So ADR-0006 is accepted: GRPO and SFT on the T4 use TRL with transformers generation, the
`trl` stack, and the plan's fallback to the 1.7B model alone isn't needed.

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

**Measured on 2026-11-16.**

- A session killed at its timeout keeps what it wrote before the kill. The CPU probe,
  pushed with `-t 900`, started at 15:56:12 UTC and wrote its last beat at 16:11:12, 900 s
  later (`limit-probe.json`: `started_at`, `last_beat_at`, and `beats` 31 at
  `interval_seconds` 30), and that file came back with `kaggle kernels output`. Its log
  ends at beat 31, 900.8 s in.
- `kaggle kernels status` reports such a session as `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`,
  which `forge kaggle wait` already reads as an end with no job outcome.
- The probe measured a timeout the push set, not the 12-hour limit itself; that Kaggle ends
  both the same way is still an assumption. The time guard (11 h 15 min) stays either way.
- At most two GPU sessions run at once: a third GPU kernel pushed while two ran was refused
  on 2026-11-16 with `Kernel push error: Maximum batch GPU session count of 2 reached.`
  (FG-307's baselines).
- The weekly quota is 30 hours. `kaggle quota -v` printed
  `GPU,0.59h,29.41h,30.00h,2026-11-21T00:00:00+00:00` after the spike (used, left, total
  and the next refresh), with 0.00 h used before it, so it refreshes at 00:00 UTC on a
  Saturday, as one source in the research said. The 50-question job's push saw the same
  29.41 h left (`kaggle_gpu_hours_left` in its `kernels.json`).

## Budget

Plan section 3.8's GPU hour budget is redone from these numbers. The throughput spike's
row is its ledger row, 0.59 h. GRPO rows use the `trl` stack's median step for the model's
size (`grpo:trl:<model>`), and evaluation rows a kernel eval job's measured setup (the
50-question job's `status.json`, the model's download and vLLM's load) and vLLM's tokens a
second (`vllm:<model>`). The note under the table gives each formula. The total came to
39.34 h on 2026-11-16, with SFT, the ablations and the reserve kept as estimates. FG-307
then worked the rows out again for the model ADR-0012 chose, and the ablations by the same
formulas, for 40.79 h, within the 45 h the Phase 3 gate allows.
