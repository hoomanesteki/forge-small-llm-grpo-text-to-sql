# Runbook

Operational notes for running Forge. Sections are filled in by the stories that need them.

## Accounts

Everything is free. Set these up once, then run `uv run forge doctor` until every line says `ok`.

1. **Kaggle (GPU runner).**
   - Create an account and **phone-verify it** under Settings. Without that, notebooks get
     no GPU and no internet.
   - Under Settings, API, create an API token. Save it to `~/.kaggle/access_token` and run
     `chmod 600 ~/.kaggle/access_token`. A legacy `~/.kaggle/kaggle.json` also works.
   - An API token doesn't name its user, so also put `FORGE_KAGGLE_USER=<your username>`
     in `.env`. Forge's Kaggle datasets are created under that name.
2. **Hugging Face (run store and published models).**
   - Create an account and a write token at huggingface.co/settings/tokens.
   - Put it in `.env` at the repo root as `HF_TOKEN=...`. `.env` is git-ignored.
   - Create the **private** model repo `hoomanesteki/forge-runs`. Override the name with
     `FORGE_RUN_STORE` if needed.
3. **Kaggle Secrets.** Not needed for kernel jobs. The CLI cannot attach a secret to a
   kernel, and FG-301 found that one attached in the web editor doesn't reach the next
   version pushed from the CLI: the secrets service refuses it as it refuses a label that
   doesn't exist (`docs/spikes/kaggle-gpu.md`, question 2). So kernel jobs run without
   `HF_TOKEN` or `WANDB_API_KEY`, their results come back through `forge kaggle output`,
   and the laptop puts them in the run store (Run store).
4. **Weights & Biases (optional).** A free account; set `WANDB_API_KEY` in `.env` for live
   charts on the laptop (a kernel can't read it, step 3). Without it, metrics go to JSONL in
   the run store.

`forge doctor` also wants 10 GB of free disk for Spider, model weights and caches. Lower the
bar with `FORGE_MIN_FREE_GB` if you know you need less. It never prints token values: any
environment variable whose name contains TOKEN, KEY or SECRET is masked in its output.

## Code and data on Kaggle

A GPU job runs the code and reads the data from two private Kaggle datasets, both owned by
`FORGE_KAGGLE_USER` (or the username in `kaggle.json`):

1. `uv run forge kaggle publish-data`, once. It checks every file `spider.sha256` lists in
   `data/spider` (or `FORGE_DATA_DIR`), copies them and that file to `build/kaggle-data/`,
   and creates `<owner>/forge-spider` with Spider's CC BY-SA 4.0 attribution and citation in
   its description. The copy is about 511 MB, so it is removed after the upload, a failed
   one too (`--keep-build` keeps it). Once `kaggle datasets status` says the dataset is ready, it
   uploads nothing; while Kaggle is still processing it, it says so and exits 1, so run it
   again later; and when the last version failed or was deleted, it uploads the files again
   as a new version.
2. `uv run forge kaggle publish-code`, on a clean tree, for each commit a job runs. It builds
   the wheel into `build/kaggle-code/` with both GPU package lists,
   `kaggle/requirements-gpu.txt` and `kaggle/requirements-train-gpu.txt` (see GPU stack), and
   `GIT_SHA`, then uploads it as a new version of `<owner>/forge-code` noted with the SHA,
   creating the dataset the first time.

Both datasets are private: Forge never passes `-u`, and each `dataset-metadata.json` says
`"isPrivate": true`, so `kaggle datasets metadata --update` keeps them private too.

## GPU stack

A kernel installs one of two package lists over Kaggle's image, after the wheel, chosen by
its job (ADR-0006): an eval job `kaggle/requirements-gpu.txt`, vLLM's stack, and a training
job (`--job sft` or `grpo`) `kaggle/requirements-train-gpu.txt`, the `trl` stack, which has
no vLLM. The install stage's result in `status.json` names the file it installed.

The eval list is vLLM's: vLLM decides most of it (FG-306). FG-301 ran it on a T4: vLLM
0.33.0 generated in fp16 on these versions, and only GRPO through it failed, at vLLM's load
of the 4-bit policy, since vLLM 0.33.0 has no bitsandbytes. pip took 392 s over the image in
the 50-question eval job, where uv takes 96 s for the same list into an empty venv
(`docs/spikes/kaggle-gpu.md`, question 3).

| Package | Version | Why this one |
|---|---|---|
| vllm | 0.33.0 | The newest release, and the newest TRL 1.18 accepts (`<0.34`) |
| torch | 2.14.0 | vllm 0.33.0 pins it; its wheel is built for CUDA 13.0, the image's CUDA. It replaces the image's 2.11.0 |
| transformers | 5.20.2 | The newest below 5.21, which vllm 0.33.0 needs; CI runs 5.23.0 |
| trl | 1.18.0 | uv.lock's; it takes transformers 4.56.2 or later |
| peft | 0.21.3 | uv.lock's |

accelerate 1.15.0, datasets 5.2.0, huggingface-hub 2.4.0 and bitsandbytes 0.50.2 complete
the list, all uv.lock's versions but bitsandbytes, which the lock doesn't have. The
transformers calls Forge makes in a kernel (`apply_chat_template` and `get_chat_template`
for prompts, the trainer callback classes for the time guard) were checked against the
5.20.2 source and do the same for Forge's arguments. A test keeps the file the same as the
spike's `trl-vllm` stack.

The training list is the spike's `trl` stack, on which FG-301 ran GRPO in 4-bit with fp16
for Qwen3 1.7B and 4B (question 5): uv.lock's trl 1.18.0, transformers 5.23.0, peft 0.21.3,
accelerate 1.15.0, datasets 5.2.0 and huggingface-hub 2.4.0, the versions CI tests, and
bitsandbytes 0.50.2. It pins no torch, so training runs on the image's torch
2.11.0+cu130, as the spike's did, and no vLLM. pip installed it in 20.0 s into a venv over
the image's packages (`install:trl`). A test keeps the file the same as that stack.

## Kaggle jobs

Every GPU job goes through `forge kaggle run`, on a clean commit that passed `make smoke`:

```bash
uv run forge kaggle run --job eval --config configs/eval/<name>.yaml   # prints the run id
uv run forge kaggle wait <run_id>      # until COMPLETE, ERROR or CANCEL_*, then the ledger row
uv run forge kaggle output <run_id>    # the kernel's output into runs/<run_id>/
```

Once the job has ended, `forge kaggle wait` adds its kernel session's row to
`docs/compute-ledger.md`, and `forge kaggle output` does too if `wait` didn't, each once a
session, as `forge runs pull` does (see Run store). A kernel that ends ERROR or CANCEL_*
has no job outcome to read, so `wait` says to fetch what it kept with `output`.

`forge kaggle run` runs these checks in order and stops at the first that fails. Steps 1
and 2 upload nothing. Step 3 may publish a new version of `<owner>/forge-code`, so a job
the budget refuses at step 4, or one declined at step 5, leaves that version behind; it
holds HEAD, so the next push from the same commit mounts it.

1. `.smoke-ok-<HEAD sha>` must be at the repo root, and the tree clean.
2. The job's config is checked as the kernel will check it. An eval config is a
   `configs/eval/*.yaml` file whose `backend` is `vllm` (FG-306; `hf` runs on the CPU, so it
   is refused on a T4), with an `ops` section giving `est_gpu_hours`. An SFT config
   (`--job sft`) is a `configs/sft/*.yaml` training config whose `run.device` is `auto` or
   `cuda`, with its `ops.est_gpu_hours`; it takes no `--reason` and no `--resume`, since an
   SFT run can't be resumed yet. `--job grpo` comes with FG-504. The run id must make a
   kernel slug Kaggle takes (`forge-<run_id>`: lowercase letters and digits with single
   hyphens between them, so not `a--b` or `run-`), the run must not clash with an earlier
   push, and a dev job needs a dev look left (see Dev). Kaggle refuses a kernel title or
   slug over 50 characters with a bare `400 Client Error: Bad Request` (FG-301), so the
   metadata check refuses one before any push, and a run's kernel name is kept within it
   (below).
3. The current version of `<owner>/forge-code` must hold HEAD's wheel (its `GIT_SHA`), or
   HEAD is published as a new version, which is waited for until Kaggle says `ready`. The
   kernel mounts that version by number, with `<owner>/forge-spider`, which must exist
   (`forge kaggle publish-data`).
4. The estimate must fit what the ledger (`forge ledger summary`) leaves of the week's 30
   hours, over the last seven UTC days, and what `kaggle quota -v` says is left when it
   answers. `--allow-over-budget` pushes anyway; the eval run's manifest config then holds
   `allow_over_budget: true`.
5. It prints the estimate and asks, unless `--yes`.
6. It writes `build/kaggle-kernel/<run_id>/`: `kaggle/entry.py` with the job's arguments
   in `KERNEL_ARGV`, and private GPU metadata with internet. It pushes with
   `--accelerator NvidiaTeslaT4` and a timeout of the estimate and half again (at most 12
   hours: plan section 5.3 stops a job 50% over its estimate), and reads the CLI's output:
   a push error fails the command although kaggle exits 0, and a CLI that fails before it
   says the kernel was pushed is checked against Kaggle's own status (see Dev). A source
   the CLI says it could not add is a warning instead, since Kaggle has saved and started
   that version without it. Either way a version Kaggle started is added to
   `runs/<run_id>/kernels.json`, which `status`, `logs`, `wait` and `output` read.

The run id is `<date>-<model>-eval-<sha7>` unless `--run-id` names it, and the kernel
`<owner>/forge-<run_id>`, titled `Forge <run_id>`. A run id that was pushed before is
refused. `--resume` pushes the run again from the same commit with the same options, as
`<owner>/forge-<run_id>-<n>` (` <n>` in the title), mounting the last attempt's kernel so
the eval carries on from its stored files. A name that would pass Kaggle's 50 characters,
such as the 57 of `Forge 2026-11-16-qwen2-5-coder-1-5b-instruct-eval-c8f4b41`, keeps the run
id up to a dash and adds 8 hex digits of its sha256 (`forge-2026-11-16-qwen2-5-coder-1-5b-`
and the digits, with `-<n>` on a resume), the same on every machine. The run id itself
stays as it is, in the kernel's arguments and everywhere else: `kernels.json` maps it to
its kernels, so `status`, `logs`, `wait`, `output` and `--resume` take the run id. An attempt
whose own kernel source Kaggle dropped failed to restore the run, so `kernels.json` marks it
(`kernel_source_dropped`) and the next resume mounts the attempt before it.

Dates are UTC everywhere Forge writes one: a run id's, a ledger row's, a
`docs/dev-usage.md` row's, and the last day of `forge ledger summary`'s week. `forge ledger
add` takes today's UTC date unless `--on` names another. West of UTC, a job pushed in the
evening is dated the next day in all of them, and this week's window counts its hours at
once. Rows already committed keep the dates they were written with.

In the kernel, `entry.py` runs its stages in order: restore (on a resume, the run's files
from the last attempt's output, first, so that an attempt failing at any later stage still
carries the run to the next resume), env (the interpreter, for the log), code (the code
dataset's `GIT_SHA` must be the pushed commit, and it must hold the wheel and both package
lists), install (`pip install --no-deps` the wheel, then the job's package list, see GPU
stack; it names any package the wheel needs that the image lacks),
data (the dataset found by its `spider.sha256`, `database.zip` unpacked if Kaggle left it
packed, every file checked against the list the installed wheel pins, not the dataset's own
copy), secrets, preflight (the same job on 5 examples, in `preflight/`), job, and upload
(`runs/<run_id>/` to the Hub run store, only with an `HF_TOKEN`; see Run store). Whatever
happens, it writes `runs/<run_id>/status.json` and marks a manifest left `running` as
failed. It exits 0 even when the job fails, since a kernel that ends in an error might not
keep its output, so a COMPLETE kernel is not yet a finished job: `forge kaggle wait` reads
`status.json` and exits 0 only for `finished`.

Sessions. The kernel holds its GPU from its start to its end, the setup, preflight, model
load and upload included, so that is the session its ledger row counts. `status.json`
gives the session's `started_at` and `ended_at` and its number in the run (`session`), and
the run's `sessions.json` gets the same span: the session the job began is stretched to
the kernel's, or, when a resumed job fails before its run starts again, the kernel's
session is added as failed. A resumed run's kernel holds the run from its restore stage,
so one that fails before its job gets that failed session too. A new run's kernel that
fails before its job holds no run and logs no row; add it with `forge ledger add`, for the
hours and date the note quotes from its `status.json`. The
upload stage first stretches the session up to the upload, so the Hub's copy misses only
the upload's own minutes; `forge kaggle wait` logs the whole session first. A resumed run's
output holds every session so far, so a session that was never waited for still gets its
row.

SFT. An SFT job trains as `forge train sft` does, with the config as `forge kaggle run`
checked it, defaults filled in, and its manifest's platform is the kernel's host. Its
preflight trains 2 steps of the same config in `preflight/`, saving the adapter as the job
will, then frees the GPU for the job, which loads the model again. The rows are built in
the temp folder; the checkpoints stay in the run's folder, which the upload stage puts in
the run store when the job ends. The job's time guard counts `ops.time_guard_minutes`
from the job's start, or less when the push's timeout comes first: counted from the
kernel's start, as an eval job's deadline is, with the same tenth of it kept back, here
for the guard's last save and the upload. A training run writes no `sessions.json` of its
own, so the kernel's session is added to the run with the run's status.

`--adapter` gives an eval job its adapter when the config names none, so a checkpoint is
scored with its base model's val config at the prompt version the run trained on, which
ADR-0007 makes p2 for every SFT run (`configs/eval/<name>-val-p2.yaml`), and the two runs
differ in the adapter alone. An adapter that is a training run's checkpoint,
`runs/<run_id>/checkpoints/<name>` as on the laptop, is fetched from the run store to the
same path in the kernel before the model loads, so it needs the `HF_TOKEN` secret; the
manifest names the adapter by that path, which `forge runs promote` reads, and records the
`adapter_sha256` of the files it loaded. The push reads the adapter as both of those do
(`forge.ops.adapters.checkpoint_ref`) and refuses, before any upload, a checkpoint outside
`runs/`, a path in `runs/` that names no checkpoint, such as `runs/<run_id>` (whatever this
machine's `runs/` holds), a folder on this machine, and a checkpoint of a run this machine
knows trained another model. A trailing `/` or a leading `./` is pushed spelled the one way.

vLLM. An eval job's `backend: vllm` loads the model with `forge.infer.vllm_backend`: fp16
(`dtype="half"`, since a T4 has no bf16), up to 2,048 tokens of prompt and reply, 85% of the
GPU's memory unless the config sets another share in `gpu_memory_utilization` (which the
manifest's config then records), and LoRA (rank up to 64) only when the config names an
`adapter`, a folder or a Hub repo fetched at its `@revision`. Every prompt is rendered by
`render_chat`, Qwen3's with thinking off, and sent as the token ids the hf backend gives its
model on the laptop, never through vLLM's own `LLM.chat`. Decoding takes only the config's settings, never a
checkpoint's `generation_config.json` defaults, and each question has a seed of its own.
A temperature between 0 and 0.01 is refused, on every backend, since vLLM would decode it
at 0.01. The preflight and the job share one engine, loaded once, since a second might not
fit beside the first. `forge eval run --backend vllm` on the laptop is refused: its manifest
would call a GPU run `local-cpu` (ADR-0016). A config with `quantization: awq` loads an
AWQ checkpoint's 4-bit weights, for a reference model too big for the T4 in fp16, as
FG-307's `Qwen/Qwen3-8B-AWQ` (ADR-0012); vLLM's AWQ kernels need compute capability 7.5,
which the T4 has, and FG-307's reference runs of 2026-11-17 loaded and scored that
checkpoint on a T4, in 0.21 h on val and 0.25 h on dev by their ledger rows. `quantization:
bitsandbytes` is refused before any push, in one line: vLLM 0.33.0 has no bitsandbytes and
refused it as "Unknown quantization method: bitsandbytes" (FG-301).

Dev. A config with `split: dev` needs `--reason`, as `forge eval run --split dev` does, and
is refused once `docs/dev-usage.md` holds six looks (ADR-0004). The push spends the look:
`forge kaggle run` writes its row, with the run id and the reason, as soon as Kaggle
accepts the push, and records the kernel whether or not that write worked: if either step
fails, the error names it and says not to push again, since the kernel runs. A push that
Kaggle refuses ("Kernel push error") fails and spends nothing. A kaggle CLI that exits with
an error before it says the kernel was pushed may only have lost Kaggle's answer, a
timeout among them, after Kaggle saved the kernel, so `forge kaggle run` then asks Kaggle
with `kaggle kernels status` whether it has the attempt's kernel, which no earlier push
made:

- Kaggle has it: the push was made. It is recorded in `kernels.json`, with no version, so
  the kernel's latest is followed, its look is logged, and a warning says the answer was
  lost.
- Kaggle has no such kernel: the push never started, so it fails and spends nothing.
- Kaggle can't be asked either: the push fails and records nothing, and the error says
  whether Kaggle saved the kernel is unknown. Check with `kaggle kernels status
  <owner>/<slug>` before pushing again: a kernel Kaggle has runs, and a dev job's then
  reads dev without its row in `docs/dev-usage.md`.

Any other answer from the CLI, one that names no version among them, is a kernel that runs,
and its look is spent. So a kernel that may read dev has its row unless the error says it
hasn't. The spike, which reads no dev, isn't checked: its push makes a new version of the
one kernel earlier pushes made, and the status names no version, so it can't tell this
push's from the one before. When its CLI fails, the spike's push fails as before.

A `--resume` carries on the same run, from the same commit with the same options, so it is
the same look: it adds no row, and it goes even once the six are spent. In the kernel the
preflight answers 5 val questions instead, so dev loads once, in the job, whose own row
goes to `dev-usage.md` in the kernel's output. A dev job has no `--emit-notebook`, since a
notebook can run any number of times and log none of them. Name a dev run with `--run-id`
when the same model's val run was pushed from the same commit on the same day, since both
would get the same default id.

Secrets. `HF_TOKEN` and `WANDB_API_KEY` would come from Kaggle Secrets attached to the
kernel, but a binding made in the web editor doesn't reach a version pushed from the CLI
(FG-301), so the secrets stage records `kaggle secrets: ConnectionError` for both. Each is
read without the line breaks and outer spaces huggingface_hub drops, and its value becomes
`***` in every stage result in `status.json`, the error, the traceback, a failed run's
`error.txt` and every log record, such as the run store's warning on stderr when it tries a
Hub call again. Without them the job still runs, nothing is uploaded, and `forge kaggle
output <run_id>` is how its results come back: the run's files go to `runs/<run_id>/`, and
the log and the preflight to `runs/<run_id>/kernel/`.

Provenance. A kernel has the wheel but no git checkout, so `entry.py` passes the code
dataset's `GIT_SHA` in `FORGE_CODE_SHA`, and the manifest's `git_sha` takes it only where
there is no checkout (`forge.ops.manifest.code_sha`); inside a checkout the variable is
never read. `RunManifest` has no field saying where the commit came from, and adding one
would change a contract, so the platform says it: a `kaggle` or `colab` run's `git_sha`
is its code dataset's, checked against HEAD before the push and against the pushed commit
in the kernel.

`--job spike` pushes FG-301's throughput script with `configs/spike/throughput.yaml`
through the same checks; it runs without Forge, so it mounts no dataset and has no
preflight. `--emit-notebook out.ipynb` writes the job as a Colab notebook and pushes
nothing: a cell fetches both datasets with the kaggle CLI (a `KAGGLE_API_TOKEN` Colab
secret), the next holds `entry.py` with the same arguments and `--host colab`, and the
last runs it. Add its hours with `forge ledger add --platform colab`. A resume has no
notebook: the fetch cell gets the two datasets, never a kernel's output, so `--resume
--emit-notebook` is refused.

## Run store

Runs keep their files off the machine that made them, in the private Hub model repo
`FORGE_RUN_STORE` (`hoomanesteki/forge-runs` by default), laid out as plan section 3.9 says:
`runs/<run_id>/manifest.json`, `config.yaml`, `eval/...`, and for training jobs
`checkpoints/step-NNNNNN/` and `checkpoints/best/`. `forge.ops.hub` holds the stores:
`HubRunStore` for the repo and `LocalRunStore` for a folder.

```bash
uv run forge runs list                          # run ids in the store; --store local for runs/
uv run forge runs show <run_id>                 # its manifest and latest checkpoint
uv run forge runs pull <run_id>                 # mirror it into runs/<run_id>/
uv run forge runs pull <run_id> --store kaggle  # the same as forge kaggle output
uv run forge runs push <run_id>                 # make the store's copy runs/<run_id>/
uv run forge runs push <run_id> --adapter-sha256 <digest>  # only if its adapter has that digest
uv run forge runs promote <run_id> --config <base val config> --eval <eval_run_id> ...  # best/
```

- A pull makes `runs/<run_id>/` the store's copy of the run, as `forge kaggle output` does with
  a kernel's: a file the copy lacks goes, such as an earlier attempt's `error.txt` or a report
  made here (`forge report run` makes it again). `kernels.json` and `kernel/` stay as they
  are unless the copy's `kernels.json` lists more pushes than this machine's, since `forge
  runs push` uploads them and the store's may be older. Other runs' folders are not touched.
- A pull, from either store, also writes a row in `docs/compute-ledger.md` for each of the
  run's sessions that has ended, once a session, as `forge kaggle wait` and `output` do:
  the UTC date the session started, the run's platform and GPU, the session's wall-clock
  hours, what the run was and how the session ended, and the run's cost so far less what
  its earlier rows hold. A run's `sessions.json` lists its sessions, one an attempt
  (`forge.ops.sessions`); a run without one has one session, from its manifest's
  `started_at` to `ended_at`. A later session's row ends its purpose with `(session N)`.
  A `local-cpu` run gets no row, and a session still running none yet: pull the run again
  once it ends. A session killed before it could record its end has no hours to log, so
  the pull names it; add it with `forge ledger add`, its purpose ending `(session N)`.
  Kernel output with no manifest, such as a spike's, gets no row either; add it the same
  way.
- Invariant I7 (`tests/integration/test_invariants.py`) checks that every session of every
  `kaggle`, `colab` or `modal` run under `docs/runs/` and `runs/` has a row, that a run's
  rows name its platform, and that the ledger's cash stays within $50.
- `forge eval run --store hub` first compares `runs/<run_id>/` with the store's copy, and fetches
  the store's when it is newer or the folder has none, so a run another machine carried on
  resumes from where that machine left it, and a finished run is never run again. It puts the
  folder in the store when the run ends, finished or failed. If that upload fails after the run
  finished, the command exits 1 and says so; `forge runs push <run_id>` tries again.
- Which copy is newer comes from the run's `sessions.json`, one session an attempt
  (`forge.ops.run_copies`): a copy whose sessions begin with all of the other's is the newer.
  When neither's do, two machines carried the run on from one point and each holds work the
  other lacks, so `forge eval run` refuses to resume it and no upload replaces either. Keep the
  store's with `forge runs pull <run_id>`, or evaluate again under a new `--run-id`.
- A run made on this machine, such as a `local-mps` pilot, has nothing to pull:
  `uv run forge ledger record <run_id>` writes its rows the same way, one for each session
  that has ended, from `runs/<run_id>/` (`FORGE_RUNS_DIR`), so its hours are never typed.
  A session still running gets no row yet: run the same command again once it ends.
- `--adapter-sha256` holds a push to the digest of the run's adapter taken where it was
  trained (`forge.ops.adapters.adapter_sha256` of the folder the manifest names as its
  adapter), for a run whose manifest records none: if the folder holds other files, nothing
  is uploaded. FG-491's SFT pilot is one; its digest is in `docs/runs/pending.yaml`.
- The end of an eval run and a push replace the store's copy: files it has that the folder lacks
  are deleted, such as the `error.txt` of an attempt that failed. The kernel's upload stage only
  adds, since a kernel that failed early may hold part of a run. None of them uploads over a
  store whose copy holds a session the folder lacks: the command says so and uploads nothing.
- A Hub call that fails on the network (a dropped or refused connection, a timeout, 408, 429
  or 5xx, downloads included) is tried again by one layer, but for one rare case.
  huggingface_hub 2.4 tries most such failures 5 more times itself, 1, 2, 4, 8 and 8 seconds
  apart; what it doesn't retry, the store tries 3 more times, 2, 4 and 8 seconds apart: a
  commit, the repo lookup that starts a pull, a connection dropped on a download, and a 5xx
  other than 500, 502, 503 and 504. The store tells which from the failure that ended the
  call (`forge.ops.hub.client_retried`), so both retry when a download's first metadata
  request is refused, times out or gets a status the client retries (408, 429, 500, 502,
  503 or 504), and the client's retries of it then drop their connections: each of the
  store's 4 tries then waits out the client's 23 s, 106 s of waiting in all with the
  store's 14 s. Otherwise a call that keeps failing waits at most 14 s between the store's
  tries or 23 s inside the client's, then says the store is unavailable; when the Hub's
  answer says how long to wait (`Retry-After`, or a 429's `RateLimit` header), the client
  waits that long and 1 s more before each of its 5 tries, so a `Retry-After: 60` costs
  305 s. A missing run is reported as missing, and a missing repo, a refused token or a
  refused upload fails at once.
- The token is `HF_TOKEN` from the environment or `.env`; without one, huggingface_hub uses a
  saved login. Nothing creates the repo: make it private by hand (Accounts, step 2), and
  `forge doctor` checks it.
- A kernel has no `HF_TOKEN` (Accounts, step 3), so its upload stage is skipped and a
  kernel run reaches the store from the laptop: `forge kaggle wait <run_id>`, `forge runs
  pull <run_id> --store kaggle`, then `forge runs push <run_id>`. That was the store's
  first real use, on 2026-11-16, for FG-303's 50-question run: `run
  2026-11-16-smoke-kaggle-50-val-c8f4b41 (eval, finished): uploaded to
  hoomanesteki/forge-runs as runs/2026-11-16-smoke-kaggle-50-val-c8f4b41`.
- Retention keeps a run's latest 2 `step-NNNNNN` checkpoints and `best/`
  (`prune_checkpoints`), for training jobs, which push a checkpoint every 50 steps (plan
  section 3.8). `latest_checkpoint` picks the highest step, which a resumed job starts from.
  A training config's `ops.keep_checkpoints_at` names shares of the run, such as `[0.5]`,
  whose checkpoints are saved and kept too, for val to compare with `checkpoints/final`;
  the manifest's outputs name each, as `kept_at_0.5`.
- `forge runs promote <sft_run_id> --config <base val config> --eval <eval_run_id> ...`
  copies the SFT run's checkpoint that val scores highest to `checkpoints/best/`, where GRPO
  starts (`forge.train.sft.promote_best`). The config is the base model's on the whole val
  split at the prompt version the run trained on, `configs/eval/<name>-val.yaml` for p1 and
  `configs/eval/<name>-val-p2.yaml` for p2. Each eval run is a finished run in `runs/` of that
  config with one of the run's checkpoints as its adapter and nothing else changed,
  `runs/<sft_run_id>/checkpoints/step-NNNNNN` or `.../final`, and it must have scored the
  files the SFT run saved there, by their `adapter_sha256`. The highest official EX wins,
  the later checkpoint on a tie, and `best.json` beside the manifest records each score and the eval
  run it comes from; the manifest's outputs name both. Dev is never read. With the hub
  store, the default, pull the SFT run and its eval runs first: the copy is made here and
  pushed, `best/` first and the manifest last.

## Resume after a session limit

A Kaggle session ends at 12 hours, and whatever is still running then is lost, so a job
stops itself first, at a deadline short of the limit (`forge.ops.timeguard`; plan section
3.8 puts it at 11 h 15 min for a full session). The guard times the job's steps, and once
the deadline is within the longest step so far, plus a margin when one is given, it stops
before the next one.

- Training: `TimeGuardCallback(deadline_minutes, store=..., run_id=...)` from
  `forge.train.callbacks` goes in the trainer's callbacks, made when the job starts, since
  the deadline counts from then. When it fires, the trainer saves `checkpoint-<step>` in its
  output folder and stops after that step; the callback pushes the folder to the run's
  `checkpoints/step-NNNNNN/` in the run store, marks the stored manifest
  `stopped_time_guard` and prunes the older checkpoints as retention says. A new session
  resumes from `latest_checkpoint`.
- Eval: `run_eval(..., deadline=...)` asks before each batch. A stopped run keeps the
  batches it stored, leaves official EX and its report to the attempt that finishes it,
  says `stopped_time_guard` in its manifest and goes to its run store like a run that
  ended. It resumes like a failed one: the same `--run-id`, or `forge kaggle run
  --resume`. Official EX, the report and the upload follow the last batch, where no
  guard asks, so an eval deadline keeps a tenth of its time back for them
  (`forge.ops.timeguard.eval_deadline`). A kernel's eval job gets the push's timeout in
  its arguments (`--timeout-s`) and counts it from the kernel's start, setup included;
  its upload stage still runs, and `forge kaggle wait` exits 1 and names the resume.
  `forge eval run --deadline-minutes N` counts from the command's start and exits 1 when
  it stops.
- A resumed run keeps its first `started_at`, but each attempt is a session of its own in
  the run's `sessions.json`, with its own start, end and status. A manifest's `gpu_hours`
  adds up its sessions' wall-clock hours on a platform with a GPU (`local-mps` too), and is
  none on `local-cpu`; each session gets its own ledger row. The time between two sessions
  holds no GPU, so it counts in neither. A kernel's job writes its manifest before the
  kernel ends, so there `gpu_hours`, which `forge runs show` prints, counts the job's own
  span, while the session's row counts the whole kernel, setup, preflight and upload
  included; the budget reads only the rows.

## Baselines

FG-307 measures the base models before any training (ADR-0012). Each run is a kernel job
from `configs/eval/`, where every model has a config per run, with the model pinned to
the commit `configs/models.yaml` names. `uv run python scripts/check_models.py` checks
those commits, licenses and gated flags against the Hub first, reading metadata only.
Push them in this order, from one clean commit that passed `make smoke`:

1. The candidates on val with p1: `qwen3-1-7b-val`, `qwen3-4b-val` and
   `qwen2-5-coder-1-5b-val`.
2. The model ADR-0012's rule chooses from step 1's runs, with p2: its `<name>-val-p2`
   (ADR-0007). The rule also needs FG-301's GRPO memory check, which every candidate
   passed (ADR-0012).
3. The reference on val, in 4-bit: `qwen3-8b-val`, the AWQ checkpoint `Qwen/Qwen3-8B-AWQ`
   (FG-398), since vLLM 0.33.0 has no bitsandbytes (FG-301). Push the reference's dev run
   (step 5) only once this one has loaded it, since that push spends a dev look. The val
   run of 2026-11-17 was the first to load the checkpoint on a T4, and it did.
4. I6's repeat: `qwen2-5-coder-1-5b-val-100`, the first 100 val questions again.
5. The model ADR-0012's rule chooses, and the reference, on dev, once each:
   `<name>-dev` and `qwen3-8b-dev`, with `--reason baseline`. These are dev looks 1 and 2.

Steps 1, 2 and 4 ran from `c8f4b41` on 2026-11-16 and are committed under `docs/runs/`: the
rule chose `Qwen/Qwen2.5-Coder-1.5B-Instruct` (ADR-0012), and p2 won (ADR-0007). So steps 3
and 5 run `qwen3-8b-val`, `qwen2-5-coder-1-5b-dev` and `qwen3-8b-dev`, whose configs use p2
since FG-307 recorded those runs: push them from that commit or a later one, never from
`c8f4b41`, where they still say p1. They ran on 2026-11-17: the reference's val run and the
chosen model's dev run from `2f77281`, and the reference's dev run from `d2d6a3a`, once its
val run had loaded the AWQ checkpoint. All of them are committed under `docs/runs/`.

The same model's runs from one commit on one day would share a default run id, so name each:
`--run-id <date>-<config name>-<sha7>`, such as `2026-11-20-qwen3-4b-val-p2-abc1234`. Kaggle
runs at most two GPU sessions at once and
refuses a third push ("Maximum batch GPU session count of 2 reached."), so push two and wait
for one to end before the next. For each run, `forge kaggle wait <run_id>`, which adds its
ledger row, then `forge runs pull <run_id> --store kaggle` (the kernel has no `HF_TOKEN`, so
the Hub store gets the run only from `forge runs push`). Render its report on the laptop
with `FORGE_DATA_DIR=data/spider uv run forge report run <run_id>`, so its rebuild compares
the chart, and copy the folder whole to `docs/runs/`.
`tests/integration/test_artifacts.py::test_phase3_baseline_reports_present` then checks
every run is there with its report and I6 on step 4 against step 1, and
`tests/integration/test_invariants.py::test_i5_agreement_on_committed_runs` checks I5 on
their replies: pooled, and for each run of the chosen model alone (ADR-0008).

## SFT runs

FG-404 trains the LoRA SFT adapters on Kaggle T4s and picks the main run's on val, in the
story's budget of about 4.5 GPU hours (FG-404 in the plan); plan section 3.8's rows give
4.4 of them, 1 for the pilot, 2.5 for the main run and 0.90 for its two checkpoints on
val, and have none for the pilot's val run. Push the jobs in this order, from one clean
commit that passed `make smoke`, with the `HF_TOKEN` secret attached to each kernel: the
runs go to the run store, and an eval job fetches its checkpoint from there. After each,
`forge kaggle wait <run_id>`, then `forge runs pull <run_id>`, which adds its ledger row.

Every SFT run trains on p2, which ADR-0007 fixed for every later phase, and its checkpoints
are scored with its base model's val config at p2, as `forge runs promote` requires. The
pilot trains the small candidate, Qwen3-1.7B (plan section 3.8), and the main run the model
ADR-0012 chose, `Qwen/Qwen2.5-Coder-1.5B-Instruct`. Both SFT configs predate FG-307's runs
and still train on p1, and the main one names Qwen3-4B, so each needs the change
`docs/runs/pending.yaml` lists before its run.

1. The pilot, 1 epoch on the small candidate, to show the setup works before the main
   run: `uv run forge kaggle run --job sft --config configs/sft/qwen3-1p7b-pilot.yaml`.
2. Its final adapter on val, with that candidate's FG-307 config at p2, so the two runs
   differ in the adapter alone: `uv run forge kaggle run --job eval --config
   configs/eval/qwen3-1-7b-val-p2.yaml --adapter runs/<pilot_run_id>/checkpoints/final
   --run-id <date>-qwen3-1-7b-sft-val-<sha7>`. The push checks the adapter first: a run's
   checkpoint in `runs/`, as here, which the kernel fetches, or a Hub repo@revision. Then
   `uv run forge runs promote <pilot_run_id> --config configs/eval/qwen3-1-7b-val-p2.yaml
   --eval <that run id>` sets the pilot's `checkpoints/best`, where FG-507's GRPO pilot
   starts.
3. The main run, 1 epoch on the chosen model: `uv run forge kaggle run --job sft --config
   <main config>`, the main SFT config once it trains `Qwen/Qwen2.5-Coder-1.5B-Instruct` on
   p2. Until that change it is `configs/sft/qwen3-4b-main.yaml`, Qwen3-4B on p1. It keeps
   its checkpoint half way (`ops.keep_checkpoints_at: [0.5]`) beside `checkpoints/final`;
   `forge runs show <run_id>` lists both, the first as `kept_at_0.5`.
4. Both checkpoints on val, each with `configs/eval/qwen2-5-coder-1-5b-val-p2.yaml`, the
   chosen model's FG-307 config at p2, and its own `--adapter
   runs/<main_run_id>/<checkpoint>` and `--run-id`.
5. `uv run forge runs promote <main_run_id> --config
   configs/eval/qwen2-5-coder-1-5b-val-p2.yaml --eval <half run id> --eval <final run id>`:
   the higher official EX goes to `checkpoints/best`, the final adapter on a tie. Each eval
   run must be that config, the base model's on the whole val split, with its checkpoint as
   the adapter and nothing else changed, and must have scored the files the run saved
   there: an eval run records the `adapter_sha256` of its adapter, and the SFT run the
   digest of each checkpoint it keeps.

Render each eval run's report on the laptop (`FORGE_DATA_DIR=data/spider uv run forge
report run <run_id>`) and copy the folder whole to `docs/runs/`. Copy each SFT run
without its checkpoints, which stay in the run store: `manifest.json`, `config.yaml`,
`metrics.jsonl` and `best.json`. Then take FG-404's entries out of
`docs/runs/pending.yaml`, and `tests/integration/test_artifacts.py::test_phase4_sft_runs_present`
checks every run, report and ledger row, and each `best/` against its eval runs' scores.
No step reads dev. If val EX was still rising at the end of the epoch and the budget
allows, a second epoch is the product owner's call (ADR-0010), recorded in
`docs/progress.md`.

Last, FG-405 compares the main model with and without SFT, on the laptop. Run b is the val
run of the main run's `checkpoints/best`, the one its `best.json` names under `eval_runs`;
run a is FG-307's run of the same config, `2026-11-16-qwen2-5-coder-1-5b-p2-c8f4b41`,
already in `docs/runs/`, so the two differ in the adapter alone:

```bash
FORGE_RUNS_DIR=docs/runs FORGE_DATA_DIR=data/spider uv run forge report compare <base val run id> <best val run id>
```

It writes `eval/compare-<base val run id>.md` in run b's folder, which
`tests/integration/test_artifacts.py::test_phase4_compare_report_rebuildable` rebuilds.
Add 3 to 5 bullets on it to `docs/notes/sft.md`, a list of their own beside the pilot's,
from that report alone and linking it, and take FG-405's entry out of
`docs/runs/pending.yaml`.

## Troubleshooting

- **`ModuleNotFoundError: forge` on macOS.** The venv can come out with the hidden file
  flag set, and Python 3.12 skips hidden `.pth` files, so the editable install vanishes
  after uv re-syncs. Every make target clears the flag first, and pytest puts `src` on the
  path itself. Before a bare `uv run forge ...`, run `make unhide` once after a sync.
