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
3. **Kaggle Secrets.** In any Kaggle notebook, open Add-ons, Secrets and add `HF_TOKEN`,
   plus `WANDB_API_KEY` if you use W&B. Kernels read them through
   `kaggle_secrets.UserSecretsClient`. The CLI cannot attach a secret to a kernel, so attach
   each one once in the web editor. FG-301 checks whether pushed runs can read them; if not,
   jobs return their results through `kaggle kernels output` instead.
4. **Weights & Biases (optional).** A free account; set `WANDB_API_KEY` in `.env` or Kaggle
   Secrets for live charts. Without it, metrics go to JSONL in the run store.

`forge doctor` also wants 10 GB of free disk for Spider, model weights and caches. Lower the
bar with `FORGE_MIN_FREE_GB` if you know you need less. It never prints token values: any
environment variable whose name contains TOKEN, KEY or SECRET is masked in its output.

## Code and data on Kaggle

A GPU job runs the code and reads the data from two private Kaggle datasets, both owned by
`FORGE_KAGGLE_USER` (or the username in `kaggle.json`):

1. `uv run forge kaggle publish-data`, once. It checks every file `spider.sha256` lists in
   `data/spider` (or `FORGE_DATA_DIR`), copies them and that file to `build/kaggle-data/`,
   and creates `<owner>/forge-spider` with Spider's CC BY-SA 4.0 attribution and citation in
   its description. Once the dataset exists, it uploads nothing.
2. `uv run forge kaggle publish-code`, on a clean tree, for each commit a job runs. It builds
   the wheel into `build/kaggle-code/` with `kaggle/requirements-gpu.txt` and `GIT_SHA`, then
   uploads it as a new version of `<owner>/forge-code` noted with the SHA, creating the
   dataset the first time.

Both datasets are private: Forge never passes `-u`, and each `dataset-metadata.json` says
`"isPrivate": true`, so `kaggle datasets metadata --update` keeps them private too.

## GPU stack

A kernel installs `kaggle/requirements-gpu.txt` over Kaggle's image, after the wheel. vLLM
decides most of it (FG-306), and none of it has run on a T4 yet (FG-301 measures it):

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

`forge kaggle run` stops at the first check that fails, before anything is uploaded:

1. `.smoke-ok-<HEAD sha>` must be at the repo root, and the tree clean.
2. The job's config is checked as the kernel will check it. An eval config is a
   `configs/eval/*.yaml` file whose `backend` is `vllm` (FG-306; `hf` runs on the CPU, so it
   is refused on a T4), with an `ops` section giving `est_gpu_hours`. `--job sft` and
   `--job grpo` come with FG-403 and FG-504.
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
   a push error fails the command although kaggle exits 0. A source the CLI says it could
   not add is a warning instead, since Kaggle has saved and started that version without
   it. Either way a version Kaggle started is added to `runs/<run_id>/kernels.json`, which
   `status`, `logs`, `wait` and `output` read.

The run id is `<date>-<model>-eval-<sha7>` unless `--run-id` names it, and the kernel
`<owner>/forge-<run_id>`. A run id that was pushed before is refused. `--resume` pushes the
run again from the same commit with the same options, as `<owner>/forge-<run_id>-<n>`,
mounting the last attempt's kernel so the eval carries on from its stored files.

Dates are UTC everywhere Forge writes one: a run id's, a ledger row's, a
`docs/dev-usage.md` row's, and the last day of `forge ledger summary`'s week. `forge ledger
add` takes today's UTC date unless `--on` names another. West of UTC, a job pushed in the
evening is dated the next day in all of them, and this week's window counts its hours at
once. Rows already committed keep the dates they were written with.

In the kernel, `entry.py` runs its stages in order: restore (on a resume, the run's files
from the last attempt's output, first, so that an attempt failing at any later stage still
carries the run to the next resume), env (the interpreter, for the log), code (the code
dataset's `GIT_SHA` must be the pushed commit), install (`pip install --no-deps` the wheel,
then `requirements-gpu.txt`; it names any package the wheel needs that the image lacks),
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
fails before its job holds no run and logs no row; add it with `forge ledger add`. The
upload stage first stretches the session up to the upload, so the Hub's copy misses only
the upload's own minutes; `forge kaggle wait` logs the whole session first. A resumed run's
output holds every session so far, so a session that was never waited for still gets its
row.

vLLM. An eval job's `backend: vllm` loads the model with `forge.infer.vllm_backend`: fp16
(`dtype="half"`, since a T4 has no bf16), up to 2,048 tokens of prompt and reply, 85% of the
GPU's memory, and LoRA (rank up to 64) only when the config names an `adapter`, a folder or
a Hub repo fetched at its `@revision`. Qwen3 prompts go through `LLM.chat` with thinking off;
other templates are rendered by `render_chat`, as on the laptop. Decoding takes only the
config's settings, never a checkpoint's `generation_config.json` defaults, and each question
has a seed of its own. The preflight and the job share one engine, loaded once, since a
second might not fit beside the first. `forge eval run --backend vllm` on the laptop is
refused: its manifest would call a GPU run `local-cpu` (ADR-0016). A config with
`quantization: bitsandbytes` loads the weights in 4-bit, for a reference model too big for
the T4 in fp16; the manifest's config then records it.

Dev. A config with `split: dev` needs `--reason`, as `forge eval run --split dev` does, and
is refused once `docs/dev-usage.md` holds six looks (ADR-0004). The push spends the look:
`forge kaggle run` writes its row, with the run id and the reason, as soon as Kaggle
accepts the push, before it records the kernel or does anything else that could fail. So a
kernel that may read dev always has its row, and a push Kaggle refuses spends nothing. A
`--resume` carries on the same run, from the same commit with the same options, so it is
the same look: it adds no row, and it goes even once the six are spent. In the kernel the
preflight answers 5 val questions instead, so dev loads once, in the job, whose own row
goes to `dev-usage.md` in the kernel's output. A dev job has no `--emit-notebook`, since a
notebook can run any number of times and log none of them. Name a dev run with `--run-id`
when the same model's val run was pushed from the same commit on the same day, since both
would get the same default id.

Secrets. `HF_TOKEN` and `WANDB_API_KEY` come from Kaggle Secrets when attached to the
kernel, which only the web editor can do. Each is read without the line breaks and outer
spaces huggingface_hub drops, and its value becomes `***` in every stage result in
`status.json`, the error, the traceback and a failed run's `error.txt`. Without them the
job still runs, nothing is uploaded, and `forge kaggle output <run_id>` is how its results
come back: the run's files go to `runs/<run_id>/`, and the log and the preflight to
`runs/<run_id>/kernel/`.

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
last runs it. Add its hours with `forge ledger add --platform colab`.

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
```

- A pull replaces each top-level file or folder of the run that the store holds and keeps the
  rest, such as `kernels.json`, as `forge kaggle output` does.
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
- `forge eval run --store hub` fetches the run from the store when `runs/<run_id>/` has no
  manifest, so a run another machine started resumes here, and puts the folder in the store
  when the run ends, finished or failed. If that upload fails after the run finished, the
  command exits 1 and says so; `forge runs push <run_id>` tries again.
- The end of an eval run and a push replace the store's copy: files it has that the folder lacks
  are deleted, such as the `error.txt` of an attempt that failed. The kernel's upload stage only
  adds, since a kernel that failed early may hold part of a run.
- A Hub call that fails on the network (a dropped connection, a timeout, 408, 429 or 5xx, a
  failed upload or Xet transfer) is tried 3 more times, 2, 4 and 8 seconds apart. A missing
  repo or a refused token fails at once.
- The token is `HF_TOKEN` from the environment or `.env`; without one, huggingface_hub uses a
  saved login. Nothing creates the repo: make it private by hand (Accounts, step 2), and
  `forge doctor` checks it.
- Retention keeps a run's latest 2 `step-NNNNNN` checkpoints and `best/`
  (`prune_checkpoints`), for training jobs, which push a checkpoint every 50 steps (plan
  section 3.8). `latest_checkpoint` picks the highest step, which a resumed job starts from.

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
  holds no GPU, so it counts in neither.

## Baselines

FG-307 measures the base models before any training (ADR-0012). Each run is a kernel job
from `configs/eval/`, where every model has a config per run, with the model pinned to
the commit `configs/models.yaml` names. `uv run python scripts/check_models.py` checks
those commits, licenses and gated flags against the Hub first, reading metadata only.
Push them in this order, from one clean commit that passed `make smoke`:

1. The candidates on val with p1: `qwen3-1-7b-val`, `qwen3-4b-val` and
   `qwen2-5-coder-1-5b-val`.
2. The top candidate by val official EX with p2: its `<name>-val-p2` (ADR-0007).
3. The reference on val, in 4-bit: `qwen3-8b-val`.
4. I6's repeat: `qwen2-5-coder-1-5b-val-100`, the first 100 val questions again.
5. The model ADR-0012's rule chooses, and the reference, on dev, once each:
   `<name>-dev` and `qwen3-8b-dev`, with `--reason baseline`. These are dev looks 1 and 2.

The same model's runs from one commit on one day would share a default run id, so name
each: `--run-id <date>-<config name>-<sha7>`, such as `2026-11-20-qwen3-4b-val-p2-abc1234`.
For each run, `forge kaggle wait <run_id>`, which adds its ledger row, then `forge runs pull
<run_id> --store kaggle` (or the hub store, with an `HF_TOKEN` in the kernel). Render its
report on the laptop with `FORGE_DATA_DIR=data/spider uv run forge report run <run_id>`, so
its rebuild compares the chart, and copy the folder whole to `docs/runs/`.
`tests/integration/test_artifacts.py::test_phase3_baseline_reports_present` then checks
every run is there with its report, I5 on its replies and I6 on step 4 against step 1.

## Troubleshooting

- **`ModuleNotFoundError: forge` on macOS.** The venv can come out with the hidden file
  flag set, and Python 3.12 skips hidden `.pth` files, so the editable install vanishes
  after uv re-syncs. Every make target clears the flag first, and pytest puts `src` on the
  path itself. Before a bare `uv run forge ...`, run `make unhide` once after a sync.
