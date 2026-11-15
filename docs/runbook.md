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

## Kaggle jobs

Every GPU job goes through `forge kaggle run`, on a clean commit that passed `make smoke`:

```bash
uv run forge kaggle run --job eval --config configs/eval/<name>.yaml   # prints the run id
uv run forge kaggle wait <run_id>      # until COMPLETE, ERROR or CANCEL_*
uv run forge kaggle output <run_id>    # the kernel's output into runs/<run_id>/
```

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
   hours, and what `kaggle quota -v` says is left when it answers. `--allow-over-budget`
   pushes anyway; the eval run's manifest config then holds `allow_over_budget: true`.
5. It prints the estimate and asks, unless `--yes`.
6. It writes `build/kaggle-kernel/<run_id>/`: `kaggle/entry.py` with the job's arguments
   in `KERNEL_ARGV`, and private GPU metadata with internet. It pushes with
   `--accelerator NvidiaTeslaT4` and a timeout of the estimate and half again (at most 12
   hours: plan section 5.3 stops a job 50% over its estimate), and reads the CLI's output:
   a push error, or a dataset source it could not add, fails the command although
   kaggle exits 0. The kernel is added to `runs/<run_id>/kernels.json`, which `status`,
   `logs`, `wait` and `output` read.

The run id is `<date>-<model>-eval-<sha7>` unless `--run-id` names it, and the kernel
`<owner>/forge-<run_id>`. A run id that was pushed before is refused. `--resume` pushes the
run again from the same commit with the same options, as `<owner>/forge-<run_id>-<n>`,
mounting the last attempt's kernel so the eval carries on from its stored files.

In the kernel, `entry.py` runs its stages in order: env (the interpreter, for the log),
code (the code dataset's `GIT_SHA` must be the pushed commit), install (`pip install
--no-deps` the wheel, then `requirements-gpu.txt`; it names any package the wheel needs
that the image lacks), data (`spider.sha256` found by glob, `database.zip` unpacked if
Kaggle left it packed, every file checked), secrets, preflight (the same job on 5 examples,
in `preflight/`), job, and upload (`runs/<run_id>/` to the Hub run store, only with an
`HF_TOKEN`; see Run store). Whatever happens, it writes `runs/<run_id>/status.json` and
marks a manifest left `running` as failed. It exits 0 even when the job fails, since a
kernel that ends in an error might not keep its output, so a COMPLETE kernel is not yet a
finished job: `forge kaggle wait` reads `status.json` and exits 0 only for `finished`.

Secrets. `HF_TOKEN` and `WANDB_API_KEY` come from Kaggle Secrets when attached to the
kernel, which only the web editor can do. Without them the job still runs, nothing is
uploaded, and `forge kaggle output <run_id>` is how its results come back: the run's files
go to `runs/<run_id>/`, and the log and the preflight to `runs/<run_id>/kernel/`.

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
  `checkpoints/step-NNNNNN/` in the run store and marks the stored manifest
  `stopped_time_guard`. A new session resumes from `latest_checkpoint`.
- Eval: `run_eval(..., deadline=Deadline(minutes))` asks before each batch. A stopped run
  keeps the batches it stored, leaves official EX to the attempt that finishes it, says
  `stopped_time_guard` in its manifest and goes to its run store like a run that ended.
  It resumes like a failed one: the same `--run-id`, or `forge kaggle run --resume`.
  `forge eval run` and the kernel's entry script don't pass a deadline yet.
- A manifest's `gpu_hours` is its wall-clock time from `started_at` to `ended_at` on a
  platform with a GPU (`local-mps` too), and none on `local-cpu`. A resumed run keeps its
  first `started_at`, so its hours include the time between its sessions.

## Troubleshooting

- **`ModuleNotFoundError: forge` on macOS.** The venv can come out with the hidden file
  flag set, and Python 3.12 skips hidden `.pth` files, so the editable install vanishes
  after uv re-syncs. Every make target clears the flag first, and pytest puts `src` on the
  path itself. Before a bare `uv run forge ...`, run `make unhide` once after a sync.
