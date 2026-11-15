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

Added in FG-303.

## Resume after a session limit

Added in FG-305.

## Troubleshooting

- **`ModuleNotFoundError: forge` on macOS.** The venv can come out with the hidden file
  flag set, and Python 3.12 skips hidden `.pth` files, so the editable install vanishes
  after uv re-syncs. Every make target clears the flag first, and pytest puts `src` on the
  path itself. Before a bare `uv run forge ...`, run `make unhide` once after a sync.
