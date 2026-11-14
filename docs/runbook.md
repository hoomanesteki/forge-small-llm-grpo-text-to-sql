# Runbook

Operational notes for running Forge. Sections are filled in by the stories that need them.

## Accounts

Everything is free. Set these up once, then run `uv run forge doctor` until every line says `ok`.

1. **Kaggle (GPU runner).**
   - Create an account and **phone-verify it** under Settings. Without that, notebooks get
     no GPU and no internet.
   - Under Settings, API, create an API token. Save it to `~/.kaggle/access_token` and run
     `chmod 600 ~/.kaggle/access_token`. A legacy `~/.kaggle/kaggle.json` also works.
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

## Kaggle jobs

Added in FG-303.

## Resume after a session limit

Added in FG-305.

## Troubleshooting

- **`ModuleNotFoundError: forge` on macOS.** The venv can come out with the hidden file
  flag set, and Python 3.12 skips hidden `.pth` files, so the editable install vanishes
  after uv re-syncs. Every make target clears the flag first, and pytest puts `src` on the
  path itself. Before a bare `uv run forge ...`, run `make unhide` once after a sync.
