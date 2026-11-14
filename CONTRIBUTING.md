# Contributing to Forge

Forge post-trains a small LLM with GRPO on text-to-SQL, on free GPUs. The plan is
[docs/PLAN.md](docs/PLAN.md). Work one story at a time; the story block is the spec.

## Commands

| Command | What it does |
|---|---|
| `make setup` | Install the dev tools and the CPU extra with uv |
| `make check` | Lint, types, unit and integration tests |
| `make smoke` | Full pipeline on CPU with a tiny model (must pass before any GPU job) |
| `make gate PHASE=N` | What a phase gate runs |
| `uv run forge --help` | CLI: data, sql, eval, train, kaggle, runs, ledger, report, export |

## Rules

1. Branch from `dev`, open a PR into `dev`, squash merge. Never push to `dev` or `main` directly.
2. Tests first. Every test has `@pytest.mark.phase(n)`.
3. Never delete, skip or weaken a test to get green. Stop and explain instead.
4. Don't change `src/forge/core/*` unless the story says so.
5. Training code never loads the val or dev split. Use `forge.data.splits.load_train()`.
6. The dev split is only evaluated with `--reason`, and only in stories that say so.
7. Reported numbers come from generated reports, never typed by hand.
8. GPU jobs only in stories marked "GPU job", only after `make smoke` passes, only through `forge kaggle run`.
9. Every SQL execution goes through `forge.sql.executor`. Never open a Spider database in write mode.
10. No secrets in code, configs, notebooks or logs. Tokens come from the environment or Kaggle Secrets.
11. Update `docs/progress.md` in the same PR.

## Style

- Type hints everywhere; mypy is strict in `core`, `sql`, `eval` and `train/rewards`.
- Comments explain why. No banners, no emoji.
- Commits follow conventional commits: imperative, lowercase, at most 72 characters in the
  subject, and `Refs: FG-xxx` in the body.

## Stop and ask when

- A dependency story isn't merged, or acceptance criteria contradict a contract.
- A GPU job fails twice, or would exceed its estimate by 50%.
- The work would need money, a new account, or a model with a gated license.
