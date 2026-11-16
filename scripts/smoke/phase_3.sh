#!/usr/bin/env bash
# Integration checkpoint 3 (docs/PLAN.md): a 50-question eval job on a Kaggle T4, about 10
# minutes of GPU, then the run pulled back with its ledger row and its report rebuilt from
# its files. The job goes through forge kaggle run like every GPU job, so HEAD must be
# clean and have passed make smoke, and its estimate must fit the week's budget. Any step
# that fails stops the script with a non-zero exit.
#
# Usage: scripts/smoke/phase_3.sh [run_id]
# The run id defaults to <date>-smoke-kaggle-50-val-<sha7>; pass another to run it again on
# the same day. Without Kaggle credentials and FORGE_KAGGLE_USER (docs/runbook.md,
# "Accounts") it says what is missing and exits 2 before anything is pushed. The run stays
# in runs/ and the ledger keeps its row, since the GPU hours are real. FORGE_DATA_DIR picks
# the data the report is rebuilt on, by default the full download in data/spider.
set -euo pipefail

cd "$(dirname "$0")/../.."
config=configs/eval/smoke-kaggle-50-val.yaml
sha="$(git rev-parse HEAD)"
run_id="${1:-$(date -u +%Y-%m-%d)-smoke-kaggle-50-val-${sha:0:7}}"
export FORGE_DATA_DIR="${FORGE_DATA_DIR:-$PWD/data/spider}"

step() { printf '\n== %s\n' "$*"; }

step "Kaggle credentials"
# Read where they are, never their values, and whose datasets the kernel mounts.
if ! uv run python -c '
import sys
from forge.ops.doctor import KAGGLE_HINT, Context, kaggle_credential_source
from forge.ops.kaggle import KaggleError, kaggle_owner
ctx = Context()
source = kaggle_credential_source(ctx)
if source is None:
    sys.exit(f"no Kaggle credentials: {KAGGLE_HINT}")
try:
    owner = kaggle_owner(ctx.env, ctx.home)
except KaggleError as exc:
    sys.exit(str(exc))
print(f"credentials in {source}, datasets of {owner}")
'; then
    echo "error: the phase 3 smoke runs on Kaggle, so nothing was pushed" >&2
    exit 2
fi

step "forge kaggle run --job eval --config $config"
uv run forge kaggle run --job eval --config "$config" --run-id "$run_id" --yes

step "forge kaggle wait $run_id"
uv run forge kaggle wait "$run_id"

step "forge runs pull $run_id --store kaggle"
uv run forge runs pull "$run_id" --store kaggle

step "forge report rebuild $run_id"
uv run forge report rebuild "$run_id"

printf '\nphase 3 smoke passed: run %s\n' "$run_id"
