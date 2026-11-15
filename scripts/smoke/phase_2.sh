#!/usr/bin/env bash
# Integration checkpoint 2 (docs/PLAN.md): the phase 2 demo. The fake backend answers val
# with the gold queries and must score 100%, the tiny model answers the same questions, the
# compare report pairs the two runs, and both runs' reports must rebuild from their stored
# files without a diff. Any step that fails stops the script with a non-zero exit.
#
# Usage: scripts/smoke/phase_2.sh
# Both runs answer the first 20 val questions by id, so the compare report can pair them.
# FORGE_DATA_DIR picks the data, by default the full download in data/spider (phase_1.sh
# fetches it); the runs go to a scratch folder that is removed at the end.
set -euo pipefail

cd "$(dirname "$0")/../.."
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
export FORGE_RUNS_DIR="$scratch/runs" FORGE_CACHE_DIR="$scratch/cache"

step() { printf '\n== %s\n' "$*"; }
tiny_model() {
    uv run python -c 'import yaml; tiny = yaml.safe_load(open("configs/models.yaml"))["tiny"]
print(tiny["repo"] + "@" + tiny["revision"])'
}

step "forge eval run --backend fake --model gold --split val --limit 20"
uv run forge eval run --backend fake --model gold --split val --limit 20 --run-id gold \
    --allow-dirty

step "forge report run gold"
uv run forge report run gold
title="$(head -n 1 "$FORGE_RUNS_DIR/gold/eval/val-report.md")"
echo "$title"
if [[ "$title" != "# Run gold: official EX 100.0% [100.0%, 100.0%] on val" ]]; then
    echo "error: the gold queries scored below 100% official EX" >&2
    exit 1
fi

model="$(tiny_model)"
step "forge eval run --backend hf --model $model --split val --limit 20"
uv run forge eval run --backend hf --model "$model" --split val --limit 20 --run-id tiny \
    --workers 2 --allow-dirty

step "forge report run tiny"
uv run forge report run tiny

step "forge report compare gold tiny"
uv run forge report compare gold tiny
cat "$FORGE_RUNS_DIR/tiny/eval/compare-gold.md"

# Rebuilding the tiny run checks its compare report too, which is in its folder.
step "forge report rebuild tiny"
uv run forge report rebuild tiny

step "forge report rebuild gold"
uv run forge report rebuild gold

printf '\nphase 2 smoke passed\n'
