#!/usr/bin/env bash
# make smoke: the whole pipeline on CPU with the tiny model, on the fixture data. It runs
# the smoke tests, then `forge eval run` with the tiny model on the fixture's 20 val
# questions and `forge report run` on that run, and prints the report. Then `forge train
# sft` trains a LoRA adapter for 3 steps (configs/sft/tiny-smoke.yaml), and `forge eval
# run --adapter` evaluates it on 10 val questions. Last, `forge train grpo` trains 2 GRPO
# steps (configs/grpo/tiny-smoke.yaml), its rewards scored on the fixture databases and
# its in-training eval on val, `forge runs plot` draws its metrics, and `forge eval run
# --adapter` evaluates that adapter on 10 val questions too. All of it goes in a scratch
# folder removed at the end.
#
# When every step passes on a clean tree it writes .smoke-ok-<commit sha> at the repo root,
# which `forge kaggle run` requires before it pushes a GPU job (FG-303). The marker vouches
# for a commit, so uncommitted changes pass without one, and so does a smoke that skipped
# some phases' tests. Each attempt first drops the marker of the commit it runs on, so a
# marker always means the last smoke on that commit passed.
#
# Usage: scripts/smoke/pipeline.sh, or make smoke. PHASE=N runs the smoke tests of phases
# 0 to N only, as make gate PHASE=N does, and writes no marker unless N is the last phase.
set -euo pipefail

cd "$(dirname "$0")/../.."
last_phase=8 # the plan's last, as in the Makefile
phase="${PHASE:-$last_phase}"
sha="$(git rev-parse HEAD)"
run_id="smoke-${sha:0:7}"
sft_id="smoke-sft-${sha:0:7}"
grpo_id="smoke-grpo-${sha:0:7}"
rm -f ".smoke-ok-$sha"

scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
export FORGE_DATA_DIR="$PWD/tests/fixtures/spider_mini"
export FORGE_RUNS_DIR="$scratch/runs" FORGE_CACHE_DIR="$scratch/cache"

step() { printf '\n== %s\n' "$*"; }
tiny_model() {
    uv run python -c 'import yaml; tiny = yaml.safe_load(open("configs/models.yaml"))["tiny"]
print(tiny["repo"] + "@" + tiny["revision"])'
}
run_commit() {
    uv run python -c 'import json, sys; print(json.load(open(sys.argv[1]))["git_sha"])' "$1"
}

model="$(tiny_model)"

step "pytest tests/smoke"
uv run pytest --phase "$phase" tests/smoke

step "forge eval run --backend hf --model $model --split val --limit 20"
# Two SQL workers: 20 questions need no more, and a 2-core CI runner has no more to give.
uv run forge eval run --backend hf --model "$model" --split val --limit 20 \
    --run-id "$run_id" --workers 2 --allow-dirty

step "forge report run $run_id"
uv run forge report run "$run_id"
# Printed whole, as the gate's demo shows it, since the scratch folder goes at the end.
report="$FORGE_RUNS_DIR/$run_id/eval/val-report.md"
step "$report"
cat "$report"

step "forge train sft --config configs/sft/tiny-smoke.yaml"
uv run forge train sft --config configs/sft/tiny-smoke.yaml --run-id "$sft_id" \
    --sft-dir "$scratch/sft" --allow-dirty

adapter="$FORGE_RUNS_DIR/$sft_id/checkpoints/final"
step "forge eval run --adapter $adapter --split val --limit 10"
uv run forge eval run --backend hf --model "$model" --adapter "$adapter" --split val \
    --limit 10 --run-id "$sft_id-eval" --workers 2 --allow-dirty

step "forge train grpo --config configs/grpo/tiny-smoke.yaml"
uv run forge train grpo --config configs/grpo/tiny-smoke.yaml --run-id "$grpo_id" --allow-dirty

step "forge runs plot $grpo_id --store local"
uv run forge runs plot "$grpo_id" --store local

adapter="$FORGE_RUNS_DIR/$grpo_id/checkpoints/final"
step "forge eval run --adapter $adapter --split val --limit 10"
uv run forge eval run --backend hf --model "$model" --adapter "$adapter" --split val \
    --limit 10 --run-id "$grpo_id-eval" --workers 2 --allow-dirty

if [[ "$phase" != "$last_phase" ]]; then
    printf '\nsmoke passed with the smoke tests of phases 0 to %s only, so no marker: ' "$phase"
    printf 'run it with every phase (PHASE unset or %s)\n' "$last_phase"
    exit 0
fi

# The manifest names the commit the run was made at, with -dirty for uncommitted changes.
made_at="$(run_commit "$FORGE_RUNS_DIR/$run_id/manifest.json")"
if [[ "$made_at" != "$sha" ]]; then
    printf '\nsmoke passed on %s, not on commit %s, so no marker: commit, then run it again\n' \
        "$made_at" "$sha"
    exit 0
fi
# Only the commit the last smoke passed on keeps a marker.
rm -f .smoke-ok-*
printf 'make smoke passed on %s with run %s\n' "$sha" "$run_id" > ".smoke-ok-$sha"
printf '\nsmoke passed: wrote .smoke-ok-%s\n' "$sha"
