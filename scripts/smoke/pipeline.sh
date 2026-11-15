#!/usr/bin/env bash
# make smoke: the whole eval pipeline on CPU with the tiny model, on the fixture data. It
# runs the smoke tests, then `forge eval run` with the tiny model on the fixture's 20 val
# questions and `forge report run` on that run, in a scratch runs folder removed at the end.
#
# When every step passes on a clean tree it writes .smoke-ok-<commit sha> at the repo root,
# which `forge kaggle run` requires before it pushes a GPU job (FG-303). The marker vouches
# for a commit, so uncommitted changes pass without one. Each attempt first drops the marker
# of the commit it runs on, so a marker always means the last smoke on that commit passed.
#
# Usage: scripts/smoke/pipeline.sh, or make smoke. PHASE=N runs the smoke tests of phases
# 0 to N only, as make gate PHASE=N does.
set -euo pipefail

cd "$(dirname "$0")/../.."
sha="$(git rev-parse HEAD)"
run_id="smoke-${sha:0:7}"
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
uv run pytest --phase "${PHASE:-8}" tests/smoke

step "forge eval run --backend hf --model $model --split val --limit 20"
# Two SQL workers: 20 questions need no more, and a 2-core CI runner has no more to give.
uv run forge eval run --backend hf --model "$model" --split val --limit 20 \
    --run-id "$run_id" --workers 2 --allow-dirty

step "forge report run $run_id"
uv run forge report run "$run_id"

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
