#!/usr/bin/env bash
# Integration checkpoint 1 (docs/PLAN.md): the phase 1 demo, run on the full Spider data.
# Fetch, stats, gold scored against itself on val, the denied DELETE, and a 500-item
# agreement check. Any step that fails stops the script with a non-zero exit.
#
# Usage: scripts/smoke/phase_1.sh [spider.zip]
# The zip defaults to $SPIDER_ZIP, then data/download/data/spider.zip; without one the
# pinned zip is downloaded. FORGE_DATA_DIR, if set, must point at the full data.
set -euo pipefail

cd "$(dirname "$0")/../.."
zip="${1:-${SPIDER_ZIP:-data/download/data/spider.zip}}"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

step() { printf '\n== %s\n' "$*"; }

if [[ -f "$zip" ]]; then
    step "forge data fetch --from-zip $zip"
    uv run forge data fetch --from-zip "$zip"
else
    step "forge data fetch"
    uv run forge data fetch
fi

step "forge data stats"
uv run forge data stats

step "forge eval gold --split val"
uv run forge eval gold --split val

step 'forge sql run --db concert_singer "DELETE FROM singer"'
if out="$(uv run forge sql run --db concert_singer "DELETE FROM singer")"; then
    echo "error: the DELETE ran: $out" >&2
    exit 1
fi
echo "$out"
if [[ "$out" != denied:* ]]; then
    echo "error: the DELETE failed, but was not denied" >&2
    exit 1
fi

step "forge eval agreement --mutations 500"
# A scratch report, so today's committed one in docs/audit is left as it is.
uv run forge eval agreement --mutations 500 --out "$scratch/agreement.md"

printf '\nphase 1 smoke passed\n'
