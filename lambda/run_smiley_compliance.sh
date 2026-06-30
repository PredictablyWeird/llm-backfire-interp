#!/usr/bin/env bash
# Small GPU experiment: plain vs smiley assertiveness ladder (stereo direction).
#
# Default: Gender_identity, n=500, collects BOTH plain+smiley matched on same examples.
#
#   bash lambda/run_smiley_compliance.sh
#
# Env: MODEL, CATEGORY, MAX_EXAMPLES (default 500), BATCH
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-32B}"
CATEGORY="${CATEGORY:-Gender_identity}"
MAX_EXAMPLES="${MAX_EXAMPLES:-500}"
BATCH="${BATCH:-16}"

run() { echo -e "\n\$ $*\n"; "$@"; }

echo "============ SMILEY LADDER: collect (plain + smiley, n=$MAX_EXAMPLES) ============"
run uv run --env-file .env python scripts/collect_smiley_ladder.py \
    --model "$MODEL" --category "$CATEGORY" \
    --max-examples "$MAX_EXAMPLES" --batch-size "$BATCH" \
    --also-plain --force

echo "============ SMILEY LADDER: analyze ============"
run uv run python scripts/analyze_smiley_compliance.py \
    --model "$MODEL" --category "$CATEGORY"

echo "============ DONE ============"
