#!/usr/bin/env bash
# GPU experiment: plain vs emoji-intensity smiley ladders (stereo direction).
#
# Default: Gender_identity, n=500, all emoji profiles except re-collecting plain if present.
#
#   bash lambda/run_smiley_compliance.sh
#
# Env:
#   MODEL, CATEGORY, MAX_EXAMPLES (default 500), BATCH
#   VARIANTS     (default "subtle,warm,enthusiastic,pressured,intense")
#   SKIP_PLAIN   (default 1)  reuse existing sensitivity_smiley_plain_* cache
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-32B}"
CATEGORY="${CATEGORY:-Gender_identity}"
MAX_EXAMPLES="${MAX_EXAMPLES:-500}"
BATCH="${BATCH:-16}"
VARIANTS="${VARIANTS:-subtle,warm,enthusiastic,pressured,intense}"
SKIP_PLAIN="${SKIP_PLAIN:-1}"

run() { echo -e "\n\$ $*\n"; "$@"; }

COLLECT_ARGS=(
  --model "$MODEL" --category "$CATEGORY"
  --max-examples "$MAX_EXAMPLES" --batch-size "$BATCH"
  --variants "$VARIANTS" --force
)
PLAIN_ARGS=()
if [ "$SKIP_PLAIN" = "1" ]; then
  PLAIN_ARGS=(--skip-plain)
else
  PLAIN_ARGS=(--also-plain)
fi

echo "============ SMILEY LADDER: collect (variants=$VARIANTS, n=$MAX_EXAMPLES) ============"
run uv run --env-file .env python scripts/collect_smiley_ladder.py \
    "${COLLECT_ARGS[@]}" "${PLAIN_ARGS[@]}"

echo "============ SMILEY LADDER: analyze (include friendly if cached) ============"
run uv run python scripts/analyze_smiley_compliance.py \
    --model "$MODEL" --category "$CATEGORY" \
    --variants "friendly,${VARIANTS}"

echo "============ DONE ============"
