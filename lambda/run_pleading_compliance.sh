#!/usr/bin/env bash
# GPU experiment: pleading ladder vs assertiveness control (Gender default).
#
#   bash lambda/run_pleading_compliance.sh
#
# Env: MODEL, CATEGORY, MAX_EXAMPLES (default 500), BATCH, VARIANTS (default all)
#      WITH_SMILEYS (default 0)  SMILEY_VARIANTS (default all, when WITH_SMILEYS=1)
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-32B}"
CATEGORY="${CATEGORY:-Gender_identity}"
MAX_EXAMPLES="${MAX_EXAMPLES:-500}"
BATCH="${BATCH:-16}"
VARIANTS="${VARIANTS:-all}"
WITH_SMILEYS="${WITH_SMILEYS:-0}"
SMILEY_VARIANTS="${SMILEY_VARIANTS:-all}"

run() { echo -e "\n\$ $*\n"; "$@"; }

COLLECT_ARGS=(
  --model "$MODEL" --category "$CATEGORY"
  --max-examples "$MAX_EXAMPLES" --batch-size "$BATCH"
  --variants "$VARIANTS" --force
)
ANALYZE_ARGS=(
  --model "$MODEL" --category "$CATEGORY" --variants "$VARIANTS"
)
if [ "$WITH_SMILEYS" = "1" ]; then
  COLLECT_ARGS+=(--with-smileys --smiley-variants "$SMILEY_VARIANTS")
  ANALYZE_ARGS+=(--with-smileys --smiley-variants "$SMILEY_VARIANTS")
fi

echo "============ PLEADING LADDER: collect (variants=$VARIANTS, n=$MAX_EXAMPLES) ============"
run uv run --env-file .env python scripts/collect_pleading_ladder.py "${COLLECT_ARGS[@]}"

echo "============ PLEADING LADDER: analyze (vs assertiveness plain) ============"
run uv run python scripts/analyze_pleading_compliance.py "${ANALYZE_ARGS[@]}"

echo "============ DONE ============"
