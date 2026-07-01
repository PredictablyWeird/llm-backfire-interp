#!/usr/bin/env bash
# GPU experiment: adjective ladder (pure + assertiveness combos) vs control.
#
#   bash lambda/run_adjective_compliance.sh
#
# Env: MODEL, CATEGORY, MAX_EXAMPLES (default 500), BATCH
#      MODES (default all)  SCALES (default all)  WITH_SMILEYS (default 0)
set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"

MODEL="${MODEL:-Qwen/Qwen3-32B}"
CATEGORY="${CATEGORY:-Gender_identity}"
MAX_EXAMPLES="${MAX_EXAMPLES:-500}"
BATCH="${BATCH:-16}"
MODES="${MODES:-all}"
SCALES="${SCALES:-all}"
WITH_SMILEYS="${WITH_SMILEYS:-0}"

run() { echo -e "\n\$ $*\n"; "$@"; }

COLLECT_ARGS=(
  --model "$MODEL" --category "$CATEGORY"
  --max-examples "$MAX_EXAMPLES" --batch-size "$BATCH"
  --modes "$MODES" --scales "$SCALES" --force
)
ANALYZE_ARGS=(
  --model "$MODEL" --category "$CATEGORY" --modes "$MODES" --scales "$SCALES"
)
if [ "$WITH_SMILEYS" = "1" ]; then
  COLLECT_ARGS+=(--with-smileys)
  ANALYZE_ARGS=(--model "$MODEL" --category "$CATEGORY" --modes all)
fi

echo "============ ADJECTIVE LADDER: collect (modes=$MODES scales=$SCALES, n=$MAX_EXAMPLES) ============"
run uv run --env-file .env python scripts/collect_adjective_ladder.py "${COLLECT_ARGS[@]}"

echo "============ ADJECTIVE LADDER: analyze (vs assertiveness plain) ============"
run uv run python scripts/analyze_adjective_compliance.py "${ANALYZE_ARGS[@]}"

echo "============ DONE ============"
