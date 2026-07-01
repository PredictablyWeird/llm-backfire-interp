#!/usr/bin/env bash
# GPU experiment: Plutchik emotion-wheel nudges vs assertiveness control.
#
#   bash lambda/run_plutchik_compliance.sh
#
# Env: MODEL, CATEGORY, MAX_EXAMPLES (default 500), BATCH
#      MODES (default all)  EMOTIONS (default all)  WITH_SMILEYS (default 0)
set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"

MODEL="${MODEL:-Qwen/Qwen3-32B}"
CATEGORY="${CATEGORY:-Gender_identity}"
MAX_EXAMPLES="${MAX_EXAMPLES:-500}"
BATCH="${BATCH:-16}"
MODES="${MODES:-all}"
EMOTIONS="${EMOTIONS:-all}"
WITH_SMILEYS="${WITH_SMILEYS:-0}"

run() { echo -e "\n\$ $*\n"; "$@"; }

COLLECT_ARGS=(
  --model "$MODEL" --category "$CATEGORY"
  --max-examples "$MAX_EXAMPLES" --batch-size "$BATCH"
  --modes "$MODES" --emotions "$EMOTIONS" --force
)
ANALYZE_ARGS=(
  --model "$MODEL" --category "$CATEGORY" --modes "$MODES" --emotions "$EMOTIONS"
)
if [ "$WITH_SMILEYS" = "1" ]; then
  COLLECT_ARGS+=(--with-smileys)
  ANALYZE_ARGS=(--model "$MODEL" --category "$CATEGORY" --modes all)
fi

echo "============ PLUTCHIK: collect (modes=$MODES emotions=$EMOTIONS, n=$MAX_EXAMPLES) ============"
run uv run --env-file .env python scripts/collect_plutchik_ladder.py "${COLLECT_ARGS[@]}"

echo "============ PLUTCHIK: analyze (vs assertiveness plain) ============"
run uv run python scripts/analyze_plutchik_compliance.py "${ANALYZE_ARGS[@]}"

echo "============ DONE ============"
