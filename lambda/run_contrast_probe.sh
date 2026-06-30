#!/usr/bin/env bash
# GPU: binary contrast probe (S vs U pair).
#
#   EXPERIMENT=binary bash lambda/run_contrast_probe.sh   # default
#   EXPERIMENT=threeway bash lambda/run_contrast_probe.sh
#
set -euo pipefail

EXPERIMENT="${EXPERIMENT:-binary}"

MODEL="${MODEL:-Qwen/Qwen3-32B}"
CATEGORY="${CATEGORY:-Gender_identity}"
MAX_EXAMPLES="${MAX_EXAMPLES:-500}"
BATCH="${BATCH:-16}"
ALL_LAYERS="${ALL_LAYERS:-0}"
WITH_NUDGE="${WITH_NUDGE:-0}"
REASON_BEFORE_ANSWER="${REASON_BEFORE_ANSWER:-0}"
WITH_REASONING_INSTR="${WITH_REASONING_INSTR:-0}"
MAX_REASONING="${MAX_REASONING:-128}"
GEN_BATCH="${GEN_BATCH:-8}"
PROBE="${PROBE:-both}"
MLP_HIDDEN="${MLP_HIDDEN:-64}"

if [ "$EXPERIMENT" = "binary" ]; then
  COLLECT=scripts/collect_contrast_probe_binary.py
  ANALYZE=scripts/analyze_contrast_probe_binary.py
elif [ "$EXPERIMENT" = "threeway" ]; then
  COLLECT=scripts/collect_contrast_probe_threeway.py
  ANALYZE=scripts/analyze_contrast_probe_threeway.py
else
  echo "EXPERIMENT must be 'binary' or 'threeway' (got: $EXPERIMENT)" >&2
  exit 1
fi

LAYER_FLAG=()
[ "$ALL_LAYERS" = "1" ] && LAYER_FLAG=(--all-layers)

NUDGE_FLAG=()
ANALYZE_NUDGE_FLAG=()
[ "$WITH_NUDGE" = "1" ] && NUDGE_FLAG=(--with-nudge) && ANALYZE_NUDGE_FLAG=(--with-nudge)

REASON_FLAG=()
ANALYZE_REASON_FLAG=()
if [ "$REASON_BEFORE_ANSWER" = "1" ]; then
  REASON_FLAG=(--reason-before-answer --max-reasoning-tokens "$MAX_REASONING" --gen-batch-size "$GEN_BATCH")
  ANALYZE_REASON_FLAG=(--reason-before-answer)
fi

INSTR_FLAG=()
[ "$WITH_REASONING_INSTR" = "1" ] && INSTR_FLAG=(--with-reasoning-instruction)

run() { echo -e "\n\$ $*\n"; "$@"; }

echo "============ CONTRAST PROBE ($EXPERIMENT): smoke test ============"
run uv run --env-file .env python "$COLLECT" \
    --model "$MODEL" --category Gender_identity --max-examples 4 \
    --batch-size 2 --force \
    "${NUDGE_FLAG[@]}" \
    "${REASON_FLAG[@]}" \
    "${INSTR_FLAG[@]}"

echo "============ CONTRAST PROBE ($EXPERIMENT): full collect ============"
run uv run --env-file .env python "$COLLECT" \
    --model "$MODEL" --category "$CATEGORY" \
    --max-examples "$MAX_EXAMPLES" \
    --batch-size "$BATCH" \
    "${LAYER_FLAG[@]}" \
    "${NUDGE_FLAG[@]}" \
    "${REASON_FLAG[@]}" \
    "${INSTR_FLAG[@]}" \
    --force

ANALYZE_EXTRA=()
[ "$EXPERIMENT" = "binary" ] && [ "$WITH_NUDGE" = "1" ] && ANALYZE_EXTRA=(--compare-sensitivity)
[ "$EXPERIMENT" = "threeway" ] && ANALYZE_EXTRA=(--probe "$PROBE" --hidden "$MLP_HIDDEN")

run uv run python "$ANALYZE" \
    --model "$MODEL" --category "$CATEGORY" \
    "${ANALYZE_NUDGE_FLAG[@]}" \
    "${ANALYZE_REASON_FLAG[@]}" \
    "${ANALYZE_EXTRA[@]}"

echo "============ DONE ($EXPERIMENT) ============"
stem="contrast_probe_${EXPERIMENT}_${CATEGORY}"
[ "$REASON_BEFORE_ANSWER" = "1" ] && stem="contrast_probe_${EXPERIMENT}_reasoning_${CATEGORY}"
[ "$WITH_NUDGE" = "1" ] && stem="${stem/_${CATEGORY}/_nudge_${CATEGORY}}"
# Note: full stem logic matches cache_stem(); see collect output path.
echo "See collect script output above for exact cache path."
