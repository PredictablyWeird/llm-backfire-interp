#!/usr/bin/env bash
# GPU: reason-before-answer nudge sensitivity (ladder + optional rep) with reasoning text.
#
# Fast defaults (~hours not days on H100):
#   CATEGORIES=Gender_identity  MAX_EXAMPLES=500  MAX_REASONING=128
#   LADDER_ONLY=1  GEN_BATCH=8
#
# Run detached:
#   tmux new -s reason 'bash lambda/run_sensitivity_reasoning.sh 2>&1 | tee reason.log'
#
# Env knobs:
#   MODEL           (default Qwen/Qwen3-32B)
#   CATEGORIES      (default Gender_identity)
#   MAX_EXAMPLES    (default 500)
#   BATCH           (default 8)   batch size for A/B/C logit scoring
#   GEN_BATCH       (default 8)   batch size for reasoning generation
#   MAX_REASONING   (default 128) max new tokens for reasoning
#   LADDER_ONLY     (default 1)   set to 0 to also collect rep axis
#   AUTOSTOP          (default 0)
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-32B}"
CATEGORIES="${CATEGORIES:-Gender_identity}"
MAX_EXAMPLES="${MAX_EXAMPLES:-500}"
BATCH="${BATCH:-8}"
GEN_BATCH="${GEN_BATCH:-8}"
MAX_REASONING="${MAX_REASONING:-128}"
LADDER_ONLY="${LADDER_ONLY:-1}"
AUTOSTOP="${AUTOSTOP:-0}"

REP_FLAG=()
if [ "$LADDER_ONLY" = "0" ]; then
  REP_FLAG=(--include-rep)
fi

run() { echo -e "\n\$ $*\n"; "$@"; }

echo "============ REASON-BEFORE-ANSWER: smoke test ============"
run uv run --env-file .env python scripts/nudge_sensitivity_reasoning.py \
    --model "$MODEL" --category Gender_identity --max-examples 4 \
    --batch-size 2 --gen-batch-size 2 --max-reasoning-tokens 64 \
    "${REP_FLAG[@]}"

echo "============ REASON-BEFORE-ANSWER: full collect ============"
echo "config: categories=$CATEGORIES max_examples=$MAX_EXAMPLES max_reasoning=$MAX_REASONING gen_batch=$GEN_BATCH ladder_only=$LADDER_ONLY"
for cat in $CATEGORIES; do
  run uv run --env-file .env python scripts/nudge_sensitivity_reasoning.py \
      --model "$MODEL" --category "$cat" \
      --max-examples "$MAX_EXAMPLES" \
      --batch-size "$BATCH" --gen-batch-size "$GEN_BATCH" \
      --max-reasoning-tokens "$MAX_REASONING" \
      "${REP_FLAG[@]}"
  run uv run python scripts/analyze_sensitivity_reasoning.py \
      --model "$MODEL" --category "$cat"
done

echo "============ DONE ============"
echo "Caches: cache/<model>/sensitivity_reasoning_<Category>.npz"
echo "Analysis: results/sensitivity_reasoning_<Category>.json"

if [ "$AUTOSTOP" = "1" ]; then
  echo "AUTOSTOP=1 → shutting down in 60s. Ctrl-C to cancel."
  sleep 60
  sudo shutdown -h now
fi
