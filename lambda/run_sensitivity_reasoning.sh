#!/usr/bin/env bash
# GPU: collect reason-before-answer nudge sensitivity (ladder + rep) with reasoning text.
#
# Run detached:
#   tmux new -s reason 'bash lambda/run_sensitivity_reasoning.sh 2>&1 | tee reason.log'
#
# Env knobs:
#   MODEL              (default Qwen/Qwen3-32B)
#   CATEGORIES         (default "Gender_identity SES Race_ethnicity")
#   BATCH              (default 8)   batch size for A/B/C logit scoring
#   GEN_BATCH          (default 4)   batch size for reasoning generation
#   MAX_REASONING      (default 512) max new tokens for reasoning
#   AUTOSTOP           (default 0)
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-32B}"
CATEGORIES="${CATEGORIES:-Gender_identity SES Race_ethnicity}"
BATCH="${BATCH:-8}"
GEN_BATCH="${GEN_BATCH:-4}"
MAX_REASONING="${MAX_REASONING:-512}"
AUTOSTOP="${AUTOSTOP:-0}"

run() { echo -e "\n\$ $*\n"; "$@"; }

echo "============ REASON-BEFORE-ANSWER: smoke test ============"
run uv run --env-file .env python scripts/nudge_sensitivity_reasoning.py \
    --model "$MODEL" --category Gender_identity --max-examples 4 \
    --batch-size 2 --gen-batch-size 2 --max-reasoning-tokens 64

echo "============ REASON-BEFORE-ANSWER: full collect ============"
for cat in $CATEGORIES; do
  run uv run --env-file .env python scripts/nudge_sensitivity_reasoning.py \
      --model "$MODEL" --category "$cat" \
      --batch-size "$BATCH" --gen-batch-size "$GEN_BATCH" \
      --max-reasoning-tokens "$MAX_REASONING"
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
