#!/usr/bin/env bash
# GPU-session orchestrator: run every model-dependent phase back-to-back, then
# (optionally) self-terminate so you never pay for idle time after completion.
#
# Run it detached so an SSH drop doesn't kill it:
#   tmux new -s run 'bash lambda/run_all.sh 2>&1 | tee run.log'
#
# Env knobs:
#   MODEL        (default Qwen/Qwen3-32B)
#   CATEGORIES   (default all)         space-separated, or "all"
#   NUDGES       (default all)         space-separated, or "all"
#   COMP_CATS    (default Gender_identity)  categories to capture mlp/attn for
#   BATCH        (default 16)
#   PATCH_LAYERS (default "0 8 12 13 14 15")  layers for true causal patching
#   AUTOSTOP     (default 0)           set to 1 to `shutdown -h now` at the end
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-32B}"
CATEGORIES="${CATEGORIES:-all}"
NUDGES="${NUDGES:-all}"
COMP_CATS="${COMP_CATS:-Gender_identity}"
BATCH="${BATCH:-16}"
PATCH_LAYERS="${PATCH_LAYERS:-0 8 12 13 14 15}"
AUTOSTOP="${AUTOSTOP:-0}"

run() { echo -e "\n\$ $*\n"; "$@"; }

echo "============ PHASE 0: smoke test ============"
run uv run --env-file .env python scripts/collect_cache.py \
    --model "$MODEL" --smoke --batch-size 4 \
    --categories Gender_identity --nudges user_preference

echo "============ PHASE 1: collect all caches (GPU) ============"
run uv run --env-file .env python scripts/collect_cache.py \
    --model "$MODEL" --categories $CATEGORIES --nudges $NUDGES \
    --components-categories $COMP_CATS --batch-size "$BATCH"

echo "============ PHASE 2: live experiments (GPU) ============"
run uv run --env-file .env python scripts/run_live_experiments.py \
    --model "$MODEL" --category Gender_identity --nudge user_preference \
    --mode causal_patch --component mlp --layers $PATCH_LAYERS --batch-size "$BATCH"
run uv run --env-file .env python scripts/run_live_experiments.py \
    --model "$MODEL" --category Gender_identity --nudge user_preference \
    --mode causal_patch --component attn --layers $PATCH_LAYERS --batch-size "$BATCH"
run uv run --env-file .env python scripts/run_live_experiments.py \
    --model "$MODEL" --category Gender_identity --nudge user_preference \
    --mode token_sweep --batch-size "$BATCH"

echo "============ PHASE 3: cached-tensor analysis (CPU; safe to run here too) ============"
# Analyze every category that has a user_preference cache.
for cat in Age Disability_status Gender_identity Nationality Physical_appearance \
           Race_ethnicity Religion SES Sexual_orientation; do
  uv run python scripts/analyze.py --model "$MODEL" --category "$cat" \
      --nudge user_preference || echo "  (skip $cat — no cache)"
done

echo "============ DONE ============"
echo "Caches + results are on the persistent volume."

if [ "$AUTOSTOP" = "1" ]; then
  echo "AUTOSTOP=1 → shutting down in 60s. Ctrl-C to cancel."
  sleep 60
  sudo shutdown -h now
fi
