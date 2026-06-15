#!/usr/bin/env bash
# Collect prompt-region attention for Gender, SES, Race on Qwen3-32B, then analyze.
#
# Run detached on Lambda:
#   tmux new -s attn 'bash lambda/run_prompt_attention.sh 2>&1 | tee prompt_attn.log'
#
# Env knobs:
#   MODEL          (default Qwen/Qwen3-32B)
#   BATCH          (default 2)     attention forwards are memory-heavy; use 1 if OOM
#   CATEGORIES     (default "Gender_identity SES Race_ethnicity")
#   MAX_EXAMPLES   (default unset = full dataset)
#   STRATIFIED     (default unset)  if set, N compliers + N non-compliers per category
#   ATTN_LAYERS    (default unset = all 64 layers; e.g. "48 52 56 59 63" if OOM)
#   SMOKE          (default 0)      set to 1 to run 16-example smoke test first
#   FORCE          (default 0)      set to 1 to pass --force to collector
#   AUTOSTOP       (default 0)      set to 1 to shutdown after completion
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen3-32B}"
BATCH="${BATCH:-2}"
CATEGORIES="${CATEGORIES:-Gender_identity SES Race_ethnicity}"
SMOKE="${SMOKE:-0}"
FORCE="${FORCE:-0}"
AUTOSTOP="${AUTOSTOP:-0}"

run() { echo -e "\n\$ $*\n"; "$@"; }

COLLECT_ARGS=(--model "$MODEL" --batch-size "$BATCH" --categories $CATEGORIES)
[ -n "${MAX_EXAMPLES:-}" ] && COLLECT_ARGS+=(--max-examples "$MAX_EXAMPLES")
[ -n "${STRATIFIED:-}" ] && COLLECT_ARGS+=(--stratified "$STRATIFIED")
[ -n "${ATTN_LAYERS:-}" ] && COLLECT_ARGS+=(--layers $ATTN_LAYERS)
[ "$FORCE" = "1" ] && COLLECT_ARGS+=(--force)

if [ "$SMOKE" = "1" ]; then
  echo "============ SMOKE: 16 examples, Gender only ============"
  run uv run --env-file .env python scripts/collect_prompt_attention.py \
      "${COLLECT_ARGS[@]}" --max-examples 16 --categories Gender_identity --force
fi

echo "============ COLLECT: prompt-region attention (GPU) ============"
run uv run --env-file .env python scripts/collect_prompt_attention.py "${COLLECT_ARGS[@]}"

echo "============ ANALYZE: CPU ============"
run uv run python scripts/analyze_prompt_attention.py \
    --model "$MODEL" --categories $CATEGORIES

echo "============ DONE ============"
echo "Caches: cache/$(echo "$MODEL" | tr '/' '_')/prompt_attn_*.npz"
echo "Results: results/prompt_attention_*.json/png, results/prompt_attention_3cats_summary.json"

if [ "$AUTOSTOP" = "1" ]; then
  echo "AUTOSTOP=1 → shutting down in 60s. Ctrl-C to cancel."
  sleep 60
  sudo shutdown -h now
fi
