#!/usr/bin/env bash
# One-time environment setup on a fresh Lambda Cloud GPU instance.
#
# Goal: get from a bare box to "ready to run experiments" with the model weights
# and HF cache living on a PERSISTENT volume, so re-launches skip the 60+GB
# download and you never pay for idle download time again.
#
# Usage:
#   export HF_TOKEN=hf_xxx
#   export PERSIST=/home/ubuntu/persist          # a persistent/attached volume mount
#   export MODEL=Qwen/Qwen3-32B
#   bash lambda/setup.sh
set -euo pipefail

PERSIST="${PERSIST:-/home/ubuntu/persist}"
MODEL="${MODEL:-Qwen/Qwen3-32B}"

echo ">>> Persistent dir: $PERSIST"
mkdir -p "$PERSIST/hf_cache" "$PERSIST/cache" "$PERSIST/results"

# Keep the big HF download on the persistent volume.
export HF_HOME="$PERSIST/hf_cache"
echo "export HF_HOME=$PERSIST/hf_cache" >> ~/.bashrc

# Symlink project cache/results onto the persistent volume so outputs survive teardown.
ln -sfn "$PERSIST/cache"   "$(pwd)/cache"
ln -sfn "$PERSIST/results" "$(pwd)/results"

# Install uv (fast, reproducible) and sync the locked environment.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# uv installs to ~/.local/bin; ensure it's on PATH for this (non-login) shell.
export PATH="$HOME/.local/bin:$PATH"
uv sync

# Write the env the run scripts need. They load this via `uv run --env-file .env`,
# so putting HF_HOME here guarantees they use the persistent cache regardless of
# which shell they run in (no accidental re-download to the ephemeral root volume).
{
  echo "HF_HOME=$HF_HOME"
  [ -n "${HF_TOKEN:-}" ] && echo "HF_TOKEN=$HF_TOKEN"
} > .env

# HF auth (needed for gated Llama; Qwen is open but a token raises rate limits).
# NOTE: the old `huggingface-cli` is removed in recent huggingface_hub — use `hf`.
if [ -n "${HF_TOKEN:-}" ]; then
  uv run hf auth login --token "$HF_TOKEN" || true
fi

# Pre-download weights to the persistent HF cache (so the run itself is GPU-bound only).
echo ">>> Pre-downloading $MODEL weights to $HF_HOME ..."
uv run hf download "$MODEL"

echo ">>> Setup complete. Weights + caches live on $PERSIST."
echo ">>> Next: bash lambda/run_all.sh"
