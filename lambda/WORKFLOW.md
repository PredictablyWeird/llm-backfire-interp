# GPU Workflow — scaling the BBQ backfire experiments to Qwen3-32B on Lambda

The guiding principle: **the GPU only produces caches.** Almost every analysis runs
on cached tensors, so we compress all GPU-bound work into one saturated session and
do the rest for free afterwards.

```
        ┌──────────────── GPU box (expensive, $/hr) ─────────────────┐
            collect_cache.py        → logits + per-layer resid (+ mlp/attn)
            run_live_experiments.py → true causal patching, token sweeps
        └─────────────────────────────────────────────────────────────┘
                                   │  caches on persistent volume
                                   ▼
        ┌──────────── laptop / cheap CPU (free) ─────────────────────┐
            analyze.py → backfire counts, margins, flip-rate,
                         ablation / residual / component patch sweeps,
                         DLA + logit lens
        └─────────────────────────────────────────────────────────────┘
```

## Phases

| Phase | Script | Needs GPU? | Output |
|------|--------|:---------:|--------|
| 1. Collect | `scripts/collect_cache.py` | ✅ | `cache/<model>/*_logits.npz`, `*_acts.npz`, `*_components.npz`, `unembed.npz` |
| 2. Live | `scripts/run_live_experiments.py` | ✅ | `results/causalpatch_*.json`, `results/tokensweep_*.json` |
| 3. Analyze | `scripts/analyze.py` | ❌ | `results/analysis_*.{json,md}` |

Caches are organized per model: every file for a run lives under
`cache/<model_slug>/` (e.g. `cache/Qwen_Qwen3-32B/`, `cache/meta-llama_Llama-3.2-1B/`),
so different models never collide. The scripts pick the subdir automatically from
`--model` and fall back to the flat `cache/` layout for any legacy files.

The two new library modules:
- `src/mech_interp_bbq/prompts.py` — shared prompt construction (identical across phases).
- `src/mech_interp_bbq/hf_backend.py` — HF loader + hook-based activation capture that
  shards across GPUs (`device_map="auto"`) and scales to 32B. Saves only the A/B/C
  unembed columns + final-norm params so Phase 3 needs no GPU and no full `lm_head`.

## Step 0 — validate locally on the small model (do this BEFORE renting a GPU)

```bash
# CPU/MPS, tiny end-to-end pass. Reproduces the existing Llama-3.2-1B cache format.
uv run --env-file .env python scripts/collect_cache.py \
    --model meta-llama/Llama-3.2-1B --device-map none --dtype float32 --smoke

uv run python scripts/analyze.py --model meta-llama/Llama-3.2-1B \
    --category Gender_identity --nudge user_preference
```

Because the cache format matches your original `run_backfire_3choice.py`, all of your
existing analysis scripts keep working on these caches too.

## Step 1 — provision Lambda + persistent storage

1. Launch a GPU instance large enough for 32B (1× H100/A100 80GB minimum; an 8× box
   shards comfortably via `device_map="auto"`).
2. Attach a **persistent filesystem** and mount it (e.g. `/home/ubuntu/persist`).
   Model weights and all caches live here so re-launches skip the ~64GB download.

```bash
git clone <your repo> && cd mech_interp
export HF_TOKEN=hf_xxx
export PERSIST=/home/ubuntu/persist
export MODEL=Qwen/Qwen3-32B
bash lambda/setup.sh        # installs uv, syncs env, pre-downloads weights to PERSIST
```

## Step 2 — run everything detached, auto-stop when done

```bash
tmux new -s run 'MODEL=Qwen/Qwen3-32B AUTOSTOP=1 bash lambda/run_all.sh 2>&1 | tee run.log'
```

`run_all.sh` does: smoke test → collect all caches → live experiments → CPU analysis →
optional `shutdown -h now`. Detached via tmux so a disconnect won't kill it.

### Avoiding idle GPU time — checklist

- **Develop on 1B locally**; only tested batch jobs touch the GPU.
- **Persistent volume** for weights + `HF_HOME` → no repeat downloads.
- **Resumable caches**: a crash costs one (category, nudge), not the whole run.
- **Smoke test first** (`--smoke`) catches breakage in ~2 min, not 2 hours.
- **`AUTOSTOP=1`** terminates the box the moment the job finishes (great for overnight).
- **Saturate the GPU**: raise `--batch-size`; prompts are short so 32B handles large batches.
- **Snapshot** the configured instance once it works to skip setup next time.

## Step 3 — pull results, analyze offline

Detach/terminate the GPU box (the persistent volume keeps everything), then run
`scripts/analyze.py` anywhere. To add a new analysis later you only touch `analyze.py`
— no GPU, no re-running the model.

## Prompt-region attention (Gender, SES, Race)

Measures **which part of the prompt the model attends to** at the `Answer:` token
(context / question / choices / nudge), using `output_attentions=True` on Qwen3-32B.

| Phase | Script | Needs GPU? | Output |
|------|--------|:---------:|--------|
| Collect | `scripts/collect_prompt_attention.py` | ✅ | `cache/<model>/prompt_attn_<Category>.npz` |
| Analyze | `scripts/analyze_prompt_attention.py` | ❌ | `results/prompt_attention_*.{json,png}` |

Requires existing `sensitivity_<Category>.npz` caches (already collected).

```bash
# On Lambda (detached):
tmux new -s attn 'bash lambda/run_prompt_attention.sh 2>&1 | tee prompt_attn.log'

# Smoke test first (16 examples):
SMOKE=1 bash lambda/run_prompt_attention.sh

# If OOM on attention forwards:
BATCH=1 ATTN_LAYERS="48 52 56 59 63" bash lambda/run_prompt_attention.sh

# Stratified subsample instead of full n (~600 examples/cat):
STRATIFIED=300 bash lambda/run_prompt_attention.sh
```

After collection, analysis runs automatically and can be re-run locally:

```bash
uv run python scripts/analyze_prompt_attention.py --model Qwen/Qwen3-32B
```

## Reason-before-answer sensitivity (ladder + optional rep)

Adds a **reasoning dimension** alongside the direct-answer sensitivity runs. Each prompt
ends with ``Think carefully step by step...`` / ``Reasoning:``; the model generates
free-form reasoning, then we append ``Answer:`` and score A/B/C logits. Reasoning text
is stored in the cache.

**Fast defaults** (``lambda/run_sensitivity_reasoning.sh``): Gender only, n=500,
128 reasoning tokens, ladder-only (13 passes), gen-batch=8 — ~hours on H100 vs ~days
for the full config.

| Phase | Script | Needs GPU? | Output |
|------|--------|:---------:|--------|
| Collect | `scripts/nudge_sensitivity_reasoning.py` | ✅ | `cache/<model>/sensitivity_reasoning_<Category>.npz` |
| Analyze | `scripts/analyze_sensitivity_reasoning.py` | ❌ | `results/sensitivity_reasoning_<Category>.json` |

```bash
# On Lambda (detached):
tmux new -s reason 'bash lambda/run_sensitivity_reasoning.sh 2>&1 | tee reason.log'

# Override defaults, e.g. full run:
CATEGORIES="Gender_identity SES Race_ethnicity" MAX_EXAMPLES=10000 MAX_REASONING=512 \
  LADDER_ONLY=0 GEN_BATCH=4 bash lambda/run_sensitivity_reasoning.sh
```

Cache keys (per category): ``base_logits``, ``reasoning_base``, ``ladder_{stereo,other}``,
``reasoning_ladder_{stereo,other}``, optional ``rep_*`` / ``reasoning_rep_*`` when
``--include-rep`` / ``LADDER_ONLY=0``, plus id arrays and metadata
(``max_examples``, ``max_reasoning_tokens``, ``ladder_only``).

## 32B gotchas

- **TransformerLens is intentionally not used here** — its weight processing ~doubles
  peak memory and Qwen3 support is unreliable. The HF backend assumes a Llama/Qwen-style
  decoder (`model.model.layers[i].{self_attn,mlp}`), which both target models use.
- **dtype**: default `bfloat16`. Use `float32` only for the tiny local smoke test.
- **Cache size**: residual-only ≈ `n_layers × d_model × 2B` per example per condition
  (~0.65 MB for 32B). Components (mlp+attn) triple that — capture them only for the
  categories you need (`--components-categories`).
- **Left padding** is set so the last real token is always at position `-1` in a batch.
