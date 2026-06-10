"""Layer ablation across all BBQ categories.

For each category and each layer we zero out that layer's contribution
(hook_attn_out + hook_mlp_out → 0) and measure the change in stereotype score:

    delta[category, layer] = mean_stereo_score(ablated) - mean_stereo_score(baseline)

    delta > 0  → layer suppresses stereotypes  (ablating it increases bias)
    delta < 0  → layer amplifies stereotypes   (ablating it decreases bias)

Plots:
  1. Heatmap  : categories × layers, colour = delta
  2. Aggregate: mean delta across all categories per layer
  3. Category lines: per-category delta curves on one axes

Run:
    uv run --env-file .env python scripts/_layer_ablation_all_categories.py
"""

from __future__ import annotations

import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from mech_interp_bbq.activations import load_model
from mech_interp_bbq.data import HITZ_CATEGORIES, load_bbq_pairwise

MAX_EXAMPLES = 150   # per category
BATCH_SIZE   = 8

# ── stereotype slot helper ────────────────────────────────────────────────────

def stereotyped_slot(ex) -> int | None:
    sg_lower = {g.lower() for g in ex.stereotyped_groups}

    def is_sg(tag: str) -> bool:
        tc = tag.lower()
        return tc in sg_lower or any(g in tc or tc in g for g in sg_lower)

    a_sg, b_sg = is_sg(ex.group_a_tag), is_sg(ex.group_b_tag)
    if ex.polarity == "neg":
        if a_sg and not b_sg: return 0
        if b_sg and not a_sg: return 1
    else:
        if not a_sg and b_sg: return 0
        if not b_sg and a_sg: return 1
    return None


# ── load model ────────────────────────────────────────────────────────────────

model    = load_model("meta-llama/Llama-3.2-1B")
n_layers = model.cfg.n_layers

tok_A = int(model.to_tokens(" A", prepend_bos=False)[0, -1])
tok_B = int(model.to_tokens(" B", prepend_bos=False)[0, -1])

# ── collect examples from all categories ─────────────────────────────────────

# category_data[cat] = list of (prompt_str, stereo_slot_idx)
category_data: dict[str, list[tuple[str, int]]] = {}

for cat in HITZ_CATEGORIES:
    examples = load_bbq_pairwise(cat, context_condition="ambig",
                                 max_examples=MAX_EXAMPLES)
    resolved = []
    for ex in examples:
        ss = stereotyped_slot(ex)
        if ss is not None:
            resolved.append((ex.prompt_with_sentence(), ss))
    category_data[cat] = resolved
    print(f"  {cat:<25}: {len(resolved):>3} resolvable examples")

all_prompts  = [p  for items in category_data.values() for p, _ in items]
all_ss       = [ss for items in category_data.values() for _, ss in items]
print(f"\nTotal resolvable examples across all categories: {len(all_prompts)}")

# ── batched inference with optional hooks ─────────────────────────────────────

@torch.inference_mode()
def run_batched(
    prompts: list[str],
    ss_list: list[int],
    fwd_hooks: list = [],
) -> np.ndarray:
    """Return stereotype score (logit_stereo - logit_other) for each prompt."""
    scores = []
    for start in range(0, len(prompts), BATCH_SIZE):
        batch_p  = prompts[start : start + BATCH_SIZE]
        batch_ss = ss_list[start : start + BATCH_SIZE]
        # to_tokens left-pads so all sequences end at position -1
        tokens   = model.to_tokens(batch_p, prepend_bos=True)
        logits   = model.run_with_hooks(
            tokens, fwd_hooks=fwd_hooks, return_type="logits"
        )                                  # (B, seq, vocab)
        last     = logits[:, -1, :]        # (B, vocab)
        for j, ss in enumerate(batch_ss):
            la = float(last[j, tok_A])
            lb = float(last[j, tok_B])
            scores.append(la - lb if ss == 0 else lb - la)
    return np.array(scores, dtype=np.float32)


# ── baseline ──────────────────────────────────────────────────────────────────

print("\nComputing baseline scores...")
baseline_all = run_batched(all_prompts, all_ss)

# Split back into per-category arrays
cat_sizes   = {cat: len(items) for cat, items in category_data.items()}
cat_offsets = {}
offset = 0
for cat in HITZ_CATEGORIES:
    cat_offsets[cat] = (offset, offset + cat_sizes[cat])
    offset += cat_sizes[cat]

baseline_by_cat = {
    cat: baseline_all[s:e].mean()
    for cat, (s, e) in cat_offsets.items()
    if cat_sizes[cat] > 0
}
print("Baseline stereotype scores by category:")
for cat, score in baseline_by_cat.items():
    print(f"  {cat:<25}: {score:+.4f}")

# ── per-layer ablation ────────────────────────────────────────────────────────

# delta_matrix[cat_idx, layer] = delta stereotype score
categories = [c for c in HITZ_CATEGORIES if cat_sizes[c] > 0]
delta_matrix = np.zeros((len(categories), n_layers), dtype=np.float32)

print(f"\nRunning {n_layers} layer ablations...")
for l in tqdm(range(n_layers), desc="layer"):
    def zero_hook(value, hook):
        return torch.zeros_like(value)

    hooks = [
        (f"blocks.{l}.hook_attn_out", zero_hook),
        (f"blocks.{l}.hook_mlp_out",  zero_hook),
    ]

    ablated_all = run_batched(all_prompts, all_ss, fwd_hooks=hooks)

    for ci, cat in enumerate(categories):
        s, e = cat_offsets[cat]
        if e > s:
            ablated_mean = ablated_all[s:e].mean()
            delta_matrix[ci, l] = ablated_mean - baseline_by_cat[cat]

# ── summary ───────────────────────────────────────────────────────────────────

print(f"\n{'─'*70}")
print(f"{'Layer':>6}", end="")
for cat in categories:
    short = cat[:8]
    print(f"  {short:>8}", end="")
print(f"  {'MEAN':>8}")
print("─" * 70)

for l in range(n_layers):
    print(f"  L{l:>2}  ", end="")
    for ci in range(len(categories)):
        d = delta_matrix[ci, l]
        print(f"  {d:>+8.4f}", end="")
    print(f"  {delta_matrix[:, l].mean():>+8.4f}")

# Top suppressor and amplifier layers (by mean delta across categories)
mean_delta = delta_matrix.mean(axis=0)
print(f"\nTop suppressor layers (mean delta across all categories):")
for l in sorted(range(n_layers), key=lambda x: -mean_delta[x])[:5]:
    print(f"  Layer {l:>2}: mean delta = {mean_delta[l]:+.4f}")
print(f"\nTop amplifier layers:")
for l in sorted(range(n_layers), key=lambda x: mean_delta[x])[:5]:
    print(f"  Layer {l:>2}: mean delta = {mean_delta[l]:+.4f}")

# ── plot ──────────────────────────────────────────────────────────────────────

layers     = list(range(n_layers))
cat_labels = [c.replace("_", "\n") for c in categories]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Panel 1: heatmap — categories × layers
ax = axes[0]
vmax = np.abs(delta_matrix).max()
im   = ax.imshow(delta_matrix, aspect="auto", cmap="RdBu_r",
                 vmin=-vmax, vmax=vmax, interpolation="nearest")
plt.colorbar(im, ax=ax, label="Δ stereotype score")
ax.set_xticks(layers)
ax.set_xticklabels([str(l) for l in layers], fontsize=8)
ax.set_yticks(range(len(categories)))
ax.set_yticklabels(cat_labels, fontsize=7)
ax.set_xlabel("Ablated layer")
ax.set_title("Δ stereotype score per category & layer\n"
             "red = suppressor (+), blue = amplifier (−)")
# Mark the strongest suppressor per category
for ci in range(len(categories)):
    best_l = int(np.argmax(delta_matrix[ci]))
    ax.plot(best_l, ci, marker="*", color="black", markersize=8, zorder=3)

# Panel 2: mean delta across categories
ax = axes[1]
colors = ["firebrick" if d > 0 else "steelblue" for d in mean_delta]
ax.bar(layers, mean_delta, color=colors, alpha=0.85)
ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
for l, d in enumerate(mean_delta):
    if abs(d) >= 0.6 * np.abs(mean_delta).max():
        ax.text(l, d + (0.001 if d > 0 else -0.001),
                f"L{l}", ha="center",
                va="bottom" if d > 0 else "top",
                fontsize=8, fontweight="bold")
ax.set_xlabel("Ablated layer")
ax.set_ylabel("Mean Δ stereotype score (across all categories)")
ax.set_title("Aggregate: suppressor vs amplifier layers\n"
             "red (+) = suppressor  |  blue (−) = amplifier")
ax.grid(alpha=0.3, axis="y")

# Panel 3: per-category delta curves
ax = axes[2]
cmap = plt.get_cmap("tab10")
for ci, cat in enumerate(categories):
    ax.plot(layers, delta_matrix[ci], marker="o", markersize=3,
            color=cmap(ci % 10), alpha=0.8,
            label=cat.replace("_", " "), linewidth=1.2)
ax.axhline(0, color="black", linestyle="--", linewidth=1.0)
ax.plot(layers, mean_delta, color="black", linewidth=2.5,
        linestyle="-", label="MEAN", zorder=5)
ax.set_xlabel("Ablated layer")
ax.set_ylabel("Δ stereotype score")
ax.set_title("Per-category layer ablation curves\n(black = mean)")
ax.legend(fontsize=6, ncol=2, loc="upper left")
ax.grid(alpha=0.3)

fig.suptitle(
    f"Llama-3.2-1B | All BBQ categories | Layer ablation stereotype analysis\n"
    f"n≤{MAX_EXAMPLES} per category, ambiguous context",
    fontsize=10,
)
fig.tight_layout()
out = pathlib.Path("probes_out/layer_ablation_all_categories.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved {out}")
plt.close(fig)
