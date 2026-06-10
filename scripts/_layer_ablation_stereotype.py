"""Layer ablation experiment: which layers suppress vs amplify stereotypes?

For each layer l we zero out its contribution to the residual stream by
patching both hook_attn_out and hook_mlp_out to zero.  This is equivalent
to asking: "what if this layer did nothing?"

We then measure the 'stereotype score' on all resolvable BBQ examples:
    stereo_score[i] = logit(stereotyped_slot) - logit(other_slot)

Positive score  → model prefers the stereotyped answer
Negative score  → model prefers the non-stereotyped answer

Comparing ablated vs baseline:
    delta[l] = mean_stereo_score(ablated_l) - mean_stereo_score(no_ablation)

    delta > 0 → ablating layer l makes model MORE stereotyped
                → layer l was SUPPRESSING stereotypes
    delta < 0 → ablating layer l makes model LESS stereotyped
                → layer l was AMPLIFYING stereotypes

Run:
    uv run --env-file .env python scripts/_layer_ablation_stereotype.py
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from mech_interp_bbq.activations import load_model
from mech_interp_bbq.data import load_bbq_pairwise
from mech_interp_bbq.nudges import group_label

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


# ── setup ─────────────────────────────────────────────────────────────────────

CATEGORY  = "Gender_identity"
BATCH     = 8
examples  = load_bbq_pairwise(CATEGORY, context_condition="ambig", max_examples=400)

# Keep only resolvable examples
resolvable = [(i, stereotyped_slot(ex)) for i, ex in enumerate(examples)]
resolvable = [(i, ss) for i, ss in resolvable if ss is not None]
print(f"Resolvable examples: {len(resolvable)} / {len(examples)}")

indices     = [i  for i, _ in resolvable]
stereo_slots = [ss for _, ss in resolvable]

prompts = [examples[i].prompt_with_sentence() for i in indices]

model   = load_model("meta-llama/Llama-3.2-1B")
n_layers = model.cfg.n_layers

# Pre-tokenise all prompts once
tokens_list = [model.to_tokens(p, prepend_bos=True) for p in prompts]

# ── helper: get stereotype logit scores for a set of prompts with hooks ───────

@torch.inference_mode()
def get_stereo_scores(
    token_batch: list[torch.Tensor],
    ss_batch: list[int],
    fwd_hooks: list = [],
) -> np.ndarray:
    """Return stereotype logit scores for a batch.

    stereo_score[i] = logit(stereotyped_slot) - logit(other_slot)
    """
    # Answer token ids: A=0, B=1
    tok_A = int(model.to_tokens(" A", prepend_bos=False)[0, -1])
    tok_B = int(model.to_tokens(" B", prepend_bos=False)[0, -1])

    scores = []
    for toks, ss in zip(token_batch, ss_batch):
        logits = model.run_with_hooks(
            toks, fwd_hooks=fwd_hooks, return_type="logits"
        )                                        # (1, seq, vocab)
        last   = logits[0, -1, :]               # (vocab,)
        logit_A = float(last[tok_A])
        logit_B = float(last[tok_B])
        stereo_logit = logit_A if ss == 0 else logit_B
        other_logit  = logit_B if ss == 0 else logit_A
        scores.append(stereo_logit - other_logit)
    return np.array(scores, dtype=np.float32)


# ── baseline (no ablation) ────────────────────────────────────────────────────

print("\nComputing baseline stereotype scores...")
baseline_scores = get_stereo_scores(tokens_list, stereo_slots)
baseline_mean   = float(baseline_scores.mean())
print(f"Baseline stereotype score: {baseline_mean:+.4f}")
print(f"  (positive = model leans stereotyped on average)")

# ── per-layer ablation ────────────────────────────────────────────────────────

layer_means = []
layer_deltas = []

print("\nAblating layers one at a time...")
for l in tqdm(range(n_layers), desc="layer ablation"):
    # Zero out this layer's attention and MLP contributions
    def zero_hook(value, hook):
        return torch.zeros_like(value)

    hooks = [
        (f"blocks.{l}.hook_attn_out", zero_hook),
        (f"blocks.{l}.hook_mlp_out",  zero_hook),
    ]

    scores_l = get_stereo_scores(tokens_list, stereo_slots, fwd_hooks=hooks)
    mean_l   = float(scores_l.mean())
    delta_l  = mean_l - baseline_mean

    layer_means.append(mean_l)
    layer_deltas.append(delta_l)
    tqdm.write(f"  layer {l:>2}: mean={mean_l:+.4f}  delta={delta_l:+.4f}  "
               f"({'MORE stereotyped' if delta_l > 0 else 'LESS stereotyped'})")

# ── summary ───────────────────────────────────────────────────────────────────

print(f"\n{'─'*55}")
print(f"Baseline stereotype score  : {baseline_mean:+.4f}")
print(f"\nTop layers whose ablation increases stereotyping (suppressors):")
top_suppress = sorted(range(n_layers), key=lambda l: -layer_deltas[l])[:5]
for l in top_suppress:
    print(f"  layer {l:>2}: delta={layer_deltas[l]:+.4f}")

print(f"\nTop layers whose ablation decreases stereotyping (amplifiers):")
top_amplify = sorted(range(n_layers), key=lambda l: layer_deltas[l])[:5]
for l in top_amplify:
    print(f"  layer {l:>2}: delta={layer_deltas[l]:+.4f}")

# ── plot ──────────────────────────────────────────────────────────────────────

layers = list(range(n_layers))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: absolute stereotype score per ablated layer
ax = axes[0]
ax.bar(layers, layer_means, color=[
    "firebrick" if d > 0 else "steelblue" for d in layer_deltas
], alpha=0.8)
ax.axhline(baseline_mean, color="black", linestyle="--", linewidth=1.5,
           label=f"baseline ({baseline_mean:+.3f})")
ax.set_xlabel("Ablated layer")
ax.set_ylabel("Mean stereotype score  (logit_stereo − logit_other)")
ax.set_title("Stereotype score when each layer is ablated\n"
             "(red = MORE stereotyped than baseline, blue = LESS)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")

# Panel 2: delta from baseline
ax = axes[1]
colors = ["firebrick" if d > 0 else "steelblue" for d in layer_deltas]
bars   = ax.bar(layers, layer_deltas, color=colors, alpha=0.8)
ax.axhline(0, color="black", linestyle="--", linewidth=1.2)

# Annotate most extreme bars
max_delta = max(layer_deltas)
min_delta = min(layer_deltas)
for l, d in enumerate(layer_deltas):
    if abs(d) >= 0.8 * max(abs(max_delta), abs(min_delta)):
        ax.text(l, d + (0.005 if d > 0 else -0.008), f"L{l}",
                ha="center", va="bottom" if d > 0 else "top",
                fontsize=8, fontweight="bold")

ax.set_xlabel("Ablated layer")
ax.set_ylabel("Δ stereotype score  (ablated − baseline)")
ax.set_title("Change in stereotyping from ablating each layer\n"
             "red (+) = layer suppressed stereotypes  |  blue (−) = layer amplified stereotypes")
ax.grid(alpha=0.3, axis="y")

# Add text labels
ax.text(0.02, 0.97, "↑ Suppressor layers\n(ablating makes model MORE biased)",
        transform=ax.transAxes, va="top", fontsize=8,
        color="firebrick", alpha=0.8)
ax.text(0.02, 0.05, "↓ Amplifier layers\n(ablating makes model LESS biased)",
        transform=ax.transAxes, va="bottom", fontsize=8,
        color="steelblue", alpha=0.8)

fig.suptitle(
    f"Llama-3.2-1B | {CATEGORY} | Layer ablation — stereotype suppression vs amplification\n"
    f"n={len(resolvable)} resolvable examples (ambiguous context)",
    fontsize=10,
)
fig.tight_layout()
out = pathlib.Path("probes_out/layer_ablation_stereotype.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved {out}")
plt.close(fig)
