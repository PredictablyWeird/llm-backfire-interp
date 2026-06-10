"""Plot activation differences across layers: strict backfire vs baseline.

For strict backfire examples, visualises how nudged activations diverge
from baseline activations at each layer, and compares that divergence
to the stayed-at-target group.

Four panels:
  1. Per-layer mean L2 norm of (nudged - baseline) for both groups
  2. Per-layer violin plots of per-example delta norms (backfire only)
  3. Heatmap of per-example delta norms across layers (backfire vs stayed)
  4. Cosine similarity between per-example nudged and baseline activations
     at each layer (how much does the nudge rotate the activation vector?)

Run:
    uv run --env-file .env python scripts/_backfire_activation_diff_plot.py
"""

from __future__ import annotations

import pathlib
import re

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from mech_interp_bbq.activations import collect_logit_diffs, collect_resid_post, load_model
from mech_interp_bbq.data import load_bbq_pairwise
from mech_interp_bbq.nudges import NUDGE_TEMPLATES, group_label, singular_group_label

# ── helpers ───────────────────────────────────────────────────────────────────

_STOP = {"people", "person", "a", "an", "the", "and", "or", "of", "with"}


def _words(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower())) - _STOP


def nudge_target_slot(ex, nudge_gl: str, category: str) -> int | None:
    nudge_words = _words(nudge_gl)
    overlap_a = len(nudge_words & _words(group_label(category, ex.group_a_tag)))
    overlap_b = len(nudge_words & _words(group_label(category, ex.group_b_tag)))
    if overlap_a > overlap_b:
        return 0
    if overlap_b > overlap_a:
        return 1
    return None


# ── setup ─────────────────────────────────────────────────────────────────────

CATEGORY = "Gender_identity"
examples = load_bbq_pairwise(CATEGORY, context_condition="ambig", max_examples=400)

tmpl       = NUDGE_TEMPLATES["user_preference"]
target_tag = examples[0].group_a_tag
nudge_gl   = group_label(CATEGORY, target_tag)
og  = group_label(CATEGORY, examples[0].group_b_tag)
sg  = singular_group_label(CATEGORY, target_tag)
sentence = "(" + tmpl.template.format(group_label=nudge_gl, other_group_label=og, singular_group_label=sg) + ")"

print(f"Nudge: {sentence}")

baseline_prompts = [ex.prompt_with_sentence() for ex in examples]
nudged_prompts   = [ex.prompt_with_sentence(sentence, position=tmpl.position) for ex in examples]

model = load_model("meta-llama/Llama-3.2-1B")

# ── collect ───────────────────────────────────────────────────────────────────

print("Collecting baseline activations...")
base_acts = collect_resid_post(model, baseline_prompts, batch_size=4).acts.numpy()    # (N, L, D)
print("Collecting nudged activations...")
nudged_acts = collect_resid_post(model, nudged_prompts, batch_size=4).acts.numpy()    # (N, L, D)

print("Collecting logit diffs...")
base_logit_diff   = collect_logit_diffs(model, baseline_prompts, n_choices=2, batch_size=4)
nudged_logit_diff = collect_logit_diffs(model, nudged_prompts,   n_choices=2, batch_size=4)

n_layers = base_acts.shape[1]
delta = nudged_acts - base_acts    # (N, L, D)

# ── group examples ────────────────────────────────────────────────────────────

base_pred   = (base_logit_diff   <= 0).astype(int)
nudged_pred = (nudged_logit_diff <= 0).astype(int)

valid = [(i, nudge_target_slot(examples[i], nudge_gl, CATEGORY)) for i in range(len(examples))]
valid = [(i, ts) for i, ts in valid if ts is not None]

strict_bf, stayed_m = [], []
for i, ts in valid:
    if (base_pred[i] == ts) and (nudged_pred[i] != ts):
        strict_bf.append(i)
    elif (base_pred[i] == ts) and (nudged_pred[i] == ts):
        stayed_m.append(i)

bf_arr = np.array(strict_bf)
sm_arr = np.array(stayed_m)
print(f"\nStrict backfire: {len(bf_arr)}   Stayed-at-target: {len(sm_arr)}")

# ── compute per-example delta norms and cosine similarities ───────────────────

# delta_norm[i, l] = ||nudged[i,l] - base[i,l]||
delta_norm = np.linalg.norm(delta, axis=2)   # (N, L)

# cosine_sim[i, l] = cos(nudged[i,l], base[i,l])
base_norm   = np.linalg.norm(base_acts,   axis=2)   # (N, L)
nudged_norm = np.linalg.norm(nudged_acts, axis=2)   # (N, L)
dot_prod    = (base_acts * nudged_acts).sum(axis=2)  # (N, L)
cosine_sim  = dot_prod / (base_norm * nudged_norm + 1e-12)   # (N, L)

layers = list(range(n_layers))

# ── print summary table ───────────────────────────────────────────────────────

print(f"\n{'layer':>5}  {'mean_δnorm_BF':>14}  {'mean_δnorm_SM':>14}  {'mean_cos_BF':>12}  {'mean_cos_SM':>12}")
print("-" * 65)
for l in layers:
    print(f"{l:>5}  {delta_norm[bf_arr, l].mean():>14.4f}  "
          f"{delta_norm[sm_arr, l].mean():>14.4f}  "
          f"{cosine_sim[bf_arr, l].mean():>12.6f}  "
          f"{cosine_sim[sm_arr, l].mean():>12.6f}")

# ── plot ──────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

# ── Panel 1: Mean delta norm per layer ────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])

mean_bf = [delta_norm[bf_arr, l].mean() for l in layers]
std_bf  = [delta_norm[bf_arr, l].std()  for l in layers]
mean_sm = [delta_norm[sm_arr, l].mean() for l in layers]
std_sm  = [delta_norm[sm_arr, l].std()  for l in layers]

ax1.plot(layers, mean_bf, marker="o", color="firebrick",
         label=f"backfire (n={len(bf_arr)})", zorder=3)
ax1.fill_between(layers,
    np.array(mean_bf) - np.array(std_bf),
    np.array(mean_bf) + np.array(std_bf),
    color="firebrick", alpha=0.15)

ax1.plot(layers, mean_sm, marker="s", color="seagreen",
         label=f"stayed-at-target (n={len(sm_arr)})", zorder=3)
ax1.fill_between(layers,
    np.array(mean_sm) - np.array(std_sm),
    np.array(mean_sm) + np.array(std_sm),
    color="seagreen", alpha=0.15)

ax1.set_xlabel("Layer")
ax1.set_ylabel("||nudged − baseline||  (L2)")
ax1.set_title("Mean activation change magnitude per layer\n(shaded = ±1 std)")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# ── Panel 2: Cosine similarity between nudged and baseline per layer ──────────
ax2 = fig.add_subplot(gs[0, 1])

mean_cos_bf = [cosine_sim[bf_arr, l].mean() for l in layers]
mean_cos_sm = [cosine_sim[sm_arr, l].mean() for l in layers]
std_cos_bf  = [cosine_sim[bf_arr, l].std()  for l in layers]
std_cos_sm  = [cosine_sim[sm_arr, l].std()  for l in layers]

ax2.plot(layers, mean_cos_bf, marker="o", color="firebrick", label="backfire")
ax2.fill_between(layers,
    np.array(mean_cos_bf) - np.array(std_cos_bf),
    np.array(mean_cos_bf) + np.array(std_cos_bf),
    color="firebrick", alpha=0.15)

ax2.plot(layers, mean_cos_sm, marker="s", color="seagreen", label="stayed-at-target")
ax2.fill_between(layers,
    np.array(mean_cos_sm) - np.array(std_cos_sm),
    np.array(mean_cos_sm) + np.array(std_cos_sm),
    color="seagreen", alpha=0.15)

ax2.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)
ax2.set_xlabel("Layer")
ax2.set_ylabel("cos(nudged, baseline)")
ax2.set_title("How much does the nudge rotate\nthe activation vector? (1 = no rotation)")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# ── Panel 3: Heatmap of per-example delta norms — backfire ────────────────────
ax3 = fig.add_subplot(gs[1, 0])

# Stack backfire rows on top, stayed rows below with a separator
heatmap_data = np.vstack([
    delta_norm[bf_arr],     # (n_bf, L)
    np.full((1, n_layers), np.nan),   # separator row
    delta_norm[sm_arr],     # (n_sm, L)
])

im = ax3.imshow(heatmap_data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
plt.colorbar(im, ax=ax3, label="||nudged − baseline||")

# Separator line
ax3.axhline(len(bf_arr) + 0.5, color="black", linewidth=1.5, linestyle="--")

# Y-axis labels
ytick_pos  = list(range(len(bf_arr))) + [len(bf_arr) + 1 + i for i in range(min(len(sm_arr), 15))]
ytick_lbl  = [f"BF-{i}" for i in range(len(bf_arr))] + [f"SM-{i}" for i in range(min(len(sm_arr), 15))]
ax3.set_yticks(ytick_pos)
ax3.set_yticklabels(ytick_lbl, fontsize=7)
ax3.set_xticks(layers)
ax3.set_xticklabels(layers)
ax3.set_xlabel("Layer")
ax3.set_title("Per-example activation change heatmap\n(BF = backfire, SM = stayed; dashed = group boundary)")

# ── Panel 4: Violin of delta norms at early / mid / final layers ──────────────
ax4 = fig.add_subplot(gs[1, 1])

highlight_layers = [0, 4, 8, 12, 15]
positions_bf = [i * 3 + 0.7 for i in range(len(highlight_layers))]
positions_sm = [i * 3 + 1.3 for i in range(len(highlight_layers))]

vp_bf = ax4.violinplot(
    [delta_norm[bf_arr, l] for l in highlight_layers],
    positions=positions_bf, widths=0.5, showmedians=True,
)
vp_sm = ax4.violinplot(
    [delta_norm[sm_arr, l] for l in highlight_layers],
    positions=positions_sm, widths=0.5, showmedians=True,
)

for body in vp_bf["bodies"]:
    body.set_facecolor("firebrick"); body.set_alpha(0.6)
for body in vp_sm["bodies"]:
    body.set_facecolor("seagreen"); body.set_alpha(0.6)
for part in ("cbars", "cmins", "cmaxes", "cmedians"):
    vp_bf[part].set_color("firebrick")
    vp_sm[part].set_color("seagreen")

ax4.set_xticks([i * 3 + 1 for i in range(len(highlight_layers))])
ax4.set_xticklabels([f"Layer {l}" for l in highlight_layers])
ax4.set_ylabel("||nudged − baseline||  (L2)")
ax4.set_title("Distribution of activation change magnitude\nat selected layers")

from matplotlib.patches import Patch
ax4.legend(handles=[
    Patch(facecolor="firebrick", alpha=0.7, label=f"backfire (n={len(bf_arr)})"),
    Patch(facecolor="seagreen",  alpha=0.7, label=f"stayed-at-target (n={len(sm_arr)})"),
], fontsize=9)
ax4.grid(alpha=0.3, axis="y")

fig.suptitle(
    f"Llama-3.2-1B | {CATEGORY} | user_preference nudge toward '{nudge_gl}'\n"
    f"Activation difference: nudged vs baseline  (strict backfire vs stayed-at-target)",
    fontsize=11,
)

out = pathlib.Path("probes_out/backfire_activation_diff.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved {out}")
plt.close(fig)
