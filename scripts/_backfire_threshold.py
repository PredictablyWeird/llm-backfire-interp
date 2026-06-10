"""Backfire threshold analysis.

Tests whether strict-backfire examples already sit near a 'stereotype resistance'
threshold in the model's BASELINE state, before any nudge is applied.

Two measurements:
  1. Baseline stereotype-direction projection:
       s[i, l] = dot(base_acts[i, l], stereo_dir[l])
     If backfire cases have more negative s at baseline, they were already leaning
     away from the stereotype before the nudge arrived.

  2. Baseline logit margin (A-logit minus B-logit):
     A small margin means the model was already near its decision boundary.
     If backfire cases have smaller margins, the nudge only had to push slightly
     to flip the output — consistent with a threshold story.

Run:
    uv run --env-file .env python scripts/_backfire_threshold.py
"""

from __future__ import annotations

import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from mech_interp_bbq.activations import collect_logit_diffs, collect_resid_post, load_model
from mech_interp_bbq.data import load_bbq_pairwise
from mech_interp_bbq.nudges import NUDGE_TEMPLATES, group_label, singular_group_label

# ── helpers ───────────────────────────────────────────────────────────────────

_STOP = {"people", "person", "a", "an", "the", "and", "or", "of", "with"}


def _words(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower())) - _STOP


def nudge_target_slot(ex, nudge_gl: str, category: str) -> int | None:
    nudge_words = _words(nudge_gl)
    words_a = _words(group_label(category, ex.group_a_tag))
    words_b = _words(group_label(category, ex.group_b_tag))
    overlap_a = len(nudge_words & words_a)
    overlap_b = len(nudge_words & words_b)
    if overlap_a > overlap_b:
        return 0
    if overlap_b > overlap_a:
        return 1
    return None


# ── setup ─────────────────────────────────────────────────────────────────────

CATEGORY = "Gender_identity"
examples = load_bbq_pairwise(CATEGORY, context_condition="ambig", max_examples=400)

tmpl = NUDGE_TEMPLATES["user_preference"]
target_tag = examples[0].group_a_tag
nudge_gl   = group_label(CATEGORY, target_tag)
og  = group_label(CATEGORY, examples[0].group_b_tag)
sg  = singular_group_label(CATEGORY, target_tag)
sentence = "(" + tmpl.template.format(group_label=nudge_gl, other_group_label=og, singular_group_label=sg) + ")"

print(f"Nudge: {sentence}")
print(f"Nudge target: '{nudge_gl}'")

baseline_prompts = [ex.prompt_with_sentence() for ex in examples]
nudged_prompts   = [ex.prompt_with_sentence(sentence, position=tmpl.position) for ex in examples]

model = load_model("meta-llama/Llama-3.2-1B")

# ── collect ───────────────────────────────────────────────────────────────────

print("\nCollecting baseline activations...")
base_acts = collect_resid_post(model, baseline_prompts, batch_size=4).acts.numpy()  # (N, L, D)

print("Collecting baseline logit diffs (A - B)...")
base_logit_diff = collect_logit_diffs(model, baseline_prompts, n_choices=2, batch_size=4)  # (N,)

print("Collecting nudged logit diffs...")
nudged_logit_diff = collect_logit_diffs(model, nudged_prompts, n_choices=2, batch_size=4)  # (N,)

n_layers = base_acts.shape[1]

# ── group examples ────────────────────────────────────────────────────────────

# A positive logit diff means model prefers slot A.
# base_pred[i] = 0 if logit_diff > 0 else 1
base_pred   = (base_logit_diff <= 0).astype(int)   # 0=A, 1=B
nudged_pred = (nudged_logit_diff <= 0).astype(int)

valid = [(i, nudge_target_slot(examples[i], nudge_gl, CATEGORY)) for i in range(len(examples))]
valid = [(i, ts) for i, ts in valid if ts is not None]

strict_bf, stayed_m = [], []
for i, ts in valid:
    chose_target_base  = (base_pred[i]   == ts)
    chose_target_nudge = (nudged_pred[i] == ts)
    if chose_target_base and not chose_target_nudge:
        strict_bf.append(i)
    elif chose_target_base and chose_target_nudge:
        stayed_m.append(i)

bf_arr = np.array(strict_bf)
sm_arr = np.array(stayed_m)
print(f"\nStrict backfire: {len(bf_arr)}   Stayed-at-target: {len(sm_arr)}")

if len(bf_arr) < 3:
    print("Too few backfire examples.")
    raise SystemExit

# ── load stereotype directions ────────────────────────────────────────────────

coef_path = pathlib.Path("probes_out/bias_probe_meta-llama_Llama-3.2-1B_Gender_identity_coefs.npz")
stereo_coefs = np.load(coef_path)["stereotype_coef"]   # (L, n_classes, D)

stereo_dirs = []
for l in range(n_layers):
    c = stereo_coefs[l]
    w = c[0] - c[-1]
    w /= (np.linalg.norm(w) + 1e-12)
    stereo_dirs.append(w)

# ── measurement 1: baseline stereotype projection ─────────────────────────────
# s[i, l] = dot(base_acts[i, l], stereo_dir[l])

print("\n── Measurement 1: Baseline stereotype-direction projection ──")
print(f"{'layer':>5}  {'mean_bf':>9}  {'mean_sm':>9}  {'diff':>8}  {'t-stat':>8}  {'p-val':>8}")
print("-" * 55)

baseline_proj = np.stack(
    [base_acts[:, l, :] @ stereo_dirs[l] for l in range(n_layers)], axis=1
)  # (N, L)

t_stats, p_vals = [], []
for l in range(n_layers):
    bf_proj = baseline_proj[bf_arr, l]
    sm_proj = baseline_proj[sm_arr, l]
    t, p = stats.ttest_ind(bf_proj, sm_proj, equal_var=False)
    t_stats.append(float(t))
    p_vals.append(float(p))
    print(f"{l:>5}  {bf_proj.mean():>9.4f}  {sm_proj.mean():>9.4f}  "
          f"{bf_proj.mean()-sm_proj.mean():>8.4f}  {t:>8.3f}  {p:>8.4f}")

best_proj_layer = int(np.argmax(np.abs(t_stats)))
print(f"\nLargest separation at layer {best_proj_layer}  "
      f"(t={t_stats[best_proj_layer]:.3f}, p={p_vals[best_proj_layer]:.4f})")

# ── measurement 2: baseline logit margin ─────────────────────────────────────
# For examples that chose the target slot at baseline, what was the margin?
# Backfire: chose slot A (target) → logit_diff > 0.  Stayed: same.
# The margin is abs(logit_diff) — how far from the boundary.

print("\n── Measurement 2: Baseline logit margin (confidence at baseline) ──")

# Sign convention: positive logit_diff = prefers A. Target slot may be A or B per example.
# Compute signed margin toward target slot:
#   margin[i] = +logit_diff[i] if target_slot==0, else -logit_diff[i]
margin = np.array([
    base_logit_diff[i] * (1 if ts == 0 else -1)
    for i, ts in valid
], dtype=np.float32)

# Get indices into 'valid' list
valid_indices = [i for i, _ in valid]
bf_valid_pos  = [valid_indices.index(i) for i in bf_arr]
sm_valid_pos  = [valid_indices.index(i) for i in sm_arr]

margin_bf = margin[bf_valid_pos]
margin_sm = margin[sm_valid_pos]

t_m, p_m = stats.ttest_ind(margin_bf, margin_sm, equal_var=False)
print(f"Mean margin at baseline (toward target slot):")
print(f"  Backfire   : {margin_bf.mean():.4f}  (std={margin_bf.std():.4f})")
print(f"  Stayed     : {margin_sm.mean():.4f}  (std={margin_sm.std():.4f})")
print(f"  t={t_m:.3f}  p={p_m:.4f}")
print()
print(f"Backfire margins  : {np.round(sorted(margin_bf), 3).tolist()}")
print(f"Stayed-at margins : {np.round(sorted(margin_sm)[:10], 3).tolist()}  ... (first 10)")

# ── plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: baseline stereotype projection at best layer
ax = axes[0]
ax.hist(baseline_proj[sm_arr, best_proj_layer], bins=15, alpha=0.6,
        color="seagreen", label=f"stayed-at-target (n={len(sm_arr)})", density=True)
ax.hist(baseline_proj[bf_arr, best_proj_layer], bins=8, alpha=0.7,
        color="firebrick", label=f"backfire (n={len(bf_arr)})", density=True)
ax.axvline(baseline_proj[sm_arr, best_proj_layer].mean(), color="seagreen",
           linestyle="--", linewidth=1.5)
ax.axvline(baseline_proj[bf_arr, best_proj_layer].mean(), color="firebrick",
           linestyle="--", linewidth=1.5)
ax.set_xlabel("Projection onto stereotype direction")
ax.set_ylabel("Density")
ax.set_title(f"Baseline stereotype projection (layer {best_proj_layer})\n"
             f"t={t_stats[best_proj_layer]:.2f}, p={p_vals[best_proj_layer]:.3f}")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 2: mean baseline projection across layers
ax = axes[1]
layers = list(range(n_layers))
ax.plot(layers, [baseline_proj[bf_arr, l].mean() for l in layers],
        marker="o", color="firebrick", label="backfire")
ax.plot(layers, [baseline_proj[sm_arr, l].mean() for l in layers],
        marker="s", color="seagreen", label="stayed-at-target")
ax.fill_between(layers,
    [baseline_proj[bf_arr, l].mean() - baseline_proj[bf_arr, l].std() for l in layers],
    [baseline_proj[bf_arr, l].mean() + baseline_proj[bf_arr, l].std() for l in layers],
    color="firebrick", alpha=0.15)
ax.fill_between(layers,
    [baseline_proj[sm_arr, l].mean() - baseline_proj[sm_arr, l].std() for l in layers],
    [baseline_proj[sm_arr, l].mean() + baseline_proj[sm_arr, l].std() for l in layers],
    color="seagreen", alpha=0.15)
ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
ax.set_xlabel("Layer")
ax.set_ylabel("Mean baseline projection onto stereotype direction")
ax.set_title("Stereotype-direction alignment at baseline\n(shaded = ±1 std)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 3: baseline logit margin distribution
ax = axes[2]
ax.hist(margin_sm, bins=15, alpha=0.6, color="seagreen",
        label=f"stayed-at-target (n={len(sm_arr)})", density=True)
ax.hist(margin_bf, bins=6, alpha=0.7, color="firebrick",
        label=f"backfire (n={len(bf_arr)})", density=True)
ax.axvline(margin_sm.mean(), color="seagreen", linestyle="--", linewidth=1.5)
ax.axvline(margin_bf.mean(), color="firebrick", linestyle="--", linewidth=1.5)
ax.axvline(0, color="black", linestyle="-", linewidth=1.0, label="decision boundary")
ax.set_xlabel("Baseline logit margin toward nudge-target slot")
ax.set_ylabel("Density")
ax.set_title(f"Confidence at baseline (before nudge)\n"
             f"t={t_m:.2f}, p={p_m:.3f}")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle(
    f"Llama-3.2-1B | {CATEGORY} | user_preference nudge toward '{nudge_gl}'\n"
    f"Are backfire cases already near a stereotype-resistance threshold at baseline?",
    fontsize=10,
)
fig.tight_layout()
out = pathlib.Path("probes_out/backfire_threshold.png")
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")
plt.close(fig)
