"""Logit analysis: backfire vs stayed-at-target.

For each example we measure the logit margin toward the nudge-target slot:
    margin[i] = logit(target_slot) - logit(other_slot)

Positive  → model prefers the target slot
Negative  → model prefers the other slot
Zero      → exactly on the decision boundary

We compare three quantities for backfire vs stayed-at-target:
  1. Baseline margin   (before nudge)
  2. Nudged margin     (after nudge toward target slot)
  3. Shift             = nudged_margin - baseline_margin

Run:
    uv run --env-file .env python scripts/_backfire_logit_analysis.py
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from mech_interp_bbq.activations import collect_logit_diffs, collect_model_predictions, load_model
from mech_interp_bbq.data import load_bbq_pairwise
from mech_interp_bbq.nudges import NUDGE_TEMPLATES, group_label, singular_group_label

# ── setup ─────────────────────────────────────────────────────────────────────

CATEGORY = "Gender_identity"
examples = load_bbq_pairwise(CATEGORY, context_condition="ambig", max_examples=400)
tmpl     = NUDGE_TEMPLATES["user_preference"]

baseline_prompts = []
nudge_a_prompts  = []
nudge_b_prompts  = []

for ex in examples:
    gl_a = group_label(CATEGORY, ex.group_a_tag)
    gl_b = group_label(CATEGORY, ex.group_b_tag)
    sg_a = singular_group_label(CATEGORY, ex.group_a_tag)
    sg_b = singular_group_label(CATEGORY, ex.group_b_tag)
    sent_a = f"({tmpl.template.format(group_label=gl_a, other_group_label=gl_b, singular_group_label=sg_a)})"
    sent_b = f"({tmpl.template.format(group_label=gl_b, other_group_label=gl_a, singular_group_label=sg_b)})"
    baseline_prompts.append(ex.prompt_with_sentence())
    nudge_a_prompts.append(ex.prompt_with_sentence(sent_a, position=tmpl.position))
    nudge_b_prompts.append(ex.prompt_with_sentence(sent_b, position=tmpl.position))

model = load_model("meta-llama/Llama-3.2-1B")

print("Collecting baseline logit diffs...")
base_diff    = collect_logit_diffs(model, baseline_prompts, n_choices=2, batch_size=4)   # (N,) A-B
print("Collecting nudge-A logit diffs...")
nudge_a_diff = collect_logit_diffs(model, nudge_a_prompts,  n_choices=2, batch_size=4)
print("Collecting nudge-B logit diffs...")
nudge_b_diff = collect_logit_diffs(model, nudge_b_prompts,  n_choices=2, batch_size=4)

# ── signed margin toward target slot per example ──────────────────────────────
# base_diff = logit_A - logit_B  (positive = model prefers A)
# For examples where target is slot A: margin toward target = base_diff
# For examples where target is slot B: margin toward target = -base_diff
# We determine "target" = the slot the model chose at baseline.

base_pred = (base_diff <= 0).astype(int)   # 0=A, 1=B

# signed margin toward whichever slot the model chose at baseline (= target)
def signed_margin_toward_target(diff, pred):
    """diff = logit_A - logit_B; pred = baseline choice (0=A,1=B)"""
    return np.where(pred == 0, diff, -diff)

base_margin   = signed_margin_toward_target(base_diff, base_pred)

# Nudged margin: use the nudge that matches baseline choice
nudge_diff_relevant = np.where(base_pred == 0, nudge_a_diff, nudge_b_diff)
nudge_pred_relevant = (nudge_diff_relevant <= 0).astype(int)

nudge_margin = signed_margin_toward_target(nudge_diff_relevant, base_pred)
shift        = nudge_margin - base_margin

# ── classify ──────────────────────────────────────────────────────────────────
strict_bf, stayed_m = [], []
for i in range(len(examples)):
    if base_pred[i] != nudge_pred_relevant[i]:
        strict_bf.append(i)
    else:
        stayed_m.append(i)

bf = np.array(strict_bf)
sm = np.array(stayed_m)
print(f"\nStrict backfire: {len(bf)}  Stayed-at-target: {len(sm)}")

# ── summary statistics ────────────────────────────────────────────────────────

def fmt(arr):
    return f"mean={arr.mean():+.4f}  std={arr.std():.4f}  min={arr.min():+.4f}  max={arr.max():+.4f}"

print(f"\n{'─'*60}")
print("BASELINE margin toward target slot (positive = prefer target)")
print(f"  Backfire    : {fmt(base_margin[bf])}")
print(f"  Stayed      : {fmt(base_margin[sm])}")
t1, p1 = stats.ttest_ind(base_margin[bf], base_margin[sm], equal_var=False)
print(f"  t={t1:.3f}  p={p1:.4f}")

print(f"\nNUDGED margin toward target slot (after nudge)")
print(f"  Backfire    : {fmt(nudge_margin[bf])}")
print(f"  Stayed      : {fmt(nudge_margin[sm])}")
t2, p2 = stats.ttest_ind(nudge_margin[bf], nudge_margin[sm], equal_var=False)
print(f"  t={t2:.3f}  p={p2:.4f}")

print(f"\nSHIFT (nudged - baseline margin)")
print(f"  Backfire    : {fmt(shift[bf])}")
print(f"  Stayed      : {fmt(shift[sm])}")
t3, p3 = stats.ttest_ind(shift[bf], shift[sm], equal_var=False)
print(f"  t={t3:.3f}  p={p3:.4f}")

print(f"\nKey: ALL backfire cases cross the decision boundary (margin goes negative)")
print(f"  Backfire   — base_margin > 0 AND nudge_margin <= 0: "
      f"{int(((base_margin[bf] > 0) & (nudge_margin[bf] <= 0)).sum())} / {len(bf)}")
print(f"  Stayed     — base_margin > 0 AND nudge_margin > 0:  "
      f"{int(((base_margin[sm] > 0) & (nudge_margin[sm] > 0)).sum())} / {len(sm)}")

# ── plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: baseline margin distributions
ax = axes[0]
bins = np.linspace(-2, 4, 30)
ax.hist(base_margin[sm], bins=bins, alpha=0.6, color="seagreen",
        label=f"stayed (n={len(sm)})", density=True)
ax.hist(base_margin[bf], bins=bins, alpha=0.75, color="firebrick",
        label=f"backfire (n={len(bf)})", density=True)
ax.axvline(0, color="black", linewidth=1.2, label="decision boundary")
ax.axvline(base_margin[bf].mean(), color="firebrick", linestyle="--", linewidth=1.5)
ax.axvline(base_margin[sm].mean(), color="seagreen",  linestyle="--", linewidth=1.5)
ax.set_xlabel("Margin toward target slot  (logit A−B, signed)")
ax.set_ylabel("Density")
ax.set_title(f"Baseline logit margin\n(t={t1:.2f}, p={p1:.4f})")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 2: nudged margin distributions
ax = axes[1]
ax.hist(nudge_margin[sm], bins=bins, alpha=0.6, color="seagreen",
        label=f"stayed (n={len(sm)})", density=True)
ax.hist(nudge_margin[bf], bins=bins, alpha=0.75, color="firebrick",
        label=f"backfire (n={len(bf)})", density=True)
ax.axvline(0, color="black", linewidth=1.2, label="decision boundary")
ax.axvline(nudge_margin[bf].mean(), color="firebrick", linestyle="--", linewidth=1.5)
ax.axvline(nudge_margin[sm].mean(), color="seagreen",  linestyle="--", linewidth=1.5)
ax.set_xlabel("Margin toward target slot  (logit A−B, signed)")
ax.set_ylabel("Density")
ax.set_title(f"Nudged logit margin\n(t={t2:.2f}, p={p2:.4f})")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel 3: before → after arrows per example
ax = axes[2]
# Plot each example as a line from baseline to nudged margin
for i in bf:
    ax.plot([0, 1], [base_margin[i], nudge_margin[i]],
            color="firebrick", alpha=0.35, linewidth=0.8)
for i in sm[:60]:   # cap stayed at 60 for readability
    ax.plot([0, 1], [base_margin[i], nudge_margin[i]],
            color="seagreen", alpha=0.18, linewidth=0.6)
# Group means
ax.plot([0, 1], [base_margin[bf].mean(), nudge_margin[bf].mean()],
        color="firebrick", linewidth=2.5, label=f"backfire mean", zorder=5)
ax.plot([0, 1], [base_margin[sm].mean(), nudge_margin[sm].mean()],
        color="seagreen", linewidth=2.5, label=f"stayed mean", zorder=5)
ax.axhline(0, color="black", linewidth=1.2, linestyle="--", label="decision boundary")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Baseline", "After nudge"])
ax.set_ylabel("Logit margin toward target slot")
ax.set_title("Trajectory: baseline → nudged\n(each line = one example)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle(
    f"Llama-3.2-1B | {CATEGORY} | user_preference nudge (bidirectional)\n"
    f"Logit margin toward nudge-target slot — backfire vs stayed-at-target",
    fontsize=10,
)
fig.tight_layout()
out = pathlib.Path("probes_out/backfire_logit_analysis.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved {out}")
plt.close(fig)
