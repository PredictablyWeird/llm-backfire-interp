"""Strict backfire analysis — bidirectional nudging.

For each example we run TWO nudges:
  1. Nudge toward group A  ("I personally think the answer involves <group_a_label>.")
  2. Nudge toward group B  ("I personally think the answer involves <group_b_label>.")

Then for each example, we pick the nudge that matches the baseline choice:
  - baseline chose A → use nudge-toward-A result
  - baseline chose B → use nudge-toward-B result

This makes EVERY example eligible for the backfire test, removing the
asymmetry of nudging only toward one fixed group.

Strict backfire (generic):
  baseline = X,  nudge → X,  result = Y      (X→Y flip)

Control:
  baseline = X,  nudge → X,  result = X      (stayed at target)

Activation analysis compares nudge-induced activation changes (delta)
between the backfire and stayed-at-target groups.
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch

from mech_interp_bbq.activations import collect_model_predictions, collect_resid_post, load_model
from mech_interp_bbq.data import load_bbq_pairwise
from mech_interp_bbq.nudges import NUDGE_TEMPLATES, group_label, singular_group_label
from mech_interp_bbq.probes import train_layer_probe

# ── setup ─────────────────────────────────────────────────────────────────────

CATEGORY = "Gender_identity"
examples = load_bbq_pairwise(CATEGORY, context_condition="ambig", max_examples=400)
tmpl     = NUDGE_TEMPLATES["user_preference"]

print(f"Loaded {len(examples)} examples")
print(f"Nudge template: {tmpl.template!r}")

# ── build per-example prompts for both nudge directions ───────────────────────

baseline_prompts = []
nudge_a_prompts  = []   # nudge toward group_a
nudge_b_prompts  = []   # nudge toward group_b

for ex in examples:
    gl_a = group_label(CATEGORY, ex.group_a_tag)
    gl_b = group_label(CATEGORY, ex.group_b_tag)
    sg_a = singular_group_label(CATEGORY, ex.group_a_tag)
    sg_b = singular_group_label(CATEGORY, ex.group_b_tag)

    sent_a = tmpl.template.format(group_label=gl_a, other_group_label=gl_b, singular_group_label=sg_a)
    sent_b = tmpl.template.format(group_label=gl_b, other_group_label=gl_a, singular_group_label=sg_b)
    if tmpl.brackets == "parentheses":
        sent_a, sent_b = f"({sent_a})", f"({sent_b})"

    baseline_prompts.append(ex.prompt_with_sentence())
    nudge_a_prompts.append(ex.prompt_with_sentence(sent_a, position=tmpl.position))
    nudge_b_prompts.append(ex.prompt_with_sentence(sent_b, position=tmpl.position))

# ── load model + collect predictions ─────────────────────────────────────────

model = load_model("meta-llama/Llama-3.2-1B")

print("\nCollecting baseline predictions...")
base_preds   = collect_model_predictions(model, baseline_prompts, n_choices=2, batch_size=4)
print("Collecting nudge-toward-A predictions...")
nudge_a_preds = collect_model_predictions(model, nudge_a_prompts,  n_choices=2, batch_size=4)
print("Collecting nudge-toward-B predictions...")
nudge_b_preds = collect_model_predictions(model, nudge_b_prompts,  n_choices=2, batch_size=4)

# ── classify every example ────────────────────────────────────────────────────
# For each example, pick the nudge that matches the baseline choice.
#   baseline = A (0) → relevant nudge = toward A → check nudge_a_pred
#   baseline = B (1) → relevant nudge = toward B → check nudge_b_pred

strict_bf = []   # baseline=X, nudge→X, result=Y
stayed_m  = []   # baseline=X, nudge→X, result=X
strict_cp = []   # baseline=Y, nudge→X, result=X  (for completeness — using opposite nudge)
stayed_f  = []   # baseline=Y, nudge→X, result=Y

for i in range(len(examples)):
    bp = base_preds[i]           # 0=A, 1=B  (baseline choice = X)
    if bp == 0:                  # baseline chose A → test nudge toward A
        np_ = nudge_a_preds[i]
    else:                        # baseline chose B → test nudge toward B
        np_ = nudge_b_preds[i]

    if bp == np_:
        stayed_m.append(i)       # nudge reinforced existing choice
    else:
        strict_bf.append(i)      # nudge caused a flip → backfire

# Also count comply cases (nudge toward the slot model did NOT choose at baseline)
for i in range(len(examples)):
    bp = base_preds[i]
    if bp == 0:
        opp_pred = nudge_b_preds[i]   # nudge toward B (opposite of baseline A)
    else:
        opp_pred = nudge_a_preds[i]   # nudge toward A (opposite of baseline B)
    if opp_pred != bp:
        strict_cp.append(i)
    else:
        stayed_f.append(i)

print(f"\n{'─'*55}")
print(f"Strict backfire  (X→Y, nudge toward X):  {len(strict_bf):>4}  ({100*len(strict_bf)/len(examples):.1f}%)")
print(f"Stayed at target (X→X, nudge toward X):  {len(stayed_m):>4}  ({100*len(stayed_m)/len(examples):.1f}%)")
print(f"Strict comply    (Y→X, nudge toward X):  {len(strict_cp):>4}  ({100*len(strict_cp)/len(examples):.1f}%)")
print(f"Stayed at other  (Y→Y, nudge toward X):  {len(stayed_f):>4}  ({100*len(stayed_f)/len(examples):.1f}%)")
print(f"Total eligible for backfire:             {len(examples):>4}  (100%)")
eligible = len(strict_bf) + len(stayed_m)
print(f"\nBackfire rate (of eligible):  {len(strict_bf)}/{eligible} = {100*len(strict_bf)/max(eligible,1):.1f}%")

# ── show backfire examples ────────────────────────────────────────────────────

def show(idx: int, label: str) -> None:
    ex = examples[idx]
    bp = base_preds[idx]
    if bp == 0:
        np_ = nudge_a_preds[idx]
        nudge_target = ex.group_a_tag
        nudge_label  = group_label(CATEGORY, ex.group_a_tag)
    else:
        np_ = nudge_b_preds[idx]
        nudge_target = ex.group_b_tag
        nudge_label  = group_label(CATEGORY, ex.group_b_tag)

    before = "A" if bp  == 0 else "B"
    after  = "A" if np_ == 0 else "B"
    print(f"[{label}]  baseline={before} → nudged={after}  (nudge→'{nudge_label}')")
    print(f"  polarity : {ex.polarity}")
    print(f"  context  : {ex.context}")
    print(f"  question : {ex.question}")
    slot_a = "A (← NUDGE TARGET X)" if bp == 0 else "A"
    slot_b = "B (← NUDGE TARGET X)" if bp == 1 else "B"
    print(f"  {slot_a}. {ex.group_a_answer}  (tag: {ex.group_a_tag})")
    print(f"  {slot_b}. {ex.group_b_answer}  (tag: {ex.group_b_tag})")
    print()

print(f"\n{'='*65}")
print("STRICT BACKFIRE examples  (baseline=X, nudge→X, result=Y)")
print(f"{'='*65}\n")
for i in strict_bf:
    show(i, "BACKFIRE")

print(f"{'='*65}")
print(f"STAYED AT TARGET (first {min(5, len(stayed_m))} shown)")
print(f"{'='*65}\n")
for i in stayed_m[:5]:
    show(i, "STAYED")

if len(strict_bf) < 3:
    print("Too few backfire examples for activation analysis.")
    raise SystemExit

# ── activation analysis ───────────────────────────────────────────────────────

print("Collecting baseline activations...")
base_acts = collect_resid_post(model, baseline_prompts, batch_size=4).acts.numpy()  # (N, L, D)

# Build "relevant nudged prompts" — for each example use the nudge that matched
# its baseline choice (the one that defines backfire)
relevant_nudged_prompts = [
    nudge_a_prompts[i] if base_preds[i] == 0 else nudge_b_prompts[i]
    for i in range(len(examples))
]
print("Collecting relevant (per-example) nudged activations...")
nudged_acts = collect_resid_post(model, relevant_nudged_prompts, batch_size=4).acts.numpy()

delta    = nudged_acts - base_acts    # (N, L, D)
n_layers = base_acts.shape[1]

bf_arr = np.array(strict_bf)
sm_arr = np.array(stayed_m)

# Load stereotype probe directions
coef_path = pathlib.Path("probes_out/bias_probe_meta-llama_Llama-3.2-1B_Gender_identity_coefs.npz")
stereo_coefs = np.load(coef_path)["stereotype_coef"]   # (L, n_classes, D)
stereo_dirs  = []
for l in range(n_layers):
    c = stereo_coefs[l]
    w = c[0] - c[-1]
    w /= (np.linalg.norm(w) + 1e-12)
    stereo_dirs.append(w)

# Per-layer stats
print(f"\n{'layer':>5}  {'cos(BF,SM)':>10}  {'L2_BF':>7}  {'L2_SM':>7}  {'align_BF':>9}  {'align_SM':>9}")
print("-" * 55)
cos_l, l2_bf_l, l2_sm_l, ab_l, as_l = [], [], [], [], []
for l in range(n_layers):
    d    = delta[:, l, :]
    mbf  = d[bf_arr].mean(axis=0)
    msm  = d[sm_arr].mean(axis=0)
    cos  = float(np.dot(mbf, msm) / (np.linalg.norm(mbf) * np.linalg.norm(msm) + 1e-12))
    l2b  = float(np.linalg.norm(mbf))
    l2s  = float(np.linalg.norm(msm))
    ab   = float(mbf @ stereo_dirs[l])
    as_  = float(msm @ stereo_dirs[l])
    cos_l.append(cos); l2_bf_l.append(l2b); l2_sm_l.append(l2s)
    ab_l.append(ab); as_l.append(as_)
    print(f"{l:>5}  {cos:>10.4f}  {l2b:>7.4f}  {l2s:>7.4f}  {ab:>9.4f}  {as_:>9.4f}")

# Probe: predict backfire vs stayed-at-target from nudged / baseline activations
mask   = np.concatenate([bf_arr, sm_arr])
labels = np.array([1]*len(bf_arr) + [0]*len(sm_arr), dtype=np.int64)
majority = float((labels == 0).mean())
print(f"\nProbe: predict backfire vs stayed-at-target")
print(f"Majority baseline: {majority:.3f}")

nudged_probe_accs = []
base_probe_accs   = []
for l in range(n_layers):
    rn = train_layer_probe(torch.from_numpy(nudged_acts[mask][:, l, :]), labels, layer=l)
    rb = train_layer_probe(torch.from_numpy(base_acts[mask][:, l, :]),   labels, layer=l)
    nudged_probe_accs.append(rn.mean_accuracy)
    base_probe_accs.append(rb.mean_accuracy)
    print(f"  layer {l:>2}: nudged={rn.mean_accuracy:.3f}  baseline={rb.mean_accuracy:.3f}")

# ── plot ──────────────────────────────────────────────────────────────────────

layers = list(range(n_layers))
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

ax = axes[0][0]
ax.plot(layers, cos_l, marker="o", color="steelblue")
ax.axhline(1, color="grey", linestyle=":", linewidth=0.8)
ax.set_ylim(0.8, 1.01)
ax.set_xlabel("Layer"); ax.set_ylabel("Cosine similarity")
ax.set_title("Similarity of mean activation change\nBackfire vs Stayed-at-target")
ax.grid(alpha=0.3)

ax = axes[0][1]
ax.plot(layers, l2_bf_l, marker="o", color="firebrick", label=f"backfire (n={len(bf_arr)})")
ax.plot(layers, l2_sm_l, marker="s", color="seagreen",  label=f"stayed-at-target (n={len(sm_arr)})")
ax.set_xlabel("Layer"); ax.set_ylabel("L2 norm of mean Δ")
ax.set_title("Magnitude of nudge-induced change")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1][0]
ax.plot(layers, ab_l, marker="o", color="firebrick", label="backfire")
ax.plot(layers, as_l, marker="s", color="seagreen",  label="stayed-at-target")
ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
ax.set_xlabel("Layer"); ax.set_ylabel("Alignment with stereotype direction")
ax.set_title("Nudge change vs stereotype direction")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1][1]
ax.plot(layers, nudged_probe_accs, marker="o", color="darkorchid", label="nudged acts")
ax.plot(layers, base_probe_accs,   marker="s", color="darkorange",  label="baseline acts")
ax.axhline(majority, color="grey", linestyle="-.", linewidth=0.8, label=f"majority ({majority:.2f})")
ax.set_xlabel("Layer"); ax.set_ylabel("5-fold CV accuracy")
ax.set_title("Probe: predict backfire\n(nudged vs baseline acts)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle(
    f"Llama-3.2-1B | {CATEGORY} | user_preference nudge (bidirectional)\n"
    f"Strict backfire (n={len(bf_arr)}) vs stayed-at-target (n={len(sm_arr)})",
    fontsize=11,
)
fig.tight_layout()
fig.savefig("probes_out/strict_backfire_corrected.png", dpi=150)
print("\nSaved probes_out/strict_backfire_corrected.png")
