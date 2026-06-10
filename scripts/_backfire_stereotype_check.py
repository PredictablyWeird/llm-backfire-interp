"""Check: after backfiring, does the model land on the stereotyped answer?

For each strict backfire example we determine which slot (A or B) is the
'stereotyped' answer using the polarity and stereotyped_groups metadata:

  neg polarity  → stereotype = blame the stigmatised group
                  → stereotyped slot = slot whose tag is in stereotyped_groups
  nonneg polarity → stereotype = credit the privileged group
                  → stereotyped slot = slot whose tag is NOT in stereotyped_groups

Then we check where the model landed after the flip.

Run:
    uv run --env-file .env python scripts/_backfire_stereotype_check.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mech_interp_bbq.activations import collect_model_predictions, load_model
from mech_interp_bbq.data import load_bbq_pairwise
from mech_interp_bbq.nudges import NUDGE_TEMPLATES, group_label, singular_group_label

# ── helpers ───────────────────────────────────────────────────────────────────

def stereotyped_slot(ex) -> int | None:
    """Return 0 (slot A) or 1 (slot B) that is the stereotyped answer.

    neg polarity:   stereotyped = blaming the stigmatised group (they're in sg)
    nonneg polarity: stereotyped = crediting the non-stigmatised/privileged group
    Returns None if the slot cannot be resolved.
    """
    sg_lower = {g.lower() for g in ex.stereotyped_groups}

    def is_sg(tag: str) -> bool:
        tc = tag.lower()
        return tc in sg_lower or any(g in tc or tc in g for g in sg_lower)

    a_sg = is_sg(ex.group_a_tag)
    b_sg = is_sg(ex.group_b_tag)

    if ex.polarity == "neg":
        if a_sg and not b_sg:
            return 0
        if b_sg and not a_sg:
            return 1
    else:   # nonneg: credit the privileged (non-sg) group
        if not a_sg and b_sg:
            return 0   # A is privileged
        if not b_sg and a_sg:
            return 1   # B is privileged
    return None


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

print("Collecting baseline predictions...")
base_preds    = collect_model_predictions(model, baseline_prompts, n_choices=2, batch_size=4)
print("Collecting nudge-A predictions...")
nudge_a_preds = collect_model_predictions(model, nudge_a_prompts,  n_choices=2, batch_size=4)
print("Collecting nudge-B predictions...")
nudge_b_preds = collect_model_predictions(model, nudge_b_prompts,  n_choices=2, batch_size=4)

# ── classify backfire examples ────────────────────────────────────────────────

strict_bf = []
stayed_m  = []
for i in range(len(examples)):
    bp  = base_preds[i]
    np_ = nudge_a_preds[i] if bp == 0 else nudge_b_preds[i]
    if bp != np_:
        strict_bf.append(i)
    else:
        stayed_m.append(i)

print(f"\nStrict backfire: {len(strict_bf)}  Stayed-at-target: {len(stayed_m)}")

# ── for each backfire example: did the flip land on the stereotype? ────────────

print(f"\n{'='*70}")
print("BACKFIRE examples: did the model flip TO the stereotype?")
print(f"{'='*70}\n")

flipped_to_stereotype   = []
flipped_to_nonstereotype = []
unresolvable            = []

for i in strict_bf:
    ex   = examples[i]
    bp   = base_preds[i]                                        # baseline choice (= nudge target X)
    np_  = nudge_a_preds[i] if bp == 0 else nudge_b_preds[i]   # post-flip choice Y
    ss   = stereotyped_slot(ex)                                  # stereotyped slot (or None)

    nudge_target_tag = ex.group_a_tag if bp == 0 else ex.group_b_tag
    nudge_target_gl  = group_label(CATEGORY, nudge_target_tag)

    if ss is None:
        outcome = "UNRESOLVABLE"
        unresolvable.append(i)
    elif np_ == ss:
        outcome = "→ STEREOTYPE ✓"
        flipped_to_stereotype.append(i)
    else:
        outcome = "→ NON-STEREOTYPE ✗"
        flipped_to_nonstereotype.append(i)

    slot_before = "A" if bp  == 0 else "B"
    slot_after  = "A" if np_ == 0 else "B"
    stereo_slot = ("A" if ss == 0 else "B") if ss is not None else "?"

    print(f"[{outcome}]  {slot_before}→{slot_after}  "
          f"(nudge→'{nudge_target_gl}', stereo_slot={stereo_slot}, "
          f"polarity={ex.polarity})")
    print(f"  Q: {ex.question}")
    print(f"  A. {ex.group_a_answer} (tag: {ex.group_a_tag})")
    print(f"  B. {ex.group_b_answer} (tag: {ex.group_b_tag})")
    print(f"  Stereotyped groups: {ex.stereotyped_groups}")
    print()

# ── summary ───────────────────────────────────────────────────────────────────

total = len(strict_bf)
n_stereo    = len(flipped_to_stereotype)
n_nonstereo = len(flipped_to_nonstereotype)
n_unresolv  = len(unresolvable)

print(f"{'='*70}")
print(f"SUMMARY  (total backfire cases = {total})")
print(f"{'='*70}")
print(f"  Flipped TO   stereotype : {n_stereo:>3}  ({100*n_stereo/total:.1f}%)")
print(f"  Flipped FROM stereotype : {n_nonstereo:>3}  ({100*n_nonstereo/total:.1f}%)")
print(f"  Unresolvable            : {n_unresolv:>3}  ({100*n_unresolv/total:.1f}%)")

# Also check baseline: how many were on the stereotype before?
print(f"\n  Baseline position (of all {total} backfire examples):")
on_stereo_baseline = sum(
    1 for i in strict_bf
    if stereotyped_slot(examples[i]) is not None
    and base_preds[i] == stereotyped_slot(examples[i])
)
off_stereo_baseline = sum(
    1 for i in strict_bf
    if stereotyped_slot(examples[i]) is not None
    and base_preds[i] != stereotyped_slot(examples[i])
)
print(f"    Were on stereotype at baseline    : {on_stereo_baseline}")
print(f"    Were off stereotype at baseline   : {off_stereo_baseline}")
print(f"    Unresolvable                      : {n_unresolv}")

# ── bar chart ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Panel 1: outcome of the flip
ax = axes[0]
labels_bar  = ["→ Stereotype", "→ Non-stereotype", "Unresolvable"]
counts_bar  = [n_stereo, n_nonstereo, n_unresolv]
colors_bar  = ["firebrick", "steelblue", "grey"]
bars = ax.bar(labels_bar, counts_bar, color=colors_bar, width=0.5, alpha=0.8)
for bar, cnt in zip(bars, counts_bar):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{cnt}\n({100*cnt/total:.0f}%)", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Number of backfire examples")
ax.set_title("After backfiring, model chose…")
ax.set_ylim(0, max(counts_bar) * 1.25)
ax.grid(alpha=0.3, axis="y")

# Panel 2: baseline position vs post-flip position (Sankey-style counts)
ax = axes[1]
# 2×2 matrix: rows = baseline (stereo/non-stereo), cols = post-flip (stereo/non-stereo)
# only for resolvable cases
mat = np.zeros((2, 2), dtype=int)
for i in strict_bf:
    ss = stereotyped_slot(examples[i])
    if ss is None:
        continue
    bp  = base_preds[i]
    np_ = nudge_a_preds[i] if bp == 0 else nudge_b_preds[i]
    row = 0 if bp  == ss else 1   # baseline: 0=stereo, 1=non-stereo
    col = 0 if np_ == ss else 1   # post-flip: 0=stereo, 1=non-stereo
    mat[row, col] += 1

im = ax.imshow(mat, cmap="YlOrRd", vmin=0)
plt.colorbar(im, ax=ax)
ax.set_xticks([0, 1]); ax.set_xticklabels(["→ Stereotype", "→ Non-stereotype"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Baseline:\nStereotype", "Baseline:\nNon-stereotype"])
ax.set_title("Baseline position → post-flip position")
for r in range(2):
    for c in range(2):
        ax.text(c, r, str(mat[r, c]), ha="center", va="center",
                fontsize=14, color="black")

fig.suptitle(
    f"Llama-3.2-1B | {CATEGORY} | user_preference nudge (bidirectional)\n"
    f"Does backfiring land on the stereotyped answer?",
    fontsize=10,
)
fig.tight_layout()
fig.savefig("probes_out/backfire_stereotype_check.png", dpi=150, bbox_inches="tight")
print("\nSaved probes_out/backfire_stereotype_check.png")
plt.close(fig)
