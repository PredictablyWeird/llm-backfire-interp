"""Backfire analysis: do activations predict when a nudge will backfire?

Two questions:

  Q1 — Are the activation differences (nudged - baseline) distinguishable
       between backfire and compliance cases?
       Method: For each layer, compute the mean activation-difference vector
       for backfire examples and compliance examples separately, then compare
       (cosine similarity, L2 norm, and layer-by-layer plots).

  Q2 — Can we predict backfire from activations alone?
       Method: Train a binary logistic-regression probe per layer on the
       activation of nudged prompts to predict whether the nudge backfired
       (logit shift < 0) or complied (shift >= 0).

"Backfire" is defined per-example as:
    nudged_logit_diff < baseline_logit_diff
i.e. the nudge made the model *less* likely to pick the nudged group (A).

Run with:

    uv run --env-file .env python scripts/run_backfire_analysis.py \\
        --model meta-llama/Llama-3.2-1B \\
        --category Gender_identity \\
        --nudge user_preference \\
        --max-examples 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from mech_interp_bbq.activations import collect_logit_diffs, collect_resid_post, load_model
from mech_interp_bbq.data import HITZ_CATEGORIES, load_bbq_pairwise
from mech_interp_bbq.nudges import NUDGE_TEMPLATES, group_label, singular_group_label
from mech_interp_bbq.probes import train_all_layers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--category", default="Gender_identity", choices=list(HITZ_CATEGORIES))
    p.add_argument("--nudge", default="user_preference",
                   help="Nudge type to analyse (single nudge for deeper analysis)")
    p.add_argument("--max-examples", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--context-condition", default="ambig", choices=["ambig", "disambig", "both"])
    p.add_argument("--out-dir", default="probes_out")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. load examples ─────────────────────────────────────────────────────
    cond_arg = None if args.context_condition == "both" else args.context_condition
    print(f"Loading BBQ pairwise | category={args.category} | condition={args.context_condition}")
    examples = load_bbq_pairwise(
        args.category,
        context_condition=cond_arg,
        max_examples=args.max_examples,
    )
    print(f"  -> {len(examples)} examples")

    target_tag = examples[0].group_a_tag
    other_tag  = examples[0].group_b_tag
    print(f"  -> nudge toward '{target_tag}' (A), away from '{other_tag}' (B)")

    # ── 2. build nudge sentence ───────────────────────────────────────────────
    tmpl = NUDGE_TEMPLATES[args.nudge]
    gl  = group_label(args.category, target_tag)
    og  = group_label(args.category, other_tag)
    sg  = singular_group_label(args.category, target_tag)
    sentence = tmpl.template.format(
        group_label=gl, other_group_label=og, singular_group_label=sg
    )
    if tmpl.brackets == "parentheses":
        sentence = f"({sentence})"
    position = tmpl.position
    print(f"  -> nudge sentence: {sentence!r}  position={position}")

    baseline_prompts = [ex.prompt_with_sentence() for ex in examples]
    nudged_prompts   = [ex.prompt_with_sentence(sentence, position=position) for ex in examples]

    # ── 3. load model ─────────────────────────────────────────────────────────
    print(f"\nLoading model: {args.model}")
    model = load_model(args.model)

    # ── 4. collect logit diffs (continuous preference) ────────────────────────
    print("Collecting baseline logit diffs (log P(A) - log P(B))")
    base_diffs   = collect_logit_diffs(model, baseline_prompts, n_choices=2,
                                       batch_size=args.batch_size)
    print("Collecting nudged logit diffs")
    nudged_diffs = collect_logit_diffs(model, nudged_prompts,   n_choices=2,
                                       batch_size=args.batch_size)

    # Per-example shift: positive = compliance, negative = backfire
    shifts = nudged_diffs - base_diffs
    backfire_mask  = shifts < 0
    comply_mask    = shifts > 0
    neutral_mask   = shifts == 0

    n_bf  = int(backfire_mask.sum())
    n_cp  = int(comply_mask.sum())
    n_neu = int(neutral_mask.sum())
    print(f"\nPer-example breakdown (nudge={args.nudge!r}, target='{target_tag}'):")
    print(f"  backfire  (shift < 0): {n_bf:4d} / {len(examples)}  ({100*n_bf/len(examples):.1f}%)")
    print(f"  comply    (shift > 0): {n_cp:4d} / {len(examples)}  ({100*n_cp/len(examples):.1f}%)")
    print(f"  neutral   (shift = 0): {n_neu:4d} / {len(examples)}  ({100*n_neu/len(examples):.1f}%)")
    print(f"  mean shift           : {shifts.mean():.4f}")
    print(f"  median shift         : {np.median(shifts):.4f}")

    # ── 5. collect activations ────────────────────────────────────────────────
    print("\nCollecting baseline activations")
    base_act_batch   = collect_resid_post(model, baseline_prompts, batch_size=args.batch_size)
    print("Collecting nudged activations")
    nudged_act_batch = collect_resid_post(model, nudged_prompts,   batch_size=args.batch_size)

    base_acts   = base_act_batch.acts.numpy()    # (n, n_layers, d_model)
    nudged_acts = nudged_act_batch.acts.numpy()

    # Activation difference: nudged - baseline for each example
    diff_acts = nudged_acts - base_acts           # (n, n_layers, d_model)

    # ── Q1: are activation diffs distinguishable? ────────────────────────────
    print("\n── Q1: Comparing activation diffs (backfire vs comply) ──")

    n_layers = base_acts.shape[1]

    # Per-layer: mean diff vector for backfire vs comply, then cosine similarity
    cosine_bf_cp = []
    l2_bf  = []
    l2_cp  = []

    for l in range(n_layers):
        diff_l = diff_acts[:, l, :]   # (n, d_model)

        if n_bf > 0 and n_cp > 0:
            mean_bf = diff_l[backfire_mask].mean(axis=0)
            mean_cp = diff_l[comply_mask].mean(axis=0)
            cos = float(
                np.dot(mean_bf, mean_cp)
                / (np.linalg.norm(mean_bf) * np.linalg.norm(mean_cp) + 1e-12)
            )
            cosine_bf_cp.append(cos)
            l2_bf.append(float(np.linalg.norm(mean_bf)))
            l2_cp.append(float(np.linalg.norm(mean_cp)))
        else:
            cosine_bf_cp.append(float("nan"))
            l2_bf.append(float("nan"))
            l2_cp.append(float("nan"))

    print(f"  {'layer':>5}  {'cos(bf,cp)':>10}  {'L2(bf)':>8}  {'L2(cp)':>8}")
    print("  " + "-" * 40)
    for l in range(n_layers):
        print(f"  {l:>5}  {cosine_bf_cp[l]:>10.3f}  {l2_bf[l]:>8.4f}  {l2_cp[l]:>8.4f}")

    # ── Q2: can we predict backfire from nudged activations? ─────────────────
    print("\n── Q2: Backfire prediction probe ──")

    if n_bf < 5 or n_cp < 5:
        print("  Not enough backfire or comply examples for probe (need ≥5 each). Skipping.")
        bf_probe_results = None
    else:
        # Label: 1 = backfire, 0 = comply-or-neutral
        bf_labels = backfire_mask.astype(np.int64)
        bf_probe_results = train_all_layers(
            torch.from_numpy(nudged_acts), bf_labels
        )
        print(f"  {'layer':>5}  {'acc':>6}")
        print("  " + "-" * 14)
        for r in bf_probe_results:
            print(f"  {r.layer:>5}  {r.mean_accuracy:>6.3f}")
        best = max(bf_probe_results, key=lambda r: r.mean_accuracy)
        print(f"\n  Best layer: {best.layer}  acc={best.mean_accuracy:.3f}")
        print(
            f"  Baseline: {1 - bf_labels.mean():.3f} "
            f"(majority-class: predict all non-backfire)"
        )

    # ── 6. save ───────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_slug = args.model.replace("/", "_")
    out_path = (
        out_dir
        / f"backfire_{model_slug}_{args.category}_{args.nudge}_{args.context_condition}.json"
    )
    payload = {
        "model": args.model,
        "category": args.category,
        "nudge": args.nudge,
        "context_condition": args.context_condition,
        "n_examples": len(examples),
        "target_tag": target_tag,
        "other_tag": other_tag,
        "n_backfire": n_bf,
        "n_comply": n_cp,
        "n_neutral": n_neu,
        "mean_shift": float(shifts.mean()),
        "q1_cosine_bf_cp": cosine_bf_cp,
        "q1_l2_bf": l2_bf,
        "q1_l2_cp": l2_cp,
        "q2_probe_accs": (
            [r.mean_accuracy for r in bf_probe_results] if bf_probe_results else None
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")

    # ── 7. plot ───────────────────────────────────────────────────────────────
    if not args.no_plot:
        layers = list(range(n_layers))
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        # Panel 1: cosine similarity between backfire and comply diff vectors
        ax = axes[0]
        ax.plot(layers, cosine_bf_cp, marker="o", color="steelblue")
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.axhline(1, color="grey", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine similarity")
        ax.set_title("Similarity of backfire vs comply\nactivation-diff vectors")
        ax.set_ylim(-1.1, 1.1)
        ax.grid(alpha=0.3)

        # Panel 2: L2 norm of mean activation diff
        ax = axes[1]
        ax.plot(layers, l2_bf, marker="o", label=f"backfire (n={n_bf})", color="firebrick")
        ax.plot(layers, l2_cp, marker="s", label=f"comply (n={n_cp})", color="seagreen")
        ax.set_xlabel("Layer")
        ax.set_ylabel("L2 norm of mean (nudged−baseline)")
        ax.set_title("Magnitude of activation change\nby outcome")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Panel 3: backfire probe accuracy
        ax = axes[2]
        if bf_probe_results is not None:
            acc = [r.mean_accuracy for r in bf_probe_results]
            majority = 1 - backfire_mask.mean()
            ax.plot(layers, acc, marker="o", color="darkorchid")
            ax.axhline(majority, color="grey", linestyle="-.", linewidth=0.8,
                       label=f"majority baseline ({majority:.2f})")
            ax.set_xlabel("Layer")
            ax.set_ylabel("5-fold CV accuracy")
            ax.set_title("Probe: predict backfire\nfrom nudged activations")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, "Insufficient data\nfor probe",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Probe: predict backfire")

        fig.suptitle(
            f"{args.model.split('/')[-1]} | {args.category} | nudge={args.nudge!r}\n"
            f"n={len(examples)}  backfire={n_bf}  comply={n_cp}  neutral={n_neu}",
            fontsize=10,
        )
        fig.tight_layout()
        plot_path = (
            out_dir
            / f"backfire_{model_slug}_{args.category}_{args.nudge}_{args.context_condition}.png"
        )
        fig.savefig(plot_path, dpi=150)
        print(f"Saved plot to {plot_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
