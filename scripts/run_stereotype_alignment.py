"""Stereotype-direction alignment analysis.

Tests the hypothesis:
  "A nudge backfires when its activation change OPPOSES the model's internal
   stereotype direction; it complies when the change ALIGNS with it."

Method
------
1. Load the stereotype probe weight vector w_s (from run_bias_probe.py output).
   At each layer, w_s is the direction in residual-stream space that the probe
   learned to associate with "the stereotyped answer is here".

2. For each pairwise example, compute the nudge-induced change:
     delta[i, l] = nudged_acts[i, l] - base_acts[i, l]

3. Project delta onto the stereotype direction:
     alignment[i, l] = dot(delta[i, l], w_s[l]) / ||w_s[l]||

   Positive → nudge pushes activations toward the stereotype direction
   Negative → nudge pushes activations away from the stereotype direction

4. Check if alignment[i, l] at the best stereotype-probe layer predicts
   backfire (shift < 0) vs comply (shift > 0) better than chance.

5. Regress alignment vs logit shift across all examples.

Run with:

    uv run --env-file .env python scripts/run_stereotype_alignment.py \\
        --model meta-llama/Llama-3.2-1B \\
        --category Gender_identity \\
        --nudge user_preference \\
        --max-examples 400
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from mech_interp_bbq.activations import collect_logit_diffs, collect_resid_post, load_model
from mech_interp_bbq.data import HITZ_CATEGORIES, load_bbq_full, load_bbq_pairwise
from mech_interp_bbq.nudges import NUDGE_TEMPLATES, group_label, singular_group_label
from mech_interp_bbq.probes import train_all_layers, train_layer_probe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--category", default="Gender_identity", choices=list(HITZ_CATEGORIES))
    p.add_argument("--nudge", default="user_preference")
    p.add_argument("--max-examples", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--context-condition", default="ambig", choices=["ambig", "disambig", "both"])
    p.add_argument("--out-dir", default="probes_out")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def _stereotype_direction(coef: np.ndarray, stereotype_class_idx: int) -> np.ndarray:
    """Extract and normalise the stereotype weight vector from probe coefficients.

    ``coef`` has shape ``(n_classes, d_model)``; we pick the row corresponding
    to the stereotyped class and normalise to unit length.
    """
    w = coef[stereotype_class_idx]
    return w / (np.linalg.norm(w) + 1e-12)


def main() -> None:
    args = parse_args()

    model_slug = args.model.replace("/", "_")
    out_dir = Path(args.out_dir)

    # ── 1. load or compute stereotype probe directions ────────────────────────
    coef_path = out_dir / f"bias_probe_{model_slug}_{args.category}_coefs.npz"

    if coef_path.exists():
        print(f"Loading saved stereotype probe coefs from {coef_path}")
        saved = np.load(coef_path)
        stereotype_coefs = saved["stereotype_coef"]  # (n_layers, n_classes, d_model)
    else:
        print("Stereotype probe coefs not found — running bias probe first.")
        print("  Loading BBQ full examples for bias probe …")
        full_examples = load_bbq_full(
            args.category, context_condition="ambig", max_examples=args.max_examples
        )
        model = load_model(args.model)
        prompts_full = [ex.prompt() for ex in full_examples]

        from mech_interp_bbq.activations import collect_model_predictions
        model_preds = collect_model_predictions(model, prompts_full, batch_size=args.batch_size)
        act_batch = collect_resid_post(model, prompts_full, batch_size=args.batch_size)

        stereotype_labels = np.array([ex.stereotyped_ans_id for ex in full_examples], dtype=np.int64)
        gold_labels       = np.array([ex.label             for ex in full_examples], dtype=np.int64)

        stereo_results = train_all_layers(act_batch.acts, stereotype_labels)
        gold_results   = train_all_layers(act_batch.acts, gold_labels)
        pred_results   = train_all_layers(act_batch.acts, model_preds)

        stereotype_coefs = np.stack([r.coef for r in stereo_results], axis=0)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            coef_path,
            stereotype_coef=stereotype_coefs,
            gold_coef=np.stack([r.coef for r in gold_results], axis=0),
            model_pred_coef=np.stack([r.coef for r in pred_results], axis=0),
        )
        print(f"  Saved coefs to {coef_path}")

    n_layers   = stereotype_coefs.shape[0]
    n_classes  = stereotype_coefs.shape[1]
    # stereotype_class_idx: for a 3-class probe the "stereotyped" class is the
    # most common non-zero label; we use class 0 as a sensible default —
    # the probe's row 0 captures that class's direction.
    # We'll use ALL class directions summed (to get net stereotype signal).

    # ── 2. load pairwise examples ─────────────────────────────────────────────
    cond_arg = None if args.context_condition == "both" else args.context_condition
    print(f"\nLoading pairwise examples | category={args.category} | n<={args.max_examples}")
    examples = load_bbq_pairwise(
        args.category, context_condition=cond_arg, max_examples=args.max_examples
    )
    print(f"  -> {len(examples)} examples")

    target_tag = examples[0].group_a_tag
    other_tag  = examples[0].group_b_tag

    # ── 3. build prompts ──────────────────────────────────────────────────────
    tmpl = NUDGE_TEMPLATES[args.nudge]
    gl = group_label(args.category, target_tag)
    og = group_label(args.category, other_tag)
    sg = singular_group_label(args.category, target_tag)
    sentence = tmpl.template.format(group_label=gl, other_group_label=og, singular_group_label=sg)
    if tmpl.brackets == "parentheses":
        sentence = f"({sentence})"
    position = tmpl.position

    baseline_prompts = [ex.prompt_with_sentence() for ex in examples]
    nudged_prompts   = [ex.prompt_with_sentence(sentence, position=position) for ex in examples]

    # ── 4. load model + collect activations + logit diffs ────────────────────
    if not coef_path.exists() or "model" not in dir():
        print(f"\nLoading model: {args.model}")
        model = load_model(args.model)

    print("\nCollecting baseline activations")
    base_acts   = collect_resid_post(model, baseline_prompts, batch_size=args.batch_size).acts.numpy()
    print("Collecting nudged activations")
    nudged_acts = collect_resid_post(model, nudged_prompts,   batch_size=args.batch_size).acts.numpy()

    print("Collecting logit diffs")
    base_diffs   = collect_logit_diffs(model, baseline_prompts, n_choices=2, batch_size=args.batch_size)
    nudged_diffs = collect_logit_diffs(model, nudged_prompts,   n_choices=2, batch_size=args.batch_size)

    shifts = nudged_diffs - base_diffs     # positive = comply, negative = backfire
    delta  = nudged_acts - base_acts       # (n, n_layers, d_model)

    n_bf = int((shifts < 0).sum())
    n_cp = int((shifts > 0).sum())
    print(f"\nBackfire: {n_bf}, Comply: {n_cp}, Neutral: {int((shifts==0).sum())}")

    # ── 5. project delta onto stereotype direction at each layer ──────────────
    print("\n── Stereotype-direction alignment ──")

    # Build unit-norm stereotype direction for each layer.
    # The probe coef has shape (n_layers, n_classes, d_model).
    # We use the first principal component across all class weight rows as the
    # "stereotype axis" — equivalently, we sum class rows (class 0 vs rest).
    # For a binary concept it's enough to take coef[layer, 0] - coef[layer, 1].
    stereo_dirs = []
    for l in range(n_layers):
        coef_l = stereotype_coefs[l]    # (n_classes, d_model)
        if n_classes >= 2:
            # Direction = first class direction minus last (captures "stereotyped vs not")
            w = coef_l[0] - coef_l[-1]
        else:
            w = coef_l[0]
        w = w / (np.linalg.norm(w) + 1e-12)
        stereo_dirs.append(w)           # unit norm (d_model,)

    # alignment[i, l] = dot(delta[i,l], stereo_dir[l])
    alignment = np.stack(
        [delta[:, l, :] @ stereo_dirs[l] for l in range(n_layers)],
        axis=1,
    )   # (n, n_layers)

    print(f"\n{'layer':>5}  {'mean_align_bf':>14}  {'mean_align_cp':>14}  {'diff':>8}  {'r_vs_shift':>10}")
    print("-" * 62)
    corr_by_layer = []
    for l in range(n_layers):
        a = alignment[:, l]
        mean_bf = float(a[shifts < 0].mean()) if n_bf > 0 else float("nan")
        mean_cp = float(a[shifts > 0].mean()) if n_cp > 0 else float("nan")
        r, pval = stats.pearsonr(a, shifts)
        corr_by_layer.append((float(r), float(pval)))
        diff = mean_cp - mean_bf if not (np.isnan(mean_bf) or np.isnan(mean_cp)) else float("nan")
        print(f"{l:>5}  {mean_bf:>14.4f}  {mean_cp:>14.4f}  {diff:>8.4f}  {r:>10.4f}")

    # ── 6. probe: can alignment predict backfire? ─────────────────────────────
    print("\n── Probe: alignment score → backfire/comply ──")
    bf_labels = (shifts < 0).astype(np.int64)
    # Feature: alignment at each layer (scalar per example per layer)
    align_tensor = torch.from_numpy(alignment[:, :, np.newaxis].astype(np.float32))  # (n, n_layers, 1)
    align_probe = train_all_layers(align_tensor, bf_labels)

    majority = float(bf_labels.mean())
    print(f"Majority baseline: {majority:.3f}")
    for r in align_probe:
        print(f"  layer {r.layer:>2}: acc={r.mean_accuracy:.3f}")

    best_align = max(align_probe, key=lambda r: r.mean_accuracy)
    print(f"\nBest alignment-probe layer: {best_align.layer}  acc={best_align.mean_accuracy:.3f}")

    # ── 7. save ───────────────────────────────────────────────────────────────
    out_path = (
        out_dir / f"stereo_align_{model_slug}_{args.category}_{args.nudge}_{args.context_condition}.json"
    )
    out_path.write_text(json.dumps({
        "model": args.model,
        "category": args.category,
        "nudge": args.nudge,
        "n_examples": len(examples),
        "n_backfire": n_bf,
        "n_comply": n_cp,
        "mean_shift": float(shifts.mean()),
        "layers": [
            {
                "layer": l,
                "mean_alignment_backfire": float(alignment[shifts < 0, l].mean()) if n_bf > 0 else None,
                "mean_alignment_comply":   float(alignment[shifts > 0, l].mean()) if n_cp > 0 else None,
                "pearson_r_vs_shift": corr_by_layer[l][0],
                "pearson_p": corr_by_layer[l][1],
                "alignment_probe_acc": align_probe[l].mean_accuracy,
            }
            for l in range(n_layers)
        ],
    }, indent=2))
    print(f"\nWrote {out_path}")

    # ── 8. plot ───────────────────────────────────────────────────────────────
    if not args.no_plot:
        layers = list(range(n_layers))
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        # Panel 1: mean alignment by outcome across layers
        ax = axes[0]
        mean_bf_by_layer = [float(alignment[shifts < 0, l].mean()) if n_bf > 0 else np.nan for l in layers]
        mean_cp_by_layer = [float(alignment[shifts > 0, l].mean()) if n_cp > 0 else np.nan for l in layers]
        ax.plot(layers, mean_bf_by_layer, marker="o", color="firebrick",  label=f"backfire (n={n_bf})")
        ax.plot(layers, mean_cp_by_layer, marker="s", color="seagreen",   label=f"comply (n={n_cp})")
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean alignment with stereotype direction")
        ax.set_title("Does the nudge push activations\ntoward or away from the stereotype?")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Panel 2: Pearson r between alignment and logit shift across layers
        ax = axes[1]
        rs = [corr_by_layer[l][0] for l in layers]
        ax.plot(layers, rs, marker="o", color="steelblue")
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Pearson r (alignment vs logit shift)")
        ax.set_title("Correlation: stereotype alignment\n→ logit shift per example")
        ax.grid(alpha=0.3)

        # Panel 3: scatter at best corr layer
        best_r_layer = int(np.argmax([abs(c[0]) for c in corr_by_layer]))
        ax = axes[2]
        a = alignment[:, best_r_layer]
        sc = ax.scatter(a, shifts, c=shifts, cmap="RdYlGn", s=18, alpha=0.7, vmin=-0.5, vmax=0.5)
        plt.colorbar(sc, ax=ax, label="logit shift")
        # Regression line
        m, b, r, p, _ = stats.linregress(a, shifts)
        x_line = np.linspace(a.min(), a.max(), 100)
        ax.plot(x_line, m * x_line + b, color="black", linewidth=1.2,
                label=f"r={r:.3f}, p={p:.3g}")
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel(f"Alignment with stereotype dir (layer {best_r_layer})")
        ax.set_ylabel("Logit shift (nudged − baseline)")
        ax.set_title(f"Scatter at layer {best_r_layer}\n(green=comply, red=backfire)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        fig.suptitle(
            f"{args.model.split('/')[-1]} | {args.category} | nudge={args.nudge!r}\n"
            f"n={len(examples)}  backfire={n_bf}  comply={n_cp}",
            fontsize=10,
        )
        fig.tight_layout()
        plot_path = (
            out_dir / f"stereo_align_{model_slug}_{args.category}_{args.nudge}_{args.context_condition}.png"
        )
        fig.savefig(plot_path, dpi=150)
        print(f"Saved plot to {plot_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
