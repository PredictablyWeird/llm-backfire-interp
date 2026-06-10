"""Bias-emergence probing experiment.

For each layer of a HookedTransformer we train three linear probes on the
last-token residual-stream activations of BBQ *ambiguous* examples:

  1. ``gold``        – predict the correct answer (always "not enough info" for ambig)
  2. ``stereotype``  – predict the stereotyped answer index
  3. ``model_pred``  – predict what the model actually outputs (A/B/C argmax)

Comparing these curves layer-by-layer reveals:
  - Where in the network the stereotype is represented (stereotype probe peak)
  - Where the model commits to its (potentially biased) output (model_pred peak)
  - Whether stereotype encoding *precedes* the model's decision (causal implication)

Run with:

    uv run --env-file .env python scripts/run_bias_probe.py \
        --model meta-llama/Llama-3.2-1B \
        --category Gender_identity \
        --max-examples 400
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from mech_interp_bbq.activations import collect_model_predictions, collect_resid_post, load_model
from mech_interp_bbq.data import HITZ_CATEGORIES, load_bbq_full
from mech_interp_bbq.probes import train_all_layers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bias-emergence probe experiment")
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--category", default="Gender_identity", choices=list(HITZ_CATEGORIES))
    p.add_argument("--max-examples", type=int, default=400,
                   help="Max ambiguous examples to use")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--out-dir", default="probes_out")
    p.add_argument("--no-plot", action="store_true", help="Skip saving the plot")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. load data ─────────────────────────────────────────────────────────
    print(f"Loading BBQ (HiTZ) | category={args.category} | condition=ambig")
    examples = load_bbq_full(
        category=args.category,
        context_condition="ambig",
        max_examples=args.max_examples,
    )
    print(f"  -> {len(examples)} examples")

    # Sanity-check: gold label should always be the unknown answer for ambig
    bad = sum(1 for ex in examples if ex.label != ex.unknown_ans_id)
    if bad:
        print(f"  WARNING: {bad} examples where gold != unknown (unexpected for ambig)")

    # ── 2. load model + collect activations and predictions ──────────────────
    print(f"Loading model: {args.model}")
    model = load_model(args.model)

    prompts = [ex.prompt() for ex in examples]

    print("Collecting model predictions (A/B/C logits)")
    model_preds = collect_model_predictions(model, prompts, batch_size=args.batch_size)

    print("Collecting residual-stream activations")
    act_batch = collect_resid_post(model, prompts, batch_size=args.batch_size)
    print(f"  -> activations shape: {tuple(act_batch.acts.shape)}")

    # ── 3. define probe labels ───────────────────────────────────────────────
    gold_labels       = np.array([ex.label             for ex in examples], dtype=np.int64)
    stereotype_labels = np.array([ex.stereotyped_ans_id for ex in examples], dtype=np.int64)
    model_pred_labels = model_preds

    print("\nLabel distributions:")
    for name, arr in [("gold", gold_labels), ("stereotype", stereotype_labels), ("model_pred", model_pred_labels)]:
        counts = np.bincount(arr, minlength=3)
        print(f"  {name:12s}: {counts}  (chance={1/3:.3f})")

    # Fraction of examples where model chose the stereotyped answer
    bias_rate = np.mean(model_preds == stereotype_labels)
    print(f"\nModel bias rate (chose stereotype on ambig): {bias_rate:.3f}")

    # ── 4. train per-layer probes ────────────────────────────────────────────
    print("\nTraining per-layer probes...")
    results = {
        "gold":       train_all_layers(act_batch.acts, gold_labels),
        "stereotype": train_all_layers(act_batch.acts, stereotype_labels),
        "model_pred": train_all_layers(act_batch.acts, model_pred_labels),
    }

    print(f"\n{'layer':>5}  {'gold':>6}  {'stereotype':>10}  {'model_pred':>10}")
    print("-" * 40)
    for l in range(act_batch.acts.shape[1]):
        g  = results["gold"][l].mean_accuracy
        s  = results["stereotype"][l].mean_accuracy
        mp = results["model_pred"][l].mean_accuracy
        print(f"{l:>5}  {g:>6.3f}  {s:>10.3f}  {mp:>10.3f}")

    # ── 5. save results ───────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_slug = args.model.replace("/", "_")
    out_path = out_dir / f"bias_probe_{model_slug}_{args.category}.json"
    payload = {
        "model": args.model,
        "category": args.category,
        "n_examples": len(examples),
        "bias_rate": float(bias_rate),
        "layers": [
            {
                "layer": l,
                "gold_acc":       results["gold"][l].mean_accuracy,
                "stereotype_acc": results["stereotype"][l].mean_accuracy,
                "model_pred_acc": results["model_pred"][l].mean_accuracy,
            }
            for l in range(act_batch.acts.shape[1])
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")

    # Also save probe weight vectors so other scripts can load the stereotype direction
    coef_path = out_dir / f"bias_probe_{model_slug}_{args.category}_coefs.npz"
    np.savez(
        coef_path,
        stereotype_coef=np.stack([r.coef for r in results["stereotype"]], axis=0),
        gold_coef=np.stack([r.coef for r in results["gold"]], axis=0),
        model_pred_coef=np.stack([r.coef for r in results["model_pred"]], axis=0),
    )
    print(f"Wrote probe coefficients to {coef_path}")

    # ── 6. plot ───────────────────────────────────────────────────────────────
    if not args.no_plot:
        layers = list(range(act_batch.acts.shape[1]))
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(layers, [results["gold"][l].mean_accuracy       for l in layers],
                marker="o", label="gold label (correct answer)")
        ax.plot(layers, [results["stereotype"][l].mean_accuracy for l in layers],
                marker="s", linestyle="--", label="stereotype label (biased answer)")
        ax.plot(layers, [results["model_pred"][l].mean_accuracy for l in layers],
                marker="^", linestyle=":", label="model prediction")
        ax.axhline(1 / 3, color="grey", linestyle="-.", linewidth=0.8, label="chance (0.333)")
        ax.set_xlabel("Layer")
        ax.set_ylabel("5-fold CV accuracy")
        ax.set_title(
            f"Bias emergence probes — {args.model.split('/')[-1]} | {args.category}\n"
            f"(ambiguous examples only, n={len(examples)}, bias rate={bias_rate:.2f})"
        )
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()

        plot_path = out_dir / f"bias_probe_{model_slug}_{args.category}.png"
        fig.savefig(plot_path, dpi=150)
        print(f"Saved plot to {plot_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
