"""Nudge-influence activation probe experiment.

Inspired by the PredictablyWeird/Choices NeurIPS-2026 repo's BBQ adapter.

For each nudge type we:
  1. Load binary BBQ examples (unknown answer dropped → A vs B).
  2. Build two prompt versions per example:
       - baseline: no nudge
       - nudged  : nudge sentence injected toward the stereotyped group
  3. Collect last-token residual-stream activations for both.
  4. Train a binary probe per layer to predict whether a prompt is nudged.
  5. Separately measure how the nudge shifts the model's A/B prediction.

This lets us ask:
  - At which layer does each nudge type first leave a detectable trace?
  - Do nudges that strongly shift model output also shift intermediate layers?
  - Are evidence-based nudges (survey_preference) vs pressure-based nudges
    (emotional) detectable at different depths?

Run with:

    uv run --env-file .env python scripts/run_nudge_probe.py \\
        --model meta-llama/Llama-3.2-1B \\
        --category Gender_identity \\
        --nudges emotional survey_preference weak_evidence user_preference \\
        --max-examples 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mech_interp_bbq.activations import collect_model_predictions, collect_resid_post, load_model
from mech_interp_bbq.data import HITZ_CATEGORIES, load_bbq_pairwise
from mech_interp_bbq.nudges import (
    NUDGE_TEMPLATES,
    NUDGE_TYPES,
    group_label,
    singular_group_label,
)
from mech_interp_bbq.probes import train_all_layers

DEFAULT_NUDGES = [
    "emotional",
    "survey_preference",
    "weak_evidence",
    "user_preference",
    "virtue_appeal",
    "always",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nudge-influence activation probe experiment")
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--category", default="Gender_identity", choices=list(HITZ_CATEGORIES))
    p.add_argument(
        "--nudges",
        nargs="+",
        default=DEFAULT_NUDGES,
        choices=[n for n in NUDGE_TYPES if n not in ("few_shot", "role_play")],
        help="Nudge types to test",
    )
    p.add_argument("--max-examples", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--context-condition", default="ambig", choices=["ambig", "disambig", "both"])
    p.add_argument("--out-dir", default="probes_out")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def render_nudge_sentence(
    nudge_type: str,
    category: str,
    target_tag: str,
    other_tag: str,
) -> tuple[str, str]:
    """Return (nudge_sentence, position) for a given nudge type and group."""
    tmpl = NUDGE_TEMPLATES[nudge_type]
    gl = group_label(category, target_tag)
    og = group_label(category, other_tag)
    sg = singular_group_label(category, target_tag)
    sentence = tmpl.template.format(
        group_label=gl,
        other_group_label=og,
        singular_group_label=sg,
    )
    if tmpl.brackets == "parentheses":
        sentence = f"({sentence})"
    elif tmpl.brackets == "quotes":
        sentence = f'"{sentence}"'
    return sentence, tmpl.position


def main() -> None:
    args = parse_args()

    # ── 1. load examples ────────────────────────────────────────────────────
    cond_arg = None if args.context_condition == "both" else args.context_condition
    print(f"Loading BBQ pairwise | category={args.category} | condition={args.context_condition}")
    examples = load_bbq_pairwise(
        category=args.category,
        context_condition=cond_arg,
        max_examples=args.max_examples,
    )
    print(f"  -> {len(examples)} pairwise examples")
    if not examples:
        raise ValueError("No examples loaded. Check category / context_condition.")

    # Determine target group for nudging: use the first group_a_tag as canonical
    # (most examples share the same binary group pair within a category).
    target_tag = examples[0].group_a_tag
    other_tag = examples[0].group_b_tag
    print(f"  -> nudging toward '{target_tag}' vs '{other_tag}'")

    # ── 2. build prompt pairs ────────────────────────────────────────────────
    baseline_prompts = [ex.prompt_with_sentence() for ex in examples]

    # ── 3. load model ────────────────────────────────────────────────────────
    print(f"\nLoading model: {args.model}")
    model = load_model(args.model)

    # ── 4. baseline activations + predictions ────────────────────────────────
    print("Collecting baseline activations")
    base_acts = collect_resid_post(model, baseline_prompts, batch_size=args.batch_size)

    print("Collecting baseline model predictions (A=0, B=1)")
    base_preds = collect_model_predictions(
        model, baseline_prompts, n_choices=2, batch_size=args.batch_size
    )

    # ── 5. per-nudge experiment ───────────────────────────────────────────────
    results_by_nudge: dict[str, dict] = {}

    for nudge_type in args.nudges:
        print(f"\n── nudge: {nudge_type} ──")
        sentence, position = render_nudge_sentence(nudge_type, args.category, target_tag, other_tag)
        print(f"   sentence: {sentence!r}  position={position}")

        nudged_prompts = [
            ex.prompt_with_sentence(sentence, position=position) for ex in examples
        ]

        # Collect nudged activations
        print("   collecting nudged activations")
        nudged_acts = collect_resid_post(model, nudged_prompts, batch_size=args.batch_size)

        # Collect nudged model predictions
        print("   collecting nudged model predictions")
        nudged_preds = collect_model_predictions(
            model, nudged_prompts, n_choices=2, batch_size=args.batch_size
        )

        # Combine baseline + nudged for probe training
        # Label: 0 = baseline, 1 = nudged
        combined_acts = np.concatenate(
            [base_acts.acts.numpy(), nudged_acts.acts.numpy()], axis=0
        )
        combined_labels = np.array(
            [0] * len(examples) + [1] * len(examples), dtype=np.int64
        )

        import torch
        probe_results = train_all_layers(torch.from_numpy(combined_acts), combined_labels)

        # How much did the nudge shift the model toward target (group A)?
        base_chose_a = float(np.mean(base_preds == 0))
        nudged_chose_a = float(np.mean(nudged_preds == 0))
        shift = nudged_chose_a - base_chose_a

        print(f"   chose A: baseline={base_chose_a:.3f}  nudged={nudged_chose_a:.3f}  shift={shift:+.3f}")

        layer_accs = [r.mean_accuracy for r in probe_results]
        best_layer = int(np.argmax(layer_accs))
        print(f"   best probe layer: {best_layer}  acc={layer_accs[best_layer]:.3f}")

        results_by_nudge[nudge_type] = {
            "sentence": sentence,
            "position": position,
            "base_chose_a": base_chose_a,
            "nudged_chose_a": nudged_chose_a,
            "shift_toward_a": shift,
            "best_layer": best_layer,
            "best_layer_acc": layer_accs[best_layer],
            "layer_accs": layer_accs,
        }

    # ── 6. summary table ──────────────────────────────────────────────────────
    print(f"\n{'nudge':22s}  {'shift':>7}  {'best_layer':>10}  {'probe_acc':>9}")
    print("-" * 56)
    for nt, res in results_by_nudge.items():
        print(
            f"{nt:22s}  {res['shift_toward_a']:>+7.3f}  "
            f"{res['best_layer']:>10}  {res['best_layer_acc']:>9.3f}"
        )

    # ── 7. save JSON ─────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_slug = args.model.replace("/", "_")
    out_path = out_dir / f"nudge_probe_{model_slug}_{args.category}_{args.context_condition}.json"
    out_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "category": args.category,
                "context_condition": args.context_condition,
                "n_examples": len(examples),
                "target_tag": target_tag,
                "other_tag": other_tag,
                "nudges": results_by_nudge,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_path}")

    # ── 8. plot ───────────────────────────────────────────────────────────────
    if not args.no_plot:
        n_layers = len(next(iter(results_by_nudge.values()))["layer_accs"])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: layer probe accuracy curves
        ax = axes[0]
        for nt, res in results_by_nudge.items():
            ax.plot(range(n_layers), res["layer_accs"], marker="o", markersize=3, label=nt)
        ax.axhline(0.5, color="grey", linestyle="-.", linewidth=0.8, label="chance (0.5)")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Probe accuracy (nudged vs baseline)")
        ax.set_title("Nudge detectability per layer")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Right: output shift vs best probe accuracy scatter
        ax2 = axes[1]
        for nt, res in results_by_nudge.items():
            ax2.scatter(
                abs(res["shift_toward_a"]),
                res["best_layer_acc"],
                s=80,
                label=nt,
                zorder=3,
            )
            ax2.annotate(
                nt,
                (abs(res["shift_toward_a"]), res["best_layer_acc"]),
                fontsize=7,
                xytext=(4, 2),
                textcoords="offset points",
            )
        ax2.set_xlabel("|output shift toward group A|")
        ax2.set_ylabel("Best layer probe accuracy")
        ax2.set_title("Output shift vs internal detectability")
        ax2.axvline(0, color="grey", linestyle="--", linewidth=0.8)
        ax2.axhline(0.5, color="grey", linestyle="-.", linewidth=0.8)
        ax2.grid(alpha=0.3)

        fig.suptitle(
            f"{args.model.split('/')[-1]} | {args.category} ({args.context_condition})\n"
            f"Nudging toward '{target_tag}', n={len(examples)}",
            fontsize=10,
        )
        fig.tight_layout()
        plot_path = out_dir / f"nudge_probe_{model_slug}_{args.category}_{args.context_condition}.png"
        fig.savefig(plot_path, dpi=150)
        print(f"Saved plot to {plot_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
