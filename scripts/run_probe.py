"""End-to-end smoke test:

  1. Load a slice of BBQ (one category, N examples).
  2. Run prompts through a small TransformerLens model.
  3. Collect last-token residual stream activations per layer.
  4. Train a linear probe per layer for the gold label and report accuracy.

Run with:

    uv run python scripts/run_probe.py --category Gender_identity --max-examples 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mech_interp_bbq.activations import collect_resid_post, load_model
from mech_interp_bbq.data import load_bbq, to_examples
from mech_interp_bbq.probes import train_all_layers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2-small")
    p.add_argument("--category", default="Gender_identity")
    p.add_argument("--max-examples", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", default="probes_out/result.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading BBQ category={args.category}, max_examples={args.max_examples}")
    ds = load_bbq(category=args.category, max_examples=args.max_examples)
    examples = to_examples(ds)
    print(f"  -> {len(examples)} examples")

    print(f"Loading model: {args.model}")
    model = load_model(args.model)

    prompts = [ex.prompt() for ex in examples]
    labels = np.array([ex.label for ex in examples], dtype=np.int64)

    print("Collecting activations")
    batch = collect_resid_post(model, prompts, batch_size=args.batch_size)
    print(f"  -> activations shape: {tuple(batch.acts.shape)}")

    print("Training per-layer probes")
    results = train_all_layers(batch.acts, labels)
    for r in results:
        print(f"  layer {r.layer:2d}: acc = {r.mean_accuracy:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "category": args.category,
                "n_examples": len(examples),
                "results": [
                    {
                        "layer": r.layer,
                        "mean_accuracy": r.mean_accuracy,
                        "fold_accuracies": r.fold_accuracies,
                    }
                    for r in results
                ],
            },
            indent=2,
        )
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
