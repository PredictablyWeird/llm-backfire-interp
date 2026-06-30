"""GPU — small smiley-ladder compliance experiment (logits-only).

Compares stereo-directed assertiveness ladder **with** vs **without** emoji on each level.
Default: Gender_identity, n=500, ladder t1–t6 only (no rep axis).

Output:
  cache/<model>/sensitivity_smiley_<Category>.npz   — smiley ladder
  (compare to existing sensitivity_<Category>.npz for plain ladder, or use --also-plain)

Example:
    uv run --env-file .env python scripts/collect_smiley_ladder.py \\
        --category Gender_identity --max-examples 500 --also-plain
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from mech_interp_bbq.hf_backend import compute_abc_logits, load_hf_model
from mech_interp_bbq.prompts import model_cache_dir
from mech_interp_bbq.sensitivity import LADDER, LADDER_SMILEY, build_examples, prompts_for


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--category", default="Gender_identity")
    ap.add_argument("--max-examples", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device-map", default="auto", choices=["auto", "none"])
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument(
        "--also-plain",
        action="store_true",
        help="Also collect plain ladder into sensitivity_smiley_plain_<Category>.npz for matched n",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    rows = build_examples(args.category, args.max_examples)
    n = len(rows)
    print(f"{args.category}: n={n}  smiley ladder experiment", flush=True)

    out_dir = model_cache_dir(args.cache_dir, args.model)
    smiley_path = out_dir / f"sensitivity_smiley_{args.category}.npz"
    if smiley_path.exists() and not args.force:
        print(f"[skip] {smiley_path} exists")
        return

    device_map = None if args.device_map == "none" else "auto"
    lm = load_hf_model(args.model, dtype=args.dtype, device_map=device_map)
    print(f"model on {lm.device}", flush=True)

    def run(prompts: list[str], tag: str) -> np.ndarray:
        t0 = time.time()
        lg = compute_abc_logits(lm, prompts, args.batch_size)
        print(f"  [{tag}] {lg.shape} in {time.time() - t0:.0f}s", flush=True)
        return lg

    saved: dict[str, np.ndarray] = {
        "base_logits": run([r["base"] for r in rows], "base"),
        "ladder_levels": np.array(LADDER_SMILEY, dtype=object),
        "ladder_variant": np.array("smiley"),
        "stereo_ids": np.array([r["stereo_id"] for r in rows], dtype=np.int64),
        "unknown_ids": np.array([r["unknown_id"] for r in rows], dtype=np.int64),
        "other_ids": np.array([r["other_id"] for r in rows], dtype=np.int64),
        "has_other": np.array([r["has_other"] for r in rows], dtype=bool),
        "n_examples": np.array(n, dtype=np.int64),
    }

    ladder_smiley = np.stack(
        [run(prompts_for(rows, t, "stereo"), f"smiley/t{i + 1}") for i, t in enumerate(LADDER_SMILEY)],
        axis=1,
    )
    saved["ladder_stereo"] = ladder_smiley

    np.savez(smiley_path, **saved)
    print(f"\n[save] {smiley_path}")

    if args.also_plain:
        plain_path = out_dir / f"sensitivity_smiley_plain_{args.category}.npz"
        plain_ladder = np.stack(
            [run(prompts_for(rows, t, "stereo"), f"plain/t{i + 1}") for i, t in enumerate(LADDER)],
            axis=1,
        )
        plain_saved = dict(saved)
        plain_saved["ladder_stereo"] = plain_ladder
        plain_saved["ladder_levels"] = np.array(LADDER, dtype=object)
        plain_saved["ladder_variant"] = np.array("plain")
        np.savez(plain_path, **plain_saved)
        print(f"[save] {plain_path}")


if __name__ == "__main__":
    main()
