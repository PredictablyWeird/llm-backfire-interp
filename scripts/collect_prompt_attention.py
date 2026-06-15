"""GPU phase — collect last-token attention mass by BBQ prompt region.

Pairs with ``sensitivity_<Category>.npz`` (same example order from ``build_examples``).
Stores per-layer region fractions for baseline and ladder-t3-stereo prompts.

Example (smoke):
    uv run python scripts/collect_prompt_attention.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct --device-map none --dtype float32 \\
        --categories Gender_identity --max-examples 32 --batch-size 4

Full run (GPU):
    uv run --env-file .env python scripts/collect_prompt_attention.py \\
        --model Qwen/Qwen3-32B --categories Gender_identity SES Race_ethnicity
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from mech_interp_bbq.hf_backend import capture_prompt_region_attention, load_hf_model
from mech_interp_bbq.prompt_regions import REGION_NAMES, append_ladder_nudge
from mech_interp_bbq.prompts import model_cache_dir

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nudge_sensitivity import LADDER, build_examples

T3_TEMPLATE = LADDER[2]
DEFAULT_CATS = ("Gender_identity", "SES", "Race_ethnicity")


def _stratified_indices(
    cache_dir: Path,
    category: str,
    n_per_class: int,
    seed: int = 0,
) -> np.ndarray:
    """Sample n_per_class compliers + n_per_class non-compliers at t3."""
    sens = np.load(cache_dir / f"sensitivity_{category}.npz")
    stereo = sens["stereo_ids"]
    comply = sens["ladder_stereo"][:, 2, :].argmax(1) == stereo
    rng = np.random.default_rng(seed)
    idx_yes = np.where(comply)[0]
    idx_no = np.where(~comply)[0]
    k = min(n_per_class, len(idx_yes), len(idx_no))
    if k == 0:
        raise SystemExit(f"Cannot stratify {category}: empty compliance class")
    pick = np.concatenate([
        rng.choice(idx_yes, k, replace=False),
        rng.choice(idx_no, k, replace=False),
    ])
    return np.sort(pick)


def collect_category(
    lm,
    category: str,
    out_path: Path,
    max_examples: int | None,
    batch_size: int,
    layer_indices: list[int] | None,
    stratified: int | None,
    cache_dir: Path,
    force: bool,
) -> None:
    if out_path.exists() and not force:
        print(f"[skip] {out_path} exists")
        return

    rows = build_examples(category, max_examples or 10_000)
    base_prompts = [r["base"] for r in rows]
    t3_prompts = [
        append_ladder_nudge(r["base"], T3_TEMPLATE.format(group=r["stereo_gl"]))
        for r in rows
    ]
    example_ids = np.array([r["example_id"] for r in rows], dtype=np.int64)

    indices = np.arange(len(base_prompts))
    if stratified is not None:
        indices = _stratified_indices(cache_dir, category, stratified)
        base_prompts = [base_prompts[i] for i in indices]
        t3_prompts = [t3_prompts[i] for i in indices]
        example_ids = example_ids[indices]

    n = len(base_prompts)
    print(f"\n{category}: n={n}  layers={layer_indices or 'all'}", flush=True)

    t0 = time.time()
    print("  baseline attention ...", flush=True)
    mass_base = capture_prompt_region_attention(
        lm, base_prompts, batch_size=batch_size, layer_indices=layer_indices
    )
    print("  t3_stereo attention ...", flush=True)
    mass_t3 = capture_prompt_region_attention(
        lm, t3_prompts, batch_size=batch_size, layer_indices=layer_indices
    )
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.0f}s  shape={mass_base.shape}", flush=True)

    layers = (
        np.array(layer_indices, dtype=np.int64)
        if layer_indices is not None
        else np.arange(lm.n_layers, dtype=np.int64)
    )

    np.savez_compressed(
        out_path,
        region_mass_baseline=mass_base,
        region_mass_t3=mass_t3,
        example_ids=example_ids,
        sample_indices=indices.astype(np.int64),
        stereo_ids=np.array([rows[i]["stereo_id"] for i in indices], dtype=np.int64),
        unknown_ids=np.array([rows[i]["unknown_id"] for i in indices], dtype=np.int64),
        other_ids=np.array([rows[i]["other_id"] for i in indices], dtype=np.int64),
        layers=layers,
        region_names=np.array(REGION_NAMES),
        t3_template=np.array(T3_TEMPLATE),
    )
    print(f"[save] {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_CATS),
        help=f"Default: {DEFAULT_CATS}",
    )
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device-map", default="auto", choices=["auto", "none"])
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="Store only these layer indices (default: all)",
    )
    ap.add_argument(
        "--stratified",
        type=int,
        default=None,
        metavar="N",
        help="Use N compliers + N non-compliers (requires sensitivity cache)",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device_map = None if args.device_map == "none" else "auto"
    lm = load_hf_model(
        args.model,
        dtype=args.dtype,
        device_map=device_map,
        attn_implementation="eager",
    )
    print(f"model={args.model}  device={lm.device}  L={lm.n_layers}  d={lm.d_model}", flush=True)

    out_dir = model_cache_dir(args.cache_dir, args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    for category in args.categories:
        out_path = out_dir / f"prompt_attn_{category}.npz"
        collect_category(
            lm,
            category,
            out_path,
            args.max_examples,
            args.batch_size,
            args.layers,
            args.stratified,
            out_dir,
            args.force,
        )


if __name__ == "__main__":
    main()
