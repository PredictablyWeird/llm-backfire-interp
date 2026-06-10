"""GPU PHASE — collect all model-dependent caches in one saturated run.

This is the *only* phase (besides ``run_live_experiments.py``) that needs the GPU.
It runs every (category, nudge) combination back-to-back, writing:

  cache/<slug>_logits.npz      base / nudge-stereo / nudge-other A/B/C logits + id arrays
  cache/<slug>_acts.npz        base / nudge-stereo / nudge-other resid_post per layer
  cache/<slug>_components.npz  (optional) per-layer mlp_out + attn_out, both directions
  cache/<model>_unembed.npz    A/B/C unembed columns + final-norm params (CPU projection)

Everything downstream (``analyze.py``) runs on these caches with no GPU.

Design for zero idle GPU time:
  * Idempotent / resumable: existing cache files are skipped, so a late crash costs
    one (category, nudge), not the whole run.
  * ``--smoke`` runs a tiny end-to-end pass first to validate before the long run.
  * Components (mlp/attn) capture is opt-in per category to control disk usage.

Examples
--------
    # Local validation on the small model (CPU/MPS ok):
    uv run --env-file .env python scripts/collect_cache.py \
        --model meta-llama/Llama-3.2-1B --device-map none --dtype float32 --smoke

    # Full run on a Lambda GPU box with Qwen3-32B:
    uv run --env-file .env python scripts/collect_cache.py \
        --model Qwen/Qwen3-32B --categories all --nudges all \
        --components-categories Gender_identity --batch-size 16
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from mech_interp_bbq.data import HITZ_CATEGORIES
from mech_interp_bbq.hf_backend import (
    capture_activations,
    compute_abc_logits,
    load_hf_model,
    save_unembed_meta,
)
from mech_interp_bbq.nudges import NUDGE_TEMPLATES
from mech_interp_bbq.prompts import build_prompts, cache_slug, model_cache_dir

# Nudges that inject a sentence (exclude the special-cased no-op ones).
RUNNABLE_NUDGES = tuple(
    k for k, v in NUDGE_TEMPLATES.items()
    if k not in ("few_shot", "role_play") and v.template
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--categories", nargs="+", default=["Gender_identity"],
                   help="Category names, or 'all'.")
    p.add_argument("--nudges", nargs="+", default=["user_preference"],
                   help="Nudge names, or 'all'.")
    p.add_argument("--condition", default="ambig", choices=["ambig", "disambig", "both"])
    p.add_argument("--max-examples", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device-map", default="auto", choices=["auto", "none"],
                   help="'auto' shards across GPUs; 'none' for single device / CPU smoke test.")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--components-categories", nargs="*", default=["Gender_identity"],
                   help="Categories to also capture per-layer mlp/attn outputs for "
                        "(needed by the component patch sweeps). Use 'none' to skip all.")
    p.add_argument("--smoke", action="store_true",
                   help="Run a tiny 16-example end-to-end pass first and exit on it "
                        "unless combined with a full category/nudge list.")
    p.add_argument("--force", action="store_true", help="Ignore existing caches.")
    return p.parse_args()


def _resolve(names: list[str], all_values: tuple[str, ...], what: str) -> list[str]:
    if names == ["all"]:
        return list(all_values)
    for n in names:
        if n not in all_values:
            raise SystemExit(f"Unknown {what}: {n!r}. Choices: {list(all_values)}")
    return names


def collect_one(
    lm,
    model_name: str,
    category: str,
    nudge: str,
    condition: str,
    max_examples: int,
    batch_size: int,
    cache_dir: Path,
    capture_components: bool,
    force: bool,
) -> None:
    bundle = build_prompts(category, nudge, condition, max_examples)
    n = bundle.n
    slug = cache_slug(model_name, category, nudge, condition, n)
    logits_path = cache_dir / f"{slug}_logits.npz"
    acts_path = cache_dir / f"{slug}_acts.npz"
    comp_path = cache_dir / f"{slug}_components.npz"

    print(f"\n=== {category} | {nudge} | n={n} ===", flush=True)

    # ── logits ────────────────────────────────────────────────────────────────
    if logits_path.exists() and not force:
        print(f"  [skip] logits cached: {logits_path.name}")
    else:
        t0 = time.time()
        print("  logits: baseline / nudge-stereo / nudge-other ...")
        base_logits = compute_abc_logits(lm, bundle.baseline_prompts, batch_size)
        ns_logits = compute_abc_logits(lm, bundle.nudge_stereo_prompts, batch_size)
        no_logits = compute_abc_logits(lm, bundle.nudge_other_prompts, batch_size)
        np.savez_compressed(
            logits_path,
            base_logits=base_logits,
            nudged_logits=ns_logits,          # original key name (backward compat)
            nudged_other_logits=no_logits,
            stereo_ids=bundle.stereo_ids,
            unknown_ids=bundle.unknown_ids,
            other_ids=bundle.other_ids,
            has_other_tag=bundle.has_other_tag,
        )
        print(f"  [save] {logits_path.name}  ({time.time() - t0:.0f}s)")

    # ── residual activations ───────────────────────────────────────────────────
    if acts_path.exists() and not force:
        print(f"  [skip] acts cached: {acts_path.name}")
    else:
        t0 = time.time()
        print("  resid_post: baseline / nudge-stereo / nudge-other ...")
        base = capture_activations(lm, bundle.baseline_prompts, batch_size, capture_components=False)
        ns = capture_activations(lm, bundle.nudge_stereo_prompts, batch_size, capture_components=False)
        no = capture_activations(lm, bundle.nudge_other_prompts, batch_size, capture_components=False)
        np.savez_compressed(
            acts_path,
            base_acts=base["resid"],
            nudge_stereo_acts=ns["resid"],
            nudge_other_acts=no["resid"],
        )
        print(f"  [save] {acts_path.name}  ({time.time() - t0:.0f}s)")

    # ── components (mlp + attn) ────────────────────────────────────────────────
    if not capture_components:
        return
    if comp_path.exists() and not force:
        print(f"  [skip] components cached: {comp_path.name}")
        return
    t0 = time.time()
    print("  components (mlp + attn per layer): baseline / nudge-stereo / nudge-other ...")
    base = capture_activations(lm, bundle.baseline_prompts, batch_size, capture_components=True)
    ns = capture_activations(lm, bundle.nudge_stereo_prompts, batch_size, capture_components=True)
    no = capture_activations(lm, bundle.nudge_other_prompts, batch_size, capture_components=True)
    np.savez_compressed(
        comp_path,
        base_mlp=base["mlp"], base_attn=base["attn"],
        nudge_stereo_mlp=ns["mlp"], nudge_stereo_attn=ns["attn"],
        nudge_other_mlp=no["mlp"], nudge_other_attn=no["attn"],
    )
    print(f"  [save] {comp_path.name}  ({time.time() - t0:.0f}s)")


def main() -> None:
    args = parse_args()
    cache_dir = model_cache_dir(args.cache_dir, args.model)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache directory (per-model): {cache_dir}")

    categories = _resolve(args.categories, HITZ_CATEGORIES, "category")
    nudges = _resolve(args.nudges, RUNNABLE_NUDGES, "nudge")
    comp_cats = set() if args.components_categories == ["none"] else set(args.components_categories)

    device_map = None if args.device_map == "none" else "auto"
    print(f"Loading model: {args.model}  (dtype={args.dtype}, device_map={device_map})", flush=True)
    t0 = time.time()
    lm = load_hf_model(args.model, dtype=args.dtype, device_map=device_map)
    print(f"  loaded in {time.time() - t0:.0f}s | n_layers={lm.n_layers} d_model={lm.d_model} "
          f"device={lm.device}", flush=True)

    unembed_path = cache_dir / "unembed.npz"
    if not unembed_path.exists() or args.force:
        save_unembed_meta(lm, str(unembed_path))
        print(f"  [save] {unembed_path.name}")

    # ── smoke test first ───────────────────────────────────────────────────────
    if args.smoke:
        print("\n--- SMOKE TEST (16 examples, Gender_identity / user_preference) ---")
        collect_one(
            lm, args.model, "Gender_identity", "user_preference", args.condition,
            max_examples=16, batch_size=min(args.batch_size, 4), cache_dir=cache_dir,
            capture_components=True, force=True,
        )
        print("--- SMOKE TEST PASSED ---\n")
        # Smoke caches are tiny (n=16) and harmless; the full run uses n>16 slugs.

    # ── full sweep ─────────────────────────────────────────────────────────────
    grand0 = time.time()
    for category in categories:
        for nudge in nudges:
            collect_one(
                lm, args.model, category, nudge, args.condition,
                args.max_examples, args.batch_size, cache_dir,
                capture_components=(category in comp_cats), force=args.force,
            )
    print(f"\nAll caches written in {(time.time() - grand0) / 60:.1f} min.")
    print(f"Cache dir: {cache_dir.resolve()}")


if __name__ == "__main__":
    main()
