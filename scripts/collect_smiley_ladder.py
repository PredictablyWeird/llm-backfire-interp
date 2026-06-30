"""GPU — smiley-ladder compliance experiment (logits-only).

Compares stereo-directed assertiveness ladder across emoji-intensity profiles.
Each profile keeps the same assertiveness text (LADDER) but escalates emoji affect t1→t6.

Output (per variant):
  cache/<model>/sensitivity_smiley_<variant>_<Category>.npz
Optional matched plain control:
  cache/<model>/sensitivity_smiley_plain_<Category>.npz

Example:
    uv run --env-file .env python scripts/collect_smiley_ladder.py \\
        --category Gender_identity --max-examples 500 \\
        --variants subtle,warm,enthusiastic,pressured,intense --skip-plain
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from mech_interp_bbq.hf_backend import compute_abc_logits, load_hf_model
from mech_interp_bbq.prompts import model_cache_dir
from mech_interp_bbq.sensitivity import (
    LADDER,
    SMILEY_LADDER_VARIANTS,
    build_examples,
    prompts_for,
    smiley_ladder,
)


def _parse_variants(raw: str | None) -> list[str]:
    if raw is None or raw.strip().lower() in {"", "all"}:
        return sorted(SMILEY_LADDER_VARIANTS)
    return [v.strip() for v in raw.split(",") if v.strip()]


def _cache_path(out_dir: Path, category: str, variant: str | None) -> Path:
    if variant is None:
        return out_dir / f"sensitivity_smiley_plain_{category}.npz"
    return out_dir / f"sensitivity_smiley_{variant}_{category}.npz"


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
        "--variants",
        default="all",
        help="Comma-separated smiley profiles (default: all). "
        f"Choices: {', '.join(sorted(SMILEY_LADDER_VARIANTS))}",
    )
    ap.add_argument(
        "--also-plain",
        action="store_true",
        help="Collect plain ladder into sensitivity_smiley_plain_<Category>.npz",
    )
    ap.add_argument(
        "--skip-plain",
        action="store_true",
        help="Do not collect plain even if --also-plain (reuse existing cache)",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    variants = _parse_variants(args.variants)
    for variant in variants:
        if variant not in SMILEY_LADDER_VARIANTS:
            known = ", ".join(sorted(SMILEY_LADDER_VARIANTS))
            raise SystemExit(f"Unknown variant {variant!r}; choose from: {known}")

    rows = build_examples(args.category, args.max_examples)
    n = len(rows)
    print(f"{args.category}: n={n}  smiley variants={variants}", flush=True)

    out_dir = model_cache_dir(args.cache_dir, args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    pending_variants = [
        v for v in variants if args.force or not _cache_path(out_dir, args.category, v).exists()
    ]
    need_plain = args.also_plain and not args.skip_plain
    plain_path = _cache_path(out_dir, args.category, None)
    if need_plain and not args.force and plain_path.exists():
        need_plain = False
        print(f"[skip plain] {plain_path} exists", flush=True)

    if not pending_variants and not need_plain:
        print("[skip] all requested caches already exist")
        return

    device_map = None if args.device_map == "none" else "auto"
    lm = load_hf_model(args.model, dtype=args.dtype, device_map=device_map)
    print(f"model on {lm.device}", flush=True)

    def run(prompts: list[str], tag: str) -> np.ndarray:
        t0 = time.time()
        lg = compute_abc_logits(lm, prompts, args.batch_size)
        print(f"  [{tag}] {lg.shape} in {time.time() - t0:.0f}s", flush=True)
        return lg

    base_logits = run([r["base"] for r in rows], "base")
    meta = {
        "base_logits": base_logits,
        "stereo_ids": np.array([r["stereo_id"] for r in rows], dtype=np.int64),
        "unknown_ids": np.array([r["unknown_id"] for r in rows], dtype=np.int64),
        "other_ids": np.array([r["other_id"] for r in rows], dtype=np.int64),
        "has_other": np.array([r["has_other"] for r in rows], dtype=bool),
        "n_examples": np.array(n, dtype=np.int64),
    }

    for variant in pending_variants:
        levels = smiley_ladder(variant)
        print(f"\n=== variant: {variant} ===", flush=True)
        ladder_smiley = np.stack(
            [run(prompts_for(rows, t, "stereo"), f"{variant}/t{i + 1}") for i, t in enumerate(levels)],
            axis=1,
        )
        saved = dict(meta)
        saved["ladder_stereo"] = ladder_smiley
        saved["ladder_levels"] = np.array(levels, dtype=object)
        saved["ladder_variant"] = np.array(variant)
        out_path = _cache_path(out_dir, args.category, variant)
        np.savez(out_path, **saved)
        print(f"[save] {out_path}", flush=True)

    if need_plain:
        print("\n=== plain control ===", flush=True)
        plain_ladder = np.stack(
            [run(prompts_for(rows, t, "stereo"), f"plain/t{i + 1}") for i, t in enumerate(LADDER)],
            axis=1,
        )
        plain_saved = dict(meta)
        plain_saved["ladder_stereo"] = plain_ladder
        plain_saved["ladder_levels"] = np.array(LADDER, dtype=object)
        plain_saved["ladder_variant"] = np.array("plain")
        np.savez(plain_path, **plain_saved)
        print(f"[save] {plain_path}", flush=True)


if __name__ == "__main__":
    main()
