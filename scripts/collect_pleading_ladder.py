"""GPU — pleading-ladder compliance experiment (logits-only).

Alternative to the assertiveness ladder: stereo-directed nudges that escalate
pleading / supplication from mild suggestion to explicit begging (t1→t6).

Output (per pleading variant):
  cache/<model>/sensitivity_pleading_<variant>_<Category>.npz

With --with-smileys (pleading text + assertiveness-style emoji profiles):
  cache/<model>/sensitivity_pleading_smiley_<profile>_<Category>.npz

Example:
    uv run --env-file .env python scripts/collect_pleading_ladder.py \\
        --category Gender_identity --variants plain \\
        --with-smileys --smiley-variants warm,pressured
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from mech_interp_bbq.hf_backend import compute_abc_logits, load_hf_model
from mech_interp_bbq.prompts import model_cache_dir
from mech_interp_bbq.sensitivity import (
    PLEADING_LADDER_VARIANTS,
    SMILEY_SUFFIX_PROFILES,
    build_examples,
    pleading_ladder,
    pleading_smiley_ladder,
    prompts_for,
)


def _parse_list(raw: str | None, all_keys: list[str]) -> list[str]:
    if raw is None or raw.strip().lower() in {"", "all"}:
        return sorted(all_keys)
    return [v.strip() for v in raw.split(",") if v.strip()]


def _pleading_cache(out_dir: Path, category: str, variant: str) -> Path:
    return out_dir / f"sensitivity_pleading_{variant}_{category}.npz"


def _pleading_smiley_cache(out_dir: Path, category: str, profile: str) -> Path:
    return out_dir / f"sensitivity_pleading_smiley_{profile}_{category}.npz"


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
        help=f"Pleading-native profiles (default: all). Choices: {', '.join(sorted(PLEADING_LADDER_VARIANTS))}",
    )
    ap.add_argument(
        "--with-smileys",
        action="store_true",
        help="Also collect pleading ladder + assertiveness-style emoji suffix profiles",
    )
    ap.add_argument(
        "--smiley-variants",
        default="all",
        help=f"Emoji profiles for --with-smileys (default: all). Choices: {', '.join(SMILEY_SUFFIX_PROFILES)}",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    variants = _parse_list(args.variants, list(PLEADING_LADDER_VARIANTS))
    for variant in variants:
        if variant not in PLEADING_LADDER_VARIANTS:
            known = ", ".join(sorted(PLEADING_LADDER_VARIANTS))
            raise SystemExit(f"Unknown pleading variant {variant!r}; choose from: {known}")

    smiley_profiles: list[str] = []
    if args.with_smileys:
        smiley_profiles = _parse_list(args.smiley_variants, list(SMILEY_SUFFIX_PROFILES))
        for profile in smiley_profiles:
            if profile not in SMILEY_SUFFIX_PROFILES:
                known = ", ".join(SMILEY_SUFFIX_PROFILES)
                raise SystemExit(f"Unknown smiley profile {profile!r}; choose from: {known}")

    rows = build_examples(args.category, args.max_examples)
    n = len(rows)
    print(
        f"{args.category}: n={n}  pleading={variants}"
        + (f"  + smiley profiles={smiley_profiles}" if smiley_profiles else ""),
        flush=True,
    )

    out_dir = model_cache_dir(args.cache_dir, args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, str, Path, list[str]]] = []
    for variant in variants:
        path = _pleading_cache(out_dir, args.category, variant)
        if args.force or not path.exists():
            jobs.append(("pleading", variant, path, pleading_ladder(variant)))

    for profile in smiley_profiles:
        path = _pleading_smiley_cache(out_dir, args.category, profile)
        if args.force or not path.exists():
            jobs.append(("pleading_smiley", profile, path, pleading_smiley_ladder(profile)))

    if not jobs:
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
        "ladder_family": np.array("pleading"),
    }

    for kind, label, out_path, levels in jobs:
        tag_prefix = f"{kind}/{label}"
        print(f"\n=== {tag_prefix} ===", flush=True)
        ladder = np.stack(
            [run(prompts_for(rows, t, "stereo"), f"{tag_prefix}/t{i + 1}") for i, t in enumerate(levels)],
            axis=1,
        )
        saved = dict(meta)
        saved["ladder_stereo"] = ladder
        saved["ladder_levels"] = np.array(levels, dtype=object)
        saved["ladder_variant"] = np.array(label)
        if kind == "pleading_smiley":
            saved["smiley_profile"] = np.array(label)
            saved["ladder_variant"] = np.array(f"smiley_{label}")
        np.savez(out_path, **saved)
        print(f"[save] {out_path}", flush=True)


if __name__ == "__main__":
    main()
