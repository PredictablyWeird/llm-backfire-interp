"""GPU — threatening-ladder compliance experiment (logits-only).

Stereo-directed nudges that escalate threat / coercion from mild pressure to
explicit warning (t1→t6). Optional --with-smileys combines threat text with
assertiveness-style emoji suffix profiles.

Output:
  cache/<model>/sensitivity_threatening_<variant>_<Category>.npz
  cache/<model>/sensitivity_threatening_smiley_<profile>_<Category>.npz  (--with-smileys)

Example:
    uv run --env-file .env python scripts/collect_threatening_ladder.py \\
        --category Gender_identity --variants plain \\
        --with-smileys --smiley-variants pressured,intense
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from mech_interp_bbq.hf_backend import compute_abc_logits, load_hf_model
from mech_interp_bbq.prompts import model_cache_dir
from mech_interp_bbq.sensitivity import (
    SMILEY_SUFFIX_PROFILES,
    THREATENING_LADDER_VARIANTS,
    build_examples,
    prompts_for,
    threatening_ladder,
    threatening_smiley_ladder,
)


def _parse_list(raw: str | None, all_keys: list[str]) -> list[str]:
    if raw is None or raw.strip().lower() in {"", "all"}:
        return sorted(all_keys)
    return [v.strip() for v in raw.split(",") if v.strip()]


def _threatening_cache(out_dir: Path, category: str, variant: str) -> Path:
    return out_dir / f"sensitivity_threatening_{variant}_{category}.npz"


def _threatening_smiley_cache(out_dir: Path, category: str, profile: str) -> Path:
    return out_dir / f"sensitivity_threatening_smiley_{profile}_{category}.npz"


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
        help=f"Threatening-native profiles (default: all). Choices: {', '.join(sorted(THREATENING_LADDER_VARIANTS))}",
    )
    ap.add_argument(
        "--with-smileys",
        action="store_true",
        help="Also collect threatening ladder + assertiveness-style emoji suffix profiles",
    )
    ap.add_argument(
        "--smiley-variants",
        default="all",
        help=f"Emoji profiles for --with-smileys (default: all). Choices: {', '.join(SMILEY_SUFFIX_PROFILES)}",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    variants = _parse_list(args.variants, list(THREATENING_LADDER_VARIANTS))
    for variant in variants:
        if variant not in THREATENING_LADDER_VARIANTS:
            known = ", ".join(sorted(THREATENING_LADDER_VARIANTS))
            raise SystemExit(f"Unknown threatening variant {variant!r}; choose from: {known}")

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
        f"{args.category}: n={n}  threatening={variants}"
        + (f"  + smiley profiles={smiley_profiles}" if smiley_profiles else ""),
        flush=True,
    )

    out_dir = model_cache_dir(args.cache_dir, args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, str, Path, list[str]]] = []
    for variant in variants:
        path = _threatening_cache(out_dir, args.category, variant)
        if args.force or not path.exists():
            jobs.append(("threatening", variant, path, threatening_ladder(variant)))

    for profile in smiley_profiles:
        path = _threatening_smiley_cache(out_dir, args.category, profile)
        if args.force or not path.exists():
            jobs.append(("threatening_smiley", profile, path, threatening_smiley_ladder(profile)))

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
        "ladder_family": np.array("threatening"),
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
        if kind == "threatening_smiley":
            saved["smiley_profile"] = np.array(label)
            saved["ladder_variant"] = np.array(f"smiley_{label}")
        np.savez(out_path, **saved)
        print(f"[save] {out_path}", flush=True)


if __name__ == "__main__":
    main()
