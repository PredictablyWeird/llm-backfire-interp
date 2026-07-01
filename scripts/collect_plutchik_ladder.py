"""GPU — Plutchik emotion-wheel nudge experiment (logits-only).

Eight primary-emotion intensity ladders (t1→t6) plus a clockwise wheel ladder,
optional smiley suffix profiles on joy and anger.

Output:
  cache/<model>/sensitivity_plutchik_<variant>_<Category>.npz
  cache/<model>/sensitivity_plutchik_smiley_<variant>_<Category>.npz

Example:
    uv run --env-file .env python scripts/collect_plutchik_ladder.py \\
        --category Gender_identity --with-smileys
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from mech_interp_bbq.hf_backend import compute_abc_logits, load_hf_model
from mech_interp_bbq.prompts import model_cache_dir
from mech_interp_bbq.sensitivity import (
    PLUTCHIK_ALL_VARIANTS,
    PLUTCHIK_EMOTION_NAMES,
    PLUTCHIK_SMILEY_VARIANTS,
    PLUTCHIK_WHEEL_VARIANTS,
    build_examples,
    plutchik_ladder,
    plutchik_smiley_ladder,
    prompts_for,
)


def _parse_list(raw: str | None, all_keys: list[str]) -> list[str]:
    if raw is None or raw.strip().lower() in {"", "all"}:
        return sorted(all_keys)
    return [v.strip() for v in raw.split(",") if v.strip()]


def _parse_modes(raw: str) -> set[str]:
    modes = {m.strip().lower() for m in raw.split(",") if m.strip()}
    if "all" in modes:
        return {"intensity", "wheel"}
    return modes or {"intensity", "wheel"}


def _variant_pool(modes: set[str], emotions: list[str], include_wheel: bool) -> list[str]:
    out: list[str] = []
    if "intensity" in modes:
        out.extend(e for e in emotions if e in PLUTCHIK_EMOTION_NAMES)
    if "wheel" in modes and include_wheel:
        out.append("wheel")
    return out


def _plutchik_cache(out_dir: Path, category: str, variant: str) -> Path:
    return out_dir / f"sensitivity_plutchik_{variant}_{category}.npz"


def _plutchik_smiley_cache(out_dir: Path, category: str, variant: str) -> Path:
    return out_dir / f"sensitivity_plutchik_smiley_{variant}_{category}.npz"


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
        "--modes",
        default="all",
        help="intensity (8 emotion ladders), wheel (clockwise 6-step), or all",
    )
    ap.add_argument(
        "--emotions",
        default="all",
        help=f"Emotion families for intensity mode (default: all). Choices: {', '.join(PLUTCHIK_EMOTION_NAMES)}",
    )
    ap.add_argument(
        "--with-smileys",
        action="store_true",
        help="Also collect joy + anger intensity ladders with each smiley suffix profile",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    modes = _parse_modes(args.modes)
    emotions = _parse_list(args.emotions, list(PLUTCHIK_EMOTION_NAMES))
    include_wheel = args.emotions.strip().lower() in {"", "all"} or "wheel" in emotions
    if "wheel" in emotions:
        include_wheel = True
        emotions = [e for e in emotions if e != "wheel"]

    variants = _variant_pool(modes, emotions, include_wheel)
    for variant in variants:
        if variant not in PLUTCHIK_ALL_VARIANTS:
            known = ", ".join(sorted(PLUTCHIK_ALL_VARIANTS))
            raise SystemExit(f"Unknown variant {variant!r}; choose from: {known}")

    smiley_variants: list[str] = []
    if args.with_smileys:
        smiley_variants = sorted(PLUTCHIK_SMILEY_VARIANTS)

    rows = build_examples(args.category, args.max_examples)
    n = len(rows)
    print(
        f"{args.category}: n={n}  modes={sorted(modes)}  variants={variants}"
        + (f"  + smiley={len(smiley_variants)}" if smiley_variants else ""),
        flush=True,
    )

    out_dir = model_cache_dir(args.cache_dir, args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, Path, list[str]]] = []
    for variant in variants:
        path = _plutchik_cache(out_dir, args.category, variant)
        if args.force or not path.exists():
            jobs.append((variant, path, plutchik_ladder(variant)))

    for variant in smiley_variants:
        path = _plutchik_smiley_cache(out_dir, args.category, variant)
        if args.force or not path.exists():
            jobs.append((variant, path, plutchik_smiley_ladder(variant)))

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
        "ladder_family": np.array("plutchik"),
    }

    for label, out_path, levels in jobs:
        print(f"\n=== plutchik/{label} ===", flush=True)
        ladder = np.stack(
            [run(prompts_for(rows, t, "stereo"), f"{label}/t{i + 1}") for i, t in enumerate(levels)],
            axis=1,
        )
        saved = dict(meta)
        saved["ladder_stereo"] = ladder
        saved["ladder_levels"] = np.array(levels, dtype=object)
        saved["ladder_variant"] = np.array(label)
        np.savez(out_path, **saved)
        print(f"[save] {out_path}", flush=True)


if __name__ == "__main__":
    main()
