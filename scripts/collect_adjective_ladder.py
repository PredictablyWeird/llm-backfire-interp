"""GPU — adjective-ladder compliance experiment (logits-only).

Investigates attributive adjectives before "answer" (t1→t6) across semantic scales,
and combo variants that merge each scale with the assertiveness ladder.

Pure:     "The {adj} answer is {group}."           (adj escalates per scale)
Combo:    assertiveness nudge + injected adjective  (e.g. "Perhaps the possible answer is …")

With --with-smileys: epistemic pure + combo_epistemic × smiley suffix profiles.

Output:
  cache/<model>/sensitivity_adjective_<variant>_<Category>.npz
  cache/<model>/sensitivity_adjective_smiley_<variant>_<Category>.npz

Example:
    uv run --env-file .env python scripts/collect_adjective_ladder.py \\
        --category Gender_identity --modes pure,combo --with-smileys
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from mech_interp_bbq.hf_backend import compute_abc_logits, load_hf_model
from mech_interp_bbq.prompts import model_cache_dir
from mech_interp_bbq.sensitivity import (
    ADJECTIVE_ALL_VARIANTS,
    ADJECTIVE_COMBO_VARIANTS,
    ADJECTIVE_LADDER_VARIANTS,
    ADJECTIVE_SMILEY_VARIANTS,
    build_examples,
    adjective_ladder,
    adjective_smiley_ladder,
    prompts_for,
)


def _parse_list(raw: str | None, all_keys: list[str]) -> list[str]:
    if raw is None or raw.strip().lower() in {"", "all"}:
        return sorted(all_keys)
    return [v.strip() for v in raw.split(",") if v.strip()]


def _parse_modes(raw: str) -> set[str]:
    modes = {m.strip().lower() for m in raw.split(",") if m.strip()}
    allowed = {"pure", "combo", "all"}
    bad = modes - allowed
    if bad:
        raise SystemExit(f"Unknown modes {bad}; choose from: pure, combo, all")
    if "all" in modes:
        return {"pure", "combo"}
    return modes or {"pure", "combo"}


def _variant_pool(modes: set[str], scales: list[str]) -> list[str]:
    out: list[str] = []
    if "pure" in modes:
        out.extend(s for s in scales if s in ADJECTIVE_LADDER_VARIANTS)
    if "combo" in modes:
        out.extend(f"combo_{s}" for s in scales if f"combo_{s}" in ADJECTIVE_COMBO_VARIANTS)
    return out


def _adjective_cache(out_dir: Path, category: str, variant: str) -> Path:
    return out_dir / f"sensitivity_adjective_{variant}_{category}.npz"


def _adjective_smiley_cache(out_dir: Path, category: str, variant: str) -> Path:
    return out_dir / f"sensitivity_adjective_smiley_{variant}_{category}.npz"


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
        help="Which families to collect: pure (The {adj} answer…), combo (+ assertiveness), all",
    )
    ap.add_argument(
        "--scales",
        default="all",
        help=f"Adjective scales (default: all). Choices: {', '.join(sorted(ADJECTIVE_LADDER_VARIANTS))}",
    )
    ap.add_argument(
        "--with-smileys",
        action="store_true",
        help="Also collect epistemic + combo_epistemic with each smiley suffix profile",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    modes = _parse_modes(args.modes)
    scales = _parse_list(args.scales, list(ADJECTIVE_LADDER_VARIANTS))
    variants = _variant_pool(modes, scales)
    for variant in variants:
        if variant not in ADJECTIVE_ALL_VARIANTS:
            known = ", ".join(sorted(ADJECTIVE_ALL_VARIANTS))
            raise SystemExit(f"Unknown variant {variant!r}; choose from: {known}")

    smiley_variants: list[str] = []
    if args.with_smileys:
        smiley_variants = sorted(ADJECTIVE_SMILEY_VARIANTS)

    rows = build_examples(args.category, args.max_examples)
    n = len(rows)
    print(
        f"{args.category}: n={n}  modes={sorted(modes)}  variants={variants}"
        + (f"  + smiley={smiley_variants}" if smiley_variants else ""),
        flush=True,
    )

    out_dir = model_cache_dir(args.cache_dir, args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, Path, list[str]]] = []
    for variant in variants:
        path = _adjective_cache(out_dir, args.category, variant)
        if args.force or not path.exists():
            jobs.append((variant, path, adjective_ladder(variant)))

    for variant in smiley_variants:
        path = _adjective_smiley_cache(out_dir, args.category, variant)
        if args.force or not path.exists():
            jobs.append((variant, path, adjective_smiley_ladder(variant)))

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
        "ladder_family": np.array("adjective"),
    }

    for label, out_path, levels in jobs:
        print(f"\n=== adjective/{label} ===", flush=True)
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
