"""CPU — compare plain vs smiley ladder stereo compliance across emoji profiles.

Uses:
  * sensitivity_smiley_<variant>_<Category>.npz  (one or more variants)
  * sensitivity_smiley_plain_<Category>.npz  (matched plain, preferred)
  * OR legacy sensitivity_smiley_<Category>.npz / sensitivity_<Category>.npz

Example:
    uv run python scripts/analyze_smiley_compliance.py --category Gender_identity --variants all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mech_interp_bbq.prompts import model_cache_dir
from mech_interp_bbq.sensitivity import SMILEY_LADDER_VARIANTS


def _comply(logits: np.ndarray, stereo_ids: np.ndarray) -> np.ndarray:
    return logits.argmax(1) == stereo_ids


def _level_rates(ladder: np.ndarray, stereo_ids: np.ndarray) -> list[float]:
    return [float(_comply(ladder[:, t, :], stereo_ids).mean()) for t in range(ladder.shape[1])]


def _parse_variants(raw: str | None) -> list[str]:
    if raw is None or raw.strip().lower() in {"", "all"}:
        return sorted(SMILEY_LADDER_VARIANTS)
    return [v.strip() for v in raw.split(",") if v.strip()]


def _load_plain(cache_dir: Path, category: str) -> tuple[np.ndarray, np.ndarray, str, list[str]]:
    matched = cache_dir / f"sensitivity_smiley_plain_{category}.npz"
    if matched.exists():
        d = np.load(matched, allow_pickle=True)
        levels = [str(x) for x in d["ladder_levels"]]
        return d["ladder_stereo"], d["stereo_ids"].astype(int), str(matched.name), levels

    legacy = cache_dir / f"sensitivity_{category}.npz"
    if legacy.exists():
        d = np.load(legacy)
        levels = [str(x) for x in d["ladder_levels"]] if "ladder_levels" in d else []
        return d["ladder_stereo"], d["stereo_ids"].astype(int), str(legacy.name), levels

    raise FileNotFoundError(f"No plain ladder cache in {cache_dir}")


def _resolve_smiley_path(cache_dir: Path, category: str, variant: str) -> Path:
    named = cache_dir / f"sensitivity_smiley_{variant}_{category}.npz"
    if named.exists():
        return named
    if variant == "friendly":
        legacy = cache_dir / f"sensitivity_smiley_{category}.npz"
        if legacy.exists():
            return legacy
    raise FileNotFoundError(f"Missing smiley cache for variant={variant!r} in {cache_dir}")


def _analyze_variant(
    cache_dir: Path,
    category: str,
    variant: str,
    plain_ladder: np.ndarray,
    plain_ids: np.ndarray,
    plain_levels: list[str],
    n_use: int,
) -> dict:
    smiley_path = _resolve_smiley_path(cache_dir, category, variant)
    sm = np.load(smiley_path, allow_pickle=True)
    smiley_levels = [str(x) for x in sm["ladder_levels"]]

    stereo_ids = sm["stereo_ids"][:n_use].astype(int)
    smiley = sm["ladder_stereo"][:n_use]
    plain = plain_ladder[:n_use]
    if not np.array_equal(stereo_ids, plain_ids[:n_use]):
        print(f"[warn] {variant}: stereo_ids differ; using smiley cache ids")

    rates_sm = _level_rates(smiley, stereo_ids)
    rates_pl = _level_rates(plain, stereo_ids)
    delta = [s - p for s, p in zip(rates_sm, rates_pl, strict=True)]

    levels_out: dict = {}
    print(f"\n--- {variant}  (plain vs {variant}) ---")
    print(f"{'level':<6} {'plain':>8} {variant:>12} {'Δ':>8}")
    for i, (pl, smr, d) in enumerate(zip(rates_pl, rates_sm, delta, strict=True)):
        lvl = f"t{i + 1}"
        levels_out[lvl] = {
            "plain_compliance": pl,
            "smiley_compliance": smr,
            "delta_smiley_minus_plain": d,
            "plain_template": plain_levels[i] if i < len(plain_levels) else None,
            "smiley_template": smiley_levels[i],
        }
        print(f"{lvl:<6} {pl:>7.1%} {smr:>11.1%} {d:>+7.1%}")

    summary = {
        "mean_delta_t1_t6": float(np.mean(delta)),
        "max_abs_delta": float(max(abs(x) for x in delta)),
        "t3_delta": delta[2] if len(delta) > 2 else None,
        "t6_delta": delta[5] if len(delta) > 5 else None,
    }
    print(f"mean Δ: {summary['mean_delta_t1_t6']:+.1%}")

    return {
        "smiley_source": smiley_path.name,
        "ladder_levels": smiley_levels,
        "levels": levels_out,
        "summary": summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--category", default="Gender_identity")
    ap.add_argument(
        "--variants",
        default="all",
        help=f"Comma-separated profiles (default: all). Choices: {', '.join(sorted(SMILEY_LADDER_VARIANTS))}",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    variants = _parse_variants(args.variants)
    for variant in variants:
        if variant not in SMILEY_LADDER_VARIANTS and variant != "friendly":
            known = ", ".join(sorted(SMILEY_LADDER_VARIANTS))
            raise SystemExit(f"Unknown variant {variant!r}; choose from: {known}")

    cache_dir = model_cache_dir(args.cache_dir, args.model)
    plain_ladder, plain_ids, plain_src, plain_levels = _load_plain(cache_dir, args.category)

    n_plain = plain_ladder.shape[0]
    print(f"\n{args.category}  plain={plain_src}  n={n_plain}")

    out: dict = {
        "experiment": "smiley_ladder_compliance",
        "category": args.category,
        "model": args.model,
        "plain_source": plain_src,
        "variants": {},
    }

    for variant in variants:
        try:
            smiley_path = _resolve_smiley_path(cache_dir, args.category, variant)
        except FileNotFoundError as exc:
            print(f"[skip] {exc}")
            continue
        sm = np.load(smiley_path, allow_pickle=True)
        n_use = min(sm["ladder_stereo"].shape[0], n_plain)
        out["variants"][variant] = _analyze_variant(
            cache_dir,
            args.category,
            variant,
            plain_ladder,
            plain_ids,
            plain_levels,
            n_use,
        )
        out["n"] = n_use

    if not out["variants"]:
        raise SystemExit("No smiley variant caches found — run collect_smiley_ladder.py on GPU first.")

    ranked = sorted(
        out["variants"].items(),
        key=lambda kv: kv[1]["summary"]["mean_delta_t1_t6"],
        reverse=True,
    )
    out["summary"] = {
        "ranked_by_mean_delta": [
            {"variant": name, **data["summary"]} for name, data in ranked
        ],
        "highest_mean_delta": ranked[0][0],
        "lowest_mean_delta": ranked[-1][0],
    }
    print("\n=== rank by mean Δ (smiley − plain) ===")
    for name, data in ranked:
        print(f"  {name:<14} {data['summary']['mean_delta_t1_t6']:+.1%}")

    out_path = Path(args.out or f"results/smiley_compliance_{args.category}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
