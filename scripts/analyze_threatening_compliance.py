"""CPU — threatening-ladder stereo compliance vs assertiveness control.

Compares each threatening profile to the matched assertiveness plain ladder at
t1–t6. Supports threatening-native variants and threatening+smiley combos.

Example:
    uv run python scripts/analyze_threatening_compliance.py \\
        --category Gender_identity --with-smileys
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mech_interp_bbq.prompts import model_cache_dir
from mech_interp_bbq.sensitivity import THREATENING_LADDER_VARIANTS


def _comply(logits: np.ndarray, stereo_ids: np.ndarray) -> np.ndarray:
    return logits.argmax(1) == stereo_ids


def _level_rates(ladder: np.ndarray, stereo_ids: np.ndarray) -> list[float]:
    return [float(_comply(ladder[:, t, :], stereo_ids).mean()) for t in range(ladder.shape[1])]


def _parse_list(raw: str | None, all_keys: list[str]) -> list[str]:
    if raw is None or raw.strip().lower() in {"", "all"}:
        return sorted(all_keys)
    return [v.strip() for v in raw.split(",") if v.strip()]


def _load_assertiveness_plain(cache_dir: Path, category: str) -> tuple[np.ndarray, np.ndarray, str, list[str]]:
    matched = cache_dir / f"sensitivity_smiley_plain_{category}.npz"
    if matched.exists():
        d = np.load(matched, allow_pickle=True)
        return d["ladder_stereo"], d["stereo_ids"].astype(int), str(matched.name), [str(x) for x in d["ladder_levels"]]

    legacy = cache_dir / f"sensitivity_{category}.npz"
    if legacy.exists():
        d = np.load(legacy, allow_pickle=True)
        levels = [str(x) for x in d["ladder_levels"]] if "ladder_levels" in d else []
        return d["ladder_stereo"], d["stereo_ids"].astype(int), str(legacy.name), levels

    raise FileNotFoundError(f"No assertiveness plain cache in {cache_dir}")


def _analyze_one(
    label: str,
    path: Path,
    assert_ladder: np.ndarray,
    assert_levels: list[str],
    n_assert: int,
) -> dict:
    d = np.load(path, allow_pickle=True)
    n_use = min(d["ladder_stereo"].shape[0], n_assert)
    stereo_ids = d["stereo_ids"][:n_use].astype(int)
    threatening = d["ladder_stereo"][:n_use]
    assertive = assert_ladder[:n_use]
    threat_levels = [str(x) for x in d["ladder_levels"]]

    rates_th = _level_rates(threatening, stereo_ids)
    rates_as = _level_rates(assertive, stereo_ids)
    delta = [t - a for t, a in zip(rates_th, rates_as, strict=True)]

    levels_out: dict = {}
    print(f"--- {label}  vs  assertiveness ---")
    print(f"{'level':<6} {'assert':>8} {'threat':>8} {'Δ':>8}")
    for i, (asr, thr, dlt) in enumerate(zip(rates_as, rates_th, delta, strict=True)):
        lvl = f"t{i + 1}"
        levels_out[lvl] = {
            "assertiveness_compliance": asr,
            "threatening_compliance": thr,
            "delta_threatening_minus_assertiveness": dlt,
            "assertiveness_template": assert_levels[i] if i < len(assert_levels) else None,
            "threatening_template": threat_levels[i],
        }
        print(f"{lvl:<6} {asr:>7.1%} {thr:>7.1%} {dlt:>+7.1%}")

    summary = {
        "mean_delta_t1_t6": float(np.mean(delta)),
        "max_abs_delta": float(max(abs(x) for x in delta)),
        "t6_delta": delta[5] if len(delta) > 5 else None,
    }
    print(f"mean Δ (threatening − assertiveness): {summary['mean_delta_t1_t6']:+.1%}\n")

    return {
        "source": path.name,
        "ladder_levels": threat_levels,
        "levels": levels_out,
        "summary": summary,
        "n": n_use,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--category", default="Gender_identity")
    ap.add_argument(
        "--variants",
        default="all",
        help=f"Threatening-native profiles (default: all). Choices: {', '.join(sorted(THREATENING_LADDER_VARIANTS))}",
    )
    ap.add_argument(
        "--with-smileys",
        action="store_true",
        help="Also analyze threatening+smiley combo caches",
    )
    ap.add_argument(
        "--smiley-variants",
        default="all",
        help=f"Smiley profiles when --with-smileys (default: all). Choices: friendly,subtle,warm,enthusiastic,pressured,intense",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    variants = _parse_list(args.variants, list(THREATENING_LADDER_VARIANTS))
    smiley_profiles: list[str] = []
    if args.with_smileys:
        from mech_interp_bbq.sensitivity import SMILEY_SUFFIX_PROFILES

        smiley_profiles = _parse_list(args.smiley_variants, list(SMILEY_SUFFIX_PROFILES))

    cache_dir = model_cache_dir(args.cache_dir, args.model)
    assert_ladder, _, assert_src, assert_levels = _load_assertiveness_plain(cache_dir, args.category)
    n_assert = assert_ladder.shape[0]

    out: dict = {
        "experiment": "threatening_ladder_compliance",
        "category": args.category,
        "model": args.model,
        "assertiveness_control": assert_src,
        "variants": {},
    }

    print(f"\n{args.category}  assertiveness control={assert_src}  n={n_assert}\n")

    for variant in variants:
        path = cache_dir / f"sensitivity_threatening_{variant}_{args.category}.npz"
        if not path.exists():
            print(f"[skip] missing {path.name}")
            continue
        result = _analyze_one(f"threatening:{variant}", path, assert_ladder, assert_levels, n_assert)
        out["variants"][variant] = result
        out["n"] = result["n"]

    for profile in smiley_profiles:
        path = cache_dir / f"sensitivity_threatening_smiley_{profile}_{args.category}.npz"
        if not path.exists():
            print(f"[skip] missing {path.name}")
            continue
        key = f"smiley_{profile}"
        result = _analyze_one(f"threatening+smiley:{profile}", path, assert_ladder, assert_levels, n_assert)
        out["variants"][key] = result
        out["n"] = result["n"]

    if not out["variants"]:
        raise SystemExit("No threatening caches found — run collect_threatening_ladder.py on GPU first.")

    ranked = sorted(
        out["variants"].items(),
        key=lambda kv: kv[1]["summary"]["mean_delta_t1_t6"],
        reverse=True,
    )
    out["summary"] = {
        "ranked_by_mean_delta": [{"variant": n, **d["summary"]} for n, d in ranked],
        "highest_mean_delta": ranked[0][0],
        "lowest_mean_delta": ranked[-1][0],
    }

    out_path = Path(args.out or f"results/threatening_compliance_{args.category}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
