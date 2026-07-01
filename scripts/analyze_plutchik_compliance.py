"""CPU — Plutchik emotion-wheel compliance vs assertiveness control.

Example:
    uv run python scripts/analyze_plutchik_compliance.py --category Gender_identity
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mech_interp_bbq.prompts import model_cache_dir
from mech_interp_bbq.sensitivity import (
    PLUTCHIK_ALL_VARIANTS,
    PLUTCHIK_EMOTION_NAMES,
    PLUTCHIK_SMILEY_VARIANTS,
)


def _comply(logits: np.ndarray, stereo_ids: np.ndarray) -> np.ndarray:
    return logits.argmax(1) == stereo_ids


def _level_rates(ladder: np.ndarray, stereo_ids: np.ndarray) -> list[float]:
    return [float(_comply(ladder[:, t, :], stereo_ids).mean()) for t in range(ladder.shape[1])]


def _parse_list(raw: str | None, all_keys: list[str]) -> list[str]:
    if raw is None or raw.strip().lower() in {"", "all"}:
        return sorted(all_keys)
    return [v.strip() for v in raw.split(",") if v.strip()]


def _parse_modes(raw: str) -> set[str]:
    modes = {m.strip().lower() for m in raw.split(",") if m.strip()}
    if "all" in modes:
        return {"intensity", "wheel", "smiley"}
    return modes or {"intensity", "wheel"}


def _load_assertiveness_plain(cache_dir: Path, category: str) -> tuple[np.ndarray, list[str], str]:
    matched = cache_dir / f"sensitivity_smiley_plain_{category}.npz"
    if matched.exists():
        d = np.load(matched, allow_pickle=True)
        return d["ladder_stereo"], [str(x) for x in d["ladder_levels"]], str(matched.name)

    legacy = cache_dir / f"sensitivity_{category}.npz"
    if legacy.exists():
        d = np.load(legacy, allow_pickle=True)
        levels = [str(x) for x in d["ladder_levels"]] if "ladder_levels" in d else []
        return d["ladder_stereo"], levels, str(legacy.name)

    raise FileNotFoundError(f"No assertiveness plain cache in {cache_dir}")


def _analyze_one(
    label: str,
    path: Path,
    assert_ladder: np.ndarray,
    assert_levels: list[str],
) -> dict:
    d = np.load(path, allow_pickle=True)
    n_use = min(d["ladder_stereo"].shape[0], assert_ladder.shape[0])
    stereo_ids = d["stereo_ids"][:n_use].astype(int)
    emo_ladder = d["ladder_stereo"][:n_use]
    assertive = assert_ladder[:n_use]
    emo_levels = [str(x) for x in d["ladder_levels"]]

    rates_em = _level_rates(emo_ladder, stereo_ids)
    rates_as = _level_rates(assertive, stereo_ids)
    delta = [e - a for e, a in zip(rates_em, rates_as, strict=True)]

    levels_out: dict = {}
    print(f"--- {label}  vs  assertiveness ---")
    print(f"{'level':<6} {'assert':>8} {'emotion':>8} {'Δ':>8}")
    for i, (asr, emr, dlt) in enumerate(zip(rates_as, rates_em, delta, strict=True)):
        lvl = f"t{i + 1}"
        levels_out[lvl] = {
            "assertiveness_compliance": asr,
            "plutchik_compliance": emr,
            "delta_plutchik_minus_assertiveness": dlt,
            "assertiveness_template": assert_levels[i] if i < len(assert_levels) else None,
            "plutchik_template": emo_levels[i],
        }
        print(f"{lvl:<6} {asr:>7.1%} {emr:>7.1%} {dlt:>+7.1%}")

    summary = {
        "mean_delta_t1_t6": float(np.mean(delta)),
        "max_abs_delta": float(max(abs(x) for x in delta)),
        "t6_delta": delta[5] if len(delta) > 5 else None,
    }
    print(f"mean Δ (Plutchik − assertiveness): {summary['mean_delta_t1_t6']:+.1%}\n")

    return {
        "source": path.name,
        "ladder_levels": emo_levels,
        "levels": levels_out,
        "summary": summary,
        "n": n_use,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--category", default="Gender_identity")
    ap.add_argument("--modes", default="all", help="intensity, wheel, smiley, or all")
    ap.add_argument(
        "--emotions",
        default="all",
        help=f"Emotion families (default: all). Choices: {', '.join(PLUTCHIK_EMOTION_NAMES)}, wheel",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    modes = _parse_modes(args.modes)
    emotions = _parse_list(args.emotions, list(PLUTCHIK_EMOTION_NAMES))

    cache_dir = model_cache_dir(args.cache_dir, args.model)
    assert_ladder, assert_levels, assert_src = _load_assertiveness_plain(cache_dir, args.category)

    out: dict = {
        "experiment": "plutchik_emotion_compliance",
        "category": args.category,
        "model": args.model,
        "assertiveness_control": assert_src,
        "variants": {},
    }

    print(f"\n{args.category}  assertiveness control={assert_src}\n")

    if "intensity" in modes:
        for emo in emotions:
            if emo not in PLUTCHIK_EMOTION_NAMES:
                continue
            path = cache_dir / f"sensitivity_plutchik_{emo}_{args.category}.npz"
            if not path.exists():
                print(f"[skip] missing {path.name}")
                continue
            out["variants"][emo] = _analyze_one(f"intensity:{emo}", path, assert_ladder, assert_levels)

    if "wheel" in modes and (args.emotions.strip().lower() in {"", "all"} or "wheel" in emotions):
        path = cache_dir / f"sensitivity_plutchik_wheel_{args.category}.npz"
        if path.exists():
            out["variants"]["wheel"] = _analyze_one("wheel", path, assert_ladder, assert_levels)
        else:
            print(f"[skip] missing {path.name}")

    if "smiley" in modes:
        for key in sorted(PLUTCHIK_SMILEY_VARIANTS):
            path = cache_dir / f"sensitivity_plutchik_smiley_{key}_{args.category}.npz"
            if not path.exists():
                print(f"[skip] missing {path.name}")
                continue
            out["variants"][f"smiley_{key}"] = _analyze_one(f"smiley:{key}", path, assert_ladder, assert_levels)

    if not out["variants"]:
        raise SystemExit("No Plutchik caches found — run collect_plutchik_ladder.py on GPU first.")

    out["n"] = next(iter(out["variants"].values()))["n"]
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

    out_path = Path(args.out or f"results/plutchik_compliance_{args.category}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
