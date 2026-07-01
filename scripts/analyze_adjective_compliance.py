"""CPU — adjective-ladder stereo compliance vs assertiveness control.

Compares pure adjective scales, assertiveness+adjective combos, and optional
epistemic smiley variants to the matched assertiveness plain ladder at t1–t6.

Example:
    uv run python scripts/analyze_adjective_compliance.py --category Gender_identity
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mech_interp_bbq.prompts import model_cache_dir
from mech_interp_bbq.sensitivity import ADJECTIVE_ALL_VARIANTS, ADJECTIVE_SMILEY_VARIANTS


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
        return {"pure", "combo", "smiley"}
    return modes or {"pure", "combo"}


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
    adj_ladder = d["ladder_stereo"][:n_use]
    assertive = assert_ladder[:n_use]
    adj_levels = [str(x) for x in d["ladder_levels"]]

    rates_adj = _level_rates(adj_ladder, stereo_ids)
    rates_as = _level_rates(assertive, stereo_ids)
    delta = [a - b for a, b in zip(rates_adj, rates_as, strict=True)]

    levels_out: dict = {}
    print(f"--- {label}  vs  assertiveness ---")
    print(f"{'level':<6} {'assert':>8} {'adj':>8} {'Δ':>8}")
    for i, (asr, adr, dlt) in enumerate(zip(rates_as, rates_adj, delta, strict=True)):
        lvl = f"t{i + 1}"
        levels_out[lvl] = {
            "assertiveness_compliance": asr,
            "adjective_compliance": adr,
            "delta_adjective_minus_assertiveness": dlt,
            "assertiveness_template": assert_levels[i] if i < len(assert_levels) else None,
            "adjective_template": adj_levels[i],
        }
        print(f"{lvl:<6} {asr:>7.1%} {adr:>7.1%} {dlt:>+7.1%}")

    summary = {
        "mean_delta_t1_t6": float(np.mean(delta)),
        "max_abs_delta": float(max(abs(x) for x in delta)),
        "t6_delta": delta[5] if len(delta) > 5 else None,
    }
    print(f"mean Δ (adjective − assertiveness): {summary['mean_delta_t1_t6']:+.1%}\n")

    return {
        "source": path.name,
        "ladder_levels": adj_levels,
        "levels": levels_out,
        "summary": summary,
        "n": n_use,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--category", default="Gender_identity")
    ap.add_argument("--modes", default="all", help="pure, combo, smiley, or all")
    ap.add_argument(
        "--scales",
        default="all",
        help=f"Adjective scales for pure/combo (default: all). Choices: {', '.join(sorted(ADJECTIVE_ALL_VARIANTS))}",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    modes = _parse_modes(args.modes)
    scales = _parse_list(args.scales, [k for k in ADJECTIVE_ALL_VARIANTS if not k.startswith("combo_")])

    cache_dir = model_cache_dir(args.cache_dir, args.model)
    assert_ladder, assert_levels, assert_src = _load_assertiveness_plain(cache_dir, args.category)

    out: dict = {
        "experiment": "adjective_ladder_compliance",
        "category": args.category,
        "model": args.model,
        "assertiveness_control": assert_src,
        "variants": {},
    }

    print(f"\n{args.category}  assertiveness control={assert_src}\n")

    if "pure" in modes:
        for scale in scales:
            if scale not in ADJECTIVE_ALL_VARIANTS:
                continue
            path = cache_dir / f"sensitivity_adjective_{scale}_{args.category}.npz"
            if not path.exists():
                print(f"[skip] missing {path.name}")
                continue
            out["variants"][scale] = _analyze_one(f"pure:{scale}", path, assert_ladder, assert_levels)

    if "combo" in modes:
        for scale in scales:
            key = f"combo_{scale}"
            if key not in ADJECTIVE_ALL_VARIANTS:
                continue
            path = cache_dir / f"sensitivity_adjective_{key}_{args.category}.npz"
            if not path.exists():
                print(f"[skip] missing {path.name}")
                continue
            out["variants"][key] = _analyze_one(f"combo:{scale}", path, assert_ladder, assert_levels)

    if "smiley" in modes:
        for key in sorted(ADJECTIVE_SMILEY_VARIANTS):
            path = cache_dir / f"sensitivity_adjective_smiley_{key}_{args.category}.npz"
            if not path.exists():
                print(f"[skip] missing {path.name}")
                continue
            out["variants"][f"smiley_{key}"] = _analyze_one(f"smiley:{key}", path, assert_ladder, assert_levels)

    if not out["variants"]:
        raise SystemExit("No adjective caches found — run collect_adjective_ladder.py on GPU first.")

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

    out_path = Path(args.out or f"results/adjective_compliance_{args.category}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
