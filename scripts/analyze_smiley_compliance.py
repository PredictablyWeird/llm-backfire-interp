"""CPU — compare plain vs smiley ladder stereo compliance.

Uses:
  * sensitivity_smiley_<Category>.npz  (smiley ladder, required)
  * sensitivity_smiley_plain_<Category>.npz  (matched plain, if --also-plain was used)
  * OR sensitivity_<Category>.npz  (fallback plain baseline)

Example:
    uv run python scripts/analyze_smiley_compliance.py --category Gender_identity
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mech_interp_bbq.prompts import model_cache_dir


def _comply(logits: np.ndarray, stereo_ids: np.ndarray) -> np.ndarray:
    return logits.argmax(1) == stereo_ids


def _level_rates(ladder: np.ndarray, stereo_ids: np.ndarray) -> list[float]:
    return [float(_comply(ladder[:, t, :], stereo_ids).mean()) for t in range(ladder.shape[1])]


def _load_plain(cache_dir: Path, category: str) -> tuple[np.ndarray, np.ndarray, str]:
    matched = cache_dir / f"sensitivity_smiley_plain_{category}.npz"
    if matched.exists():
        d = np.load(matched, allow_pickle=True)
        return d["ladder_stereo"], d["stereo_ids"].astype(int), str(matched.name)

    legacy = cache_dir / f"sensitivity_{category}.npz"
    if legacy.exists():
        d = np.load(legacy)
        return d["ladder_stereo"], d["stereo_ids"].astype(int), str(legacy.name)

    raise FileNotFoundError(f"No plain ladder cache in {cache_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--category", default="Gender_identity")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cache_dir = model_cache_dir(args.cache_dir, args.model)
    smiley_path = cache_dir / f"sensitivity_smiley_{args.category}.npz"
    if not smiley_path.exists():
        raise SystemExit(f"Missing {smiley_path} — run collect_smiley_ladder.py on GPU first.")

    sm = np.load(smiley_path, allow_pickle=True)
    plain_ladder, plain_ids, plain_src = _load_plain(cache_dir, args.category)
    plain_meta = np.load(cache_dir / plain_src, allow_pickle=True)
    plain_levels = [str(x) for x in plain_meta["ladder_levels"]]
    smiley_levels = [str(x) for x in sm["ladder_levels"]]

    n = sm["ladder_stereo"].shape[0]
    n_plain = plain_ladder.shape[0]
    n_use = min(n, n_plain)
    if n != n_plain:
        print(f"[warn] aligning n={n_use} (smiley={n} plain={n_plain})")

    stereo_ids = sm["stereo_ids"][:n_use].astype(int)
    smiley = sm["ladder_stereo"][:n_use]
    plain = plain_ladder[:n_use]
    if not np.array_equal(stereo_ids, plain_ids[:n_use]):
        print("[warn] stereo_ids differ between caches; using smiley cache ids")

    rates_sm = _level_rates(smiley, stereo_ids)
    rates_pl = _level_rates(plain, stereo_ids)
    delta = [s - p for s, p in zip(rates_sm, rates_pl, strict=True)]

    out: dict = {
        "experiment": "smiley_ladder_compliance",
        "category": args.category,
        "model": args.model,
        "n": n_use,
        "plain_source": plain_src,
        "smiley_source": smiley_path.name,
        "levels": {},
    }

    print(f"\n{args.category}  n={n_use}  plain={plain_src}  vs  smiley\n")
    print(f"{'level':<6} {'plain':>8} {'smiley':>8} {'Δ':>8}")
    for i, (pl, smr, d) in enumerate(zip(rates_pl, rates_sm, delta, strict=True)):
        lvl = f"t{i + 1}"
        out["levels"][lvl] = {
            "plain_compliance": pl,
            "smiley_compliance": smr,
            "delta_smiley_minus_plain": d,
            "plain_template": plain_levels[i] if i < len(plain_levels) else None,
            "smiley_template": smiley_levels[i],
        }
        print(f"{lvl:<6} {pl:>7.1%} {smr:>7.1%} {d:>+7.1%}")

    out["summary"] = {
        "mean_delta_t1_t6": float(np.mean(delta)),
        "max_abs_delta": float(max(abs(x) for x in delta)),
        "t3_delta": delta[2] if len(delta) > 2 else None,
    }
    print(f"\nmean Δ (smiley−plain) across ladder: {out['summary']['mean_delta_t1_t6']:+.1%}")

    out_path = Path(args.out or f"results/smiley_compliance_{args.category}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
