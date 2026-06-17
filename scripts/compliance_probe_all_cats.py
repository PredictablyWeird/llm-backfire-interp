"""Baseline-method vs activation-probe compliance prediction across BBQ categories.

Uses sensitivity_<cat>.npz for labels + logit baseline.
Uses base_acts from ladder_acts_<cat>.npz or 3choice_*_<cat>_acts.npz when present.

Example:
    uv run python scripts/compliance_probe_all_cats.py --layers 56 48
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mech_interp_bbq.prompts import model_cache_dir

ALL_CATS = [
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Religion",
    "SES",
    "Sexual_orientation",
]


def _find_acts(cache_dir: Path, category: str) -> np.ndarray | None:
    ladder = cache_dir / f"ladder_acts_{category}.npz"
    if ladder.exists():
        return np.load(ladder, mmap_mode="r")["base_acts"]
    hits = glob.glob(str(cache_dir / f"3choice_*_{category}_*_acts.npz"))
    if not hits:
        return None
    # Prefer largest n (full run over smoke)
    best = max(hits, key=lambda p: np.load(p, mmap_mode="r")["base_acts"].shape[0])
    return np.load(best, mmap_mode="r")["base_acts"]


def _comp(ladder: np.ndarray, tid: np.ndarray, t: int) -> np.ndarray:
    return ladder[:, t, :].argmax(1) == tid


def _acc(clf, X, y, cv) -> float:
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return float(cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out", default="results/compliance_probe_all_cats.json")
    ap.add_argument("--stereo-layer", type=int, default=56)
    ap.add_argument("--other-layer", type=int, default=48)
    ap.add_argument(
        "--categories",
        nargs="*",
        default=ALL_CATS,
        help="BBQ categories (default: all with sensitivity cache)",
    )
    args = ap.parse_args()

    cache_dir = model_cache_dir(args.cache_dir, args.model)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    clf_margin = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")
    )
    clf_act = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")
    )

    out: dict = {
        "model": args.model,
        "stereo_layer": args.stereo_layer,
        "other_layer": args.other_layer,
        "categories": {},
    }

    for cat in args.categories:
        sens_path = cache_dir / f"sensitivity_{cat}.npz"
        if not sens_path.exists():
            print(f"[skip] no sensitivity cache for {cat}")
            continue

        sens = np.load(sens_path)
        n = sens["base_logits"].shape[0]
        rows = np.arange(n)
        uids = sens["unknown_ids"]
        base_logits = sens["base_logits"]
        acts = _find_acts(cache_dir, cat)
        has_acts = acts is not None
        if has_acts and acts.shape[0] != n:
            print(f"[warn] {cat}: acts n={acts.shape[0]} != sens n={n}; probe skipped")
            has_acts = False

        cat_out: dict = {"n": n, "has_base_acts": has_acts, "levels": {}}
        print(f"\n{cat}  n={n}  acts={'yes' if has_acts else 'no'}")

        for t in range(6):
            lvl = f"t{t + 1}"
            cat_out["levels"][lvl] = {}
            for direction, ladder_key, tid, layer in [
                ("stereo", "ladder_stereo", sens["stereo_ids"], args.stereo_layer),
                ("other", "ladder_other", sens["other_ids"], args.other_layer),
            ]:
                y = _comp(sens[ladder_key], tid, t)
                margin = base_logits[rows, tid] - base_logits[rows, uids]
                base_acc = _acc(clf_margin, margin.astype(np.float64), y, cv)
                probe_acc = None
                if has_acts:
                    probe_acc = _acc(clf_act, acts[:, layer, :], y, cv)
                cat_out["levels"][lvl][direction] = {
                    "compliance_rate": float(y.mean()),
                    "baseline_method_acc": base_acc,
                    "probe_acc": probe_acc,
                }
                probe_s = f"{probe_acc:.3f}" if probe_acc is not None else "  n/a"
                print(
                    f"  {lvl} {direction:>5}: rate={y.mean():.1%}  "
                    f"baseline={base_acc:.3f}  probe={probe_s}"
                )

        out["categories"][cat] = cat_out

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
