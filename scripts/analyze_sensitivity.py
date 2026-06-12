"""Nudge dose-response sensitivity analysis (CPU). Consumes sensitivity_<cat>.npz.

Reports, per direction (toward stereo / toward other):
  * population flip-rate vs nudge intensity (ladder + repetition)
  * per-example flip threshold (smallest level that flips argmax to target) & slope
  * H2: does baseline abstention margin predict sensitivity? (AUC of "flips by t")
  * H3: is Unknown->Other backfire dose-dependent? (backfire rate vs intensity)
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CDIR_GLOB = "cache/*"


def _softmax3(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _target_id(direction, sids, oids):
    return sids if direction == "stereo" else oids


def flip_threshold(argmax_levels, base_arg, target):
    """First level index (1-based) where argmax==target; 0 if already, np.inf if never."""
    n, T = argmax_levels.shape
    out = np.full(n, np.inf)
    out[base_arg == target] = 0
    for t in range(T):
        hit = (argmax_levels[:, t] == target) & ~np.isfinite(out)
        out[hit] = t + 1
    return out


def analyze_axis(name, levels_logits, base_logits, sids, uids, oids, direction, mask):
    """levels_logits: (n, T, 3); base_logits: (n,3)."""
    n, T, _ = levels_logits.shape
    tgt = _target_id(direction, sids, oids)
    base_arg = base_logits.argmax(1)
    arg = levels_logits.argmax(2)                                   # (n,T)
    p = _softmax3(levels_logits)
    rows = np.arange(n)
    p_tgt = p[rows[:, None], np.arange(T)[None, :], tgt[:, None]]   # (n,T) P(target)
    p_tgt_base = _softmax3(base_logits)[rows, tgt]

    m = mask
    print(f"\n  [{name} | toward {direction}]  (n={m.sum()})")
    print("   level:  " + "  ".join(f"t{t+1}" for t in range(T)))
    fr = [(arg[m, t] == tgt[m]).mean() for t in range(T)]
    print("   P(argmax=target): " + "  ".join(f"{v:.2f}" for v in fr))
    pt = [p_tgt[m, t].mean() for t in range(T)]
    print(f"   mean P(target):  base={p_tgt_base[m].mean():.2f}  " + "  ".join(f"{v:.2f}" for v in pt))

    thr = flip_threshold(arg[m], base_arg[m], tgt[m])
    ever = np.isfinite(thr)
    print(f"   flips ever: {ever.mean():.2f}   median threshold (flippers): "
          f"{np.median(thr[ever & (thr > 0)]) if (ever & (thr>0)).any() else float('nan')}")
    # slope = ΔP(target) from base to max level
    slope = p_tgt[m, -1] - p_tgt_base[m]
    print(f"   mean ΔP(target) base→max: {slope.mean():.2f}")
    return thr, base_arg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Gender_identity")
    ap.add_argument("--model-glob", default="Qwen_Qwen3-32B",
                    help="substring of the per-model cache dir to search")
    args = ap.parse_args()

    f = glob.glob(f"cache/*{args.model_glob}*/sensitivity_{args.category}.npz")
    if not f:
        f = glob.glob(f"cache/**/sensitivity_{args.category}.npz", recursive=True)
    D = np.load(f[0], allow_pickle=True)
    print(f"loaded {f[0]}")
    base = D["base_logits"]
    sids, uids, oids = D["stereo_ids"], D["unknown_ids"], D["other_ids"]
    has_other = D["has_other"]
    n = len(base)

    for direction in ("stereo", "other"):
        mask = has_other if direction == "other" else np.ones(n, bool)
        analyze_axis("LADDER", D[f"ladder_{direction}"], base, sids, uids, oids, direction, mask)
        analyze_axis("REP", D[f"rep_{direction}"], base, sids, uids, oids, direction, mask)

    # ── H2: baseline margin predicts sensitivity (flip-by-level) for nudge→stereo ──
    print("\n=== H2: does baseline geometry predict sensitivity? (ladder, toward stereo) ===")
    base_p = _softmax3(base)
    rows = np.arange(n)
    margin = base[rows, uids] - np.maximum(base[rows, sids], base[rows, oids])  # U lead
    feats = np.stack([margin, base_p[rows, uids]], 1)
    arg = D["ladder_stereo"].argmax(2)
    base_arg = base.argmax(1)
    thr = flip_threshold(arg, base_arg, sids)
    for t in (2, 4, 6):
        y = (thr <= t).astype(int)                       # flipped to stereo by level t
        if 0 < y.mean() < 1:
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            pred = cross_val_predict(clf, feats, y, cv=5, method="predict_proba")[:, 1]
            print(f"  flip-by-t{t}: rate={y.mean():.2f}  AUC(margin,P_U)={roc_auc_score(y, pred):.3f}")

    # ── H3: dose-dependence of Unknown→Other backfire (toward stereo) ──
    print("\n=== H3: is Unknown→Other backfire dose-dependent? (toward stereo) ===")
    unk = base.argmax(1) == uids
    ls = D["ladder_stereo"].argmax(2)                    # (n,T)
    T = ls.shape[1]
    print(f"  baseline-Unknown cases: {unk.sum()}")
    print("   level:           " + "  ".join(f"t{t+1}" for t in range(T)))
    bf = [((ls[unk, t] == oids[unk])).mean() for t in range(T)]        # ->Other (backfire)
    comp = [((ls[unk, t] == sids[unk])).mean() for t in range(T)]      # ->Stereo (comply)
    print("   P(→Other  backfire): " + "  ".join(f"{v:.2f}" for v in bf))
    print("   P(→Stereo comply):   " + "  ".join(f"{v:.2f}" for v in comp))


if __name__ == "__main__":
    main()
