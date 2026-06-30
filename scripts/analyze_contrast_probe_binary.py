"""CPU — binary contrast probe analysis (stereo vs not).

Trains/evaluates:
  * direct argmax compliance rate
  * logit-margin logistic regression (S−U)
  * supervised contrast probe on centered Δ(S-U)
  * unsupervised PCA on Δ(S-U)

Example:
    uv run python scripts/analyze_contrast_probe_binary.py --category Gender_identity
    uv run python scripts/analyze_contrast_probe_binary.py --category Gender_identity --with-nudge
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mech_interp_bbq.contrast_analysis import (
    centered_delta,
    comply_labels,
    load_probe_arrays,
    resolve_probe_cache,
)
from mech_interp_bbq.contrast_probe import cache_stem
from mech_interp_bbq.prompts import model_cache_dir


def _margin(logits: np.ndarray, stereo_ids: np.ndarray, unknown_ids: np.ndarray) -> np.ndarray:
    rows = np.arange(logits.shape[0])
    return (logits[rows, stereo_ids] - logits[rows, unknown_ids]).astype(np.float64)


def _cv_acc(clf, X: np.ndarray, y: np.ndarray, cv) -> float:
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return float(cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean())


def _cv_auc(clf, X: np.ndarray, y: np.ndarray, cv) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, prob))


def _unsupervised_pc_acc(delta: np.ndarray, y: np.ndarray, cv: StratifiedKFold) -> tuple[float, float | None]:
    scores = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in cv.split(delta, y):
        pca = PCA(n_components=1)
        pca.fit(delta[train_idx])
        pc = pca.components_[0]
        proj_tr = delta[train_idx] @ pc
        corr = np.corrcoef(proj_tr, y[train_idx])[0, 1]
        sign = 1.0 if (corr >= 0 if np.isfinite(corr) else True) else -1.0
        scores[test_idx] = delta[test_idx] @ pc * sign
    acc = float(((scores >= np.median(scores)).astype(int) == y).mean())
    auc = float(roc_auc_score(y, scores)) if len(np.unique(y)) >= 2 else None
    return acc, auc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--category", default="Gender_identity")
    ap.add_argument("--out", default=None)
    ap.add_argument("--reason-before-answer", action="store_true")
    ap.add_argument("--with-nudge", action="store_true")
    ap.add_argument("--compare-sensitivity", action="store_true")
    args = ap.parse_args()

    cache_dir = model_cache_dir(args.cache_dir, args.model)
    probe_path = resolve_probe_cache(
        cache_dir,
        args.category,
        "binary",
        reason_before_answer=args.reason_before_answer,
        with_nudge=args.with_nudge,
    )
    if not probe_path.exists():
        raise SystemExit(f"Missing {probe_path} — run collect_contrast_probe_binary.py on GPU first.")

    loaded = load_probe_arrays(np.load(probe_path, allow_pickle=True))
    if loaded["mode"] == "threeway":
        raise SystemExit(f"{probe_path} is a 3-way cache — use analyze_contrast_probe_threeway.py")

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    clf = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")
    )

    stem = cache_stem(
        args.category,
        mode="binary",
        reason_before_answer=args.reason_before_answer,
        with_nudge=args.with_nudge,
    )
    out: dict = {
        "experiment": "binary",
        "model": args.model,
        "category": args.category,
        "n": loaded["n"],
        "contrast_suffix": loaded["contrast_suffix"],
        "with_nudge": loaded["with_nudge"],
        "reason_before_answer": loaded["reason_before_answer"],
        "with_reasoning_instruction": loaded["with_reasoning_instruction"],
        "levels": {},
    }

    print(f"[binary] {args.category}  n={loaded['n']}\n")

    for t, lvl in enumerate(loaded["level_names"]):
        direct_t = loaded["direct"][:, t, :]
        y = comply_labels(direct_t, loaded["stereo_ids"])
        margin = _margin(direct_t, loaded["stereo_ids"], loaded["unknown_ids"])
        delta = centered_delta(loaded["phi_plus"][:, t, :], loaded["phi_minus"][:, t, :])

        direct_acc = float(y.mean())
        margin_acc = _cv_acc(clf, margin, y, cv)
        sup_acc = _cv_acc(clf, delta, y, cv)
        sup_auc = _cv_auc(clf, delta, y, cv)
        unsup_acc, unsup_auc = _unsupervised_pc_acc(delta, y, cv)

        out["levels"][lvl] = {
            "compliance_rate": direct_acc,
            "logit_margin_cv_acc": margin_acc,
            "contrast_supervised_cv_acc": sup_acc,
            "contrast_supervised_cv_auc": sup_auc,
            "contrast_unsupervised_pc_cv_acc": unsup_acc,
            "contrast_unsupervised_pc_cv_auc": unsup_auc,
            "mean_delta_norm": float(np.linalg.norm(delta, axis=1).mean()),
        }
        print(
            f"  {lvl}: comply={direct_acc:.1%}  margin={margin_acc:.3f}  "
            f"sup={sup_acc:.3f}  unsup={unsup_acc:.3f}"
        )

    if args.compare_sensitivity and loaded["with_nudge"]:
        sens_path = cache_dir / f"sensitivity_{args.category}.npz"
        if sens_path.exists():
            sens = np.load(sens_path)
            n_use = min(loaded["n"], sens["ladder_stereo"].shape[0])
            out["sensitivity_levels"] = {}
            for t, lvl in enumerate(loaded["level_names"]):
                if t >= 6:
                    break
                y_p = comply_labels(loaded["direct"][:n_use, t, :], loaded["stereo_ids"][:n_use])
                y_s = (sens["ladder_stereo"][:n_use, t, :].argmax(1) == sens["stereo_ids"][:n_use])
                agree = float((y_p == y_s).mean())
                out["sensitivity_levels"][lvl] = {"probe_vs_sensitivity_agreement": agree}
                print(f"  {lvl} probe↔sensitivity: {agree:.1%}")

    out_path = Path(args.out or f"results/{stem}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
