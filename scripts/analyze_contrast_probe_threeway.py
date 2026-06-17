"""CPU — 3-way contrast probe analysis with linear or MLP heads.

Features: concat centered Δ for pairs S-U, S-O, U-O → R^{3d}.
Labels: direct-path argmax in {0, 1, 2} (A/B/C choice index).

Probes (5-fold CV):
  * **direct_logits** — multinomial LR on raw A/B/C logits
  * **linear** — multinomial LR on concat Δ (single linear layer → 3 logits)
  * **mlp** — one hidden layer → 3 logits (MLPClassifier)

Example:
    uv run python scripts/analyze_contrast_probe_threeway.py --category Gender_identity
    uv run python scripts/analyze_contrast_probe_threeway.py --category Gender_identity --probe mlp --hidden 64
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mech_interp_bbq.contrast_analysis import (
    centered_delta,
    choice_labels,
    load_probe_arrays,
    resolve_probe_cache,
)
from mech_interp_bbq.contrast_probe import cache_stem
from mech_interp_bbq.prompts import model_cache_dir


def _threeway_features(phi_plus: np.ndarray, phi_minus: np.ndarray) -> np.ndarray:
    """phi_* shape (n, n_pairs, d) → features (n, n_pairs * d)."""
    parts = []
    for p in range(phi_plus.shape[1]):
        parts.append(centered_delta(phi_plus[:, p, :], phi_minus[:, p, :]))
    return np.concatenate(parts, axis=1)


def _cv_multiclass_acc(clf, X: np.ndarray, y: np.ndarray, cv: StratifiedKFold) -> tuple[float, float]:
    preds = np.zeros_like(y)
    for train_idx, test_idx in cv.split(X, y):
        clf.fit(X[train_idx], y[train_idx])
        preds[test_idx] = clf.predict(X[test_idx])
    acc = float(accuracy_score(y, preds))
    macro_f1 = float(f1_score(y, preds, average="macro"))
    return acc, macro_f1


def _class_rates(
    y: np.ndarray,
    stereo_ids: np.ndarray,
    unknown_ids: np.ndarray,
    other_ids: np.ndarray,
) -> dict[str, float]:
    n = len(y)
    stereo = float(np.mean(y == stereo_ids))
    unknown = float(np.mean(y == unknown_ids))
    other = float(np.mean(y == other_ids))
    return {"stereo": stereo, "unknown": unknown, "other": other}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--category", default="Gender_identity")
    ap.add_argument("--out", default=None)
    ap.add_argument("--reason-before-answer", action="store_true")
    ap.add_argument("--with-nudge", action="store_true")
    ap.add_argument(
        "--probe",
        choices=("linear", "mlp", "both"),
        default="both",
        help="Probe head: linear (multinomial LR) and/or mlp (one hidden layer → 3 logits)",
    )
    ap.add_argument("--hidden", type=int, default=64, help="MLP hidden size")
    ap.add_argument("--mlp-alpha", type=float, default=0.01, help="L2 penalty for MLP")
    args = ap.parse_args()

    cache_dir = model_cache_dir(args.cache_dir, args.model)
    probe_path = resolve_probe_cache(
        cache_dir,
        args.category,
        "threeway",
        reason_before_answer=args.reason_before_answer,
        with_nudge=args.with_nudge,
    )
    if not probe_path.exists():
        raise SystemExit(f"Missing {probe_path} — run collect_contrast_probe_threeway.py on GPU first.")

    loaded = load_probe_arrays(np.load(probe_path))
    if loaded["mode"] != "threeway":
        raise SystemExit(f"{probe_path} is not a 3-way cache (mode={loaded['mode']})")

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    stem = cache_stem(
        args.category,
        mode="threeway",
        reason_before_answer=args.reason_before_answer,
        with_nudge=args.with_nudge,
    )

    out: dict = {
        "experiment": "threeway",
        "model": args.model,
        "category": args.category,
        "n": loaded["n"],
        "pair_names": loaded["pair_names"] or ["su", "so", "uo"],
        "probe": args.probe,
        "mlp_hidden": args.hidden,
        "levels": {},
    }

    print(f"[threeway] {args.category}  n={loaded['n']}  probe={args.probe}\n")

    for t, lvl in enumerate(loaded["level_names"]):
        direct_t = loaded["direct"][:, t, :]
        y = choice_labels(direct_t)
        rates = _class_rates(y, loaded["stereo_ids"], loaded["unknown_ids"], loaded["other_ids"])
        phi_p = loaded["phi_plus"][:, t, :, :]
        phi_m = loaded["phi_minus"][:, t, :, :]
        X_delta = _threeway_features(phi_p, phi_m)

        rates = _class_rates(y, loaded["stereo_ids"], loaded["unknown_ids"], loaded["other_ids"])

        row: dict = {
            "class_rates": rates,
            "feature_dim": int(X_delta.shape[1]),
        }

        logit_clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=0.1, multi_class="multinomial"),
        )
        logit_acc, logit_f1 = _cv_multiclass_acc(logit_clf, direct_t.astype(np.float64), y, cv)
        row["direct_logits_cv_acc"] = logit_acc
        row["direct_logits_cv_macro_f1"] = logit_f1

        parts = []
        if args.probe in ("linear", "both"):
            linear_clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, C=0.1, multi_class="multinomial"),
            )
            lin_acc, lin_f1 = _cv_multiclass_acc(linear_clf, X_delta, y, cv)
            row["contrast_linear_cv_acc"] = lin_acc
            row["contrast_linear_cv_macro_f1"] = lin_f1
            parts.append(f"linear={lin_acc:.3f}")

        if args.probe in ("mlp", "both"):
            mlp_clf = make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(args.hidden,),
                    activation="relu",
                    max_iter=2000,
                    alpha=args.mlp_alpha,
                    random_state=0,
                    early_stopping=True,
                    validation_fraction=0.1,
                ),
            )
            mlp_acc, mlp_f1 = _cv_multiclass_acc(mlp_clf, X_delta, y, cv)
            row["contrast_mlp_cv_acc"] = mlp_acc
            row["contrast_mlp_cv_macro_f1"] = mlp_f1
            parts.append(f"mlp={mlp_acc:.3f}")

        out["levels"][lvl] = row
        print(
            f"  {lvl}: logits={logit_acc:.3f}  "
            f"{'  '.join(parts)}  "
            f"[S/U/O={rates['stereo']:.0%}/{rates['unknown']:.0%}/{rates['other']:.0%}]"
        )

    out_path = Path(args.out or f"results/{stem}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
