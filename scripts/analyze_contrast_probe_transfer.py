"""Cross-condition contrast probe transfer (CPU).

Train a probe on one cache/condition (default: baseline no-nudge), then evaluate
whether the same fixed probe predicts compliance on another condition (e.g. t1).

Example:
    # After collecting both caches on GPU:
    uv run python scripts/analyze_contrast_probe_transfer.py \\
        --category Gender_identity \\
        --test-with-nudge --test-level t1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
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


def _level_index(level_names: list[str], level: str) -> int:
    if level in level_names:
        return level_names.index(level)
    raise ValueError(f"Level {level!r} not in cache conditions: {level_names}")


def _fit_probe(X: np.ndarray, y: np.ndarray):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced"),
    )
    clf.fit(X, y)
    return clf


def _eval_probe(clf, X: np.ndarray, y: np.ndarray) -> dict:
    pred = clf.predict(X)
    prob = clf.predict_proba(X)[:, 1]
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "compliance_rate": float(y.mean()),
    }
    if len(np.unique(y)) >= 2:
        out["auc"] = float(roc_auc_score(y, prob))
    else:
        out["auc"] = None
    return out


def _load_split(
    cache_dir: Path,
    category: str,
    *,
    with_nudge: bool,
    reason_before_answer: bool,
) -> dict:
    path = resolve_probe_cache(
        cache_dir,
        category,
        "binary",
        reason_before_answer=reason_before_answer,
        with_nudge=with_nudge,
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return load_probe_arrays(np.load(path, allow_pickle=True))


def _delta_and_labels(loaded: dict, level: str) -> tuple[np.ndarray, np.ndarray]:
    t = _level_index(loaded["level_names"], level)
    delta = centered_delta(loaded["phi_plus"][:, t, :], loaded["phi_minus"][:, t, :])
    y = comply_labels(loaded["direct"][:, t, :], loaded["stereo_ids"])
    return delta, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--category", default="Gender_identity")
    ap.add_argument("--out", default=None)
    ap.add_argument("--train-with-nudge", action="store_true")
    ap.add_argument("--test-with-nudge", action="store_true")
    ap.add_argument("--train-level", default="base", help="Train condition tag (base or t1..t6)")
    ap.add_argument("--test-level", default="t1", help="Test condition tag (base or t1..t6)")
    ap.add_argument("--reason-before-answer", action="store_true")
    args = ap.parse_args()

    cache_dir = model_cache_dir(args.cache_dir, args.model)
    train = _load_split(
        cache_dir,
        args.category,
        with_nudge=args.train_with_nudge,
        reason_before_answer=args.reason_before_answer,
    )
    test = _load_split(
        cache_dir,
        args.category,
        with_nudge=args.test_with_nudge,
        reason_before_answer=args.reason_before_answer,
    )

    n = min(train["n"], test["n"])
    if train["n"] != test["n"]:
        print(f"[warn] aligning n={n} (train={train['n']} test={test['n']})")

    X_train, y_train = _delta_and_labels(train, args.train_level)
    X_test, y_test = _delta_and_labels(test, args.test_level)
    X_train, y_train = X_train[:n], y_train[:n]
    X_test, y_test = X_test[:n], y_test[:n]

    clf = _fit_probe(X_train, y_train)
    train_in_domain = _eval_probe(clf, X_train, y_train)
    test_transfer = _eval_probe(clf, X_test, y_test)

    # In-domain upper bound on test condition (retrain on test Δ with CV-style holdout note)
    clf_test = _fit_probe(X_test, y_test)
    test_in_domain = _eval_probe(clf_test, X_test, y_test)

    out = {
        "experiment": "binary_probe_transfer",
        "model": args.model,
        "category": args.category,
        "n": n,
        "train": {
            "with_nudge": args.train_with_nudge,
            "level": args.train_level,
            "in_domain": train_in_domain,
        },
        "test": {
            "with_nudge": args.test_with_nudge,
            "level": args.test_level,
            "transfer_from_train": test_transfer,
            "in_domain_refit": test_in_domain,
        },
    }

    stem = cache_stem(args.category, mode="binary", with_nudge=False)
    out_path = Path(
        args.out
        or f"results/{stem}_transfer_{args.train_level}_to_{args.test_level}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(f"Train {args.train_level} (nudge={args.train_with_nudge}): comply={train_in_domain['compliance_rate']:.1%}")
    print(f"  in-domain acc={train_in_domain['accuracy']:.3f}  auc={train_in_domain.get('auc')}")
    print(f"Test  {args.test_level} (nudge={args.test_with_nudge}): comply={test_transfer['compliance_rate']:.1%}")
    print(f"  transfer acc={test_transfer['accuracy']:.3f}  auc={test_transfer.get('auc')}")
    print(f"  in-domain refit acc={test_in_domain['accuracy']:.3f}  auc={test_in_domain.get('auc')}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
