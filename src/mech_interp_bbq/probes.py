"""Linear probes over per-layer activations.

We train one probe per layer using logistic regression, then report
cross-validated accuracy so you can see at which depth a concept becomes
linearly readable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class ProbeResult:
    layer: int
    mean_accuracy: float
    fold_accuracies: list[float]
    coef: np.ndarray  # shape (n_classes, d_model)
    intercept: np.ndarray  # shape (n_classes,)


def train_layer_probe(
    activations: torch.Tensor,
    labels: np.ndarray,
    layer: int,
    n_splits: int = 5,
    C: float = 1.0,
    max_iter: int = 2000,
    seed: int = 0,
) -> ProbeResult:
    """Train a logistic-regression probe for a single layer.

    Args:
        activations: Tensor of shape (n_examples, d_model).
        labels: Integer labels of shape (n_examples,).
        layer: Index used only for bookkeeping in the returned object.
    """
    X = activations.detach().cpu().float().numpy()
    y = np.asarray(labels)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_acc: list[float] = []
    for train_idx, val_idx in skf.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(C=C, max_iter=max_iter)
        clf.fit(scaler.transform(X[train_idx]), y[train_idx])
        fold_acc.append(float(clf.score(scaler.transform(X[val_idx]), y[val_idx])))

    full_scaler = StandardScaler().fit(X)
    full_clf = LogisticRegression(C=C, max_iter=max_iter)
    full_clf.fit(full_scaler.transform(X), y)

    return ProbeResult(
        layer=layer,
        mean_accuracy=float(np.mean(fold_acc)),
        fold_accuracies=fold_acc,
        coef=full_clf.coef_,
        intercept=full_clf.intercept_,
    )


def train_all_layers(
    acts: torch.Tensor,
    labels: np.ndarray,
    **kwargs,
) -> list[ProbeResult]:
    """Train a probe on every layer in an ``(n, n_layers, d_model)`` tensor."""
    results: list[ProbeResult] = []
    for layer in range(acts.shape[1]):
        results.append(train_layer_probe(acts[:, layer, :], labels, layer=layer, **kwargs))
    return results
