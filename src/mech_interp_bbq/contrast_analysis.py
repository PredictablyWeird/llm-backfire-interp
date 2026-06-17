"""Shared CPU analysis helpers for contrast-probe experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mech_interp_bbq.contrast_probe import ProbeMode, cache_stem, legacy_binary_cache_stem


def resolve_probe_cache(
    cache_dir: Path,
    category: str,
    mode: ProbeMode,
    *,
    reason_before_answer: bool = False,
    with_nudge: bool = False,
) -> Path:
    path = cache_dir / f"{cache_stem(category, mode=mode, reason_before_answer=reason_before_answer, with_nudge=with_nudge)}.npz"
    if path.exists():
        return path
    if mode == "binary":
        legacy = cache_dir / f"{legacy_binary_cache_stem(category, reason_before_answer=reason_before_answer, with_nudge=with_nudge)}.npz"
        if legacy.exists():
            return legacy
    return path


def load_probe_arrays(data: np.lib.npyio.NpzFile) -> dict:
    direct = data["direct_logits"]
    n, n_levels = direct.shape[:2]

    phi_plus = data["phi_plus"]
    phi_minus = data["phi_minus"]

    if phi_plus.ndim == 5:
        # (n, n_levels, n_pairs, n_layers, d)
        phi_plus = phi_plus[:, :, :, -1, :]
        phi_minus = phi_minus[:, :, :, -1, :]
    elif phi_plus.ndim == 4:
        if phi_plus.shape[2] == 3:
            # (n, n_levels, n_pairs, d) — 3-way, keep
            pass
        else:
            # (n, n_levels, n_layers, d) — binary, last layer only
            phi_plus = phi_plus[:, :, -1, :]
            phi_minus = phi_minus[:, :, -1, :]
    elif phi_plus.ndim != 3:
        raise ValueError(f"Unexpected phi_plus shape {phi_plus.shape}")

    if "probe_mode" in data:
        mode = str(data["probe_mode"])
    elif phi_plus.ndim == 4 and phi_plus.shape[2] == 3:
        mode = "threeway"
    else:
        mode = "binary"

    if "with_nudge" in data:
        with_nudge = bool(data["with_nudge"])
    else:
        with_nudge = len(data.get("ladder_levels", [])) > 0

    if "condition_tags" in data:
        level_names = [str(x) for x in data["condition_tags"]]
    elif with_nudge:
        level_names = [f"t{i + 1}" for i in range(n_levels)]
    else:
        level_names = ["base"] if n_levels == 1 else [f"base_{i}" for i in range(n_levels)]

    return {
        "mode": mode,
        "phi_plus": phi_plus,
        "phi_minus": phi_minus,
        "direct": direct,
        "stereo_ids": data["stereo_ids"].astype(int),
        "unknown_ids": data["unknown_ids"].astype(int),
        "other_ids": data["other_ids"].astype(int),
        "n": n,
        "n_levels": n_levels,
        "level_names": level_names,
        "with_nudge": with_nudge,
        "reason_before_answer": bool(data["reason_before_answer"]) if "reason_before_answer" in data else False,
        "with_reasoning_instruction": (
            bool(data["with_reasoning_instruction"]) if "with_reasoning_instruction" in data else False
        ),
        "contrast_suffix": str(data["contrast_suffix"]),
        "pair_names": [str(x) for x in data["pair_names"]] if "pair_names" in data else None,
    }


def centered_delta(phi_plus: np.ndarray, phi_minus: np.ndarray) -> np.ndarray:
    mu_p = phi_plus.mean(axis=0)
    mu_m = phi_minus.mean(axis=0)
    return (phi_plus - mu_p) - (phi_minus - mu_m)


def choice_labels(direct_logits: np.ndarray) -> np.ndarray:
    """3-way label: argmax index in {0, 1, 2}."""
    return direct_logits.argmax(1).astype(np.int64)


def comply_labels(direct_logits: np.ndarray, stereo_ids: np.ndarray) -> np.ndarray:
    return (direct_logits.argmax(1) == stereo_ids).astype(np.int64)
