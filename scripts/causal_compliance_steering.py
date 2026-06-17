"""Causal test: steer baseline compliance-probe direction into t3 forward pass.

Uses cached activations + unembed (CPU-only). Approximates intervening at layer L
during the t3 (stereo/other) forward pass by patching the final residual:

  patched_final = final_t3 - h_t3[L] + h_base[L] + alpha * v     [layer_replace]
  patched_final = final_t3 + alpha * v                           [final_add]

where v is the logistic-regression probe direction trained on *baseline* activations
at L to predict compliance at the chosen ladder level.

If v captures a causally relevant compliance direction, increasing alpha should
monotonically change compliance rate (under linear patch approximation).

Example:
    uv run python scripts/causal_compliance_steering.py --category Gender_identity
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mech_interp_bbq.hf_backend import project_resid_to_abc
from mech_interp_bbq.prompts import model_cache_dir

T3_IDX = 2


def _load(cache_dir: Path, category: str) -> dict:
    acts_f = cache_dir / f"ladder_acts_{category}.npz"
    sens_f = cache_dir / f"sensitivity_{category}.npz"
    unembed_f = cache_dir / "unembed.npz"
    for f in (acts_f, sens_f, unembed_f):
        if not f.exists():
            raise SystemExit(f"Missing {f}")

    acts = np.load(acts_f, mmap_mode="r")
    sens = np.load(sens_f)
    um = np.load(unembed_f)
    return {
        "base": acts["base_acts"],
        "ladder": acts["ladder_stereo_acts"],
        "ladder_other": sens["ladder_other"],  # logits only
        "base_logits": sens["base_logits"],
        "ladder_logits": sens["ladder_stereo"],
        "ladder_other_logits": sens["ladder_other"],
        "stereo_ids": sens["stereo_ids"].astype(int),
        "other_ids": sens["other_ids"].astype(int),
        "unknown_ids": sens["unknown_ids"].astype(int),
        "unembed": {
            "abc_unembed": um["abc_unembed"],
            "norm_weight": um["norm_weight"],
            "norm_eps": float(um["norm_eps"]),
        },
    }


def _proj(resid: np.ndarray, um: dict) -> np.ndarray:
    return project_resid_to_abc(resid, um["norm_weight"], um["norm_eps"], um["abc_unembed"])


def _comply_from_logits(logits: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    return logits.argmax(1) == target_ids


def _probe_direction(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit vector in activation space pointing toward class 1, plus train-set RMS scale."""
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced"),
    )
    pipe.fit(X[train_idx], y[train_idx])
    scaler = pipe.named_steps["standardscaler"]
    coef = pipe.named_steps["logisticregression"].coef_.ravel()
    v = coef / scaler.scale_
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v, 1.0
    Xf = X[train_idx].astype(np.float32)
    # Mean L2 norm at this layer (activations are ~O(1e3) for Qwen3-32B)
    rms = float(np.mean(np.linalg.norm(Xf, axis=1)))
    return v, rms


def _sweep(
    final: np.ndarray,
    h_t3: np.ndarray,
    h_base: np.ndarray,
    v: np.ndarray,
    act_rms: float,
    target_ids: np.ndarray,
    um: dict,
    alphas: np.ndarray,
    mode: str,
) -> list[dict]:
    rows = []
    for alpha in alphas:
        delta = alpha * act_rms * v  # alpha = fraction of layer RMS norm
        if mode == "final_add":
            patched = final + delta
        elif mode == "layer_replace":
            patched = final - h_t3 + h_base + delta
        elif mode == "layer_steer":
            patched = final - h_t3 + h_t3 + delta  # add steer at layer L
        else:
            raise ValueError(mode)
        pred = _proj(patched, um).argmax(1)
        comply = pred == target_ids
        rows.append({
            "alpha": float(alpha),
            "compliance_rate": float(comply.mean()),
            "compliance_n": int(comply.sum()),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--category", default="Gender_identity")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--level", type=int, default=3, help="Ladder level 1-6 (default t3)")
    ap.add_argument("--patch-layer", type=int, default=56)
    ap.add_argument("--directions", nargs="+", default=["stereo", "other"])
    ap.add_argument("--alpha-min", type=float, default=-0.5)
    ap.add_argument("--alpha-max", type=float, default=0.5)
    ap.add_argument("--alpha-step", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.2,
                    help="Hold out this fraction to evaluate steering (direction from train)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    level_idx = args.level - 1
    cache_dir = model_cache_dir(args.cache_dir, args.model)
    c = _load(cache_dir, args.category)
    n = c["base"].shape[0]
    L = args.patch_layer
    um = c["unembed"]
    alphas = np.arange(args.alpha_min, args.alpha_max + 1e-9, args.alpha_step)

    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_frac, random_state=args.seed, stratify=None
    )

    results = {
        "model": args.model,
        "category": args.category,
        "level": args.level,
        "patch_layer": L,
        "n": n,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "directions": {},
    }

    final_t3 = c["ladder"][:, level_idx, -1, :].astype(np.float32)
    h_t3_L = c["ladder"][:, level_idx, L, :].astype(np.float32)
    h_base_L = c["base"][:, L, :].astype(np.float32)

    for direction in args.directions:
        if direction == "stereo":
            logits = c["ladder_logits"]
            target_ids = c["stereo_ids"]
        else:
            logits = c["ladder_other_logits"]
            target_ids = c["other_ids"]

        y = _comply_from_logits(logits[:, level_idx, :], target_ids)
        v, act_rms = _probe_direction(c["base"][:, L, :], y, train_idx)
        print(f"  probe direction RMS scale at L{L}: {act_rms:.1f}")

        # Natural compliance on test set (from cached logits)
        natural_rate = float(y[test_idx].mean())
        natural_n = int(y[test_idx].sum())

        if direction != "stereo":
            print(
                f"\n[skip] {direction}: ladder_other_acts not cached — "
                "collect with collect_ladder_acts.py for both directions"
            )
            continue

        dir_out = {
            "natural_compliance_test": natural_rate,
            "natural_compliance_n_test": natural_n,
            "probe_auc_note": "direction trained on train split only",
            "modes": {},
        }

        for mode in ("final_add", "layer_steer", "layer_replace"):
            dir_out["modes"][mode] = _sweep(
                final_t3[test_idx],
                h_t3_L[test_idx],
                h_base_L[test_idx],
                v,
                act_rms,
                target_ids[test_idx],
                um,
                alphas,
                mode,
            )

        results["directions"][direction] = dir_out
        print(f"\n=== {direction} nudge @ t{args.level}, layer L{L} (test n={len(test_idx)}) ===")
        print(f"  Natural compliance: {natural_rate:.1%} ({natural_n})")
        for mode in ("final_add", "layer_steer", "layer_replace"):
            print(f"  [{mode}]")
            for r in dir_out["modes"][mode]:
                print(
                    f"    alpha={r['alpha']:+.2f}  "
                    f"compliance={r['compliance_rate']:.1%} ({r['compliance_n']})"
                )

    stem = f"causal_steering_{args.model.replace('/', '_')}_{args.category}_t{args.level}_L{L}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {json_path}")

    # Plot
    fig, axes = plt.subplots(1, len(args.directions), figsize=(5 * len(args.directions), 4), squeeze=False)
    for ax, direction in zip(axes[0], args.directions, strict=True):
        d = results["directions"][direction]
        nat = d["natural_compliance_test"]
        for mode, color, ls in (
            ("final_add", "C0", "-"),
            ("layer_steer", "C2", "-."),
            ("layer_replace", "C1", "--"),
        ):
            rows = d["modes"][mode]
            ax.plot(
                [r["alpha"] for r in rows],
                [r["compliance_rate"] for r in rows],
                ls=ls,
                marker="o",
                color=color,
                label=mode,
            )
        ax.axhline(nat, color="grey", ls=":", label=f"natural t{args.level} ({nat:.1%})")
        ax.set_xlabel("α × layer RMS (steering strength)")
        ax.set_ylabel("Compliance rate (test set)")
        ax.set_title(f"{direction} nudge")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Causal steering — {args.category} L{L} t{args.level}", fontsize=11)
    fig.tight_layout()
    png_path = out_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
