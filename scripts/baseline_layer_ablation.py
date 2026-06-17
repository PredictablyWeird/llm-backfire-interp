"""Baseline layer ablation: which layers drive Unknown vs Stereo margin?

Linear patch on cached base_acts (no GPU):
  ablated_final(L) = final - (base[L] - base[L-1])

Positive Δ(U−S margin) after ablating L → layer L normally *lowered* Unknown margin
(i.e. it pushed toward Unknown / away from Stereo when present).

Example:
    uv run python scripts/baseline_layer_ablation.py --category Gender_identity
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mech_interp_bbq.hf_backend import project_resid_to_abc
from mech_interp_bbq.prompts import model_cache_dir


def _find_base_acts(cache_dir: Path, category: str) -> np.ndarray:
    ladder = cache_dir / f"ladder_acts_{category}.npz"
    if ladder.exists():
        return np.load(ladder, mmap_mode="r")["base_acts"]
    import glob

    hits = glob.glob(str(cache_dir / f"3choice_*_{category}_*_acts.npz"))
    if not hits:
        raise SystemExit(f"No base_acts for {category}")
    best = max(hits, key=lambda p: np.load(p, mmap_mode="r")["base_acts"].shape[0])
    return np.load(best, mmap_mode="r")["base_acts"]


def _metrics(abc: np.ndarray, sids: np.ndarray, uids: np.ndarray) -> dict:
    rows = np.arange(len(sids))
    margin_us = abc[rows, uids] - abc[rows, sids]  # U - S
    margin_su = -margin_us
    arg = abc.argmax(1)
    return {
        "mean_margin_u_minus_s": float(margin_us.mean()),
        "mean_margin_s_minus_u": float(margin_su.mean()),
        "frac_argmax_unknown": float((arg == uids).mean()),
        "frac_argmax_stereo": float((arg == sids).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--category", default="Gender_identity")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    cache_dir = model_cache_dir(args.cache_dir, args.model)
    sens = np.load(cache_dir / f"sensitivity_{args.category}.npz")
    um = np.load(cache_dir / "unembed.npz")
    umd = {
        "abc_unembed": um["abc_unembed"],
        "norm_weight": um["norm_weight"],
        "norm_eps": float(um["norm_eps"]),
    }
    base = _find_base_acts(cache_dir, args.category).astype(np.float32)
    sids = sens["stereo_ids"].astype(int)
    uids = sens["unknown_ids"].astype(int)
    if base.shape[0] != len(sids):
        raise SystemExit(f"acts n={base.shape[0]} != sens n={len(sids)}")

    n_layers = base.shape[1]
    final = base[:, -1, :]
    full_abc = project_resid_to_abc(final, umd["norm_weight"], umd["norm_eps"], umd["abc_unembed"])
    full_m = _metrics(full_abc, sids, uids)

    rows = []
    for L in range(1, n_layers):
        abl = final - (base[:, L, :] - base[:, L - 1, :])
        abc = project_resid_to_abc(abl, umd["norm_weight"], umd["norm_eps"], umd["abc_unembed"])
        m = _metrics(abc, sids, uids)
        rows.append({
            "layer": L,
            **m,
            "delta_margin_u_minus_s": m["mean_margin_u_minus_s"] - full_m["mean_margin_u_minus_s"],
            "delta_frac_unknown": m["frac_argmax_unknown"] - full_m["frac_argmax_unknown"],
            "delta_frac_stereo": m["frac_argmax_stereo"] - full_m["frac_argmax_stereo"],
        })

    stem = f"baseline_ablation_{args.model.replace('/', '_')}_{args.category}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": args.model,
        "category": args.category,
        "n": int(base.shape[0]),
        "full_baseline": full_m,
        "per_layer": rows,
    }
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    layers = [r["layer"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(layers, [r["mean_margin_u_minus_s"] for r in rows], "o-", ms=3, label="ablated")
    ax.axhline(full_m["mean_margin_u_minus_s"], color="grey", ls="--", label="full baseline")
    ax.set_xlabel("Layer ablated")
    ax.set_ylabel("Mean logit(U) − logit(S)")
    ax.set_title("Unknown − Stereo margin after ablation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.bar(layers, [r["delta_margin_u_minus_s"] for r in rows], width=0.8, color="steelblue")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Layer ablated")
    ax.set_ylabel("Δ(U−S margin) vs full")
    ax.set_title("Layer effect on Unknown margin\n(negative = layer boosted Unknown)")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 0]
    ax.plot(layers, [r["frac_argmax_unknown"] for r in rows], "o-", ms=3, label="P(argmax=U)")
    ax.plot(layers, [r["frac_argmax_stereo"] for r in rows], "s-", ms=3, label="P(argmax=S)")
    ax.axhline(full_m["frac_argmax_unknown"], color="C0", ls=":", alpha=0.7)
    ax.axhline(full_m["frac_argmax_stereo"], color="C1", ls=":", alpha=0.7)
    ax.set_xlabel("Layer ablated")
    ax.set_ylabel("Fraction")
    ax.set_title("Argmax rates after ablation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.bar(layers, [r["delta_frac_unknown"] for r in rows], width=0.8, color="C0", alpha=0.7, label="Δ Unknown")
    ax.bar(layers, [r["delta_frac_stereo"] for r in rows], width=0.8, color="C1", alpha=0.7, label="Δ Stereo")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Layer ablated")
    ax.set_ylabel("Δ argmax fraction")
    ax.set_title("Change in argmax vs full baseline")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Baseline layer ablation — {args.category}", fontsize=11)
    fig.tight_layout()
    png_path = out_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    # Print summary
    print(f"{args.category} n={base.shape[0]}")
    print(f"Full baseline: margin U-S={full_m['mean_margin_u_minus_s']:.2f}  "
          f"argmax U={full_m['frac_argmax_unknown']:.1%} S={full_m['frac_argmax_stereo']:.1%}")
    top_u_boost = sorted(rows, key=lambda r: -r["delta_margin_u_minus_s"])[:8]
    top_u_supp = sorted(rows, key=lambda r: r["delta_margin_u_minus_s"])[:8]
    print("\nLayers whose REMOVAL most *increases* U-S margin (normally pushed toward Stereo):")
    for r in top_u_boost:
        print(f"  L{r['layer']:2d}: Δmargin={r['delta_margin_u_minus_s']:+.2f}  "
              f"ΔP(U)={r['delta_frac_unknown']:+.1%}")
    print("\nLayers whose REMOVAL most *decreases* U-S margin (normally boosted Unknown):")
    for r in top_u_supp:
        print(f"  L{r['layer']:2d}: Δmargin={r['delta_margin_u_minus_s']:+.2f}  "
              f"ΔP(U)={r['delta_frac_unknown']:+.1%}")
    print(f"\nWrote {json_path}\nWrote {png_path}")


if __name__ == "__main__":
    main()
