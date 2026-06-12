"""Single plot: assertiveness ladder vs compliance (flip→stereo), all categories pooled.

Compliance = nudge toward the stereotyped group and argmax equals that group.
Pools Gender_identity, Race_ethnicity, and SES into one curve; optional faint
per-category lines for reference.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LADDER_LABELS = [
    "t1\nPerhaps",
    "t2\nI think",
    "t3\nFairly sure",
    "t4\nCertain",
    "t5\nNo doubt",
    "t6\nMust answer",
]
DEFAULT_CATS = [
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


def compliance_rates(ladder_logits, stereo_ids):
    """P(argmax == stereotyped group) at each ladder level."""
    arg = ladder_logits.argmax(2)
    return np.array([(arg[:, t] == stereo_ids).mean() for t in range(arg.shape[1])])


def load_category(cat: str, model_glob: str):
    f = glob.glob(f"cache/*{model_glob}*/sensitivity_{cat}.npz")
    if not f:
        f = glob.glob(f"cache/**/sensitivity_{cat}.npz", recursive=True)
    if not f:
        raise FileNotFoundError(f"no sensitivity cache for {cat}")
    D = np.load(f[0])
    return D["ladder_stereo"], D["stereo_ids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATS)
    ap.add_argument("--model-glob", default="Qwen_Qwen3-32B")
    ap.add_argument("--out", default="results/sensitivity_ladder_compliance_combined.png")
    ap.add_argument("--no-per-cat", action="store_true", help="Hide per-category lines")
    args = ap.parse_args()

    ladders, sids, labels = [], [], []
    for cat in args.categories:
        lad, sid = load_category(cat, args.model_glob)
        ladders.append(lad)
        sids.append(sid)
        labels.append(cat.replace("_", " "))

    ladder_all = np.concatenate(ladders, axis=0)
    sid_all = np.concatenate(sids, axis=0)
    n_total = len(sid_all)
    pooled = compliance_rates(ladder_all, sid_all)

    x = np.arange(1, len(pooled) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))

    if not args.no_per_cat:
        for lad, sid, lab in zip(ladders, sids, labels, strict=True):
            r = compliance_rates(lad, sid)
            ax.plot(x, r, "--", alpha=0.45, lw=1.5, label=lab)

    ax.plot(
        x, pooled, "o-", color="C0", lw=2.5, ms=9,
        label=f"All categories (n={n_total:,})",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(LADDER_LABELS[: len(x)], fontsize=9)
    ax.set_xlabel("Assertiveness ladder level")
    ax.set_ylabel("Compliance rate  P(argmax = stereotyped group)")
    ax.set_ylim(0, max(0.55, pooled.max() * 1.15))
    ax.set_title("Qwen3-32B · nudge toward stereotyped group")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    for i, v in enumerate(pooled, 1):
        print(f"  t{i}: {v:.3f}")


if __name__ == "__main__":
    main()
