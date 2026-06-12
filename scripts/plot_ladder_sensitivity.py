"""Plot assertiveness ladder vs flip rate from sensitivity_<cat>.npz."""
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


def flip_rates(ladder_logits, base_logits, target_ids):
    """Fraction where argmax == target at each ladder level."""
    tgt = target_ids
    arg = ladder_logits.argmax(2)
    return [(arg[:, t] == tgt).mean() for t in range(arg.shape[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Gender_identity")
    ap.add_argument("--model-glob", default="Qwen_Qwen3-32B")
    ap.add_argument("--out", default="results/sensitivity_ladder_flip_rate.png")
    args = ap.parse_args()

    f = glob.glob(f"cache/*{args.model_glob}*/sensitivity_{args.category}.npz")
    if not f:
        f = glob.glob(f"cache/**/sensitivity_{args.category}.npz", recursive=True)
    D = np.load(f[0])
    base = D["base_logits"]
    sids, oids = D["stereo_ids"], D["other_ids"]
    has_other = D["has_other"]

    stereo_fr = flip_rates(D["ladder_stereo"], base, sids)
    other_mask = has_other
    other_fr = flip_rates(D["ladder_other"][other_mask], base[other_mask], oids[other_mask])

    x = np.arange(1, len(stereo_fr) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, stereo_fr, "o-", linewidth=2, markersize=8, label="Nudge → stereotyped group")
    ax.plot(x, other_fr, "s-", linewidth=2, markersize=8, label="Nudge → other group")
    ax.set_xticks(x)
    ax.set_xticklabels(LADDER_LABELS[: len(x)], fontsize=9)
    ax.set_xlabel("Assertiveness ladder level")
    ax.set_ylabel("Flip rate  P(argmax = target)")
    ax.set_ylim(0, 0.75)
    ax.set_title(f"Qwen3-32B · {args.category} · n={len(base)}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
