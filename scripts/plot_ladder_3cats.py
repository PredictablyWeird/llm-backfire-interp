"""3-panel ladder vs flip rate for Gender, Race, SES."""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LADDER_LABELS = ["t1", "t2", "t3", "t4", "t5", "t6"]
CATS = ["Gender_identity", "Race_ethnicity", "SES"]


def flip_rates(ladder, tgt):
    arg = ladder.argmax(2)
    return [(arg[:, t] == tgt).mean() for t in range(arg.shape[1])]


fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
for ax, cat in zip(axes, CATS, strict=True):
    f = glob.glob(f"cache/*Qwen*/sensitivity_{cat}.npz")[0]
    D = np.load(f)
    s, o, ho = D["stereo_ids"], D["other_ids"], D["has_other"]
    st = flip_rates(D["ladder_stereo"], s)
    ot = flip_rates(D["ladder_other"][ho], o[ho])
    x = np.arange(1, 7)
    ax.plot(x, st, "o-", lw=2, ms=7, label="→ stereo")
    ax.plot(x, ot, "s-", lw=2, ms=7, label="→ other")
    ax.set_xticks(x)
    ax.set_xticklabels(LADDER_LABELS)
    ax.set_title(cat.replace("_", " "))
    ax.set_xlabel("Ladder level")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel("Flip rate  P(argmax = target)")
fig.suptitle("Qwen3-32B · assertiveness ladder sensitivity", y=1.02)
fig.tight_layout()
out = Path("results/sensitivity_ladder_flip_rate_3cats.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
