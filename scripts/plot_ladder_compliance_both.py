"""Assertiveness ladder vs compliance for both nudge directions (pooled categories).

Compliance = nudge toward target group and argmax equals that group.
t0 is the no-nudge baseline; t1–t6 are assertiveness ladder levels.
Stereo uses all examples; other uses only examples with an 'other' label.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LADDER_LABELS = [
    "t0\nNo nudge",
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


def baseline_compliance(base_logits, target_ids):
    """P(argmax == target) with no nudge."""
    return (base_logits.argmax(1) == target_ids).mean()


def compliance_rates(ladder_logits, target_ids, base_logits=None):
    """P(argmax == target) at t0 (baseline) and each ladder level."""
    rates = []
    if base_logits is not None:
        rates.append(baseline_compliance(base_logits, target_ids))
    arg = ladder_logits.argmax(2)
    rates.extend((arg[:, t] == target_ids).mean() for t in range(arg.shape[1]))
    return np.array(rates)


def load_category(cat: str, model_glob: str):
    f = glob.glob(f"cache/*{model_glob}*/sensitivity_{cat}.npz")
    if not f:
        f = glob.glob(f"cache/**/sensitivity_{cat}.npz", recursive=True)
    if not f:
        raise FileNotFoundError(f"no sensitivity cache for {cat}")
    D = np.load(f[0])
    return (
        D["base_logits"],
        D["ladder_stereo"],
        D["ladder_other"],
        D["stereo_ids"],
        D["other_ids"],
        D["has_other"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATS)
    ap.add_argument("--model-glob", default="Qwen_Qwen3-32B")
    ap.add_argument("--out", default="results/sensitivity_ladder_compliance_both.png")
    args = ap.parse_args()

    bases, bases_other = [], []
    stereo_ladders, other_ladders = [], []
    stereo_ids_all, other_ids_all = [], []

    for cat in args.categories:
        base, lad_s, lad_o, sid, oid, has_o = load_category(cat, args.model_glob)
        bases.append(base)
        stereo_ladders.append(lad_s)
        stereo_ids_all.append(sid)
        m = has_o
        bases_other.append(base[m])
        other_ladders.append(lad_o[m])
        other_ids_all.append(oid[m])

    base_all = np.concatenate(bases, axis=0)
    base_other = np.concatenate(bases_other, axis=0)
    ladder_stereo = np.concatenate(stereo_ladders, axis=0)
    sid_all = np.concatenate(stereo_ids_all, axis=0)
    ladder_other = np.concatenate(other_ladders, axis=0)
    oid_all = np.concatenate(other_ids_all, axis=0)

    n_stereo = len(sid_all)
    n_other = len(oid_all)
    stereo_rates = compliance_rates(ladder_stereo, sid_all, base_all)
    other_rates = compliance_rates(ladder_other, oid_all, base_other)

    x = np.arange(len(stereo_rates))
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        x, stereo_rates, "o-", color="C0", lw=2.5, ms=9,
        label=f"Nudge → stereotyped group (n={n_stereo:,})",
    )
    ax.plot(
        x, other_rates, "s-", color="C1", lw=2.5, ms=9,
        label=f"Nudge → other group (n={n_other:,})",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(LADDER_LABELS[: len(x)], fontsize=9)
    ax.set_xlabel("Assertiveness ladder level")
    ax.set_ylabel("Compliance rate  P(argmax = target group)")
    ax.set_ylim(0, max(0.55, stereo_rates.max(), other_rates.max()) * 1.1)
    ax.set_title("Qwen3-32B · all categories pooled")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    print("  toward stereo:")
    for i, v in enumerate(stereo_rates):
        print(f"    t{i}: {v:.3f}")
    print("  toward other:")
    for i, v in enumerate(other_rates):
        print(f"    t{i}: {v:.3f}")


if __name__ == "__main__":
    main()
