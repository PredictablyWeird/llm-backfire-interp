"""Show A/B/C logits for strict-backfire cases: baseline vs nudged (flipped)."""
from __future__ import annotations

import glob
import sys

import numpy as np

LETTERS = ["A", "B", "C"]


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def role_row(i, s_ids, u_ids, o_ids):
    """Map each column 0/1/2 -> 'S'/'U'/'O'."""
    role = ["?"] * 3
    role[s_ids[i]] = "S"
    role[u_ids[i]] = "U"
    role[o_ids[i]] = "O"
    return role


def fmt(vec, role, argmax):
    parts = []
    for c in range(3):
        mark = "*" if c == argmax else " "
        parts.append(f"{LETTERS[c]}({role[c]}){mark}{vec[c]:+6.2f}")
    return "  ".join(parts)


def main(category):
    f = glob.glob(f"cache/Qwen_Qwen3-32B/*{category}*logits.npz")[0]
    d = np.load(f)
    base, nstereo = d["base_logits"], d["nudged_logits"]
    s, u, o = d["stereo_ids"], d["unknown_ids"], d["other_ids"]
    bp, sp = base.argmax(1), nstereo.argmax(1)
    # strict backfire from stereo: baseline picks stereo, stereo-nudge flips to other
    bf = np.where((bp == s) & (sp == o))[0]
    print(f"\n=== {category}: {len(bf)} strict backfire (stereo->other) cases ===")
    print("legend: S=stereo answer, U=unknown, O=other group; * = argmax\n")
    for i in bf:
        rb = role_row(i, s, u, o)
        pb, pn = softmax(base[i]), softmax(nstereo[i])
        print(f"[idx {i}]")
        print(f"  baseline      logits: {fmt(base[i], rb, bp[i])}")
        print(f"                 probs: S={pb[s[i]]:.2f} U={pb[u[i]]:.2f} O={pb[o[i]]:.2f}  -> picks {rb[bp[i]]}")
        print(f"  nudge->stereo logits: {fmt(nstereo[i], rb, sp[i])}")
        print(f"                 probs: S={pn[s[i]]:.2f} U={pn[u[i]]:.2f} O={pn[o[i]]:.2f}  -> picks {rb[sp[i]]}  [FLIP]")
        print()


if __name__ == "__main__":
    cats = sys.argv[1:] or ["Gender_identity", "Race_ethnicity", "SES"]
    for c in cats:
        main(c)
