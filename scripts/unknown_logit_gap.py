"""When baseline predicts Unknown, how do the S vs O option logits compare?

Looks at the no-nudge (baseline) pass. For cases where argmax == Unknown,
reports the 3-way option logits/probs and the S-vs-O gap, both for ALL
Unknown-baseline cases and for the Unknown->O backfire subset.
"""
from __future__ import annotations

import glob
import sys

import numpy as np

CDIR = "cache/Qwen_Qwen3-32B"


def _softmax3(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def summarize(tag, lg_s, lg_u, lg_o):
    p = _softmax3(np.stack([lg_s, lg_u, lg_o], axis=1))
    pS, pU, pO = p[:, 0], p[:, 1], p[:, 2]
    gap = np.abs(lg_s - lg_o)                       # |logit_S - logit_O|
    # margin of U over the better of S/O (in logit space)
    u_margin = lg_u - np.maximum(lg_s, lg_o)
    print(f"\n  [{tag}]  n={len(lg_u)}")
    print(f"    mean prob   S={pS.mean():.3f}  U={pU.mean():.3f}  O={pO.mean():.3f}")
    print(f"    mean logit  S={lg_s.mean():.2f}  U={lg_u.mean():.2f}  O={lg_o.mean():.2f}")
    print(f"    |logit_S - logit_O|: mean={gap.mean():.2f} median={np.median(gap):.2f} "
          f"p90={np.percentile(gap, 90):.2f}")
    print(f"    U margin over max(S,O) [logits]: mean={u_margin.mean():.2f} "
          f"median={np.median(u_margin):.2f}")
    print(f"    mean prob mass on (S+O)={ (pS+pO).mean():.3f}")


def run(cat):
    lg = glob.glob(f"{CDIR}/*{cat}*logits.npz")[0]
    L = np.load(lg)
    base, nst = L["base_logits"], L["nudged_logits"]
    s, u, o = L["stereo_ids"], L["unknown_ids"], L["other_ids"]
    rows = np.arange(len(base))
    bs, bu, bo = base[rows, s], base[rows, u], base[rows, o]
    bp = base.argmax(1)
    sp = nst.argmax(1)

    unk = bp == u                                    # baseline predicts Unknown
    backfire = unk & (sp == o)                       # Unknown -> O after nudge->S
    print(f"\n=== {cat} ===")
    summarize("all Unknown@base", bs[unk], bu[unk], bo[unk])
    summarize("Unknown@base -> O backfire", bs[backfire], bu[backfire], bo[backfire])


if __name__ == "__main__":
    for c in sys.argv[1:] or ["Gender_identity", "Race_ethnicity", "SES"]:
        run(c)
