"""Logit-lens: at which layer does Unknown@base -> O (other) commit, under nudge->S?

For the Unknown-baseline backfire set (baseline predicts Unknown, nudge-toward-
stereo final prediction = other group), project each layer's resid_post through
the final RMSNorm + A/B/C unembed and track when the per-layer argmax becomes O.
"""
from __future__ import annotations

import glob
import sys

import numpy as np

from mech_interp_bbq.hf_backend import project_resid_to_abc

CDIR = "cache/Qwen_Qwen3-32B"


def run(cat):
    lg = glob.glob(f"{CDIR}/*{cat}*logits.npz")[0]
    ac = glob.glob(f"{CDIR}/*{cat}*acts.npz")[0]
    um = np.load(f"{CDIR}/unembed.npz")
    L = np.load(lg)
    A = np.load(ac)
    base, nst = L["base_logits"], L["nudged_logits"]
    s, u, o = L["stereo_ids"], L["unknown_ids"], L["other_ids"]
    bp, sp = base.argmax(1), nst.argmax(1)
    mask = (bp == u) & (sp == o)            # Unknown@base -> O after nudge->S
    idx = np.where(mask)[0]
    acts = A["nudge_stereo_acts"]           # (n, nLayers, d)
    nL = acts.shape[1]
    nw, eps, abcU = um["norm_weight"], float(um["norm_eps"]), um["abc_unembed"]

    sub = acts[idx].astype(np.float32)      # (m, nLayers, d)
    proj = project_resid_to_abc(sub.reshape(-1, sub.shape[-1]), nw, eps, abcU)
    proj = proj.reshape(len(idx), nL, 3)     # logits over [A,B,C] cols per layer
    am = proj.argmax(-1)                     # (m, nLayers) column index A/B/C
    si, ui, oi = s[idx], u[idx], o[idx]
    isO = am == oi[:, None]                  # per layer: lens predicts O?
    isU = am == ui[:, None]

    # softmax over the 3 option columns -> per-case prob of S/U/O at each layer
    ex = np.exp(proj - proj.max(-1, keepdims=True))
    p3 = ex / ex.sum(-1, keepdims=True)
    rows = np.arange(len(idx))
    pS = p3[rows[:, None], np.arange(nL)[None, :], si[:, None]]
    pU = p3[rows[:, None], np.arange(nL)[None, :], ui[:, None]]
    pO = p3[rows[:, None], np.arange(nL)[None, :], oi[:, None]]

    print(f"\n=== {cat}: {len(idx)} Unknown@base -> O backfire cases ===")
    fracO, fracU = isO.mean(0), isU.mean(0)
    print(" L   %O   %U   meanP(S/U/O)        argmax=O")
    for layer in range(nL):
        bar = "#" * int(round(fracO[layer] * 30))
        print(f"  L{layer:02d} {fracO[layer]:.2f} {fracU[layer]:.2f}  "
              f"{pS[:, layer].mean():.2f}/{pU[:, layer].mean():.2f}/{pO[:, layer].mean():.2f}  {bar}")

    # (a) first layer O becomes the lens argmax (per case)
    first_O = np.array([np.argmax(isO[r]) if isO[r].any() else nL for r in range(len(idx))])
    # (b) commit layer: first layer from which lens stays O until the final layer
    commit = []
    for r in range(len(idx)):
        last = nL
        for layer in range(nL - 1, -1, -1):
            if isO[r, layer]:
                last = layer
            else:
                break
        commit.append(last)
    commit = np.array(commit)
    # (c) population-level: first layer where O leads in >=50% of cases
    maj = next((layer for layer in range(nL) if fracO[layer] >= 0.5), None)

    print(f"first layer O is argmax:  median={int(np.median(first_O))} mean={first_O.mean():.1f}")
    print(f"commit-to-O (stays O->end): median={int(np.median(commit))} mean={commit.mean():.1f} "
          f"min={commit.min()} max={commit.max()}")
    print(f"first layer O leads >=50% of cases: {maj}   (total {nL} layers, 0-indexed)")


if __name__ == "__main__":
    for c in sys.argv[1:] or ["Gender_identity", "Race_ethnicity", "SES"]:
        run(c)
