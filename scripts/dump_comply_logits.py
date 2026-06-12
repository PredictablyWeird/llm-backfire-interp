"""Dump 'comply' (non-backfire) cases: baseline picks S, nudge->S keeps S."""
from __future__ import annotations

import glob
import json
import sys

import numpy as np


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def collect(category, limit=10):
    f = glob.glob(f"cache/Qwen_Qwen3-32B/*{category}*logits.npz")[0]
    d = np.load(f)
    base, nstereo = d["base_logits"], d["nudged_logits"]
    s, u, o = d["stereo_ids"], d["unknown_ids"], d["other_ids"]
    bp, sp = base.argmax(1), nstereo.argmax(1)
    # comply: baseline picks stereo, nudge toward stereo stays on stereo
    comply = np.where((bp == s) & (sp == s))[0][:limit]
    rows = []
    for i in comply:
        pb, pn = softmax(base[i]), softmax(nstereo[i])
        idx = (s[i], u[i], o[i])
        rows.append({
            "idx": int(i),
            "bl": [round(float(base[i][j]), 1) for j in idx],
            "bp": [round(float(pb[j]), 2) for j in idx],
            "nl": [round(float(nstereo[i][j]), 1) for j in idx],
            "np": [round(float(pn[j]), 2) for j in idx],
        })
    return rows


if __name__ == "__main__":
    cat = sys.argv[1] if len(sys.argv) > 1 else "Gender_identity"
    print(json.dumps({cat: collect(cat)}, indent=0))
