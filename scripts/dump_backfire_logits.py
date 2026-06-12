"""Dump strict-backfire cases (baseline + nudged logits/probs) to JSON."""
from __future__ import annotations

import glob
import json

import numpy as np


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def collect(category):
    f = glob.glob(f"cache/Qwen_Qwen3-32B/*{category}*logits.npz")[0]
    d = np.load(f)
    base, nstereo = d["base_logits"], d["nudged_logits"]
    s, u, o = d["stereo_ids"], d["unknown_ids"], d["other_ids"]
    bp, sp = base.argmax(1), nstereo.argmax(1)
    bf = np.where((bp == s) & (sp == o))[0]
    rows = []
    for i in bf:
        pb, pn = softmax(base[i]), softmax(nstereo[i])
        idx = (s[i], u[i], o[i])
        bl = [round(float(base[i][j]), 1) for j in idx]
        bpr = [round(float(pb[j]), 2) for j in idx]
        nl = [round(float(nstereo[i][j]), 1) for j in idx]
        npr = [round(float(pn[j]), 2) for j in idx]
        rows.append({"idx": int(i), "bl": bl, "bp": bpr, "nl": nl, "np": npr})
    return rows


if __name__ == "__main__":
    data = {c: collect(c) for c in ["Gender_identity", "Race_ethnicity", "SES"]}
    # Emit a TS literal ready to paste into the canvas.
    print("const CASES: Record<string, Case[]> = " + json.dumps(data) + ";")
