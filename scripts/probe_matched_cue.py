"""Probe the SURFACE-MATCHED cues: does pressure-vs-evidence generalize once form is
controlled? Same metrics as probe_cue_features.py (CV, LOCO, geometry), on the matched
4+4 cue set. LOCO is the decisive test for a shared semantic influence feature.
"""
from __future__ import annotations

import glob

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CDIR = "cache/meta-llama_Llama-3.2-1B"
CAT = "Gender_identity"
PRESSURE = ["pref", "emotion", "social", "insist"]
EVIDENCE = ["survey", "expert", "stats", "study"]
ALL = PRESSURE + EVIDENCE
LABEL = {c: 0 for c in PRESSURE} | {c: 1 for c in EVIDENCE}


def _clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


def main():
    base = np.load(glob.glob(f"{CDIR}/*{CAT}_user_preference_*_acts.npz")[0])["base_acts"].astype(np.float32)
    B = np.load(f"{CDIR}/matched_cue_{CAT}_acts.npz")
    cues = {c: B[c].astype(np.float32) for c in ALL}
    nL = base.shape[1]
    print(f"layers={nL} n_per_cue={base.shape[0]}\nPRESSURE={PRESSURE}\nEVIDENCE={EVIDENCE}\n")
    print(f"{'L':>3} {'2cls_CV':>8} {'LOCO':>6} {'8way_CV':>8}   per-held-out-cue acc")
    best = (-1, -1.0)
    for L in range(nL):
        feat = {c: (cues[c][:, L, :] - base[:, L, :]) for c in ALL}
        X = np.concatenate([feat[c] for c in ALL], 0)
        y = np.concatenate([np.full(feat[c].shape[0], LABEL[c]) for c in ALL])
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        cv2 = cross_val_score(_clf(), X, y, cv=skf, scoring="accuracy").mean()

        loco = {}
        for held in ALL:
            tr = [c for c in ALL if c != held]
            Xtr = np.concatenate([feat[c] for c in tr], 0)
            ytr = np.concatenate([np.full(feat[c].shape[0], LABEL[c]) for c in tr])
            pred = _clf().fit(Xtr, ytr).predict(feat[held])
            loco[held] = (pred == LABEL[held]).mean()
        loco_mean = float(np.mean(list(loco.values())))
        if loco_mean > best[1]:
            best = (L, loco_mean)

        y8 = np.concatenate([np.full(feat[c].shape[0], i) for i, c in enumerate(ALL)])
        cv8 = cross_val_score(_clf(), X, y8, cv=skf, scoring="accuracy").mean()
        perc = " ".join(f"{c[:4]}={loco[c]:.2f}" for c in ALL)
        print(f"{L:>3} {cv2:>8.3f} {loco_mean:>6.2f} {cv8:>8.3f}   {perc}")

    print(f"\nbest LOCO = {best[1]:.2f} at layer {best[0]}  (chance = 0.50)")

    L = best[0]
    mids = {c: (cues[c][:, L, :] - base[:, L, :]).mean(0) for c in ALL}
    M = np.stack([mids[c] / (np.linalg.norm(mids[c]) + 1e-8) for c in ALL])
    S = M @ M.T
    wi = [S[i, j] for i in range(len(ALL)) for j in range(i + 1, len(ALL)) if LABEL[ALL[i]] == LABEL[ALL[j]]]
    bw = [S[i, j] for i in range(len(ALL)) for j in range(i + 1, len(ALL)) if LABEL[ALL[i]] != LABEL[ALL[j]]]
    print(f"layer {L} geometry: within-class cos={np.mean(wi):.2f}  between-class cos={np.mean(bw):.2f}")


if __name__ == "__main__":
    main()
