"""Probe Llama-1B for higher-level influence features: SOCIAL PRESSURE vs EVIDENCE.

Hypothesis: the model compresses many surface cue types into a small set of latent
"influence" representations. If so, a linear probe trained to separate pressure-cues
from evidence-cues should *generalize to held-out cue types* it never saw in training
(leave-one-cue-out), not just memorize individual cues.

Inputs (Gender_identity, same 2836 examples, target = stereotyped group):
  * user_preference, baseline  <- existing  ..._user_preference_..._acts.npz
  * emotional, role_play, survey_preference, weak_evidence, few_shot
                                <- cue_probe_Gender_identity_acts.npz

We probe on the cue-induced DELTA (cue_resid - baseline_resid) to isolate the
influence from the (shared) example content.
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

PRESSURE = ["user_preference", "emotional", "role_play"]
EVIDENCE = ["survey_preference", "weak_evidence", "few_shot"]
ALL_CUES = PRESSURE + EVIDENCE
LABEL = {c: 0 for c in PRESSURE} | {c: 1 for c in EVIDENCE}   # 0=pressure 1=evidence


def load_acts():
    up = glob.glob(f"{CDIR}/*{CAT}_user_preference_*_acts.npz")[0]
    A = np.load(up)
    base = A["base_acts"].astype(np.float32)          # (n, L, d)
    cues = {"user_preference": A["nudge_stereo_acts"].astype(np.float32)}
    B = np.load(f"{CDIR}/cue_probe_{CAT}_acts.npz")
    for c in B.files:
        cues[c] = B[c].astype(np.float32)
    n = base.shape[0]
    for c in cues:
        assert cues[c].shape[0] == n, f"{c} n mismatch"
    return base, cues, base.shape[1]


def _clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


def main():
    base, cues, nL = load_acts()
    print(f"layers={nL}  cues={list(cues)}  n_per_cue={base.shape[0]}")
    print(f"PRESSURE={PRESSURE}\nEVIDENCE={EVIDENCE}\n")

    rng = np.random.default_rng(0)
    print(f"{'L':>3} {'2cls_CV':>8} {'LOCO':>6} {'6way_CV':>8}   per-held-out-cue acc")
    for L in range(nL):
        # delta = cue - baseline, isolates the influence at the answer position
        feat = {c: (cues[c][:, L, :] - base[:, L, :]) for c in ALL_CUES}

        # ---- (1) 2-class CV (pressure vs evidence), all cues pooled ----
        X = np.concatenate([feat[c] for c in ALL_CUES], 0)
        y = np.concatenate([np.full(feat[c].shape[0], LABEL[c]) for c in ALL_CUES])
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        cv2 = cross_val_score(_clf(), X, y, cv=skf, scoring="accuracy").mean()

        # ---- (2) leave-one-cue-out: train on other cues, test on held-out cue ----
        loco = {}
        for held in ALL_CUES:
            tr = [c for c in ALL_CUES if c != held]
            Xtr = np.concatenate([feat[c] for c in tr], 0)
            ytr = np.concatenate([np.full(feat[c].shape[0], LABEL[c]) for c in tr])
            clf = _clf().fit(Xtr, ytr)
            pred = clf.predict(feat[held])
            loco[held] = (pred == LABEL[held]).mean()
        loco_mean = np.mean(list(loco.values()))

        # ---- (3) 6-way cue identity CV (chance=1/6) ----
        y6 = np.concatenate([np.full(feat[c].shape[0], i) for i, c in enumerate(ALL_CUES)])
        cv6 = cross_val_score(_clf(), X, y6, cv=skf, scoring="accuracy").mean()

        perc = "  ".join(f"{c[:4]}={loco[c]:.2f}" for c in ALL_CUES)
        print(f"{L:>3} {cv2:>8.3f} {loco_mean:>6.2f} {cv6:>8.3f}   {perc}")

    # ---- (4) geometry: cosine sim of mean cue-delta directions at best layer ----
    print("\nper-cue mean-delta cosine similarity (layer with best LOCO):")
    # recompute LOCO per layer cheaply via the 2-class direction is overkill; just
    # show geometry at a representative mid/late layer.
    for L in [nL // 2, nL - 2]:
        mids = {c: (cues[c][:, L, :] - base[:, L, :]).mean(0) for c in ALL_CUES}
        M = np.stack([mids[c] / (np.linalg.norm(mids[c]) + 1e-8) for c in ALL_CUES])
        S = M @ M.T
        print(f"\n  layer {L}:")
        print("        " + " ".join(f"{c[:4]:>5}" for c in ALL_CUES))
        for i, c in enumerate(ALL_CUES):
            print(f"  {c[:6]:>6} " + " ".join(f"{S[i, j]:>5.2f}" for j in range(len(ALL_CUES))))
        # within vs between class mean cosine
        within, between = [], []
        for i in range(len(ALL_CUES)):
            for j in range(i + 1, len(ALL_CUES)):
                (within if LABEL[ALL_CUES[i]] == LABEL[ALL_CUES[j]] else between).append(S[i, j])
        print(f"  within-class cos={np.mean(within):.2f}  between-class cos={np.mean(between):.2f}")


if __name__ == "__main__":
    main()
