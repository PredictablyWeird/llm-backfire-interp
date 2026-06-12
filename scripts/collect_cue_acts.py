"""Collect Llama-1B last-token residuals for multiple *cue types*, fixed target group.

Goal: probe whether the model compresses many surface cue types into a small set
of latent "influence" features (e.g. SOCIAL PRESSURE vs EVIDENCE).

To isolate the cue *framing* from group *content*, every cue nudges toward the SAME
target group (the stereotyped group) for a given example. The only thing that varies
across the classes is how the influence is phrased.

``user_preference`` and the no-nudge baseline are already in the standard cache
(``..._user_preference_..._acts.npz`` -> nudge_stereo_acts / base_acts), so this script
only collects the remaining cues and writes them to a single npz keyed by cue name.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from mech_interp_bbq.data import load_bbq_full, load_bbq_pairwise
from mech_interp_bbq.hf_backend import capture_activations, load_hf_model
from mech_interp_bbq.nudges import (
    apply_nudge,
    build_few_shot_prefix,
    group_label,
    singular_group_label,
)
from mech_interp_bbq.prompts import model_cache_dir

# Cues to collect here (user_preference + baseline come from the existing cache).
CUES = ["emotional", "role_play", "survey_preference", "weak_evidence", "few_shot"]


def _pair_pool(category: str) -> list[dict]:
    pairs = load_bbq_pairwise(category, context_condition="ambig")
    return [
        {
            "group_a_tag": p.group_a_tag, "group_b_tag": p.group_b_tag,
            "group_a_answer": p.group_a_answer, "group_b_answer": p.group_b_answer,
            "context": p.context, "question": p.question,
        }
        for p in pairs
    ]


def build_cue_prompts(category: str, cue: str, max_examples: int) -> list[str]:
    """Build one prompt per example, nudging toward that example's stereo group."""
    examples = load_bbq_full(category, context_condition="ambig", max_examples=max_examples)
    pool = _pair_pool(category) if cue == "few_shot" else None
    fewshot_cache: dict[str, str] = {}

    prompts: list[str] = []
    for ex in examples:
        stereo_tag = ex.stereotyped_groups[0] if ex.stereotyped_groups else "unknown"
        gl = group_label(category, stereo_tag)
        sg = singular_group_label(category, stereo_tag)
        base = ex.prompt()

        if cue == "role_play":
            # system_replace persona, rendered as a leading instruction line.
            prompts.append(f"You are {sg}.\n\n{base}")
        elif cue == "few_shot":
            if stereo_tag not in fewshot_cache:
                try:
                    fewshot_cache[stereo_tag] = build_few_shot_prefix(pool, stereo_tag, k=3)
                except ValueError:
                    fewshot_cache[stereo_tag] = ""
            prefix = fewshot_cache[stereo_tag]
            prompts.append(f"{prefix}\n\n---\n\n{base}" if prefix else base)
        else:
            prompts.append(
                apply_nudge(
                    base, nudge_type=cue,
                    group_label_str=gl, other_group_label_str="the other group",
                    singular_group_label_str=sg,
                )
            )
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--category", default="Gender_identity")
    ap.add_argument("--max-examples", type=int, default=10_000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cache-dir", default="cache")
    args = ap.parse_args()

    lm = load_hf_model(args.model, dtype="float32", device_map=None)
    # Prefer Apple MPS if available (much faster than CPU for a 1B model).
    if torch.backends.mps.is_available() and lm.device == "cpu":
        lm.model = lm.model.to("mps")
        lm.device = "mps"
    print(f"model on {lm.device} | layers={lm.n_layers} d={lm.d_model}", flush=True)

    out_dir = model_cache_dir(args.cache_dir, args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cue_probe_{args.category}_acts.npz"

    saved: dict[str, np.ndarray] = {}
    for cue in CUES:
        t0 = time.time()
        prompts = build_cue_prompts(args.category, cue, args.max_examples)
        print(f"\n[{cue}] n={len(prompts)}  e.g.: {prompts[0][:120]!r}", flush=True)
        acts = capture_activations(lm, prompts, args.batch_size, capture_components=False)
        saved[cue] = acts["resid"].astype(np.float16)
        print(f"[{cue}] captured {saved[cue].shape} in {time.time() - t0:.0f}s", flush=True)

    np.savez(out_path, **saved)
    print(f"\n[save] {out_path}  ({sum(v.nbytes for v in saved.values()) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
