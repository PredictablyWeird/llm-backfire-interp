"""Nudge dose-response with REASON-BEFORE-ANSWER (GPU phase).

Same ladder (t1..t6) and repetition (k∈{1,2,3,5,8}) axes as ``nudge_sensitivity.py``,
but each prompt ends with a reasoning scaffold instead of ``Answer:``.  The model
generates free-form reasoning, then we append ``Answer:`` and score A/B/C logits.

Output: ``sensitivity_reasoning_<Category>.npz`` with logits + reasoning text arrays.

Lambda usage:
    uv run --env-file .env python scripts/nudge_sensitivity_reasoning.py \\
        --model Qwen/Qwen3-32B --category Gender_identity --batch-size 8 --gen-batch-size 4

Local smoke:
    uv run --env-file .env python scripts/nudge_sensitivity_reasoning.py \\
        --model meta-llama/Llama-3.2-1B --device-map none --max-examples 8 \\
        --batch-size 4 --gen-batch-size 2 --max-reasoning-tokens 64
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from mech_interp_bbq.hf_backend import load_hf_model, reason_then_score_abc
from mech_interp_bbq.prompts import model_cache_dir
from mech_interp_bbq.reasoning import REASONING_INSTRUCTION, append_reasoning_scaffold
from mech_interp_bbq.sensitivity import LADDER, REP_KS, REP_SENT, build_examples, prompts_for


def _scaffolds_for(rows, template: str, direction: str) -> list[str]:
    return [append_reasoning_scaffold(p) for p in prompts_for(rows, template, direction)]


def _to_object_array(nested: list[list[str]]) -> np.ndarray:
    """(n_levels, n) list-of-lists → (n, n_levels) object array."""
    arr = np.array(nested, dtype=object).T
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--category", default="Gender_identity")
    ap.add_argument("--max-examples", type=int, default=10_000)
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size for A/B/C logit scoring")
    ap.add_argument(
        "--gen-batch-size",
        type=int,
        default=4,
        help="Batch size for reasoning generation (keep small — uses KV cache)",
    )
    ap.add_argument("--max-reasoning-tokens", type=int, default=512)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device-map", default="auto", choices=["auto", "none"])
    ap.add_argument("--cache-dir", default="cache")
    args = ap.parse_args()

    rows = build_examples(args.category, args.max_examples)
    n = len(rows)
    print(
        f"{args.category}: n={n} examples  "
        f"({sum(r['has_other'] for r in rows)} with an 'other' label)",
        flush=True,
    )

    device_map = None if args.device_map == "none" else "auto"
    lm = load_hf_model(args.model, dtype=args.dtype, device_map=device_map)
    print(f"model on {lm.device} | layers={lm.n_layers} d={lm.d_model}", flush=True)

    out_dir = model_cache_dir(args.cache_dir, args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sensitivity_reasoning_{args.category}.npz"

    def run(scaffolds: list[str], tag: str) -> tuple[np.ndarray, np.ndarray]:
        t0 = time.time()
        reasonings, logits = reason_then_score_abc(
            lm,
            scaffolds,
            batch_size=args.batch_size,
            gen_batch_size=args.gen_batch_size,
            max_new_tokens=args.max_reasoning_tokens,
        )
        elapsed = time.time() - t0
        mean_len = np.mean([len(r) for r in reasonings])
        print(
            f"  [{tag}] logits={logits.shape} mean_reasoning_chars={mean_len:.0f} "
            f"in {elapsed:.0f}s",
            flush=True,
        )
        return np.array(reasonings, dtype=object), logits

    saved: dict = {}

    base_scaffolds = [append_reasoning_scaffold(r["base"]) for r in rows]
    base_reasoning, base_logits = run(base_scaffolds, "base")
    saved["base_logits"] = base_logits
    saved["reasoning_base"] = base_reasoning

    for direction in ("stereo", "other"):
        ladder_reasonings: list[list[str]] = []
        ladder_logits: list[np.ndarray] = []
        for i, template in enumerate(LADDER):
            reas, lg = run(_scaffolds_for(rows, template, direction), f"ladder/{direction}/t{i+1}")
            ladder_reasonings.append(reas)
            ladder_logits.append(lg)
        saved[f"ladder_{direction}"] = np.stack(ladder_logits, axis=1)
        saved[f"reasoning_ladder_{direction}"] = _to_object_array(ladder_reasonings)

        rep_reasonings: list[list[str]] = []
        rep_logits: list[np.ndarray] = []
        for k in REP_KS:
            template = " ".join([REP_SENT] * k)
            reas, lg = run(_scaffolds_for(rows, template, direction), f"rep/{direction}/k{k}")
            rep_reasonings.append(reas)
            rep_logits.append(lg)
        saved[f"rep_{direction}"] = np.stack(rep_logits, axis=1)
        saved[f"reasoning_rep_{direction}"] = _to_object_array(rep_reasonings)

    saved["stereo_ids"] = np.array([r["stereo_id"] for r in rows], dtype=np.int64)
    saved["unknown_ids"] = np.array([r["unknown_id"] for r in rows], dtype=np.int64)
    saved["other_ids"] = np.array([r["other_id"] for r in rows], dtype=np.int64)
    saved["has_other"] = np.array([r["has_other"] for r in rows], dtype=bool)
    saved["example_ids"] = np.array([r["example_id"] for r in rows], dtype=np.int64)
    saved["ladder_levels"] = np.array(LADDER)
    saved["rep_ks"] = np.array(REP_KS, dtype=np.int64)
    saved["reasoning_instruction"] = np.array(REASONING_INSTRUCTION)
    saved["max_reasoning_tokens"] = np.array(args.max_reasoning_tokens, dtype=np.int64)

    np.savez(out_path, **saved)
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
