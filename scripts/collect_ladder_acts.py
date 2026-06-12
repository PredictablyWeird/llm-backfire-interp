"""GPU phase — collect per-layer activations for the assertiveness ladder (Gender_identity).

Pairs with ``nudge_sensitivity.py`` logits cache. Stores:
  cache/<model>/ladder_acts_<Category>.npz
    base_acts          (n, n_layers, d_model)
    ladder_stereo_acts (n, 6, n_layers, d_model)   t1..t6 toward stereotyped group

Optional (``--capture-t3-components``):
    t3_stereo_mlp / t3_stereo_attn  (n, n_layers, d_model) at ladder t3 only

Lambda:
    nohup /home/ubuntu/.local/bin/uv run --env-file .env python scripts/collect_ladder_acts.py \
        --model Qwen/Qwen3-32B --batch-size 16 > ladder_acts.log 2>&1 &
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from mech_interp_bbq.hf_backend import capture_activations, load_hf_model
from mech_interp_bbq.prompts import model_cache_dir

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nudge_sensitivity import LADDER, _prompts_for, build_examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--category", default="Gender_identity")
    ap.add_argument("--max-examples", type=int, default=10_000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device-map", default="auto", choices=["auto", "none"])
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--capture-t3-components", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = model_cache_dir(args.cache_dir, args.model)
    out_path = out_dir / f"ladder_acts_{args.category}.npz"
    if out_path.exists() and not args.force:
        print(f"[skip] {out_path} exists")
        return

    rows = build_examples(args.category, args.max_examples)
    n = len(rows)
    print(f"{args.category}: n={n}", flush=True)

    device_map = None if args.device_map == "none" else "auto"
    lm = load_hf_model(args.model, dtype=args.dtype, device_map=device_map)
    print(f"model on {lm.device} | layers={lm.n_layers} d={lm.d_model}", flush=True)

    def run(prompts, tag, components=False):
        t0 = time.time()
        out = capture_activations(lm, prompts, args.batch_size, capture_components=components)
        resid = out["resid"]
        print(f"  [{tag}] {resid.shape} in {time.time() - t0:.0f}s", flush=True)
        return out

    saved: dict[str, np.ndarray] = {}
    saved["base_acts"] = run([r["base"] for r in rows], "base")["resid"]

    ladder_chunks = []
    for i, template in enumerate(LADDER):
        acts = run(_prompts_for(rows, template, "stereo"), f"ladder/stereo/t{i+1}")["resid"]
        ladder_chunks.append(acts)
    saved["ladder_stereo_acts"] = np.stack(ladder_chunks, axis=1)

    if args.capture_t3_components:
        t3_prompts = _prompts_for(rows, LADDER[2], "stereo")
        comp = run(t3_prompts, "ladder/stereo/t3/components", components=True)
        saved["t3_stereo_mlp"] = comp["mlp"]
        saved["t3_stereo_attn"] = comp["attn"]
        base_comp = run([r["base"] for r in rows], "base/components", components=True)
        saved["base_mlp"] = base_comp["mlp"]
        saved["base_attn"] = base_comp["attn"]

    saved["n_examples"] = np.array(n, dtype=np.int64)
    np.savez(out_path, **saved)
    print(f"\n[save] {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
