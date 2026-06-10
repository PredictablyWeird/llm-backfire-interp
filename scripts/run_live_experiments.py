"""GPU PHASE (live model) — experiments that cannot be reconstructed from caches.

Two experiments:

  1. ``causal_patch``  — run the nudge forward pass but overwrite a component's
     last-token output (mlp / attn / block) with the value it had in the BASE run.
     Layers after the patch re-process the corrected residual, so this is a true
     causal intervention (not the linear approximation done in ``analyze.py``).
     Baseline predictions are untouched, so the eligible pool is unchanged.

  2. ``token_sweep`` — for each backfire example, add the nudge sentence one token
     at a time and record which token first triggers the flip.

Both need the live model and so should run during the GPU session.

Examples:
    uv run --env-file .env python scripts/run_live_experiments.py \
        --model Qwen/Qwen3-32B --category Gender_identity --nudge user_preference \
        --mode causal_patch --component mlp --layers 11 12 13 14 15

    uv run --env-file .env python scripts/run_live_experiments.py \
        --model Qwen/Qwen3-32B --category Gender_identity --nudge user_preference \
        --mode token_sweep
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from mech_interp_bbq.hf_backend import abc_token_ids, load_hf_model
from mech_interp_bbq.prompts import build_prompts, model_cache_dir


def _find_logits(cache_dir, model, category, nudge, condition):
    pat = f"3choice_{model.replace('/', '_')}_{category}_{nudge}_{condition}_n*_logits.npz"
    # Prefer the per-model subdir; fall back to the flat cache dir.
    for d in (model_cache_dir(cache_dir, model), Path(cache_dir)):
        hits = sorted(glob.glob(str(d / pat)))
        if hits:
            return hits[-1]
    return None


def _input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


@torch.inference_mode()
def _predict_batch(lm, prompts, abc, batch_size):
    dev = _input_device(lm.model)
    preds = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = lm.tokenizer(batch, return_tensors="pt", padding=True).to(dev)
        logits = lm.model(**enc).logits[:, -1, abc]
        preds.extend(logits.argmax(-1).cpu().tolist())
    return np.array(preds)


@torch.inference_mode()
def _capture_component_last(lm, prompts, component, layers, batch_size):
    """Capture last-token component output for given layers. Returns {L: (n, d) tensor}."""
    blocks = lm.layers
    dev = _input_device(lm.model)
    store = {L: [] for L in layers}
    scratch: dict = {}

    def mk(L):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            scratch[L] = h[:, -1, :].detach().float().cpu()
        return hook

    handles = []
    for L in layers:
        target = blocks[L] if component == "block" else getattr(blocks[L], "self_attn" if component == "attn" else "mlp")
        handles.append(target.register_forward_hook(mk(L)))
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            enc = lm.tokenizer(batch, return_tensors="pt", padding=True).to(dev)
            scratch.clear()
            lm.model(**enc)
            for L in layers:
                store[L].append(scratch[L])
    finally:
        for h in handles:
            h.remove()
    return {L: torch.cat(store[L], dim=0) for L in layers}


@torch.inference_mode()
def _predict_patched(lm, prompts, component, layer, base_vals, abc, batch_size):
    """Predict with last-token output of `component`@`layer` overwritten by base_vals."""
    blocks = lm.layers
    dev = _input_device(lm.model)
    target = blocks[layer] if component == "block" else getattr(blocks[layer], "self_attn" if component == "attn" else "mlp")
    preds = []
    state: dict = {}

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        bv = state["bv"].to(h.dtype).to(h.device)
        h[:, -1, :] = bv
        if isinstance(out, tuple):
            return (h, *out[1:])
        return h

    handle = target.register_forward_hook(hook)
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            enc = lm.tokenizer(batch, return_tensors="pt", padding=True).to(dev)
            state["bv"] = base_vals[start : start + len(batch)]
            logits = lm.model(**enc).logits[:, -1, abc]
            preds.extend(logits.argmax(-1).cpu().tolist())
    finally:
        handle.remove()
    return np.array(preds)


def causal_patch(args, lm, bundle, abc):
    ld = np.load(_find_logits(args.cache_dir, args.model, args.category, args.nudge, args.condition))
    s, o = ld["stereo_ids"].astype(int), ld["other_ids"].astype(int)
    bp = ld["base_logits"].argmax(1)
    full_bf_s = int(((bp == s) & (ld["nudged_logits"].argmax(1) == o)).sum())
    full_bf_o = int(((bp == o) & (ld["nudged_other_logits"].argmax(1) == s)).sum())

    layers = args.layers
    print(f"Capturing base {args.component} outputs for layers {layers} ...", flush=True)
    base_vals = _capture_component_last(lm, bundle.baseline_prompts, args.component, layers, args.batch_size)

    rows = []
    print(f"\nFull: bf_s={full_bf_s} bf_o={full_bf_o} total={full_bf_s + full_bf_o}")
    print(f"{'layer':>6} {'bf_s':>6} {'bf_o':>6} {'total':>6} {'Δ':>6}")
    for L in layers:
        sp = _predict_patched(lm, bundle.nudge_stereo_prompts, args.component, L, base_vals[L], abc, args.batch_size)
        op = _predict_patched(lm, bundle.nudge_other_prompts, args.component, L, base_vals[L], abc, args.batch_size)
        bf_s = int(((bp == s) & (sp == o)).sum())
        bf_o = int(((bp == o) & (op == s)).sum())
        tot = bf_s + bf_o
        d = tot - (full_bf_s + full_bf_o)
        rows.append({"layer": int(L), "bf_s": bf_s, "bf_o": bf_o, "total": tot, "delta_vs_full": d})
        print(f"{L:>6} {bf_s:>6} {bf_o:>6} {tot:>6} {d:>+6}")

    out = {"mode": "causal_patch", "component": args.component, "model": args.model,
           "category": args.category, "nudge": args.nudge,
           "full": {"bf_s": full_bf_s, "bf_o": full_bf_o}, "rows": rows}
    _save(args, out, f"causalpatch_{args.component}")


def token_sweep(args, lm, bundle, abc):
    ld = np.load(_find_logits(args.cache_dir, args.model, args.category, args.nudge, args.condition))
    s, o = ld["stereo_ids"].astype(int), ld["other_ids"].astype(int)
    bp, sp_full, op_full = ld["base_logits"].argmax(1), ld["nudged_logits"].argmax(1), ld["nudged_other_logits"].argmax(1)
    bf_s_idx = np.where((bp == s) & (sp_full == o))[0]
    bf_o_idx = np.where((bp == o) & (op_full == s))[0]
    tok = lm.tokenizer

    def find_inserted(base_p, nudge_p):
        bt = tok(base_p, add_special_tokens=False)["input_ids"]
        nt = tok(nudge_p, add_special_tokens=False)["input_ids"]
        plen = 0
        for b, nn in zip(bt, nt, strict=False):
            if b != nn:
                break
            plen += 1
        suffix = bt[plen:]
        ins = nt[plen : len(nt) - len(suffix)]
        return tok.decode(bt[:plen]), ins, tok.decode(suffix)

    def sweep(indices, prompts, desc):
        ks, toks = [], []
        for cnt, idx in enumerate(indices):
            base_p = bundle.baseline_prompts[idx]
            nudge_p = prompts[idx]
            prefix, ins, suffix = find_inserted(base_p, nudge_p)
            base_pred = _predict_batch(lm, [base_p], abc, 1)[0]
            flip_k = -1
            flip_tok = None
            partials = [prefix + tok.decode(ins[:k]) + suffix for k in range(1, len(ins) + 1)]
            preds = _predict_batch(lm, partials, abc, args.batch_size)
            for k, pr in enumerate(preds, 1):
                if pr != base_pred:
                    flip_k, flip_tok = k, tok.decode([ins[k - 1]])
                    break
            ks.append(flip_k)
            toks.append(flip_tok)
            if (cnt + 1) % 10 == 0:
                print(f"  {cnt + 1}/{len(indices)} {desc}", flush=True)
        return ks, toks

    print(f"bf_s={len(bf_s_idx)} bf_o={len(bf_o_idx)}")
    bfs_k, bfs_t = sweep(bf_s_idx, bundle.nudge_stereo_prompts, "bf_s")
    bfo_k, bfo_t = sweep(bf_o_idx, bundle.nudge_other_prompts, "bf_o")

    def summarize(ks, toks):
        flipped = [k for k in ks if k != -1]
        return {
            "n": len(ks), "never_flipped": int(sum(1 for k in ks if k == -1)),
            "flip_at_k1": float(np.mean([k == 1 for k in flipped])) if flipped else None,
            "flip_at_k_le2": float(np.mean([k <= 2 for k in flipped])) if flipped else None,
            "position_hist": dict(Counter(flipped)),
            "top_tokens": Counter(t for t in toks if t).most_common(10),
        }

    out = {"mode": "token_sweep", "model": args.model, "category": args.category,
           "nudge": args.nudge,
           "bf_from_stereo": summarize(bfs_k, bfs_t),
           "bf_from_other": summarize(bfo_k, bfo_t)}
    print(json.dumps(out, indent=2, default=str))
    _save(args, out, "tokensweep")


def _save(args, payload, tag):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{tag}_{args.model.replace('/', '_')}_{args.category}_{args.nudge}_{args.condition}"
    path = out_dir / f"{stem}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--category", default="Gender_identity")
    p.add_argument("--nudge", default="user_preference")
    p.add_argument("--condition", default="ambig", choices=["ambig", "disambig", "both"])
    p.add_argument("--mode", required=True, choices=["causal_patch", "token_sweep"])
    p.add_argument("--component", default="mlp", choices=["mlp", "attn", "block"])
    p.add_argument("--layers", type=int, nargs="+", default=[11, 12, 13, 14, 15])
    p.add_argument("--max-examples", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device-map", default="auto", choices=["auto", "none"])
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    device_map = None if args.device_map == "none" else "auto"
    print(f"Loading model {args.model} ...", flush=True)
    lm = load_hf_model(args.model, dtype=args.dtype, device_map=device_map)
    abc = abc_token_ids(lm.tokenizer)
    bundle = build_prompts(args.category, args.nudge, args.condition, args.max_examples)

    if args.mode == "causal_patch":
        causal_patch(args, lm, bundle, abc)
    else:
        token_sweep(args, lm, bundle, abc)


if __name__ == "__main__":
    main()
