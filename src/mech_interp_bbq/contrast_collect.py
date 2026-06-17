"""Shared GPU collection helpers for binary and 3-way contrast-probe experiments."""

from __future__ import annotations

import argparse
import time

import numpy as np

from mech_interp_bbq.contrast_probe import (
    ProbeMode,
    build_direct_prompt,
    cache_stem,
    contrast_pair_for_row,
    contrast_suffix,
    letter_for_id,
    threeway_pairs_for_row,
)
from mech_interp_bbq.hf_backend import (
    capture_activations,
    compute_abc_logits,
    generate_continuations,
    load_hf_model,
)
from mech_interp_bbq.prompts import model_cache_dir
from mech_interp_bbq.reasoning import append_reasoning_scaffold, finalize_after_reasoning, truncate_reasoning
from mech_interp_bbq.sensitivity import LADDER, build_examples, prompts_for


def add_shared_collect_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--category", default="Gender_identity")
    ap.add_argument("--max-examples", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device-map", default="auto", choices=["auto", "none"])
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--all-layers", action="store_true")
    ap.add_argument("--with-nudge", action="store_true")
    ap.add_argument("--reason-before-answer", action="store_true")
    ap.add_argument("--with-reasoning-instruction", action="store_true")
    ap.add_argument("--max-reasoning-tokens", type=int, default=128)
    ap.add_argument("--gen-batch-size", type=int, default=8)
    ap.add_argument("--force", action="store_true")


def stimulus_conditions(rows, *, with_nudge: bool) -> list[tuple[str, list[str]]]:
    if with_nudge:
        return [(f"t{i + 1}", prompts_for(rows, template, "stereo")) for i, template in enumerate(LADDER)]
    return [("base", [r["base"] for r in rows])]


def _stimulus_prefix(stimulus: str, reasoning: str | None) -> tuple[str, str | None]:
    """Return (contrast_prefix, direct_prompt). direct_prompt is None if reasoning path used."""
    if reasoning is None:
        return stimulus, build_direct_prompt(stimulus)
    scaffold = append_reasoning_scaffold(stimulus)
    return scaffold + truncate_reasoning(reasoning), None


def build_binary_prompts(
    rows,
    stimuli: list[str],
    reasonings: list[str] | None,
    *,
    suffix: str,
) -> tuple[list[str], list[str], list[str]]:
    plus_prompts: list[str] = []
    minus_prompts: list[str] = []
    direct_prompts: list[str] = []

    for row, stimulus, reasoning in zip(
        rows,
        stimuli,
        reasonings if reasonings is not None else [None] * len(stimuli),
        strict=True,
    ):
        prefix, direct = _stimulus_prefix(stimulus, reasoning)
        p, m = contrast_pair_for_row(prefix, row["stereo_id"], row["unknown_id"], suffix=suffix)
        plus_prompts.append(p)
        minus_prompts.append(m)
        if direct is not None:
            direct_prompts.append(direct)
        else:
            direct_prompts.append(finalize_after_reasoning(append_reasoning_scaffold(stimulus), reasoning))

    return plus_prompts, minus_prompts, direct_prompts


def build_threeway_prompts(
    rows,
    stimuli: list[str],
    reasonings: list[str] | None,
    *,
    suffix: str,
) -> tuple[list[list[str]], list[list[str]], list[str]]:
    """Returns plus/minus as [pair_idx][example], direct as flat list."""
    n_pairs = 3
    plus_by_pair: list[list[str]] = [[] for _ in range(n_pairs)]
    minus_by_pair: list[list[str]] = [[] for _ in range(n_pairs)]
    direct_prompts: list[str] = []

    for row, stimulus, reasoning in zip(
        rows,
        stimuli,
        reasonings if reasonings is not None else [None] * len(stimuli),
        strict=True,
    ):
        prefix, direct = _stimulus_prefix(stimulus, reasoning)
        for i, (_name, p, m) in enumerate(threeway_pairs_for_row(row, prefix, suffix=suffix)):
            plus_by_pair[i].append(p)
            minus_by_pair[i].append(m)
        if direct is not None:
            direct_prompts.append(direct)
        else:
            direct_prompts.append(finalize_after_reasoning(append_reasoning_scaffold(stimulus), reasoning))

    return plus_by_pair, minus_by_pair, direct_prompts


def run_collect(args: argparse.Namespace, mode: ProbeMode) -> None:
    suffix = contrast_suffix(with_reasoning_instruction=args.with_reasoning_instruction)
    out_dir = model_cache_dir(args.cache_dir, args.model)
    out_path = out_dir / f"{cache_stem(args.category, mode=mode, reason_before_answer=args.reason_before_answer, with_nudge=args.with_nudge)}.npz"
    if out_path.exists() and not args.force:
        print(f"[skip] {out_path} exists")
        return

    rows = build_examples(args.category, args.max_examples)
    n = len(rows)
    conditions = stimulus_conditions(rows, with_nudge=args.with_nudge)
    print(
        f"[{mode}] {args.category}: n={n}  conditions={len(conditions)}  "
        f"layers={'all' if args.all_layers else 'last'}  "
        f"with_nudge={args.with_nudge}  reason_before_answer={args.reason_before_answer}",
        flush=True,
    )

    device_map = None if args.device_map == "none" else "auto"
    lm = load_hf_model(args.model, dtype=args.dtype, device_map=device_map)
    print(f"model on {lm.device} | layers={lm.n_layers} d={lm.d_model}", flush=True)

    def run_acts(prompts: list[str], tag: str) -> np.ndarray:
        t0 = time.time()
        out = capture_activations(lm, prompts, args.batch_size)
        resid = out["resid"]
        if not args.all_layers:
            resid = resid[:, -1, :]
        print(f"  [{tag}] {resid.shape} in {time.time() - t0:.0f}s", flush=True)
        return resid

    def run_logits(prompts: list[str], tag: str) -> np.ndarray:
        t0 = time.time()
        lg = compute_abc_logits(lm, prompts, args.batch_size)
        print(f"  [{tag}] {lg.shape} in {time.time() - t0:.0f}s", flush=True)
        return lg

    direct_chunks: list[np.ndarray] = []
    reasoning_chunks: list[np.ndarray] | None = [] if args.reason_before_answer else None
    condition_tags: list[str] = []

    if mode == "binary":
        plus_chunks: list[np.ndarray] = []
        minus_chunks: list[np.ndarray] = []
    else:
        plus_chunks = [[] for _ in range(3)]
        minus_chunks = [[] for _ in range(3)]

    for tag, stimuli in conditions:
        condition_tags.append(tag)
        reasonings: list[str] | None = None
        if args.reason_before_answer:
            scaffolds = [append_reasoning_scaffold(p) for p in stimuli]
            t0 = time.time()
            reasonings = generate_continuations(
                lm,
                scaffolds,
                max_new_tokens=args.max_reasoning_tokens,
                batch_size=args.gen_batch_size,
            )
            print(
                f"  [reason/{tag}] mean_chars={np.mean([len(r) for r in reasonings]):.0f} "
                f"in {time.time() - t0:.0f}s",
                flush=True,
            )
            assert reasoning_chunks is not None
            reasoning_chunks.append(np.array(reasonings, dtype=object))

        if mode == "binary":
            plus_p, minus_p, direct_p = build_binary_prompts(rows, stimuli, reasonings, suffix=suffix)
            plus_chunks.append(run_acts(plus_p, f"plus/{tag}"))
            minus_chunks.append(run_acts(minus_p, f"minus/{tag}"))
        else:
            plus_by_pair, minus_by_pair, direct_p = build_threeway_prompts(
                rows, stimuli, reasonings, suffix=suffix
            )
            for i, pair_name in enumerate(("su", "so", "uo")):
                plus_chunks[i].append(run_acts(plus_by_pair[i], f"plus/{pair_name}/{tag}"))
                minus_chunks[i].append(run_acts(minus_by_pair[i], f"minus/{pair_name}/{tag}"))

        direct_chunks.append(run_logits(direct_p, f"direct/{tag}"))

    saved: dict[str, np.ndarray] = {
        "probe_mode": np.array(mode),
        "direct_logits": np.stack(direct_chunks, axis=1),
        "stereo_ids": np.array([r["stereo_id"] for r in rows], dtype=np.int64),
        "unknown_ids": np.array([r["unknown_id"] for r in rows], dtype=np.int64),
        "other_ids": np.array([r["other_id"] for r in rows], dtype=np.int64),
        "example_ids": np.array([r["example_id"] for r in rows], dtype=np.int64),
        "condition_tags": np.array(condition_tags, dtype=object),
        "ladder_levels": np.array(LADDER if args.with_nudge else [], dtype=object),
        "contrast_suffix": np.array(suffix),
        "all_layers": np.array(args.all_layers),
        "with_nudge": np.array(args.with_nudge),
        "reason_before_answer": np.array(args.reason_before_answer),
        "with_reasoning_instruction": np.array(args.with_reasoning_instruction),
        "n_examples": np.array(n, dtype=np.int64),
    }

    if mode == "binary":
        saved["phi_plus"] = np.stack(plus_chunks, axis=1)
        saved["phi_minus"] = np.stack(minus_chunks, axis=1)
    else:
        saved["pair_names"] = np.array(["su", "so", "uo"], dtype=object)
        saved["phi_plus"] = np.stack([np.stack(ch, axis=1) for ch in plus_chunks], axis=2)
        saved["phi_minus"] = np.stack([np.stack(ch, axis=1) for ch in minus_chunks], axis=2)

    if args.reason_before_answer:
        saved["reasoning"] = np.stack(reasoning_chunks, axis=1)
        saved["max_reasoning_tokens"] = np.array(args.max_reasoning_tokens, dtype=np.int64)

    np.savez(out_path, **saved)
    print(f"\n[save] {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
