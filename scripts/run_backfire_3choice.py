"""Backfire analysis on the original 3-choice BBQ — bidirectional.

Strict backfire (both directions):
  Case 1 — stereo-direction:
        baseline = stereo answer (X)
        nudge → stereo (X)
        result = other non-stereo answer (Y)   [NOT unknown]

  Case 2 — other-direction:
        baseline = other non-stereo answer (Y)
        nudge → other (Y)
        result = stereo answer (X)             [NOT unknown]

  "Fled to unknown" (either direction):
        eligible base (X or Y), nudge → that base,
        result = cannot-be-determined          [tracked separately, NOT backfire]

  "Stayed" (control, either direction):
        eligible base, nudge → base, result = same base

The "other group" nudge requires the non-stereo group's tag.  We derive it by
matching each BBQFullExample to its BBQPairExample (same example_id), then
checking which pair group tag corresponds to the other answer slot.

Caching
-------
  cache/<slug>_logits.npz  — base + nudge-stereo + nudge-other logits
  cache/<slug>_acts.npz    — base + nudge-stereo + nudge-other residual acts
                             (only written if enough backfire examples exist)

  If the cache exists but is missing nudged_other_* (e.g. from a prior single-
  direction run), only the missing forward pass is re-run and the cache is
  updated.  Use --force-recompute to ignore the cache entirely.

Usage:
    uv run --env-file .env python scripts/run_backfire_3choice.py \\
        --model meta-llama/Llama-3.2-1B \\
        --category Gender_identity \\
        --nudge user_preference \\
        --max-examples 10000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from mech_interp_bbq.activations import collect_resid_post, load_model
from mech_interp_bbq.data import HITZ_CATEGORIES, load_bbq_full, load_bbq_pairwise
from mech_interp_bbq.nudges import NUDGE_TEMPLATES, apply_nudge, group_label, singular_group_label
from mech_interp_bbq.probes import train_layer_probe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--category", default="Gender_identity", choices=list(HITZ_CATEGORIES))
    p.add_argument("--nudge", default="user_preference")
    p.add_argument("--max-examples", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--context-condition", default="ambig",
                   choices=["ambig", "disambig", "both"])
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--force-recompute", action="store_true")
    p.add_argument("--out-dir", default="probes_out")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def cache_slug(args: argparse.Namespace, n: int) -> str:
    return (f"3choice_{args.model.replace('/', '_')}"
            f"_{args.category}_{args.nudge}_{args.context_condition}_n{n}")


@torch.inference_mode()
def collect_3choice_logits(model, prompts: list[str], batch_size: int = 4) -> np.ndarray:
    """Raw logits over A/B/C at the last token position. Shape: (n, 3)."""
    choice_token_ids = [
        int(model.to_tokens(f" {l}", prepend_bos=False)[0][-1])
        for l in ["A", "B", "C"]
    ]
    chunks = []
    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start : start + batch_size])
        tokens = model.to_tokens(batch, prepend_bos=True)
        logits = model(tokens, return_type="logits")[:, -1, :]
        chunks.append(logits[:, choice_token_ids].cpu().float())
    return torch.cat(chunks, dim=0).numpy()


def _ensure_model(model_name: str, _cache: dict) -> object:
    """Load model once and cache in the provided dict."""
    if "model" not in _cache:
        print(f"\nLoading model: {model_name}")
        _cache["model"] = load_model(model_name)
    return _cache["model"]


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_cache: dict = {}

    # ── 1. load full examples ─────────────────────────────────────────────────
    cond_arg = None if args.context_condition == "both" else args.context_condition
    print(f"Loading BBQ full (3-choice) | category={args.category} | condition={args.context_condition}")
    examples = load_bbq_full(args.category, context_condition=cond_arg,
                             max_examples=args.max_examples)
    n = len(examples)
    print(f"  -> {n} examples")

    # ── 2. load pairwise examples to get other-group tags ─────────────────────
    # BBQFullExample knows the stereo group tag (stereotyped_groups[0]) but not
    # the OTHER group's tag.  BBQPairExample stores both as group_a_tag /
    # group_b_tag.  We match by example_id and identify which tag belongs to the
    # "other" (non-stereo, non-unknown) answer slot.
    print("Loading pairwise examples for group-tag lookup...")
    pair_examples = load_bbq_pairwise(args.category, context_condition=cond_arg)
    pair_by_id = {ex.example_id: ex for ex in pair_examples}

    # ── 3. build per-example prompts + metadata ───────────────────────────────
    baseline_prompts:    list[str] = []
    nudge_stereo_prompts: list[str] = []
    nudge_other_prompts:  list[str] = []
    stereo_ids:  list[int] = []
    unknown_ids: list[int] = []
    has_other_tag: list[bool] = []  # False if pairwise match failed

    tmpl = NUDGE_TEMPLATES[args.nudge]

    for ex in examples:
        stereo_tag = ex.stereotyped_groups[0] if ex.stereotyped_groups else "unknown"
        stereo_gl  = group_label(args.category, stereo_tag)
        stereo_sg  = singular_group_label(args.category, stereo_tag)

        base_prompt = ex.prompt()
        baseline_prompts.append(base_prompt)
        stereo_ids.append(ex.stereotyped_ans_id)
        unknown_ids.append(ex.unknown_ans_id)

        nudge_stereo_prompts.append(apply_nudge(
            base_prompt, nudge_type=args.nudge,
            group_label_str=stereo_gl,
            other_group_label_str="the other group",
            singular_group_label_str=stereo_sg,
        ))

        # Derive other-group tag from the matching pairwise example
        other_idx = ({0, 1, 2} - {ex.stereotyped_ans_id, ex.unknown_ans_id}).pop()
        other_ans_text = ex.answers[other_idx]
        pair = pair_by_id.get(ex.example_id)
        other_tag = None
        if pair is not None:
            if pair.group_a_answer.strip() == other_ans_text.strip():
                other_tag = pair.group_a_tag
            elif pair.group_b_answer.strip() == other_ans_text.strip():
                other_tag = pair.group_b_tag

        if other_tag is not None:
            other_gl = group_label(args.category, other_tag)
            other_sg = singular_group_label(args.category, other_tag)
            nudge_other_prompts.append(apply_nudge(
                base_prompt, nudge_type=args.nudge,
                group_label_str=other_gl,
                other_group_label_str=stereo_gl,
                singular_group_label_str=other_sg,
            ))
            has_other_tag.append(True)
        else:
            nudge_other_prompts.append(base_prompt)  # no-op fallback
            has_other_tag.append(False)

    stereo_ids_arr  = np.array(stereo_ids,  dtype=np.int64)
    unknown_ids_arr = np.array(unknown_ids, dtype=np.int64)
    other_ids_arr   = np.array(
        [({0, 1, 2} - {int(stereo_ids_arr[i]), int(unknown_ids_arr[i])}).pop()
         for i in range(n)],
        dtype=np.int64,
    )
    has_other_tag_arr = np.array(has_other_tag, dtype=bool)
    n_missing_tag = int((~has_other_tag_arr).sum())
    if n_missing_tag:
        print(f"  Warning: {n_missing_tag} examples had no pairwise tag match "
              f"— excluded from other-direction analysis")

    slug = cache_slug(args, n)
    logits_cache_path = cache_dir / f"{slug}_logits.npz"
    acts_cache_path   = cache_dir / f"{slug}_acts.npz"

    # ── 4. logits: load or compute ────────────────────────────────────────────
    need_base    = True
    need_stereo  = True
    need_other   = True

    if logits_cache_path.exists() and not args.force_recompute:
        print(f"\nLoading cached logits from {logits_cache_path}")
        cached = np.load(logits_cache_path)
        base_logits          = cached["base_logits"]
        nudge_stereo_logits  = cached["nudged_logits"]         # backward-compat key
        need_base = need_stereo = False
        if "nudged_other_logits" in cached:
            nudge_other_logits = cached["nudged_other_logits"]
            need_other = False
        else:
            print("  Cache missing nudged_other_logits — collecting now...")

    if need_base or need_stereo:
        model = _ensure_model(args.model, model_cache)
        if need_base:
            print("Collecting baseline logits (A/B/C)...")
            base_logits = collect_3choice_logits(model, baseline_prompts, args.batch_size)
        if need_stereo:
            print("Collecting nudge-toward-stereo logits...")
            nudge_stereo_logits = collect_3choice_logits(
                model, nudge_stereo_prompts, args.batch_size)

    if need_other:
        model = _ensure_model(args.model, model_cache)
        print("Collecting nudge-toward-other logits...")
        nudge_other_logits = collect_3choice_logits(
            model, nudge_other_prompts, args.batch_size)

    if need_base or need_stereo or need_other:
        np.savez_compressed(
            logits_cache_path,
            base_logits=base_logits,
            nudged_logits=nudge_stereo_logits,
            nudged_other_logits=nudge_other_logits,
            stereo_ids=stereo_ids_arr,
            unknown_ids=unknown_ids_arr,
            other_ids=other_ids_arr,
        )
        print(f"  Cached logits -> {logits_cache_path}")

    base_preds         = base_logits.argmax(axis=1)
    nudge_stereo_preds = nudge_stereo_logits.argmax(axis=1)
    nudge_other_preds  = nudge_other_logits.argmax(axis=1)

    # ── 5. bidirectional strict backfire classification ───────────────────────
    #
    # For each example, the "relevant nudge" is the one that matches the
    # baseline choice:
    #   base = stereo → test nudge-toward-stereo
    #   base = other  → test nudge-toward-other  (only if has_other_tag)
    #   base = unknown → not eligible
    #
    # Strict backfire: nudge caused a flip to the OPPOSITE real group (not unknown)
    # Fled to unknown: nudge caused a flip to 'cannot be determined'
    # Stayed: nudge did not change the prediction

    elig_stereo = base_preds == stereo_ids_arr
    elig_other  = (base_preds == other_ids_arr) & has_other_tag_arr

    # stereo-direction outcomes
    bf_from_stereo    = elig_stereo & (nudge_stereo_preds == other_ids_arr)
    fled_from_stereo  = elig_stereo & (nudge_stereo_preds == unknown_ids_arr)
    stayed_stereo     = elig_stereo & (nudge_stereo_preds == stereo_ids_arr)

    # other-direction outcomes
    bf_from_other     = elig_other  & (nudge_other_preds  == stereo_ids_arr)
    fled_from_other   = elig_other  & (nudge_other_preds  == unknown_ids_arr)
    stayed_other      = elig_other  & (nudge_other_preds  == other_ids_arr)

    # combined
    all_bf    = bf_from_stereo    | bf_from_other
    all_stayed = stayed_stereo    | stayed_other
    all_elig  = elig_stereo       | elig_other

    n_elig_s  = int(elig_stereo.sum())
    n_elig_o  = int(elig_other.sum())
    n_bf_s    = int(bf_from_stereo.sum())
    n_bf_o    = int(bf_from_other.sum())
    n_fled_s  = int(fled_from_stereo.sum())
    n_fled_o  = int(fled_from_other.sum())
    n_stay_s  = int(stayed_stereo.sum())
    n_stay_o  = int(stayed_other.sum())
    n_bf_tot  = int(all_bf.sum())
    n_stay_tot = int(all_stayed.sum())
    n_elig_tot = int(all_elig.sum())

    print(f"\n=== Bidirectional strict backfire | 3-choice BBQ | nudge={args.nudge!r} ===")
    print(f"  total examples                           : {n}")
    print(f"")
    print(f"  ── Stereo direction (base=stereo, nudge→stereo) ──")
    print(f"  eligible                                 : {n_elig_s}  ({100*n_elig_s/n:.1f}%)")
    print(f"  strict backfire  (→ other group Y)       : {n_bf_s}  ({100*n_bf_s/max(n_elig_s,1):.1f}% of eligible)")
    print(f"  fled to unknown  (→ cannot determine)    : {n_fled_s}  ({100*n_fled_s/max(n_elig_s,1):.1f}% of eligible)")
    print(f"  stayed-at-stereo (→ X)                   : {n_stay_s}  ({100*n_stay_s/max(n_elig_s,1):.1f}% of eligible)")
    print(f"")
    print(f"  ── Other direction (base=other, nudge→other) ──")
    print(f"  eligible                                 : {n_elig_o}  ({100*n_elig_o/n:.1f}%)")
    print(f"  strict backfire  (→ stereo group X)      : {n_bf_o}  ({100*n_bf_o/max(n_elig_o,1):.1f}% of eligible)")
    print(f"  fled to unknown  (→ cannot determine)    : {n_fled_o}  ({100*n_fled_o/max(n_elig_o,1):.1f}% of eligible)")
    print(f"  stayed-at-other  (→ Y)                   : {n_stay_o}  ({100*n_stay_o/max(n_elig_o,1):.1f}% of eligible)")
    print(f"")
    print(f"  ── Combined ──")
    print(f"  total eligible                           : {n_elig_tot}  ({100*n_elig_tot/n:.1f}%)")
    print(f"  total strict backfire                    : {n_bf_tot}  ({100*n_bf_tot/max(n_elig_tot,1):.1f}% of eligible)")
    print(f"  total stayed                             : {n_stay_tot}  ({100*n_stay_tot/max(n_elig_tot,1):.1f}% of eligible)")

    # ── 6. show example backfires ─────────────────────────────────────────────
    label_map = ["A", "B", "C"]

    def show_examples(mask: np.ndarray, preds_after: np.ndarray, label: str, k: int = 3) -> None:
        idx = np.where(mask)[0][:k]
        if not len(idx):
            return
        print(f"\n── {label} (showing up to {k}) ──")
        for i in idx:
            ex = examples[i]
            before = label_map[int(base_preds[i])]
            after  = label_map[int(preds_after[i])]
            print(f"\n[BACKFIRE] {before}→{after}")
            print(f"  context  : {ex.context}")
            print(f"  question : {ex.question}")
            for j, ans in enumerate(ex.answers):
                tag = ""
                if j == ex.stereotyped_ans_id: tag = "  ← STEREO"
                elif j == ex.unknown_ans_id:   tag = "  ← UNKNOWN"
                print(f"  {label_map[j]}. {ans}{tag}")

    print(f"\n{'='*65}")
    show_examples(bf_from_stereo, nudge_stereo_preds,
                  "Backfire: base=stereo → nudge→stereo → other group")
    show_examples(bf_from_other,  nudge_other_preds,
                  "Backfire: base=other  → nudge→other  → stereo group")

    # ── 7. activation analysis ────────────────────────────────────────────────
    if n_bf_tot < 3 or n_stay_tot < 3:
        print(f"\nToo few examples for activation analysis "
              f"(backfire={n_bf_tot}, stayed={n_stay_tot}).")
        probe_results = base_acts = nudged_acts_combined = None
        majority = nudged_accs = base_accs = cos_l = l2_bf_l = l2_sm_l = None
        n_layers = None
    else:
        # For each eligible example, pick the "relevant" nudged activations:
        #   stereo-eligible → nudge-stereo acts
        #   other-eligible  → nudge-other acts
        need_base_acts        = True
        need_stereo_acts      = True
        need_other_acts       = True

        if acts_cache_path.exists() and not args.force_recompute:
            print(f"\nLoading cached activations from {acts_cache_path}")
            cached_acts = np.load(acts_cache_path)
            base_acts = cached_acts["base_acts"]
            # backward-compat: old cache used "nudged_acts", new cache uses "nudge_stereo_acts"
            if "nudge_stereo_acts" in cached_acts:
                nudge_stereo_acts = cached_acts["nudge_stereo_acts"]
            elif "nudged_acts" in cached_acts:
                nudge_stereo_acts = cached_acts["nudged_acts"]
            else:
                nudge_stereo_acts = None  # will recompute
            need_base_acts = False
            need_stereo_acts = nudge_stereo_acts is None
            if "nudge_other_acts" in cached_acts:
                nudge_other_acts = cached_acts["nudge_other_acts"]
                need_other_acts  = False
            else:
                print("  Cache missing nudge_other_acts — collecting now...")

        if need_base_acts or need_stereo_acts:
            model = _ensure_model(args.model, model_cache)
            if need_base_acts:
                print("Collecting baseline activations...")
                base_acts = collect_resid_post(
                    model, baseline_prompts, batch_size=args.batch_size
                ).acts.numpy()
            if need_stereo_acts:
                print("Collecting nudge-toward-stereo activations...")
                nudge_stereo_acts = collect_resid_post(
                    model, nudge_stereo_prompts, batch_size=args.batch_size
                ).acts.numpy()

        if need_other_acts:
            model = _ensure_model(args.model, model_cache)
            print("Collecting nudge-toward-other activations...")
            nudge_other_acts = collect_resid_post(
                model, nudge_other_prompts, batch_size=args.batch_size
            ).acts.numpy()

        if need_base_acts or need_stereo_acts or need_other_acts:
            np.savez_compressed(
                acts_cache_path,
                base_acts=base_acts,
                nudge_stereo_acts=nudge_stereo_acts,
                nudge_other_acts=nudge_other_acts,
            )
            print(f"  Cached activations -> {acts_cache_path}")

        # Build per-example "relevant nudge acts" array
        n_layers = base_acts.shape[1]
        nudged_acts_combined = np.where(
            elig_stereo[:, None, None],
            nudge_stereo_acts,
            nudge_other_acts,
        )  # (n, L, D): uses stereo acts for stereo-eligible, other acts for other-eligible

        delta = nudged_acts_combined - base_acts

        bf_arr  = np.where(all_bf)[0]
        sm_arr  = np.where(all_stayed)[0]

        print(f"\n{'layer':>5}  {'cos(BF,SM)':>10}  {'L2_BF':>7}  {'L2_SM':>7}")
        print("-" * 38)
        cos_l, l2_bf_l, l2_sm_l = [], [], []
        for l in range(n_layers):
            d   = delta[:, l, :]
            mbf = d[bf_arr].mean(axis=0)
            msm = d[sm_arr].mean(axis=0)
            cos = float(np.dot(mbf, msm) / (np.linalg.norm(mbf) * np.linalg.norm(msm) + 1e-12))
            l2b = float(np.linalg.norm(mbf))
            l2s = float(np.linalg.norm(msm))
            cos_l.append(cos); l2_bf_l.append(l2b); l2_sm_l.append(l2s)
            print(f"{l:>5}  {cos:>10.4f}  {l2b:>7.4f}  {l2s:>7.4f}")

        subset_idx = np.concatenate([bf_arr, sm_arr])
        labels     = np.array([1]*len(bf_arr) + [0]*len(sm_arr), dtype=np.int64)
        majority   = float((labels == 0).mean())

        print(f"\n── Probe: all backfire vs all stayed ──")
        print(f"  Majority baseline (predict stayed): {majority:.3f}")
        print(f"  {'layer':>5}  {'nudged':>8}  {'baseline':>8}")

        acts_nudged_sub = torch.from_numpy(nudged_acts_combined[subset_idx])
        acts_base_sub   = torch.from_numpy(base_acts[subset_idx])
        probe_results   = []
        nudged_accs, base_accs = [], []
        for l in range(n_layers):
            rn = train_layer_probe(acts_nudged_sub[:, l, :], labels, layer=l)
            rb = train_layer_probe(acts_base_sub[:, l, :],   labels, layer=l)
            nudged_accs.append(rn.mean_accuracy)
            base_accs.append(rb.mean_accuracy)
            probe_results.append(rn)
            print(f"  {l:>5}  {rn.mean_accuracy:>8.3f}  {rb.mean_accuracy:>8.3f}")

        best = max(probe_results, key=lambda r: r.mean_accuracy)
        print(f"  Best (nudged acts): layer {best.layer}  acc={best.mean_accuracy:.3f}")

    # ── 8. save results ───────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_slug = args.model.replace("/", "_")
    out_path = (
        out_dir
        / f"backfire_3choice_bidir_{model_slug}_{args.category}_{args.nudge}_{args.context_condition}.json"
    )
    payload = {
        "model": args.model, "category": args.category,
        "nudge": args.nudge, "context_condition": args.context_condition,
        "n_examples": n,
        "n_elig_stereo": n_elig_s, "n_elig_other": n_elig_o,
        "n_bf_from_stereo": n_bf_s, "n_bf_from_other": n_bf_o,
        "n_fled_from_stereo": n_fled_s, "n_fled_from_other": n_fled_o,
        "n_stayed_stereo": n_stay_s, "n_stayed_other": n_stay_o,
        "n_total_backfire": n_bf_tot,
        "probe_nudged_accs": nudged_accs if probe_results else None,
        "probe_base_accs":   base_accs   if probe_results else None,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")

    # ── 9. plot ───────────────────────────────────────────────────────────────
    if args.no_plot or probe_results is None:
        return

    layers = list(range(n_layers))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    ax = axes[0]
    ax.plot(layers, cos_l, marker="o", color="steelblue")
    ax.axhline(0, color="grey", linestyle="--", lw=0.8)
    ax.axhline(1, color="grey", linestyle=":", lw=0.8)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Layer"); ax.set_ylabel("Cosine similarity")
    ax.set_title("Similarity of activation change\nAll backfire vs all stayed")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(layers, l2_bf_l, marker="o", color="firebrick",
            label=f"backfire (n={n_bf_tot})")
    ax.plot(layers, l2_sm_l, marker="s", color="seagreen",
            label=f"stayed (n={n_stay_tot})")
    ax.set_xlabel("Layer"); ax.set_ylabel("L2 norm of mean (nudged−baseline)")
    ax.set_title("Magnitude of activation change\n(3-choice BBQ, bidirectional)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(layers, nudged_accs, marker="o", color="darkorchid", label="nudged acts")
    ax.plot(layers, base_accs,   marker="s", color="darkorange",  label="baseline acts")
    ax.axhline(majority, color="grey", linestyle="-.", lw=0.8,
               label=f"majority ({majority:.2f})")
    ax.set_xlabel("Layer"); ax.set_ylabel("5-fold CV accuracy")
    ax.set_title("Probe: predict backfire\nvs stayed (bidirectional)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(
        f"{args.model.split('/')[-1]} | {args.category} | nudge={args.nudge!r} | 3-choice bidir\n"
        f"BF from stereo={n_bf_s}  BF from other={n_bf_o}  total BF={n_bf_tot}  "
        f"stayed={n_stay_tot}",
        fontsize=10,
    )
    fig.tight_layout()
    plot_path = (
        out_dir
        / f"backfire_3choice_bidir_{model_slug}_{args.category}_{args.nudge}_{args.context_condition}.png"
    )
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
