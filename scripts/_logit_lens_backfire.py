"""Logit lens: which tokens distinguish backfire from stayed-at-target?

At each layer l we project the residual stream to vocabulary space:
    logits[i, l] = unembed(layernorm(resid_post[i, l]))   shape: (vocab,)
    probs[i, l]  = softmax(logits[i, l])

Then for each layer we compute the mean probability difference:
    diff[l, token] = mean_prob(backfire, l, token) - mean_prob(stayed, l, token)

Tokens with large positive diff are "over-represented" in backfire cases.
Tokens with large negative diff are "over-represented" in stayed cases.

We report:
  1. Top-K differentially likely tokens at a set of key layers
  2. Heatmap of token probability across layers for the top differential tokens
  3. Logit lens score for answer tokens A and B across layers, split by group

Run:
    uv run --env-file .env python scripts/_logit_lens_backfire.py
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from mech_interp_bbq.activations import load_model
from mech_interp_bbq.data import load_bbq_pairwise
from mech_interp_bbq.nudges import NUDGE_TEMPLATES, group_label, singular_group_label

TOP_K      = 15     # top tokens to show per layer
KEY_LAYERS = [0, 4, 7, 10, 11, 13, 14, 15]   # layers to inspect closely
BATCH_SIZE = 8

# ── stereotype slot helper ────────────────────────────────────────────────────

def stereotyped_slot(ex) -> int | None:
    sg_lower = {g.lower() for g in ex.stereotyped_groups}
    def is_sg(tag: str) -> bool:
        tc = tag.lower()
        return tc in sg_lower or any(g in tc or tc in g for g in sg_lower)
    a_sg, b_sg = is_sg(ex.group_a_tag), is_sg(ex.group_b_tag)
    if ex.polarity == "neg":
        if a_sg and not b_sg: return 0
        if b_sg and not a_sg: return 1
    else:
        if not a_sg and b_sg: return 0
        if not b_sg and a_sg: return 1
    return None

# ── setup ─────────────────────────────────────────────────────────────────────

CATEGORY = "Gender_identity"
examples  = load_bbq_pairwise(CATEGORY, context_condition="ambig", max_examples=400)
tmpl      = NUDGE_TEMPLATES["user_preference"]

baseline_prompts = []
nudge_a_prompts  = []
nudge_b_prompts  = []

for ex in examples:
    gl_a = group_label(CATEGORY, ex.group_a_tag)
    gl_b = group_label(CATEGORY, ex.group_b_tag)
    sg_a = singular_group_label(CATEGORY, ex.group_a_tag)
    sg_b = singular_group_label(CATEGORY, ex.group_b_tag)
    sent_a = f"({tmpl.template.format(group_label=gl_a, other_group_label=gl_b, singular_group_label=sg_a)})"
    sent_b = f"({tmpl.template.format(group_label=gl_b, other_group_label=gl_a, singular_group_label=sg_b)})"
    baseline_prompts.append(ex.prompt_with_sentence())
    nudge_a_prompts.append(ex.prompt_with_sentence(sent_a, position=tmpl.position))
    nudge_b_prompts.append(ex.prompt_with_sentence(sent_b, position=tmpl.position))

model    = load_model("meta-llama/Llama-3.2-1B")
n_layers = model.cfg.n_layers
vocab_size = model.cfg.d_vocab

# ── collect predictions to classify backfire / stayed ──────────────────────

from mech_interp_bbq.activations import collect_model_predictions

print("Collecting predictions...")
base_preds    = collect_model_predictions(model, baseline_prompts, n_choices=2, batch_size=BATCH_SIZE)
nudge_a_preds = collect_model_predictions(model, nudge_a_prompts,  n_choices=2, batch_size=BATCH_SIZE)
nudge_b_preds = collect_model_predictions(model, nudge_b_prompts,  n_choices=2, batch_size=BATCH_SIZE)

strict_bf, stayed_m = [], []
for i in range(len(examples)):
    bp  = base_preds[i]
    np_ = nudge_a_preds[i] if bp == 0 else nudge_b_preds[i]
    if bp != np_:
        strict_bf.append(i)
    else:
        stayed_m.append(i)

bf_arr = np.array(strict_bf)
sm_arr = np.array(stayed_m)
print(f"Strict backfire: {len(bf_arr)}   Stayed-at-target: {len(sm_arr)}")

# Relevant nudged prompts per example
relevant_nudged = [
    nudge_a_prompts[i] if base_preds[i] == 0 else nudge_b_prompts[i]
    for i in range(len(examples))
]

# ── logit lens: project resid_post at each layer to vocab ─────────────────────

@torch.inference_mode()
def logit_lens_batched(prompts: list[str]) -> np.ndarray:
    """Return log-probabilities of shape (N, n_layers, vocab_size).

    At each layer l, applies ln_final + unembed to resid_post[:, last_token, :].
    """
    all_logprobs = []

    for start in tqdm(range(0, len(prompts), BATCH_SIZE), desc="logit lens"):
        batch = prompts[start : start + BATCH_SIZE]
        tokens = model.to_tokens(batch, prepend_bos=True)

        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda n: "hook_resid_post" in n,
            return_type=None,
        )

        # (n_layers, batch, seq, d_model) → last token position
        batch_logprobs = []
        for l in range(n_layers):
            resid = cache[f"blocks.{l}.hook_resid_post"][:, -1, :]  # (B, D)
            normed = model.ln_final(resid)                            # (B, D)
            logits = model.unembed(normed)                            # (B, V)
            lp     = torch.log_softmax(logits.float(), dim=-1)        # (B, V)
            batch_logprobs.append(lp.cpu().numpy())

        # stack: (n_layers, B, V) → (B, n_layers, V)
        batch_arr = np.stack(batch_logprobs, axis=1)
        all_logprobs.append(batch_arr)

    return np.concatenate(all_logprobs, axis=0)   # (N, n_layers, V)


CACHE_FILE = pathlib.Path("probes_out/logit_lens_nudged_lp.npz")

if CACHE_FILE.exists():
    print(f"\nLoading cached logit lens from {CACHE_FILE} ...")
    nudged_lp = np.load(CACHE_FILE)["nudged_lp"]   # (N, n_layers, V)
    print(f"  Loaded shape: {nudged_lp.shape}")
else:
    print("\nCollecting logit lens for nudged prompts...")
    nudged_lp = logit_lens_batched(relevant_nudged)   # (N, n_layers, V)
    np.savez_compressed(CACHE_FILE, nudged_lp=nudged_lp)
    print(f"  Saved logit lens cache to {CACHE_FILE}")

# ── compute differential token probabilities ──────────────────────────────────

# mean log-prob per group at each layer
mean_lp_bf = nudged_lp[bf_arr].mean(axis=0)   # (n_layers, V)
mean_lp_sm = nudged_lp[sm_arr].mean(axis=0)   # (n_layers, V)
diff        = mean_lp_bf - mean_lp_sm          # (n_layers, V)  positive = more in BF

# ── report: top differentially likely tokens per key layer ────────────────────

print(f"\n{'='*70}")
print("Top tokens distinguishing BACKFIRE from STAYED at each key layer")
print(f"{'='*70}")

# Answer token ids
tok_A = int(model.to_tokens(" A", prepend_bos=False)[0, -1])
tok_B = int(model.to_tokens(" B", prepend_bos=False)[0, -1])

for l in KEY_LAYERS:
    d = diff[l]   # (V,)
    top_pos = np.argsort(d)[::-1][:TOP_K]    # more likely in backfire
    top_neg = np.argsort(d)[:TOP_K]           # more likely in stayed

    print(f"\n── Layer {l} ──")
    print(f"  More likely in BACKFIRE (+):")
    for t in top_pos:
        word = model.to_string([t])
        print(f"    {word!r:20s}  Δlog-prob = {d[t]:+.4f}")

    print(f"  More likely in STAYED (−):")
    for t in top_neg:
        word = model.to_string([t])
        print(f"    {word!r:20s}  Δlog-prob = {d[t]:+.4f}")

    # Answer tokens specifically
    print(f"  Answer tokens:  A Δ={d[tok_A]:+.4f}   B Δ={d[tok_B]:+.4f}")

# ── plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Panel 1: answer token A and B log-prob across layers, split by group
ax = axes[0]
layers = list(range(n_layers))
ax.plot(layers, mean_lp_bf[:, tok_A], marker="o", color="firebrick",
        linestyle="-",  label="backfire — logP(A)")
ax.plot(layers, mean_lp_sm[:, tok_A], marker="o", color="seagreen",
        linestyle="-",  label="stayed — logP(A)")
ax.plot(layers, mean_lp_bf[:, tok_B], marker="s", color="firebrick",
        linestyle="--", label="backfire — logP(B)")
ax.plot(layers, mean_lp_sm[:, tok_B], marker="s", color="seagreen",
        linestyle="--", label="stayed — logP(B)")
ax.set_xlabel("Layer")
ax.set_ylabel("Mean log-probability (logit lens)")
ax.set_title("Answer token probabilities across layers\n(logit lens, nudged prompts)")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

# Panel 2: Δlog-prob for A and B tokens across layers
ax = axes[1]
diff_A = diff[:, tok_A]
diff_B = diff[:, tok_B]
ax.plot(layers, diff_A, marker="o", color="steelblue",
        label="Δlog-prob(A)  [backfire − stayed]")
ax.plot(layers, diff_B, marker="s", color="darkorange",
        label="Δlog-prob(B)  [backfire − stayed]")
ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
ax.set_xlabel("Layer")
ax.set_ylabel("Δlog-probability (backfire − stayed)")
ax.set_title("Which answer token does the residual stream\nfavour more in backfire cases?")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 3: heatmap of top differential tokens across layers
# Find the union of top-K tokens across all key layers
top_tokens_union = set()
for l in KEY_LAYERS:
    d = diff[l]
    top_tokens_union.update(np.argsort(np.abs(d))[::-1][:8].tolist())
# Add answer tokens
top_tokens_union.update([tok_A, tok_B])
top_tokens = sorted(top_tokens_union)

token_labels = [repr(model.to_string([t])) for t in top_tokens]
heat_data    = diff[np.ix_(KEY_LAYERS, top_tokens)]   # (len(KEY_LAYERS), len(top_tokens))

ax = axes[2]
vmax = np.abs(heat_data).max()
im   = ax.imshow(heat_data, aspect="auto", cmap="RdBu_r",
                 vmin=-vmax, vmax=vmax, interpolation="nearest")
plt.colorbar(im, ax=ax, label="Δlog-prob (backfire − stayed)")
ax.set_xticks(range(len(top_tokens)))
ax.set_xticklabels(token_labels, rotation=60, ha="right", fontsize=7)
ax.set_yticks(range(len(KEY_LAYERS)))
ax.set_yticklabels([f"L{l}" for l in KEY_LAYERS])
ax.set_title("Differential token log-probs\nacross key layers (red=more in BF)")

fig.suptitle(
    f"Llama-3.2-1B | {CATEGORY} | Logit lens: backfire vs stayed-at-target\n"
    f"Which tokens does the residual stream predict differently at each layer?",
    fontsize=10,
)
fig.tight_layout()
out = pathlib.Path("probes_out/logit_lens_backfire.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved {out}")
plt.close(fig)

# ── actual log-probs for stayed group at layers 14 & 15 ──────────────────────

print(f"\n{'='*70}")
print("Actual mean log-probs for STAYED group — layers 14 & 15")
print(f"{'='*70}")

for l in [14, 15]:
    lp_stayed_l = mean_lp_sm[l]           # (V,) — mean log-prob per token for stayed group
    top_idx     = np.argsort(lp_stayed_l)[::-1][:20]
    print(f"\n── Layer {l}  (top-20 most likely tokens in stayed group) ──")
    print(f"  {'token':<25}  {'mean log-prob':>14}  {'prob (approx)':>14}")
    print("  " + "─" * 57)
    for idx in top_idx:
        tok_str = model.tokenizer.decode([idx])
        lp_val  = float(lp_stayed_l[idx])
        prob    = float(np.exp(lp_val))
        print(f"  {repr(tok_str):<25}  {lp_val:>14.4f}  {prob:>14.6f}")
    print(f"\n  Answer tokens specifically:")
    print(f"    ' A'  log-prob = {float(lp_stayed_l[tok_A]):.4f}  "
          f"prob ≈ {float(np.exp(lp_stayed_l[tok_A])):.6f}")
    print(f"    ' B'  log-prob = {float(lp_stayed_l[tok_B]):.4f}  "
          f"prob ≈ {float(np.exp(lp_stayed_l[tok_B])):.6f}")

# ── stereotype-aware logit lens ───────────────────────────────────────────────
# For each example, determine which answer token (A or B) is stereotyped.
# Then track logP(stereo_token) and logP(nonstereo_token) across layers.

print(f"\n{'='*70}")
print("Stereotype-aware logit lens")
print(f"{'='*70}")

stereo_slots = [stereotyped_slot(examples[i]) for i in range(len(examples))]

# Split backfire cases by post-flip direction
# bf_to_stereo  : baseline chose non-stereo, nudge toward non-stereo, flipped to stereo
# bf_from_stereo: baseline chose stereo,     nudge toward stereo,     flipped to non-stereo
bf_to_stereo   = []   # flip lands ON stereotype
bf_from_stereo = []   # flip lands AWAY from stereotype
bf_unresolvable = []

for i in bf_arr:
    ss = stereo_slots[i]
    if ss is None:
        bf_unresolvable.append(i)
        continue
    bp  = base_preds[i]
    np_ = nudge_a_preds[i] if bp == 0 else nudge_b_preds[i]
    if np_ == ss:
        bf_to_stereo.append(i)
    else:
        bf_from_stereo.append(i)

print(f"Backfire toward stereotype  (flip → stereo)   : {len(bf_to_stereo)}")
print(f"Backfire away from stereotype (flip → non-stereo): {len(bf_from_stereo)}")
print(f"Backfire unresolvable                           : {len(bf_unresolvable)}")

# For resolvable examples, compute per-example stereo/non-stereo log-probs
# lp_stereo[i, l]    = log-prob of the stereotyped answer token at layer l
# lp_nonstereo[i, l] = log-prob of the non-stereotyped answer token

resolvable_idx = [i for i in range(len(examples)) if stereo_slots[i] is not None]
lp_stereo    = np.zeros((len(resolvable_idx), n_layers), dtype=np.float32)
lp_nonstereo = np.zeros((len(resolvable_idx), n_layers), dtype=np.float32)

for j, i in enumerate(resolvable_idx):
    ss = stereo_slots[i]
    t_stereo    = tok_A if ss == 0 else tok_B
    t_nonstereo = tok_B if ss == 0 else tok_A
    lp_stereo[j]    = nudged_lp[i, :, t_stereo]
    lp_nonstereo[j] = nudged_lp[i, :, t_nonstereo]

# stereo_margin[i, l] = log-prob of stereo token minus non-stereo token
stereo_margin = lp_stereo - lp_nonstereo   # (n_resolvable, n_layers)

# Map group indices from resolvable_idx space
def to_resolvable(indices):
    return [resolvable_idx.index(i) for i in indices if i in resolvable_idx]

ri_bf_to   = to_resolvable(bf_to_stereo)
ri_bf_from = to_resolvable(bf_from_stereo)
ri_sm      = to_resolvable([i for i in sm_arr if stereo_slots[i] is not None])

print(f"\nResolvable counts: bf→stereo={len(ri_bf_to)}, bf→nons={len(ri_bf_from)}, stayed={len(ri_sm)}")

print(f"\n{'layer':>6}  {'BF→stereo':>12}  {'BF→non-stereo':>14}  {'Stayed':>8}  "
      f"{'Δ(BFts-SM)':>12}  {'Δ(BFfs-SM)':>12}")
print("─" * 72)
for l in layers:
    m_bfts = stereo_margin[ri_bf_to,   l].mean() if ri_bf_to   else float("nan")
    m_bffs = stereo_margin[ri_bf_from, l].mean() if ri_bf_from else float("nan")
    m_sm   = stereo_margin[ri_sm,      l].mean() if ri_sm      else float("nan")
    print(f"  L{l:>2}  {m_bfts:>12.4f}  {m_bffs:>14.4f}  {m_sm:>8.4f}  "
          f"{m_bfts-m_sm:>+12.4f}  {m_bffs-m_sm:>+12.4f}")

# ── plot: stereotype margin across layers ─────────────────────────────────────

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

ax = axes2[0]
if ri_sm:
    ax.plot(layers, [stereo_margin[ri_sm, l].mean() for l in layers],
            marker="s", color="seagreen", linewidth=2,
            label=f"stayed (n={len(ri_sm)})")
if ri_bf_to:
    ax.plot(layers, [stereo_margin[ri_bf_to, l].mean() for l in layers],
            marker="o", color="firebrick", linewidth=2,
            label=f"backfire → stereo (n={len(ri_bf_to)})")
if ri_bf_from:
    ax.plot(layers, [stereo_margin[ri_bf_from, l].mean() for l in layers],
            marker="^", color="darkorange", linewidth=2,
            label=f"backfire → non-stereo (n={len(ri_bf_from)})")
ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
ax.set_xlabel("Layer")
ax.set_ylabel("log-prob(stereotyped token) − log-prob(other token)")
ax.set_title("Stereotype margin across layers\n(positive = model leans stereotyped)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 2: delta from stayed baseline
ax = axes2[1]
if ri_sm:
    sm_means = np.array([stereo_margin[ri_sm, l].mean() for l in layers])
    if ri_bf_to:
        bfts_means = np.array([stereo_margin[ri_bf_to,   l].mean() for l in layers])
        ax.plot(layers, bfts_means - sm_means, marker="o", color="firebrick",
                linewidth=2, label=f"backfire → stereo (n={len(ri_bf_to)})")
    if ri_bf_from:
        bffs_means = np.array([stereo_margin[ri_bf_from, l].mean() for l in layers])
        ax.plot(layers, bffs_means - sm_means, marker="^", color="darkorange",
                linewidth=2, label=f"backfire → non-stereo (n={len(ri_bf_from)})")
ax.axhline(0, color="grey", linestyle="--", linewidth=0.8,
           label="stayed (reference = 0)")
ax.set_xlabel("Layer")
ax.set_ylabel("Δ stereotype margin vs stayed group")
ax.set_title("Does the residual stream lean more or less\nstereotyped in backfire cases?")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

fig2.suptitle(
    f"Llama-3.2-1B | {CATEGORY} | Stereotype-aware logit lens\n"
    f"Tracking stereotyped answer token log-prob across layers",
    fontsize=10,
)
fig2.tight_layout()
out2 = pathlib.Path("probes_out/logit_lens_stereotype.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"\nSaved {out2}")
plt.close(fig2)
