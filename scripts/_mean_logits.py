"""Show mean token distribution at final layer across 10 BBQ examples."""

import os, warnings
os.environ["TRANSFORMERLENS_ALLOW_MPS"] = "1"
warnings.filterwarnings("ignore")

import torch
import numpy as np
from mech_interp_bbq.activations import load_model
from mech_interp_bbq.data import load_bbq_pairwise

N = 10
examples = load_bbq_pairwise("Gender_identity", context_condition="ambig", max_examples=N)
prompts  = [ex.prompt_with_sentence() for ex in examples]

model = load_model("meta-llama/Llama-3.2-1B")

# ── for each prompt, get the final-layer logits at the last token ─────────────
all_probs = []   # list of (vocab_size,) arrays

with torch.inference_mode():
    for i, prompt in enumerate(prompts):
        tokens = model.to_tokens(prompt, prepend_bos=True)   # (1, seq)
        logits = model(tokens)                                # (1, seq, V)
        last_logits = logits[0, -1, :]                       # (V,)
        probs = torch.softmax(last_logits.float(), dim=-1)   # (V,)
        all_probs.append(probs.cpu().numpy())

        tok_A = int(model.to_tokens(" A", prepend_bos=False)[0, -1])
        tok_B = int(model.to_tokens(" B", prepend_bos=False)[0, -1])
        p_A   = float(probs[tok_A])
        p_B   = float(probs[tok_B])
        pred  = "A" if p_A > p_B else "B"
        print(f"  ex {i:>2}: P(A)={p_A:.4f}  P(B)={p_B:.4f}  → model picks {pred}")

# ── mean probabilities across all 10 examples ─────────────────────────────────
mean_probs = np.stack(all_probs).mean(axis=0)   # (V,)

tok_A = int(model.to_tokens(" A", prepend_bos=False)[0, -1])
tok_B = int(model.to_tokens(" B", prepend_bos=False)[0, -1])

top_idx = np.argsort(mean_probs)[::-1][:20]

print(f"\n{'='*60}")
print(f"Mean probability distribution at final layer ({N} examples)")
print(f"{'='*60}")
print(f"  {'rank':>4}  {'token':<20}  {'mean prob':>10}  {'mean log-prob':>14}")
print("  " + "-" * 52)
for rank, idx in enumerate(top_idx, 1):
    ts  = model.tokenizer.decode([idx])
    p   = float(mean_probs[idx])
    lp  = float(np.log(p)) if p > 0 else float("-inf")
    tag = " ← A" if idx == tok_A else (" ← B" if idx == tok_B else "")
    print(f"  {rank:>4}  {repr(ts):<20}  {p:>10.6f}  {lp:>14.4f}{tag}")

print(f"\n  Answer tokens:")
print(f"    ' A'  mean prob = {mean_probs[tok_A]:.6f}  ({100*mean_probs[tok_A]:.3f}%)")
print(f"    ' B'  mean prob = {mean_probs[tok_B]:.6f}  ({100*mean_probs[tok_B]:.3f}%)")
print(f"\n  Sum of top-20 token probs = {mean_probs[top_idx].sum():.4f}")
print(f"  Remaining prob mass in other {len(mean_probs)-20:,} tokens = "
      f"{1 - mean_probs[top_idx].sum():.4f}")
