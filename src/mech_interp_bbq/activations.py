"""Extract residual-stream activations from a TransformerLens model.

The standard probe-friendly site is the post-resid stream at each layer
(``blocks.{l}.hook_resid_post``). We collect the activation at the last
token position of each prompt, which encodes the model's summary state
just before the answer is generated.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer


@dataclass(slots=True)
class ActivationBatch:
    """Activations of shape (n_examples, n_layers, d_model)."""

    acts: torch.Tensor
    layer_names: list[str]

    def for_layer(self, layer: int) -> torch.Tensor:
        return self.acts[:, layer, :]


def load_model(
    model_name: str = "gpt2-small",
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> HookedTransformer:
    """Load a HookedTransformer model.

    Sensible defaults to start with: ``gpt2-small`` for fast CPU iteration,
    ``pythia-410m`` / ``pythia-1.4b`` for a step up, ``meta-llama/Llama-3.2-1B``
    if you have a GPU and a HF token.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model = model.to(dtype)
    model.eval()
    return model


@torch.inference_mode()
def collect_resid_post(
    model: HookedTransformer,
    prompts: Sequence[str],
    layers: Iterable[int] | None = None,
    batch_size: int = 8,
    pool: str = "last",
) -> ActivationBatch:
    """Run ``prompts`` through ``model`` and gather residual-stream activations.

    Args:
        model: A HookedTransformer (eval mode).
        prompts: List of input strings.
        layers: Which layers to keep. Defaults to all.
        batch_size: How many prompts to run per forward pass.
        pool: How to reduce over sequence dimension. ``"last"`` keeps the
            final non-pad token; ``"mean"`` averages over real tokens.
    """
    if layers is None:
        layers = list(range(model.cfg.n_layers))
    layer_list = list(layers)
    hook_names = [f"blocks.{l}.hook_resid_post" for l in layer_list]

    out_chunks: list[torch.Tensor] = []
    pbar = tqdm(range(0, len(prompts), batch_size), desc="activations")
    for start in pbar:
        batch = list(prompts[start : start + batch_size])
        tokens = model.to_tokens(batch, prepend_bos=True)
        attn_mask = (tokens != model.tokenizer.pad_token_id).long() if model.tokenizer.pad_token_id is not None else torch.ones_like(tokens)
        _, cache = model.run_with_cache(
            tokens,
            names_filter=hook_names,
            return_type=None,
        )
        per_layer = []
        for name in hook_names:
            resid = cache[name]
            if pool == "last":
                lengths = attn_mask.sum(dim=1) - 1
                idx = lengths.view(-1, 1, 1).expand(-1, 1, resid.shape[-1])
                vec = resid.gather(1, idx).squeeze(1)
            elif pool == "mean":
                mask = attn_mask.unsqueeze(-1).to(resid.dtype)
                vec = (resid * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                raise ValueError(f"Unknown pool: {pool!r}")
            per_layer.append(vec.cpu().float())
        stacked = torch.stack(per_layer, dim=1)
        out_chunks.append(stacked)
        del cache

    return ActivationBatch(acts=torch.cat(out_chunks, dim=0), layer_names=hook_names)


@torch.inference_mode()
def collect_model_predictions(
    model: HookedTransformer,
    prompts: Sequence[str],
    n_choices: int = 3,
    batch_size: int = 8,
) -> np.ndarray:
    """Return the model's predicted answer index (0/1/2) for each prompt.

    We inspect the logits at the last token position over the answer-choice
    tokens ``A``, ``B``, ``C`` (and optionally ``D``) and take the argmax.
    This reflects what the model would actually output as its answer.

    Args:
        model: A HookedTransformer (eval mode).
        prompts: List of formatted prompts ending with ``"Answer:"``.
        n_choices: Number of choices (3 for BBQ).
        batch_size: Forward-pass batch size.

    Returns:
        Integer array of shape ``(n_examples,)`` with values in ``{0, …, n_choices-1}``.
    """
    # Tokenise single letters to get answer-token IDs.
    # We use the first token of each tokenised letter (no BOS).
    choice_letters = [chr(ord("A") + i) for i in range(n_choices)]
    answer_token_ids: list[int] = []
    for letter in choice_letters:
        toks = model.to_tokens(f" {letter}", prepend_bos=False)[0]
        # Take the last token in case the tokeniser splits it (rare).
        answer_token_ids.append(int(toks[-1]))

    predictions: list[int] = []
    pbar = tqdm(range(0, len(prompts), batch_size), desc="model predictions")
    for start in pbar:
        batch = list(prompts[start : start + batch_size])
        tokens = model.to_tokens(batch, prepend_bos=True)
        logits = model(tokens, return_type="logits")  # (B, seq, vocab)
        last_logits = logits[:, -1, :]                # (B, vocab)
        choice_logits = last_logits[:, answer_token_ids]  # (B, n_choices)
        preds = choice_logits.argmax(dim=-1).cpu().tolist()
        predictions.extend(preds)

    return np.array(predictions, dtype=np.int64)


@torch.inference_mode()
def collect_logit_diffs(
    model: HookedTransformer,
    prompts: Sequence[str],
    n_choices: int = 2,
    batch_size: int = 8,
) -> np.ndarray:
    """Return per-example logit differences between consecutive answer choices.

    For binary (A vs B) prompts returns ``log_P(A) - log_P(B)`` as a
    continuous preference score:
      - Positive → model prefers A
      - Negative → model prefers B
      - Large magnitude → model is confident

    For 3-choice prompts returns an array of shape ``(n, n_choices)`` with
    the raw logits over each answer token (unnormalised but comparable).

    Args:
        model: A HookedTransformer (eval mode).
        prompts: List of formatted prompts.
        n_choices: 2 for binary A/B; 3 for full BBQ A/B/C.
        batch_size: Forward-pass batch size.

    Returns:
        Float array of shape ``(n,)`` for binary, ``(n, n_choices)`` for multi.
    """
    choice_letters = [chr(ord("A") + i) for i in range(n_choices)]
    answer_token_ids: list[int] = []
    for letter in choice_letters:
        toks = model.to_tokens(f" {letter}", prepend_bos=False)[0]
        answer_token_ids.append(int(toks[-1]))

    all_logits: list[np.ndarray] = []
    pbar = tqdm(range(0, len(prompts), batch_size), desc="logit diffs")
    for start in pbar:
        batch = list(prompts[start : start + batch_size])
        tokens = model.to_tokens(batch, prepend_bos=True)
        logits = model(tokens, return_type="logits")[:, -1, :]
        choice_logits = logits[:, answer_token_ids].cpu().float().numpy()
        all_logits.append(choice_logits)

    stacked = np.concatenate(all_logits, axis=0)  # (n, n_choices)
    if n_choices == 2:
        return stacked[:, 0] - stacked[:, 1]       # scalar diff: A - B
    return stacked
