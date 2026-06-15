"""HuggingFace backend for activation capture, designed to scale to large models.

Why not TransformerLens for the big runs?
-----------------------------------------
``HookedTransformer.from_pretrained`` re-processes weights on load (folding norms,
etc.), which roughly doubles peak memory and does not reliably support every newer
architecture (e.g. Qwen3).  For a 32B model that is the difference between fitting
and OOM.  Plain HF ``AutoModelForCausalLM`` with ``device_map="auto"`` shards across
GPUs, supports bf16/quantization, and exposes the residual stream through ordinary
forward hooks — everything these experiments need.

This backend assumes a **Llama/Qwen-style decoder** where each block does:

    residual = h
    h = self_attn(input_layernorm(h));      h = residual + h     # resid after attn
    residual = h
    h = mlp(post_attention_layernorm(h));    h = residual + h     # resid_post (block out)

so that:
  * the decoder block's output hidden state  == ``hook_resid_post``
  * ``self_attn`` output                     == ``hook_attn_out``
  * ``mlp`` output                           == ``hook_mlp_out``

Both ``meta-llama/Llama-3.2-*`` and ``Qwen/Qwen3-*`` follow this layout.

Everything is captured at the **last real token** position only (the slot that
produces the answer), keeping cache sizes tractable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── model loading / introspection ────────────────────────────────────────────

@dataclass
class LoadedModel:
    model: object
    tokenizer: object
    n_layers: int
    d_model: int
    device: str

    @property
    def layers(self):
        return _decoder_layers(self.model)

    @property
    def final_norm(self):
        return _final_norm(self.model)

    @property
    def lm_head(self):
        return self.model.get_output_embeddings()


def _decoder_layers(model):
    """Return the ``nn.ModuleList`` of decoder blocks for a Llama/Qwen model."""
    base = getattr(model, "model", model)
    if hasattr(base, "layers"):
        return base.layers
    raise AttributeError(
        "Could not find decoder layers at model.model.layers — "
        "this backend supports Llama/Qwen-style architectures only."
    )


def _final_norm(model):
    base = getattr(model, "model", model)
    if hasattr(base, "norm"):
        return base.norm
    raise AttributeError("Could not find final norm at model.model.norm.")


def load_hf_model(
    model_name: str,
    dtype: str = "bfloat16",
    device_map: str | None = "auto",
) -> LoadedModel:
    """Load an HF causal LM ready for inference.

    Args:
        model_name: HF repo id, e.g. ``"Qwen/Qwen3-32B"``.
        dtype: ``"bfloat16"``, ``"float16"``, or ``"float32"``.
        device_map: ``"auto"`` to shard across visible GPUs; ``None`` to place on a
            single device (CPU if no CUDA).  Use ``None`` + CPU for local smoke tests.
    """
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-pad so the last token is always at position -1 across a batch.
    tokenizer.padding_side = "left"

    kwargs: dict = {"torch_dtype": torch_dtype}
    if device_map is not None and torch.cuda.is_available():
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if "device_map" not in kwargs:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
    model.eval()

    cfg = model.config
    device = str(next(model.parameters()).device)
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        n_layers=cfg.num_hidden_layers,
        d_model=cfg.hidden_size,
        device=device,
    )


def abc_token_ids(tokenizer) -> list[int]:
    """Token ids for the answer letters A/B/C (with a leading space, last subtoken)."""
    ids = []
    for letter in ["A", "B", "C"]:
        toks = tokenizer(f" {letter}", add_special_tokens=False)["input_ids"]
        ids.append(int(toks[-1]))
    return ids


def _input_device(model):
    """Device that token inputs should be placed on (embedding device for sharded models)."""
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


# ── forward-pass helpers ──────────────────────────────────────────────────────

def _forward_last_logits(model, enc):
    """Forward pass returning logits, computing only the last position when supported.

    ``logits_to_keep=1`` (transformers ≥ 4.49; ``num_logits_to_keep`` on older
    versions) skips the full ``(B, S, vocab)`` projection — a big memory/compute
    saving for large vocabularies. ``use_cache=False`` avoids allocating a KV cache
    we never use (no generation).
    """
    for kw in ({"logits_to_keep": 1}, {"num_logits_to_keep": 1}, {}):
        try:
            return model(**enc, use_cache=False, **kw).logits
        except TypeError:
            continue
    return model(**enc).logits


@torch.inference_mode()
def compute_abc_logits(
    lm: LoadedModel,
    prompts: list[str],
    batch_size: int = 8,
    progress: bool = True,
) -> np.ndarray:
    """Raw logits over A/B/C at the last token. Shape ``(n, 3)`` float32."""
    abc = abc_token_ids(lm.tokenizer)
    dev = _input_device(lm.model)
    out: list[np.ndarray] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = lm.tokenizer(batch, return_tensors="pt", padding=True).to(dev)
        logits = _forward_last_logits(lm.model, enc)[:, -1, :]  # left-padded → -1 is last real token
        out.append(logits[:, abc].float().cpu().numpy())
        if progress and (start // batch_size) % 25 == 0:
            print(f"    logits {min(start + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return np.concatenate(out, axis=0)


@torch.inference_mode()
def capture_activations(
    lm: LoadedModel,
    prompts: list[str],
    batch_size: int = 8,
    capture_components: bool = False,
    progress: bool = True,
) -> dict[str, np.ndarray]:
    """Capture last-token per-layer activations in a single forward pass per batch.

    Returns a dict with:
      * ``resid``  — (n, n_layers, d_model): block outputs == ``hook_resid_post``
      * ``mlp``    — (n, n_layers, d_model): only if ``capture_components``
      * ``attn``   — (n, n_layers, d_model): only if ``capture_components``

    Stored as float16 to keep cache size manageable for large models.
    """
    layers = lm.layers
    n_layers = lm.n_layers
    dev = _input_device(lm.model)

    buf: dict = {"resid": [[] for _ in range(n_layers)]}
    if capture_components:
        buf["mlp"] = [[] for _ in range(n_layers)]
        buf["attn"] = [[] for _ in range(n_layers)]

    # Per-batch scratch updated by the hooks.
    scratch: dict = {}

    def _last_token(t: torch.Tensor) -> torch.Tensor:
        # t: (B, seq, d). Left-padded → last token is at -1.
        return t[:, -1, :].float().cpu()

    def make_block_hook(layer_idx: int):
        def hook(_module, _inp, output):
            h = output[0] if isinstance(output, tuple) else output
            scratch.setdefault("resid", {})[layer_idx] = _last_token(h)
        return hook

    def make_attn_hook(layer_idx: int):
        def hook(_module, _inp, output):
            h = output[0] if isinstance(output, tuple) else output
            scratch.setdefault("attn", {})[layer_idx] = _last_token(h)
        return hook

    def make_mlp_hook(layer_idx: int):
        def hook(_module, _inp, output):
            h = output[0] if isinstance(output, tuple) else output
            scratch.setdefault("mlp", {})[layer_idx] = _last_token(h)
        return hook

    handles = []
    for i, blk in enumerate(layers):
        handles.append(blk.register_forward_hook(make_block_hook(i)))
        if capture_components:
            handles.append(blk.self_attn.register_forward_hook(make_attn_hook(i)))
            handles.append(blk.mlp.register_forward_hook(make_mlp_hook(i)))

    # Run the base decoder, not the CausalLM head: the hooks fire on the same block
    # modules, but we skip the (B, S, vocab) lm_head projection we'd only discard.
    base_model = getattr(lm.model, "model", lm.model)

    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            enc = lm.tokenizer(batch, return_tensors="pt", padding=True).to(dev)
            scratch.clear()
            base_model(**enc, use_cache=False)
            for i in range(n_layers):
                buf["resid"][i].append(scratch["resid"][i])
                if capture_components:
                    buf["mlp"][i].append(scratch["mlp"][i])
                    buf["attn"][i].append(scratch["attn"][i])
            if progress and (start // batch_size) % 25 == 0:
                print(f"    acts {min(start + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    finally:
        for h in handles:
            h.remove()

    out: dict[str, np.ndarray] = {}
    for key in buf:
        # buf[key]: list over layers of list over batches of (B, d)
        per_layer = [torch.cat(chunks, dim=0) for chunks in buf[key]]  # list of (n, d)
        out[key] = torch.stack(per_layer, dim=1).numpy().astype(np.float16)  # (n, L, d)
    return out


@torch.inference_mode()
def capture_prompt_region_attention(
    lm: LoadedModel,
    prompts: list[str],
    batch_size: int = 2,
    layer_indices: list[int] | None = None,
    progress: bool = True,
) -> np.ndarray:
    """Aggregate last-token attention mass by prompt region.

    For each example and layer, averages over heads then sums attention from the
    last real token to all tokens in each region.  Returns **fractions** that sum
    to 1 over assigned (non-pad) regions at each layer.

    Args:
        lm: Loaded HF model.
        prompts: Input strings.
        batch_size: Keep small — ``output_attentions=True`` is memory-heavy.
        layer_indices: Subset of layers to store; ``None`` = all layers.

    Returns:
        ``(n, n_layers, n_regions)`` float32 region attention fractions.
    """
    from .prompt_regions import N_REGIONS, batch_token_region_labels

    n_layers_total = lm.n_layers
    if layer_indices is None:
        layer_indices = list(range(n_layers_total))
    n_store = len(layer_indices)

    dev = _input_device(lm.model)
    n = len(prompts)
    out = np.zeros((n, n_store, N_REGIONS), dtype=np.float32)

    for start in range(0, n, batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch_labels = batch_token_region_labels(lm.tokenizer, batch_prompts)
        enc = lm.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(dev)
        attn_mask = enc["attention_mask"]  # (B, S)

        outputs = lm.model(**enc, output_attentions=True, use_cache=False)
        attentions = outputs.attentions  # tuple len n_layers of (B, H, S, S)

        for b in range(len(batch_prompts)):
            labels = batch_labels[b]
            if len(labels) != attn_mask.shape[1]:
                raise ValueError(
                    f"Label length {len(labels)} != encoded seq {attn_mask.shape[1]}"
                )
            valid = (labels >= 0) & (attn_mask[b].cpu().numpy() > 0)
            if not valid.any():
                continue

            for li, layer_idx in enumerate(layer_indices):
                # (H, S) — query is last token (left-padded → index -1)
                w = attentions[layer_idx][b, :, -1, :].float().mean(dim=0).cpu().numpy()
                w = w * valid
                total = w.sum()
                if total <= 0:
                    continue
                for r in range(N_REGIONS):
                    mask = valid & (labels == r)
                    out[start + b, li, r] = float(w[mask].sum() / total)

        if progress and (start // batch_size) % 10 == 0:
            print(
                f"    attn {min(start + batch_size, n)}/{n}",
                flush=True,
            )

    return out


def save_unembed_meta(lm: LoadedModel, path: str) -> None:
    """Save just the A/B/C unembed columns + final-norm params for CPU-side projection.

    This lets ``analyze.py`` reconstruct A/B/C logits from any cached residual on a
    laptop, with no GPU and no full ``lm_head`` (which is ~1.5GB for a 32B vocab).
    """
    abc = abc_token_ids(lm.tokenizer)
    w = lm.lm_head.weight.detach().float().cpu().numpy()  # (vocab, d_model)
    abc_unembed = w[abc, :].T  # (d_model, 3)
    norm_w = lm.final_norm.weight.detach().float().cpu().numpy()  # (d_model,)
    eps = float(getattr(lm.final_norm, "variance_epsilon", getattr(lm.final_norm, "eps", 1e-6)))
    np.savez_compressed(
        path,
        abc_unembed=abc_unembed,
        norm_weight=norm_w,
        norm_eps=np.array(eps, dtype=np.float64),
        abc_ids=np.array(abc, dtype=np.int64),
        n_layers=np.array(lm.n_layers, dtype=np.int64),
        d_model=np.array(lm.d_model, dtype=np.int64),
    )


# ── CPU-side projection (used by analyze.py, no GPU needed) ────────────────────

def project_resid_to_abc(
    resid: np.ndarray,
    norm_weight: np.ndarray,
    norm_eps: float,
    abc_unembed: np.ndarray,
) -> np.ndarray:
    """Apply final RMSNorm + A/B/C unembed to a residual stream.

    Args:
        resid: (..., d_model) residual activations (last token, any layer).
        norm_weight: (d_model,) final RMSNorm weight.
        norm_eps: RMSNorm epsilon.
        abc_unembed: (d_model, 3) the three answer-letter unembed columns.

    Returns:
        (..., 3) A/B/C logits.
    """
    x = resid.astype(np.float32)
    ms = np.mean(x * x, axis=-1, keepdims=True)
    x_normed = x / np.sqrt(ms + norm_eps) * norm_weight
    return x_normed @ abc_unembed
