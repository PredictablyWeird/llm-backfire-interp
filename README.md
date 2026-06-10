# Mech Interp on BBQ via TransformerLens

Investigate **activation probes** on the [BBQ](https://github.com/nyu-mll/BBQ)
(Bias Benchmark for QA) dataset using [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).

The project pulls per-layer residual-stream activations from a HookedTransformer
on BBQ prompts, then fits linear probes (logistic regression) to ask:
*at what depth is a given concept linearly decodable?*

## Setup

This project is managed with [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

That creates `.venv/`, resolves a lockfile, and installs everything in
`pyproject.toml`. To activate the env (optional — `uv run` works without it):

```bash
source .venv/bin/activate
```

### Hugging Face access

If you point the model loader at a gated model (Llama-3, Gemma-2, …), set:

```bash
export HF_TOKEN=hf_xxx
```

## Smoke test

Run the end-to-end pipeline on a small slice of one BBQ category:

```bash
uv run python scripts/run_probe.py \
    --model gpt2-small \
    --category Gender_identity \
    --max-examples 200
```

Output goes to `probes_out/result.json` (per-layer 5-fold CV accuracies).

## Notebook

```bash
uv run jupyter lab notebooks/01_explore_bbq.ipynb
```

## Layout

```
src/mech_interp_bbq/
    data.py          # BBQ loader + dataclass
    activations.py   # HookedTransformer + residual-stream extraction
    probes.py        # Per-layer logistic-regression probes
scripts/run_probe.py # CLI driver
notebooks/           # Interactive exploration
data/                # Cached datasets (gitignored)
probes_out/          # Probe results (gitignored)
figures/             # Plots (gitignored)
```

## Suggested next experiments

- Compare probe accuracy across BBQ categories — does bias localise to
  different layers depending on the social axis?
- Train probes on the *ambiguous* subset and test on *disambiguated* (and
  vice versa) to separate stereotype-following from genuine inference.
- Swap `gpt2-small` for `pythia-1.4b`, `meta-llama/Llama-3.2-1B`, or
  `google/gemma-2-2b` to see whether bias-relevant features sharpen with
  scale.
- Move from `hook_resid_post` to attention-head outputs
  (`blocks.{l}.attn.hook_z`) to find which heads carry stereotype signals.
- Use the probe direction as a steering vector (TransformerLens
  `add_hook`) to causally test the feature.
