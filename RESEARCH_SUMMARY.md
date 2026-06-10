# Mechanistic Interpretability of Social Bias in LLMs
## Research Summary — BBQ Dataset + Llama-3.2-1B

---

## 1. Project Overview

This project investigates **how social biases are encoded and processed inside a language model**, using mechanistic interpretability tools. The core setup:

- **Model**: `meta-llama/Llama-3.2-1B` via TransformerLens
- **Dataset**: [BBQ (Bias Benchmark for QA)](https://huggingface.co/datasets/HiTZ/bbq) — multiple-choice questions about social groups under ambiguous contexts
- **Task format**: Binary forced-choice (A or B) rather than the original 3-way format
- **Tools**: TransformerLens, scikit-learn probes, matplotlib/seaborn plots
- **Package manager**: `uv`

---

## 2. Repository Structure
n
```
mech_interp/
├── src/mech_interp_bbq/
│   ├── data.py          # BBQ loading, prompt formatting, pairwise conversion
│   ├── activations.py   # Model loading, residual stream collection, predictions
│   ├── probes.py        # Linear probe training (LogisticRegression per layer)
│   └── nudges.py        # Nudge sentence templates (from PredictablyWeird/Choices)
├── scripts/
│   ├── run_bias_probe.py                    # Experiment 1: bias emergence probing
│   ├── _strict_backfire_corrected.py        # Experiment 2: bidirectional backfire analysis
│   ├── _backfire_threshold.py               # Experiment 3: baseline margin analysis
│   ├── _backfire_stereotype_check.py        # Experiment 4: post-flip stereotype direction
│   ├── _backfire_control_nudge.py           # Experiment 5: control (mention-only) nudge
│   ├── _backfire_logit_analysis.py          # Experiment 6: logit shift analysis
│   ├── _layer_ablation_stereotype.py        # Experiment 7: single-category layer ablation
│   ├── _layer_ablation_all_categories.py    # Experiment 8: all-category layer ablation
│   └── _logit_lens_backfire.py              # Experiment 9: logit lens analysis
└── probes_out/                              # All saved figures and results
```

---

## 3. Experiments & Findings

### Experiment 1 — Bias Emergence Probing

**Question**: Can we detect emerging bias in intermediate layers *before* the model outputs its answer?

**Setup**: Trained three linear probes (logistic regression) on residual stream activations at each of the 16 layers:
- `gold` probe: predict the correct BBQ answer
- `stereotype` probe: predict the stereotyped answer
- `model_pred` probe: predict the model's actual output

**Key finding**: The `stereotype` probe achieves high accuracy in middle layers (L8–L12), earlier than the `model_pred` probe peaks, suggesting the model internally "commits" to a stereotyped representation before its final answer token is determined.

---

### Experiment 2 — Strict Backfire Analysis (Bidirectional Nudging)

**Question**: When we nudge the model toward its current baseline choice, does it sometimes flip away (backfire)?

**Definition of strict backfire**:
> Baseline = X. Nudge toward X. Model output = Y. (A flip against the nudge.)

**Setup**: For each of 400 ambiguous BBQ examples from `Gender_identity`, ran **two nudges** — one toward group A, one toward group B. Picked the nudge matching each example's baseline choice, making all 400 examples eligible.

**Nudge template** (`user_preference`): *"I personally think the answer involves \<group\>."*

**Results**:
- **~11% strict backfire rate** (baseline = X, nudge → X, result = Y)
- **~89% stayed at target** (nudge reinforced the baseline choice)
- Probing showed backfire cases are weakly separable from stayed cases using layer 8–12 activations

---

### Experiment 3 — Baseline Margin Analysis

**Question**: Are backfire cases already near a "stereotype resistance" threshold before any nudge?

**Method**: Measured the logit margin (`logit(A) − logit(B)`) at baseline for backfire vs. stayed groups.

**Key finding**:

| Group | Mean baseline margin |
|---|---|
| Strict backfire | +0.037 (near decision boundary) |
| Stayed at target | +0.204 (confidently committed) |

Backfire cases are significantly closer to the decision boundary before any nudge — they are in a genuinely uncertain state.

---

### Experiment 4 — Post-Flip Stereotype Direction

**Question**: When the model backfires, does it flip *toward* or *away from* the stereotyped answer?

**Method**: Used `polarity` and `stereotyped_groups` metadata from BBQ to identify which answer slot (A or B) is stereotyped.

**Results** (of 44 backfire cases, 24 resolvable):
- **67%** flipped *away* from the stereotype (were on stereotype at baseline, nudge reinforced, flipped off)
- **33%** flipped *toward* the stereotype

Interpretation: The majority of backfire cases involve the model *resisting* its own stereotype when nudged toward it — a form of stereotype suppression under pressure.

---

### Experiment 5 — Control (Mention-Only) Nudge

**Question**: Does backfire require a *directional* nudge, or does merely *mentioning* the group cause instability?

**Method**: Replaced the directional nudge ("I think the answer involves \<group\>") with a neutral mention ("This question is about \<group\>.") and measured flip rates and logit shifts.

**Results**:

| Nudge type | Flip rate | Mean logit shift |
|---|---|---|
| Directional (user_preference) | 11.0% | +0.123 |
| Mention-only (control) | 16.2% | −0.014 |

**Interpretation**: Merely mentioning the group destabilizes the model *more* than the directional nudge, but the directional content determines *which way* the model shifts. The mention alone triggers instability without providing a resolution direction.

---

### Experiment 6 — Logit Shift Analysis

**Question**: What happens to the logit margin after nudging, for backfire vs. stayed cases?

**Results**:

| Group | Baseline margin | Nudged margin | Shift |
|---|---|---|---|
| Strict backfire | +0.14 | −0.18 | **−0.32** |
| Stayed at target | +0.66 | +0.84 | **+0.18** |

The nudge pushes stayed cases *further* from the boundary (more confident) but pushes backfire cases *across* the boundary in the opposite direction.

---

### Experiment 7 — Layer Ablation (Gender_identity)

**Question**: Which layers are responsible for suppressing vs. amplifying stereotypes?

**Method**: Zeroed out each layer's contribution to the residual stream one at a time, measured change in stereotype score (`logit(stereo_slot) − logit(non_stereo_slot)`).

**Delta interpretation**: Positive delta = removing the layer *increases* stereotype score = that layer normally *suppresses* stereotypes.

**Key findings**:
- **Layer 15** (final): dominant stereotype **suppressor** (+0.145 delta) — removing it dramatically increases stereotype preference
- **Layer 6, 9, 10, 13**: stereotype **amplifiers** (negative delta) — removing them *reduces* stereotype preference

Pattern: stereotypes are built up in middle layers (6–13), then actively suppressed at the final layer (15).

---

### Experiment 8 — Layer Ablation (All Categories)

**Question**: Is the layer 15 suppressor pattern consistent across all BBQ categories?

**Categories tested**: Age, Disability_status, Gender_identity, Nationality, Physical_appearance, Race_ethnicity, Religion, SES, Sexual_orientation, Socioeconomic_status

**Key findings**:
- **Layer 0** (mean Δ +0.11) and **Layer 15** (mean Δ +0.04): most consistent suppressors
- **Layer 9** (mean Δ −0.07): strongest aggregate amplifier
- The pattern is **highly category-specific** — some layers are suppressors for one category and amplifiers for another
- Categories with insufficient metadata (`Age`, `Disability_status`, `Nationality`, `SES`) could not be resolved for stereotype-slot identification

---

### Experiment 9 — Logit Lens Analysis

**Question**: What token does the model's residual stream point toward at each intermediate layer, and how does this differ between backfire and stayed cases?

**Method**: At each layer, projected the residual stream at the last token position through `ln_final + unembed` to get log-probabilities over the full vocabulary.

#### Part A — Differential token analysis

Most differentially likely tokens at **Layer 14 and 15** for the **stayed group** vs. backfire:

| Layer | Stayed group tokens | Backfire group tokens |
|---|---|---|
| 14 | `' B'`, `' A'`, `' Both'`, `' both'` | uncertain/other |
| 15 | `' B'`, `' A'`, `' The'` | `' Could'`, `' Couldn'`, `' Think'`, `' Look'` |

**Interpretation**: The stayed group's residual stream at layers 14–15 is confidently pointing at answer tokens. The backfire group is pointing at *deliberation* and *uncertainty* tokens — the model is genuinely in an unresolved state just before generating its answer.

#### Part B — Stereotype-aware logit lens

Tracked `logP(stereotyped answer token) − logP(non-stereotyped answer token)` across layers, split by backfire subtype:

| Layer range | BF → stereo | BF → non-stereo | Stayed |
|---|---|---|---|
| L0–L5 (early) | strongly negative (−1 to −4) | strongly positive (+0.4 to +4) | near zero |
| L6–L14 (mid/late) | positive (+0.5 to +3) | strongly negative (−0.4 to −3.4) | near zero |
| L15 (final) | ~0 | ~0 | ~0 |

**Key observation**: The two backfire subtypes (flip-to-stereo, flip-from-stereo) are **mirror images** across layers. The stereotype-direction flip happens between layers 5 and 6. Layer 15 collapses both subtypes to near zero — consistent with it being the stereotype suppressor identified in ablation.

---

## 4. Consolidated Mechanistic Picture

```
Early layers (0–5)
  → Stereotype signal is established based on group mentions in the context
  → Backfire cases show strong early leaning in the "wrong" direction (opposite to their final output)

Middle layers (6–13)
  → Stereotype amplification occurs
  → Model builds up confident representations toward an answer
  → Backfire cases show the characteristic flip between L5 and L6

Layer 15 (final transformer block)
  → Dominant stereotype suppressor: actively cancels stereotype leaning
  → Collapsed stereotype margin for all groups

Last token residual stream at layer 14–15
  → Stayed cases: strongly committed to answer tokens (A or B)
  → Backfire cases: diffuse probability mass on uncertainty tokens (Could, Think, Both)
  → Backfire cases have ~5 log-prob units less probability on answer tokens than stayed cases
```

---

## 5. Open Questions / Possible Extensions

1. **Causal intervention**: Apply activation patching from stayed→backfire cases at specific layers to see if backfire can be *prevented* by patching in the stayed group's activations at layer 14.
2. **Steering vectors**: Use the difference between nudged and baseline activations as a steering vector; inject it at layer 6 (where the flip occurs) to see if it can suppress backfire.
3. **Cross-category generalization**: Test whether the same layer-6 flip and layer-15 suppression pattern holds across all BBQ categories with enough resolvable examples.
4. **Model scale**: Repeat the layer ablation on a larger model (e.g. Llama-3.2-3B) to see if stereotype suppression migrates to later layers as depth increases.
5. **True answer correlation**: In the 11% backfire cases, what fraction of flips happen to land on the *correct* BBQ answer? If high, backfire may actually be *bias correction* rather than random instability.
6. **Attention head attribution**: Identify which attention heads in layers 6 and 15 drive the stereotype-direction flip and the final suppression, using TransformerLens attention pattern analysis.

---

## 6. References

- [BBQ Dataset — Parrish et al. (2022)](https://arxiv.org/abs/2110.08193)
- [TransformerLens — Neel Nanda](https://github.com/TransformerLensOrg/TransformerLens)
- [PredictablyWeird/Choices — Nudge experiment design](https://github.com/PredictablyWeird/Choices)
- [Activation Differences as Steering Vectors — arXiv:2510.13900](https://arxiv.org/abs/2510.13900)
- [Logit Lens — nostalgebraist](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
