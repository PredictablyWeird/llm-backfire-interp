"""One-shot script: strict backfire activation analysis."""
import pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt

from mech_interp_bbq.activations import collect_model_predictions, collect_resid_post, load_model
from mech_interp_bbq.data import load_bbq_pairwise
from mech_interp_bbq.nudges import NUDGE_TEMPLATES, group_label, singular_group_label
from mech_interp_bbq.probes import train_layer_probe

examples = load_bbq_pairwise("Gender_identity", context_condition="ambig", max_examples=400)
target_tag, other_tag = examples[0].group_a_tag, examples[0].group_b_tag

tmpl = NUDGE_TEMPLATES["user_preference"]
gl = group_label("Gender_identity", target_tag)
og = group_label("Gender_identity", other_tag)
sg = singular_group_label("Gender_identity", target_tag)
sentence = "(" + tmpl.template.format(group_label=gl, other_group_label=og, singular_group_label=sg) + ")"

baseline_prompts = [ex.prompt_with_sentence() for ex in examples]
nudged_prompts   = [ex.prompt_with_sentence(sentence, position=tmpl.position) for ex in examples]

model = load_model("meta-llama/Llama-3.2-1B")

base_preds   = collect_model_predictions(model, baseline_prompts, n_choices=2, batch_size=4)
nudged_preds = collect_model_predictions(model, nudged_prompts,   n_choices=2, batch_size=4)

strict_bf = np.array([i for i in range(len(examples)) if base_preds[i] == 0 and nudged_preds[i] == 1])
stayed_a  = np.array([i for i in range(len(examples)) if base_preds[i] == 0 and nudged_preds[i] == 0])
print(f"Strict backfire (A->B): {len(strict_bf)}")
print(f"Stayed A        (A->A): {len(stayed_a)}")

print("Collecting baseline activations...")
base_acts   = collect_resid_post(model, baseline_prompts, batch_size=4).acts.numpy()
print("Collecting nudged activations...")
nudged_acts = collect_resid_post(model, nudged_prompts,   batch_size=4).acts.numpy()

delta    = nudged_acts - base_acts
n_layers = base_acts.shape[1]

coef_path = pathlib.Path("probes_out/bias_probe_meta-llama_Llama-3.2-1B_Gender_identity_coefs.npz")
saved = np.load(coef_path)
stereo_coefs = saved["stereotype_coef"]
stereo_dirs = []
for l in range(n_layers):
    c = stereo_coefs[l]
    w = c[0] - c[-1]
    w /= (np.linalg.norm(w) + 1e-12)
    stereo_dirs.append(w)

header = f"{'layer':>5}  {'cos(BF,SA)':>10}  {'L2_BF':>8}  {'L2_SA':>8}  {'align_BF':>10}  {'align_SA':>10}"
print()
print("=== Mean activation-change vectors: backfire vs stayed-A ===")
print(header)
print("-" * len(header))

cos_l, l2_bf_l, l2_sa_l, align_bf_l, align_sa_l = [], [], [], [], []

for l in range(n_layers):
    d = delta[:, l, :]
    mean_bf = d[strict_bf].mean(axis=0)
    mean_sa = d[stayed_a].mean(axis=0)
    cos = float(np.dot(mean_bf, mean_sa) / (np.linalg.norm(mean_bf) * np.linalg.norm(mean_sa) + 1e-12))
    l2_bf = float(np.linalg.norm(mean_bf))
    l2_sa = float(np.linalg.norm(mean_sa))
    ab = float(mean_bf @ stereo_dirs[l])
    as_ = float(mean_sa @ stereo_dirs[l])
    cos_l.append(cos); l2_bf_l.append(l2_bf); l2_sa_l.append(l2_sa)
    align_bf_l.append(ab); align_sa_l.append(as_)
    print(f"{l:>5}  {cos:>10.4f}  {l2_bf:>8.4f}  {l2_sa:>8.4f}  {ab:>10.4f}  {as_:>10.4f}")

# Per-layer probe: backfire vs stayed-A from nudged activations
print()
print("=== Probe: predict backfire vs stayed-A from nudged activations ===")
mask   = np.concatenate([strict_bf, stayed_a])
labels = np.array([1] * len(strict_bf) + [0] * len(stayed_a), dtype=np.int64)
acts_subset = torch.from_numpy(nudged_acts[mask])

majority = float((labels == 0).mean())
print(f"Majority baseline (predict stayed-A): {majority:.3f}")
probe_accs = []
best_layer, best_acc = 0, 0.0
for l in range(n_layers):
    r = train_layer_probe(acts_subset[:, l, :], labels, layer=l)
    probe_accs.append(r.mean_accuracy)
    marker = " <-- best" if r.mean_accuracy > best_acc else ""
    print(f"  layer {l:>2}: {r.mean_accuracy:.3f}{marker}")
    if r.mean_accuracy > best_acc:
        best_acc = r.mean_accuracy
        best_layer = l
print(f"Best: layer {best_layer}, acc={best_acc:.3f}")

# Also compare baseline activations of backfire vs stayed-A
# (before nudge — does the model's initial state predict future backfire?)
print()
print("=== Probe: predict backfire vs stayed-A from BASELINE activations ===")
acts_base_subset = torch.from_numpy(base_acts[mask])
best_base_layer, best_base_acc = 0, 0.0
base_probe_accs = []
for l in range(n_layers):
    r = train_layer_probe(acts_base_subset[:, l, :], labels, layer=l)
    base_probe_accs.append(r.mean_accuracy)
    marker = " <-- best" if r.mean_accuracy > best_base_acc else ""
    print(f"  layer {l:>2}: {r.mean_accuracy:.3f}{marker}")
    if r.mean_accuracy > best_base_acc:
        best_base_acc = r.mean_accuracy
        best_base_layer = l
print(f"Best: layer {best_base_layer}, acc={best_base_acc:.3f}")

# Plot
layers = list(range(n_layers))
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Top-left: cosine similarity of mean delta vectors
ax = axes[0][0]
ax.plot(layers, cos_l, marker="o", color="steelblue")
ax.set_ylim(0.85, 1.01)
ax.axhline(1, color="grey", linestyle=":", linewidth=0.8)
ax.set_xlabel("Layer"); ax.set_ylabel("Cosine similarity")
ax.set_title("Similarity of mean nudge-change vector\nBackfire vs Stayed-A")
ax.grid(alpha=0.3)

# Top-right: L2 magnitude
ax = axes[0][1]
ax.plot(layers, l2_bf_l, marker="o", color="firebrick", label=f"backfire (n={len(strict_bf)})")
ax.plot(layers, l2_sa_l, marker="s", color="seagreen",  label=f"stayed-A (n={len(stayed_a)})")
ax.set_xlabel("Layer"); ax.set_ylabel("L2 norm of mean delta")
ax.set_title("Magnitude of nudge-induced change")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Bottom-left: stereotype direction alignment
ax = axes[1][0]
ax.plot(layers, align_bf_l, marker="o", color="firebrick", label="backfire")
ax.plot(layers, align_sa_l, marker="s", color="seagreen",  label="stayed-A")
ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
ax.set_xlabel("Layer"); ax.set_ylabel("Alignment with stereotype direction")
ax.set_title("Does nudge push toward/away from\nstereotype direction?")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Bottom-right: probe accuracy from baseline vs nudged acts
ax = axes[1][1]
ax.plot(layers, probe_accs,      marker="o", color="darkorchid", label="nudged acts")
ax.plot(layers, base_probe_accs, marker="s", color="darkorange",  label="baseline acts")
ax.axhline(majority, color="grey", linestyle="-.", linewidth=0.8, label=f"majority ({majority:.2f})")
ax.set_xlabel("Layer"); ax.set_ylabel("5-fold CV accuracy")
ax.set_title("Probe: predict backfire from activations\n(nudged vs baseline)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle(
    "Llama-3.2-1B | Gender_identity | user_preference nudge\n"
    f"Strict backfire (A→B, n={len(strict_bf)}) vs Stayed-A (A→A, n={len(stayed_a)})",
    fontsize=11,
)
fig.tight_layout()
fig.savefig("probes_out/strict_backfire_activation_analysis.png", dpi=150)
print("\nSaved probes_out/strict_backfire_activation_analysis.png")
