"""CPU analysis — which layers/components mediate assertiveness-ladder compliance?

Requires:
  cache/<model>/ladder_acts_<Category>.npz   (from collect_ladder_acts.py)
  cache/<model>/sensitivity_<Category>.npz   (from nudge_sensitivity.py)
  cache/<model>/unembed.npz

Analyses (nudge → stereotyped group, peak at t3):
  1. Assertiveness-level probe  — linear readout of ladder level per layer
  2. Residual patch sweep       — undo nudge delta at layer L → compliance drop
  3. Logit lens                 — when does P(stereo) cross 50% at t3?
  4. DLA                        — marginal logit contribution toward stereo per layer
  5. Delta norm                 — ||act(t3) − act(base)|| per layer
  6. MLP/attn patch at t3       — if component caches present

Example:
    uv run python scripts/analyze_assertiveness_circuits.py \\
        --category Gender_identity
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mech_interp_bbq.hf_backend import project_resid_to_abc
from mech_interp_bbq.prompts import model_cache_dir

T3_IDX = 2  # t3 = index 2 in ladder_stereo (0-based)


def _load(cache_dir: Path, model: str, category: str) -> dict:
    mdir = model_cache_dir(cache_dir, model)
    acts_f = mdir / f"ladder_acts_{category}.npz"
    sens_glob = glob.glob(str(mdir / f"sensitivity_{category}.npz"))
    if not acts_f.exists():
        raise SystemExit(f"Missing {acts_f} — run collect_ladder_acts.py on GPU first.")
    if not sens_glob:
        raise SystemExit(f"Missing sensitivity cache for {category}")
    unembed_f = mdir / "unembed.npz"
    if not unembed_f.exists():
        raise SystemExit(f"Missing {unembed_f}")

    acts = np.load(acts_f, mmap_mode="r")
    sens = np.load(sens_glob[0])
    um = np.load(unembed_f)
    return {
        "base_acts": acts["base_acts"],
        "ladder_acts": acts["ladder_stereo_acts"],
        "base_logits": sens["base_logits"],
        "ladder_logits": sens["ladder_stereo"],
        "stereo_ids": sens["stereo_ids"].astype(int),
        "unknown_ids": sens["unknown_ids"].astype(int),
        "other_ids": sens["other_ids"].astype(int),
        "unembed": {
            "abc_unembed": um["abc_unembed"],
            "norm_weight": um["norm_weight"],
            "norm_eps": float(um["norm_eps"]),
        },
        "base_mlp": acts["base_mlp"] if "base_mlp" in acts else None,
        "base_attn": acts["base_attn"] if "base_attn" in acts else None,
        "t3_mlp": acts["t3_stereo_mlp"] if "t3_stereo_mlp" in acts else None,
        "t3_attn": acts["t3_stereo_attn"] if "t3_stereo_attn" in acts else None,
    }


def _proj(resid: np.ndarray, um: dict) -> np.ndarray:
    return project_resid_to_abc(resid, um["norm_weight"], um["norm_eps"], um["abc_unembed"])


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def compliance_at_level(logits: np.ndarray, stereo_ids: np.ndarray, level: int) -> np.ndarray:
    """Boolean mask: argmax == stereotyped group at ladder level."""
    return logits[:, level, :].argmax(1) == stereo_ids


def assertiveness_probe(c: dict, max_examples: int = 500, n_splits: int = 3) -> dict:
    """Multiclass probe: predict ladder level (1–6) from activations at each layer."""
    ladder = c["ladder_acts"]  # (n, 6, L, d)
    n, n_levels, n_layers, d = ladder.shape
    if max_examples < n:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, max_examples, replace=False)
        ladder = ladder[idx]
        n = max_examples
    rows = []
    for layer in range(n_layers):
        X = ladder[:, :, layer, :].reshape(n * n_levels, d)
        y = np.tile(np.arange(1, n_levels + 1), n)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.1))
        acc = float(cross_val_score(clf, X, y, cv=n_splits, scoring="accuracy").mean())
        rows.append({"layer": layer, "accuracy": acc})
    best = max(rows, key=lambda r: r["accuracy"])
    return {"per_layer": rows, "best_layer": best["layer"], "best_accuracy": best["accuracy"]}


def assertiveness_regression(c: dict, max_examples: int = 500, n_splits: int = 3) -> dict:
    """Ridge: predict continuous ladder level from activation delta (relative to base)."""
    base, ladder = c["base_acts"], c["ladder_acts"]
    n, n_levels, n_layers, _ = ladder.shape
    if max_examples < n:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, max_examples, replace=False)
        base, ladder = base[idx], ladder[idx]
        n = max_examples
    rows = []
    for layer in range(n_layers):
        delta = ladder[:, :, layer, :] - base[:, layer, :][:, None, :]
        X = delta.reshape(n * n_levels, -1)
        y = np.tile(np.arange(1, n_levels + 1, dtype=np.float64), n)
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        r2 = float(cross_val_score(pipe, X, y, cv=n_splits, scoring="r2").mean())
        rows.append({"layer": layer, "r2": r2})
    best = max(rows, key=lambda r: r["r2"])
    return {"per_layer": rows, "best_layer": best["layer"], "best_r2": best["r2"]}


def residual_patch_sweep(c: dict, level: int = T3_IDX) -> list[dict]:
    """At each layer L, patch t3 final resid by removing (ladder-base) delta at L."""
    um = c["unembed"]
    sids = c["stereo_ids"]
    base, ladder = c["base_acts"], c["ladder_acts"]
    final = ladder[:, level, -1, :]
    full_comply = int(compliance_at_level(c["ladder_logits"], sids, level).sum())
    rows = []
    for L in range(base.shape[1]):
        patched = final - (ladder[:, level, L, :] - base[:, L, :])
        pred = _proj(patched, um).argmax(1)
        comply = int((pred == sids).sum())
        rows.append({
            "layer": L,
            "compliance": comply,
            "delta_vs_full": comply - full_comply,
        })
    return rows


def component_patch_sweep(c: dict, comp: str, level: int = T3_IDX) -> list[dict]:
    """Linear patch: replace t3 component at L with baseline component."""
    base_key = f"base_{comp}"
    t3_key = f"t3_{comp}" if comp != "stereo" else f"t3_stereo_{comp}"
    base_c = c["base_mlp"] if comp == "mlp" else c["base_attn"]
    t3_c = c["t3_mlp"] if comp == "mlp" else c["t3_attn"]
    if base_c is None or t3_c is None:
        return []

    um = c["unembed"]
    sids = c["stereo_ids"]
    final = c["ladder_acts"][:, level, -1, :]
    full_comply = int(compliance_at_level(c["ladder_logits"], sids, level).sum())
    rows = []
    for L in range(base_c.shape[1]):
        patched = final + (base_c[:, L, :] - t3_c[:, L, :])
        pred = _proj(patched, um).argmax(1)
        comply = int((pred == sids).sum())
        rows.append({
            "layer": L,
            "compliance": comply,
            "delta_vs_full": comply - full_comply,
        })
    return rows


def logit_lens(c: dict, level: int = T3_IDX, mask: np.ndarray | None = None) -> list[dict]:
    """Per-layer lens logits at t3; optional subset mask."""
    um = c["unembed"]
    sids = c["stereo_ids"]
    uids = c["unknown_ids"]
    acts = c["ladder_acts"][:, level, :, :]
    idx = np.arange(len(sids)) if mask is None else np.where(mask)[0]
    if len(idx) == 0:
        return []
    s, u = sids[idx], uids[idx]
    rows = []
    for L in range(acts.shape[1]):
        abc = _proj(acts[idx, L, :], um)
        p = _softmax(abc)
        rows.append({
            "layer": L,
            "P_stereo": float(p[np.arange(len(idx)), s].mean()),
            "P_unknown": float(p[np.arange(len(idx)), u].mean()),
            "frac_argmax_stereo": float((abc.argmax(1) == s).mean()),
        })
    return rows


def dla_stereo(c: dict, level: int = T3_IDX) -> list[dict]:
    """Marginal DLA toward stereo logit from cumulative resid at each layer (t3)."""
    um = c["unembed"]
    sids = c["stereo_ids"]
    ladder = c["ladder_acts"][:, level, :, :]  # (n, L, d)
    idx = np.arange(len(sids))
    rows = []
    prev = None
    for L in range(ladder.shape[1]):
        abc = _proj(ladder[:, L, :], um)
        toward_stereo = abc[idx, sids]
        marginal = toward_stereo if prev is None else toward_stereo - prev
        prev = toward_stereo
        rows.append({
            "layer": L,
            "mean_logit_stereo": float(toward_stereo.mean()),
            "mean_marginal": float(marginal.mean()),
        })
    return rows


def delta_norm(c: dict, level: int = T3_IDX) -> list[dict]:
    base, ladder = c["base_acts"], c["ladder_acts"]
    return [
        {"layer": L, "mean_l2": float(np.linalg.norm(
            ladder[:, level, L, :] - base[:, L, :], axis=1
        ).mean())}
        for L in range(base.shape[1])
    ]


def plot_results(c: dict, results: dict, out_dir: Path, stem: str) -> None:
    n_layers = c["base_acts"].shape[1]
    layers = np.arange(n_layers)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # 1. Assertiveness probe
    ax = axes[0, 0]
    probe = results["assertiveness_probe"]["per_layer"]
    ax.plot(layers, [r["accuracy"] for r in probe], "o-", color="C0")
    ax.axhline(1 / 6, color="grey", ls=":", label="chance (1/6)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("CV accuracy")
    ax.set_title(f"Assertiveness level probe\n(best L{results['assertiveness_probe']['best_layer']})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 2. Residual patch sweep
    ax = axes[0, 1]
    patch = results["residual_patch_t3"]
    ax.plot(layers, [r["compliance"] for r in patch], "o-", color="C1")
    full = compliance_at_level(c["ladder_logits"], c["stereo_ids"], T3_IDX).sum()
    ax.axhline(full, color="grey", ls="--", label=f"full t3 ({full})")
    ax.set_xlabel("Layer patched")
    ax.set_ylabel("# complying at t3")
    ax.set_title("Residual patch sweep (undo nudge at L)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 3. Logit lens — all examples at t3
    ax = axes[0, 2]
    lens = results["logit_lens_t3_all"]
    ax.plot(layers, [r["P_stereo"] for r in lens], "o-", label="P(stereo)", color="C0")
    ax.plot(layers, [r["P_unknown"] for r in lens], "s-", label="P(unknown)", color="C2")
    ax.axhline(0.5, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean probability")
    ax.set_title("Logit lens at t3 (all examples)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 4. DLA marginal
    ax = axes[1, 0]
    dla = results["dla_stereo_t3"]
    ax.bar(layers, [r["mean_marginal"] for r in dla], color="steelblue", width=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Marginal Δ logit(stereo)")
    ax.set_title("DLA toward stereo at t3")
    ax.grid(alpha=0.3, axis="y")

    # 5. Delta norm
    ax = axes[1, 1]
    dn = results["delta_norm_t3"]
    ax.plot(layers, [r["mean_l2"] for r in dn], "o-", color="C4")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean L2 ||Δact||")
    ax.set_title("Activation change base → t3")
    ax.grid(alpha=0.3)

    # 6. Component patches or compliance curve
    ax = axes[1, 2]
    if results.get("mlp_patch_t3"):
        ax.plot(layers, [r["compliance"] for r in results["mlp_patch_t3"]], "o-", label="MLP patch")
    if results.get("attn_patch_t3"):
        ax.plot(layers, [r["compliance"] for r in results["attn_patch_t3"]], "s-", label="Attn patch")
    if results.get("mlp_patch_t3") or results.get("attn_patch_t3"):
        ax.axhline(full, color="grey", ls="--", label=f"full t3")
        ax.set_xlabel("Layer patched")
        ax.set_ylabel("# complying")
        ax.set_title("Component patch at t3")
        ax.legend(fontsize=8)
    else:
        lvls = np.arange(7)
        comp_rates = [
            compliance_at_level(c["ladder_logits"], c["stereo_ids"], t - 1).mean() if t > 0
            else (c["base_logits"].argmax(1) == c["stereo_ids"]).mean()
            for t in lvls
        ]
        ax.plot(lvls, comp_rates, "o-", color="C0")
        ax.set_xticks(lvls)
        ax.set_xticklabels(["t0", "t1", "t2", "t3", "t4", "t5", "t6"])
        ax.set_xlabel("Ladder level")
        ax.set_ylabel("Compliance rate")
        ax.set_title("Compliance curve (from logits cache)")
    ax.grid(alpha=0.3)

    fig.suptitle(stem.replace("_", " "), fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"assertiveness_circuits_{stem}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--category", default="Gender_identity")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--max-probe-examples", type=int, default=500)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading caches for {args.category}...")
    c = _load(cache_dir, args.model, args.category)
    n = len(c["stereo_ids"])
    print(f"  n={n}, layers={c['base_acts'].shape[1]}")

    t3_comply = compliance_at_level(c["ladder_logits"], c["stereo_ids"], T3_IDX)
    print(f"  t3 compliance: {t3_comply.mean():.3f} ({t3_comply.sum()} examples)")

    print("Running assertiveness probe...")
    probe = assertiveness_probe(c, max_examples=args.max_probe_examples)
    reg = assertiveness_regression(c, max_examples=args.max_probe_examples)
    print(f"  best probe layer L{probe['best_layer']} acc={probe['best_accuracy']:.3f}")
    print(f"  best regression layer L{reg['best_layer']} R²={reg['best_r2']:.3f}")

    print("Running residual patch sweep at t3...")
    patch = residual_patch_sweep(c, T3_IDX)
    worst = min(patch, key=lambda r: r["compliance"])
    print(f"  worst patch L{worst['layer']}: compliance={worst['compliance']} "
          f"(Δ={worst['delta_vs_full']:+d})")

    comply_mask = t3_comply
    lens_all = logit_lens(c, T3_IDX, mask=None)
    lens_comply = logit_lens(c, T3_IDX, mask=comply_mask)
    dla = dla_stereo(c, T3_IDX)
    dn = delta_norm(c, T3_IDX)
    mlp_patch = component_patch_sweep(c, "mlp", T3_IDX)
    attn_patch = component_patch_sweep(c, "attn", T3_IDX)

    # First layer where P(stereo) >= 0.5 in lens (all examples)
    maj_stereo = next((r["layer"] for r in lens_all if r["P_stereo"] >= 0.5), None)

    stem = f"{args.model.replace('/', '_')}_{args.category}"
    results = {
        "model": args.model,
        "category": args.category,
        "n": n,
        "t3_compliance_rate": float(t3_comply.mean()),
        "t3_compliance_n": int(t3_comply.sum()),
        "assertiveness_probe": probe,
        "assertiveness_regression": reg,
        "residual_patch_t3": patch,
        "mlp_patch_t3": mlp_patch,
        "attn_patch_t3": attn_patch,
        "logit_lens_t3_all": lens_all,
        "logit_lens_t3_compliers": lens_comply,
        "first_layer_P_stereo_50pct": maj_stereo,
        "dla_stereo_t3": dla,
        "delta_norm_t3": dn,
    }

    json_path = out_dir / f"assertiveness_circuits_{stem}.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {json_path}")

    plot_results(c, results, out_dir, stem)
    print(f"Wrote {out_dir / f'assertiveness_circuits_{stem}.png'}")

    # Text summary
    md_lines = [
        f"# Assertiveness circuits — {args.model} | {args.category}\n",
        f"- n = {n}, t3 compliance = {t3_comply.mean():.1%} ({t3_comply.sum()} examples)\n",
        "## Where is assertiveness encoded?\n",
        f"- Multiclass ladder probe peaks at **L{probe['best_layer']}** "
        f"(CV acc = {probe['best_accuracy']:.3f})\n",
        f"- Ridge regression on Δact peaks at **L{reg['best_layer']}** (R² = {reg['best_r2']:.3f})\n",
        f"- Logit lens: P(stereo) ≥ 50% from layer **{maj_stereo}**\n",
        "## Causal layers (residual patch at t3)\n",
        "Patching layer L removes the nudge-induced delta at L from the final residual.\n",
        "| layer | compliance | Δ vs full |",
        "|--:|--:|--:|",
    ]
    for r in sorted(patch, key=lambda x: x["compliance"])[:10]:
        md_lines.append(f"| {r['layer']} | {r['compliance']} | {r['delta_vs_full']:+d} |")
    md_lines.append("\n*(top 10 layers with largest compliance drop)*\n")

    md_path = out_dir / f"assertiveness_circuits_{stem}.md"
    md_path.write_text("\n".join(md_lines))
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
