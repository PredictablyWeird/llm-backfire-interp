"""CPU analysis — which BBQ prompt regions does the model attend to at Answer:?

Requires ``prompt_attn_<Category>.npz`` from ``collect_prompt_attention.py`` and
``sensitivity_<Category>.npz`` for compliance labels.

Example:
    uv run python scripts/analyze_prompt_attention.py \\
        --model Qwen/Qwen3-32B --categories Gender_identity SES Race_ethnicity
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mech_interp_bbq.prompt_regions import REGION_NAMES
from mech_interp_bbq.prompts import model_cache_dir

DEFAULT_CATS = ("Gender_identity", "SES", "Race_ethnicity")
LATE_LAYERS = (48, 52, 56, 59, 63)
T3_IDX = 2


def _rank_regions(mass: np.ndarray, region_names: list[str]) -> list[dict]:
    """mass: (n_regions,) mean fractions."""
    order = np.argsort(-mass)
    return [
        {"region": region_names[i], "mean_mass": float(mass[i]), "rank": r + 1}
        for r, i in enumerate(order)
    ]


def _layer_idx_for(layers: np.ndarray, target: int) -> int | None:
    hits = np.where(layers == target)[0]
    return int(hits[0]) if len(hits) else None


def _load_validation(results_dir: Path, model: str, category: str) -> dict:
    """Pull existing cache-only checks for interpretation."""
    stem = model.replace("/", "_")
    out: dict = {}

    abl_path = results_dir / f"baseline_ablation_{stem}_{category}.json"
    if abl_path.exists():
        abl = json.loads(abl_path.read_text())
        rows = abl["per_layer"]
        top_unknown = sorted(rows, key=lambda r: r["delta_margin_u_minus_s"])[:5]
        out["baseline_ablation"] = {
            "path": str(abl_path),
            "full_frac_unknown": abl["full_baseline"]["frac_argmax_unknown"],
            "layers_boosting_unknown": [
                {"layer": r["layer"], "delta_margin_u_minus_s": r["delta_margin_u_minus_s"]}
                for r in top_unknown
            ],
        }

    if category == "Gender_identity":
        circ_path = results_dir / f"assertiveness_circuits_{stem}_{category}.json"
        if circ_path.exists():
            circ = json.loads(circ_path.read_text())
            patch = circ.get("residual_patch_t3", [])
            full = circ.get("t3_compliance_n", None)
            if patch and full is not None:
                worst = min(patch, key=lambda r: r["compliance"])
                best_drop = min(patch, key=lambda r: r["delta_vs_full"])
                out["residual_patch_t3"] = {
                    "path": str(circ_path),
                    "full_compliance_n": full,
                    "worst_layer": worst,
                    "largest_drop": best_drop,
                }

    return out


def analyze_category(
    cache_dir: Path,
    results_dir: Path,
    model: str,
    category: str,
) -> dict | None:
    attn_path = cache_dir / f"prompt_attn_{category}.npz"
    sens_path = cache_dir / f"sensitivity_{category}.npz"
    if not attn_path.exists():
        print(f"[skip] {category}: missing {attn_path}")
        return None
    if not sens_path.exists():
        print(f"[skip] {category}: missing {sens_path}")
        return None

    attn = np.load(attn_path)
    sens = np.load(sens_path)
    region_names = [str(x) for x in attn["region_names"].tolist()]
    layers = attn["layers"].astype(int)
    sample_idx = attn["sample_indices"].astype(int)

    mass_base = attn["region_mass_baseline"]  # (n, n_store, 4)
    mass_t3 = attn["region_mass_t3"]
    n = mass_base.shape[0]
    final_li = mass_base.shape[1] - 1
    final_layer = int(layers[final_li])

    mean_base = mass_base.mean(axis=0)
    mean_t3 = mass_t3.mean(axis=0)
    delta_t3_minus_base = mean_t3 - mean_base

    ranking_base_final = _rank_regions(mean_base[final_li], region_names)
    ranking_t3_final = _rank_regions(mean_t3[final_li], region_names)

    late_idx = [_layer_idx_for(layers, L) for L in LATE_LAYERS]
    late_idx = [i for i in late_idx if i is not None]
    late_mean_base = mean_base[late_idx].mean(axis=0) if late_idx else mean_base[final_li]
    late_mean_t3 = mean_t3[late_idx].mean(axis=0) if late_idx else mean_t3[final_li]

    stereo = sens["stereo_ids"][sample_idx]
    comply = sens["ladder_stereo"][sample_idx, T3_IDX, :].argmax(1) == stereo
    nudge_i = region_names.index("nudge")
    ctx_i = region_names.index("context")

    comply_mask = comply
    nudge_comply = float(mass_t3[comply_mask, :, nudge_i].mean()) if comply_mask.any() else None
    nudge_not = float(mass_t3[~comply_mask, :, nudge_i].mean()) if (~comply_mask).any() else None
    ctx_comply = float(mass_t3[comply_mask, :, ctx_i].mean()) if comply_mask.any() else None
    ctx_not = float(mass_t3[~comply_mask, :, ctx_i].mean()) if (~comply_mask).any() else None

    late_layers_report = {}
    for L in LATE_LAYERS:
        li = _layer_idx_for(layers, L)
        if li is None:
            continue
        late_layers_report[f"L{L}"] = {
            "baseline": {region_names[r]: float(mean_base[li, r]) for r in range(len(region_names))},
            "t3": {region_names[r]: float(mean_t3[li, r]) for r in range(len(region_names))},
        }

    payload = {
        "model": model,
        "category": category,
        "n": int(n),
        "n_comply_t3": int(comply.sum()),
        "layers_stored": layers.tolist(),
        "final_layer": final_layer,
        "ranking_baseline_final": ranking_base_final,
        "ranking_t3_final": ranking_t3_final,
        "ranking_baseline_late_avg": _rank_regions(late_mean_base, region_names),
        "ranking_t3_late_avg": _rank_regions(late_mean_t3, region_names),
        "delta_t3_minus_base_final": {
            region_names[r]: float(delta_t3_minus_base[final_li, r]) for r in range(len(region_names))
        },
        "compliance_split_t3_mean_over_layers": {
            "nudge_compliers": nudge_comply,
            "nudge_non_compliers": nudge_not,
            "context_compliers": ctx_comply,
            "context_non_compliers": ctx_not,
        },
        "late_layers": late_layers_report,
        "validation": _load_validation(results_dir, model, category),
    }

    _plot_category(payload, mass_base, mass_t3, layers, region_names, results_dir, model, category)
    return payload


def _plot_category(
    payload: dict,
    mass_base: np.ndarray,
    mass_t3: np.ndarray,
    layers: np.ndarray,
    region_names: list[str],
    results_dir: Path,
    model: str,
    category: str,
) -> None:
    stem = f"prompt_attention_{model.replace('/', '_')}_{category}"
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    x = layers.astype(int)
    colors = ["C0", "C1", "C2", "C3"]

    for ax, mass, title in [
        (axes[0, 0], mass_base.mean(axis=0), "Baseline — mean region attention"),
        (axes[0, 1], mass_t3.mean(axis=0), "t3 stereo — mean region attention"),
    ]:
        for r, name in enumerate(region_names):
            ax.plot(x, mass[:, r], "o-", ms=3, color=colors[r], label=name)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Attention mass fraction")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    ax = axes[1, 0]
    final_li = mass_base.shape[1] - 1
    width = 0.35
    idx = np.arange(len(region_names))
    ax.bar(idx - width / 2, mass_base.mean(axis=0)[final_li], width, label="baseline")
    ax.bar(idx + width / 2, mass_t3.mean(axis=0)[final_li], width, label="t3")
    ax.set_xticks(idx)
    ax.set_xticklabels(region_names)
    ax.set_ylabel("Attention mass")
    ax.set_title(f"Final layer L{layers[final_li]} — region comparison")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    delta = mass_t3.mean(axis=0) - mass_base.mean(axis=0)
    for r, name in enumerate(region_names):
        ax.plot(x, delta[:, r], "o-", ms=3, color=colors[r], label=name)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Δ mass (t3 − baseline)")
    ax.set_title("Attention shift when nudge added")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Prompt-region attention — {category}", fontsize=11)
    fig.tight_layout()
    results_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(results_dir / f"{stem}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_CATS),
    )
    args = ap.parse_args()

    cache_dir = model_cache_dir(args.cache_dir, args.model)
    results_dir = Path(args.out_dir)
    per_cat: dict[str, dict] = {}

    for category in args.categories:
        print(f"\n=== {category} ===")
        result = analyze_category(cache_dir, results_dir, args.model, category)
        if result is None:
            continue
        per_cat[category] = result
        out_path = results_dir / f"prompt_attention_{args.model.replace('/', '_')}_{category}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"Top regions at t3 (final layer L{result['final_layer']}):")
        for row in result["ranking_t3_final"]:
            print(f"  #{row['rank']} {row['region']}: {row['mean_mass']:.3f}")
        split = result["compliance_split_t3_mean_over_layers"]
        print(
            f"Compliance split — nudge mass: compliers={split['nudge_compliers']:.3f}  "
            f"non={split['nudge_non_compliers']:.3f}"
        )
        print(f"Wrote {out_path}")

    summary_path = results_dir / "prompt_attention_3cats_summary.json"
    summary = {
        "model": args.model,
        "categories": {
            cat: {
                "n": data["n"],
                "top_t3_region": data["ranking_t3_final"][0]["region"],
                "top_t3_mass": data["ranking_t3_final"][0]["mean_mass"],
                "nudge_delta_final": data["delta_t3_minus_base_final"].get("nudge"),
                "nudge_compliers_vs_non": data["compliance_split_t3_mean_over_layers"],
            }
            for cat, data in per_cat.items()
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
