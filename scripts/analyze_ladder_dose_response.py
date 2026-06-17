"""Population dose–response from sensitivity_<cat>.npz (CPU-only).

Reports per category and direction (stereo / other):
  * P(argmax = target) vs ladder t1–t6 and rep k∈{1,2,3,5,8}
  * mean P(target) from softmax
  * flip-ever rate and median first-flip level

Example:
    uv run python scripts/analyze_ladder_dose_response.py \\
        --categories Gender_identity SES Race_ethnicity
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mech_interp_bbq.prompts import model_cache_dir

DEFAULT_CATS = ("Gender_identity", "SES", "Race_ethnicity")
LADDER_LABELS = ["t1", "t2", "t3", "t4", "t5", "t6"]


def _softmax3(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _flip_threshold(argmax_levels: np.ndarray, base_arg: np.ndarray, target: np.ndarray) -> np.ndarray:
    """First 1-based level where argmax==target; 0 if already; inf if never."""
    n, t = argmax_levels.shape
    out = np.full(n, np.inf)
    out[base_arg == target] = 0
    for i in range(t):
        hit = (argmax_levels[:, i] == target) & ~np.isfinite(out)
        out[hit] = i + 1
    return out


def _axis_summary(
    name: str,
    levels_logits: np.ndarray,
    base_logits: np.ndarray,
    target_ids: np.ndarray,
    mask: np.ndarray,
    level_labels: list[str],
) -> dict:
    """levels_logits: (n, T, 3)."""
    base_logits = base_logits[mask]
    levels_logits = levels_logits[mask]
    target_ids = target_ids[mask]
    n, t, _ = levels_logits.shape

    base_arg = base_logits.argmax(1)
    arg = levels_logits.argmax(2)
    p = _softmax3(levels_logits)
    rows = np.arange(n)
    p_tgt = p[rows[:, None], np.arange(t)[None, :], target_ids[:, None]]
    p_tgt_base = _softmax3(base_logits)[rows, target_ids]

    p_argmax = [(arg[:, i] == target_ids).mean() for i in range(t)]
    mean_p = [float(p_tgt[:, i].mean()) for i in range(t)]
    thr = _flip_threshold(arg, base_arg, target_ids)
    ever = np.isfinite(thr)
    flippers = ever & (thr > 0)

    return {
        "axis": name,
        "n": int(n),
        "level_labels": level_labels,
        "baseline": {
            "p_argmax_target": float((base_arg == target_ids).mean()),
            "mean_p_target": float(p_tgt_base.mean()),
        },
        "levels": [
            {
                "label": level_labels[i],
                "p_argmax_target": float(p_argmax[i]),
                "mean_p_target": mean_p[i],
                "delta_mean_p_vs_base": mean_p[i] - float(p_tgt_base.mean()),
            }
            for i in range(t)
        ],
        "flip_ever_rate": float(ever.mean()),
        "median_flip_level": float(np.median(thr[flippers])) if flippers.any() else None,
        "mean_delta_p_base_to_max": float((p_tgt[:, -1] - p_tgt_base).mean()),
    }


def analyze_category(cache_dir: Path, category: str) -> dict | None:
    path = cache_dir / f"sensitivity_{category}.npz"
    if not path.exists():
        print(f"[skip] {path}")
        return None

    d = np.load(path)
    base = d["base_logits"]
    sids, uids, oids = d["stereo_ids"], d["unknown_ids"], d["other_ids"]
    has_other = d["has_other"].astype(bool)
    n = len(base)
    rep_ks = [int(k) for k in d["rep_ks"].tolist()]
    rep_labels = [f"k{k}" for k in rep_ks]

    out: dict = {
        "category": category,
        "n": n,
        "n_has_other_tag": int(has_other.sum()),
        "ladder_templates": [str(x) for x in d["ladder_levels"].tolist()],
        "rep_ks": rep_ks,
        "directions": {},
    }

    for direction in ("stereo", "other"):
        mask = has_other if direction == "other" else np.ones(n, dtype=bool)
        tgt = sids if direction == "stereo" else oids
        out["directions"][direction] = {
            "ladder": _axis_summary(
                "ladder",
                d[f"ladder_{direction}"],
                base,
                tgt,
                mask,
                LADDER_LABELS,
            ),
            "rep": _axis_summary(
                "rep",
                d[f"rep_{direction}"],
                base,
                tgt,
                mask,
                rep_labels,
            ),
        }

    return out


def _print_table(cat_result: dict) -> None:
    cat = cat_result["category"]
    print(f"\n{'=' * 72}\n{cat}  (n={cat_result['n']})\n{'=' * 72}")
    for direction in ("stereo", "other"):
        d = cat_result["directions"][direction]
        print(f"\n--- toward {direction} (n={d['ladder']['n']}) ---")
        for axis in ("ladder", "rep"):
            s = d[axis]
            labels = s["level_labels"]
            print(f"\n  [{axis.upper()}]")
            print(f"  {'level':<6} {'P(argmax)':>10} {'mean P(tgt)':>12} {'ΔP vs base':>10}")
            print(f"  {'base':<6} {s['baseline']['p_argmax_target']:>10.3f} "
                  f"{s['baseline']['mean_p_target']:>12.3f}")
            for row in s["levels"]:
                print(
                    f"  {row['label']:<6} {row['p_argmax_target']:>10.3f} "
                    f"{row['mean_p_target']:>12.3f} {row['delta_mean_p_vs_base']:>+10.3f}"
                )
            med = s["median_flip_level"]
            med_s = f"{med:.1f}" if med is not None else "—"
            print(
                f"  flip-ever={s['flip_ever_rate']:.3f}  "
                f"median flip level={med_s}  "
                f"ΔP base→max={s['mean_delta_p_base_to_max']:+.3f}"
            )


def _plot(all_results: dict, out_dir: Path, model: str) -> None:
    cats = list(all_results.keys())
    fig, axes = plt.subplots(len(cats), 2, figsize=(12, 3.5 * len(cats)), sharey="row")
    if len(cats) == 1:
        axes = np.array([axes])

    for row, cat in enumerate(cats):
        res = all_results[cat]
        for col, (axis, title) in enumerate([("ladder", "Assertiveness ladder"), ("rep", "Repetition")]):
            ax = axes[row, col]
            for direction, marker, color in [
                ("stereo", "o-", "C0"),
                ("other", "s--", "C1"),
            ]:
                s = res["directions"][direction][axis]
                y = [lv["p_argmax_target"] for lv in s["levels"]]
                x = np.arange(1, len(y) + 1)
                ax.plot(x, y, marker, color=color, lw=2, ms=6, label=f"→ {direction}")
            ax.axhline(
                res["directions"]["stereo"][axis]["baseline"]["p_argmax_target"],
                color="C0",
                ls=":",
                alpha=0.4,
            )
            labels = res["directions"]["stereo"][axis]["level_labels"]
            ax.set_xticks(np.arange(1, len(labels) + 1))
            ax.set_xticklabels(labels)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
            ax.set_title(f"{cat.replace('_', ' ')} · {title}")
            if col == 0:
                ax.set_ylabel("P(argmax = target)")
            if row == 0 and col == 1:
                ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"{model} · dose–response (flip rate)", y=1.01)
    fig.tight_layout()
    stem = f"ladder_dose_response_{model.replace('/', '_')}"
    fig.savefig(out_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # mean P(target) panel
    fig, axes = plt.subplots(len(cats), 2, figsize=(12, 3.5 * len(cats)), sharey="row")
    if len(cats) == 1:
        axes = np.array([axes])
    for row, cat in enumerate(cats):
        res = all_results[cat]
        for col, axis in enumerate(("ladder", "rep")):
            ax = axes[row, col]
            for direction, marker, color in [("stereo", "o-", "C0"), ("other", "s--", "C1")]:
                s = res["directions"][direction][axis]
                y = [lv["mean_p_target"] for lv in s["levels"]]
                x = np.arange(1, len(y) + 1)
                ax.plot(x, y, marker, color=color, lw=2, ms=6, label=f"→ {direction}")
            labels = res["directions"]["stereo"][axis]["level_labels"]
            ax.set_xticks(np.arange(1, len(labels) + 1))
            ax.set_xticklabels(labels)
            ax.grid(alpha=0.3)
            if row == len(cats) - 1:
                ax.set_xlabel("Level")
            if col == 0:
                ax.set_ylabel("mean P(target)")
    fig.suptitle(f"{model} · dose–response (mean softmax prob)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_mean_p.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--categories", nargs="+", default=list(DEFAULT_CATS))
    args = ap.parse_args()

    cache_dir = model_cache_dir(args.cache_dir, args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict = {}
    for cat in args.categories:
        res = analyze_category(cache_dir, cat)
        if res is None:
            continue
        all_results[cat] = res
        _print_table(res)

    if not all_results:
        raise SystemExit("No sensitivity caches found.")

    payload = {"model": args.model, "categories": all_results}
    stem = f"ladder_dose_response_{args.model.replace('/', '_')}"
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2))
    _plot(all_results, out_dir, args.model)
    print(f"\nWrote {json_path}")
    print(f"Wrote {out_dir}/{stem}.png")
    print(f"Wrote {out_dir}/{stem}_mean_p.png")


if __name__ == "__main__":
    main()
