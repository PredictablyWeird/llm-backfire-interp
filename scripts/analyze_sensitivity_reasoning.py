"""CPU analysis: compare direct vs reason-before-answer sensitivity caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mech_interp_bbq.prompts import model_cache_dir
from mech_interp_bbq.sensitivity import LADDER, REP_KS


def _compliance_rate(logits: np.ndarray, target_ids: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    return float((logits[mask].argmax(1) == target_ids[mask]).mean())


def _axis_summary(name: str, logits: np.ndarray, target_ids: np.ndarray, mask: np.ndarray) -> dict:
    rows = []
    for i in range(logits.shape[1]):
        if name == "ladder":
            level = LADDER[i]
        else:
            level = f"k={REP_KS[i]}"
        rows.append({
            "level": level,
            "compliance": _compliance_rate(logits, target_ids, mask),
        })
    return {"name": name, "levels": rows}


def _summarize_npz(d: np.lib.npyio.NpzFile, direction: str, mask: np.ndarray) -> dict:
    tgt = d["stereo_ids"]
    out = {
        "base_compliance": _compliance_rate(d["base_logits"], tgt, mask),
        "ladder": _axis_summary("ladder", d[f"ladder_{direction}"], tgt, mask),
    }
    rep_key = f"rep_{direction}"
    if rep_key in d:
        out["rep"] = _axis_summary("rep", d[rep_key], tgt, mask)
    return out


def analyze_category(cache_dir: Path, category: str) -> dict:
    direct_path = cache_dir / f"sensitivity_{category}.npz"
    reason_path = cache_dir / f"sensitivity_reasoning_{category}.npz"
    if not reason_path.exists():
        raise FileNotFoundError(reason_path)

    rd = np.load(reason_path, allow_pickle=True)
    n = int(rd["base_logits"].shape[0])
    stereo_mask = np.ones(n, dtype=bool)
    other_mask = rd["has_other"]

    out: dict = {
        "category": category,
        "n": n,
        "max_reasoning_tokens": int(rd["max_reasoning_tokens"]),
        "max_examples": int(rd["max_examples"]) if "max_examples" in rd else n,
        "ladder_only": bool(rd["ladder_only"]) if "ladder_only" in rd else False,
        "reasoning_instruction": str(rd["reasoning_instruction"]),
        "mean_reasoning_chars": {
            "base": float(np.mean([len(str(x)) for x in rd["reasoning_base"]])),
            "ladder_stereo": float(np.mean([len(str(x)) for x in rd["reasoning_ladder_stereo"].ravel()])),
            "ladder_other": float(np.mean([len(str(x)) for x in rd["reasoning_ladder_other"].ravel()])),
        },
        "reasoning": {
            "stereo": _summarize_npz(rd, "stereo", stereo_mask),
            "other": _summarize_npz(rd, "other", other_mask),
        },
        "direct": {},
    }

    if direct_path.exists():
        dd = np.load(direct_path)
        out["direct"] = {
            "stereo": _summarize_npz(dd, "stereo", stereo_mask),
            "other": _summarize_npz(dd, "other", other_mask),
        }

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--category", default="Gender_identity")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    cache_dir = model_cache_dir(args.cache_dir, args.model)
    summary = analyze_category(cache_dir, args.category)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sensitivity_reasoning_{args.category.replace('/', '_')}"
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(summary, indent=2))

    print(f"\n=== {args.category} reason-before-answer ===")
    print(f"mean reasoning chars (base): {summary['mean_reasoning_chars']['base']:.0f}")
    rs = summary["reasoning"]["stereo"]
    print(f"stereo base compliance (reasoning): {rs['base_compliance']:.3f}")
    print("ladder t1..t6:", [round(x["compliance"], 3) for x in rs["ladder"]["levels"]])
    if summary.get("direct"):
        ds = summary["direct"]["stereo"]
        print(f"stereo base compliance (direct):   {ds['base_compliance']:.3f}")
        print("direct ladder:", [round(x["compliance"], 3) for x in ds["ladder"]["levels"]])
    print(f"\n[save] {json_path}")


if __name__ == "__main__":
    main()
