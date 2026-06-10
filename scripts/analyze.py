"""CPU PHASE — run all analyses on cached tensors. No GPU required.

Consumes the caches written by ``collect_cache.py`` and reproduces the analysis
suite developed for Llama-3.2-1B, now model-agnostic:

  * backfire / eligible counts
  * baseline margin analysis (backfire vs comply)
  * temperature flip-rate (Monte-Carlo sampling from cached logits)
  * layer-ablation sweep            (resid delta subtraction → A/B/C projection)
  * residual-stream patching sweep  (undo nudge divergence per layer)
  * component patch sweep           (mlp / attn, from *_components.npz)
  * DLA + logit lens                (per-layer marginal contribution to the flip)

Projection to A/B/C logits uses ``<model>_unembed.npz`` (final RMSNorm + 3 unembed
columns), so this script never loads the full model.

Example:
    uv run python scripts/analyze.py --model Qwen/Qwen3-32B \
        --category Gender_identity --nudge user_preference
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from mech_interp_bbq.hf_backend import project_resid_to_abc
from mech_interp_bbq.prompts import model_cache_dir

# ── cache loading ─────────────────────────────────────────────────────────────

def _resolve_dir(cache_dir: Path, model: str, category: str, nudge: str, condition: str) -> Path:
    """Prefer the per-model subdir; fall back to the flat cache dir for legacy layouts."""
    mdir = model_cache_dir(cache_dir, model)
    pat = f"3choice_{model.replace('/', '_')}_{category}_{nudge}_{condition}_n*_logits.npz"
    if glob.glob(str(mdir / pat)):
        return mdir
    return cache_dir


def _stem(search_dir: Path, model: str, category: str, nudge: str, condition: str) -> str | None:
    """Find the logits cache and return its full stem (so acts/components share the same n)."""
    pat = f"3choice_{model.replace('/', '_')}_{category}_{nudge}_{condition}_n*_logits.npz"
    hits = sorted(glob.glob(str(search_dir / pat)),
                  key=lambda p: int(p.split("_n")[-1].split("_")[0]))
    if not hits:
        return None
    return Path(hits[-1]).name[: -len("_logits.npz")]


def load_caches(cache_dir: Path, model: str, category: str, nudge: str, condition: str) -> dict:
    search_dir = _resolve_dir(cache_dir, model, category, nudge, condition)
    stem = _stem(search_dir, model, category, nudge, condition)
    if stem is None:
        logits_f = None
    else:
        logits_f = str(search_dir / f"{stem}_logits.npz")
    acts_p = search_dir / f"{stem}_acts.npz" if stem else None
    comp_p = search_dir / f"{stem}_components.npz" if stem else None
    acts_f = str(acts_p) if acts_p and acts_p.exists() else None
    comp_f = str(comp_p) if comp_p and comp_p.exists() else None
    # unembed: per-model-dir name first, then legacy flat name.
    unembed_f = search_dir / "unembed.npz"
    if not unembed_f.exists():
        unembed_f = cache_dir / f"{model.replace('/', '_')}_unembed.npz"
    if logits_f is None:
        raise SystemExit(f"No logits cache for {model}/{category}/{nudge}/{condition} in {cache_dir}")

    ld = np.load(logits_f)
    out = {
        "logits_file": logits_f,
        "s_ids": ld["stereo_ids"].astype(int),
        "o_ids": ld["other_ids"].astype(int),
        "u_ids": ld["unknown_ids"].astype(int),
        "base_logits": ld["base_logits"],
        "ns_logits": ld["nudged_logits"],
        "no_logits": ld["nudged_other_logits"],
        "has_other": ld["has_other_tag"] if "has_other_tag" in ld else None,
    }
    if acts_f:
        ad = np.load(acts_f)
        out["base_acts"] = ad["base_acts"]
        out["ns_acts"] = ad["nudge_stereo_acts"]
        out["no_acts"] = ad["nudge_other_acts"]
    if comp_f:
        out["comp"] = dict(np.load(comp_f))
    if unembed_f.exists():
        um = np.load(unembed_f)
        out["unembed"] = {
            "abc_unembed": um["abc_unembed"],
            "norm_weight": um["norm_weight"],
            "norm_eps": float(um["norm_eps"]),
        }
    return out


def _proj(resid: np.ndarray, um: dict) -> np.ndarray:
    return project_resid_to_abc(resid, um["norm_weight"], um["norm_eps"], um["abc_unembed"])


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ── analyses ──────────────────────────────────────────────────────────────────

def backfire_summary(c: dict) -> dict:
    s, o = c["s_ids"], c["o_ids"]
    bp, sp, op = c["base_logits"].argmax(1), c["ns_logits"].argmax(1), c["no_logits"].argmax(1)
    elig_s, elig_o = (bp == s), (bp == o)
    bf_s = int((elig_s & (sp == o)).sum())
    bf_o = int((elig_o & (op == s)).sum())
    return {
        "n": int(len(bp)),
        "elig_stereo": int(elig_s.sum()), "elig_other": int(elig_o.sum()),
        "bf_from_stereo": bf_s, "bf_from_other": bf_o, "total_backfire": bf_s + bf_o,
        "rate_stereo": bf_s / max(int(elig_s.sum()), 1),
        "rate_other": bf_o / max(int(elig_o.sum()), 1),
    }


def margin_analysis(c: dict) -> dict:
    s, o = c["s_ids"], c["o_ids"]
    bp, sp, op = c["base_logits"].argmax(1), c["ns_logits"].argmax(1), c["no_logits"].argmax(1)
    probs = _softmax(c["base_logits"])
    sp_sorted = np.sort(probs, axis=1)
    margin = sp_sorted[:, -1] - sp_sorted[:, -2]
    elig = (bp == s) | (bp == o)
    bf = ((bp == s) & (sp == o)) | ((bp == o) & (op == s))
    comply = ((bp == s) & (sp == s)) | ((bp == o) & (op == o))
    return {
        "margin_all_eligible": float(margin[elig].mean()) if elig.any() else None,
        "margin_backfire": float(margin[bf].mean()) if bf.any() else None,
        "margin_comply": float(margin[comply].mean()) if comply.any() else None,
        "n_backfire": int(bf.sum()), "n_comply": int(comply.sum()),
    }


def flip_rate(c: dict, n_runs: int = 3, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for name, key in [("baseline", "base_logits"), ("nudge_stereo", "ns_logits"),
                      ("nudge_other", "no_logits")]:
        probs = _softmax(c[key])
        n = len(probs)
        samples = np.stack(
            [[rng.choice(3, p=probs[i]) for i in range(n)] for _ in range(n_runs)], axis=1
        )
        any_diff = np.any(samples != samples[:, [0]], axis=1)
        out[name] = float(any_diff.mean())
    return out


def layer_ablation_sweep(c: dict) -> list[dict]:
    if "base_acts" not in c or "unembed" not in c:
        return []
    s, o = c["s_ids"], c["o_ids"]
    um = c["unembed"]
    bp = c["base_logits"].argmax(1)
    base, ns, no = c["base_acts"], c["ns_acts"], c["no_acts"]
    n_layers = base.shape[1]
    full_bf = backfire_summary(c)["total_backfire"]
    rows = []
    for L in range(1, n_layers):
        ns_abl = ns[:, -1, :] - (ns[:, L, :] - ns[:, L - 1, :])
        no_abl = no[:, -1, :] - (no[:, L, :] - no[:, L - 1, :])
        sp = _proj(ns_abl, um).argmax(1)
        op = _proj(no_abl, um).argmax(1)
        bf_s = int(((bp == s) & (sp == o)).sum())
        bf_o = int(((bp == o) & (op == s)).sum())
        rows.append({"layer": L, "bf_s": bf_s, "bf_o": bf_o, "total": bf_s + bf_o,
                     "delta_vs_full": (bf_s + bf_o) - full_bf})
    return rows


def residual_patch_sweep(c: dict) -> list[dict]:
    if "base_acts" not in c or "unembed" not in c:
        return []
    s, o = c["s_ids"], c["o_ids"]
    um = c["unembed"]
    bp = c["base_logits"].argmax(1)
    base, ns, no = c["base_acts"], c["ns_acts"], c["no_acts"]
    ns_final, no_final = ns[:, -1, :], no[:, -1, :]
    full_bf = backfire_summary(c)["total_backfire"]
    rows = []
    for L in range(base.shape[1]):
        sp = _proj(ns_final - (ns[:, L, :] - base[:, L, :]), um).argmax(1)
        op = _proj(no_final - (no[:, L, :] - base[:, L, :]), um).argmax(1)
        bf_s = int(((bp == s) & (sp == o)).sum())
        bf_o = int(((bp == o) & (op == s)).sum())
        rows.append({"layer": L, "bf_s": bf_s, "bf_o": bf_o, "total": bf_s + bf_o,
                     "delta_vs_full": (bf_s + bf_o) - full_bf})
    return rows


def component_patch_sweep(c: dict, comp: str) -> list[dict]:
    """Linear-approx patch of a component (mlp/attn) per layer, both directions."""
    if "comp" not in c or "base_acts" not in c or "unembed" not in c:
        return []
    s, o = c["s_ids"], c["o_ids"]
    um = c["unembed"]
    bp = c["base_logits"].argmax(1)
    ns_final, no_final = c["ns_acts"][:, -1, :], c["no_acts"][:, -1, :]
    cd = c["comp"]
    base_c = cd[f"base_{comp}"]
    ns_c = cd[f"nudge_stereo_{comp}"]
    no_c = cd[f"nudge_other_{comp}"]
    full_bf = backfire_summary(c)["total_backfire"]
    rows = []
    for L in range(base_c.shape[1]):
        sp = _proj(ns_final + (base_c[:, L, :] - ns_c[:, L, :]), um).argmax(1)
        op = _proj(no_final + (base_c[:, L, :] - no_c[:, L, :]), um).argmax(1)
        bf_s = int(((bp == s) & (sp == o)).sum())
        bf_o = int(((bp == o) & (op == s)).sum())
        rows.append({"layer": L, "bf_s": bf_s, "bf_o": bf_o, "total": bf_s + bf_o,
                     "delta_vs_full": (bf_s + bf_o) - full_bf})
    return rows


def dla_logit_lens(c: dict) -> dict:
    """Per-layer logit lens + marginal DLA toward the flip, for bf_s examples."""
    if "base_acts" not in c or "unembed" not in c:
        return {}
    s, o = c["s_ids"], c["o_ids"]
    um = c["unembed"]
    bp, sp = c["base_logits"].argmax(1), c["ns_logits"].argmax(1)
    bf_s = (bp == s) & (sp == o)
    if bf_s.sum() == 0:
        return {}
    ns = c["ns_acts"]
    n_layers = ns.shape[1]
    idx = np.where(bf_s)[0]
    s_sel, o_sel = s[idx], o[idx]
    rows = []
    prev = None
    for L in range(n_layers):
        abc = _proj(ns[idx, L, :], um)  # (m, 3)
        p = _softmax(abc)
        p_s = float(p[np.arange(len(idx)), s_sel].mean())
        p_o = float(p[np.arange(len(idx)), o_sel].mean())
        cum = abc[np.arange(len(idx)), o_sel] - abc[np.arange(len(idx)), s_sel]  # other-stereo
        marginal = float((cum - prev).mean()) if prev is not None else float(cum.mean())
        prev = cum
        rows.append({"layer": L, "P_stereo": p_s, "P_other": p_o, "marginal_flip": marginal})
    return {"n_bf_s": int(bf_s.sum()), "per_layer": rows}


# ── report ────────────────────────────────────────────────────────────────────

def _fmt_sweep(title: str, rows: list[dict]) -> str:
    if not rows:
        return f"\n### {title}\n_(no cache available)_\n"
    lines = [f"\n### {title}\n", "| layer | bf_s | bf_o | total | Δ vs full |", "|--:|--:|--:|--:|--:|"]
    for r in rows:
        lines.append(f"| {r['layer']} | {r['bf_s']} | {r['bf_o']} | {r['total']} | {r['delta_vs_full']:+d} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--category", default="Gender_identity")
    p.add_argument("--nudge", default="user_preference")
    p.add_argument("--condition", default="ambig")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--flip-runs", type=int, default=3)
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    c = load_caches(cache_dir, args.model, args.category, args.nudge, args.condition)

    results = {
        "model": args.model, "category": args.category, "nudge": args.nudge,
        "condition": args.condition,
        "backfire": backfire_summary(c),
        "margins": margin_analysis(c),
        "flip_rate": flip_rate(c, args.flip_runs),
        "layer_ablation": layer_ablation_sweep(c),
        "residual_patch": residual_patch_sweep(c),
        "mlp_patch": component_patch_sweep(c, "mlp"),
        "attn_patch": component_patch_sweep(c, "attn"),
        "dla_logit_lens": dla_logit_lens(c),
    }

    stem = f"{args.model.replace('/', '_')}_{args.category}_{args.nudge}_{args.condition}"
    json_path = out_dir / f"analysis_{stem}.json"
    json_path.write_text(json.dumps(results, indent=2))

    bf = results["backfire"]
    mg = results["margins"]
    fr = results["flip_rate"]
    md = [
        f"# Analysis — {args.model} | {args.category} | {args.nudge}\n",
        "## Backfire summary\n",
        f"- n = {bf['n']}, eligible stereo/other = {bf['elig_stereo']}/{bf['elig_other']}",
        f"- bf_from_stereo = {bf['bf_from_stereo']} ({bf['rate_stereo']*100:.1f}%)",
        f"- bf_from_other  = {bf['bf_from_other']} ({bf['rate_other']*100:.1f}%)",
        f"- **total backfire = {bf['total_backfire']}**\n",
        "## Baseline margin (top − 2nd choice)\n",
        f"- all eligible: {mg['margin_all_eligible']}",
        f"- backfire (n={mg['n_backfire']}): {mg['margin_backfire']}",
        f"- comply (n={mg['n_comply']}): {mg['margin_comply']}\n",
        "## Temperature=1 flip rate (re-sampling same prompt)\n",
        f"- baseline: {fr['baseline']*100:.1f}%  |  nudge→stereo: {fr['nudge_stereo']*100:.1f}%  "
        f"|  nudge→other: {fr['nudge_other']*100:.1f}%\n",
        _fmt_sweep("Layer ablation sweep (resid delta)", results["layer_ablation"]),
        _fmt_sweep("Residual-stream patch sweep", results["residual_patch"]),
        _fmt_sweep("MLP component patch sweep", results["mlp_patch"]),
        _fmt_sweep("Attention component patch sweep", results["attn_patch"]),
    ]
    md_path = out_dir / f"analysis_{stem}.md"
    md_path.write_text("\n".join(md))

    print(f"Backfire: bf_s={bf['bf_from_stereo']} bf_o={bf['bf_from_other']} "
          f"total={bf['total_backfire']} / n={bf['n']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
