"""
paired_bootstrap.py
====================
Implements the analysis docs/analysis-plan.md explicitly deferred and
findings_gps.txt Sec.9 item 2 flagged as "NOT YET DONE": a PAIRED bootstrap of
rho_rel(PE) - rho_rel(No-PE), matched by graph identity, instead of two separate
marginal CIs.

WHY THIS IS DIFFERENT FROM Table 3's CIs -- [PER-CELL, SEEDS POOLED] there computes
one CI for arm A and one for arm B and calls an effect real only if they don't
overlap. That throws away the fact that both arms are probed on the SAME 256 (or
634, for VOC) sampled test graphs: molecule-to-molecule variance is shared and
should cancel. Pairing computes, for each graph g common to both arms,
    diff(g) = rho_rel_PE(g) - rho_rel_none(g)
(each side first averaged over that arm's own seeds for graph g), then bootstraps
the MEAN of diff(g) by resampling graph identities. This is expected to be tighter
than Table 3's marginal comparison whenever cross-graph variance dominates --
exactly the case flagged for peptides-func, where nothing separated marginally.

    python scripts/paired_bootstrap.py --in-dir results_all --out results_all/paired_bootstrap.csv

Output columns are labelled [PAIRED, GRAPH-CLUSTERED BOOTSTRAP, SEEDS AVERAGED
WITHIN EACH ARM PER GRAPH BEFORE PAIRING] -- never averaged across backbones or
datasets.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from dataset_meta import DATASETS, REL_BINS, REL_RHO_WINDOW  # noqa: E402
from sensitivity import average_curves, long_range_fraction, to_relative_curve  # noqa: E402


def _as_curve(raw):
    out = {}
    for k, v in raw.items():
        out[int(k)] = dict(v) if isinstance(v, dict) else {"mean": float(v), "count": 1}
    return out


def load_records(results_dir):
    records = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            r = json.load(f)
        if not r.get("smoke_test"):
            records.append(r)
    return records


def per_graph_rel_curves(records, backbone, pe, dataset):
    """graph_id -> list of RELATIVE per-graph curves, one per seed that has it.

    Kept as a list (not yet averaged) so the caller decides how to combine seeds;
    here we average a graph's seed-copies into one curve before pairing, since the
    graph's TOPOLOGY (and hence its relative binning) is identical across seeds --
    only the trained model differs -- so seed-averaging first is a within-arm
    operation, not a cross-arm one, and does not leak information between PEs.
    """
    by_graph = defaultdict(list)
    for r in records:
        if r["backbone"] != backbone or r["pe"] != pe or r["dataset"] != dataset:
            continue
        for entry in r.get("sensitivity_curves_per_graph") or []:
            if not (isinstance(entry, dict) and "curve" in entry and entry.get("graph_id") is not None):
                continue
            diam = entry.get("diameter")
            if not diam:
                continue
            rel = to_relative_curve(_as_curve(entry["curve"]), diam, REL_BINS)
            if rel:
                by_graph[entry["graph_id"]].append(rel)
    return {g: average_curves(cs) for g, cs in by_graph.items()}


def paired_bootstrap_diff(none_by_graph, pe_by_graph, n_boot=1000, seed=0):
    """Bootstrap CI for mean_g [rho_rel(pe, g) - rho_rel(none, g)] over graphs
    common to both arms. Returns (n_paired, mean_diff, ci_lo, ci_hi)."""
    b_lo, b_hi = REL_RHO_WINDOW
    common = sorted(set(none_by_graph) & set(pe_by_graph))
    if len(common) < 2:
        return len(common), float("nan"), float("nan"), float("nan")
    diffs = []
    for g in common:
        r_pe = long_range_fraction(pe_by_graph[g], b_lo, b_hi)
        r_none = long_range_fraction(none_by_graph[g], b_lo, b_hi)
        if r_pe == r_pe and r_none == r_none:  # drop NaN (graph missing that bin)
            diffs.append(r_pe - r_none)
    if len(diffs) < 2:
        return len(diffs), float("nan"), float("nan"), float("nan")
    point = sum(diffs) / len(diffs)
    rng = torch.Generator().manual_seed(seed)
    n = len(diffs)
    vals = []
    for _ in range(n_boot):
        idx = torch.randint(n, (n,), generator=rng).tolist()
        vals.append(sum(diffs[i] for i in idx) / n)
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[min(len(vals) - 1, int(0.975 * len(vals)))]
    return len(diffs), point, lo, hi


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="results_all")
    ap.add_argument("--out", default="results_all/paired_bootstrap.csv")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    records = load_records(args.in_dir)
    cells = sorted({(r["backbone"], r["pe"], r["dataset"]) for r in records})
    backbones_datasets = sorted({(b, d) for b, _, d in cells})

    rows = []
    for backbone, dataset in backbones_datasets:
        none_curves = per_graph_rel_curves(records, backbone, "none", dataset)
        if not none_curves:
            continue
        pes = sorted({p for b, p, d in cells if b == backbone and d == dataset and p != "none"})
        for pe in pes:
            pe_curves = per_graph_rel_curves(records, backbone, pe, dataset)
            n, diff, lo, hi = paired_bootstrap_diff(none_curves, pe_curves, args.n_boot)
            sig = (lo == lo) and (lo > 0 or hi < 0)
            rows.append({
                "backbone": backbone, "dataset": dataset, "pe": pe,
                "n_paired_graphs": n, "mean_diff_rho_rel_vs_none": diff,
                "ci_lo": lo, "ci_hi": hi, "significant": sig,
            })
    df = pd.DataFrame(rows).sort_values(["dataset", "backbone", "pe"])
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(df)} rows)")
    print(df.to_string(index=False))
