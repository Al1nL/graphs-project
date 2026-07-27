"""
aggregate_results.py
=====================
Run this AFTER scripts/run_all.sh has produced real (non-placeholder) results/*.json files.

    python scripts/aggregate_results.py [--d-min N] [--d-max N] [--weight-by-count]

Produces:
  - results/summary_table.csv          : one row per (backbone, pe, dataset) -- task metric,
                                         rho with a graph-clustered bootstrap CI, and the
                                         raw gain s_bar(1). This is Table 1.
  - results/sensitivity_norm_<ds>.png  : s_tilde(d) = s_bar(d)/s_bar(1). PRIMARY figure.
  - results/sensitivity_raw_<ds>.png   : raw s_bar(d). APPENDIX figure -- see below.
  - results/criterion_b_<ds>.csv       : rho-vs-task-metric rank correlation per backbone.

Why three statistics and not one
--------------------------------
Raw s_bar(d) conflates decay SHAPE with overall GAIN: different LayerNorm placement,
residual scaling, depth and width put Graphormer's and GraphGPS's Jacobian norms on
different scales, so "slower decay" read off raw curves is not a statement about the PE.
rho = sum_{d>=d_min} s_bar(d) / sum_{d>=1} s_bar(d) is a ratio of sums, so the gain
cancels exactly. See the long comment block in src/sensitivity.py for the full argument,
and docs/analysis-plan.md for the pre-registered windows and the amended success criteria.

The raw curves are still emitted, on a log-y axis where a gain difference is a vertical
SHIFT and decay shape is SLOPE -- so the eye can separate what a table of raw numbers
cannot. Keep them in the appendix; do not rank cells with them.
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from sensitivity import (  # noqa: E402
    average_curves,
    bootstrap_over_graphs,
    long_range_fraction,
    normalized_curve,
)

# ---------------------------------------------------------------------------
# rho windows (d_min, d_max), per dataset.
#
# PROVISIONAL -- these are the values in force as of docs/analysis-plan.md and they are
# the weakest part of this script. Two known problems, both resolved by fix 3:
#
#   * d_max = 20 truncates Peptides at roughly the first third of its range (average
#     diameter 57), so "rho over d in [5,20]" is a MID-range statistic wearing a
#     long-range label.
#   * d_min = 5 is not "long range" for a diameter-57 graph -- it captures most of the
#     curve's mass. The cutoff should be defined RELATIVE to diameter (e.g. 0.5*diam),
#     which also puts the three datasets on a common axis and makes rho comparable
#     across them rather than only within one.
#
# rho is a property of the pair (curve, window), so whatever lands here must be reported
# alongside every rho in the paper. Override at the CLI to check sensitivity to the choice.
# ---------------------------------------------------------------------------
RHO_WINDOW = {
    "peptides-func": (5, 20),
    "peptides-struct": (5, 20),
    "pascalvoc-sp": (5, 20),
}
DEFAULT_WINDOW = (5, 20)

METRIC_HIGHER_IS_BETTER = {"ap": True, "macro_f1": True, "mae": False}


def load_all(results_dir="results"):
    records = []
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        with open(path) as f:
            records.append(json.load(f))
    return records


def _as_curve(raw):
    """Normalize a stored curve to {int d: {"mean": float, "count": int}}.

    Tolerates the pre-fix-1 flat {d: mean} shape, which carries no pair counts; those
    records get count=1 so they still plot, but they cannot be count-weighted or pooled
    correctly. Re-run the probe rather than trusting a weighted rho computed from them.
    """
    out = {}
    for k, v in raw.items():
        out[int(k)] = dict(v) if isinstance(v, dict) else {"mean": float(v), "count": 1}
    return out


def _per_graph_curves(record):
    raw = record.get("sensitivity_curves_per_graph")
    return [_as_curve(c) for c in raw] if raw else []


def build_summary_table(records, n_boot, weight_by_count, out_dir="results"):
    out_csv = os.path.join(out_dir, "summary_table.csv")
    by_cell = defaultdict(lambda: {"metric": [], "curves": [], "per_graph": []})
    for r in records:
        cell = by_cell[(r["backbone"], r["pe"], r["dataset"])]
        if r.get("metric_value") is not None:
            cell["metric"].append(r["metric_value"])
        if r.get("sensitivity_curve"):
            cell["curves"].append(_as_curve(r["sensitivity_curve"]))
            cell["per_graph"].extend(_per_graph_curves(r))

    rows = []
    for (backbone, pe, dataset), cell in sorted(by_cell.items()):
        if not cell["metric"] and not cell["curves"]:
            continue
        d_min, d_max = RHO_WINDOW.get(dataset, DEFAULT_WINDOW)
        row = {
            "backbone": backbone, "pe": pe, "dataset": dataset,
            "n_seeds": len(cell["metric"]),
            "rho_d_min": d_min, "rho_d_max": d_max,
            "rho_weighting": "pair_count" if weight_by_count else "per_distance",
        }
        if cell["metric"]:
            s = pd.Series(cell["metric"])
            row["metric_mean"], row["metric_std"] = s.mean(), s.std()

        if cell["curves"]:
            pooled = average_curves(cell["curves"])
            stat = lambda c: long_range_fraction(c, d_min, d_max, weight_by_count)  # noqa: E731
            row["rho"] = stat(pooled)
            # s_bar(1) is the gain rho exists to divide out. Reported so a reviewer can see
            # HOW different the scales were, and so pathological runs are visible.
            row["s_bar_1_raw_gain"] = pooled.get(1, {}).get("mean", float("nan"))
            row["n_pairs_total"] = sum(b["count"] for b in pooled.values())
            row["n_pairs_tail"] = sum(
                b["count"] for d, b in pooled.items() if d_min <= d <= d_max
            )
            row["max_d_populated"] = max(pooled)

            if cell["per_graph"]:
                _, lo, hi = bootstrap_over_graphs(cell["per_graph"], stat, n_boot=n_boot)
                row["rho_ci_lo"], row["rho_ci_hi"] = lo, hi
                row["n_graphs"] = len(cell["per_graph"])
            else:
                row["rho_ci_lo"] = row["rho_ci_hi"] = float("nan")
                row["n_graphs"] = 0
        rows.append(row)

    if not rows:
        print("No completed runs found (all metric_value are null placeholders and no "
              "sensitivity curves present). Run scripts/run_all.sh with real backbones first.")
        return None

    df = pd.DataFrame(rows).sort_values(["dataset", "backbone", "rho"], ascending=[1, 1, 0])
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(df)} rows)")

    missing_ci = df[df.get("n_graphs", 0) == 0] if "n_graphs" in df else df
    if len(missing_ci):
        print(f"  NOTE: {len(missing_ci)} cell(s) have no per-graph curves, so rho has no "
              "confidence interval. Have run_experiment.py store "
              "`sensitivity_curves_per_graph` -- without it, rho is a point estimate with "
              "no way to tell a real gap from noise.")
    widths = df.groupby("dataset")["max_d_populated"].nunique() if "max_d_populated" in df else {}
    for ds, n in (widths.items() if hasattr(widths, "items") else []):
        if n > 1:
            print(f"  WARNING: cells for '{ds}' have differing max populated distance. "
                  "rho is only comparable over an IDENTICAL window -- check max_dist "
                  "was the same for every run before ranking these.")
    return df


def _curves_by_dataset(records):
    by_ds = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get("sensitivity_curve"):
            by_ds[r["dataset"]][(r["backbone"], r["pe"])].append(_as_curve(r["sensitivity_curve"]))
    return by_ds


def plot_curves(records, out_dir="results"):
    for dataset, cells in _curves_by_dataset(records).items():
        for kind in ("norm", "raw"):
            plt.figure(figsize=(7.5, 5))
            plotted = 0
            for (backbone, pe), curves in sorted(cells.items()):
                pooled = average_curves(curves)
                xs = sorted(pooled)
                try:
                    ys = ([pooled[d]["mean"] for d in xs] if kind == "raw"
                          else [normalized_curve(pooled)[d] for d in xs])
                except (KeyError, ZeroDivisionError) as exc:
                    print(f"  skipped {backbone}-{pe} on the normalized plot: {exc}")
                    continue
                plt.plot(xs, ys, marker="o", ms=3, label=f"{backbone}-{pe}")
                plotted += 1
            if not plotted:
                plt.close()
                continue
            plt.xlabel("Hop distance $d$")
            if kind == "raw":
                plt.ylabel(r"$\bar{s}(d)$  (mean Jacobian Frobenius norm)")
                plt.title(f"Raw sensitivity — {dataset}  [APPENDIX: gain-confounded]")
            else:
                plt.ylabel(r"$\tilde{s}(d) = \bar{s}(d)\,/\,\bar{s}(1)$")
                plt.title(f"Normalized sensitivity — {dataset}")
            plt.yscale("log")
            plt.grid(alpha=0.25, which="both", lw=0.5)
            plt.legend(fontsize=7, ncol=2)
            plt.tight_layout()
            path = os.path.join(out_dir, f"sensitivity_{kind}_{dataset}.png")
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"Wrote {path}")


def criterion_b(df, out_dir="results"):
    """Does the PE ranking by rho track the PE ranking by task metric?

    POWER WARNING: with 5 PEs a Spearman correlation needs |r| ~ 0.9 to clear p<0.05, so
    a per-backbone test is close to uninformative on its own. Treat the per-cell numbers
    as descriptive and read the pooled row; the proposal's criterion (b) as written is
    underpowered and docs/analysis-plan.md records that.
    """
    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("scipy not installed; skipping criterion (b) rank correlations.")
        return
    if df is None or "rho" not in df or "metric_mean" not in df:
        return

    for dataset, sub in df.groupby("dataset"):
        rows, pooled_x, pooled_y = [], [], []
        higher_better = METRIC_HIGHER_IS_BETTER.get(
            {"peptides-func": "ap", "peptides-struct": "mae"}.get(dataset, "macro_f1"), True
        )
        for backbone, cell in sub.groupby("backbone"):
            cell = cell.dropna(subset=["rho", "metric_mean"])
            if len(cell) < 3:
                continue
            # flip MAE so "higher is better" holds and the sign of r is interpretable
            y = cell["metric_mean"] if higher_better else -cell["metric_mean"]
            r, p = spearmanr(cell["rho"], y)
            rows.append({"backbone": backbone, "n_pes": len(cell), "spearman_r": r, "p": p})
            pooled_x += list(cell["rho"])
            pooled_y += list(y)
        if len(pooled_x) >= 4:
            r, p = spearmanr(pooled_x, pooled_y)
            rows.append({"backbone": "ALL (pooled)", "n_pes": len(pooled_x),
                         "spearman_r": r, "p": p})
        if rows:
            path = os.path.join(out_dir, f"criterion_b_{dataset}.csv")
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"Wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--d-min", type=int, default=None, help="override rho lower cutoff")
    ap.add_argument("--d-max", type=int, default=None, help="override rho upper cutoff")
    ap.add_argument("--weight-by-count", action="store_true",
                    help="weight rho by pair count (information flow) instead of per "
                         "distance (curve shape); see long_range_fraction's docstring")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    if args.d_min is not None or args.d_max is not None:
        for ds, (lo, hi) in list(RHO_WINDOW.items()):
            RHO_WINDOW[ds] = (args.d_min or lo, args.d_max or hi)
        print(f"rho window overridden to {RHO_WINDOW}")

    recs = load_all(args.results_dir)
    summary = build_summary_table(recs, args.n_boot, args.weight_by_count,
                                  out_dir=args.results_dir)
    plot_curves(recs, args.results_dir)
    criterion_b(summary, args.results_dir)
