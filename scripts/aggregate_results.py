"""
aggregate_results.py
=====================
Run this AFTER scripts/run_all.sh has produced real (non-placeholder) results/*.json files.

Produces:
  - results/summary_table.csv   : one row per (backbone, pe, dataset), mean +/- std over seeds
  - results/sensitivity_plot_<dataset>.png : one figure per dataset, s_bar(d) curves for all
    backbone x PE combinations, to visually compare decay rates (the core plot for the paper)

This script is what turns the raw JSON dump into Table 1 / Figure 1 referenced in the
final report draft -- it is intentionally separate from run_experiment.py so it can be
re-run cheaply (no retraining) whenever new seeds finish.
"""

import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd


def load_all(results_dir="results"):
    records = []
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        with open(path) as f:
            records.append(json.load(f))
    return records


def build_summary_table(records, out_csv="results/summary_table.csv"):
    rows = defaultdict(list)
    for r in records:
        if r.get("metric_value") is None:
            continue  # skip unrun placeholders
        key = (r["backbone"], r["pe"], r["dataset"])
        rows[key].append(r["metric_value"])

    if not rows:
        print("No completed runs found yet (all metric_value are null placeholders). "
              "Nothing to aggregate -- run scripts/run_all.sh with real backbones first.")
        return

    out = []
    for (backbone, pe, dataset), vals in sorted(rows.items()):
        s = pd.Series(vals)
        out.append({"backbone": backbone, "pe": pe, "dataset": dataset,
                     "mean": s.mean(), "std": s.std(), "n_seeds": len(vals)})
    df = pd.DataFrame(out)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(df)} rows)")
    return df


def plot_sensitivity_curves(records, out_dir="results"):
    by_dataset = defaultdict(list)
    for r in records:
        if r.get("sensitivity_curve"):
            by_dataset[r["dataset"]].append(r)

    if not by_dataset:
        print("No sensitivity curves found yet -- nothing to plot.")
        return

    for dataset, recs in by_dataset.items():
        plt.figure(figsize=(7, 5))
        for r in recs:
            curve = {int(k): v for k, v in r["sensitivity_curve"].items()}
            xs = sorted(curve.keys())
            ys = [curve[x] for x in xs]
            plt.plot(xs, ys, marker="o", label=f"{r['backbone']}-{r['pe']}")
        plt.xlabel("Hop distance d")
        plt.ylabel(r"$\bar{s}(d)$  (mean Jacobian norm)")
        plt.yscale("log")
        plt.title(f"Long-range sensitivity — {dataset}")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"sensitivity_plot_{dataset}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    records = load_all()
    build_summary_table(records)
    plot_sensitivity_curves(records)
