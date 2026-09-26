"""
make_combined_curves_figure.py
================================
aggregate_results.py's plot_curves() writes one PNG per dataset with a per-axis
legend (up to 14 backbone-PE lines on Peptides). Re-plots the SAME normalized
curves -- identical data/normalization, just re-pooled here rather than
re-computed -- as ONE combined figure for a single figure* in the paper.

Layout: small multiples, one row per PE variant x one column per dataset, with
color = backbone (only 3 levels, so a panel never has more than 3 overlapping
lines and a single shared legend covers every panel). This replaces an earlier
1x3 layout that used linestyle to distinguish 5 PE variants within one axis --
five dash patterns at 1.1pt line width become indistinguishable once curves
cross, especially on a log-scaled, noisy y-axis. Splitting PE onto its own row
removes that channel entirely.

    python scripts/make_combined_curves_figure.py --in-dir results_all --out figures/sensitivity_norm_combined.png
"""
import argparse
import os
import sys

import matplotlib.lines as mlines
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_results import _curves_by_dataset, load_all  # noqa: E402
from sensitivity import average_curves, normalized_curve  # noqa: E402

DATASET_LABEL = {"peptides-func": "Peptides-func", "peptides-struct": "Peptides-struct",
                  "pascalvoc-sp": "PascalVOC-SP"}
BACKBONE_COLOR = {"gps": "#1b78b4", "san": "#d1495b", "graphormer": "#2ca25f"}
BACKBONE_LABEL = {"gps": "GraphGPS", "san": "SAN", "graphormer": "Graphormer"}
BACKBONE_ORDER = ["gps", "san", "graphormer"]
PE_ORDER = ["none", "lappe", "rwse", "signnet", "grpe"]
PE_LABEL = {"none": "No-PE", "lappe": "LapPE", "rwse": "RWSE",
            "signnet": "SignNet-PE", "grpe": "GRPE"}


def main(in_dir, out_path):
    """CRITICAL: figsize is the figure's ACTUAL PRINT SIZE (~6.6in wide at
    \\includegraphics width=0.95\\textwidth in a two-column page), not an
    oversized canvas LaTeX shrinks back down -- that shrink takes the fonts
    down with it and silently undoes any fontsize increase made here. dpi is
    raised for sharpness only; it does not change point/physical sizes."""
    records = load_all(in_dir)
    by_ds = _curves_by_dataset(records)
    datasets = [d for d in ["peptides-func", "peptides-struct", "pascalvoc-sp"] if d in by_ds]

    # Pre-pool every (dataset, backbone, pe) cell once so we know which PE rows
    # actually have data before laying out the grid.
    pooled_xy = {}
    pes_present = set()
    backbones_present = set()
    for dataset in datasets:
        for (backbone, pe), curves in by_ds[dataset].items():
            if dataset == "pascalvoc-sp" and backbone == "san":
                continue
            pooled = average_curves(curves)
            xs = sorted(pooled)
            try:
                ys = [normalized_curve(pooled)[d] for d in xs]
            except (KeyError, ZeroDivisionError):
                continue
            pooled_xy[(dataset, backbone, pe)] = (xs, ys)
            pes_present.add(pe)
            backbones_present.add(backbone)
    pes = [p for p in PE_ORDER if p in pes_present]

    n_rows, n_cols = len(pes), len(datasets)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.6, 1.55 * n_rows + 0.6),
                              sharex="col", squeeze=False)

    # Last row with data in each column -- that's where the x-axis label goes,
    # since a column's final row can be an empty (data-less) panel we hide.
    last_data_row = {}
    for col, dataset in enumerate(datasets):
        for row, pe in enumerate(pes):
            if any(pooled_xy.get((dataset, b, pe)) is not None for b in BACKBONE_ORDER):
                last_data_row[col] = row

    for row, pe in enumerate(pes):
        for col, dataset in enumerate(datasets):
            ax = axes[row][col]
            any_curve = False
            for backbone in BACKBONE_ORDER:
                xy = pooled_xy.get((dataset, backbone, pe))
                if xy is None:
                    continue
                xs, ys = xy
                ax.plot(xs, ys, lw=1.1, color=BACKBONE_COLOR[backbone])
                any_curve = True
            if not any_curve:
                ax.axis("off")
                continue
            ax.set_yscale("log")
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.25, which="both", lw=0.4)
            if row == 0:
                ax.set_title(DATASET_LABEL[dataset], fontsize=8.5)
            if row == last_data_row[col]:
                ax.set_xlabel("Hop distance $d$", fontsize=7)
                ax.tick_params(labelbottom=True)
            if col == 0:
                ax.set_ylabel(PE_LABEL[pe], fontsize=7.5, fontweight="bold")

    fig.suptitle(r"Normalized sensitivity $\tilde s(d) = \bar s(d)/\bar s(1)$ by PE variant",
                 fontsize=9, y=0.995)

    legend_handles = [mlines.Line2D([], [], color=BACKBONE_COLOR[b], lw=1.5, label=BACKBONE_LABEL[b])
                       for b in BACKBONE_ORDER if b in backbones_present]
    fig.legend(handles=legend_handles, title="Backbone", fontsize=6.5, title_fontsize=7,
               loc="lower center", ncol=len(legend_handles), frameon=False,
               bbox_to_anchor=(0.5, -0.01 / n_rows))

    fig.tight_layout(rect=[0, 0.045, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="results_all")
    ap.add_argument("--out", default="figures/sensitivity_norm_combined.png")
    args = ap.parse_args()
    main(args.in_dir, args.out)
