"""
make_combined_curves_figure.py
================================
aggregate_results.py's plot_curves() writes one PNG per dataset with a per-axis
legend (up to 14 backbone-PE lines on Peptides). Re-plots the SAME normalized
curves -- identical data/normalization, just re-pooled here rather than
re-computed -- as ONE combined figure (3 subplots) for a single figure* in the
paper, with:
  - color = backbone, linestyle = PE (fixed maps below), so the same PE is
    visually comparable across the three panels instead of an arbitrary
    per-panel color cycle, and
  - ONE shared legend for the whole figure instead of three cluttered
    per-panel ones -- with up to 14 series per panel, no per-panel legend at
    readable font size fits in a 2.2in-wide subplot regardless of point size.

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
PE_STYLE = {"none": (0, ()), "lappe": (0, (4, 1)), "rwse": (0, (1, 1)),
            "signnet": (0, (3, 1, 1, 1)), "grpe": (0, (5, 1, 1, 1, 1, 1))}
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

    fig, axes = plt.subplots(1, len(datasets), figsize=(6.6, 2.5))
    if len(datasets) == 1:
        axes = [axes]
    seen_pes, seen_backbones = set(), set()
    for ax, dataset in zip(axes, datasets):
        cells = by_ds[dataset]
        for (backbone, pe), curves in sorted(cells.items()):
            pooled = average_curves(curves)
            xs = sorted(pooled)
            try:
                ys = [normalized_curve(pooled)[d] for d in xs]
            except (KeyError, ZeroDivisionError):
                continue
            ax.plot(xs, ys, lw=1.1, color=BACKBONE_COLOR[backbone], linestyle=PE_STYLE[pe])
            seen_pes.add(pe)
            seen_backbones.add(backbone)
        ax.set_xlabel("Hop distance $d$", fontsize=7)
        ax.set_ylabel(r"$\tilde{s}(d) = \bar{s}(d)/\bar{s}(1)$", fontsize=7)
        ax.set_title(DATASET_LABEL[dataset], fontsize=8.5)
        ax.set_yscale("log")
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25, which="both", lw=0.4)

    # ONE shared legend: color = backbone, linestyle = PE, instead of one
    # cluttered per-panel legend with up to 14 entries each.
    color_handles = [mlines.Line2D([], [], color=BACKBONE_COLOR[b], lw=1.5, label=BACKBONE_LABEL[b])
                      for b in ["gps", "san", "graphormer"] if b in seen_backbones]
    style_handles = [mlines.Line2D([], [], color="black", lw=1.1, linestyle=PE_STYLE[p], label=PE_LABEL[p])
                      for p in ["none", "lappe", "rwse", "signnet", "grpe"] if p in seen_pes]
    fig.legend(handles=color_handles, title="Backbone (color)", fontsize=6, title_fontsize=6.5,
               loc="lower center", bbox_to_anchor=(0.27, -0.06), ncol=len(color_handles), frameon=False)
    fig.legend(handles=style_handles, title="PE (linestyle)", fontsize=6, title_fontsize=6.5,
               loc="lower center", bbox_to_anchor=(0.75, -0.06), ncol=len(style_handles), frameon=False)
    fig.suptitle(r"Normalized sensitivity $\tilde s(d) = \bar s(d)/\bar s(1)$",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0.08, 1, 0.90])
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
