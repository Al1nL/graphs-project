"""
make_ranking_figures.py
========================
Two things the plain summary_table.csv cannot show at a glance: whether a PE's
RANK (not raw rho_rel) is stable across backbones, and whether that agreement is
more than chance. Both are computed PER DATASET -- datasets are never pooled.

    python scripts/make_ranking_figures.py --in-dir results_all --out-dir figures

Produces:
  figures/ranking_heatmap_<dataset>.png   [PER-CELL] rank of rho_rel within each
      backbone's own available PEs (rank 1 = most local / lowest rho_rel), one
      row per backbone, one column per PE. Cells for a (backbone, PE) that never
      ran are left blank. This is a visualization, not a new statistic.

  results_all/kendall_w.csv               [PER-DATASET, BACKBONES POOLED] Kendall's
      coefficient of concordance W over the PE set common to every backbone that
      has ANY data on that dataset (the intersection, since GraphGPS never ran
      GRPE and Graphormer never ran PascalVOC-SP). W=1 means every backbone ranks
      the PEs identically; W=0 means no more agreement than random rankings. A
      chi-square test (df = n_pe - 1) is the standard significance check for W,
      reported alongside; it is asymptotic and is a rough guide only at n_pe<=5.
      SAN's PascalVOC-SP rho_rel is identically 0 for all PEs (see paper
      Limitations) so its ranking there is undefined (all ties) and it is
      EXCLUDED from that dataset's W rather than silently contributing a
      degenerate all-tied rank.
"""
import argparse
import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, rankdata

PE_ORDER = ["none", "lappe", "rwse", "signnet", "grpe"]
PE_LABEL = {"none": "No-PE", "lappe": "LapPE", "rwse": "RWSE",
            "signnet": "SignNet-PE", "grpe": "GRPE"}
BACKBONE_ORDER = ["gps", "san", "graphormer"]
BACKBONE_LABEL = {"gps": "GraphGPS", "san": "SAN", "graphormer": "Graphormer"}
DATASET_LABEL = {"peptides-func": "Peptides-func", "peptides-struct": "Peptides-struct",
                  "pascalvoc-sp": "PascalVOC-SP"}


def kendalls_w(rank_matrix: np.ndarray):
    """rank_matrix: [n_raters, n_items], each row a rater's 1..n_items ranking.

    W = 12*S / (m^2 * (n^3 - n)), S = sum_j (R_j - mean(R))^2 over item rank-sums
    R_j. Standard coefficient of concordance (Kendall & Babington Smith, 1939);
    chi2 = m*(n-1)*W is asymptotically chi-square with n-1 df under H0: no
    agreement (Siegel & Castellan, Nonparametric Statistics, the usual citation
    for this test -- not specific to this project or its source papers).
    """
    m, n = rank_matrix.shape
    if n < 2 or m < 2:
        return float("nan"), float("nan"), float("nan")
    R = rank_matrix.sum(axis=0)
    S = float(np.sum((R - R.mean()) ** 2))
    W = 12 * S / (m ** 2 * (n ** 3 - n))
    stat = m * (n - 1) * W
    p = float(chi2.sf(stat, df=n - 1))
    return W, stat, p


def _one_heatmap(ax, df, ds, show_cbar):
    sub = df[df.dataset == ds]
    backbones = [b for b in BACKBONE_ORDER if (sub.backbone == b).any()]
    pes = [p for p in PE_ORDER if (sub.pe == p).any()]
    rank_grid = np.full((len(backbones), len(pes)), np.nan)
    val_grid = np.full((len(backbones), len(pes)), np.nan)
    for bi, b in enumerate(backbones):
        row = sub[sub.backbone == b].set_index("pe")["rho_rel"]
        have = row.dropna()
        if len(have) > 0:
            ranks = rankdata(have.values)  # rank 1 = MOST LOCAL (lowest rho_rel)
            for pe, r, v in zip(have.index, ranks, have.values):
                if pe in pes:
                    rank_grid[bi, pes.index(pe)] = r
                    val_grid[bi, pes.index(pe)] = v

    im = ax.imshow(rank_grid, cmap="RdYlBu_r", vmin=1, vmax=max(len(pes), 2), aspect="auto")
    ax.set_xticks(range(len(pes)))
    ax.set_xticklabels([PE_LABEL[p] for p in pes], rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(backbones)))
    ax.set_yticklabels([BACKBONE_LABEL[b] for b in backbones], fontsize=7)
    for bi in range(len(backbones)):
        for pi in range(len(pes)):
            if not np.isnan(rank_grid[bi, pi]):
                ax.text(pi, bi, f"{rank_grid[bi, pi]:.0f}\n({val_grid[bi, pi]:.3f})",
                        ha="center", va="center", fontsize=6.5)
            else:
                ax.text(pi, bi, "n/a", ha="center", va="center", fontsize=6.5, color="gray")
    ax.set_title(DATASET_LABEL[ds], fontsize=8.5)
    if show_cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("rank", fontsize=6.5)
        cb.ax.tick_params(labelsize=6)
    return im


def build_heatmaps(df, out_dir):
    """One COMBINED figure (3 subplots) rather than 3 separate small PNGs.

    CRITICAL: figsize is set to the figure's ACTUAL PRINT SIZE in the paper
    (~6.6in wide, at \\includegraphics width=0.95\\textwidth in a standard
    two-column page), not some larger canvas that LaTeX then shrinks -- shrinking
    a big raster back down to page width shrinks its fonts right along with it,
    silently undoing any fontsize bump made here. dpi is raised instead for
    print sharpness, which does not affect the physical/point size of anything.
    """
    datasets = [d for d in ["peptides-func", "peptides-struct", "pascalvoc-sp"]
                if not df[df.dataset == d].empty]
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.6, 3.4))
    if len(datasets) == 1:
        axes = [axes]
    fig.suptitle("Rank of $\\rho_{rel}$ within each backbone's own PEs "
                 "(1 = most local; value shown below rank)", fontsize=8.5, y=0.985)
    for ax, ds in zip(axes, datasets):
        _one_heatmap(ax, df, ds, show_cbar=True)
    # Explicit margins, not tight_layout: tight_layout's automatic left-margin
    # estimate for long y-tick labels ("Graphormer") was too small, so labels
    # rendered on top of the first column's cell text. Manual margins avoid
    # that failure mode outright instead of tuning tight_layout's heuristic.
    fig.subplots_adjust(left=0.105, right=0.99, top=0.84, bottom=0.20, wspace=0.55)
    path = os.path.join(out_dir, "ranking_heatmap_combined.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Wrote {path}")


def build_kendall_table(df, out_csv):
    rows = []
    for ds in ["peptides-func", "peptides-struct", "pascalvoc-sp"]:
        sub = df[df.dataset == ds]
        if sub.empty:
            continue
        backbones = [b for b in BACKBONE_ORDER if (sub.backbone == b).any()]
        # SAN's PascalVOC-SP rho_rel is identically 0 for every PE (degenerate,
        # not a null result -- see paper Limitations): its "ranking" there is an
        # all-way tie carrying no information, so it is excluded from W rather
        # than silently contributing a degenerate rater.
        usable = []
        for b in backbones:
            vals = sub[sub.backbone == b]["rho_rel"]
            if vals.nunique(dropna=True) > 1:
                usable.append(b)
        if len(usable) < 2:
            rows.append({"dataset": ds, "backbones": ",".join(usable), "n_pe_common": 0,
                         "W": float("nan"), "chi2": float("nan"), "p": float("nan"),
                         "note": "fewer than 2 backbones with a non-degenerate ranking"})
            continue
        # common PE set: only PEs every USABLE backbone actually ran
        pe_sets = [set(sub[sub.backbone == b]["pe"]) for b in usable]
        common = sorted(set.intersection(*pe_sets), key=lambda p: PE_ORDER.index(p))
        if len(common) < 3:
            rows.append({"dataset": ds, "backbones": ",".join(usable), "n_pe_common": len(common),
                         "W": float("nan"), "chi2": float("nan"), "p": float("nan"),
                         "note": f"only {len(common)} PE(s) common to all usable backbones"})
            continue
        rank_matrix = np.zeros((len(usable), len(common)))
        for bi, b in enumerate(usable):
            row = sub[(sub.backbone == b) & (sub.pe.isin(common))].set_index("pe")["rho_rel"]
            rank_matrix[bi] = rankdata([row[p] for p in common])
        W, stat, p = kendalls_w(rank_matrix)
        rows.append({"dataset": ds, "backbones": ",".join(usable), "n_pe_common": len(common),
                     "pe_set": ",".join(common), "W": W, "chi2": stat, "p": p, "note": ""})
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="results_all")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(args.in_dir, "summary_table.csv"))
    build_heatmaps(df, args.out_dir)
    build_kendall_table(df, os.path.join(args.in_dir, "kendall_w.csv"))
