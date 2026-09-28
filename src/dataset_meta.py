"""
dataset_meta.py
===============
Single source of truth for per-dataset constants, distance thresholds, and evaluation windows.
"""

import math

# ---------------------------------------------------------------------------
# Per-dataset constants
# ---------------------------------------------------------------------------
DATASETS = {
    "peptides-func": {
        "avg_nodes": 150,
        "avg_diameter": 57,
        "median_diameter": 51,
        "max_diameter": 159,
        "max_dist": 159,
        "abs_rho_window": (26, 80),
    },
    "peptides-struct": {
        "avg_nodes": 150,
        "avg_diameter": 57,
        "median_diameter": 51,
        "max_diameter": 159,
        "max_dist": 159,
        "abs_rho_window": (26, 80),
    },
    "pascalvoc-sp": {
        "avg_nodes": 480,
        "avg_diameter": 28,
        "median_diameter": 28,
        "max_diameter": 54,
        "max_dist": 54,
        "abs_rho_window": (14, 36),
    },
}

# Node-feature width per dataset, MEASURED from the downloaded data. This is the
# `n_shared_feats` the Jacobian is taken over, and it must be identical across all five PE
# variants (sensitivity.assert_shared_width).
NODE_FEATURE_DIM = {
    "peptides-func": 9,
    "peptides-struct": 9,
    "pascalvoc-sp": 14,
}

# Relative-distance binning, shared by all datasets -- this is what makes rho comparable
# across them. Deciles of d / diam(G); the window is the top half.
REL_BINS = 10
REL_RHO_WINDOW = (6, 10)  # bins 6..10  <=>  d/diam(G) > 0.5

# Spatial distance bucketing: exact for d <= 8, log-spaced beyond, dedicated bucket for unreachable.
# ---------------------------------------------------------------------------
SPD_EXACT_UPTO = 8      # d = 1..8 get their own bucket
SPD_NUM_BUCKETS = 24    # total, including bucket 0 (self) and unreachable bucket
SPD_HORIZON = 128       # distance at which log-spaced buckets saturate
SPD_UNREACHABLE = SPD_NUM_BUCKETS - 1


def spd_bucket_id(d: int) -> int:
    """Map a shortest-path distance to a log-spaced bucket index."""
    if d is None or d < 0:
        return SPD_UNREACHABLE
    if d <= SPD_EXACT_UPTO:
        return int(d)
    n_log = SPD_UNREACHABLE - SPD_EXACT_UPTO - 1
    scaled = math.log(d / SPD_EXACT_UPTO) / math.log(SPD_HORIZON / SPD_EXACT_UPTO)
    return min(SPD_EXACT_UPTO + 1 + int(scaled * n_log), SPD_UNREACHABLE - 1)


def min_max_dist_for_relative_tail(max_diameter: int) -> int:
    """Smallest `max_dist` giving every graph at least one pair in the relative tail."""
    return max_diameter // 2 + 1


def max_dist(dataset: str) -> int:
    return DATASETS[dataset]["max_dist"]


def abs_rho_window(dataset: str):
    return tuple(DATASETS[dataset]["abs_rho_window"])


def verify_diameters(dataset: str, graphs, sample=256):
    """Recompute diameter statistics from sample graphs and compare to dataset metadata."""
    from sensitivity import graph_diameter

    diams = [graph_diameter(g.edge_index, g.num_nodes) for g in list(graphs)[:sample]]
    diams = [d for d in diams if d > 0]
    if not diams:
        return {"n": 0}
    cap = max_dist(dataset)
    diams_sorted = sorted(diams)
    return {
        "n": len(diams),
        "measured_mean": sum(diams) / len(diams),
        "measured_median": diams_sorted[len(diams_sorted) // 2],
        "measured_max": diams_sorted[-1],
        "quoted_avg_diameter": DATASETS[dataset]["avg_diameter"],
        "max_dist": cap,
        "frac_graphs_exceeding_max_dist": sum(d > cap for d in diams) / len(diams),
    }
