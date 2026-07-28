"""
dataset_meta.py
===============
Single source of truth for the per-dataset constants that fix 3 turns from one hardcoded
`20` into a deliberate, dataset-aware choice.

Before this module, `20` appeared in five places meaning two DIFFERENT things:

    sensitivity.py    max_dist=20     a MEASUREMENT parameter -- how far out we look
    compute_pe.py     SPD_CAP=20      a MODEL parameter -- GRPE's bias table size
    graphgps_adapter, san_adapter, graphormer_adapter, and 15 SAN JSON configs

Changing the first costs a re-run of the probe. Changing the second changes the model and
costs a re-train of every GRPE cell. Collapsing them into one number made it impossible to
move either without silently moving the other.

--------------------------------------------------------------------------------------
Why 20 was wrong for Peptides
--------------------------------------------------------------------------------------
Peptides has an average diameter of 57 (proposal, Dataset section). Capping the probe at 20
measures roughly its first third -- on the dataset chosen precisely because it has long
range. Worse, rho computed over d in [5, 20] is then a MID-range statistic wearing a
long-range label.

`max_dist` is therefore set per dataset. It is not set to the full diameter: buckets past
~40 are supported by a shrinking minority of graphs (see the population-shift note below),
so they cost compute for measurements that are simultaneously noisy and unrepresentative.

--------------------------------------------------------------------------------------
The population-shift problem, and why the relative axis is primary
--------------------------------------------------------------------------------------
A bucket at d=50 can only contain pairs from graphs whose diameter is at least 50. So as d
grows, the SUBPOPULATION OF GRAPHS being averaged silently changes. A curve that flattens
at large d might show genuine long-range flow -- or might just show that only the big
graphs survive out there, and big graphs behave differently. The absolute-d curve cannot
distinguish these.

Indexing by relative distance d / diam(G) largely normalizes this away, and additionally
puts datasets with wildly different diameters (Peptides ~57, PascalVOC-SP ~27) on a common
axis, so a single rho is comparable ACROSS datasets rather than only within one.

The cost, stated honestly: "relative hop 0.5" mixes 5 hops in a diam-10 graph with 28 hops
in a diam-57 graph, and over-squashing theory (Di Giovanni et al., ref [7]) is stated in
ABSOLUTE hops. So both axes are reported, with distinct jobs:

    absolute d        primary WITHIN a dataset; what connects to the theory we cite
    relative d/diam   primary ACROSS datasets; the axis rho is ranked on

Per-bucket graph counts are recorded either way, so the composition shift is visible rather
than assumed away.
"""

import math

# ---------------------------------------------------------------------------
# Per-dataset constants.
#
# `avg_diameter` is quoted from the proposal / LRGB paper and is used only to derive
# defaults and to sanity-check runs -- `verify_diameters()` in this module recomputes it
# from the actual data. Re-run that before trusting the numbers in a paper; do not treat
# these literals as measurements.
# ---------------------------------------------------------------------------
DATASETS = {
    "peptides-func": {
        "avg_nodes": 151,
        "avg_diameter": 57,        # proposal, Dataset section
        "max_dist": 40,            # MEASUREMENT cap; see module docstring for why not 57
        "abs_rho_window": (20, 40),
    },
    "peptides-struct": {           # same graphs as peptides-func, different task
        "avg_nodes": 151,
        "avg_diameter": 57,
        "max_dist": 40,
        "abs_rho_window": (20, 40),
    },
    "pascalvoc-sp": {
        "avg_nodes": 480,
        "avg_diameter": 27,        # superpixel graphs: many more nodes, far shorter paths
        "max_dist": 28,
        "abs_rho_window": (10, 28),
    },
}

# Relative-distance binning, shared by all datasets -- this is what makes rho comparable
# across them. Deciles of d / diam(G); the window is the top half.
REL_BINS = 10
REL_RHO_WINDOW = (6, 10)  # bins 6..10  <=>  d/diam(G) > 0.5

# ---------------------------------------------------------------------------
# GRPE spatial-bias bucketing -- a MODEL parameter. Changing anything here changes the
# model and requires re-training every GRPE cell.
#
# The decision fix 3 forces (see docs/analysis-plan.md, Amendment 5):
#
# The old scheme capped shortest-path distance at 20 and gave every d >= 20 the same
# learned bias. Measuring sensitivity out to d=40 while GRPE cannot tell d=25 from d=40
# would make GRPE's tail behaviour an artefact of our cap rather than a property of the
# encoding -- and GRPE is one of the five arms being ranked, so that would corrupt the
# headline result in a direction that looks like a finding.
#
# Rejected: simply raising the cap to 40. It doubles the table and starves the far buckets,
# which would see very few training examples on chain-like peptides.
#
# Adopted: T5-style bucketing -- exact buckets for d <= EXACT_UPTO where pairs are dense,
# logarithmically widening buckets beyond, to a horizon well past any dataset's diameter.
# Fine resolution where it matters, real coverage in the tail, small table.
#
# Also fixed here: unreachable pairs get their OWN bucket. Previously they were initialised
# to the cap and were therefore indistinguishable from "far", so GRPE learned a single bias
# meaning "far OR in a different connected component" -- two different structural relations.
# ---------------------------------------------------------------------------
SPD_EXACT_UPTO = 8      # d = 1..8 get their own bucket
SPD_NUM_BUCKETS = 24    # total, including bucket 0 (self) and the unreachable bucket
SPD_HORIZON = 128       # distance at which the log-spaced buckets saturate
SPD_UNREACHABLE = SPD_NUM_BUCKETS - 1


def spd_bucket_id(d: int) -> int:
    """Map a shortest-path distance to a GRPE bias-table index.

    d < 0 (or None) means unreachable -- a distinct relation, not a large distance.

        d = 0            -> 0                     (self)
        1 <= d <= 8      -> d                     (exact)
        d > 8            -> 9 .. 22               (log-spaced, saturating at SPD_HORIZON)
        unreachable      -> 23
    """
    if d is None or d < 0:
        return SPD_UNREACHABLE
    if d <= SPD_EXACT_UPTO:
        return int(d)
    n_log = SPD_UNREACHABLE - SPD_EXACT_UPTO - 1  # log buckets available
    scaled = math.log(d / SPD_EXACT_UPTO) / math.log(SPD_HORIZON / SPD_EXACT_UPTO)
    return min(SPD_EXACT_UPTO + 1 + int(scaled * n_log), SPD_UNREACHABLE - 1)


def max_dist(dataset: str) -> int:
    return DATASETS[dataset]["max_dist"]


def abs_rho_window(dataset: str):
    return tuple(DATASETS[dataset]["abs_rho_window"])


def verify_diameters(dataset: str, graphs, sample=256):
    """Recompute diameter statistics from actual graphs and compare to the literals above.

    The `avg_diameter` entries are quoted from papers, and `max_dist` is derived from them.
    If the real distribution differs, `max_dist` is truncating (or wasting compute on) the
    wrong range. Run this once per dataset before the main grid.

    Returns a dict with the measured stats and the fraction of graphs whose diameter
    exceeds `max_dist` -- i.e. how much range the probe is choosing not to look at.
    """
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
