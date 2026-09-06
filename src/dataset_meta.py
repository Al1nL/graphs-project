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

`max_dist` is therefore set per dataset, to that dataset's exact max diameter, so that
every graph can populate its whole relative tail (see the DATASETS comment for the
measured coverage). Measuring that far does NOT mean reporting that far: buckets past
~40 are supported by a shrinking minority of graphs (see the population-shift note
below), so the reported ABSOLUTE window stops well short of the cap. The cap is a
measurement range; the window is a reporting choice, applied post-hoc.

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
# MEASURED 2026-07-29 on 400 test graphs per dataset (verify_diameters + percentiles),
# after downloading the real LRGB data. The quoted paper figures were accurate on average
# -- Peptides 56.5 vs 57, VOC-SP 27.7 vs 27 -- but the AVERAGE was the wrong summary to
# design around for Peptides, whose diameter distribution is heavily right-skewed:
#
#   Peptides   p25 36   median 51   p75 66   p90 95   p95 116   max 146
#   VOC-SP     p25 26   median 28   p75 29   p90 30   p95  31   max  36
#
# max_dist was originally 40 (Peptides) and 28 (VOC-SP), chosen on the reasoning that
# buckets past those points would be backed by a shrinking minority of graphs. The data
# says otherwise: at 40, 68% of Peptides graphs are truncated, and -- the serious part --
# only 81% of them can reach their own relative-window tail (d >= 0.5*diam). The other 19%
# contribute NOTHING to relative rho, and they are the largest, longest-range graphs in the
# set. That is a selection bias pointing exactly the wrong way for this paper.
#
# The caps below are the smallest round values at which every graph can reach its relative
# tail (Peptides 80 -> 100%, VOC-SP 36 -> 100% and nothing truncated at all).
#
# Raising them is nearly free: after the fix-1 restructuring the probe's cost is per TARGET
# NODE, not per distance bucket -- the full Jacobian block is computed regardless, so a
# larger cap only populates more buckets from work already done.
# `max_diameter` is the exact maximum over EVERY split, taken from the built PE cache
# manifests (2026-07-29), not from a sample. That distinction cost two mistakes:
#
#   sampled 400 test graphs     true (all splits)
#   Peptides  max 146            max 159
#   VOC-SP    max  36            max  54
#
# Both sample maxima were low, and both were used to justify a cap. VOC-SP's was described
# as "the observed maximum: truncates nothing" -- false, since graphs of diameter 37..54
# exist and are truncated on the absolute axis. Sample-derived extrema are systematically
# optimistic; only the whole-dataset value can license a claim about "every graph".
#
# max_dist == max_diameter: measure the FULL diameter, report over a narrower window.
#
# The earlier rule (max_dist >= max_diameter//2 + 1) guarantees only that a graph has AT
# LEAST ONE pair in the relative tail -- not that the tail is sampled. Relative bin b of a
# diameter-D graph covers absolute distances ((b-1)/10*D, b/10*D], so a D=159 graph at
# max_dist=80 puts only d=80 into bin 6 and leaves bins 7..10 empty. Measured on 600
# Peptides test graphs, the share of graphs with ALL FIVE tail bins populated was:
#
#     max_dist   40      80      90     100     159
#     coverage  39.3%   85.2%   91.2%   94.3%  100.0%
#
# The 14.8% shortfall at 80 is a BIAS, not just missing data: rho = tail/total, so dropping
# bins 7..10 removes numerator terms while the denominator stays dominated by bins 1..2,
# pushing rho_rel DOWN. It hits the largest, longest-range graphs -- exactly where this
# project expects the effect to be strongest. Truncation suppresses the signal being
# looked for.
#
# Any cap between 80 and the diameter is an arbitrary point on that curve. max_diameter is
# the one value where coverage is complete by construction, so that is what is used.
#
# This costs essentially nothing. The probe's work is per TARGET NODE -- the full Jacobian
# block is computed regardless of max_dist -- and the BFS cutoff is trivial either way. The
# PE caches do NOT need rebuilding: max_dist is a probe parameter, not a cache one.
#
# The ABSOLUTE window stays narrow and is unaffected. long_range_fraction skips every
# bucket past d_max, so measuring to 159 and reporting over (26, 80) gives exactly the same
# absolute rho that max_dist=80 would have. Far absolute buckets are deliberately excluded
# from that statistic: bucket d=140 is backed by <5% of graphs, which is the population
# shift the relative axis exists to avoid.
DATASETS = {
    "peptides-func": {
        "avg_nodes": 150,          # measured 150.0 (paper: 151)
        "avg_diameter": 57,        # measured 56.5; median 51, p90 95
        "median_diameter": 51,
        "p90_diameter": 95,        # d_max must stay at or below this -- see the note above
        "max_diameter": 159,       # exact, all splits, from the cache manifest
        "max_dist": 159,           # == max_diameter: relative tail complete for every graph
        "abs_rho_window": (26, 80),   # reported window; d_min ~ half the MEDIAN diameter
    },
    "peptides-struct": {           # same graphs as peptides-func, different task
        "avg_nodes": 150,
        "avg_diameter": 57,
        "median_diameter": 51,
        "p90_diameter": 95,
        "max_diameter": 159,
        "max_dist": 159,
        "abs_rho_window": (26, 80),
    },
    "pascalvoc-sp": {
        "avg_nodes": 480,          # measured 480.4
        "avg_diameter": 28,        # measured 27.7; tight -- median 28
        "median_diameter": 28,
        "p90_diameter": 30,        # measured on 256 test graphs, 2026-09-06
        "max_diameter": 54,        # exact, all splits (a 400-graph sample said 36)
        # Raised 36 -> 54 for the SAME reason as Peptides, though the request named only
        # Peptides: at 36, graphs of diameter 37..54 had partially sampled relative tails
        # and the identical downward bias. Leaving one dataset with a defect just argued to
        # be unacceptable in the other would be incoherent. Revert to 36 if unwanted.
        "max_dist": 54,
        # 36 -> 28. The old d_max was the MAXIMUM diameter of a 400-graph sample, i.e. the
        # single most extreme graph, so the top of the window was populated only by rare
        # outliers. Measured on 256 test graphs (2026-09-06): median 28, p90 30, p99 34,
        # max 36. Roughly 1% of graphs reach 34, and whether their far pairs get sampled
        # depends on T -- so rho ratcheted upward as buckets crossed from empty to
        # populated and calibration never converged, ending with
        # "rho had not converged in T by T=128".
        #
        # 28 is the MEDIAN diameter: report over distances at least half the graphs
        # actually contain. Projected bucket counts for a 256-graph probe at T=8, scaled
        # from a measured 8-graph run: d=26 ~1300, d=27 ~660, d=28 ~150, d=29 ~40,
        # d=30 ~8. So 28 leaves every in-window bucket estimable and 30 does not.
        #
        # Peptides needs no such change and shows why: its d_max of 80 sits BELOW its p90
        # diameter of 95, so many graphs realise it, and both its calibrations converged.
        # VOC's 36 sat above its p99. That is the invariant -- d_max <= p90_diameter --
        # and it is now asserted in tests/test_distance_axis.py.
        "abs_rho_window": (14, 28),
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


def min_max_dist_for_relative_tail(max_diameter: int) -> int:
    """Smallest `max_dist` giving every graph AT LEAST ONE pair in the relative tail.

    NECESSARY BUT NOT SUFFICIENT, and no longer the rule in force -- `max_dist` is now set
    to `max_diameter` outright (see the DATASETS comment). Retained as a floor and as a
    diagnostic, since it marks the point below which a graph drops out of relative rho
    *entirely* rather than merely being under-sampled.

    Why it is too weak: the relative window is d/diam(G) > 0.5, so a diameter-D graph needs
    d >= floor(D/2) + 1 for any pair at all -- but relative bin b spans
    ((b-1)/10*D, b/10*D], so satisfying this bound only fills bin 6. A D=159 graph at
    max_dist=80 gets exactly one distance (d=80) in bin 6 and nothing in bins 7..10. It is
    counted as "reaching its tail" while contributing a fifth of one.

    Evaluate against the WHOLE dataset's max diameter, never a sample: a 400-graph sample
    gave 146 for Peptides (true 159) and 36 for VOC-SP (true 54), and both were used to
    justify a cap. Asserted in tests/test_distance_axis.py against the recorded value.
    """
    return max_diameter // 2 + 1


def max_dist(dataset: str) -> int:
    return DATASETS[dataset]["max_dist"]


def abs_rho_window(dataset: str):
    return tuple(DATASETS[dataset]["abs_rho_window"])


def verify_diameters(dataset: str, graphs, sample=256):
    """Recompute diameter statistics from actual graphs and compare to the literals above.

    The `avg_diameter` entries are quoted from papers; `max_diameter` and `max_dist` are
    measured, taken from the built cache manifests over every split. This function is what
    checks the quoted averages against reality -- and it is what showed the average was the
    wrong summary to design around, since Peptides' distribution is heavily right-skewed.
    Run it once per dataset before the main grid.

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
