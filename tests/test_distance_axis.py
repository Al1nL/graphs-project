"""Regression tests for fix 3: the distance axis and the GRPE bucketing decision.

    python -m pytest tests/test_distance_axis.py   (or: python tests/test_distance_axis.py)

Two separate concerns that used to share one hardcoded `20`:
  * max_dist -- a MEASUREMENT parameter; changing it costs a re-run of the probe
  * the GRPE spatial-bias table -- a MODEL parameter; changing it costs a re-train

Collapsing them meant neither could move without silently moving the other, and it made
GRPE's tail behaviour an artefact of the probe's cap rather than a property of the encoding.
"""

import os
import sys
import types

import networkx as nx
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset_meta import (  # noqa: E402
    DATASETS,
    REL_BINS,
    REL_RHO_WINDOW,
    SPD_EXACT_UPTO,
    SPD_HORIZON,
    SPD_NUM_BUCKETS,
    SPD_UNREACHABLE,
    abs_rho_window,
    max_dist,
    spd_bucket_id,
)
from sensitivity import (  # noqa: E402
    average_curves,
    graph_diameter,
    long_range_fraction,
    to_relative_curve,
)


# --------------------------------------------------------------------------- max_dist
def test_max_dist_is_per_dataset_and_covers_more_than_the_old_cap():
    """20 truncated Peptides (avg diameter 57) at roughly its first third."""
    assert max_dist("peptides-func") == 40 > 20
    assert max_dist("peptides-struct") == max_dist("peptides-func")   # same graphs
    # PascalVOC-SP has 3x the nodes but far shorter paths -- one cap cannot serve both
    assert max_dist("pascalvoc-sp") < max_dist("peptides-func")
    for ds, meta in DATASETS.items():
        lo, hi = abs_rho_window(ds)
        assert 1 <= lo < hi <= meta["max_dist"], ds


# --------------------------------------------------------------------------- relative axis
def test_relative_rebinning_puts_different_diameters_on_a_common_axis():
    """Two graphs with 3x different diameters and the SAME shape in RELATIVE terms.

    The claim being tested is comparative, not an identity: rebinning must bring the two
    into far closer agreement than the absolute axis does. Exact agreement is not
    achievable and is not claimed -- with 10 bins, a diameter-10 graph puts one absolute
    distance in each bin (an endpoint) while a diameter-30 graph puts three (a midpoint
    average). That granularity gap is the documented caveat in `to_relative_curve`, not a
    defect: it is why small graphs are said to give coarse relative resolution.
    """
    small = {d: {"mean": 1.0 - d / 12, "count": 100} for d in range(1, 11)}
    large = {d: {"mean": 1.0 - d / 36, "count": 100} for d in range(1, 31)}

    rs = to_relative_curve(small, diameter=10, n_bins=REL_BINS)
    rl = to_relative_curve(large, diameter=30, n_bins=REL_BINS)
    assert set(rs) == set(rl) == set(range(1, REL_BINS + 1))

    # relative rho agrees closely; absolute rho over a shared window does not, because the
    # window means completely different things to a diameter-10 and a diameter-30 graph
    rel_gap = abs(long_range_fraction(rs, *REL_RHO_WINDOW)
                  - long_range_fraction(rl, *REL_RHO_WINDOW))
    abs_gap = abs(long_range_fraction(small, 5, 20) - long_range_fraction(large, 5, 20))
    assert rel_gap < 0.02, rel_gap
    assert abs_gap > 5 * rel_gap, (abs_gap, rel_gap)

    # pointwise, the residual is bounded by the within-bin averaging offset (~half a bin's
    # worth of decay), not by any systematic distortion
    half_bin_decay = (10 / 12) / REL_BINS / 2
    for b in rs:
        assert abs(rs[b]["mean"] - rl[b]["mean"]) <= half_bin_decay + 1e-9, b


def test_relative_rho_reuses_long_range_fraction_unchanged():
    """Relative curves are integer-keyed, so the existing statistic applies as-is."""
    curve = {d: {"mean": 1.0, "count": 10} for d in range(1, 41)}
    rel = to_relative_curve(curve, diameter=40, n_bins=REL_BINS)
    rho = long_range_fraction(rel, *REL_RHO_WINDOW)
    assert abs(rho - 0.5) < 1e-9          # flat curve, top half of bins -> exactly half


def test_relative_rebinning_preserves_pair_counts_and_weights_by_them():
    curve = {1: {"mean": 10.0, "count": 900}, 2: {"mean": 0.0, "count": 100}}
    rel = to_relative_curve(curve, diameter=2, n_bins=2)
    assert sum(b["count"] for b in rel.values()) == 1000
    assert rel[1]["count"] == 900 and abs(rel[1]["mean"] - 10.0) < 1e-12


def test_relative_rebinning_edge_cases():
    assert to_relative_curve({1: {"mean": 1.0, "count": 5}}, diameter=0) == {}
    # d beyond the largest finite eccentricity (disconnected graph) clamps into the last bin
    rel = to_relative_curve({7: {"mean": 2.0, "count": 3}}, diameter=5, n_bins=REL_BINS)
    assert list(rel) == [REL_BINS]
    try:
        to_relative_curve({1: {"mean": 1.0, "count": 1}}, diameter=5, n_bins=0)
    except ValueError:
        return
    raise AssertionError("n_bins=0 must be rejected")


# --------------------------------------------------------------------------- diameter
def test_graph_diameter():
    path = torch.tensor([[i, i + 1] for i in range(9)] +
                        [[i + 1, i] for i in range(9)]).t()
    assert graph_diameter(path, 10) == 9
    cycle = nx.cycle_graph(10)
    ei = torch.tensor(list(cycle.edges()) + [(b, a) for a, b in cycle.edges()]).t()
    assert graph_diameter(ei, 10) == 5
    assert graph_diameter(torch.zeros((2, 0), dtype=torch.long), 5) == 0   # edgeless

    # disconnected: the LARGEST FINITE eccentricity, not infinity
    two = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1], [3, 4], [4, 3]]).t()
    assert graph_diameter(two, 5) == 2


# --------------------------------------------------------------------------- population shift
def test_pooling_records_how_many_graphs_back_each_bucket():
    """A far bucket can only draw from graphs big enough to have it. Without this column,
    'the curve flattens at large d' cannot be distinguished from 'only big graphs are
    left out there, and big graphs differ'."""
    small = {d: {"mean": 1.0, "count": 50} for d in range(1, 6)}     # diameter ~5
    big = {d: {"mean": 1.0, "count": 50} for d in range(1, 31)}      # diameter ~30
    pooled = average_curves([small, small, small, big])
    assert pooled[1]["n_graphs"] == 4        # every graph reaches d=1
    assert pooled[20]["n_graphs"] == 1       # only the big one reaches d=20


# --------------------------------------------------------------------------- GRPE buckets
def test_spd_buckets_are_exact_where_pairs_are_dense():
    for d in range(1, SPD_EXACT_UPTO + 1):
        assert spd_bucket_id(d) == d
    assert spd_bucket_id(0) == 0


def test_spd_buckets_separate_far_from_unreachable():
    """The old scheme initialised unreachable pairs to the cap, so GRPE learned ONE bias
    meaning 'far OR in a different component' -- two different structural relations."""
    assert spd_bucket_id(-1) == SPD_UNREACHABLE
    assert spd_bucket_id(None) == SPD_UNREACHABLE
    assert spd_bucket_id(10_000) != SPD_UNREACHABLE
    assert spd_bucket_id(10_000) < SPD_UNREACHABLE


def test_spd_buckets_distinguish_distances_the_old_cap_collapsed():
    """The decision fix 3 forces: measuring to d=40 while GRPE cannot tell d=25 from d=40
    would make its tail an artefact of our cap. Under the old cap=20 scheme every one of
    these shared a bucket."""
    assert len({spd_bucket_id(d) for d in (20, 25, 30, 40, 57)}) > 1
    assert spd_bucket_id(57) > spd_bucket_id(20)


def test_spd_buckets_are_monotone_and_bounded():
    prev = -1
    for d in range(0, SPD_HORIZON * 3):
        b = spd_bucket_id(d)
        assert 0 <= b < SPD_NUM_BUCKETS, d
        assert b >= prev, d                    # never decreases with distance
        prev = b
    # saturates at the horizon rather than overflowing the table
    assert spd_bucket_id(SPD_HORIZON * 10) == spd_bucket_id(SPD_HORIZON * 100)


def test_spd_bucket_table_is_smaller_than_raising_the_cap_would_be():
    """Rejected alternative was cap=40, i.e. a 41-row table with starved far buckets.
    Log-spacing covers 128 hops in fewer rows."""
    assert SPD_NUM_BUCKETS < 41
    assert spd_bucket_id(SPD_HORIZON) < SPD_UNREACHABLE


def test_no_hardcoded_caps_remain_in_src():
    """Regression guard: the constant lived in five places meaning two different things."""
    src = os.path.join(os.path.dirname(__file__), "..", "src")
    offenders = []
    for root, _, files in os.walk(src):
        for f in files:
            if not f.endswith(".py") or f == "dataset_meta.py":
                continue
            path = os.path.join(root, f)
            with open(path, encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    if "spd_cap" in line.lower() and "dataset_meta" not in line:
                        offenders.append(f"{f}:{i}")
    assert not offenders, f"hardcoded spd cap still present: {offenders}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nall tests passed")
