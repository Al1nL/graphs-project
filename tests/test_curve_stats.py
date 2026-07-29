"""Regression tests for the scale-free curve statistics in src/sensitivity.py.

    python -m pytest tests/test_curve_stats.py     (or: python tests/test_curve_stats.py)

The whole reason rho exists is that raw s_bar(d) conflates decay SHAPE with overall GAIN.
So the load-bearing test here is not "rho computes the formula" -- it is that rho is
invariant to gain, and that it ranks a slowly-decaying low-gain curve above a rapidly-
decaying high-gain one. That is exactly the inversion raw s_bar(d) gets wrong, and it is
the failure mode that would have put a spurious backbone effect in the paper.

Needs torch (for the bootstrap RNG) but no GPU, dataset, or backbone repo.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sensitivity import (  # noqa: E402
    average_curves,
    bootstrap_over_graphs,
    long_range_fraction,
    normalized_curve,
)

D_MIN, D_MAX = 5, 20


def _curve(fn, gain=1.0, count=100, d_max=D_MAX):
    return {d: {"mean": gain * fn(d), "count": count} for d in range(1, d_max + 1)}


def _fast(d):   # sharp exponential decay -- poor long-range flow
    return math.exp(-0.6 * d)


def _slow(d):   # decays then flattens, as GT curves actually do
    return math.exp(-0.6 * d) + 0.02


def test_rho_is_invariant_to_gain():
    """The defining property: rho is a ratio of sums, so C cancels exactly."""
    base = long_range_fraction(_curve(_slow), D_MIN, D_MAX)
    for gain in (1e-4, 0.5, 1.0, 7.3, 1e5):
        scaled = long_range_fraction(_curve(_slow, gain=gain), D_MIN, D_MAX)
        assert abs(scaled - base) < 1e-12, gain


def test_rho_ranks_shape_not_gain():
    """A high-gain fast-decaying curve must NOT beat a low-gain slow-decaying one.

    This is the inversion that ranking on raw s_bar(d) produces, and the concrete reason
    proposal success criterion (a) was amended (docs/analysis-plan.md).
    """
    loud_but_local = _curve(_fast, gain=1000.0)
    quiet_but_global = _curve(_slow, gain=0.001)

    # raw sensitivity at every long distance says the WRONG curve is better...
    for d in range(D_MIN, D_MAX + 1):
        assert loud_but_local[d]["mean"] > quiet_but_global[d]["mean"]
    # ...while rho gets it right
    assert (long_range_fraction(quiet_but_global, D_MIN, D_MAX)
            > long_range_fraction(loud_but_local, D_MIN, D_MAX))


def test_rho_window_bounds_are_respected():
    curve = _curve(_slow, d_max=40)
    assert long_range_fraction(curve, 1, D_MAX) == 1.0          # whole window is the tail
    r20 = long_range_fraction(curve, D_MIN, 20)
    r40 = long_range_fraction(curve, D_MIN, 40)
    assert r40 > r20                                            # truncation hides tail mass
    for bad in ((0, 20), (5, 4)):
        try:
            long_range_fraction(curve, *bad)
        except ValueError:
            continue
        raise AssertionError(f"window {bad} should have been rejected")


def test_rho_weighting_modes_differ_and_both_ignore_gain():
    """Pair-count weighting answers a different question; state which one you used."""
    curve = {d: {"mean": _slow(d), "count": 200 // d} for d in range(1, D_MAX + 1)}
    shape = long_range_fraction(curve, D_MIN, D_MAX, weight_by_count=False)
    flow = long_range_fraction(curve, D_MIN, D_MAX, weight_by_count=True)
    assert abs(shape - flow) > 1e-3
    scaled = {d: {"mean": 42.0 * b["mean"], "count": b["count"]} for d, b in curve.items()}
    assert abs(long_range_fraction(scaled, D_MIN, D_MAX, True) - flow) < 1e-12


def test_normalized_curve_divides_out_gain():
    a = normalized_curve(_curve(_slow, gain=0.01))
    b = normalized_curve(_curve(_slow, gain=1000.0))
    assert a[1] == 1.0
    for d in a:
        assert abs(a[d] - b[d]) < 1e-12


def test_normalized_curve_rejects_missing_or_zero_anchor():
    try:
        normalized_curve({2: {"mean": 1.0, "count": 5}})
    except KeyError:
        pass
    else:
        raise AssertionError("missing anchor must raise")
    try:
        normalized_curve({1: {"mean": 0.0, "count": 5}})
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError("zero anchor must raise")


def test_bootstrap_brackets_point_estimate_and_narrows_with_agreement():
    """CI must contain the point estimate, and be tighter when graphs agree."""
    stat = lambda c: long_range_fraction(c, D_MIN, D_MAX)  # noqa: E731

    agree = [_curve(_slow, gain=1.0 + 0.01 * i) for i in range(24)]
    p1, lo1, hi1 = bootstrap_over_graphs(agree, stat, n_boot=200, seed=0)
    assert lo1 <= p1 <= hi1

    disagree = [_curve(_slow if i % 2 else _fast) for i in range(24)]
    p2, lo2, hi2 = bootstrap_over_graphs(disagree, stat, n_boot=200, seed=0)
    assert lo2 <= p2 <= hi2
    assert (hi1 - lo1) < (hi2 - lo2)      # gain-only spread carries no rho uncertainty


def test_clustering_by_graph_identity_does_not_let_seeds_fake_precision():
    """THE test for clustering. Probing the same graphs under 3 seeds must not shrink the
    interval as if it were 3x the data.

    The same test graphs are probed under every training run, so graph 17 at seeds 0/1/2
    is one molecule measured three times -- its topology, diameter and distance profile are
    identical across all three, and topology is what drives the distance profile. Treating
    those as independent understates the standard error by up to sqrt(n_seeds).
    """
    stat = lambda c: long_range_fraction(c, D_MIN, D_MAX)  # noqa: E731
    graphs = [_curve(_slow if i % 3 else _fast, gain=1.0 + 0.05 * i) for i in range(30)]

    # what one seed alone would give
    _, lo1, hi1 = bootstrap_over_graphs(graphs, stat, n_boot=400, seed=0)

    # three seeds over the SAME graphs; ids repeat, gains differ slightly per seed
    tripled, ids = [], []
    for seed in range(3):
        for i, g in enumerate(graphs):
            tripled.append({d: {"mean": b["mean"] * (1 + 0.01 * seed), "count": b["count"]}
                            for d, b in g.items()})
            ids.append(i)                      # graph identity, stable across seeds

    _, lo_flat, hi_flat = bootstrap_over_graphs(tripled, stat, n_boot=400, seed=0)
    _, lo_cl, hi_cl = bootstrap_over_graphs(tripled, stat, groups=ids, n_boot=400, seed=0)

    w1, w_flat, w_cl = hi1 - lo1, hi_flat - lo_flat, hi_cl - lo_cl
    assert w_flat < w1 * 0.75, (w_flat, w1)    # flattening fakes ~sqrt(3) more precision
    assert w_cl > w_flat * 1.3, (w_cl, w_flat)  # clustering refuses to
    assert abs(w_cl - w1) / w1 < 0.35, (w_cl, w1)  # and lands near the one-seed width


def test_clustering_is_a_noop_for_single_seed_runs():
    """Under the reduced grid SAN and Graphormer run 1 seed, where every id is unique."""
    stat = lambda c: long_range_fraction(c, D_MIN, D_MAX)  # noqa: E731
    graphs = [_curve(_slow, gain=1.0 + 0.1 * i) for i in range(20)]
    a = bootstrap_over_graphs(graphs, stat, n_boot=200, seed=3)
    b = bootstrap_over_graphs(graphs, stat, groups=list(range(20)), n_boot=200, seed=3)
    assert a == b


def test_clustering_carries_all_members_of_a_drawn_cluster():
    """A cluster drawn twice must contribute all its members twice, so a group of
    identical curves behaves like a single heavier observation."""
    stat = lambda c: long_range_fraction(c, D_MIN, D_MAX)  # noqa: E731
    a = _curve(_slow)
    b = _curve(_fast)
    # two identities, three copies each -> only 2 clusters, so no usable interval
    point, lo, hi = bootstrap_over_graphs(
        [a, a, a, b, b, b], stat, groups=["g0"] * 3 + ["g1"] * 3, n_boot=100
    )
    assert not math.isnan(point)
    assert point == long_range_fraction(average_curves([a, a, a, b, b, b]), D_MIN, D_MAX)


def test_bootstrap_rejects_misaligned_groups():
    stat = lambda c: long_range_fraction(c, D_MIN, D_MAX)  # noqa: E731
    try:
        bootstrap_over_graphs([_curve(_slow)] * 3, stat, groups=[1, 2])
    except ValueError as exc:
        assert "parallel" in str(exc)
        return
    raise AssertionError("misaligned groups must be rejected")


def test_bootstrap_degenerate_inputs():
    stat = lambda c: long_range_fraction(c, D_MIN, D_MAX)  # noqa: E731
    p, lo, hi = bootstrap_over_graphs([], stat)
    assert all(math.isnan(v) for v in (p, lo, hi))
    p, lo, hi = bootstrap_over_graphs([_curve(_slow)], stat, n_boot=50)
    assert not math.isnan(p) and math.isnan(lo) and math.isnan(hi)  # n=1 -> no interval


def test_bootstrap_is_deterministic_under_seed():
    stat = lambda c: long_range_fraction(c, D_MIN, D_MAX)  # noqa: E731
    graphs = [_curve(_slow, gain=1.0 + 0.1 * i) for i in range(12)]
    a = bootstrap_over_graphs(graphs, stat, n_boot=100, seed=7)
    b = bootstrap_over_graphs(graphs, stat, n_boot=100, seed=7)
    c = bootstrap_over_graphs(graphs, stat, n_boot=100, seed=8)
    assert a == b and a[1:] != c[1:]


def test_pooling_then_rho_uses_pair_counts():
    """rho computed on the pooled curve must reflect count-weighted pooling, not a
    plain average of per-graph curves."""
    big = {5: {"mean": 1.0, "count": 500}, 1: {"mean": 1.0, "count": 500}}
    small = {5: {"mean": 9.0, "count": 1}, 1: {"mean": 1.0, "count": 500}}
    pooled = average_curves([big, small])
    assert abs(pooled[5]["mean"] - (1.0 * 500 + 9.0) / 501) < 1e-12
    assert pooled[5]["count"] == 501


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nall tests passed")
