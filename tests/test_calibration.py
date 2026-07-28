"""Regression tests for src/calibration.py.

    python -m pytest tests/test_calibration.py   (or: python tests/test_calibration.py)

The decision rule is the part worth pinning. A convergence check that accepts too early is
worse than no check at all -- it launders an arbitrary T into a claim of empirical
stability, which is exactly the thing this module was written to avoid.
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from calibration import (  # noqa: E402
    recommend_target_nodes,
    report_sentence,
    sweep_target_nodes,
)

LADDER = (4, 8, 16, 32, 64, 128)


def _rows(rhos, ci_width=0.01, n_graphs=10, saturated=0, min_bucket=50):
    return [
        {"T": t, "rho": r, "rho_ci_lo": r - ci_width / 2, "rho_ci_hi": r + ci_width / 2,
         "ci_width": ci_width, "n_graphs": n_graphs, "pairs_in_window": 5000,
         "min_bucket_count": min_bucket, "graphs_saturated": saturated, "seconds": 1.0}
        for t, r in zip(LADDER, rhos)
    ]


def test_picks_smallest_settled_rung():
    """rho drifts, then settles from T=16 onward -> recommend 16, not 4 and not 128."""
    rows = _rows([0.20, 0.24, 0.2695, 0.2700, 0.2701, 0.2700], ci_width=0.01)
    rec = recommend_target_nodes(rows, tol=0.5)   # band = 0.5 * 0.005 = 0.0025
    assert rec["recommended_T"] == 16
    assert rec["converged"] is True
    assert rec["reference_T"] == 128


def test_rejects_a_curve_that_merely_crosses_the_reference():
    """T=4 happens to equal rho(T_max) on the way past it. Accepting it would be wrong.

    This is why the rule requires every LARGER rung to hold as well, not just this one.
    """
    rows = _rows([0.270, 0.400, 0.500, 0.400, 0.300, 0.270], ci_width=0.01)
    rec = recommend_target_nodes(rows, tol=0.5)
    assert rec["recommended_T"] == 128       # only the reference itself qualifies
    assert rec["converged"] is False


def test_flags_non_convergence_instead_of_guessing():
    """Still climbing at the densest rung -> say so; do not hand back a number."""
    rows = _rows([0.10, 0.15, 0.20, 0.25, 0.30, 0.35], ci_width=0.001)
    rec = recommend_target_nodes(rows, tol=0.5)
    assert rec["converged"] is False
    assert rec["recommended_T"] == 128
    assert "had not settled" in rec["reason"]
    assert "WARNING" in report_sentence(rec, rows, 5, 20)


def test_tolerance_is_relative_to_the_ci_not_absolute():
    """The same rho curve converges at different T depending on how wide the CI is.

    That is the intended behaviour: the claim is "subsampling bias is negligible RELATIVE
    to the uncertainty we already report", so a study with tight CIs must sample harder.
    """
    rhos = [0.20, 0.24, 0.263, 0.268, 0.2695, 0.270]
    loose = recommend_target_nodes(_rows(rhos, ci_width=0.05), tol=0.5)
    tight = recommend_target_nodes(_rows(rhos, ci_width=0.0005), tol=0.5)
    assert loose["recommended_T"] < tight["recommended_T"]


def test_unbiased_but_underpowered_rung_is_rejected():
    """rho is flat from T=4, but small T inflates the CI -- reject until it tightens.

    This is criterion part (iii), and it is what actually bound on the demo data: rho was
    unbiased at every rung while the CI ran 3.39e-2 at T=4 against 2.19e-2 at T=128. The
    cost of a small T is not bias, it is power -- bootstrap_over_graphs cannot separate a
    noisily-measured graph from a genuinely deviant one, so measurement noise lands in the
    interval.
    """
    rows = _rows([0.138] * 6)
    widths = {4: 3.39e-2, 8: 2.86e-2, 16: 2.64e-2, 32: 2.39e-2, 64: 2.24e-2, 128: 2.19e-2}
    for r in rows:
        r["ci_width"] = widths[r["T"]]
        r["rho_ci_lo"], r["rho_ci_hi"] = 0.138 - widths[r["T"]] / 2, 0.138 + widths[r["T"]] / 2
    rec = recommend_target_nodes(rows, tol=0.5, min_bucket=5, max_ci_inflation=0.15)
    assert rec["recommended_T"] == 32          # 2.39e-2 <= 1.15 * 2.19e-2 = 2.52e-2
    assert rec["limited_by"] == "ci_inflation"
    assert "inflating the CI" in rec["reason"]

    # a looser power budget accepts a cheaper rung; a stricter one demands a denser rung
    assert recommend_target_nodes(rows, max_ci_inflation=0.60)["recommended_T"] == 4
    assert recommend_target_nodes(rows, max_ci_inflation=0.03)["recommended_T"] == 64
    # tighter than any rung below the reference can deliver (2.19e-2 * 1.02 = 2.234e-2,
    # under T=64's 2.24e-2) -> report failure rather than hand back an unqualified rung
    strict = recommend_target_nodes(rows, max_ci_inflation=0.02)
    assert strict["converged"] is False and strict["recommended_T"] == 128


def test_ci_inflation_is_inert_when_widths_are_flat():
    """If the CI does not shrink with T, criterion (iii) must not fire spuriously."""
    rows = _rows([0.20, 0.270, 0.270, 0.270, 0.270, 0.270], ci_width=0.01)
    rec = recommend_target_nodes(rows, tol=0.5, max_ci_inflation=0.15)
    assert rec["recommended_T"] == 8
    assert rec["limited_by"] == "rho_stability"


def test_stable_but_sparse_rung_is_rejected():
    """rho settles at T=8, but T=8 leaves a bucket with 2 pairs. Must climb to T=32.

    This is criterion part (ii). Part (i) alone would accept T=8 -- and does, which is the
    whole reason (ii) exists: the bootstrap CI is dominated by between-graph variance and
    barely moves with T, so rho-stability is close to vacuous on its own.
    """
    rows = _rows([0.20, 0.270, 0.270, 0.270, 0.270, 0.270], ci_width=0.01)
    for r in rows:                       # tail fills in only as T grows
        r["min_bucket_count"] = {4: 1, 8: 2, 16: 3, 32: 12, 64: 40, 128: 90}[r["T"]]
    rec = recommend_target_nodes(rows, tol=0.5, min_bucket=5)
    assert rec["recommended_T"] == 32
    assert rec["converged"] is True
    assert rec["limited_by"] == "tail_density"
    assert "8" in rec["reason"] and "rejected" in rec["reason"]


def test_tail_density_binds_only_when_it_actually_bites():
    """Same rho curve, dense buckets everywhere -> part (i) governs and T=8 is accepted."""
    rows = _rows([0.20, 0.270, 0.270, 0.270, 0.270, 0.270], ci_width=0.01, min_bucket=99)
    rec = recommend_target_nodes(rows, tol=0.5, min_bucket=5)
    assert rec["recommended_T"] == 8
    assert rec["limited_by"] == "rho_stability"
    assert "rejected" not in rec["reason"]


def test_all_stable_rungs_too_sparse_reports_failure():
    """If every rung where rho is stable is also too thin, that is not convergence."""
    rows = _rows([0.20, 0.270, 0.270, 0.270, 0.270, 0.270], ci_width=0.01, min_bucket=1)
    rec = recommend_target_nodes(rows, tol=0.5, min_bucket=5)
    assert rec["converged"] is False
    assert rec["recommended_T"] == 128
    assert rec["limited_by"] == "tail_density"
    assert "Extend the ladder" in rec["reason"]
    assert "WARNING" in report_sentence(rec, rows, 5, 20)


def test_reference_saturation_is_surfaced():
    """A saturated top rung is a different estimator, not denser sampling -- a bad anchor."""
    clean = _rows([0.20, 0.270, 0.270, 0.270, 0.270, 0.270], ci_width=0.01)
    assert recommend_target_nodes(clean)["reference_saturated"] is False

    dirty = _rows([0.20, 0.270, 0.270, 0.270, 0.270, 0.270], ci_width=0.01)
    dirty[-1]["graphs_saturated"] = 8          # 8 of 10 graphs smaller than the top rung
    rec = recommend_target_nodes(dirty)
    assert rec["reference_saturated"] is True
    assert "reference rung is itself saturated" in rec["reason"]


def test_degenerate_ci_falls_back_and_says_why():
    rows = _rows([0.1, 0.2, 0.3, 0.3, 0.3, 0.3], ci_width=float("nan"))
    rec = recommend_target_nodes(rows, tol=0.5)
    assert rec["recommended_T"] == 128
    assert rec["converged"] is False
    assert "bootstrap CI" in rec["reason"]
    assert math.isnan(rec["band"])


def test_saturation_is_reported_not_silently_accepted():
    """Once T >= n, randperm(n)[:T] returns every node, so the curve flattens for a reason
    that has nothing to do with convergence. The recommendation must say so."""
    rows = _rows([0.20, 0.24, 0.270, 0.270, 0.270, 0.270], ci_width=0.01, saturated=0)
    for r in rows:
        if r["T"] >= 16:
            r["graphs_saturated"] = r["n_graphs"]
    rec = recommend_target_nodes(rows, tol=0.5)
    assert rec["recommended_T"] == 16
    assert "saturat" in rec["reason"]


def test_target_sets_are_nested_prefixes_across_the_ladder():
    """The sweep's key assumption: holding `seed` fixed makes the T=k target set a subset
    of the T=2k set, so consecutive rungs differ by added targets rather than by a fresh
    random draw. If this breaks, the sweep measures resampling jitter, not convergence.
    """
    n = 150
    sets = {}
    for t in LADDER:
        rng = torch.Generator().manual_seed(0)      # same seed each rung, as sweep does
        sets[t] = torch.randperm(n, generator=rng)[:t].tolist()
    for small, large in zip(LADDER, LADDER[1:]):
        assert sets[large][: len(sets[small])] == sets[small]
        assert set(sets[small]).issubset(sets[large])


def test_sweep_runs_and_returns_one_row_per_rung():
    """Smoke test of the full sweep against a tiny toy model."""
    import types

    torch.manual_seed(0)
    n = 40
    edges = [[i, i + 1] for i in range(n - 1)]
    ei = torch.tensor(edges + [[b, a] for a, b in edges]).t()
    adj = torch.zeros(n, n)
    adj[ei[0], ei[1]] = 1.0
    adj = adj / adj.sum(1, keepdim=True).clamp(min=1)
    data = types.SimpleNamespace(x=torch.randn(n, 6), edge_index=ei, num_nodes=n)

    lin = torch.nn.Linear(6, 8)
    factory = lambda d: (lambda x: torch.tanh(adj @ torch.tanh(lin(x))))  # noqa: E731

    rows = sweep_target_nodes(
        factory, [data] * 4, n_shared_feats=4, ladder=(2, 4, 8),
        max_dist=10, d_min=3, d_max=10, n_boot=25, verbose=False,
    )
    assert [r["T"] for r in rows] == [2, 4, 8]
    for r in rows:
        assert 0.0 <= r["rho"] <= 1.0
        assert r["pairs_in_window"] > 0
        assert r["seconds"] >= 0
    # more targets -> strictly more measured pairs
    assert rows[0]["pairs_in_window"] < rows[2]["pairs_in_window"]


def test_sweep_rejects_bad_input():
    for bad in ([], None):
        try:
            sweep_target_nodes(lambda d: None, bad or [], 4, ladder=(4,))
        except ValueError:
            continue
        raise AssertionError("empty graph list must be rejected")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nall tests passed")
