"""Tests for scripts/check_probe_coverage.py.

    python tests/test_probe_coverage.py

The script answers one question -- "was this T large enough?" -- from finished results,
so these build result files with known coverage and check the verdict.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import check_probe_coverage as cpc  # noqa: E402
from dataset_meta import abs_rho_window  # noqa: E402
from sensitivity import long_range_fraction  # noqa: E402

D_MIN, D_MAX = abs_rho_window("peptides-func")


def _record(pe, populated, count_of, n_graphs=20, T=32):
    """A result whose in-window buckets are populated exactly on `populated`."""
    pooled = {}
    for d in range(1, D_MAX + 1):
        if d >= D_MIN and d not in populated:
            continue
        pooled[str(d)] = {"mean": 0.9 ** d, "count": count_of(d)}
    per_graph = [
        {"graph_id": g, "diameter": 90, "num_nodes": 150,
         "curve": {k: {"mean": v["mean"], "count": max(1, v["count"] // n_graphs)}
                   for k, v in pooled.items()}}
        for g in range(n_graphs)
    ]
    return {"backbone": "san", "pe": pe, "dataset": "peptides-func", "seed": 0,
            "metric_name": "ap", "metric_value": 0.6, "smoke_test": False,
            "num_target_nodes": T, "sensitivity_curve": pooled,
            "sensitivity_curves_per_graph": per_graph}


def test_full_coverage_with_deep_buckets_is_ok():
    rec = _record("rwse", set(range(D_MIN, D_MAX + 1)), lambda d: 4000 // d, T=128)
    out = cpc.analyse(rec, n_boot=100)
    assert out["verdict"] == "OK", out["why"]
    assert out["n_populated"] == out["n_expected"]
    assert not out["missing"]
    # rho must not depend on the count threshold when every bucket is deep
    assert len(set(round(v, 6) for v in out["rhos"].values())) == 1


def test_unpopulated_in_window_buckets_are_inadequate():
    """The failure this script exists to catch: far buckets never sampled."""
    rec = _record("lappe", set(range(D_MIN, 55)), lambda d: max(1, 300 // d), T=8)
    out = cpc.analyse(rec, n_boot=100)
    assert out["verdict"] == "INADEQUATE"
    assert out["missing"], "missing in-window distances were not reported"
    assert str(len(out["missing"])) in out["why"]


def test_dropping_far_buckets_biases_rho_downward():
    """Not a property of the script but the reason it exists, so it is pinned here.

    long_range_fraction skips empty buckets, so a dropped far term x leaves
    (N - x)/(D - x) < N/D whenever N < D. An unmeasured tail therefore UNDERSTATES
    long-range mass -- the same direction as the effect the study looks for, which is
    what makes a too-small T dangerous rather than merely noisy.
    """
    full = {d: {"mean": 0.9 ** d, "count": 100} for d in range(1, D_MAX + 1)}
    starved = {d: b for d, b in full.items() if d < 55}

    rho_full = long_range_fraction(full, D_MIN, D_MAX)
    rho_starved = long_range_fraction(starved, D_MIN, D_MAX)

    assert rho_starved < rho_full, (
        f"expected the starved curve to understate rho; got {rho_starved:.6f} "
        f"vs {rho_full:.6f}")


def test_mixed_T_across_records_is_flagged():
    """rho is only comparable at a fixed T, so a results dir holding several must say so
    rather than let them be averaged together."""
    with tempfile.TemporaryDirectory() as tmp:
        for pe, T in (("rwse", 32), ("lappe", 128)):
            rec = _record(pe, set(range(D_MIN, D_MAX + 1)), lambda d: 4000 // d, T=T)
            with open(os.path.join(tmp, f"san_{pe}_peptides-func_seed0.json"), "w") as f:
                json.dump(rec, f)

        recs = cpc.load_all(tmp)
        assert len({r["num_target_nodes"] for r in recs}) == 2


def test_smoke_records_are_not_analysed():
    """A smoke result carries a real-looking curve from 2 graphs. Judging T from it would
    condemn every T ever chosen."""
    with tempfile.TemporaryDirectory() as tmp:
        rec = _record("rwse", set(range(D_MIN, D_MAX + 1)), lambda d: 4000 // d)
        rec["smoke_test"] = True
        with open(os.path.join(tmp, "san_rwse_peptides-func_seed0_smoke.json"), "w") as f:
            json.dump(rec, f)

        assert cpc.load_all(tmp) == []


def test_records_without_pair_counts_are_unknown_not_inadequate():
    """The pre-fix-1 flat {d: mean} schema carries no counts, and _as_curve fabricates
    count=1 so those records still plot. Judged by the depth and threshold checks, every
    one of them would be condemned as INADEQUATE -- a schema fact reported as a finding
    about T. They must come back UNKNOWN instead."""
    rec = _record("rwse", set(range(D_MIN, D_MAX + 1)), lambda d: 4000 // d)
    rec["sensitivity_curve"] = {d: b["mean"] for d, b in rec["sensitivity_curve"].items()}

    out = cpc.analyse(rec, n_boot=50)
    assert out["verdict"] == "UNKNOWN", out["why"]
    assert "count" in out["why"]


def test_a_flat_zero_window_is_degenerate_not_ok():
    """Regression test for a real green light on a result with no signal in it.

    Two of Lior's SAN/VOC cells came back OK with all 23 in-window buckets populated and
    1,651+ pairs in the thinnest -- and rho 0.0000 at every count threshold. Coverage and
    drift are both vacuously clean when the window is flat zero: every bucket is
    "populated", and a rho of zero cannot move. The T checks all passed and reported a
    healthy cell.

    Asking "was T big enough?" presumes there is something for T to resolve, so this has
    to be tested before the others and reported as its own verdict.
    """
    rec = _record("lappe", set(range(D_MIN, D_MAX + 1)), lambda d: 5000)
    for d, b in rec["sensitivity_curve"].items():
        if int(d) >= D_MIN:
            b["mean"] = 0.0
    for g in rec["sensitivity_curves_per_graph"]:
        for d, b in g["curve"].items():
            if int(d) >= D_MIN:
                b["mean"] = 0.0

    out = cpc.analyse(rec, n_boot=50)
    assert out["verdict"] == "DEGENERATE", out["why"]
    assert not out["missing"], "coverage was fine; the problem is the signal, not sampling"
    assert "not a t problem" in out["why"].lower()


def test_a_small_but_real_tail_is_still_judged_on_T():
    """The degenerate check must not swallow genuinely small long-range mass -- a steeply
    decaying but non-zero tail is the normal case and the thing the study measures."""
    rec = _record("rwse", set(range(D_MIN, D_MAX + 1)), lambda d: 5000)
    out = cpc.analyse(rec, n_boot=50)
    assert out["verdict"] != "DEGENERATE"
    assert out["rhos"][1] > cpc.RHO_DEGENERATE


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nall tests passed")
