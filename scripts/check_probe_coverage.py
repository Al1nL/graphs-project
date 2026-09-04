"""
check_probe_coverage.py
=======================
Is the T (`num_target_nodes`) a set of finished results was probed at large enough?

    python scripts/check_probe_coverage.py                      # every result in results/
    python scripts/check_probe_coverage.py --results-dir /path/to/liors/results
    python scripts/check_probe_coverage.py --backbone san       # just one arm

Read-only. Loads nothing but results/*.json, runs no model, and re-runs no probe -- the
data needed to answer this is already stored in each record's per-bucket `count`.

WHY THIS EXISTS
---------------
rho is a ratio of sums over distance buckets, and `long_range_fraction` skips buckets with
no sampled pairs: they contribute to NEITHER sum. Writing rho = N/D and dropping a far
term x from both,

    (N - x) / (D - x)  <  N / D        whenever  N < D,  which always holds here

so an unpopulated far bucket does not merely add noise -- it biases rho DOWNWARD, in the
same direction as the effect the study is looking for. Two arms probed at different T are
therefore not comparable, and a T too small for either understates long-range mass.

The corollary that makes this script worth running rather than guessing: T only has to be
large enough that the far buckets inside the rho window are populated and their means are
stable. That is checkable from finished results, so a T chosen for one arm can be VALIDATED
for the others without re-running anything.

WHAT "ADEQUATE" MEANS HERE
--------------------------
Three checks per cell, in increasing order of how much they should worry you:

  1. COVERAGE  -- every distance in the rho window is populated in the pooled curve.
                  A missing one is silently dropped, per the algebra above.
  2. DEPTH     -- populated in-window buckets hold enough pairs for their mean to mean
                  something. A bucket with 3 pairs is a number, not an estimate.
  3. SENSITIVITY -- and this is the decisive one: recompute rho while requiring at least k
                  pairs per bucket. If rho moves by more than its own bootstrap CI as k
                  rises, the thin buckets are carrying the estimate, and T is too small.
                  If rho barely moves, the tail is being measured, not guessed at.

Check 3 subsumes the other two in principle, but 1 and 2 say WHY when it fails, and a
count is easier to act on than a delta.

WHAT THIS CANNOT TELL YOU
-------------------------
Whether a LARGER T would change rho. Only the buckets actually sampled are stored, so this
validates from below -- it detects a T that is too small, and cannot certify one as
optimal. `scripts/calibrate_target_nodes.py` sweeps T against a live model and is the tool
for that question; this one is for results that already exist.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aggregate_results import _as_curve, _cluster_keys, _per_graph_curves, load_all  # noqa: E402
from dataset_meta import abs_rho_window  # noqa: E402
from sensitivity import bootstrap_over_graphs, long_range_fraction  # noqa: E402

# Pair-count thresholds for check 3. 1 is "as computed"; 30 is a conventional point past
# which a sample mean is usually taken seriously. The ladder matters more than the exact
# rungs -- what is being read is whether rho MOVES, not its value at any one of them.
COUNT_THRESHOLDS = (1, 10, 30)

# A bucket thinner than this is reported as shallow. Not a failure on its own: the far
# tail is legitimately sparse, which is why check 3 decides and this only explains.
THIN_BUCKET = 30


def _filter_by_count(curve, min_count):
    return {d: b for d, b in curve.items() if b["count"] >= min_count}


def _pooled_from(record):
    """Pooled curve as stored, plus the per-graph curves the bootstrap needs."""
    return _as_curve(record.get("sensitivity_curve") or {}), _per_graph_curves(record)


def analyse(record, n_boot=1000, weight_by_count=False):
    dataset = record.get("dataset")
    d_min, d_max = abs_rho_window(dataset)
    pooled, per_graph = _pooled_from(record)

    in_window = {d: b for d, b in pooled.items() if d_min <= d <= d_max}
    expected = list(range(d_min, d_max + 1))
    missing = [d for d in expected if d not in in_window]
    thin = sorted(d for d, b in in_window.items() if b["count"] < THIN_BUCKET)
    counts = sorted(b["count"] for b in in_window.values())

    # check 3: does rho depend on the thin buckets?
    rhos = {}
    for k in COUNT_THRESHOLDS:
        rhos[k] = long_range_fraction(_filter_by_count(pooled, k), d_min, d_max,
                                      weight_by_count)

    # scale for "materially": the CI rho already carries, clustered on the molecule so a
    # graph's seed-copies do not count as independent evidence
    ci_width = None
    if per_graph:
        curves = [e["curve"] for e in per_graph]
        groups = _cluster_keys(per_graph)
        try:
            _point, lo, hi = bootstrap_over_graphs(
                curves,
                lambda c: long_range_fraction(c, d_min, d_max, weight_by_count),
                groups=groups, n_boot=n_boot, seed=0)
            ci_width = hi - lo
        except Exception as exc:  # noqa: BLE001 - a degenerate curve should not abort the sweep
            print(f"    (bootstrap unavailable: {type(exc).__name__}: {exc})")

    finite = [v for v in rhos.values() if v == v]  # drop NaN
    drift = (max(finite) - min(finite)) if len(finite) > 1 else 0.0

    if missing:
        verdict = "INADEQUATE"
        why = (f"{len(missing)} of {len(expected)} in-window distances have NO sampled "
               f"pairs and are dropped from both sums, biasing rho down: {missing[:8]}"
               + (" ..." if len(missing) > 8 else ""))
    elif ci_width is not None and drift > ci_width:
        verdict = "INADEQUATE"
        why = (f"rho moves {drift:.4f} across count thresholds, more than its own "
               f"bootstrap CI width {ci_width:.4f} -- the thin buckets are carrying it")
    elif ci_width is not None and drift > 0.5 * ci_width:
        verdict = "MARGINAL"
        why = (f"rho moves {drift:.4f}, over half its CI width {ci_width:.4f}")
    elif thin:
        verdict = "OK"
        why = (f"rho stable (moves {drift:.4f}); {len(thin)} in-window buckets are thin "
               f"(<{THIN_BUCKET} pairs) but are not driving it")
    else:
        verdict = "OK"
        why = f"every in-window bucket populated and >= {THIN_BUCKET} pairs"

    return {
        "backbone": record.get("backbone"), "pe": record.get("pe"), "dataset": dataset,
        "seed": record.get("seed"), "T": record.get("num_target_nodes"),
        "window": (d_min, d_max), "n_expected": len(expected),
        "n_populated": len(in_window), "missing": missing, "thin": thin,
        "min_count": counts[0] if counts else 0,
        "median_count": counts[len(counts) // 2] if counts else 0,
        "rhos": rhos, "drift": drift, "ci_width": ci_width,
        "verdict": verdict, "why": why,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--backbone", default=None, help="filter, e.g. san")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--weight-by-count", action="store_true",
                    help="analyse the pair-weighted rho instead of the per-distance one. "
                         "It is far less T-sensitive, being dominated by the mid-range "
                         "buckets where most pairs live -- so if the default verdict is "
                         "INADEQUATE and this one is not, switching the primary statistic "
                         "is an alternative to re-running. It answers a different "
                         "question and must be declared, not quietly substituted.")
    args = ap.parse_args()

    records = [r for r in load_all(args.results_dir)
               if r.get("sensitivity_curve")
               and (args.backbone is None or r.get("backbone") == args.backbone)
               and (args.dataset is None or r.get("dataset") == args.dataset)]

    if not records:
        print(f"no probed results found in {args.results_dir}/")
        print("  (cells with an empty sensitivity_curve -- e.g. a backbone whose probe is "
              "not wired -- are skipped, as are smoke-test records)")
        return 1

    ts = sorted({r.get("num_target_nodes") for r in records})
    print(f"{len(records)} probed cell(s) in {args.results_dir}/")
    print(f"T values present: {ts}")
    if len(ts) > 1:
        print("  WARNING: more than one T across these results. rho is only comparable "
              "at a FIXED T -- a bucket unpopulated at one T and populated at another "
              "shifts rho systematically, not just noisily. Cells at different T should "
              "not be compared to each other.")
    print()

    rows = [analyse(r, args.n_boot, args.weight_by_count) for r in records]
    rows.sort(key=lambda r: ({"INADEQUATE": 0, "MARGINAL": 1, "OK": 2}[r["verdict"]],
                             str(r["dataset"]), str(r["pe"]), r["seed"] or 0))

    for r in rows:
        rho_str = "  ".join(f"k>={k}: {v:.4f}" for k, v in r["rhos"].items())
        print(f"[{r['verdict']:10}] {r['backbone']}/{r['pe']}/{r['dataset']} "
              f"seed={r['seed']} T={r['T']}")
        print(f"             window d={r['window'][0]}..{r['window'][1]}  "
              f"populated {r['n_populated']}/{r['n_expected']}  "
              f"counts min={r['min_count']} median={r['median_count']}")
        print(f"             rho  {rho_str}")
        print(f"             {r['why']}")
        print()

    bad = [r for r in rows if r["verdict"] == "INADEQUATE"]
    marginal = [r for r in rows if r["verdict"] == "MARGINAL"]
    print(f"summary: {len(rows) - len(bad) - len(marginal)} OK, {len(marginal)} marginal, "
          f"{len(bad)} inadequate")
    if bad:
        print("\nT is too small for the cells above. Options, in the order I would weigh "
              "them:")
        print("  1. re-run those cells at a larger T -- correct, and costs the reruns")
        print("  2. narrow the rho window so it covers only distances that ARE measured; "
              "honest, but changes what the headline number means and must be declared")
        print("  3. --weight-by-count, which is far less T-sensitive; also a change of "
              "question, also declarable")
        print("  Shrinking the window silently to make the number look defensible is the "
              "one option that is not available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
