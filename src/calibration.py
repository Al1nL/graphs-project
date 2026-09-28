"""
calibration.py
==============
Empirically calibrates `num_target_nodes` (T) by sweeping target-node subsampling budgets
and evaluating convergence of long-range sensitivity metrics.
"""

import time
from typing import Callable, Dict, List, Sequence

from sensitivity import (
    average_curves,
    bootstrap_over_graphs,
    compute_sensitivity_curve,
    long_range_fraction,
)

DEFAULT_LADDER = (4, 8, 16, 32, 64, 128)


def sweep_target_nodes(
    model_fn_factory: Callable[[object], Callable],
    graphs: Sequence,
    n_shared_feats: int,
    ladder: Sequence[int] = DEFAULT_LADDER,
    max_dist: int = 20,
    d_min: int = 5,
    d_max: int = 20,
    weight_by_count: bool = False,
    n_boot: int = 500,
    seed: int = 0,
    chunk_size: int = 16,
    verbose: bool = True,
) -> List[Dict]:
    """Probe every graph at each T in `ladder`; return one row per rung.

    Args:
        model_fn_factory: given one PyG Data object, returns the `model_fn(x) -> [n, p]`
            callable for that graph (the model itself must be shared across graphs -- see
            run_experiment.make_model_fn for the contract the wrapper must satisfy).
        graphs: the sampled test graphs to calibrate on. ~10 is plenty; this is a
            convergence check, not an estimate of rho itself.
        n_shared_feats: as in compute_sensitivity_curve -- the shared original feature
            width, identical across all five PE variants.
        d_min, d_max: the rho window. Must match what aggregate_results.py will use, or
            the calibration is for a different statistic than the one you report.

    Returns rows with rho, its graph-clustered bootstrap CI, pair counts, saturation
    diagnostics, and wall time.
    """
    if not graphs:
        raise ValueError("no graphs supplied")
    ladder = sorted(set(int(t) for t in ladder))
    if any(t < 1 for t in ladder):
        raise ValueError(f"ladder entries must be >= 1, got {ladder}")

    rows = []
    for t in ladder:
        started = time.time()
        per_graph = []
        for data in graphs:
            per_graph.append(
                compute_sensitivity_curve(
                    model_fn_factory(data),
                    data,
                    n_shared_feats=n_shared_feats,
                    max_dist=max_dist,
                    num_target_nodes=t,
                    chunk_size=chunk_size,
                    seed=seed,  # fixed across the ladder -> nested target sets
                )
            )

        def stat(c, _lo=d_min, _hi=d_max, _w=weight_by_count):
            return long_range_fraction(c, _lo, _hi, _w)

        rho, lo, hi = bootstrap_over_graphs(per_graph, stat, n_boot=n_boot, seed=seed)
        pooled = average_curves(per_graph)
        pooled_tail = sum(
            sum(b["count"] for d, b in c.items() if d_min <= d <= d_max) for c in per_graph
        )
        # Density of the SPARSEST bucket THAT IS ACTUALLY REPORTED, pooled across graphs.
        #
        # Two things this deliberately does not do. It does not take the minimum over all
        # buckets: with max_dist == max_diameter, the extreme buckets (d near 159 on
        # Peptides) are populated by a handful of pairs from one or two of the largest
        # graphs, so an all-bucket minimum would sit at ~1 forever and criterion (ii) would
        # reject every rung, reporting non-convergence no matter how dense the sampling. And
        # it does not take a per-graph minimum: rho is computed on the POOLED curve, so
        # pooled counts are what determine whether a reported bucket is estimable.
        #
        # Restricting to [d_min, d_max] is the correct scope regardless of max_dist -- a
        # bucket outside the reported window contributes to no statistic, so its emptiness
        # says nothing about whether T is large enough.
        in_window = [b["count"] for d, b in pooled.items() if d_min <= d <= d_max]
        # T saturates once it exceeds a graph's node count: randperm(n)[:T] returns all n
        # nodes, so larger T buys nothing there and the curve flattens for the wrong reason.
        n_saturated = sum(1 for g in graphs if t >= g.num_nodes)
        row = {
            "T": t,
            "rho": rho,
            "rho_ci_lo": lo,
            "rho_ci_hi": hi,
            "ci_width": hi - lo,
            "n_graphs": len(graphs),
            "pairs_in_window": pooled_tail,
            "min_bucket_count": min(in_window) if in_window else 0,
            "n_buckets_in_window": len(in_window),
            "graphs_saturated": n_saturated,
            "seconds": time.time() - started,
        }
        rows.append(row)
        if verbose:
            print(
                f"  T={t:4d}  rho={rho:.5f}  CI=[{lo:.5f}, {hi:.5f}]  "
                f"width={hi - lo:.2e}  pairs_in_window={pooled_tail:,}  "
                f"{row['seconds']:.1f}s"
                + (f"  [{n_saturated}/{len(graphs)} graphs saturated]" if n_saturated else "")
            )
    return rows


def recommend_target_nodes(
    rows: List[Dict],
    tol: float = 0.5,
    min_bucket: int = 5,
    max_ci_inflation: float = 0.15,
) -> Dict:
    """Select the smallest target node count T meeting stability, tail density, and CI bounds."""
    if not rows:
        raise ValueError("no rows to analyse")
    rows = sorted(rows, key=lambda r: r["T"])
    ref = rows[-1]
    half = ref["ci_width"] / 2.0
    # A saturated reference is a bad anchor: once T >= n the rung samples every node, so it
    # is not "denser sampling" but a different estimator, and rho can shift for that reason
    # alone. Surfaced on the result so callers can refuse to trust the recommendation.
    ref_saturated = ref["graphs_saturated"] > 0

    if not (half == half) or half <= 0:  # NaN or degenerate (n_graphs < 2)
        return {
            "recommended_T": ref["T"],
            "reference_T": ref["T"],
            "tol": tol,
            "band": float("nan"),
            "converged": False,
            "limited_by": "no_ci",
            "reference_saturated": ref_saturated,
            "reason": "no usable bootstrap CI at the reference rung (need >= 2 graphs); "
                      "falling back to the densest rung sampled",
        }

    band = tol * half
    # Search rungs BELOW the reference only. The reference trivially satisfies the
    # criterion against itself, so including it would make every sweep report success and
    # the non-convergence branch below unreachable -- laundering an unvalidated T into a
    # claim of empirical stability, which is the one outcome this module must not produce.
    max_width = ref["ci_width"] * (1.0 + max_ci_inflation)
    stable_at, sparse_rejected, inflated_rejected = None, [], []

    def _binding():
        if inflated_rejected and (
            not sparse_rejected or max(inflated_rejected) > max(sparse_rejected)
        ):
            return "ci_inflation"
        return "tail_density" if sparse_rejected else "rho_stability"

    for i, r in enumerate(rows[:-1]):
        if not all(abs(rr["rho"] - ref["rho"]) <= band for rr in rows[i:]):
            continue
        if stable_at is None:
            stable_at = r["T"]
        if r["min_bucket_count"] < min_bucket:
            sparse_rejected.append(r["T"])
            continue  # rho is stable here, but the tail buckets are too thin to trust
        if r["ci_width"] > max_width:
            inflated_rejected.append(r["T"])
            continue  # unbiased, but pays for it with a materially wider interval
        saturated = r["graphs_saturated"] == r["n_graphs"]
        return {
            "recommended_T": r["T"],
            "reference_T": ref["T"],
            "tol": tol,
            "band": band,
            "converged": True,
            "limited_by": _binding(),
            "reference_saturated": ref_saturated,
            "reason": (
                "every rung from this T upward is within "
                f"{tol:g}x the reference CI half-width ({band:.2e}) of "
                f"rho(T={ref['T']})={ref['rho']:.5f}, its sparsest distance bucket "
                f"holds >= {min_bucket} pairs, and its CI is within "
                f"{max_ci_inflation:.0%} of the reference CI"
                + (
                    f"; rho was already stable at T={stable_at} but rungs "
                    f"{sparse_rejected} were rejected for leaving a bucket with "
                    f"<{min_bucket} pairs"
                    if sparse_rejected else ""
                )
                + (
                    f"; rungs {inflated_rejected} were rejected for inflating the CI by "
                    f"more than {max_ci_inflation:.0%} over the reference "
                    f"({ref['ci_width']:.2e})"
                    if inflated_rejected else ""
                )
                + (
                    "; NOTE this rung is SATURATED -- T exceeds the node count of every "
                    "calibration graph, so it samples all nodes and the ladder cannot "
                    "probe further. The flatness here is saturation, not demonstrated "
                    "convergence"
                    if saturated else ""
                )
                + (
                    "; WARNING the reference rung is itself saturated on "
                    f"{ref['graphs_saturated']}/{ref['n_graphs']} graphs, so it is a "
                    "different estimator rather than strictly denser sampling -- shorten "
                    "the ladder or calibrate on larger graphs"
                    if ref_saturated else ""
                )
            ),
        }
    return {
        "recommended_T": ref["T"],
        "reference_T": ref["T"],
        "tol": tol,
        "band": band,
        "converged": False,
        "limited_by": _binding(),
        "reference_saturated": ref_saturated,
        "reason": (
            f"no rung below T={ref['T']} qualified -- "
            + (
                f"rho was stable from T={stable_at}, but every such rung either left a "
                f"bucket with <{min_bucket} pairs {sparse_rejected} or inflated the CI by "
                f">{max_ci_inflation:.0%} {inflated_rejected}"
                if (sparse_rejected or inflated_rejected) else
                "rho had not settled by the densest rung sampled"
            )
            + ". Extend the ladder before trusting any T in it"
        ),
    }


def report_sentence(rec: Dict, rows: List[Dict], d_min: int, d_max: int) -> str:
    """Format calibration result summary sentence."""
    ref = max(rows, key=lambda r: r["T"])
    n_graphs = ref["n_graphs"]
    if not rec["converged"]:
        return (
            f"WARNING: rho had not converged in T by T={ref['T']}. Do not claim stability; "
            f"extend the ladder past {ref['T']} and re-run."
        )
    ladder = [r["T"] for r in sorted(rows, key=lambda r: r["T"])]
    # Name the constraint that actually bound. Reporting only the rho-stability clause
    # when a different criterion selected T would overstate what was verified.
    binding = {
        "rho_stability": (
            f"rho (window d in [{d_min}, {d_max}]) changes by less than {rec['tol']:g}x "
            f"the bootstrap CI half-width relative to the densest sampling T={ref['T']}"
        ),
        "tail_density": (
            f"rho (window d in [{d_min}, {d_max}]) is stable relative to T={ref['T']} and "
            "every distance bucket retains enough sampled pairs to be estimated; smaller "
            "T left buckets in the tail too sparse to trust"
        ),
        "ci_inflation": (
            f"rho (window d in [{d_min}, {d_max}]) is unbiased relative to T={ref['T']} at "
            "every rung, and this is the smallest T whose graph-clustered bootstrap "
            "interval is no wider than the densest sampling's; smaller T left rho "
            "unbiased but widened the interval, since per-graph measurement noise is not "
            "separable from between-graph variation"
        ),
    }[rec["limited_by"]]
    return (
        f"We verified that rho is stable in the number of sampled target nodes: sweeping T "
        f"over {ladder} on {n_graphs} test graphs, {binding}. We use "
        f"T={rec['recommended_T']}."
    )
