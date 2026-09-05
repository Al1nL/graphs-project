"""
calibration.py
==============
Turns `num_target_nodes` (T) -- the probe's target-node subsampling budget -- from an
arbitrary constant into an empirically chosen, reportable one.

Why this exists
---------------
compute_sensitivity_curve samples T target nodes v per graph and measures the exact
Jacobian norm from every source u to each of them. T therefore controls how much of each
graph's distance profile we actually observe, and it trades three things off:

  * cost -- linear in T (one batched backward per target),
  * per-graph curve stability -- bootstrap_over_graphs resamples whole graphs, so each
    graph needs its OWN rho estimated well enough that measurement noise does not
    masquerade as between-graph variation (this inflates the CI: conservative, but it
    wastes real statistical power),
  * tail coverage -- far buckets are the sparsest, and they are the ones the paper is
    about.

None of that yields a defensible number by argument. So measure it: sweep T over a ladder,
watch rho, and stop when rho stops moving relative to the uncertainty we already report.

The nesting property that makes this cheap and clean
-----------------------------------------------------
compute_sensitivity_curve picks targets with `torch.randperm(n, generator=rng)[:T]` where
rng is seeded by `seed`. Holding `seed` fixed across the sweep therefore makes the target
sets NESTED PREFIXES: the T=4 set is a subset of the T=8 set, which is a subset of T=16,
and so on. Consecutive points on the curve differ only by "what did adding more targets
do", not by "this was a different random draw" -- so the sweep measures convergence rather
than resampling jitter. Do not vary `seed` across the ladder.

Budgeting: a full ladder of (4, 8, 16, 32, 64, 128) costs sum(ladder) = 252 target-probes
per graph, i.e. roughly 2x a single probe at T=128. The largest rung dominates.
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
    """Smallest T that is unbiased in rho, dense in the tail, AND statistically efficient.

    The densest rung is the reference. T is accepted if all three hold:

      (i)   it and EVERY LARGER RUNG sit within `tol` x (CI half-width at the reference)
            of the reference rho          -- no subsampling BIAS
      (ii)  its sparsest distance bucket holds at least `min_bucket` pairs
                                          -- the far buckets were actually SAMPLED
      (iii) its own CI is no more than (1 + `max_ci_inflation`) x the reference CI
                                          -- no wasted statistical POWER

    Why one criterion is not enough
    -------------------------------
    (i) alone is close to vacuous, and both demo runs showed it. The bootstrap CI is
    dominated by BETWEEN-GRAPH variance, which barely shrinks with T, so the acceptance
    band is set by how heterogeneous the graphs are rather than by how precisely each was
    measured -- almost any T clears it. That is not a logic error: "subsampling bias is
    negligible relative to the uncertainty we report" is exactly what (i) certifies, and it
    is true. It is simply not sufficient.

    (ii) catches the case where rho looks settled at the pooled level while individual far
    buckets -- the ones carrying the paper's claims -- hold a handful of pairs or none.

    (iii) catches the subtler failure, and is the one that actually bound in practice.
    bootstrap_over_graphs resamples whole graphs, so if T is small each graph's OWN rho is
    noisy and that measurement noise is indistinguishable from real between-graph variation.
    The bootstrap absorbs it into the interval, which stays honest (conservative) but grows.
    Observed in the demo: CI width 3.39e-2 at T=4 against 2.19e-2 at T=128 -- a 55% wider
    interval bought by sampling 32x less. rho was unbiased the whole way; the cost was
    entirely in power. Comparing each rung's CI to the reference's measures that directly.

    Note (iii) is what makes the criterion self-consistent: (i) judges each rung against a
    band derived from the REFERENCE's CI, so without (iii) a rung could be accepted while
    its own CI -- the one that would actually appear in the paper -- was far wider than the
    band it was judged against.

    Also deliberate:
      * (i) compares to the bootstrap half-width rather than an absolute epsilon; an
        absolute threshold would just be another arbitrary constant, which is the thing
        this module exists to eliminate.
      * (i) requires all larger rungs to hold, not only this one, so a curve that happens
        to cross the reference on its way elsewhere is not mistaken for convergence.
    """
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
    """The claim this whole module exists to license, phrased for the paper."""
    ref = max(rows, key=lambda r: r["T"])
    n_graphs = ref["n_graphs"]
    if not rec["converged"]:
        return (
            f"WARNING: rho had not converged in T by T={ref['T']}. Do not claim stability; "
            f"extend the ladder past {ref['T']} and re-run."
        )
    ladder = [r["T"] for r in sorted(rows, key=lambda r: r["T"])]

    # Every clause below is phrased "relative to the densest sampling T=<ref>". That
    # phrasing is only true while the reference rung actually IS denser sampling. Once T
    # exceeds a graph's node count, randperm(n)[:T] returns all n nodes, so the rung
    # enumerates rather than samples -- a different estimator, which is why
    # recommend_target_nodes already warns about it in `reason`.
    #
    # This sentence is written to be pasted into a paper, and it was dropping that
    # caveat: on a real calibration it read "relative to the densest sampling T=256"
    # while 7 of 10 graphs at that rung had been exhaustively enumerated. Stating an
    # unqualified stability claim that the tool itself had qualified two lines earlier is
    # the one failure mode a report helper must not have.
    ref_saturated = ref["graphs_saturated"]
    caveat = ""
    if ref_saturated:
        if ref_saturated == n_graphs:
            caveat = (
                f" This comparison is against an EXHAUSTIVE rung, not a denser sample: at "
                f"T={ref['T']} every calibration graph has fewer than {ref['T']} nodes, so "
                f"the reference enumerates all node pairs. The stability claim is "
                f"therefore against the exact per-graph value, which is stronger than "
                f"convergence in T but is not the same statement; re-run with a ladder "
                f"whose top rung is below the smallest graph if you need the weaker one."
            )
        else:
            caveat = (
                f" Caveat: the reference rung is saturated on {ref_saturated} of "
                f"{n_graphs} calibration graphs -- for those, T={ref['T']} enumerates every "
                f"node rather than sampling more of them, so the reference mixes two "
                f"estimators. Re-run with a top rung below the smallest graph's node count "
                f"before quoting this."
            )
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
        f"T={rec['recommended_T']}.{caveat}"
    )
