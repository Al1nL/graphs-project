"""
sensitivity.py
==============
Backbone-agnostic long-range sensitivity metric, following Di Giovanni et al. (2023)
"On over-squashing in message passing neural networks" (Jacobian-based sensitivity), used
here to compare PE variants across all three backbones on equal footing.

    s_bar(d) = E_{(u,v): dist(u,v)=d} [ || d h_v^(L) / d x_u^(0) ||_F ]

A slower decay of s_bar(d) with hop distance d indicates the model preserves more
information flow between distant nodes -- this is the core quantity the whole project is
about, so it is intentionally decoupled from any one backbone's internals: it only needs a
callable `model_fn(x) -> node_embeddings [n, p]` and the graph's node distances.

--------------------------------------------------------------------------------------
THE INPUT-SPACE CONTRACT (`n_shared_feats`) -- read before wiring up a backbone
--------------------------------------------------------------------------------------
The Jacobian is taken with respect to `x`, so *what lives in `x` defines the comparison*.
If a PE's channels are concatenated into `x`, then ||.||_F sums over more non-negative
terms for LapPE (+16 dims) and RWSE (+20 dims) than for No-PE, and GRPE -- which enters as
an attention bias and contributes no columns at all -- is structurally disadvantaged. The
five PE variants would then be ranked partly by how many input channels they add.

`n_shared_feats` fixes this: only the FIRST `n_shared_feats` columns of `x` are
differentiated against, so every PE variant is measured on an identical input space.

Two valid ways to satisfy the contract:
  (a) PE concatenated into x  -> lay out x as [shared_original_features | PE channels]
                                 and pass n_shared_feats = (original feature width).
  (b) PE enters via a separate path (closure over cached tensors, attention bias, ...)
                              -> pass n_shared_feats = x.shape[1].
Either way the value MUST be identical across all five PE variants for a given dataset;
`assert_shared_width` below is provided to enforce that at the call site.

Semantically (b) is the cleaner design: a PE channel describes *graph structure*, not node
u's *content*, so sensitivity to it was never the quantity of interest at any width.

--------------------------------------------------------------------------------------
Estimator notes
--------------------------------------------------------------------------------------
We compute the TRUE Frobenius norm of each Jacobian block, via the identity

    ||J||_F^2 = sum_k || J^T e_k ||^2

i.e. the sum of squared vector-Jacobian products over an orthonormal basis of the output
space. Collapsing the output first (e.g. backprop from `h[v].sum()`) instead measures
|| J^T 1 ||, the norm of the Jacobian's *column sum*, which is a different quantity:
signed cancellation across hidden channels can drive it to zero for a genuinely strongly
coupled pair. It is also worst-case degenerate under LayerNorm, which annihilates the
all-ones direction exactly (at init, where gamma = 1, the gradient of 1^T h is identically
zero), so the size of that error would vary by backbone -- precisely the confound this
project needs to avoid.

Because the identity is a sum over output directions, it is *exactly* decomposable: we
accumulate it in chunks of `chunk_size` basis vectors to bound peak memory.

Sampling: we sample `num_target_nodes` target nodes v and, for each, take ONE batched
backward pass that yields the full [p, n, q] Jacobian block -- giving exact norms for
*every* source node u at once. Bucketing by dist(u, v) then costs nothing. (Sampling
pairs and backpropagating per pair recomputes the same rows O(n) times over.)

Each returned bucket carries its pair `count` alongside the mean, so curves can be pooled
across graphs by pair count rather than by unweighted graph average -- far pairs are
concentrated in large graphs, so an unweighted average is biased, not merely noisy.
"""

import warnings
from collections import defaultdict
from typing import Callable, Dict, List

import networkx as nx
import torch


def _build_nx_graph(edge_index, num_nodes) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from(edge_index.t().tolist())
    return g


def _distances_from(g: nx.Graph, v: int, num_nodes: int, max_dist: int) -> torch.Tensor:
    """Hop distance from every node to target `v`; -1 for unreachable or beyond max_dist.

    Single-source BFS per sampled target, rather than all-pairs: we only ever need the
    columns of the distance matrix belonging to sampled targets.
    """
    d = torch.full((num_nodes,), -1, dtype=torch.long)
    for u, dist in nx.single_source_shortest_path_length(g, v, cutoff=max_dist).items():
        d[u] = dist
    return d


def _frobenius_per_source(h_v, x, n_shared_feats, chunk_size, batched_ok):
    """|| d h_v / d x_u ||_F for every source node u, as a [n] tensor.

    Returns (norms, batched_ok). `batched_ok` is threaded through so the vmap capability
    probe happens once per run, not once per target node.
    """
    p = h_v.numel()
    acc = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    basis = torch.eye(p, device=h_v.device, dtype=h_v.dtype)

    for chunk in basis.split(chunk_size):
        jac = None
        if batched_ok:
            try:
                (jac,) = torch.autograd.grad(
                    h_v, x, grad_outputs=chunk, is_grads_batched=True, retain_graph=True
                )
            except (RuntimeError, NotImplementedError) as exc:
                # is_grads_batched runs under vmap; an op in this backbone may lack a
                # batching rule. Fall back to a plain loop -- same number, less speed.
                warnings.warn(
                    f"is_grads_batched failed ({type(exc).__name__}: {exc}); falling back "
                    "to an unbatched VJP loop for the rest of this run. Results are "
                    "unchanged, throughput is lower.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                batched_ok = False
        if jac is None:
            jac = torch.stack([
                torch.autograd.grad(h_v, x, grad_outputs=w, retain_graph=True)[0]
                for w in chunk
            ])
        # jac: [chunk, n, q] -> slice to the shared input channels, accumulate ||.||_F^2
        acc = acc + jac[:, :, :n_shared_feats].pow(2).sum(dim=(0, 2))

    return acc.sqrt().detach(), batched_ok


def compute_sensitivity_curve(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    data,
    n_shared_feats: int,
    max_dist: int = 20,
    num_target_nodes: int = None,
    chunk_size: int = 16,
    seed: int = 0,
    _batched_ok: bool = True,
) -> Dict[int, Dict[str, float]]:
    """
    Args:
        model_fn: callable taking the input feature tensor `x` (this function sets
                  requires_grad on it) and returning final-layer node embeddings [n, p].
                  Wrap each backbone's forward pass to match (see run_experiment.py's
                  `make_model_fn`). Any PE that does not live in `x` should be closed over.
        data: a single PyG Data object with `.x`, `.edge_index`, `.num_nodes`.
        n_shared_feats: number of leading columns of `x` to differentiate against. MUST be
                  identical across all five PE variants -- see the input-space contract at
                  the top of this module.
        max_dist: hop-distance cutoff for bucketing.
        num_target_nodes: target nodes v sampled per graph (all of them if n is smaller).
                  REQUIRED, deliberately with no default: this controls how much of each
                  graph's distance profile is observed, and the paper's claims live in the
                  sparse far buckets. Calibrate it per (backbone, dataset) with
                  `scripts/calibrate_target_nodes.py`, which sweeps it and reports the
                  smallest value at which rho is stable AND the tail buckets are populated.
                  A default here would just be an arbitrary constant wearing a plausible
                  number, which is what that script exists to eliminate.
        chunk_size: output basis vectors per batched backward pass; lower to cut memory.
    Returns:
        {hop distance d: {"mean": mean ||d h_v / d x_u||_F, "count": pairs in the bucket}}
    """
    if num_target_nodes is None:
        raise ValueError(
            "num_target_nodes is required and has no default. Calibrate it for this "
            "(backbone, dataset) with `python scripts/calibrate_target_nodes.py "
            "--backbone ... --pe ... --dataset ...` and pass the value it reports; see "
            "docs/analysis-plan.md for why it is not allowed to be an arbitrary constant."
        )
    if num_target_nodes < 1:
        raise ValueError(f"num_target_nodes must be >= 1, got {num_target_nodes}")

    x = data.x.clone().detach().requires_grad_(True)
    if x.dim() != 2:
        raise ValueError(f"expected data.x of shape [n, q], got {tuple(x.shape)}")
    if not 1 <= n_shared_feats <= x.shape[1]:
        raise ValueError(
            f"n_shared_feats={n_shared_feats} out of range for x with {x.shape[1]} "
            "columns; pass the shared original feature width (or x.shape[1] if this "
            "backbone feeds its PE in through a separate path)"
        )

    h = model_fn(x)
    if h.dim() != 2 or h.shape[0] != x.shape[0]:
        raise ValueError(
            f"model_fn must return node embeddings [n, p]; got {tuple(h.shape)} for "
            f"n={x.shape[0]}"
        )

    n = data.num_nodes
    g = _build_nx_graph(data.edge_index, n)
    rng = torch.Generator().manual_seed(seed)
    targets = torch.randperm(n, generator=rng)[:num_target_nodes].tolist()

    sums = torch.zeros(max_dist + 1, dtype=torch.float64)
    counts = torch.zeros(max_dist + 1, dtype=torch.float64)

    for v in targets:
        norms, _batched_ok = _frobenius_per_source(
            h[v], x, n_shared_feats, chunk_size, _batched_ok
        )
        dists = _distances_from(g, v, n, max_dist)
        keep = dists >= 1  # drops d == 0 (u is v), unreachable, and beyond max_dist
        if not bool(keep.any()):
            continue
        sums.index_add_(0, dists[keep], norms[keep].double().cpu())
        counts.index_add_(0, dists[keep], torch.ones(int(keep.sum()), dtype=torch.float64))

    return {
        d: {"mean": (sums[d] / counts[d]).item(), "count": int(counts[d])}
        for d in range(1, max_dist + 1)
        if counts[d] > 0
    }


def average_curves(curves: List[Dict[int, Dict[str, float]]]) -> Dict[int, Dict[str, float]]:
    """Pool s_bar(d) across sampled test graphs, weighted by each bucket's pair count.

    Weighting matters: the number of far pairs a graph contributes scales with its size,
    so an unweighted mean over graphs lets a graph offering 2 pairs at d=30 count as much
    as one offering 128 -- and since sensitivity itself correlates with graph size, that
    is a biased estimator, not just a noisy one.

    NOTE the pooled `count` treats pairs as exchangeable, which they are not (pairs within
    one graph share a model and a topology). Use it for weighting, not for standard
    errors; for uncertainty, compute a per-graph statistic and bootstrap over graphs.
    """
    sums = defaultdict(float)
    counts = defaultdict(int)
    for c in curves:
        for d, rec in c.items():
            d = int(d)
            sums[d] += rec["mean"] * rec["count"]
            counts[d] += rec["count"]
    return {d: {"mean": sums[d] / counts[d], "count": counts[d]} for d in sorted(sums)}


# ---------------------------------------------------------------------------
# Scale-free summaries of a curve
# ---------------------------------------------------------------------------
# Raw s_bar(d) is NOT comparable across backbones. Model it as
#
#       s_bar(d) ~= C * f(d)
#
# where C is an overall gain and f is the decay shape. The scientific claim is
# about f, but the raw curve reports the product, and C absorbs weight init
# scale, LayerNorm-vs-BatchNorm placement, residual scaling, depth and hidden
# width -- none of which are the PE. A backbone with 10x the gain "wins" at
# every d regardless of how fast it decays.
#
# We therefore report three things, in increasing order of authority:
#   raw s_bar(d)                     -- appendix; reproducibility, and it exposes
#                                       vanishing/exploding pathologies that
#                                       ratios hide
#   s_tilde(d) = s_bar(d)/s_bar(1)   -- secondary; divides out C, but anchors on
#                                       a single architecture-loaded point (d=1
#                                       is where GraphGPS's GatedGCN branch
#                                       dominates and where Graphormer has no
#                                       message-passing branch at all), and its
#                                       noise propagates into every other point
#   rho                              -- PRIMARY; a ratio of sums, so C cancels
#                                       exactly and algebraically
#
# Why rho rather than fitting s_bar(d) ~ exp(-lambda*d) and ranking by lambda:
# GT decay curves flatten, because global attention supplies a distance-
# independent floor. Fitting an exponential to a decay-then-flat curve yields a
# lambda dominated by the early hops, and the fitted parameter then describes
# the misspecification rather than the data. rho assumes no functional form --
# it is a nonparametric tail-mass statistic, the ECDF-tail analogue for curves.


def normalized_curve(curve: Dict[int, Dict[str, float]], anchor: int = 1) -> Dict[int, float]:
    """s_tilde(d) = s_bar(d) / s_bar(anchor). Secondary statistic -- see note above."""
    curve = {int(d): rec for d, rec in curve.items()}
    if anchor not in curve:
        raise KeyError(f"anchor d={anchor} not populated; buckets present: {sorted(curve)}")
    a = curve[anchor]["mean"]
    if a == 0:
        raise ZeroDivisionError(f"s_bar({anchor}) == 0; cannot normalize")
    return {d: rec["mean"] / a for d, rec in sorted(curve.items())}


def long_range_fraction(
    curve: Dict[int, Dict[str, float]],
    d_min: int,
    d_max: int,
    weight_by_count: bool = False,
) -> float:
    """rho = sum_{d >= d_min} s_bar(d) / sum_{d >= 1} s_bar(d), both truncated at d_max.

    PRIMARY statistic: scale-free, one number per cell, directly rankable, and robust to
    the fact that GT curves are often not exponential.

    rho is only comparable across cells when (d_min, d_max) are IDENTICAL -- it is a
    property of the pair (curve, window), not of the curve alone. Always report the window
    next to the number, and see docs/analysis-plan.md for the values in force.

    `weight_by_count` picks which question rho answers:
      False (default) -- each DISTANCE contributes equally: rho describes the curve's
                         SHAPE, and is sensitive to the tail, which is the point.
      True            -- each PAIR contributes equally: rho describes actual information
                         flow in a typical graph, and is dominated by the mid-range
                         buckets where most pairs live.
    Either is defensible and neither confounds the comparison (the distance distribution
    is a property of the dataset, identical across PEs and backbones), but they give
    different numbers, so the choice must be stated rather than left implicit.

    Buckets with no sampled pairs are skipped, contributing to neither sum.
    """
    if d_min < 1 or d_max < d_min:
        raise ValueError(f"need 1 <= d_min <= d_max, got d_min={d_min}, d_max={d_max}")
    num = den = 0.0
    for d, rec in curve.items():
        d = int(d)
        if not 1 <= d <= d_max:
            continue
        val = rec["mean"] * (rec["count"] if weight_by_count else 1.0)
        den += val
        if d >= d_min:
            num += val
    return num / den if den > 0 else float("nan")


def bootstrap_over_graphs(
    per_graph_curves: List[Dict[int, Dict[str, float]]],
    stat_fn: Callable[[Dict[int, Dict[str, float]]], float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
):
    """Point estimate and confidence interval for a curve statistic, CLUSTERED BY GRAPH.

    Resamples whole graphs with replacement, re-pools, and recomputes the statistic.

    This is the part that survives review. Node pairs within one graph are NOT
    independent -- they share a model and a topology, and their paths overlap -- so
    treating ~50,000 sampled pairs as 50,000 independent observations understates the
    standard error badly. Resampling at the level of the independent unit (the graph)
    gives n = number of sampled graphs, which is smaller, honest, and immune to the
    "your error bars assume independence you don't have" objection.

    Returns (point, lo, hi); lo/hi are NaN if fewer than 2 graphs are supplied.
    """
    if not per_graph_curves:
        return float("nan"), float("nan"), float("nan")
    point = stat_fn(average_curves(per_graph_curves))
    n_graphs = len(per_graph_curves)
    if n_graphs < 2:
        return point, float("nan"), float("nan")

    rng = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(n_boot):
        idx = torch.randint(n_graphs, (n_graphs,), generator=rng).tolist()
        v = stat_fn(average_curves([per_graph_curves[i] for i in idx]))
        if v == v:  # drop NaN
            vals.append(v)
    if len(vals) < 2:
        return point, float("nan"), float("nan")
    vals.sort()
    lo_i = int((1 - ci) / 2 * len(vals))
    hi_i = min(len(vals) - 1, int((1 + ci) / 2 * len(vals)))
    return point, vals[lo_i], vals[hi_i]


def assert_shared_width(widths_by_pe: Dict[str, int]) -> int:
    """Guard for the input-space contract: every PE variant must be probed on the same
    input width. Call this once per (backbone, dataset) before the grid runs.

        assert_shared_width({"none": 9, "lappe": 9, "rwse": 9, "signnet": 9, "grpe": 9})
    """
    distinct = set(widths_by_pe.values())
    if len(distinct) != 1:
        raise ValueError(
            "n_shared_feats differs across PE variants, so their Jacobian norms are not "
            f"comparable: {widths_by_pe}. Every variant must be differentiated against "
            "the same shared original node-feature channels."
        )
    return distinct.pop()
