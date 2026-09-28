"""
sensitivity.py
==============
Backbone-agnostic long-range sensitivity metric, following Di Giovanni et al. (2023)
"On over-squashing in message passing neural networks" (Jacobian-based sensitivity).

    s_bar(d) = E_{(u,v): dist(u,v)=d} [ || d h_v^(L) / d x_u^(0) ||_F ]

Computes the Frobenius norm of node embedding outputs with respect to input features
differentiated against `n_shared_feats` leading feature columns across hop distances d.
"""

import math
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


def graph_diameter(edge_index, num_nodes: int) -> int:
    """Longest shortest-path distance in the graph; for disconnected graphs, the largest finite one."""
    g = _build_nx_graph(edge_index, num_nodes)
    best = 0
    for _, lengths in nx.all_pairs_shortest_path_length(g):
        if lengths:
            best = max(best, max(lengths.values()))
    return best


def _distances_from(g: nx.Graph, v: int, num_nodes: int, max_dist: int) -> torch.Tensor:
    """Hop distance from every node to target `v`; -1 for unreachable or beyond max_dist."""
    d = torch.full((num_nodes,), -1, dtype=torch.long)
    for u, dist in nx.single_source_shortest_path_length(g, v, cutoff=max_dist).items():
        d[u] = dist
    return d


def _frobenius_per_source(h_v, x, n_shared_feats, chunk_size, batched_ok):
    """Compute || d h_v / d x_u ||_F for every source node u, returning (norms, batched_ok)."""
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
            except (RuntimeError, NotImplementedError, TypeError) as exc:
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
    """Compute mean Frobenius sensitivity ||d h_v / d x_u||_F binned by hop distance d.

    Args:
        model_fn: Callable `x -> node_embeddings [n, p]`.
        data: PyG Data object with `.x`, `.edge_index`, `.num_nodes`.
        n_shared_feats: Number of leading columns of `x` to differentiate against.
        max_dist: Hop-distance cutoff for bucketing.
        num_target_nodes: Number of target nodes v sampled per graph.
        chunk_size: Output basis vectors per batched backward pass.
        seed: Random seed for target node sampling.

    Returns:
        {hop distance d: {"mean": mean_sensitivity, "count": pair_count}}
    """
    if num_target_nodes is None:
        raise ValueError("num_target_nodes is required. Calibrate it using scripts/calibrate_target_nodes.py.")
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


def to_relative_curve(
    curve: Dict[int, Dict[str, float]],
    diameter: int,
    n_bins: int = 10,
) -> Dict[int, Dict[str, float]]:
    """Re-index a graph's curve from absolute hop distance d to relative distance d / diam(G)."""
    if diameter <= 0:
        return {}
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    sums, counts = defaultdict(float), defaultdict(int)
    for d, rec in curve.items():
        d = int(d)
        if d < 1:
            continue
        # d/diam in (0, 1] -> bin 1..n_bins; d beyond the diameter clamps in.
        b = min(n_bins, max(1, math.ceil(d / diameter * n_bins)))
        sums[b] += rec["mean"] * rec["count"]
        counts[b] += rec["count"]
    return {b: {"mean": sums[b] / counts[b], "count": counts[b]} for b in sorted(sums)}


def average_curves(curves: List[Dict[int, Dict[str, float]]]) -> Dict[int, Dict[str, float]]:
    """Pool s_bar(d) across sampled test graphs, weighted by each bucket's pair count."""
    sums = defaultdict(float)
    counts = defaultdict(int)
    graphs = defaultdict(int)
    for c in curves:
        for d, rec in c.items():
            d = int(d)
            sums[d] += rec["mean"] * rec["count"]
            counts[d] += rec["count"]
            graphs[d] += 1
    return {
        d: {"mean": sums[d] / counts[d], "count": counts[d], "n_graphs": graphs[d]}
        for d in sorted(sums)
    }


# ---------------------------------------------------------------------------
# Scale-free summaries of a curve
# ---------------------------------------------------------------------------


def normalized_curve(curve: Dict[int, Dict[str, float]], anchor: int = 1) -> Dict[int, float]:
    """Compute normalized curve s_tilde(d) = s_bar(d) / s_bar(anchor)."""
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
    """Compute long-range fraction rho = sum_{d >= d_min} s_bar(d) / sum_{d >= 1} s_bar(d)."""
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
    groups: List = None,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
):
    """Resample whole clusters (graphs) with replacement to compute point estimate and confidence interval."""
    if not per_graph_curves:
        return float("nan"), float("nan"), float("nan")
    point = stat_fn(average_curves(per_graph_curves))

    if groups is None:
        members = [[i] for i in range(len(per_graph_curves))]
    else:
        if len(groups) != len(per_graph_curves):
            raise ValueError(
                f"groups has {len(groups)} entries but there are "
                f"{len(per_graph_curves)} curves; they must be parallel"
            )
        by_key = defaultdict(list)
        for i, key in enumerate(groups):
            by_key[key].append(i)
        members = list(by_key.values())

    n_clusters = len(members)
    if n_clusters < 2:
        return point, float("nan"), float("nan")

    rng = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(n_boot):
        drawn = torch.randint(n_clusters, (n_clusters,), generator=rng).tolist()
        sample = [per_graph_curves[j] for c in drawn for j in members[c]]
        v = stat_fn(average_curves(sample))
        if v == v:  # drop NaN
            vals.append(v)
    if len(vals) < 2:
        return point, float("nan"), float("nan")
    vals.sort()
    lo_i = int((1 - ci) / 2 * len(vals))
    hi_i = min(len(vals) - 1, int((1 + ci) / 2 * len(vals)))
    return point, vals[lo_i], vals[hi_i]


def assert_shared_width(widths_by_pe: Dict[str, int]) -> int:
    """Guard for input feature contract: verify all PE variants use identical shared input widths."""
    distinct = set(widths_by_pe.values())
    if len(distinct) != 1:
        raise ValueError(
            "n_shared_feats differs across PE variants, so their Jacobian norms are not "
            f"comparable: {widths_by_pe}. Every variant must be differentiated against "
            "the same shared original node-feature channels."
        )
    return distinct.pop()

