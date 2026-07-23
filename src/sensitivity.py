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
callable `model_fn(data) -> node_embeddings [n, hidden]` and the graph's node distances.

Sampling: for tractability we do NOT compute the full n x n Jacobian. We sample
`num_target_nodes` target nodes v, and for each, a bounded number of source nodes u per
distance bucket (`max_pairs_per_bucket`), following the sampling budget in the proposal
(256 test graphs, up to 128 pairs/distance bin).
"""

from collections import defaultdict
from typing import Callable, Dict, List

import networkx as nx
import torch
from torch_geometric.utils import to_networkx


def _all_pairs_by_distance(edge_index, num_nodes, max_dist=None) -> Dict[int, List[tuple]]:
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from(edge_index.t().tolist())
    buckets = defaultdict(list)
    for src, lengths in nx.all_pairs_shortest_path_length(g, cutoff=max_dist):
        for dst, d in lengths.items():
            if d > 0:
                buckets[d].append((src, dst))
    return buckets


def compute_sensitivity_curve(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    data,
    max_dist: int = 20,
    max_pairs_per_bucket: int = 128,
    seed: int = 0,
) -> Dict[int, float]:
    """
    Args:
        model_fn: callable taking `data.x` (with requires_grad=True set by caller before
                  the forward pass) plus any other fields on `data`, returning final-layer
                  node embeddings [n, hidden]. Wrap each backbone's forward pass to match
                  this signature (see run_experiment.py's `make_model_fn`).
        data: a single PyG Data object (one graph) with `.x`, `.edge_index`, `.num_nodes`.
    Returns:
        dict mapping hop distance d -> mean ||d h_v / d x_u|| over sampled pairs.
    """
    rng = torch.Generator().manual_seed(seed)
    buckets = _all_pairs_by_distance(data.edge_index, data.num_nodes, max_dist=max_dist)

    x = data.x.clone().detach().requires_grad_(True)
    h = model_fn(x)  # [n, hidden]

    curve = {}
    for d, pairs in buckets.items():
        if len(pairs) > max_pairs_per_bucket:
            idx = torch.randperm(len(pairs), generator=rng)[:max_pairs_per_bucket].tolist()
            pairs = [pairs[i] for i in idx]

        norms = []
        for (u, v) in pairs:
            # sum over hidden dims of h_v to get a scalar to backprop from, matching the
            # Frobenius-norm-of-Jacobian-row definition used in the proposal
            grad_u = torch.autograd.grad(
                h[v].sum(), x, retain_graph=True, create_graph=False
            )[0][u]
            norms.append(grad_u.norm().item())
        if norms:
            curve[d] = sum(norms) / len(norms)
    return curve


def average_curves(curves: List[Dict[int, float]]) -> Dict[int, float]:
    """Average s_bar(d) across multiple sampled test graphs."""
    agg = defaultdict(list)
    for c in curves:
        for d, v in c.items():
            agg[d].append(v)
    return {d: sum(vs) / len(vs) for d, vs in agg.items()}
