"""
compute_pe.py
=============
Backbone-agnostic computation of the 5 positional-encoding (PE) variants used in the
study: No-PE, LapPE, RWSE, SignNet-PE, GRPE.

Design intent: every backbone (GraphGPS, SAN, Graphormer) must see the *same* underlying
PE definition, so that a difference in downstream metric or long-range sensitivity can be
attributed to how the backbone *uses* the PE rather than a difference in how the PE itself
was computed. This module is the single source of truth; the three adapters in
`src/adapters/` only reshape these tensors into whatever input format each backbone's code
expects (node-feature concat vs. attention bias, etc.).

Usage:
    python compute_pe.py --dataset peptides-func --out cache/peptides-func/

Output: one .pt file per graph (or one sharded .pt per split) containing a dict:
    {
      "lap_pe":      FloatTensor [n, k]        top-k non-trivial Laplacian eigenvectors
      "lap_eigvals": FloatTensor [k]           corresponding eigenvalues
      "rwse":        FloatTensor [n, k]        k-step random-walk landing probabilities
      "signnet_in":  FloatTensor [n, k]        raw input to the SignNet encoder (= lap_pe,
                                                 sign ambiguity resolved inside the encoder,
                                                 not here -- see SignNetEncoder below)
      "spd":         LongTensor  [n, n]        all-pairs shortest-path distance (capped)
      "edge_type_id":LongTensor  [n, n]        bucketed edge-type id per pair, for GRPE bias
    }
"""

import argparse
import os
import sys
import networkx as nx
import numpy as np
import scipy.linalg
import torch
from torch_geometric.utils import to_networkx, get_laplacian, to_dense_adj
from torch_geometric.datasets import LRGBDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_meta import DATASETS  # noqa: E402
from cache import PECacheWriter, SPLITS, estimate_cache_bytes  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
K_LAP = 16          # number of non-trivial Laplacian eigenvectors kept (LapPE, SignNet-PE input)
K_RWSE = 20         # number of random-walk steps for RWSE
# GRPE's distance bucketing now lives in src/dataset_meta.py -- it is a MODEL parameter
# shared by all three adapters, and duplicating it here is what previously let the probe's
# measurement cap and the model's resolution drift apart. See that module for the scheme.

# Approximate split-total graph counts, used only for the pre-flight size estimate.
_APPROX_GRAPH_COUNT = {
    "peptides-func": 15535, "peptides-struct": 15535, "pascalvoc-sp": 11355,
}

DATASET_NAME_MAP = {
    "peptides-func": "Peptides-func",
    "peptides-struct": "Peptides-struct",
    "pascalvoc-sp": "PascalVOC-SP",
}


def compute_lap_pe(edge_index, num_nodes, k=K_LAP):
    """Smallest-k non-trivial eigenvectors/eigenvalues of the normalized graph Laplacian.

    Uses a PARTIAL solver: we need the lowest k+1 eigenpairs, and `np.linalg.eigh` computes
    all n of them. On PascalVOC-SP (n ~ 480) that dominated the whole precompute at 449 of
    601 ms per graph. `scipy.linalg.eigh(..., subset_by_index=[0, k])` dispatches to LAPACK
    syevr, which computes only the requested range: 11.2x faster, and numerically identical
    on real VOC graphs (eigenvalues agree to 5.6e-16, eigenvectors to 8.9e-15 in absolute
    value). Total precompute drops from ~2h to ~45min.

    Note the comparison is on |eigenvector|: sign is arbitrary for any eigensolver, and in a
    DEGENERATE eigenspace so is the basis. That ambiguity is inherent to LapPE, not
    introduced here -- it is exactly what SignNet-PE exists to be invariant to.
    """
    lap_index, lap_weight = get_laplacian(edge_index, normalization="sym", num_nodes=num_nodes)
    L = to_dense_adj(lap_index, edge_attr=lap_weight, max_num_nodes=num_nodes)[0].numpy()
    k_eff = min(k, num_nodes - 1)
    # indices 0..k_eff inclusive -> k_eff+1 pairs; index 0 is the trivial lambda=0 one
    eigvals, eigvecs = scipy.linalg.eigh(L, subset_by_index=[0, k_eff])
    vals = eigvals[1:1 + k_eff]
    vecs = eigvecs[:, 1:1 + k_eff]
    if k_eff < k:  # pad small graphs
        vals = np.pad(vals, (0, k - k_eff))
        vecs = np.pad(vecs, ((0, 0), (0, k - k_eff)))
    return torch.tensor(vecs, dtype=torch.float32), torch.tensor(vals, dtype=torch.float32)


def compute_rwse(edge_index, num_nodes, k=K_RWSE):
    """Diagonal of the k-step random walk transition matrix (return probabilities)."""
    A = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].numpy()
    deg = A.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    P = A / deg
    diag = np.zeros((num_nodes, k))
    Pk = np.eye(num_nodes)
    for step in range(k):
        Pk = Pk @ P
        diag[:, step] = np.diag(Pk)
    return torch.tensor(diag, dtype=torch.float32)


def compute_spd(edge_index, num_nodes):
    """All-pairs shortest-path distance + a trivial edge-type bucket id, both needed for
    GRPE-style attention bias (Park et al., 2022). We use unweighted BFS distance; for
    PascalVOC-SP, edge weights exist but GRPE's original formulation buckets by hop count,
    not by weight, so we follow that convention here for consistency across datasets.

    Returns the raw hop-distance matrix, UNCAPPED, with -1 for unreachable.

    `spd_bucket` (the GRPE bias-table index) and `edge_type_id` are NOT returned or stored:
    both are pure functions of this matrix, so caching them tripled the on-disk footprint
    and made the cache stale whenever the bucketing scheme changed. cache.py derives them
    on read via a 256-entry lookup table.

    Two bugs fixed here relative to the original version, both of which silently corrupted
    the GRPE arm:

    1. Unreachable pairs were initialised to the cap and BFS ran with `cutoff=cap`, so a
       pair in a different connected component and a pair at distance >= cap ended up with
       the SAME value. GRPE then learned one bias meaning "far OR disconnected" -- two
       different structural relations collapsed into one parameter. They now get distinct
       buckets.

    2. The cap was applied destructively at cache time, so raising the model's distance
       resolution required recomputing the entire PE cache rather than editing a config.
       Raw distances are now stored uncapped and bucketed at model-input time.
    """
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from(edge_index.t().tolist())
    spd = np.full((num_nodes, num_nodes), -1, dtype=np.int64)  # -1 = unreachable
    for src, lengths in nx.all_pairs_shortest_path_length(g):   # no cutoff: keep the tail
        for dst, d in lengths.items():
            spd[src, dst] = d
    return spd


def _graph_records(ds):
    """Yield (lap_pe, lap_eigvals, rwse, spd) per graph. A GENERATOR on purpose: the writer
    consumes it one graph at a time, so no split is ever fully resident."""
    for data in ds:
        n = data.num_nodes
        lap_pe, lap_eigvals = compute_lap_pe(data.edge_index, n)
        rwse = compute_rwse(data.edge_index, n)
        spd = compute_spd(data.edge_index, n)
        yield lap_pe.numpy(), lap_eigvals.numpy(), rwse.numpy(), spd


def process_dataset(name, out_dir):
    """Precompute every PE for one dataset, once, streaming to disk.

    All five variants are served from this single pass: LapPE and SignNet-PE from the
    Laplacian eigenvectors (SignNet's sign-invariance lives in its learned encoder, not
    here), RWSE from the random-walk return probabilities, GRPE from the all-pairs
    distances, and No-PE from nothing at all.
    """
    os.makedirs(out_dir, exist_ok=True)
    pyg_name = DATASET_NAME_MAP[name]

    meta = DATASETS.get(name, {})
    if meta:
        est = estimate_cache_bytes(
            n_graphs=_APPROX_GRAPH_COUNT.get(name, 0),
            avg_nodes=meta.get("avg_nodes", 0), k_lap=K_LAP, k_rwse=K_RWSE)
        print(f"[{name}] estimated cache size ~{est['total_gb']:.1f} GB "
              f"(dense term {est['dense_bytes'] / 1e9:.1f} GB, quadratic in node count)")

    writer = PECacheWriter(out_dir, name, K_LAP, K_RWSE)
    for split in SPLITS:
        ds = LRGBDataset(root=f"./raw_data/{pyg_name}", name=pyg_name, split=split)
        writer.write_split(split, _graph_records(ds), total=len(ds))
    return writer.finalize()


class SignNetEncoder(torch.nn.Module):
    """Sign- and basis-invariant encoder over Laplacian eigenvectors (Lim et al., 2023).
    This is applied *inside* the GraphGPS/SAN model at train time (it has learnable
    parameters), not baked into the cached PE file -- the cache only stores the raw,
    sign-ambiguous eigenvectors it consumes.
    """

    def __init__(self, k=K_LAP, hidden=64, out_dim=32):
        super().__init__()
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(1, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden)
        )
        self.rho = torch.nn.Sequential(
            torch.nn.Linear(hidden * k, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, out_dim)
        )
        self.k = k

    def forward(self, eigvecs):  # eigvecs: [n, k]
        n = eigvecs.shape[0]
        v = eigvecs.unsqueeze(-1)             # [n, k, 1]
        pos = self.phi(v)                     # [n, k, hidden]
        neg = self.phi(-v)                    # [n, k, hidden]
        sign_inv = pos + neg                  # sign invariance: phi(v) + phi(-v)
        sign_inv = sign_inv.reshape(n, -1)    # [n, k*hidden]
        return self.rho(sign_inv)             # [n, out_dim]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_NAME_MAP.keys()))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    process_dataset(args.dataset, args.out)
