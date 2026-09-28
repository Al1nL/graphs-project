"""
compute_pe.py
=============
Backbone-agnostic computation of positional encodings: No-PE, LapPE, RWSE, SignNet-PE.

Output per graph:
    {
      "lap_pe":      FloatTensor [n, k]  top-k Laplacian eigenvectors
      "lap_eigvals": FloatTensor [k]     corresponding eigenvalues
      "rwse":        FloatTensor [n, k]  k-step random-walk landing probabilities
      "signnet_in":  FloatTensor [n, k]  raw input to SignNet encoder (= lap_pe)
      "spd":         LongTensor  [n, n]  all-pairs shortest-path distance
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

K_LAP = 16
K_RWSE = 20

_APPROX_GRAPH_COUNT = {
    "peptides-func": 15535, "peptides-struct": 15535, "pascalvoc-sp": 11355,
}

DATASET_NAME_MAP = {
    "peptides-func": "Peptides-func",
    "peptides-struct": "Peptides-struct",
    "pascalvoc-sp": "PascalVOC-SP",
}


def compute_lap_pe(edge_index, num_nodes, k=K_LAP):
    """Smallest-k non-trivial eigenvectors/eigenvalues of the normalized graph Laplacian."""
    lap_index, lap_weight = get_laplacian(edge_index, normalization="sym", num_nodes=num_nodes)
    L = to_dense_adj(lap_index, edge_attr=lap_weight, max_num_nodes=num_nodes)[0].numpy()
    k_eff = min(k, num_nodes - 1)
    eigvals, eigvecs = scipy.linalg.eigh(L, subset_by_index=[0, k_eff])
    vals = eigvals[1:1 + k_eff]
    vecs = eigvecs[:, 1:1 + k_eff]
    if k_eff < k:
        vals = np.pad(vals, (0, k - k_eff))
        vecs = np.pad(vecs, ((0, 0), (0, k - k_eff)))
    return torch.tensor(vecs, dtype=torch.float32), torch.tensor(vals, dtype=torch.float32)


def compute_rwse(edge_index, num_nodes, k=K_RWSE):
    """Diagonal of the k-step random walk transition matrix."""
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
    """All-pairs shortest-path distance matrix, with -1 for unreachable nodes."""
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from(edge_index.t().tolist())
    spd = np.full((num_nodes, num_nodes), -1, dtype=np.int64)
    for src, lengths in nx.all_pairs_shortest_path_length(g):
        for dst, d in lengths.items():
            spd[src, dst] = d
    return spd


def _graph_records(ds):
    """Yield (lap_pe, lap_eigvals, rwse, spd) per graph."""
    for data in ds:
        n = data.num_nodes
        lap_pe, lap_eigvals = compute_lap_pe(data.edge_index, n)
        rwse = compute_rwse(data.edge_index, n)
        spd = compute_spd(data.edge_index, n)
        yield lap_pe.numpy(), lap_eigvals.numpy(), rwse.numpy(), spd


def process_dataset(name, out_dir):
    """Precompute PEs for one dataset and stream to disk."""
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
