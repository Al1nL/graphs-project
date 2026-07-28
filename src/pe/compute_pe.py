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
import torch
from torch_geometric.utils import to_networkx, get_laplacian, to_dense_adj
from torch_geometric.datasets import LRGBDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_meta import SPD_NUM_BUCKETS, spd_bucket_id  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
K_LAP = 16          # number of non-trivial Laplacian eigenvectors kept (LapPE, SignNet-PE input)
K_RWSE = 20         # number of random-walk steps for RWSE
# GRPE's distance bucketing now lives in src/dataset_meta.py -- it is a MODEL parameter
# shared by all three adapters, and duplicating it here is what previously let the probe's
# measurement cap and the model's resolution drift apart. See that module for the scheme.

DATASET_NAME_MAP = {
    "peptides-func": "Peptides-func",
    "peptides-struct": "Peptides-struct",
    "pascalvoc-sp": "PascalVOC-SP",
}


def compute_lap_pe(edge_index, num_nodes, k=K_LAP):
    """Smallest-k non-trivial eigenvectors/eigenvalues of the normalized graph Laplacian."""
    lap_index, lap_weight = get_laplacian(edge_index, normalization="sym", num_nodes=num_nodes)
    L = to_dense_adj(lap_index, edge_attr=lap_weight, max_num_nodes=num_nodes)[0].numpy()
    eigvals, eigvecs = np.linalg.eigh(L)  # ascending order, eigvals[0] ~ 0
    k_eff = min(k, num_nodes - 1)
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


def compute_spd_and_edge_buckets(edge_index, num_nodes):
    """All-pairs shortest-path distance + a trivial edge-type bucket id, both needed for
    GRPE-style attention bias (Park et al., 2022). We use unweighted BFS distance; for
    PascalVOC-SP, edge weights exist but GRPE's original formulation buckets by hop count,
    not by weight, so we follow that convention here for consistency across datasets.

    Returns (spd, spd_bucket, edge_type_id):
      spd         raw hop distance, UNCAPPED, -1 for unreachable
      spd_bucket  GRPE bias-table index via dataset_meta.spd_bucket_id
      edge_type_id 0 = no edge, 1 = edge exists

    Two bugs fixed here relative to the previous version, both of which silently corrupted
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
    bucket = np.vectorize(spd_bucket_id, otypes=[np.int64])(spd)
    edge_type_id = (spd == 1).astype(np.int64)
    return torch.tensor(spd), torch.tensor(bucket), torch.tensor(edge_type_id)


def process_dataset(name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pyg_name = DATASET_NAME_MAP[name]
    for split in ("train", "val", "test"):
        ds = LRGBDataset(root=f"./raw_data/{pyg_name}", name=pyg_name, split=split)
        split_records = []
        for i, data in enumerate(ds):
            n = data.num_nodes
            lap_pe, lap_eigvals = compute_lap_pe(data.edge_index, n)
            rwse = compute_rwse(data.edge_index, n)
            spd, spd_bucket, edge_type_id = compute_spd_and_edge_buckets(data.edge_index, n)
            split_records.append({
                "lap_pe": lap_pe,
                "lap_eigvals": lap_eigvals,
                "rwse": rwse,
                "signnet_in": lap_pe.clone(),  # SignNet encoder consumes raw eigenvectors
                "spd": spd,                    # raw hop distance, uncapped, -1 = unreachable
                "spd_bucket": spd_bucket,      # GRPE bias-table index
                "edge_type_id": edge_type_id,
            })
            if (i + 1) % 500 == 0:
                print(f"[{name}/{split}] {i + 1}/{len(ds)} graphs processed")
        torch.save(split_records, os.path.join(out_dir, f"{split}_pe.pt"))
        print(f"Saved {out_dir}/{split}_pe.pt ({len(split_records)} graphs)")


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
