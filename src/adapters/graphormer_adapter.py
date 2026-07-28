"""
graphormer_adapter.py
======================
Maps the shared PE cache into Graphormer's (Ying et al., 2021) input format.

Graphormer's whole design point is that structure enters as an *attention bias*
(spatial encoding = shortest-path bucket, edge encoding = edge feature along the path) plus
one feature-level term (centrality encoding = degree). It does not have a slot for
dense feature-level PEs like LapPE/RWSE/SignNet-PE.

- No-PE: centrality encoding only (Graphormer's own default already includes this --
  "no PE" here means "no *additional* PE beyond Graphormer's built-in centrality term",
  stated explicitly in the paper so the baseline is comparable across backbones).
- GRPE: **native fit.** GRPE (Park et al., 2022) was explicitly designed as a drop-in
  replacement/extension of Graphormer's spatial+edge bias terms, adding node-to-relation
  interaction that vanilla Graphormer's bias lacks. We use `spd` + `edge_type_id` from the
  shared cache directly as `spatial_pos` / `edge_input` tensors in Graphormer's collator.
- LapPE / RWSE / SignNet-PE: **do not fit natively.** These are concatenated onto the
  token (node) embedding at the input layer, in addition to Graphormer's centrality
  encoding, following the "APE-inside-a-bias-native-model" ablation used by Black et al.
  (2024) to compare APE vs. RPE within one architecture. This is an explicit adaptation:
  it changes Graphormer's node-embedding dimensionality and adds parameters the original
  model doesn't have, so again reported with its own parameter count.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_meta import SPD_NUM_BUCKETS  # noqa: E402

import torch
import torch.nn as nn


def build_graphormer_config(pe_name: str, cache_dir: str) -> dict:
    base = {
        "centrality_encoding": True,   # Graphormer's own default, always on
        "spatial_encoding": False,     # attention-bias term (native slot)
        "edge_encoding": False,        # attention-bias term (native slot)
        "extra_node_pe": None,         # feature-level PE bolted on (non-native)
        "cache_dir": cache_dir,
    }
    if pe_name == "none":
        return base
    if pe_name == "grpe":
        return {**base, "spatial_encoding": True, "edge_encoding": True}
    if pe_name in ("lappe", "rwse", "signnet"):
        return {**base, "extra_node_pe": pe_name}
    raise ValueError(f"Unknown pe_name: {pe_name}")


def collate_spatial_and_edge(spd_bucket: torch.Tensor, edge_type_id: torch.Tensor):
    """Graphormer's data collator expects `spatial_pos` (n x n long, clamped) and
    `edge_input` (n x n x path_len x edge_feat, here simplified to a single categorical
    edge-type id per pair since LRGB edge features are scalar weights, not multi-hop
    bond paths as in the original molecular-graph setting)."""
    spatial_pos = spd_bucket  # pre-bucketed upstream; see src/dataset_meta.py
    edge_input = edge_type_id.unsqueeze(-1)  # [n, n, 1] -- single "path step" placeholder
    return spatial_pos, edge_input


class ExtraNodePEProjection(nn.Module):
    """Small linear projection so a feature-level PE (LapPE/RWSE/SignNet-PE) can be summed
    into Graphormer's node embedding without changing its hidden size."""

    def __init__(self, pe_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(pe_dim, hidden_dim)

    def forward(self, node_emb: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        return node_emb + self.proj(pe)
