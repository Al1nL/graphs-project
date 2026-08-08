"""
san_adapter.py
==============
Maps the shared PE cache into SAN's (Kreuzer et al., 2021) input format.

SAN natively expects a Learned Positional Encoding (LPE): raw Laplacian eigenvectors +
eigenvalues, passed through a small Transformer encoder, then added to node features. That
is exactly our `lap_pe`/`lap_eigvals` cache fields, so LapPE is a near-faithful drop-in.

- No-PE: disable SAN's LPE module entirely (feed zeros / skip the add).
- LapPE: native fit, SAN's own LPE encoder consumes `lap_pe` + `lap_eigvals` unchanged.
- RWSE: not part of SAN's original design. We concatenate `rwse` to the node input features
  alongside (or instead of, per config flag) the LPE output -- a straightforward feature-
  level extension, no architecture change needed.
- SignNet-PE: replaces SAN's own LPE encoder with the SignNetEncoder from
  `src/pe/compute_pe.py` (both operate on the same raw eigenvectors, so this is a clean
  swap of "how do we make eigenvectors sign-invariant", not a structural change).
- GRPE: **does not fit SAN's architecture naturally.** SAN already has one additive
  attention bias baked in -- a learned scalar `gamma` that separately weights attention
  contributions from existing vs. non-existing edges (see Kreuzer et al., Eq. 5-ish). We
  extend that mechanism: instead of a single scalar gamma, SAN's attention gets an
  additional per-head bias indexed by GRPE's (shortest-path bucket, edge-type bucket),
  added at the same point in the computation where gamma is applied. This is explicitly a
  research choice made for this project, not part of the published SAN model -- it is
  flagged in the paper as "SAN+GRPE (adapted)" throughout, and its parameter count is
  reported separately so it isn't silently compared as if it were stock SAN.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_meta import SPD_NUM_BUCKETS  # noqa: E402

import torch
import torch.nn as nn


def build_san_config(pe_name: str, cache_dir: str) -> dict:
    base = {
        "lpe_enable": False,
        "extra_node_feat": None,     # e.g. "rwse" to concat
        "signnet_replaces_lpe": False,
        "grpe_bias_enable": False,
        "cache_dir": cache_dir,
    }
    if pe_name == "none":
        return base
    if pe_name == "lappe":
        return {**base, "lpe_enable": True}
    if pe_name == "rwse":
        return {**base, "lpe_enable": True, "extra_node_feat": "rwse"}
    if pe_name == "signnet":
        return {**base, "lpe_enable": True, "signnet_replaces_lpe": True}
    if pe_name == "grpe":
        return {**base, "lpe_enable": False, "grpe_bias_enable": True}
    raise ValueError(f"Unknown pe_name: {pe_name}")


class SANGammaGRPEBias(nn.Module):
    """Extension of SAN's edge-existence gamma bias to a full GRPE-style bias.

    Original SAN: score_ij += gamma if (i,j) is an edge else 0   (single learnable scalar)
    This module:  score_ij += b_spd[spd_bucket(i,j)] + b_edge[edge_type(i,j)]  (per head)

    Kept as a standalone module (rather than editing SAN's attention class in place) so a
    reviewer can diff exactly what changed relative to stock SAN.
    """

    def __init__(self, num_heads, num_spd_buckets=SPD_NUM_BUCKETS, num_edge_types=2):
        super().__init__()
        self.b_spd = nn.Parameter(torch.zeros(num_heads, num_spd_buckets))
        self.b_edge = nn.Parameter(torch.zeros(num_heads, num_edge_types))

    def forward(self, scores, spd_bucket, edge_type_id):
        # spd_bucket is pre-bucketed by dataset_meta.spd_bucket_id; see graphgps_adapter.
        return scores + self.b_spd[:, spd_bucket] + self.b_edge[:, edge_type_id]
