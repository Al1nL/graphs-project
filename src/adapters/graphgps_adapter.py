"""
graphgps_adapter.py
====================
Maps the shared PE cache (src/pe/compute_pe.py) into whatever GraphGPS (Rampasek et al.,
2022) natively expects.

GraphGPS is config-driven (GraphGym YAML). For the three feature-level PEs (No-PE, LapPE,
RWSE, SignNet-PE) this adapter is nearly a no-op: it just points GraphGPS's own
`posenc_LapPE` / `posenc_RWSE` blocks at our cached tensors instead of letting GraphGPS
recompute them internally, so the numbers are identical to what SAN/Graphormer see.

For GRPE (attention bias), GraphGPS has no native hook -- its GPS layer only supports
adding structural information as node features or as an edge-feature-conditioned MPNN
branch, not as an additive bias on the *global attention* logits. We add a small hook
(`GRPEBiasedAttention`) that wraps GraphGPS's `torch.nn.MultiheadAttention` call inside the
GPS layer's Transformer branch and adds the GRPE bias term before softmax, mirroring
Graphormer's original formulation. This is a genuine architectural addition to GraphGPS,
not present in the original paper -- flagged in the report as an adaptation, not a
faithful ablation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_posenc_config(pe_name: str, cache_dir: str) -> dict:
    """Returns the `cfg.posenc_*` dict GraphGPS's main.py expects, pointed at our cache."""
    common = {"enable": False}
    if pe_name == "none":
        return {"posenc_LapPE": dict(common), "posenc_RWSE": dict(common), "posenc_SignNet": dict(common)}

    if pe_name == "lappe":
        cfg = {"posenc_LapPE": {"enable": True, "dim_pe": 16, "model": "DeepSet",
                                  "precomputed_cache": f"{cache_dir}/{{split}}_pe.pt", "field": "lap_pe"},
               "posenc_RWSE": dict(common), "posenc_SignNet": dict(common)}
        return cfg

    if pe_name == "rwse":
        cfg = {"posenc_RWSE": {"enable": True, "dim_pe": 20, "kernel": {"times_func": "range(1,21)"},
                                 "precomputed_cache": f"{cache_dir}/{{split}}_pe.pt", "field": "rwse"},
               "posenc_LapPE": dict(common), "posenc_SignNet": dict(common)}
        return cfg

    if pe_name == "signnet":
        cfg = {"posenc_SignNet": {"enable": True, "dim_pe": 32, "phi_hidden_dim": 64, "phi_out_dim": 32,
                                    "precomputed_cache": f"{cache_dir}/{{split}}_pe.pt", "field": "signnet_in"},
               "posenc_LapPE": dict(common), "posenc_RWSE": dict(common)}
        return cfg

    if pe_name == "grpe":
        # No native posenc block -- consumed instead by the GRPEBiasedAttention hook below.
        return {"posenc_LapPE": dict(common), "posenc_RWSE": dict(common), "posenc_SignNet": dict(common),
                "custom_attn_bias": {"enable": True, "source": "grpe",
                                      "precomputed_cache": f"{cache_dir}/{{split}}_pe.pt"}}

    raise ValueError(f"Unknown pe_name: {pe_name}")


class GRPEBiasedAttention(nn.Module):
    """Drop-in replacement for the Transformer branch's self-attention inside a GPS layer,
    adding a learnable per-head bias indexed by (shortest-path bucket, edge-type bucket),
    following GRPE (Park et al., 2022), Eq. for a_ij with the query/key term untouched:

        a_ij = (q_i . k_j) / sqrt(d) + b_spd[spd_bucket(i,j)] + b_edge[edge_type(i,j)]

    Only b_spd and b_edge are new learnable parameters; q, k, v projections are GraphGPS's
    existing ones, so parameter-count deltas between GRPE and the other 4 PEs stay small
    and comparable (relevant for the "#Param." column in the results table).
    """

    def __init__(self, dim, num_heads, spd_cap=20, num_edge_types=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.b_spd = nn.Parameter(torch.zeros(num_heads, spd_cap + 1))
        self.b_edge = nn.Parameter(torch.zeros(num_heads, num_edge_types))

    def forward(self, x, spd, edge_type_id, attn_mask=None):
        n, d = x.shape
        qkv = self.qkv(x).reshape(n, 3, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [heads, n, head_dim]
        scores = torch.einsum("hid,hjd->hij", q, k) / (self.head_dim ** 0.5)
        spd_clamped = spd.clamp(max=self.b_spd.shape[1] - 1)
        scores = scores + self.b_spd[:, spd_clamped]        # broadcast over heads
        scores = scores + self.b_edge[:, edge_type_id]
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("hij,hjd->ihd", attn, v).reshape(n, d)
        return self.out_proj(out)
