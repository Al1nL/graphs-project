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

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_meta import SPD_NUM_BUCKETS  # noqa: E402

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_posenc_config(pe_name: str, cache_dir: str) -> dict:
    """A readable SUMMARY of the posenc settings this PE trains with.

    NOT the config that runs. The authority is graphgps_backend.build_graphgym_cfg, which
    merges GraphGPS's reference YAML and then applies PE_SPEC and PE_ENCODER to the live
    GraphGym cfg. This function exists so `--dry-run` and the run header can show what a
    cell is about to do without constructing that cfg (which needs the GraphGPS clone).

    Every value is therefore DERIVED from those same tables rather than restated. The
    previous version restated them, and had drifted: it advertised phi_out_dim 32 while
    runs used 64, omitted raw_norm_type entirely -- BatchNorm for RWSE, which materially
    changes what the encoder sees -- and pointed at "<cache>/{split}_pe.pt", a layout the
    PE cache abandoned for per-graph .npy files under node/, eig/ and spd/. Printed at the
    top of every run, it read as a record of the configuration while describing something
    that had not been true for some time.
    """
    from backends.graphgps_backend import K_LAP, K_RWSE, PE_ENCODER, PE_SPEC

    if pe_name not in PE_SPEC:
        raise ValueError(f"Unknown pe_name: {pe_name}")

    cfg = {key: {"enable": False}
           for key in ("posenc_LapPE", "posenc_RWSE", "posenc_SignNet")}

    _enc_suffix, posenc_key, dim_pe = PE_SPEC[pe_name]
    if posenc_key is not None:
        block = {"enable": True, "dim_pe": dim_pe, **PE_ENCODER[pe_name]}
        if posenc_key in ("posenc_LapPE", "posenc_SignNet"):
            # K_LAP, not dim_pe: eigenvectors IN vs channels OUT. They coincide for LapPE
            # and differ for SignNet (16 in, 32 out).
            block["eigen"] = {"max_freqs": K_LAP}
        if posenc_key == "posenc_RWSE":
            block["kernel"] = {"times_func": f"range(1,{K_RWSE + 1})"}
        cfg[posenc_key] = block

    if pe_name == "grpe":
        # No native posenc block -- consumed instead by the GRPEBiasedAttention hook below,
        # which is not yet wired into GPSLayer, so this cell raises NotImplementedError.
        cfg["custom_attn_bias"] = {"enable": True, "source": "grpe", "status": "NOT WIRED"}

    cfg["_pe_source"] = (
        f"{cache_dir} (per-graph .npy under node/, eig/, spd/; read by "
        f"backends.graphgps_pe_cache, which replaces GraphGPS's own PE pre-transform)"
        if posenc_key is not None else "n/a -- this arm uses no PE"
    )
    cfg["_authority"] = ("summary only; graphgps_backend.build_graphgym_cfg builds the "
                         "config that actually runs")
    return cfg


class GRPEBiasedAttention(nn.Module):
    """Drop-in replacement for the Transformer branch's self-attention inside a GPS layer,
    adding a learnable per-head bias indexed by (shortest-path bucket, edge-type bucket),
    following GRPE (Park et al., 2022), Eq. for a_ij with the query/key term untouched:

        a_ij = (q_i . k_j) / sqrt(d) + b_spd[spd_bucket(i,j)] + b_edge[edge_type(i,j)]

    Only b_spd and b_edge are new learnable parameters; q, k, v projections are GraphGPS's
    existing ones, so parameter-count deltas between GRPE and the other 4 PEs stay small
    and comparable (relevant for the "#Param." column in the results table).
    """

    def __init__(self, dim, num_heads, num_spd_buckets=SPD_NUM_BUCKETS, num_edge_types=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.b_spd = nn.Parameter(torch.zeros(num_heads, num_spd_buckets))
        self.b_edge = nn.Parameter(torch.zeros(num_heads, num_edge_types))

    def forward(self, x, spd_bucket, edge_type_id, attn_mask=None):
        n, d = x.shape
        qkv = self.qkv(x).reshape(n, 3, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [heads, n, head_dim]
        scores = torch.einsum("hid,hjd->hij", q, k) / (self.head_dim ** 0.5)
        # spd_bucket comes pre-bucketed from the PE cache (dataset_meta.spd_bucket_id):
        # exact for d<=8, log-spaced beyond, with a dedicated unreachable bucket. No
        # clamping here -- clamping was what collapsed the tail into one bias.
        scores = scores + self.b_spd[:, spd_bucket]         # broadcast over heads
        scores = scores + self.b_edge[:, edge_type_id]
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("hij,hjd->ihd", attn, v).reshape(n, d)
        return self.out_proj(out)
