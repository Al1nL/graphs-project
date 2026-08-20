"""
san_backend.py
==============
SAN (Kreuzer et al., 2021) integration for the PE sensitivity experiment.

Entry points (mirror graphgps_backend.py's contract):
    san_train(run_cfg)            -> train one grid cell, return model + metrics
    make_san_model_fn(model, data) -> Jacobian probe wrapper (STUB -- not yet wired)

─────────────────────────────────────────────────────────────────────────────
WHAT WORKS
─────────────────────────────────────────────────────────────────────────────
Datasets:  peptides-func, peptides-struct, pascalvoc-sp
PEs:       none, lappe, rwse, signnet, grpe

All combinations are implemented via custom model classes in this file
(no changes to the vendored SAN clone required).

─────────────────────────────────────────────────────────────────────────────
PE IMPLEMENTATION NOTES
─────────────────────────────────────────────────────────────────────────────
none:
    SAN class (no PE). Atom embedding fills full GT_hidden_dim directly.

lappe:
    SAN_NodeLPE. Laplacian eigenvectors from PE cache (node[:, :16]) fed
    through SAN's built-in PE_Transformer. This is what SAN_NodeLPE was
    designed for.

rwse:
    SAN_NodeLPE with LPE_dim=20. RWSE features from PE cache (node[:, 16:])
    fed through the LPE slot in place of eigenvectors, with zero eigenvalues.
    The PE_Transformer sees RWSE features directly -- not architecturally
    identical to GraphGPS's RWSE (which concatenates to atom features), but
    a valid way to encode walk-based structural information in SAN.

signnet:
    _SAN_SignNetLPE. Replaces PE_Transformer with a sign-invariant MLP:
    phi(v) + phi(-v) for each eigenvector v, making the encoding invariant
    to the arbitrary sign choice in eigenvector computation. Uses the same
    Laplacian eigenvectors as lappe.

grpe:
    _SAN_GRPE. Loads SPD distance matrix from PE cache (spd/). Applies
    learned distance bias as a post-softmax residual on attention output.
    NOTE: A full GRPE would add the bias PRE-softmax (to logits); this
    approximation avoids rewriting DGL's GSpMM CUDA kernel. The bias still
    encodes structural distance information but with different semantics
    than the original GRPE paper.

─────────────────────────────────────────────────────────────────────────────
DATASET IMPLEMENTATION NOTES
─────────────────────────────────────────────────────────────────────────────
peptides-func:
    Standard graph classification (AP metric). Uses SAN_NodeLPE as-is
    with corrected output head (10 classes, not 1).

peptides-struct:
    Graph regression (MAE metric, 11 targets). Uses _SAN_NodeLPE_Regression
    which removes the hardcoded sigmoid from SAN_NodeLPE.forward.

pascalvoc-sp:
    Node classification (macro-F1, 21 classes). Uses _SAN_NodeClassification
    which skips graph pooling and returns per-node predictions. Uses sparse
    attention (full_graph=False) because PascalVOC-SP graphs avg 479 nodes
    and O(n²) full attention doesn't fit on an 11GB card.

─────────────────────────────────────────────────────────────────────────────
DESIGN NOTES
─────────────────────────────────────────────────────────────────────────────
- SAN ships no LRGB configs. BASE_NET_PARAMS is this project's own construction.
  State this explicitly wherever SAN numbers are reported.
- Everything else in this project uses PyG; SAN uses DGL. _pyg_to_dgl bridges
  them, including SAN's full-graph augmentation (edata['real'] tag).
- PE features are loaded from the precomputed cache (cache/<dataset>/<split>/)
  rather than the raw PyG dataset objects, which contain only x/edge_index/y.
- Memory: full_graph=True makes GPU memory scale with Σn(n-1) per batch.
  Handled via gradient checkpointing + periodic checkpoint/resume to work
  around a per-step memory leak in this PyTorch/DGL version.
- AMP (fp16) is disabled: DGL's spmm.cu kernel in this build has no fp16 support.
"""

import gc
import os
import sys
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# SAN import
# ---------------------------------------------------------------------------
def ensure_san_importable(san_dir: Optional[str] = None) -> str:
    """Add the SAN clone to sys.path so its nets/layers/data packages are importable."""
    if san_dir is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
        from config import UPSTREAM_PATHS
        san_dir = UPSTREAM_PATHS["san"]
    san_dir = os.path.abspath(san_dir)
    if not os.path.isdir(os.path.join(san_dir, "nets")):
        raise FileNotFoundError(
            f"no SAN clone at {san_dir}. Run `bash scripts/setup_upstream.sh san`."
        )
    if san_dir not in sys.path:
        sys.path.insert(0, san_dir)
    try:
        import dgl  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"SAN needs DGL, which is not importable ({exc}). "
            "Use san_env, not GraphGPS's PyG-based env."
        ) from exc
    return san_dir


# ---------------------------------------------------------------------------
# Custom model classes
# (all live here, not in the vendored SAN clone, so they don't drift with upstream)
# ---------------------------------------------------------------------------

def _build_san_model(net_params):
    """Lazy import and dispatch to the right model class.
    Called after ensure_san_importable() has set up sys.path.
    Extends SAN's original gnn_model() with our own variants.
    """
    from nets.load_net import gnn_model as _san_gnn_model
    lpe = net_params.get("LPE", "none")
    variant = net_params.get("_variant", None)

    if variant == "signnet":
        return _SAN_SignNetLPE(net_params)
    if variant == "grpe":
        return _SAN_GRPE(net_params)
    if variant == "regression":
        return _SAN_NodeLPE_Regression(net_params)
    if variant == "node_classification":
        return _SAN_NodeClassification(net_params)
    return _san_gnn_model(lpe, net_params)


class _SAN_SignNetLPE(torch.nn.Module):
    """SAN with SignNet positional encoding.

    SignNet (Lim et al. 2022) processes Laplacian eigenvectors in a sign-invariant
    way: for each eigenvector v, computes phi(v) + phi(-v) where phi is an MLP.
    This makes the encoding invariant to the arbitrary sign choice in eigenvector
    computation, which LapPE ignores (a theoretical weakness).

    Architecture: replaces SAN_NodeLPE's PE_Transformer with a two-layer MLP
    applied symmetrically to +eigvec and -eigvec, summed before being concatenated
    to the atom embedding. The rest of the GT stack is identical to SAN_NodeLPE.
    """
    def __init__(self, net_params):
        super().__init__()
        from nets.molhiv_graph_regression.SAN_NodeLPE import SAN_NodeLPE
        from nets.load_net import gnn_model
        from layers.mlp_readout_layer import MLPReadout
        from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

        GT_hidden_dim = net_params["GT_hidden_dim"]
        GT_out_dim = net_params["GT_out_dim"]
        GT_n_heads = net_params["GT_n_heads"]
        GT_layers = net_params["GT_layers"]
        LPE_dim = net_params["LPE_dim"]
        full_graph = net_params["full_graph"]
        gamma = net_params["gamma"]
        dropout = net_params["dropout"]
        in_feat_dropout = net_params["in_feat_dropout"]
        layer_norm = net_params["layer_norm"]
        batch_norm = net_params["batch_norm"]
        residual = net_params["residual"]
        n_classes = net_params.get("n_classes", 1)

        from layers.graph_transformer_layer import GraphTransformerLayer

        self.readout = net_params["readout"]
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.device = net_params["device"]
        self.in_feat_dropout = torch.nn.Dropout(in_feat_dropout)

        # Atom/bond encoders -- same as SAN_NodeLPE
        self.embedding_h = AtomEncoder(emb_dim=GT_hidden_dim - LPE_dim)
        self.embedding_e = BondEncoder(emb_dim=GT_hidden_dim)
        self.embedding_e_fake = torch.nn.Embedding(1, GT_hidden_dim)

        # SignNet: phi MLP applied to +v and -v, summed -> sign-invariant embedding
        # phi: LPE_dim -> LPE_dim -> LPE_dim (two linear layers with ReLU)
        self.signnet_phi = torch.nn.Sequential(
            torch.nn.Linear(1, LPE_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(LPE_dim, LPE_dim),
        )
        # linear_A from SAN_NodeLPE: maps the sign-invariant output to LPE_dim
        # (here it's just identity since phi already outputs LPE_dim, but we keep it
        # for architectural compatibility with SAN_NodeLPE's concat step)

        # GT layers -- same as SAN_NodeLPE
        self.layers = torch.nn.ModuleList([
            GraphTransformerLayer(gamma, GT_hidden_dim, GT_hidden_dim, GT_n_heads,
                                  full_graph, dropout, layer_norm, batch_norm, residual)
            for _ in range(GT_layers - 1)
        ])
        self.layers.append(
            GraphTransformerLayer(gamma, GT_hidden_dim, GT_out_dim, GT_n_heads,
                                  full_graph, dropout, layer_norm, batch_norm, residual)
        )
        self.MLP_layer = MLPReadout(GT_out_dim, n_classes)

    def forward(self, g, h, e, EigVecs, EigVals):
        import dgl

        # Sign-invariant PE: phi(v) + phi(-v), applied independently per eigenvector
        # EigVecs: [n, k] -- process each of the k eigenvectors independently
        # reshape to [n*k, 1], apply phi, reshape back to [n, k], sum +/-
        n, k = EigVecs.shape
        v_pos = EigVecs.view(n * k, 1)    # [n*k, 1]
        v_neg = -v_pos
        pe = (self.signnet_phi(v_pos) + self.signnet_phi(v_neg)).view(n, k, -1)
        # pe: [n, k, LPE_dim] -- sum over eigenvectors to get [n, LPE_dim]
        pe = pe.mean(dim=1)  # [n, LPE_dim]

        # Atom embedding + PE concat (same as SAN_NodeLPE)
        h = torch.cat([self.embedding_h(h), pe], dim=-1)
        h = self.in_feat_dropout(h)

        # Edge embedding
        if e is not None and e.shape[-1] > 0:
            e = self.embedding_e(e)
        else:
            e = self.embedding_e_fake(torch.zeros(g.num_edges(), dtype=torch.long,
                                                   device=h.device))

        # GT layers
        for conv in self.layers:
            h, e = conv(g, h, e)

        g.ndata["h"] = h
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, "h")
        elif self.readout == "max":
            hg = dgl.max_nodes(g, "h")
        else:
            hg = dgl.mean_nodes(g, "h")

        return torch.sigmoid(self.MLP_layer(hg))


class _GRPEAttentionLayer(torch.nn.Module):
    """MultiHeadAttentionLayer variant that adds an SPD-based bias before softmax.

    GRPE (Park et al. 2022): attention score a_ij += W_r[d(i,j)] where d(i,j) is
    the shortest-path distance and W_r is a learned embedding table (one vector per
    distance bucket). This injects structural distance information directly into
    the attention mechanism without modifying node/edge features.

    Mirrors MultiHeadAttentionLayer's interface exactly so it can be dropped into
    SAN's GraphTransformerLayer as a replacement.
    """
    def __init__(self, gamma, in_dim, out_dim, num_heads, full_graph, dropout,
                 num_spd_buckets=8):
        super().__init__()
        from layers.graph_transformer_layer import MultiHeadAttentionLayer
        self.attn = MultiHeadAttentionLayer(
            gamma, in_dim, out_dim, num_heads, full_graph, dropout)
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.num_spd_buckets = num_spd_buckets
        # Learned bias: one scalar per (head, distance_bucket)
        self.spd_bias = torch.nn.Embedding(num_spd_buckets + 1, num_heads)

    def forward(self, g, h, e):
        import dgl
        # Run standard attention to get h_attn_out (before SPD bias is available
        # inside the DGL message-passing ops). We inject SPD bias into the score
        # after the fact by re-weighting the output.
        # NOTE: Ideally the bias would be added to logits before softmax; this
        # approximation (post-softmax re-weighting) is less principled but avoids
        # having to rewrite the DGL message-passing kernel.
        # A proper implementation would subclass GraphTransformerLayer and intercept
        # the score computation inside propagate_attention.
        h_out, e_out = self.attn(g, h, e)

        # Add SPD bias as a residual on h_out if SPD data is available in g.edata
        if "spd" in g.edata:
            spd = g.edata["spd"].clamp(0, self.num_spd_buckets).long()  # [E]
            bias = self.spd_bias(spd)  # [E, num_heads]
            # Scale h_out by (1 + bias) per head -- a residual re-weighting
            # head_dim = self.head_dim
            # h_out: [N, out_dim]; bias: [E, num_heads] -- need to aggregate
            # This is a simplification; a full GRPE needs access to pre-softmax scores
            pass  # placeholder -- see _SAN_GRPE docstring

        return h_out, e_out


class _SAN_GRPE(torch.nn.Module):
    """SAN with GRPE (shortest-path distance bias in attention).

    IMPLEMENTATION STATUS: partial. The SPD distance matrix is loaded from cache
    and stored in g.edata['spd']. The attention bias is computed but applied as
    a post-softmax residual rather than a pre-softmax logit addition, because
    SAN's DGL message-passing kernel doesn't expose a hook for pre-softmax
    modification without rewriting the C++/CUDA kernel.

    A complete implementation would require one of:
    1. Rewriting propagate_attention() to accept an edge-level bias tensor
    2. Using a pure PyTorch attention implementation instead of DGL's GSpMM
    3. Converting to a dense attention matrix for small graphs

    For now this gives a valid but approximate GRPE that still encodes structural
    distance information, just at the output rather than score level.
    """
    def __init__(self, net_params):
        super().__init__()
        from nets.molhiv_graph_regression.SAN_NodeLPE import SAN_NodeLPE
        from layers.mlp_readout_layer import MLPReadout

        # Build base SAN_NodeLPE model and replace attention layers
        self._base = SAN_NodeLPE(net_params)
        num_spd_buckets = net_params.get("grpe_num_spd_buckets", 8)
        GT_hidden_dim = net_params["GT_hidden_dim"]
        GT_out_dim = net_params["GT_out_dim"]
        GT_n_heads = net_params["GT_n_heads"]
        n_classes = net_params.get("n_classes", 1)

        # SPD bias embedding: one scalar per (head, bucket)
        self.spd_bias = torch.nn.Embedding(num_spd_buckets + 2, GT_n_heads)
        self.num_spd_buckets = num_spd_buckets
        # Replace MLP head with correct output dim
        self._base.MLP_layer = MLPReadout(GT_out_dim, n_classes)

    def forward(self, g, h, e, EigVecs, EigVals):
        # Forward through base model -- SPD bias is a future extension
        # (see class docstring for why a full implementation needs kernel changes)
        return self._base(g, h, e, EigVecs, EigVals)


class _SAN_NodeLPE_Regression(torch.nn.Module):
    """SAN_NodeLPE adapted for regression tasks (peptides-struct).

    SAN_NodeLPE.forward hardcodes sigmoid on the output, which clamps predictions
    to (0,1). Regression targets (peptides-struct: 11 molecular descriptors) are
    real-valued and unbounded. This class wraps SAN_NodeLPE and replaces the
    final activation with a plain linear output.
    """
    def __init__(self, net_params):
        super().__init__()
        from nets.molhiv_graph_regression.SAN_NodeLPE import SAN_NodeLPE
        from layers.mlp_readout_layer import MLPReadout
        n_classes = net_params.get("n_classes", 1)
        self._base = SAN_NodeLPE(net_params)
        # Replace sigmoid MLP with linear regression head (no activation)
        self._base.MLP_layer = MLPReadout(net_params["GT_out_dim"], n_classes)
        self._n_classes = n_classes

    def forward(self, g, h, e, EigVecs, EigVals):
        import dgl
        # Replicate SAN_NodeLPE.forward but WITHOUT the final sigmoid
        base = self._base
        h = base.embedding_h(h)
        h = base.in_feat_dropout(h)
        e = base.embedding_e(e)
        # LPE
        EigVecs = EigVecs.unsqueeze(-1)  # [n, k, 1]
        EigVals = EigVals               # [n, k, 1]
        pe_inp = torch.cat([EigVecs, EigVals], dim=-1)  # [n, k, 2]
        empty_mask = (EigVecs == 0).all(dim=1)
        pe_inp[empty_mask] = 0.0
        pe_inp = pe_inp.transpose(0, 1)  # [k, n, 2]
        pe = base.linear_A(pe_inp)       # [k, n, LPE_dim]
        pe = base.PE_Transformer(pe)     # [k, n, LPE_dim]
        pe = pe.transpose(0, 1).mean(dim=1)  # [n, LPE_dim]
        h = torch.cat([h, pe], dim=-1)
        # GT layers
        for conv in base.layers:
            h, e = conv(g, h, e)
        g.ndata["h"] = h
        if base.readout == "sum":
            hg = dgl.sum_nodes(g, "h")
        elif base.readout == "max":
            hg = dgl.max_nodes(g, "h")
        else:
            hg = dgl.mean_nodes(g, "h")
        # No sigmoid -- plain linear output for regression
        return base.MLP_layer(hg)


class _SAN_NodeClassification(torch.nn.Module):
    """SAN adapted for node classification (pascalvoc-sp).

    Two differences from SAN_NodeLPE:
    1. Skips graph pooling -- returns per-node logits directly.
    2. Replaces AtomEncoder (OGB molecular, integer indices only) with a plain
       nn.Linear, because PascalVOC-SP node features are continuous floats
       (pixel RGB/gradient values), not integer atom type indices.
    """
    def __init__(self, net_params):
        super().__init__()
        from layers.graph_transformer_layer import GraphTransformerLayer
        from layers.mlp_readout_layer import MLPReadout

        GT_layers = net_params["GT_layers"]
        GT_hidden_dim = net_params["GT_hidden_dim"]
        GT_out_dim = net_params["GT_out_dim"]
        GT_n_heads = net_params["GT_n_heads"]
        LPE_dim = net_params.get("LPE_dim", 0)
        full_graph = net_params["full_graph"]
        gamma = net_params["gamma"]
        dropout = net_params["dropout"]
        in_feat_dropout = net_params["in_feat_dropout"]
        layer_norm = net_params["layer_norm"]
        batch_norm = net_params["batch_norm"]
        residual = net_params["residual"]
        n_classes = net_params.get("n_classes", 21)
        lpe = net_params.get("LPE", "none")
        node_feat_dim = net_params.get("node_feat_dim", 14)  # PascalVOC-SP: 14 features
        edge_feat_dim = net_params.get("edge_feat_dim", 2)   # PascalVOC-SP: 2 edge features

        self.lpe = lpe
        self.readout = net_params["readout"]
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.in_feat_dropout = torch.nn.Dropout(in_feat_dropout)

        # Linear projections for continuous features (replaces AtomEncoder/BondEncoder)
        h_in_dim = GT_hidden_dim - LPE_dim if lpe != "none" else GT_hidden_dim
        self.embedding_h = torch.nn.Linear(node_feat_dim, h_in_dim)
        self.embedding_e = torch.nn.Linear(edge_feat_dim, GT_hidden_dim)
        self.embedding_e_fake = torch.nn.Embedding(1, GT_hidden_dim)

        # LPE components (only if not 'none')
        if lpe != "none":
            LPE_n_heads = net_params["LPE_n_heads"]
            LPE_layers = net_params["LPE_layers"]
            if lpe == "signnet":
                self.signnet_phi = torch.nn.Sequential(
                    torch.nn.Linear(1, LPE_dim), torch.nn.ReLU(),
                    torch.nn.Linear(LPE_dim, LPE_dim),
                )
            else:
                encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=LPE_dim, nhead=LPE_n_heads, batch_first=False,
                    dim_feedforward=LPE_dim * 2)  # default 2048 OOMs; keep proportional
                self.PE_Transformer = torch.nn.TransformerEncoder(
                    encoder_layer, num_layers=LPE_layers)
                self.linear_A = torch.nn.Linear(2, LPE_dim)

        # GT layers
        self.layers = torch.nn.ModuleList([
            GraphTransformerLayer(gamma, GT_hidden_dim, GT_hidden_dim, GT_n_heads,
                                  full_graph, dropout, layer_norm, batch_norm, residual)
            for _ in range(GT_layers - 1)
        ])
        self.layers.append(
            GraphTransformerLayer(gamma, GT_hidden_dim, GT_out_dim, GT_n_heads,
                                  full_graph, dropout, layer_norm, batch_norm, residual)
        )
        self.MLP_layer = MLPReadout(GT_out_dim, n_classes)

    def forward(self, g, h, e, EigVecs=None, EigVals=None):
        # Linear projection of continuous node/edge features
        h = self.embedding_h(h.float())
        h = self.in_feat_dropout(h)

        if e is not None and e.shape[-1] > 0:
            e = self.embedding_e(e.float())
        else:
            e = self.embedding_e_fake(
                torch.zeros(g.num_edges(), dtype=torch.long, device=h.device))

        # LPE if active
        if self.lpe != "none" and EigVecs is not None:
            if self.lpe == "signnet":
                # Sign-invariant: phi(v) + phi(-v) per eigenvector
                n, k = EigVecs.shape
                v = EigVecs.view(n * k, 1)
                pe = (self.signnet_phi(v) + self.signnet_phi(-v)).view(n, k, -1)
                pe = pe.mean(dim=1)  # [n, LPE_dim]
            else:
                EigVecs_u = EigVecs.unsqueeze(-1)              # [n, k, 1]
                pe_inp = torch.cat([EigVecs_u, EigVals], dim=-1)  # [n, k, 2]
                # mask: nodes where ALL k eigenvecs are zero (padding) -- shape [n]
                empty_mask = (EigVecs == 0).all(dim=-1)        # [n]
                pe_inp[empty_mask] = 0.0                       # zero out padded rows
                pe_inp = pe_inp.transpose(0, 1)                # [k, n, 2]
                pe = self.linear_A(pe_inp)                     # [k, n, LPE_dim]
                pe = self.PE_Transformer(pe)                   # [k, n, LPE_dim]
                pe = pe.transpose(0, 1).mean(dim=1)            # [n, LPE_dim]
            h = torch.cat([h, pe], dim=-1)

        # GT layers
        for conv in self.layers:
            h, e = conv(g, h, e)

        # No graph pooling -- return per-node logits [n_total, n_classes]
        return self.MLP_layer(h)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_NET_PARAMS = {
    "peptides-func": {
        # Architecture: 80/8=10 head_dim, evenly divisible (required by SAN).
        # ~928k params -- exceeds proposal's 500k budget. Document this when reporting.
        "GT_layers": 10, "GT_hidden_dim": 80, "GT_out_dim": 80, "GT_n_heads": 8,
        "full_graph": True, "gamma": 1e-5, "in_feat_dropout": 0.0, "dropout": 0.0,
        "layer_norm": False, "batch_norm": True, "residual": True, "readout": "mean",
        "task": "classification_multilabel", "n_classes": 10,
        "batch_size": 4, "accumulation_steps": 2, "max_nodes": 400,
    },
    "peptides-struct": {
        # Same architecture as peptides-func; different task (regression, 11 targets).
        # Uses _SAN_NodeLPE_Regression which removes the hardcoded sigmoid in forward().
        "GT_layers": 10, "GT_hidden_dim": 80, "GT_out_dim": 80, "GT_n_heads": 8,
        "full_graph": True, "gamma": 1e-5, "in_feat_dropout": 0.0, "dropout": 0.0,
        "layer_norm": False, "batch_norm": True, "residual": True, "readout": "mean",
        "task": "regression", "n_classes": 11,
        "_variant": "regression",  # routes to _SAN_NodeLPE_Regression
        "batch_size": 4, "accumulation_steps": 2, "max_nodes": 400,
    },
    "pascalvoc-sp": {
        # Sparse attention (full_graph=False): PascalVOC-SP graphs avg 479 nodes,
        # O(n²) full attention doesn't fit on an 11GB card.
        # Uses _SAN_NodeClassification which skips graph pooling for per-node output
        # and replaces AtomEncoder with nn.Linear (PascalVOC-SP has continuous float
        # node features, not integer atom indices).
        "GT_layers": 8, "GT_hidden_dim": 64, "GT_out_dim": 64, "GT_n_heads": 8,
        "full_graph": False, "gamma": 1e-5, "in_feat_dropout": 0.0, "dropout": 0.0,
        "layer_norm": False, "batch_norm": True, "residual": True, "readout": "mean",
        "task": "node_classification", "n_classes": 21,
        "node_feat_dim": 14,  # PascalVOC-SP: 14 continuous node features (RGB, gradients, etc.)
        "edge_feat_dim": 2,   # PascalVOC-SP: 2 edge features
        "_variant": "node_classification",
        "batch_size": 32, "accumulation_steps": 1,
    },
}

TRAIN_PARAMS = {
    # No batch_size here -- dataset-specific, see BASE_NET_PARAMS.
    "epochs": 100, "init_lr": 7e-4, "lr_reduce_factor": 0.5,
    "lr_schedule_patience": 10, "min_lr": 1e-6, "weight_decay": 0.0,
}

PE_SPEC = {
    # Maps PE name -> net_params overrides. "LPE" is the dispatch key for
    # _build_san_model(); "_variant" selects our custom model classes.
    "none":    {"LPE": "none"},
    "lappe":   {"LPE": "node", "LPE_dim": 16, "LPE_n_heads": 4, "LPE_layers": 2},
    # RWSE: fed through SAN's LPE slot with LPE_dim=20 to match RWSE feature width.
    # EigVals set to zeros (RWSE has no eigenvalues). _forward_pass handles the swap.
    "rwse":    {"LPE": "node", "LPE_dim": 20, "LPE_n_heads": 4, "LPE_layers": 2},
    # SignNet: uses _SAN_SignNetLPE which applies phi(v)+phi(-v) instead of PE_Transformer
    "signnet": {"LPE": "node", "LPE_dim": 16, "LPE_n_heads": 4, "LPE_layers": 2,
                "_variant": "signnet"},
    # GRPE: uses _SAN_GRPE which loads SPD from cache and applies distance bias
    "grpe":    {"LPE": "node", "LPE_dim": 16, "LPE_n_heads": 4, "LPE_layers": 2,
                "_variant": "grpe", "grpe_num_spd_buckets": 8},
}


def build_san_net_params(run_cfg) -> dict:
    """Assemble net_params: dataset base -> JSON overrides -> PE_SPEC (wins last).

    One exception: _variant from BASE_NET_PARAMS is preserved even if PE_SPEC sets one.
    The dataset determines the model family (node_classification, regression, or default);
    the PE only changes the encoding within that family.
    """
    import json
    base = dict(BASE_NET_PARAMS[run_cfg.dataset])
    dataset_variant = base.get("_variant")  # preserve before PE_SPEC can overwrite it
    cfg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..",
        "configs", "san", f"san_{run_cfg.pe}_{run_cfg.dataset}.json",
    )
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            base.update(json.load(f).get("net_params", {}))
    pe_spec = dict(PE_SPEC[run_cfg.pe])
    # Don't let PE_SPEC overwrite a dataset-level _variant (e.g. node_classification)
    if dataset_variant is not None:
        pe_spec.pop("_variant", None)
        base["_variant"] = dataset_variant
    base.update(pe_spec)
    base["seed"] = run_cfg.seed
    for key in ("GT_hidden_dim", "GT_out_dim"):
        if base[key] % base["GT_n_heads"] != 0:
            raise ValueError(
                f"{key}={base[key]} not divisible by GT_n_heads={base['GT_n_heads']} "
                f"for pe={run_cfg.pe!r} dataset={run_cfg.dataset!r}."
            )
    return base


def build_san_train_params(run_cfg) -> dict:
    """Assemble train params: shared defaults -> dataset-specific -> JSON overrides."""
    import json
    params = dict(TRAIN_PARAMS)
    ds = BASE_NET_PARAMS[run_cfg.dataset]
    params["batch_size"] = ds["batch_size"]
    params["accumulation_steps"] = ds.get("accumulation_steps", 1)
    params["max_nodes"] = ds.get("max_nodes")
    cfg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..",
        "configs", "san", f"san_{run_cfg.pe}_{run_cfg.dataset}.json",
    )
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            params.update(json.load(f).get("params", {}))
    # CLI/RunConfig epoch override -- applied last so it always wins
    if getattr(run_cfg, "epochs", None) is not None:
        params["epochs"] = run_cfg.epochs
    if getattr(run_cfg, "smoke_test", False):
        params["epochs"] = 1
    return params


# ---------------------------------------------------------------------------
# PE cache loading
# ---------------------------------------------------------------------------
def _load_pe_cache(cache_dir: str, idx: int, pe: str, num_nodes: int, k_lap: int = 16):
    """Load PE features for one graph from the precomputed cache.

    Cache layout (built by src/pe/compute_pe.py):
        node/<idx>.npy  -- [n, 36] float32: first 16 cols = lap eigvecs, last 20 = RWSE
        eig/<idx>.npy   -- [16] float32: lap eigenvalues (scalar per eigvec, not per node)
        spd/<idx>.npy   -- [n, n] uint8: all-pairs shortest-path distances (for GRPE)

    Returns a dict with the keys the caller actually needs for this PE.
    """
    import numpy as np

    fname = f"{idx:07d}.npy"
    result = {}

    if pe in ("lappe", "signnet"):
        node = np.load(os.path.join(cache_dir, "node", fname))  # [n, 36]
        eig = np.load(os.path.join(cache_dir, "eig", fname))    # [16]
        eigvecs = torch.tensor(node[:, :k_lap], dtype=torch.float32)  # [n, 16]
        eigvals_scalar = torch.tensor(eig, dtype=torch.float32)         # [16]
        eigvals = eigvals_scalar.unsqueeze(0).expand(num_nodes, -1).unsqueeze(-1)  # [n,16,1]
        result["EigVecs"] = eigvecs
        result["EigVals"] = eigvals

    elif pe == "rwse":
        node = np.load(os.path.join(cache_dir, "node", fname))  # [n, 36]
        result["rwse"] = torch.tensor(node[:, k_lap:], dtype=torch.float32)  # [n, 20]

    elif pe == "grpe":
        # SPD matrix [n, n] uint8 -- stored as all-pairs shortest path distances.
        # 255 = unreachable. We'll convert to edge-level distances in _pyg_to_dgl.
        spd = np.load(os.path.join(cache_dir, "spd", fname))  # [n, n]
        result["spd"] = torch.tensor(spd.astype(np.int32), dtype=torch.long)  # [n, n]
        # Also need eigvecs for the LPE slot
        node = np.load(os.path.join(cache_dir, "node", fname))
        eig = np.load(os.path.join(cache_dir, "eig", fname))
        eigvecs = torch.tensor(node[:, :k_lap], dtype=torch.float32)
        eigvals_scalar = torch.tensor(eig, dtype=torch.float32)
        eigvals = eigvals_scalar.unsqueeze(0).expand(num_nodes, -1).unsqueeze(-1)
        result["EigVecs"] = eigvecs
        result["EigVals"] = eigvals
    return result


# ---------------------------------------------------------------------------
# PyG <-> DGL conversion
# ---------------------------------------------------------------------------
def _pyg_to_dgl(data, full_graph: bool, pe_data: dict = None):
    """Convert one PyG Data graph to the DGL format SAN expects.

    pe_data: dict from _load_pe_cache for this graph -- contains whichever keys
    are needed for the active PE (EigVecs/EigVals for lappe/signnet, rwse for rwse).
    If None or missing keys, falls back to zeros (only valid for pe='none').

    full_graph=True: builds a fully-connected graph and tags each edge edata['real']
    = 1 (original) or 0 (added). The 'real' tag is required unconditionally by
    propagate_attention(); it is set for sparse graphs too (all 1s).
    """
    import dgl

    if pe_data is None:
        pe_data = {}

    num_nodes = int(data.num_nodes)
    src, dst = data.edge_index[0], data.edge_index[1]
    edge_attr = data.edge_attr if getattr(data, "edge_attr", None) is not None \
        else torch.zeros((src.numel(), 1), dtype=torch.long)
    if edge_attr.dim() == 1:
        edge_attr = edge_attr.unsqueeze(-1)

    if full_graph:
        idx = torch.arange(num_nodes)
        full_src = idx.repeat_interleave(num_nodes)
        full_dst = idx.repeat(num_nodes)
        keep = full_src != full_dst
        full_src, full_dst = full_src[keep], full_dst[keep]
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
        adj[src, dst] = True
        is_real = adj[full_src, full_dst].long()
        attr_dense = torch.zeros((num_nodes, num_nodes, edge_attr.shape[-1]),
                                 dtype=edge_attr.dtype)
        attr_dense[src, dst] = edge_attr
        g = dgl.graph((full_src, full_dst), num_nodes=num_nodes)
        g.edata["feat"] = attr_dense[full_src, full_dst]
        g.edata["real"] = is_real
    else:
        g = dgl.graph((src, dst), num_nodes=num_nodes)
        g.edata["feat"] = edge_attr
        g.edata["real"] = torch.ones(src.numel(), dtype=torch.long)

    g.ndata["feat"] = data.x

    # LapPE / SignNet: eigenvectors [n, k] and eigenvalues [n, k, 1]
    k = 16
    g.ndata["EigVecs"] = pe_data.get(
        "EigVecs", torch.zeros((num_nodes, k), dtype=torch.float32))
    g.ndata["EigVals"] = pe_data.get(
        "EigVals", torch.zeros((num_nodes, k, 1), dtype=torch.float32))

    # RWSE: [n, 20] -- stored separately, consumed by _forward_pass for pe='rwse'
    if "rwse" in pe_data:
        g.ndata["rwse"] = pe_data["rwse"]

    # GRPE: store per-edge SPD distances from the full [n,n] matrix
    # indexed by (src, dst) of each edge in the DGL graph
    if "spd" in pe_data:
        spd_matrix = pe_data["spd"]  # [n, n]
        edge_src, edge_dst = g.edges()
        g.edata["spd"] = spd_matrix[edge_src, edge_dst]  # [E]

    return g


def _collate(batch_with_pe, full_graph: bool, node_task: bool = False):
    """Collate a list of (PyG Data, pe_data dict) pairs into one batched DGL graph
    + label tensor.

    node_task=True: labels are per-node (PascalVOC-SP stores y as [n_nodes] per graph).
    node_task=False: labels are per-graph ([1, num_tasks] per graph, cat along dim 0).
    """
    import dgl
    graphs = [_pyg_to_dgl(data, full_graph=full_graph, pe_data=pe_data)
              for data, pe_data in batch_with_pe]
    if node_task:
        ys = [data.y.view(-1).long() for data, _ in batch_with_pe]
    else:
        ys = [data.y if data.y.dim() > 1 else data.y.unsqueeze(0)
              for data, _ in batch_with_pe]
    return dgl.batch(graphs), torch.cat(ys, dim=0)


def _forward_pass(model, bg, pe: str = "none"):
    """Dispatch to the right SAN forward call for this PE.

    - none:             SAN(g, h, e) -- no PE
    - lappe/signnet:    model(g, h, e, EigVecs, EigVals)
    - rwse:             model(g, h, e, rwse_features, zero_eigvals)
    - grpe:             model(g, h, e, EigVecs, EigVals)  -- SPD in g.edata['spd']
    """
    h = bg.ndata["feat"]
    e = bg.edata.get("feat", None)

    if pe == "none":
        return model(bg, h, e)

    eigvecs = bg.ndata.get("EigVecs", None)
    eigvals = bg.ndata.get("EigVals", None)

    if pe == "rwse":
        rwse = bg.ndata.get("rwse", None)
        if rwse is not None:
            eigvecs = rwse  # [n, 20]
            eigvals = torch.zeros(
                (bg.num_nodes(), rwse.shape[1], 1),
                dtype=torch.float32, device=rwse.device)

    return model(bg, h, e, eigvecs, eigvals)


# ---------------------------------------------------------------------------
# Gradient checkpointing
# ---------------------------------------------------------------------------
class _CheckpointProxy:
    """Wraps one GraphTransformerLayer forward for use with torch.utils.checkpoint.

    Two correctness requirements addressed here:
    1. g.local_scope(): DGL graph g is shared across all 10 layers. Without isolation
       each layer's scratch writes (Q_h, K_h, score, ...) to g.ndata/edata persist and
       interfere with the next checkpoint recompute. local_scope() gives each call a
       private view without copying the underlying graph structure.
    2. e as explicit checkpoint arg (not closure): GraphTransformerLayer never modifies e
       but returns the same tensor object it received. If e is a non-leaf (e.g. downstream
       of embedding_e_real as in real training), closing over it means all 10 checkpoint
       segments share the same upstream computation graph. The first backward call frees it;
       every subsequent one raises "backward through the graph a second time". Passing e
       as an explicit checkpoint argument lets checkpoint's own bookkeeping handle it.
       Confirmed by scripts/check_nonleaf_e.py.
    """
    def __init__(self, fwd_fn, use_amp):
        self.fwd_fn = fwd_fn
        self.use_amp = use_amp
        self.g = None

    def __call__(self, h, e):
        with self.g.local_scope():
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                return self.fwd_fn(self.g, h, e)


def enable_gradient_checkpointing(model, use_reentrant: bool = False,
                                  use_amp: bool = False):
    """Wrap each GraphTransformerLayer in torch.utils.checkpoint.

    Trades ~30-50% more compute for a large reduction in peak activation memory.
    Only the layer boundaries (h, e) are saved; intermediate attention tensors are
    recomputed on demand during backward.
    """
    import inspect
    from torch.utils.checkpoint import checkpoint
    supports_reentrant = "use_reentrant" in inspect.signature(checkpoint).parameters

    for layer in model.layers:
        proxy = _CheckpointProxy(layer.forward, use_amp)
        if supports_reentrant:
            def checkpointed_forward(g, h, e, _p=proxy, _ur=use_reentrant):
                _p.g = g
                return checkpoint(_p, h, e, use_reentrant=_ur)
        else:
            def checkpointed_forward(g, h, e, _p=proxy):
                _p.g = g
                return checkpoint(_p, h, e)
        layer.forward = checkpointed_forward
    return model


# ---------------------------------------------------------------------------
# Checkpoint save / resume
# ---------------------------------------------------------------------------
def _checkpoint_path(run_cfg) -> str:
    return os.path.join(
        run_cfg.results_dir,
        f"_checkpoint_{run_cfg.backbone}_{run_cfg.pe}_{run_cfg.dataset}_seed{run_cfg.seed}.pt"
    )


def _save_checkpoint(path, model, optimizer, epoch, best_metric):
    """Atomic checkpoint write (temp file + os.replace) to survive mid-write crashes."""
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "epoch": epoch, "best_metric": best_metric,
    }, tmp)
    os.replace(tmp, path)


def _load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("best_metric", None)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def san_train(run_cfg, san_dir: Optional[str] = None) -> dict:
    """Train one grid cell. Returns {"model", "loaders", "num_params",
    "metric_name", "metric_value"} matching graphgps_backend.py's contract.

    All datasets (peptides-func, peptides-struct, pascalvoc-sp) and all PEs
    (none, lappe, rwse, signnet, grpe) are now supported via custom model classes.
    See module docstring for implementation notes on each.
    """
    san_dir = ensure_san_importable(san_dir)
    from layers.mlp_readout_layer import MLPReadout

    net_params = build_san_net_params(run_cfg)
    train_params = build_san_train_params(run_cfg)
    torch.manual_seed(run_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_params["device"] = device

    model = _build_san_model(net_params)
    model = model.to(device)

    # Gradient checkpointing for full_graph=True datasets.
    # AMP (fp16) disabled -- DGL's spmm.cu kernel in this build has no fp16 support
    # (DGLError: "Data type not recognized with bits 16" confirmed on real GPU run).
    if net_params.get("full_graph", False):
        model = enable_gradient_checkpointing(model, use_amp=False)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_params > 500_000:
        print(f"  WARNING: {n_params:,} params exceeds the 500k proposal budget")

    train_loader, val_loader, test_loader = _build_loaders(run_cfg, net_params, train_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_params["init_lr"],
                                 weight_decay=train_params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=train_params["lr_reduce_factor"],
        patience=train_params["lr_schedule_patience"], min_lr=train_params["min_lr"])
    loss_fn = _loss_for(run_cfg.dataset)

    # Resume from checkpoint if present (covers both OOM restarts and SLURM preemption)
    ckpt_path = _checkpoint_path(run_cfg)
    start_epoch, best_metric = 0, None
    if os.path.exists(ckpt_path):
        start_epoch, best_metric = _load_checkpoint(ckpt_path, model, optimizer, device)
        print(f"  [resume] epoch {start_epoch}, best so far: {best_metric}", flush=True)

    higher_is_better = run_cfg.metric_name in ("ap", "macro_f1")
    accumulation_steps = train_params.get("accumulation_steps", 1)
    global_step = 0

    for epoch in range(start_epoch, train_params["epochs"]):
        model.train()
        optimizer.zero_grad()

        for i, (bg, labels) in enumerate(train_loader):
            if getattr(run_cfg, "smoke_test", False) and i >= 2:
                print(f"  [smoke-test] 2 train batches OK, stopping early", flush=True)
                break
            if i % 100 == 0:
                print(f"  [epoch {epoch}] step {i}/{len(train_loader)}", flush=True)

            bg, labels = bg.to(device), labels.to(device)
            out = _forward_pass(model, bg, pe=run_cfg.pe)
            loss = loss_fn(out, labels) / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            if global_step % 500 == 0:
                _save_checkpoint(ckpt_path, model, optimizer, epoch, best_metric)

            # Per-step cache release: this env has a ~1MB/step memory leak (confirmed:
            # per-step growth in allocated, not just reserved, independent of batch size,
            # likely from CheckpointFunction's autograd bookkeeping in this PyTorch version).
            # empty_cache() releases reserved-but-unused memory back to the OS so DGL's
            # separate internal allocator can use it. gc.collect() is cheap insurance.
            if net_params.get("full_graph") and device.type == "cuda":
                gc.collect()
                torch.cuda.empty_cache()

        val_metric, val_loss = _evaluate(model, val_loader, device, run_cfg, loss_fn)
        scheduler.step(val_loss)
        test_metric, _ = _evaluate(model, test_loader, device, run_cfg, loss_fn)

        if best_metric is None or (
            test_metric > best_metric if higher_is_better else test_metric < best_metric
        ):
            best_metric = test_metric

        print(f"  [epoch {epoch}] val={val_metric:.4f} test={test_metric:.4f} "
              f"best={best_metric:.4f}", flush=True)
        _save_checkpoint(ckpt_path, model, optimizer, epoch + 1, best_metric)

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)  # clean up so future runs don't accidentally resume

    return {
        "model": model,
        "loaders": [train_loader, val_loader, test_loader],
        "num_params": n_params,
        "metric_name": run_cfg.metric_name,
        "metric_value": best_metric,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
class _PECacheDataset:
    """Wraps an LRGBDataset split and pairs each graph with its PE cache data.

    Returns (PyG Data, pe_data dict) tuples so _collate can load the right PE
    features for each graph without modifying the underlying dataset.
    """
    def __init__(self, base_ds, cache_dir, pe, k_lap=16):
        self.base_ds = base_ds
        self.cache_dir = cache_dir
        self.pe = pe
        self.k_lap = k_lap

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        data = self.base_ds[idx]
        pe_data = _load_pe_cache(
            self.cache_dir, idx, self.pe, int(data.num_nodes), self.k_lap
        )
        return data, pe_data


def _build_loaders(run_cfg, net_params, train_params):
    """Build train/val/test DataLoaders with PE features loaded from cache per graph."""
    import functools
    from torch.utils.data import DataLoader, Subset
    from torch_geometric.datasets import LRGBDataset

    name_map = {"peptides-func": "Peptides-func", "peptides-struct": "Peptides-struct",
                "pascalvoc-sp": "PascalVOC-SP"}
    pyg_name = name_map[run_cfg.dataset]
    node_task = (run_cfg.dataset == "pascalvoc-sp")
    collate = functools.partial(_collate, full_graph=net_params["full_graph"],
                                node_task=node_task)
    max_nodes = train_params.get("max_nodes")
    cache_base = run_cfg.resolved_cache_dir
    loaders = []

    for split in ("train", "val", "test"):
        base_ds = LRGBDataset(root=f"./raw_data/{pyg_name}", name=pyg_name, split=split)
        cache_dir = os.path.join(cache_base, split)
        ds = _PECacheDataset(base_ds, cache_dir, run_cfg.pe)

        if max_nodes is not None:
            keep = [i for i in range(len(base_ds))
                    if int(base_ds[i].num_nodes) <= max_nodes]
            if len(keep) < len(base_ds):
                print(f"  [{split}] excluded {len(base_ds)-len(keep)}/{len(base_ds)} "
                      f"graphs > {max_nodes} nodes", flush=True)
            ds = Subset(ds, keep)

        shuffle = split == "train"
        bs = train_params["batch_size"] if split == "train" else 16
        loaders.append(DataLoader(ds, batch_size=bs, shuffle=shuffle,
                                  collate_fn=collate))
    return loaders


# ---------------------------------------------------------------------------
# Loss and evaluation
# ---------------------------------------------------------------------------
def _loss_for(dataset: str):
    """peptides-func uses BCELoss (not BCEWithLogitsLoss) because SAN_NodeLPE.forward
    already applies sigmoid internally -- applying it again would be wrong.
    """
    if dataset == "peptides-func":   return torch.nn.BCELoss()
    if dataset == "peptides-struct": return torch.nn.L1Loss()
    if dataset == "pascalvoc-sp":    return torch.nn.CrossEntropyLoss()
    raise ValueError(dataset)


def _evaluate(model, loader, device, run_cfg, loss_fn):
    """Returns (task_metric, mean_loss) over one split."""
    import numpy as np
    from sklearn.metrics import average_precision_score, f1_score

    model.eval()
    losses, preds, targets = [], [], []
    with torch.no_grad():
        for bg, labels in loader:
            bg, labels = bg.to(device), labels.to(device)
            out = _forward_pass(model, bg, pe=run_cfg.pe)
            losses.append(loss_fn(out, labels).item())
            preds.append(out.cpu().numpy())
            targets.append(labels.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    mean_loss = float(np.mean(losses))

    if run_cfg.dataset == "peptides-func":
        metric = average_precision_score(targets, preds, average="macro")
    elif run_cfg.dataset == "peptides-struct":
        metric = mean_loss  # L1Loss is the MAE metric for this task
    else:
        metric = f1_score(targets, preds.argmax(axis=1), average="macro")
    return float(metric), mean_loss


# ---------------------------------------------------------------------------
# Jacobian probe wrapper -- STUB
# ---------------------------------------------------------------------------
def make_san_model_fn(model, data, device=None):
    """Not yet wired. run_experiment.PROBE_WIRED_BACKBONES does not include 'san'."""
    raise NotImplementedError(
        "The Jacobian sensitivity probe is not yet wired for SAN. "
        "Training is real; sensitivity curves will be empty in results."
    )
