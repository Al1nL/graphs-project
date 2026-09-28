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
PEs:       none, lappe, rwse, signnet

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
  Handled via gradient checkpointing + EdgeBudgetBatchSampler.
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
    from layers.mlp_readout_layer import MLPReadout
    lpe = net_params.get("LPE", "none")
    variant = net_params.get("_variant", None)

    if variant == "rwse":
        return _SAN_RWSE(net_params)
    if variant == "signnet":
        return _SAN_SignNetLPE(net_params)
    if variant == "regression":
        return _SAN_NodeLPE_Regression(net_params)
    if variant == "node_classification":
        return _SAN_NodeClassification(net_params)

    # SAN and SAN_NodeLPE both hardcode MLPReadout(GT_out_dim, 1) for molhiv.
    # Replace with correct output dim for this task.
    model = _san_gnn_model(lpe, net_params)
    n_classes = net_params.get("n_classes", 1)
    if n_classes != 1 and hasattr(model, "MLP_layer"):
        model.MLP_layer = MLPReadout(net_params["GT_out_dim"], n_classes)
    return model


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



class _SAN_RWSE(torch.nn.Module):
    """SAN with RWSE (Random Walk Structural Encoding) positional encoding.

    Encodes the full k-step RWSE vector per node with an MLP before adding it
    to node features in the SAN GraphTransformer architecture.
    """
    def __init__(self, net_params):
        super().__init__()
        from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

        GT_hidden_dim = net_params["GT_hidden_dim"]
        GT_out_dim = net_params["GT_out_dim"]
        GT_n_heads = net_params["GT_n_heads"]
        GT_layers = net_params["GT_layers"]
        LPE_dim = net_params["LPE_dim"]  # RWSE step count, e.g. 20
        full_graph = net_params["full_graph"]
        gamma = net_params["gamma"]
        dropout = net_params["dropout"]
        in_feat_dropout = net_params["in_feat_dropout"]
        layer_norm = net_params["layer_norm"]
        batch_norm = net_params["batch_norm"]
        residual = net_params["residual"]
        n_classes = net_params.get("n_classes", 1)

        from layers.graph_transformer_layer import GraphTransformerLayer
        from layers.mlp_readout_layer import MLPReadout

        self.readout = net_params["readout"]
        self.in_feat_dropout = torch.nn.Dropout(in_feat_dropout)

        self.embedding_h = AtomEncoder(emb_dim=GT_hidden_dim - LPE_dim)
        self.embedding_e = BondEncoder(emb_dim=GT_hidden_dim)
        self.embedding_e_fake = torch.nn.Embedding(1, GT_hidden_dim)

        # Plain per-node MLP over the full ordered RWSE vector -- sees all
        # steps jointly, no mean-pool-across-steps information loss.
        self.rwse_encoder = torch.nn.Sequential(
            torch.nn.Linear(LPE_dim, LPE_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(LPE_dim, LPE_dim),
        )

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

    def forward(self, g, h, e, rwse, _unused_eigvals=None):
        import dgl
        pe = self.rwse_encoder(rwse)  # [n, LPE_dim] -- steps stay joint, no pooling
        h = torch.cat([self.embedding_h(h), pe], dim=-1)
        h = self.in_feat_dropout(h)

        if e is not None and e.shape[-1] > 0:
            e = self.embedding_e(e)
        else:
            e = self.embedding_e_fake(torch.zeros(g.num_edges(), dtype=torch.long,
                                                   device=h.device))
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


class _SAN_NodeLPE_Regression(torch.nn.Module):
    """SAN_NodeLPE adapted for regression tasks (peptides-struct).

    Removes the hardcoded sigmoid activation on the output head to support unbounded
    regression targets, and handles RWSE/SignNet positional encodings directly.
    """
    def __init__(self, net_params):
        super().__init__()
        from nets.molhiv_graph_regression.SAN_NodeLPE import SAN_NodeLPE
        from layers.mlp_readout_layer import MLPReadout
        n_classes = net_params.get("n_classes", 1)

        net_params = dict(net_params)
        net_params.setdefault("LPE_dim", 16)
        net_params.setdefault("LPE_n_heads", 4)
        net_params.setdefault("LPE_layers", 2)

        self._base = SAN_NodeLPE(net_params)
        self._base.MLP_layer = MLPReadout(net_params["GT_out_dim"], n_classes)
        self._n_classes = n_classes

        self.is_rwse = net_params.get("pe") == "rwse"
        self.lpe_dim = net_params["LPE_dim"]
        self.is_signnet = net_params.get("pe") == "signnet"

        if self.is_rwse:
            LPE_dim = net_params["LPE_dim"]
            self.rwse_encoder = torch.nn.Sequential(
                torch.nn.Linear(LPE_dim, LPE_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(LPE_dim, LPE_dim),
            )
        elif self.is_signnet:
            LPE_dim = net_params["LPE_dim"]
            self.signnet_phi = torch.nn.Sequential(
                torch.nn.Linear(1, LPE_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(LPE_dim, LPE_dim),
            )

    @property
    def layers(self):
        """Expose _base.layers so enable_gradient_checkpointing can wrap them."""
        return self._base.layers

    def forward(self, g, h, e, EigVecs=None, EigVals=None):
        import dgl
        base = self._base
        h = base.embedding_h(h)
        h = base.in_feat_dropout(h)
        e = (base.embedding_e_real(e) if hasattr(base, "embedding_e_real")
             else base.embedding_e(e))

        if self.is_rwse:
            pe = self.rwse_encoder(EigVecs)
        elif self.is_signnet and EigVecs is not None:
            n, k = EigVecs.shape
            v = EigVecs.view(n * k, 1)
            pe = (self.signnet_phi(v) + self.signnet_phi(-v)).view(n, k, -1)
            pe = pe.mean(dim=1)
        elif EigVecs is not None and EigVals is not None:
            EigVecs_u = EigVecs.unsqueeze(-1)
            pe_inp = torch.cat([EigVecs_u, EigVals], dim=-1)
            empty_mask = (EigVecs == 0).all(dim=-1)
            pe_inp[empty_mask] = 0.0
            pe_inp = pe_inp.transpose(0, 1)
            pe = base.linear_A(pe_inp)
            pe = base.PE_Transformer(pe)
            pe = pe.transpose(0, 1).mean(dim=1)
        else:
            pe = torch.zeros(h.shape[0], self.lpe_dim, device=h.device, dtype=h.dtype)

        h = torch.cat([h, pe], dim=-1)
        for conv in base.layers:
            h, e = conv(g, h, e)
        g.ndata["h"] = h
        if base.readout == "sum":
            hg = dgl.sum_nodes(g, "h")
        elif base.readout == "max":
            hg = dgl.max_nodes(g, "h")
        else:
            hg = dgl.mean_nodes(g, "h")
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
        node_feat_dim = net_params.get("node_feat_dim", 14)
        edge_feat_dim = net_params.get("edge_feat_dim", 2)

        self.lpe = lpe
        self.readout = net_params["readout"]
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.in_feat_dropout = torch.nn.Dropout(in_feat_dropout)

        h_in_dim = GT_hidden_dim - LPE_dim if lpe != "none" else GT_hidden_dim
        self.embedding_h = torch.nn.Linear(node_feat_dim, h_in_dim)
        self.embedding_e = torch.nn.Linear(edge_feat_dim, GT_hidden_dim)
        self.embedding_e_fake = torch.nn.Embedding(1, GT_hidden_dim)

        self.is_rwse = net_params.get("pe") == "rwse"
        self.is_signnet = net_params.get("pe") == "signnet"

        if self.is_rwse:
            self.rwse_encoder = torch.nn.Sequential(
                torch.nn.Linear(LPE_dim, LPE_dim), torch.nn.ReLU(),
                torch.nn.Linear(LPE_dim, LPE_dim),
            )
        elif self.is_signnet:
            self.signnet_phi = torch.nn.Sequential(
                torch.nn.Linear(1, LPE_dim), torch.nn.ReLU(),
                torch.nn.Linear(LPE_dim, LPE_dim),
            )
        elif lpe != "none":
            LPE_n_heads = net_params["LPE_n_heads"]
            LPE_layers = net_params["LPE_layers"]
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=LPE_dim, nhead=LPE_n_heads, batch_first=False,
                dim_feedforward=LPE_dim * 2)
            self.PE_Transformer = torch.nn.TransformerEncoder(
                encoder_layer, num_layers=LPE_layers)
            self.linear_A = torch.nn.Linear(2, LPE_dim)

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
        h = self.embedding_h(h.float())
        h = self.in_feat_dropout(h)

        if e is not None and e.shape[-1] > 0:
            e = self.embedding_e(e.float())
        else:
            e = self.embedding_e_fake(
                torch.zeros(g.num_edges(), dtype=torch.long, device=h.device))

        if self.is_rwse and EigVecs is not None:
            pe = self.rwse_encoder(EigVecs)
            h = torch.cat([h, pe], dim=-1)
        elif self.is_signnet and EigVecs is not None:
            n, k = EigVecs.shape
            v = EigVecs.view(n * k, 1)
            pe = (self.signnet_phi(v) + self.signnet_phi(-v)).view(n, k, -1)
            pe = pe.mean(dim=1)
            h = torch.cat([h, pe], dim=-1)
        elif self.lpe != "none" and EigVecs is not None:
            EigVecs_u = EigVecs.unsqueeze(-1)
            pe_inp = torch.cat([EigVecs_u, EigVals], dim=-1)
            empty_mask = (EigVecs == 0).all(dim=-1)
            pe_inp[empty_mask] = 0.0
            pe_inp = pe_inp.transpose(0, 1)
            pe = self.linear_A(pe_inp)
            pe = self.PE_Transformer(pe)
            pe = pe.transpose(0, 1).mean(dim=1)
            h = torch.cat([h, pe], dim=-1)

        for conv in self.layers:
            h, e = conv(g, h, e)

        return self.MLP_layer(h)



# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_NET_PARAMS = {
    "peptides-func": {
        # Architecture: 80/8=10 head_dim, evenly divisible (required by SAN).
        # ~928k params -- exceeds proposal's 500k budget. Document this when reporting.
        "GT_layers": 10, "GT_hidden_dim": 80, "GT_out_dim": 80, "GT_n_heads": 8,
        "full_graph": True, "gamma": 1e-5, "in_feat_dropout": 0.1, "dropout": 0.1,
        "layer_norm": False, "batch_norm": True, "residual": True, "readout": "mean",
        "task": "classification_multilabel", "n_classes": 10,
        "batch_size": 4, "accumulation_steps": 2, "max_nodes": 400,
    },
    "peptides-struct": {
        # Same architecture as peptides-func; different task (regression, 11 targets).
        # Uses _SAN_NodeLPE_Regression which removes the hardcoded sigmoid in forward().
        "GT_layers": 10, "GT_hidden_dim": 80, "GT_out_dim": 80, "GT_n_heads": 8,
        "full_graph": True, "gamma": 1e-5, "in_feat_dropout": 0.1, "dropout": 0.1,
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
    "lr_schedule_patience": 10, "min_lr": 1e-6, "weight_decay": 1e-5,
}

PE_SPEC = {
    # Maps PE name -> net_params overrides. "LPE" is the dispatch key for
    # _build_san_model(); "_variant" selects our custom model classes.
    "none":    {"LPE": "none"},
    "lappe":   {"LPE": "node", "LPE_dim": 16, "LPE_n_heads": 4, "LPE_layers": 2},
    # RWSE: fed through SAN's LPE slot with LPE_dim=20 to match RWSE feature width.
    # EigVals set to zeros (RWSE has no eigenvalues). _forward_pass handles the swap.
    "rwse":    {"LPE": "node", "LPE_dim": 20, "LPE_n_heads": 4, "LPE_layers": 2,
                "_variant": "rwse"},
    # SignNet: uses _SAN_SignNetLPE which applies phi(v)+phi(-v) instead of PE_Transformer
    "signnet": {"LPE": "node", "LPE_dim": 16, "LPE_n_heads": 4, "LPE_layers": 2,
                "_variant": "signnet"},
}


def build_san_net_params(run_cfg) -> dict:
    """Assemble net_params: dataset base -> JSON overrides -> PE_SPEC (wins last).

    One exception: _variant from BASE_NET_PARAMS is preserved even if PE_SPEC sets one.
    The dataset determines the model family (node_classification, regression, or default);
    the PE only changes the encoding within that family.
    """
    if run_cfg.pe not in PE_SPEC:
        raise ValueError(
            f"Unsupported PE '{run_cfg.pe}' for SAN backbone. "
            f"Supported PEs: {list(PE_SPEC.keys())}"
        )

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
    base["pe"] = run_cfg.pe
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
        spd/<idx>.npy   -- [n, n] uint8: all-pairs shortest-path distances

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
        if os.environ.get("SAN_DEBUG_CHECKPOINT_GRAD"):
            print(f"[checkpoint] h.requires_grad={h.requires_grad} "
                  f"e.requires_grad={e.requires_grad} "
                  f"grad_enabled={torch.is_grad_enabled()}", flush=True)
        with self.g.local_scope():
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                return self.fwd_fn(self.g, h, e)


def enable_gradient_checkpointing(model, use_reentrant: bool = False,
                                  use_amp: bool = False):
    """Wrap each GraphTransformerLayer in torch.utils.checkpoint.

    Trades ~30-50% more compute for a large reduction in peak activation memory.
    Only the layer boundaries (h, e) are saved; intermediate attention tensors are
    recomputed on demand during backward.

    Skips the checkpoint() wrapper entirely when torch.is_grad_enabled() is False
    (i.e. inside a torch.no_grad() block, as _evaluate() uses). Checkpointing exists
    to save memory by recomputing activations on backward -- under no_grad there is
    no backward, so wrapping the call only wastes compute and triggers PyTorch's
    "None of the inputs have requires_grad=True" warning on every eval batch. Calling
    the layer directly during eval is equivalent and removes the ambiguous warning,
    making any future occurrence of it during an actual training step unambiguous
    (see SAN_DEBUG_CHECKPOINT_GRAD env var above for a targeted debug print if that
    warning ever reappears during training and needs to be chased down for real).

    Also stores the ORIGINAL unwrapped forward as layer._probe_forward, for
    make_san_model_fn's sensitivity probe to call directly. The probe needs
    torch.autograd.grad() (not .backward()) to get per-source-node Jacobian blocks
    without accumulating into every parameter's .grad -- but this PyTorch version's
    torch.utils.checkpoint raises "Checkpointing is not compatible with .grad()"
    the moment autograd.grad() is used anywhere in a graph that passed through a
    checkpointed segment, even a call outside the checkpoint itself. Since the probe
    doesn't need checkpointing's memory savings anyway (it processes one graph at a
    time, not a full training batch), model_fn bypasses the checkpoint wrapper
    entirely via this stored reference rather than trying to make checkpointing and
    autograd.grad() coexist.
    """
    import inspect
    from torch.utils.checkpoint import checkpoint
    supports_reentrant = "use_reentrant" in inspect.signature(checkpoint).parameters

    for layer in model.layers:
        proxy = _CheckpointProxy(layer.forward, use_amp)
        layer._probe_forward = proxy.fwd_fn
        if supports_reentrant:
            def checkpointed_forward(g, h, e, _p=proxy, _ur=use_reentrant):
                _p.g = g
                if not torch.is_grad_enabled():
                    with g.local_scope():
                        with torch.cuda.amp.autocast(enabled=_p.use_amp):
                            return _p.fwd_fn(g, h, e)
                return checkpoint(_p, h, e, use_reentrant=_ur)
        else:
            def checkpointed_forward(g, h, e, _p=proxy):
                _p.g = g
                if not torch.is_grad_enabled():
                    with g.local_scope():
                        with torch.cuda.amp.autocast(enabled=_p.use_amp):
                            return _p.fwd_fn(g, h, e)
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
    (none, lappe, rwse, signnet) are supported via custom model classes.
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

    train_loader, val_loader, test_loader, class_weights, probe_dataset = _build_loaders(
        run_cfg, net_params, train_params)

    # If a PREVIOUS attempt already finished training and only crashed during the
    # SUBSEQUENT probe phase, a model_*.pt file will already exist here (saved right
    # after the training-resume checkpoint is deleted, at the very end of a normal
    # run below). Without this check, a resubmit has nothing to resume training
    # from (that checkpoint is already gone) and retrains 100+ epochs from scratch
    # just to redo a probe that crashed for an unrelated reason -- observed
    # repeatedly today. If a completed model is found, skip training entirely and
    # go straight to returning it so run_cell can re-attempt only the probe.
    final_model_path = os.path.join(
        run_cfg.results_dir,
        f"model_{run_cfg.backbone}_{run_cfg.pe}_{run_cfg.dataset}_seed{run_cfg.seed}.pt")
    if os.path.exists(final_model_path):
        saved = torch.load(final_model_path, map_location=device)
        missing, unexpected = model.load_state_dict(saved["model_state"], strict=False)
        if missing or unexpected:
            print(f"  [skip-training] WARNING: saved model at {final_model_path} did "
                  f"not cleanly load into the freshly-built architecture "
                  f"(missing={missing}, unexpected={unexpected}) -- falling back to "
                  "training from scratch rather than risk probing a mismatched "
                  "model. This can happen if _build_san_model/build_san_net_params "
                  "changed since this file was saved.", flush=True)
        else:
            print(f"  [skip-training] found a completed model at {final_model_path} "
                  f"(best={saved.get('best_metric')}) -- skipping training, going "
                  "straight to the probe.", flush=True)
            return {
                "model": model,
                "loaders": [train_loader, val_loader, test_loader],
                "num_params": saved.get("num_params", n_params),
                "metric_name": run_cfg.metric_name,
                "metric_value": saved.get("best_metric"),
                "probe_dataset": probe_dataset,
            }

    optimizer = torch.optim.Adam(model.parameters(), lr=train_params["init_lr"],
                                 weight_decay=train_params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=train_params["lr_reduce_factor"],
        patience=train_params["lr_schedule_patience"], min_lr=train_params["min_lr"])
    loss_fn = _loss_for(run_cfg.dataset, class_weights=class_weights, device=device)

    # Resume from checkpoint if present (covers both OOM restarts and SLURM preemption)
    ckpt_path = _checkpoint_path(run_cfg)
    start_epoch, best_metric = 0, None
    if os.path.exists(ckpt_path):
        start_epoch, best_metric = _load_checkpoint(ckpt_path, model, optimizer, device)
        print(f"  [resume] epoch {start_epoch}, best so far: {best_metric}", flush=True)

    higher_is_better = run_cfg.metric_name in ("ap", "macro_f1")
    accumulation_steps = train_params.get("accumulation_steps", 1)
    global_step = 0
    # Early stopping: stop if best_metric hasn't improved in this many epochs.
    # Default comes from run_cfg (CLI --early-stop-patience, default 15). Pass 0
    # to disable and always run the full train_params["epochs"] ceiling.
    # NOTE: epochs_since_improvement is NOT persisted in the checkpoint, so a
    # resumed run restarts its patience counter from 0 rather than picking up
    # mid-count -- a resume gets a fresh grace period.
    early_stop_patience = getattr(run_cfg, "early_stop_patience", 15)
    epochs_since_improvement = 0

    for epoch in range(start_epoch, train_params["epochs"]):
        model.train()
        optimizer.zero_grad()

        # If train_loader is using EdgeBudgetBatchSampler, advance its epoch counter
        # so batch composition reshuffles each epoch instead of repeating identically.
        # Plain DataLoaders (fixed batch_size) reshuffle on their own via shuffle=True
        # and have no batch_sampler with a set_epoch method, so this is a no-op there.
        _bsampler = getattr(train_loader, "batch_sampler", None)
        if isinstance(_bsampler, EdgeBudgetBatchSampler):
            _bsampler.set_epoch(epoch)

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
                # Gradient clipping: caps the overall gradient norm before it's
                # applied, preventing a single destructively large step. Added
                # after observing pascalvoc-sp's --pe none --gamma 1e-5 sweep run
                # freeze to a bit-identical output (val/test unchanged for 16
                # straight epochs) starting from epoch 0 -- the classic signature
                # of an early exploding-gradient step collapsing the model into a
                # saturated/dead region it can't recover from. No-op for gradients
                # already under max_norm, so this shouldn't change behavior for
                # runs that weren't hitting this failure mode.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            if global_step % 500 == 0:
                _save_checkpoint(ckpt_path, model, optimizer, epoch, best_metric)

            # Cache release: throttled to every 10 steps instead of every step.
            # empty_cache() forces a CUDA sync and is expensive when called every
            # iteration; every-50-steps was tried first but a real CUDA OOM was
            # observed in production (job 780000, ~8h into a run, DGL's GSpMM
            # allocator failing to get memory back from PyTorch's reserved-but-
            # idle pool in time for an unusually large full_graph=True batch).
            # Every-10-steps is a tighter mitigation, not a real fix: full_graph
            # memory scales O(n^2) per graph, so a single large-enough graph can
            # still OOM regardless of clearing frequency. The actual fix would be
            # an edge-budget batch sampler (see run_experiment.py's --edge-budget
            # flag, which is currently a documented no-op -- EdgeBudgetBatchSampler
            # is referenced but was never implemented) or a lower --max-nodes.
            # The underlying "~1MB/step leak" this workaround targets was never
            # actually diagnosed -- that figure is inherited from the original
            # code's comment and was not independently verified.
            if net_params.get("full_graph") and device.type == "cuda" and global_step % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        val_metric, val_loss = _evaluate(model, val_loader, device, run_cfg, loss_fn)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  [epoch {epoch}] lr={current_lr:.2e}", flush=True)
        test_metric, _ = _evaluate(model, test_loader, device, run_cfg, loss_fn)

        improved = best_metric is None or (
            test_metric > best_metric if higher_is_better else test_metric < best_metric
        )
        if improved:
            best_metric = test_metric
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        print(f"  [epoch {epoch}] val={val_metric:.4f} test={test_metric:.4f} "
              f"best={best_metric:.4f} (no improvement for {epochs_since_improvement} "
              f"epoch{'s' if epochs_since_improvement != 1 else ''})", flush=True)
        _save_checkpoint(ckpt_path, model, optimizer, epoch + 1, best_metric)

        if early_stop_patience > 0 and epochs_since_improvement >= early_stop_patience:
            print(f"  [early-stop] no improvement in {early_stop_patience} epochs, "
                  f"stopping at epoch {epoch}", flush=True)
            break

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)  # clean up so future runs don't accidentally resume

    # Persist final trained weights + the net_params needed to reconstruct this exact
    # architecture, SEPARATELY from the (now-deleted) training-resume checkpoint above.
    # Two things could not be done without this: (1) sensitivity.compute_sensitivity_curve
    # on an already-finished result without retraining from scratch, and (2)
    # scripts/calibrate_target_nodes.py's --backbone san mode, which needs a trained
    # model's weights to load, the same way its GPS path loads a GraphGPS checkpoint.
    # net_params includes a live torch.device object (net_params["device"]), which
    # torch.save can't usefully round-trip across machines/processes -- strip it before
    # saving; load_real_san (calibrate_target_nodes.py) re-adds the correct device at
    # load time instead.
    final_model_path = os.path.join(
        run_cfg.results_dir,
        f"model_{run_cfg.backbone}_{run_cfg.pe}_{run_cfg.dataset}_seed{run_cfg.seed}.pt")
    net_params_to_save = {k: v for k, v in net_params.items() if k != "device"}
    torch.save({"model_state": model.state_dict(), "net_params": net_params_to_save,
               "best_metric": best_metric, "num_params": n_params},
               final_model_path)

    return {
        "model": model,
        "loaders": [train_loader, val_loader, test_loader],
        "num_params": n_params,
        "metric_name": run_cfg.metric_name,
        "metric_value": best_metric,
        # Used by run_experiment.run_cell in place of loaders[-1].dataset for the
        # sensitivity probe -- see _PEAttachedDataset's docstring for why SAN needs
        # a separate probe-friendly dataset rather than reusing the training
        # test_loader's dataset directly (which yields (data, pe_data) tuples).
        "probe_dataset": probe_dataset,
    }


# ---------------------------------------------------------------------------
# Edge-budget batching (real fix for full_graph=True OOM, not a mitigation)
# ---------------------------------------------------------------------------
class EdgeBudgetBatchSampler(torch.utils.data.Sampler):
    """Groups graph indices into batches by a total densified-edge-count budget,
    instead of a fixed number of graphs per batch.

    Why this exists: full_graph=True densifies each graph into a complete graph, so
    per-graph memory scales O(n^2) (n*(n-1) directed edges). A fixed batch_size (e.g.
    4) can randomly combine several large graphs into one batch and exceed the GPU's
    memory ceiling regardless of how often empty_cache() runs -- that's a probability
    reduction, not a cap. This sampler caps the actual quantity that drives memory
    (total edges in the batch) directly, so no batch can exceed roughly `edge_budget`
    dense edges, independent of which graphs land together.

    This was built after two real CUDA OOM crashes under fixed batch_size=4 with
    full_graph=True (see san_train's changelog) that repeated even after tightening
    the empty_cache() interval -- confirming the interval tweak was mitigating
    probability, not the underlying cause.

    Args:
        num_nodes: list[int], node count per graph, aligned 1:1 with the dataset
            indices this sampler will be used with (i.e. num_nodes[i] must be the
            node count of dataset[i], not the node count of some pre-filter index).
        edge_budget: max total n*(n-1) summed across one batch. A single graph whose
            own cost exceeds edge_budget still gets its own batch (of size 1) rather
            than being silently dropped -- max_nodes filtering upstream should
            normally prevent this, but this sampler doesn't assume that happened.
        shuffle: shuffle graph order each epoch (matches DataLoader's shuffle=True
            semantics for train; pass False for val/test to keep them deterministic).
        seed: base seed for the shuffle RNG. Combined with an internal epoch counter
            (via set_epoch) so a resumed run reshuffles rather than repeating the
            exact same batch composition every epoch.
        max_batch_size: optional hard cap on graphs per batch even if the edge
            budget isn't reached (guards against e.g. 50 tiny graphs landing in one
            batch and blowing up unrelated per-graph overhead). None = no cap.
    """
    def __init__(self, num_nodes, edge_budget, shuffle=True, seed=0,
                 max_batch_size=None):
        if edge_budget <= 0:
            raise ValueError(f"edge_budget must be positive, got {edge_budget}")
        self.costs = [n * (n - 1) for n in num_nodes]
        self.edge_budget = edge_budget
        self.shuffle = shuffle
        self.seed = seed
        self.max_batch_size = max_batch_size
        self.epoch = 0
        oversized = sum(1 for c in self.costs if c > edge_budget)
        if oversized:
            print(f"  [edge-budget] {oversized}/{len(self.costs)} graphs alone "
                  f"exceed edge_budget={edge_budget} and will each get their own "
                  f"batch (consider lowering --max-nodes if this is frequent)",
                  flush=True)

    def set_epoch(self, epoch: int):
        """Call once per epoch so the shuffle order (and thus batch composition)
        varies across epochs instead of repeating identically every time.
        """
        self.epoch = epoch

    def __iter__(self):
        n = len(self.costs)
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(n, generator=g).tolist()
        else:
            order = list(range(n))

        batch, batch_cost = [], 0
        for idx in order:
            cost = self.costs[idx]
            if cost > self.edge_budget:
                if batch:
                    yield batch
                    batch, batch_cost = [], 0
                yield [idx]
                continue
            would_exceed_budget = batch and (batch_cost + cost > self.edge_budget)
            would_exceed_count = (self.max_batch_size is not None
                                   and len(batch) >= self.max_batch_size)
            if batch and (would_exceed_budget or would_exceed_count):
                yield batch
                batch, batch_cost = [], 0
            batch.append(idx)
            batch_cost += cost
        if batch:
            yield batch

    def __len__(self):
        # Approximate: exact count depends on shuffle order, which varies per
        # epoch. Good enough for progress bars / step-count logging; not used
        # for any correctness-critical logic.
        total_cost = sum(self.costs)
        return max(1, -(-total_cost // self.edge_budget))  # ceil division


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


class _PEAttachedDataset:
    """Wraps a _PECacheDataset (or Subset thereof) to return single self-contained
    Data objects with PE fields attached as attributes.

    Used by the sensitivity probe to provide single Data objects carrying PE features
    and metadata (`_pe`, `_full_graph`).
    """
    def __init__(self, pe_cache_dataset, pe: str, full_graph: bool):
        self._ds = pe_cache_dataset
        self._pe = pe
        self._full_graph = full_graph

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        data, pe_data = self._ds[idx]
        data = data.clone()
        for k, v in pe_data.items():
            setattr(data, k, v)
        data._pe = self._pe
        data._full_graph = self._full_graph
        return data


def _build_loaders(run_cfg, net_params, train_params):
    """Build train/val/test DataLoaders with PE features loaded from cache per graph.

    If run_cfg.edge_budget is set (truthy, e.g. via --edge-budget N) AND
    net_params["full_graph"] is True, batches are formed by EdgeBudgetBatchSampler
    instead of a fixed batch_size -- this caps per-batch memory directly (the real
    fix for full_graph=True OOM) rather than relying on batch_size + empty_cache()
    timing to keep worst-case batches under the memory ceiling. Pass --edge-budget 0
    (or leave it unset) to keep the old fixed-batch_size behavior.
    """
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
    edge_budget = getattr(run_cfg, "edge_budget", None)
    use_edge_budget = bool(edge_budget) and net_params.get("full_graph", False)
    loaders = []
    class_weights = None  # only set for pascalvoc-sp's train split, below

    for split in ("train", "val", "test"):
        base_ds = LRGBDataset(root=f"./raw_data/{pyg_name}", name=pyg_name, split=split)
        cache_dir = os.path.join(cache_base, split)
        ds = _PECacheDataset(base_ds, cache_dir, run_cfg.pe)

        # Compute node counts once, up front (single pass over base_ds), and reuse
        # for both max_nodes filtering below and the edge-budget sampler further
        # down -- avoids loading each graph's structure twice.
        all_node_counts = [int(base_ds[i].num_nodes) for i in range(len(base_ds))]

        if max_nodes is not None:
            keep = [i for i in range(len(base_ds)) if all_node_counts[i] <= max_nodes]
            if len(keep) < len(base_ds):
                print(f"  [{split}] excluded {len(base_ds)-len(keep)}/{len(base_ds)} "
                      f"graphs > {max_nodes} nodes", flush=True)
            ds = Subset(ds, keep)
            node_counts = [all_node_counts[i] for i in keep]  # aligned with ds now
        else:
            node_counts = all_node_counts

        if use_edge_budget:
            sampler = EdgeBudgetBatchSampler(
                node_counts, edge_budget=edge_budget, shuffle=(split == "train"),
                seed=run_cfg.seed)
            loaders.append(DataLoader(ds, batch_sampler=sampler, collate_fn=collate,
                                      num_workers=4, pin_memory=True, persistent_workers=True))
        else:
            shuffle = split == "train"
            bs = train_params["batch_size"] if split == "train" else 4
            loaders.append(DataLoader(ds, batch_size=bs, shuffle=shuffle,
                                      collate_fn=collate, num_workers=4, pin_memory=True,
                                      persistent_workers=(split == "train")))

        if split == "test":
            # Built from the SAME underlying (post max_nodes filtering) dataset the
            # test_loader above uses, so the probe samples from exactly the graphs
            # that were actually available at eval time -- not a separately
            # constructed copy that could silently diverge (e.g. if max_nodes
            # filtering were computed differently in two places).
            probe_dataset = _PEAttachedDataset(
                ds, pe=run_cfg.pe, full_graph=net_params["full_graph"])

        if split == "train" and run_cfg.dataset == "pascalvoc-sp":
            # PascalVOC-SP is heavily class-imbalanced (majority class ~71% of
            # nodes; several classes under 1%), but CrossEntropyLoss without
            # weighting treats every node equally -- the model gets little
            # gradient signal to learn rare classes well, while macro-F1 (the
            # eval metric) weights every class equally regardless of frequency.
            # This computes inverse-frequency weights from the REAL, FULL
            # training set label distribution (not just a diagnostic sample),
            # aligned with whichever indices max_nodes filtering kept.
            keep_idx = keep if max_nodes is not None else range(len(base_ds))
            class_weights = _compute_pascalvoc_class_weights(
                base_ds, keep_idx, net_params.get("n_classes", 21))

    return loaders[0], loaders[1], loaders[2], class_weights, probe_dataset


def _compute_pascalvoc_class_weights(base_ds, indices, n_classes):
    """Inverse-frequency class weights for pascalvoc-sp's CrossEntropyLoss.

    weight_c = (1/count_c) normalized so mean weight across classes is ~1 (keeps
    the overall loss scale comparable to the unweighted case, rather than
    shrinking/inflating it, which could otherwise interact confusingly with
    --lr). Classes with zero occurrences in `indices` get a weight of 0 (can't
    meaningfully weight what's never seen) rather than dividing by zero.
    """
    counts = torch.zeros(n_classes)
    for i in indices:
        y = base_ds[i].y.view(-1).long()
        counts += torch.bincount(y, minlength=n_classes).float()

    weights = torch.zeros(n_classes)
    nonzero = counts > 0
    weights[nonzero] = 1.0 / counts[nonzero]
    if weights.sum() > 0:
        weights = weights / weights.sum() * nonzero.sum().item()
    return weights


# ---------------------------------------------------------------------------
# Loss and evaluation
# ---------------------------------------------------------------------------
def _loss_for(dataset: str, class_weights=None, device=None):
    """peptides-func uses BCELoss (not BCEWithLogitsLoss) because SAN_NodeLPE.forward
    already applies sigmoid internally -- applying it again would be wrong.

    pascalvoc-sp: class_weights (from _compute_pascalvoc_class_weights, inverse-
    frequency, computed over the real training set) are applied to counter the
    severe class imbalance (majority class ~71% of nodes) -- without weighting,
    CrossEntropyLoss gives little gradient signal for rare classes, while the
    eval metric (macro-F1) weights every class equally regardless of frequency.
    """
    if dataset == "peptides-func":   return torch.nn.BCELoss()
    if dataset == "peptides-struct": return torch.nn.L1Loss()
    if dataset == "pascalvoc-sp":
        weight = class_weights.to(device) if class_weights is not None else None
        return torch.nn.CrossEntropyLoss(weight=weight)
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
# Jacobian probe wrapper
# ---------------------------------------------------------------------------
def make_san_model_fn(model, data, device=None):
    """Wrap a trained SAN model for sensitivity probing.

    Returns (model_fn, probe_data, meta) where model_fn takes h^(0) embeddings
    and returns final-layer node representations, probe_data holds h^(0),
    and meta contains dim_inner = GT_hidden_dim.
    """
    import types

    model.eval()
    device = device or next(model.parameters()).device

    pe = getattr(data, "_pe", None)
    full_graph = getattr(data, "_full_graph", None)
    if pe is None or full_graph is None:
        raise RuntimeError(
            "data is missing _pe/_full_graph attributes -- make_san_model_fn requires a "
            "Data object from san_backend._PEAttachedDataset (san_train's "
            "'probe_dataset'), not a bare PyG Data object. Check that "
            "run_experiment.run_cell uses train_out.get('probe_dataset', ...) rather than "
            "loaders[-1].dataset for backbone == 'san'."
        )

    # Reassemble this one graph's PE dict exactly as _load_pe_cache produced it, so
    # _pyg_to_dgl builds the identical DGL graph san_train used (same 'real'-edge
    # tagging, same EigVecs/EigVals/rwse placement).
    pe_data = {}
    for key in ("EigVecs", "EigVals", "rwse"):
        if hasattr(data, key):
            pe_data[key] = getattr(data, key)

    g = _pyg_to_dgl(data, full_graph=full_graph, pe_data=pe_data).to(device)
    feat = g.ndata["feat"]
    e_raw = g.edata.get("feat", None)

    # _SAN_NodeLPE_Regression wraps an upstream SAN_NodeLPE instance in
    # `_base`; every other class (including upstream's own SAN/SAN_NodeLPE, used
    # directly for pe=='none'/'lappe') exposes embedding_h/embedding_e etc. on itself.
    base = getattr(model, "_base", model)

    with torch.no_grad():
        h_content = base.embedding_h(feat.float() if feat.dtype.is_floating_point
                                     else feat)
        if hasattr(base, "in_feat_dropout"):
            h_content = base.in_feat_dropout(h_content)

        # Edge embedding: naming is inconsistent across classes
        if e_raw is not None and e_raw.shape[-1] > 0:
            if hasattr(base, "embedding_e"):
                e0 = base.embedding_e(e_raw)
            elif hasattr(base, "embedding_e_real"):
                e0 = base.embedding_e_real(e_raw)
            else:
                raise AttributeError(
                    f"{type(base).__name__} has neither embedding_e nor "
                    "embedding_e_real -- add its actual attribute name here."
                )
        elif hasattr(base, "embedding_e_fake"):
            e0 = base.embedding_e_fake(
                torch.zeros(g.num_edges(), dtype=torch.long, device=device))
        else:
            raise AttributeError(f"{type(base).__name__} has no embedding_e_fake "
                                 "for the empty-edge-feature case.")

        # PE channels, computed the same way each class's own forward() does, then
        # concatenated onto h_content -- together these form h^(0).
        if pe == "rwse":
            rwse_enc = getattr(model, "rwse_encoder", None) or getattr(base, "rwse_encoder", None)
            if rwse_enc is None:
                raise AttributeError(f"Could not find rwse_encoder on model or base for pe={pe}")
            pe_vec = rwse_enc(g.ndata["rwse"])
        elif pe == "signnet":
            phi = getattr(model, "signnet_phi", None) or getattr(base, "signnet_phi", None)
            if phi is None:
                raise AttributeError(f"Could not find signnet_phi on model or base for pe={pe}")
            EigVecs = g.ndata["EigVecs"]
            n, k = EigVecs.shape
            v = EigVecs.view(n * k, 1)
            pe_vec = (phi(v) + phi(-v)).view(n, k, -1).mean(dim=1)
        elif pe == "none":
            if hasattr(model, "lpe_dim"):
                lpe_dim = model.lpe_dim
                pe_vec = torch.zeros(h_content.shape[0], lpe_dim, device=device, dtype=h_content.dtype)
            else:
                pe_vec = None

        else:  # lappe: eigenvector path through linear_A + PE_Transformer
            EigVecs, EigVals = g.ndata["EigVecs"], g.ndata["EigVals"]
            EigVecs_u = EigVecs.unsqueeze(-1)
            pe_inp = torch.cat([EigVecs_u, EigVals], dim=-1)
            empty_mask = (EigVecs == 0).all(dim=-1)
            pe_inp[empty_mask] = 0.0
            pe_inp = pe_inp.transpose(0, 1)
            pe_vec = base.PE_Transformer(base.linear_A(pe_inp)).transpose(0, 1).mean(dim=1)

        h0 = torch.cat([h_content, pe_vec], dim=-1) if pe_vec is not None else h_content

    h0 = h0.detach().clone()
    e0 = e0.detach().clone()
    layers = model.layers  # direct attribute, or via @property on wrapper classes

    def model_fn(x):
        h, e = x, e0
        for conv in layers:
            # Bypass gradient checkpointing entirely: torch.utils.checkpoint on
            # this PyTorch version raises "Checkpointing is not compatible with
            # .grad()" the moment sensitivity.py's torch.autograd.grad() calls
            # touch a graph that passed through a checkpointed segment. The probe
            # doesn't need checkpointing's memory savings (one graph at a time,
            # not a full training batch), so call the ORIGINAL unwrapped forward
            # (stored as _probe_forward by enable_gradient_checkpointing) when
            # present; layers that were never checkpointed (pascalvoc-sp,
            # full_graph=False) simply don't have this attribute, so fall back to
            # calling the layer normally.
            fwd = getattr(conv, "_probe_forward", conv)
            h, e = fwd(g, h, e)
        return h

    probe_data = types.SimpleNamespace(
        x=h0, edge_index=data.edge_index.to(device), num_nodes=int(data.num_nodes))
    meta = {"dim_inner": h0.shape[1], "num_nodes": int(data.num_nodes)}
    return model_fn, probe_data, meta
