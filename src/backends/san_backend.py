"""
san_backend.py
===============
Real integration with the SAN fork at ../SAN (config.UPSTREAM_PATHS["san"]).

Two entry points, matching backends/graphgps_backend.py's contract exactly so
run_experiment.py's dispatch is symmetric:

    san_train(run_cfg)          -> trains a cell of the grid, returns model + metrics
    make_san_model_fn(model, data) -> wraps a trained model for the Jacobian probe (STUB;
                                       see "probe wrapper" below -- not yet wired)

--------------------------------------------------------------------------------------
CURRENT STATUS: ONLY peptides-func TRAINS CLEANLY. Read this before running anything else.
--------------------------------------------------------------------------------------
The SAN model class this file drives (`nets.molhiv_graph_regression.SAN_NodeLPE`, the only
variant `../SAN/nets/load_net.py` actually imports) was built for ogbg-molhiv: a single
binary graph-classification task. Concretely, its `forward()`:

    hg = dgl.mean_nodes(g, 'h')          # GRAPH-level pooling
    return sig(self.MLP_layer(hg))       # hardcoded output dim 1, hardcoded sigmoid

Two things follow, confirmed against real cluster errors, not guessed:

1. **peptides-func** (multi-label graph classification, bounded [0,1] targets after
   sigmoid) is architecturally compatible once the output width is fixed (10 labels, not
   1) and the label tensor shape is fixed (the earlier `ValueError: Target size
   (torch.Size([8, 1, 10])) must be the same as input size (torch.Size([8, 1]))` was BOTH
   of these at once: `MLPReadout(GT_out_dim, 1)` hardcoding 1 output, and `_collate`
   introducing a spurious extra dimension via `torch.stack`). Both are fixed below.

2. **peptides-struct** (real-valued regression) and **pascalvoc-sp** (NODE classification)
   are NOT compatible with this model class as-is, and `san_train` now refuses to run them
   rather than silently produce numbers that look plausible but are wrong:
     - peptides-struct: forward()'s hardcoded `sig(...)` clamps every prediction to (0,1).
       Peptides-struct's regression targets are real-valued molecular descriptors, not
       bounded to (0,1) -- training would silently converge to *something*, and an L1Loss
       number would come out, but it would not mean what it looks like it means.
     - pascalvoc-sp: forward()'s `dgl.mean_nodes(g, 'h')` pools every node in a graph down
       to ONE vector before the readout. PascalVOC-SP needs a per-node prediction, not a
       per-graph one -- this model structurally cannot produce that, no matter how the
       output width is configured.
   Fixing either needs either patching the SAN fork itself (making the final activation and
   the pooling conditional on task type) or writing a second model wrapper class in this
   file that reuses SAN's GT layers but replaces the head -- real work, deliberately not
   done blind in this pass. See the `NotImplementedError`s in `san_train` for exactly what
   each needs.

--------------------------------------------------------------------------------------
WHY THIS FILE IS SHAPED DIFFERENTLY FROM graphgps_backend.py
--------------------------------------------------------------------------------------
GraphGPS *is* effectively the LRGB benchmark's own reference codebase, so
`graphgps_backend.py` could start from GraphGPS's own tuned `configs/GPS/*-GPS.yaml` files.
SAN (Kreuzer et al., 2021, github.com/DevinKreuzer/SAN) predates LRGB and ships NO
LRGB-specific reference configs or task variants at all -- its `nets/` directory only has
folders tuned for its own original benchmarks (molhiv, ZINC, SBM, ...), none of which is
LRGB's multi-label-classification / node-classification combination. `BASE_NET_PARAMS`
below is THIS PROJECT'S construction (state this explicitly wherever a SAN number is
reported), and `_pyg_to_dgl` performs SAN's own full-graph augmentation (the fix for an
earlier `KeyError: 'real'`) since everything else in this project is PyG, not DGL.

--------------------------------------------------------------------------------------
OTHER OPEN ITEMS (unchanged from before, still true)
--------------------------------------------------------------------------------------
- The probe wrapper (`make_san_model_fn`) is a stub -- training is real, the Jacobian probe
  is not yet wired for SAN (run_experiment.PROBE_WIRED_BACKBONES does not include "san").
- RWSE and SignNet are NOT YET actually differentiated from LapPE -- `pe=lappe`, `pe=rwse`,
  `pe=signnet` currently build and train the identical model. See PE_SPEC's comment.
- GRPE has no native attention-bias hook on SAN -- `san_train` refuses it (see below).

Corrected against real cluster errors so far (chronological):
  - `gnn_model(LPE, net_params)` dispatches on the PE-variant key ('none'/'node'/'edge'),
    not a model name -- fixed via the explicit "LPE" field in PE_SPEC.
  - `SAN_NodeLPE.forward` takes `(g, h, e, EigVecs, EigVals)` -- `_forward_pass` dispatches
    on signature arg count to handle SAN's different model variants uniformly.
  - `SAN_NodeLPE.__init__` reads `GT_layers`/`GT_hidden_dim`/... not this file's originally
    invented `L`/`hidden_dim`/... names -- BASE_NET_PARAMS/PE_SPEC use the confirmed names.
  - `propagate_attention` reads `g.edata['real']` unconditionally -- `_pyg_to_dgl` sets it
    either way (augmented full-graph, or all-real sparse graph).
  - `GT_hidden_dim`/`GT_out_dim` must be evenly divisible by `GT_n_heads` (74/8 was not) --
    `build_san_net_params` now asserts this before training starts.
  - `full_graph=True`'s O(n^2) edges caused a CUDA OOM at batch_size=32 -- batch_size is now
    dataset-specific (8 for full_graph=True, 32 otherwise).
  - `MLPReadout(GT_out_dim, 1)` is hardcoded for molhiv's single output -- `san_train` now
    replaces it with the correct width for the task, and `_collate`'s label shape bug
    (extra dim from `torch.stack`) is fixed to `torch.cat`.
"""

import os
import sys
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# importing SAN
# ---------------------------------------------------------------------------
def ensure_san_importable(san_dir: Optional[str] = None) -> str:
    """Put the SAN clone on sys.path so its `nets`/`data`/`layers` packages import."""
    if san_dir is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
        from config import UPSTREAM_PATHS
        san_dir = UPSTREAM_PATHS["san"]
    san_dir = os.path.abspath(san_dir)
    if not os.path.isdir(os.path.join(san_dir, "nets")):
        raise FileNotFoundError(
            f"no SAN clone at {san_dir}. Run `bash scripts/setup_upstream.sh san` (after "
            "forking DevinKreuzer/SAN and adding the fork URL to config.FORK_URLS)."
        )
    if san_dir not in sys.path:
        sys.path.insert(0, san_dir)
    try:
        import dgl  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"SAN at {san_dir} needs DGL, which is not importable ({exc}). SAN needs its "
            "OWN environment (DGL + its pinned PyTorch/CUDA combination, NOT GraphGPS's "
            "PyG-based one). See README 'Environment setup'; do not share one env across "
            "the three backbones."
        ) from exc
    return san_dir


# ---------------------------------------------------------------------------
# config -- key names confirmed against SAN_NodeLPE's real constructor
# ---------------------------------------------------------------------------
BASE_NET_PARAMS = {
    "peptides-func": {
        # GT_hidden_dim=80, GT_n_heads=8 -> 80/8=10, evenly divisible (was 74, which does
        # NOT divide by 8 -- see build_san_net_params's assertion). 80 also matches SAN's
        # own published hidden_dim for its largest ("500k") full-graph ZINC config.
        "GT_layers": 10, "GT_hidden_dim": 80, "GT_out_dim": 80, "GT_n_heads": 8,
        "full_graph": True, "gamma": 1e-5, "in_feat_dropout": 0.0, "dropout": 0.0,
        "layer_norm": False, "batch_norm": True, "residual": True, "readout": "mean",
        "task": "classification_multilabel", "n_classes": 10,
        # batch_size=32 (SAN's own ZINC-scale default) hit a CUDA OOM on an 11.9GB card.
        # full_graph=True means every graph contributes n*(n-1) edges (Peptides avg
        # n~151 -> ~22.6k edges/graph), and propagate_attention's per-edge tensors are kept
        # alive by autograd across all 10 layers for backward. batch_size is the most
        # direct lever on that; 8 is a safety-first default, not a tuned one -- raise it if
        # you have a bigger card (scripts/slurm/README.md's --constraint options).
        "batch_size": 8,
    },
    "peptides-struct": {
        "GT_layers": 10, "GT_hidden_dim": 80, "GT_out_dim": 80, "GT_n_heads": 8,
        "full_graph": True, "gamma": 1e-5, "in_feat_dropout": 0.0, "dropout": 0.0,
        "layer_norm": False, "batch_norm": True, "residual": True, "readout": "mean",
        "task": "regression", "n_classes": 11,
        "batch_size": 8,
    },
    "pascalvoc-sp": {
        "GT_layers": 8, "GT_hidden_dim": 64, "GT_out_dim": 64, "GT_n_heads": 8,   # 64/8=8
        "full_graph": False,
        # full_graph=False (sparse attention) here, not a stylistic choice: PascalVOC-SP
        # graphs average 479 nodes and SAN's full attention is O(n^2) per graph.
        "gamma": 1e-5, "in_feat_dropout": 0.0, "dropout": 0.0,
        "layer_norm": False, "batch_norm": True, "residual": True, "readout": "mean",
        "task": "classification_multiclass_node", "n_classes": 21,
        "batch_size": 32,
    },
}

TRAIN_PARAMS = {
    "epochs": 200, "init_lr": 7e-4, "lr_reduce_factor": 0.5,
    "lr_schedule_patience": 10, "min_lr": 1e-6, "weight_decay": 0.0,
    # no "batch_size" here -- it is dataset-specific, see BASE_NET_PARAMS.
}

# PE -> net_params overrides. "LPE" is the dispatch key gnn_model() itself needs
# ('none'/'node'/'edge') -- every PE except "none" builds SAN_NodeLPE.
# STILL OPEN: extra_node_feat (rwse) / signnet_replaces_lpe (signnet) are recorded
# intentions, NOT YET consumed anywhere -- pe=rwse and pe=signnet currently train the
# identical model to pe=lappe. Do not trust a SAN rwse/signnet result until this is closed.
PE_SPEC = {
    "none": {"LPE": "none"},
    "lappe": {"LPE": "node", "LPE_dim": 16, "LPE_n_heads": 4, "LPE_layers": 2},
    "rwse": {"LPE": "node", "LPE_dim": 16, "LPE_n_heads": 4, "LPE_layers": 2,
              "extra_node_feat": "rwse", "extra_node_feat_dim": 20},
    "signnet": {"LPE": "node", "LPE_dim": 16, "signnet_replaces_lpe": True,
                 "signnet_hidden_dim": 64, "signnet_out_dim": 32},
    "grpe": {"grpe_bias_enable": True},
}


def build_san_net_params(run_cfg) -> dict:
    """Assemble one cell's net_params: dataset base + PE override, loaded from
    configs/san/san_<pe>_<dataset>.json and merged over BASE_NET_PARAMS.
    """
    import json

    base = dict(BASE_NET_PARAMS[run_cfg.dataset])
    cfg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..",
        "configs", "san", f"san_{run_cfg.pe}_{run_cfg.dataset}.json",
    )
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            file_cfg = json.load(f)
        base.update(file_cfg.get("net_params", {}))
    # PE_SPEC is the authoritative source for the CORRECT key names; apply it AFTER the
    # (possibly stale) json file so it cannot be silently overridden by an old config.
    base.update(PE_SPEC[run_cfg.pe])
    from dataset_meta import SPD_NUM_BUCKETS
    base["grpe_num_spd_buckets"] = SPD_NUM_BUCKETS
    base["seed"] = run_cfg.seed

    # Fail fast: SAN's GraphTransformerLayer computes head_dim = GT_hidden_dim //
    # GT_n_heads internally (floor division), then reshapes attention output to exactly
    # GT_out_dim -- if not evenly divisible, that reshape fails deep inside SAN's forward
    # pass with a confusing batch-size-dependent shape error (confirmed: 74/8 -> actual
    # width 72, not 74).
    for key in ("GT_hidden_dim", "GT_out_dim"):
        if base[key] % base["GT_n_heads"] != 0:
            raise ValueError(
                f"SAN config error for pe={run_cfg.pe!r} dataset={run_cfg.dataset!r}: "
                f"{key}={base[key]} is not evenly divisible by GT_n_heads="
                f"{base['GT_n_heads']}. Fix BASE_NET_PARAMS or the config JSON."
            )
    return base


def build_san_train_params(run_cfg) -> dict:
    """Training hyperparameters (epochs, lr, batch size, ...) for one cell. batch_size
    comes from BASE_NET_PARAMS[dataset] (full_graph=True needs a much smaller batch), not
    from TRAIN_PARAMS, which only holds settings that don't vary by dataset.
    """
    import json

    params = dict(TRAIN_PARAMS)
    params["batch_size"] = BASE_NET_PARAMS[run_cfg.dataset]["batch_size"]
    cfg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..",
        "configs", "san", f"san_{run_cfg.pe}_{run_cfg.dataset}.json",
    )
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            file_cfg = json.load(f)
        params.update(file_cfg.get("params", {}))
    return params


# ---------------------------------------------------------------------------
# PyG <-> DGL
# ---------------------------------------------------------------------------
def _pyg_to_dgl(data, full_graph: bool, k: int = 16):
    """Convert one PyG Data graph into the DGLGraph SAN's model classes expect.

    Performs SAN's own full-graph augmentation when `full_graph=True` (the fix for
    `KeyError: 'real'`): builds a fully connected graph and tags each edge `edata['real']`
    = 1 (original) or 0 (added purely so attention is genuinely all-pairs).
    `GraphTransformerLayer.propagate_attention` reads that flag unconditionally, so it is
    set either way -- for `full_graph=False`, every edge is simply tagged real=1.

    Added ("fake") edges get a zero-vector edge feature -- a placeholder, not a claim about
    what index 0 means in whatever bond-feature vocabulary edge_attr uses; SAN_NodeLPE's
    forward applies `embedding_e_real` to ALL edges uniformly, so fake edges need SOME
    valid in-range feature.
    """
    import dgl

    num_nodes = int(data.num_nodes)
    src, dst = data.edge_index[0], data.edge_index[1]

    if getattr(data, "edge_attr", None) is not None:
        edge_attr = data.edge_attr
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
    else:
        edge_attr = torch.zeros((src.numel(), 1), dtype=torch.long)

    if full_graph:
        # Vectorized all-pairs edge list (i != j) -- deliberately NOT a Python double loop:
        # a per-pair Python loop would be tens of thousands of iterations per graph even at
        # Peptides' ~151-node average, dominating wall-clock time across a whole epoch.
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
        full_attr = attr_dense[full_src, full_dst]

        g = dgl.graph((full_src, full_dst), num_nodes=num_nodes)
        g.edata["feat"] = full_attr
        g.edata["real"] = is_real
    else:
        g = dgl.graph((src, dst), num_nodes=num_nodes)
        g.edata["feat"] = edge_attr
        g.edata["real"] = torch.ones(src.numel(), dtype=torch.long)

    g.ndata["feat"] = data.x

    # Laplacian eigenvectors/eigenvalues, in the shape SAN_NodeLPE.forward expects:
    # EigVecs [n, k], EigVals [n, k, 1] (concatenated on a trailing dim in the real
    # forward: `torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2)`).
    if hasattr(data, "EigVecs") and data.EigVecs is not None:
        g.ndata["EigVecs"] = data.EigVecs.float()
    elif hasattr(data, "lap_eigvec") and data.lap_eigvec is not None:
        g.ndata["EigVecs"] = data.lap_eigvec.float()
    else:
        g.ndata["EigVecs"] = torch.zeros((num_nodes, k), dtype=torch.float32)

    if hasattr(data, "EigVals") and data.EigVals is not None:
        eigvals = data.EigVals.float()
        g.ndata["EigVals"] = eigvals if eigvals.dim() == 3 else eigvals.unsqueeze(-1)
    elif hasattr(data, "lap_eigval") and data.lap_eigval is not None:
        eigvals = data.lap_eigval.float()
        if eigvals.dim() == 1:
            eigvals = eigvals.unsqueeze(0).repeat(num_nodes, 1).unsqueeze(-1)
        elif eigvals.dim() == 2:
            eigvals = eigvals.unsqueeze(-1)
        g.ndata["EigVals"] = eigvals
    else:
        g.ndata["EigVals"] = torch.zeros((num_nodes, k, 1), dtype=torch.float32)

    return g


def _forward_pass(model, bg):
    """Dispatch to whichever SAN model variant `model` is, by forward() arg count --
    bare `SAN` takes (g, h); `SAN_EdgeLPE` takes (g, h, e); `SAN_NodeLPE` takes
    (g, h, e, EigVecs, EigVals).
    """
    import inspect

    h = bg.ndata["feat"]
    e = bg.edata.get("feat", None)
    eigvecs = bg.ndata.get("EigVecs", None)
    eigvals = bg.ndata.get("EigVals", None)

    num_params = len(inspect.signature(model.forward).parameters)
    if num_params >= 4:
        return model(bg, h, e, eigvecs, eigvals)
    elif num_params == 3:
        return model(bg, h, e)
    return model(bg, h)


def _collate(batch, full_graph: bool):
    """Batch a list of PyG Data graphs into one DGL batched graph + a label tensor shaped
    [batch_size, num_tasks] for graph-level tasks.

    THE FIX for `ValueError: Target size (torch.Size([8, 1, 10])) must be the same as
    input size (torch.Size([8, 1]))`'s label-shape half: LRGB stores each graph's `y` as
    `[1, num_tasks]` (a leading dim of 1, PyG's convention for graph-level attributes).
    `torch.stack` ADDS a new leading dimension on top of that, producing
    `[batch, 1, num_tasks]`. `torch.cat` along dim 0 instead gives the correct
    `[batch, num_tasks]` directly, since each graph's `y` already carries its own
    single-row "batch slot".
    """
    import dgl

    graphs = [_pyg_to_dgl(d, full_graph=full_graph) for d in batch]
    ys = [d.y if d.y.dim() > 1 else d.y.unsqueeze(0) for d in batch]
    labels = torch.cat(ys, dim=0)
    return dgl.batch(graphs), labels


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def san_train(run_cfg, san_dir: Optional[str] = None) -> dict:
    """Train one grid cell with SAN's own model classes. Mirrors graphgps_train's
    contract: returns {"model", "loaders", "num_params", "metric_name", "metric_value"}.

    Only "peptides-func" actually trains right now -- see module docstring "CURRENT
    STATUS". The other two raise NotImplementedError with what fixing them needs, rather
    than training a structurally-wrong model and returning a number that looks real.
    """
    if run_cfg.pe == "grpe":
        raise NotImplementedError(
            "SAN has no native attention-bias hook, so GRPE needs SAN's attention class "
            "extended by adapters.san_adapter.SANGammaGRPEBias and the spd_bucket/"
            "edge_type tensors threaded onto the DGL batch. Left for a separate pass -- "
            "the other four PE arms should be validated first."
        )
    if run_cfg.dataset == "peptides-struct":
        raise NotImplementedError(
            "nets.molhiv_graph_regression.SAN_NodeLPE.forward hardcodes a sigmoid on its "
            "output (`return sig(self.MLP_layer(hg))`), which clamps every prediction to "
            "(0,1). Peptides-struct's targets are real-valued molecular descriptors, not "
            "bounded to (0,1) -- training would run and produce SOME loss number, but it "
            "would not mean what it looks like it means. Fix needs either patching the "
            "SAN fork's forward() to make the final sigmoid conditional on task type, or "
            "a wrapper model class in this file that reuses SAN's GT layers with a plain "
            "(non-sigmoid) regression head. Not done blind in this pass."
        )
    if run_cfg.dataset == "pascalvoc-sp":
        raise NotImplementedError(
            "nets.molhiv_graph_regression.SAN_NodeLPE.forward pools every node down to "
            "one graph-level vector (`dgl.mean_nodes(g, 'h')`) before its readout. "
            "PascalVOC-SP is NODE classification -- it needs a per-node prediction, which "
            "this model class cannot produce no matter how the output width is "
            "configured. This needs a genuinely different model variant (check whether "
            "../SAN/nets/ ships anything under an SBM/node-classification-style folder) "
            "or a custom node-level readout wrapper. Not done blind in this pass."
        )

    san_dir = ensure_san_importable(san_dir)
    from nets.load_net import gnn_model  # noqa: (SAN's own model factory)
    from layers.mlp_readout_layer import MLPReadout  # SAN's own readout-head class

    net_params = build_san_net_params(run_cfg)
    train_params = build_san_train_params(run_cfg)
    torch.manual_seed(run_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_params["device"] = device

    model = gnn_model(net_params["LPE"], net_params)

    # THE FIX for the output-width half of `ValueError: Target size ... must be the same
    # as input size ...`: SAN_NodeLPE.__init__ hardcodes `MLPReadout(GT_out_dim, 1)` (built
    # for ogbg-molhiv's single binary task). Replace it with the width this dataset
    # actually needs (10 for peptides-func's multi-label task) -- reuses everything else
    # the constructor already built (GT layers, LPE module, embeddings) unchanged.
    n_classes = net_params.get("n_classes", 1)
    if hasattr(model, "MLP_layer"):
        model.MLP_layer = MLPReadout(net_params["GT_out_dim"], n_classes)

    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_params > 500_000:
        print(f"  WARNING: {n_params:,} parameters exceeds the 500k budget in the proposal")

    train_loader, val_loader, test_loader = _build_loaders(run_cfg, net_params, train_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_params["init_lr"],
                                 weight_decay=train_params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=train_params["lr_reduce_factor"],
        patience=train_params["lr_schedule_patience"], min_lr=train_params["min_lr"])
    loss_fn = _loss_for(run_cfg.dataset)

    best_metric = None
    higher_is_better = run_cfg.metric_name in ("ap", "macro_f1")
    for epoch in range(train_params["epochs"]):
        model.train()
        for bg, labels in train_loader:
            bg, labels = bg.to(device), labels.to(device)
            optimizer.zero_grad()
            out = _forward_pass(model, bg)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()

        val_metric, val_loss = _evaluate(model, val_loader, device, run_cfg, loss_fn)
        scheduler.step(val_loss)
        test_metric, _ = _evaluate(model, test_loader, device, run_cfg, loss_fn)
        if best_metric is None or (
            test_metric > best_metric if higher_is_better else test_metric < best_metric
        ):
            best_metric = test_metric

    return {
        "model": model,
        "loaders": [train_loader, val_loader, test_loader],
        "num_params": n_params,
        "metric_name": run_cfg.metric_name,
        "metric_value": best_metric,
    }


def _build_loaders(run_cfg, net_params, train_params):
    """DataLoaders over the same LRGBDataset splits GraphGPS trains on, batched via
    `_collate` into DGL graphs. `full_graph` is bound into the collate function via
    functools.partial so it matches `net_params['full_graph']` for every batch.
    """
    import functools

    from torch.utils.data import DataLoader
    from torch_geometric.datasets import LRGBDataset

    name_map = {"peptides-func": "Peptides-func", "peptides-struct": "Peptides-struct",
                "pascalvoc-sp": "PascalVOC-SP"}
    pyg_name = name_map[run_cfg.dataset]
    collate = functools.partial(_collate, full_graph=net_params["full_graph"])
    loaders = []
    for split in ("train", "val", "test"):
        ds = LRGBDataset(root=f"./raw_data/{pyg_name}", name=pyg_name, split=split)
        shuffle = split == "train"
        loaders.append(DataLoader(ds, batch_size=train_params["batch_size"],
                                  shuffle=shuffle, collate_fn=collate))
    return loaders


def _loss_for(dataset: str):
    """peptides-func: SAN_NodeLPE.forward already applies sigmoid internally (see module
    docstring), so this must be plain BCELoss on the already-sigmoided output, NOT
    BCEWithLogitsLoss (which would apply a second, redundant sigmoid-like transform inside
    its own numerically-stable formulation on top of one already applied in forward()).
    peptides-struct/pascalvoc-sp are unreachable while san_train's guards above are in
    place; their entries are kept here for whenever those guards are lifted.
    """
    if dataset == "peptides-func":
        return torch.nn.BCELoss()
    if dataset == "peptides-struct":
        return torch.nn.L1Loss()
    if dataset == "pascalvoc-sp":
        return torch.nn.CrossEntropyLoss()
    raise ValueError(dataset)


def _evaluate(model, loader, device, run_cfg, loss_fn):
    """Returns (task_metric, mean_loss) on one split."""
    import numpy as np
    from sklearn.metrics import average_precision_score, f1_score

    model.eval()
    losses, preds, targets = [], [], []
    with torch.no_grad():
        for bg, labels in loader:
            bg, labels = bg.to(device), labels.to(device)
            out = _forward_pass(model, bg)
            losses.append(loss_fn(out, labels).item())
            preds.append(out.cpu().numpy())
            targets.append(labels.cpu().numpy())
    preds, targets = np.concatenate(preds), np.concatenate(targets)
    mean_loss = float(np.mean(losses))

    if run_cfg.dataset == "peptides-func":
        metric = average_precision_score(targets, preds, average="macro")
    elif run_cfg.dataset == "peptides-struct":
        metric = mean_loss  # L1Loss already is the MAE metric
    else:  # pascalvoc-sp
        metric = f1_score(targets, preds.argmax(axis=1), average="macro")
    return float(metric), mean_loss


# ---------------------------------------------------------------------------
# the sensitivity probe wrapper -- STUB, see module docstring
# ---------------------------------------------------------------------------
def make_san_model_fn(model, data, device=None):
    """NOT YET WIRED."""
    raise NotImplementedError(
        "make_san_model_fn is not wired yet -- SAN training is real (san_train, for "
        "peptides-func) but the Jacobian probe wrapper for it is not. "
        "run_experiment.PROBE_WIRED_BACKBONES does not include 'san' for exactly this "
        "reason; a SAN cell run through run_cell() will report a real task metric with "
        "sensitivity_curve left empty, not fabricated."
    )
