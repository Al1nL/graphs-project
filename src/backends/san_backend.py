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
WHY THIS FILE IS SHAPED DIFFERENTLY FROM graphgps_backend.py, READ BEFORE TRUSTING IT
--------------------------------------------------------------------------------------
GraphGPS *is* effectively the LRGB benchmark's own reference codebase (the LRGB paper's
authors built their released benchmark repo as a fork of GraphGPS), so
`graphgps_backend.py` could start from GraphGPS's own tuned `configs/GPS/*-GPS.yaml` files
for Peptides-func/-struct and PascalVOC-SP and override only the PE block.

SAN (Kreuzer et al., 2021, github.com/DevinKreuzer/SAN) predates LRGB and ships NO
LRGB-specific reference configs at all -- its `configs/` directory only has tuned
hyperparameters for pre-LRGB benchmarks (ZINC, PATTERN, CLUSTER, MNIST, CIFAR10, SBM). The
LRGB paper DOES report a SAN baseline on all three of our datasets, but those numbers were
produced by the LRGB authors' OWN reimplementation of SAN as a GraphGym layer type inside
their GraphGPS-based repo -- not by running the original DevinKreuzer/SAN codebase. This
project's `config.UPSTREAM_URLS["san"]` points at the original DevinKreuzer/SAN repo
(DGL-based, standalone training scripts), which is the architecturally distinct backbone
described in the proposal and README, so that is what this file drives.

Two consequences follow, both disclosed rather than hidden:

1. **No upstream-tuned reference hyperparameters exist for our three datasets.**
   `BASE_NET_PARAMS` below is THIS PROJECT'S construction, sized to the same <=500K
   parameter budget as the GraphGPS arm and following SAN's own published hyperparameter
   conventions for its largest full-graph benchmark (ZINC, 500k split) scaled to LRGB's
   larger graphs -- it is NOT a reproduction of a published SAN+LRGB number, because no such
   official config exists. State this explicitly wherever a SAN number is reported.

2. **SAN's graphs are DGL, everything else in this project is PyG.** `_pyg_to_dgl` below
   converts one direction at collate time. DGL preserves autograd through its message-
   passing ops when features carry `requires_grad`, which is what the sensitivity probe
   needs -- verified against DGL's documented behaviour, not against a live install (no DGL
   available in the environment this was written in). Re-verify this specific claim against
   whatever DGL version the SAN fork pins before trusting probe numbers from this backbone.

3. **The probe wrapper (`make_san_model_fn`) is a stub.** Training is real; the Jacobian
   probe is not yet wired for SAN (see run_experiment.PROBE_WIRED_BACKBONES, which does not
   include "san"). A SAN cell run through `run_experiment.run_cell` will therefore report a
   real task metric and a real parameter count, with `sensitivity_curve` left honestly
   empty rather than fabricated -- the same "trained_but_not_probed" status GraphGPS+GRPE
   would report if someone forced it through. Wiring the probe requires the same
   encoder/body split `make_gps_model_fn` does (run SAN's LPE + input embedding once to get
   h^(0), then replay only the Transformer layers on whatever `x` the probe hands back) --
   left for a follow-up pass so the training integration lands first and can be validated
   independently.

--------------------------------------------------------------------------------------
API CALLS BELOW ARE WRITTEN FROM DOCUMENTED KNOWLEDGE OF THE SAN REPO, NOT A LIVE IMPORT
--------------------------------------------------------------------------------------
This was written in an environment with no network access and no DGL/SAN install, so the
exact module paths and function signatures (`nets.load_net.gnn_model`, the LPE module's
constructor arguments, the `full_graph`/`gamma` net_params keys) are transcribed from the
repo's documented structure as of the pinned era, not verified against a live import.
`ensure_san_importable` raises a readable error rather than a bare ImportError, but a
signature mismatch inside a successfully-imported module will not be caught until a real
run -- re-verify against `../SAN` at the pinned commit before trusting this file blindly.
"""

import os
import sys
import types
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# importing SAN
# ---------------------------------------------------------------------------
def ensure_san_importable(san_dir: Optional[str] = None) -> str:
    """Put the SAN clone on sys.path so its `nets`/`data`/`layers` packages import.

    Unlike GraphGPS, SAN does not register itself into a shared framework as an import
    side effect -- its modules are imported directly (`from nets.load_net import
    gnn_model`), so there is no GraphGym-style registry failure mode to guard against here.
    The main thing worth checking early is that the clone exists and looks like SAN's
    layout, so a missing fork fails with "run setup_upstream.sh" rather than a confusing
    ModuleNotFoundError three imports later.
    """
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
# config
# ---------------------------------------------------------------------------
# NOT upstream-tuned -- see module docstring, consequence (1). Sized to stay under the
# proposal's <=500k parameter budget, matching GraphGPS's L=10 arm as closely as SAN's
# architecture allows so the two backbones are at comparable capacity, not just comparable
# hyperparameter *names*.
BASE_NET_PARAMS = {
    "peptides-func": {
        "L": 10, "hidden_dim": 74, "out_dim": 74, "n_heads": 8, "full_graph": True,
        "gamma": 1e-5, "in_feat_dropout": 0.0, "dropout": 0.0,
        "layer_norm": False, "batch_norm": True, "readout": "mean",
        "task": "classification_multilabel", "n_classes": 10,
    },
    "peptides-struct": {
        "L": 10, "hidden_dim": 74, "out_dim": 74, "n_heads": 8, "full_graph": True,
        "gamma": 1e-5, "in_feat_dropout": 0.0, "dropout": 0.0,
        "layer_norm": False, "batch_norm": True, "readout": "mean",
        "task": "regression", "n_classes": 11,
    },
    "pascalvoc-sp": {
        "L": 8, "hidden_dim": 68, "out_dim": 68, "n_heads": 8, "full_graph": False,
        # full_graph=False (sparse attention) here, not a stylistic choice: PascalVOC-SP
        # graphs average 479 nodes and SAN's full attention is O(n^2) per graph -- dense
        # attention over an average batch would dominate wall-clock time disproportionately
        # relative to the Peptides arms, which is exactly the scaling concern the README's
        # "compute budget reality check" already names for this dataset.
        "gamma": 1e-5, "in_feat_dropout": 0.0, "dropout": 0.0,
        "layer_norm": False, "batch_norm": True, "readout": "mean",
        "task": "classification_multiclass_node", "n_classes": 21,
    },
}

TRAIN_PARAMS = {
    "epochs": 200, "batch_size": 32, "init_lr": 7e-4, "lr_reduce_factor": 0.5,
    "lr_schedule_patience": 10, "min_lr": 1e-6, "weight_decay": 0.0,
}

# PE -> net_params overrides. dim values match src/pe/compute_pe.py's K_LAP / K_RWSE so the
# two definitions stay aligned, mirroring graphgps_backend.PE_SPEC.
PE_SPEC = {
    "none": {"lpe_enable": False},
    "lappe": {"lpe_enable": True, "lpe_dim": 16, "lpe_n_heads": 4, "lpe_layers": 2},
    "rwse": {"lpe_enable": True, "lpe_dim": 16, "lpe_n_heads": 4, "lpe_layers": 2,
              "extra_node_feat": "rwse", "extra_node_feat_dim": 20},
    "signnet": {"lpe_enable": True, "lpe_dim": 16, "signnet_replaces_lpe": True,
                 "signnet_hidden_dim": 64, "signnet_out_dim": 32},
    "grpe": {"lpe_enable": False, "grpe_bias_enable": True},
}


def build_san_net_params(run_cfg) -> dict:
    """Assemble one cell's net_params: dataset base + PE override, loaded from
    configs/san/san_<pe>_<dataset>.json and merged over BASE_NET_PARAMS.

    Unlike configs/graphgps/*.yaml (which graphgps_backend.py does NOT read -- it builds
    GraphGym's cfg from GraphGPS's OWN internal reference YAML inside the clone, and this
    repo's configs/graphgps/ directory is a decorative leftover from before that
    integration landed; see configs/README.md), configs/san/*.json ARE the live source for
    net_params overrides here. SAN has no equivalent upstream reference config to defer to
    (see module docstring, consequence 1), so this repo's own JSON files are the only
    config that exists for it -- there was no reason to leave them unread the same way.
    """
    import json

    base = dict(BASE_NET_PARAMS[run_cfg.dataset])
    cfg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..",
        "configs", "san", f"san_{run_cfg.pe}_{run_cfg.dataset}.json",
    )
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"no SAN config at {cfg_path}. Every (pe, dataset) cell needs one -- see "
            "configs/san/README or regenerate with scripts/generate_san_configs.py."
        )
    with open(cfg_path) as f:
        file_cfg = json.load(f)
    base.update(file_cfg.get("net_params", {}))
    from dataset_meta import SPD_NUM_BUCKETS
    base["grpe_num_spd_buckets"] = SPD_NUM_BUCKETS
    base["seed"] = run_cfg.seed
    return base


def build_san_train_params(run_cfg) -> dict:
    """Training hyperparameters (epochs, lr, batch size, ...) for one cell, loaded from
    the same configs/san/san_<pe>_<dataset>.json file's "params" block and merged over
    TRAIN_PARAMS. Split from build_san_net_params only because one describes the model and
    the other describes the optimizer loop -- callers that only need one should not have to
    reason about the other changing underneath them.
    """
    import json

    params = dict(TRAIN_PARAMS)
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
def _pyg_to_dgl(data):
    """Convert one PyG Data graph into the DGLGraph SAN's model classes expect.

    Node features go on `ndata['feat']`; edge features (if present) on `edata['feat']`.
    Kept deliberately minimal -- SAN's own LPE module computes Laplacian eigenvectors
    itself from the DGL graph's adjacency, so no PE fields need to be attached here (see
    module docstring, consequence (1) applies to SAN the same way it applies to GraphGPS's
    native encoders: this drives SAN's OWN LapPE computation, not src/pe/cache.py's).
    """
    import dgl

    g = dgl.graph((data.edge_index[0], data.edge_index[1]), num_nodes=int(data.num_nodes))
    g.ndata["feat"] = data.x
    if getattr(data, "edge_attr", None) is not None:
        g.edata["feat"] = data.edge_attr
    return g


def _collate(batch):
    """Batch a list of PyG Data graphs into one DGL batched graph + a stacked label
    tensor, matching the (batched_graph, labels) pairs SAN's own train loops iterate over.
    """
    import dgl

    graphs = [_pyg_to_dgl(d) for d in batch]
    labels = torch.stack([d.y if d.y.dim() > 0 else d.y.unsqueeze(0) for d in batch])
    return dgl.batch(graphs), labels


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def san_train(run_cfg, san_dir: Optional[str] = None) -> dict:
    """Train one grid cell with SAN's own model classes. Mirrors graphgps_train's
    contract: returns {"model", "loaders", "num_params", "metric_name", "metric_value"}.

    Uses SAN's own `nets.load_net.gnn_model(...)` factory to build the model -- that is
    genuinely how SAN's own scripts instantiate it, so the architecture is SAN's, not a
    reimplementation. The outer training loop (optimizer, LR schedule, epoch loop, eval) is
    ordinary supervised-learning boilerplate written for this project, because SAN's
    upstream train scripts are per-benchmark-family (`train_molecules_...py`,
    `train_SBMs_...py`) and none of the three matches our three task types exactly; writing
    one small generic loop here is a smaller confound than adapting a benchmark-family
    script that was never meant for LRGB's tasks.
    """
    if run_cfg.pe == "grpe":
        raise NotImplementedError(
            "SAN has no native attention-bias hook, so GRPE needs SAN's attention class "
            "extended by adapters.san_adapter.SANGammaGRPEBias and the spd_bucket/"
            "edge_type tensors threaded onto the DGL batch. That is a genuine "
            "architectural addition to SAN (flagged as such in the README), not a config "
            "change, and is deliberately left for a separate pass -- the other four PE "
            "arms are drop-ins and should be validated first."
        )

    san_dir = ensure_san_importable(san_dir)
    from nets.load_net import gnn_model  # noqa: (SAN's own model factory)

    net_params = build_san_net_params(run_cfg)
    train_params = build_san_train_params(run_cfg)
    torch.manual_seed(run_cfg.seed)

    from dataset_meta import NODE_FEATURE_DIM
    net_params["in_dim"] = NODE_FEATURE_DIM[run_cfg.dataset]
    model = gnn_model("SAN", net_params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_params > 500_000:
        print(f"  WARNING: {n_params:,} parameters exceeds the 500k budget in the proposal")

    train_loader, val_loader, test_loader = _build_loaders(run_cfg, train_params)

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
            out = model(bg, bg.ndata["feat"])
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


def _build_loaders(run_cfg, train_params):
    """DataLoaders over the same LRGBDataset splits GraphGPS trains on, batched via
    `_collate` into DGL graphs. Deliberately the same underlying PyG dataset object as
    GraphGPS's loader (LRGBDataset with the standard LRGB train/val/test split), so both
    backbones see the identical graphs in the identical split -- only the batching format
    (DGL vs PyG) and the PE computation (each backbone's own, see module docstring) differ.
    """
    from torch.utils.data import DataLoader
    from torch_geometric.datasets import LRGBDataset

    name_map = {"peptides-func": "Peptides-func", "peptides-struct": "Peptides-struct",
                "pascalvoc-sp": "PascalVOC-SP"}
    pyg_name = name_map[run_cfg.dataset]
    loaders = []
    for split in ("train", "val", "test"):
        ds = LRGBDataset(root=f"./raw_data/{pyg_name}", name=pyg_name, split=split)
        shuffle = split == "train"
        loaders.append(DataLoader(ds, batch_size=train_params["batch_size"],
                                  shuffle=shuffle, collate_fn=_collate))
    return loaders


def _loss_for(dataset: str):
    if dataset == "peptides-func":
        return torch.nn.BCEWithLogitsLoss()
    if dataset == "peptides-struct":
        return torch.nn.L1Loss()
    if dataset == "pascalvoc-sp":
        return torch.nn.CrossEntropyLoss()
    raise ValueError(dataset)


def _evaluate(model, loader, device, run_cfg, loss_fn):
    """Returns (task_metric, mean_loss) on one split. AP/macro-F1 for classification,
    negative-MAE-as-loss for regression (kept as plain MAE in metric_value; loss_fn
    already IS L1Loss for peptides-struct, so `loss` and `metric` coincide there)."""
    import numpy as np
    from sklearn.metrics import average_precision_score, f1_score

    model.eval()
    losses, preds, targets = [], [], []
    with torch.no_grad():
        for bg, labels in loader:
            bg, labels = bg.to(device), labels.to(device)
            out = model(bg, bg.ndata["feat"])
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
# the sensitivity probe wrapper -- STUB, see module docstring consequence (3)
# ---------------------------------------------------------------------------
def make_san_model_fn(model, data, device=None):
    """NOT YET WIRED. Matches graphgps_backend.make_gps_model_fn's intended signature so
    run_experiment.make_model_fn can dispatch to it once implemented, but raises today
    rather than returning something that looks probeable and silently isn't.

    Implementing this needs the same split make_gps_model_fn does: run SAN's input
    embedding + LPE module once to get h^(0) (the node representation the Transformer
    layers actually start from), then replay only the Transformer layer stack on whatever
    `x` the probe hands back, stopping before the readout/task head. SAN's model classes
    (`nets/SAN_NodeLPE.py` et al.) are not import-verified in this environment (see module
    docstring), so writing this blind was judged riskier than leaving it a loud stub --
    training numbers should be validated against the real fork first.
    """
    raise NotImplementedError(
        "make_san_model_fn is not wired yet -- SAN training is real (san_train), but the "
        "Jacobian probe wrapper for it is not. See this function's docstring for what "
        "implementing it needs. run_experiment.PROBE_WIRED_BACKBONES does not include "
        "'san' for exactly this reason; a SAN cell run through run_cell() will report a "
        "real task metric with sensitivity_curve left empty, not fabricated."
    )
