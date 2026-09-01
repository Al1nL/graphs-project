"""
graphgps_backend.py
===================
Real integration with the GraphGPS fork at ../GraphGPS (config.UPSTREAM_PATHS["gps"]).

Two entry points, replacing the stubs in run_experiment.py:

    graphgps_train(run_cfg)          -> trains a cell of the grid, returns model + metrics
    make_gps_model_fn(model, data)   -> wraps a trained model for the Jacobian probe

Both drive GraphGPS's own code rather than reimplementing it: the config is GraphGPS's
tuned reference YAML for the dataset with only the PE block overridden, and training is
GraphGPS's registered `custom` train loop. The point of the study is to vary the PE inside
a fixed backbone, so anything we reimplement is a confound.

--------------------------------------------------------------------------------------
WHERE THE JACOBIAN IS TAKEN, AND WHY IT CANNOT BE THE RAW FEATURES
--------------------------------------------------------------------------------------
LRGB node features are INTEGER atom-type indices consumed by an nn.Embedding lookup
(AtomEncoder). d h / d x is undefined for a discrete index, so the probe cannot
differentiate the raw features -- there is no derivative to take.

The probe therefore differentiates with respect to h^(0), the node representation *after*
the feature encoder. This is the standard reading of Di Giovanni et al.: h^(0) is the
initial node representation the network actually starts from.

--------------------------------------------------------------------------------------
AN INCOMPARABILITY THIS EXPOSES -- READ BEFORE TRUSTING CROSS-PE NUMBERS
--------------------------------------------------------------------------------------
GraphGPS keeps `dim_inner` constant and makes room for the PE by SHRINKING the atom
encoder (graphgps/encoder/composed_encoders.py):

    self.encoder1 = AtomEncoder(dim_emb - dim_pe)      # content channels
    self.encoder2 = PEEncoder(dim_emb, expand_x=False) # concatenated after
    ...
    batch.x = torch.cat((h, pos_enc), 1)               # content FIRST, then PE

So with dim_inner = 96 the CONTENT width differs per variant:

    No-PE   96      GRPE     96   (attention bias, adds no channels)
    LapPE   80      RWSE     96 - dim_pe      SignNet  96 - dim_pe

This is fix 1's problem in its real form, and worse than anticipated: it is not that some
PEs add channels, it is that they SUBTRACT content channels. Two consequences:

  * Slicing to the content channels (`n_content_feats`) compares ||J||_F over a DIFFERENT
    number of columns per variant -- 80 terms vs 96 -- so the raw Frobenius norms are not
    comparable. `sensitivity.assert_shared_width` will refuse them, correctly.
  * Using the full width (`dim_inner`, identical for every variant) removes the
    dimensional problem entirely, at the cost of including PE channels in the perturbation.

`make_gps_model_fn` returns both widths and takes no side. See the docstring of
`probe_widths` for the recommendation and the open question it leaves.
"""

import copy
import os
import sys
import types
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# importing GraphGPS
# ---------------------------------------------------------------------------
def ensure_graphgps_importable(graphgps_dir: Optional[str] = None) -> str:
    """Put the GraphGPS clone on sys.path and import it so its modules register.

    GraphGPS registers its networks, encoders, heads and train loops into GraphGym's
    global registries as an import side effect (`import graphgps  # noqa`). Nothing works
    until that happens, and the failure mode is a confusing KeyError from a registry
    lookup rather than an ImportError, so this raises something readable instead.
    """
    if graphgps_dir is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
        from config import UPSTREAM_PATHS
        graphgps_dir = UPSTREAM_PATHS["gps"]
    graphgps_dir = os.path.abspath(graphgps_dir)
    if not os.path.isdir(os.path.join(graphgps_dir, "graphgps")):
        raise FileNotFoundError(
            f"no GraphGPS clone at {graphgps_dir}. Run `bash scripts/setup_upstream.sh gps`"
        )
    if graphgps_dir not in sys.path:
        sys.path.insert(0, graphgps_dir)
    try:
        import graphgps  # noqa: F401  -- registers custom modules into GraphGym
    except ImportError as exc:
        raise ImportError(
            f"GraphGPS at {graphgps_dir} could not be imported ({exc}). It needs its OWN "
            "environment -- yacs and pytorch_lightning for GraphGym, plus GraphGPS's "
            "pinned PyG. See README 'Environment setup'; do not share one env across the "
            "three backbones."
        ) from exc
    return graphgps_dir


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
# GraphGPS ships tuned reference configs; we start from those and override ONLY the PE, so
# every arm of the grid differs in exactly one thing. Re-tuning per PE would confound the
# comparison with hyperparameter search.
BASE_CONFIG = {
    "peptides-func": "configs/GPS/peptides-func-GPS.yaml",
    "peptides-struct": "configs/GPS/peptides-struct-GPS.yaml",
    "pascalvoc-sp": "configs/GPS/vocsuperpixels-GPS.yaml",
}

# The dataset-specific (non-PE) node encoder each dataset uses on its own.
DATASET_NODE_ENCODER = {
    "peptides-func": "Atom",
    "peptides-struct": "Atom",
    "pascalvoc-sp": "VOCNode",
}

# Widths of the cache these encoders are fed from -- src/pe/compute_pe.py's K_LAP/K_RWSE.
# Duplicated here rather than imported so that build_graphgym_cfg stays importable without
# torch_geometric.utils (the launcher's --dry-run and most of the test suite depend on
# that). graphgps_pe_cache.install cross-checks both against the built cache's manifest
# before a run starts, so a drift here fails loudly rather than silently.
K_LAP = 16
K_RWSE = 20

# PE -> (GraphGym encoder suffix, posenc config key, dim_pe). dim_pe is the encoder's
# OUTPUT width; see build_graphgym_cfg for why that is not the same as max_freqs.
PE_SPEC = {
    "none": (None, None, 0),
    "lappe": ("LapPE", "posenc_LapPE", 16),
    "rwse": ("RWSE", "posenc_RWSE", 20),
    "signnet": ("SignNet", "posenc_SignNet", 32),
    "grpe": (None, None, 0),   # attention bias -- no node-feature channels
}


def build_graphgym_cfg(run_cfg, graphgps_dir: str):
    """Populate GraphGym's global cfg for one grid cell.

    GraphGym's `cfg` is a process-global singleton, so this mutates shared state and two
    cells cannot be configured concurrently in one process. The launcher runs cells
    sequentially, which is why that is acceptable here.
    """
    from torch_geometric.graphgym.config import cfg, set_cfg

    set_cfg(cfg)
    base = os.path.join(graphgps_dir, BASE_CONFIG[run_cfg.dataset])
    if not os.path.exists(base):
        raise FileNotFoundError(
            f"GraphGPS reference config not found: {base}. The upstream layout may have "
            f"changed under the pin -- check config.PINNED_COMMITS['gps']."
        )
    cfg.merge_from_file(base)

    enc_suffix, posenc_key, dim_pe = PE_SPEC[run_cfg.pe]

    # Disable every PE block the base config may have switched on, so the only PE active
    # is the one this cell is testing.
    for key in ("posenc_LapPE", "posenc_RWSE", "posenc_SignNet", "posenc_HKdiagSE",
                "posenc_ElstaticSE", "posenc_EquivStableLapPE"):
        if hasattr(cfg, key):
            getattr(cfg, key).enable = False

    node_enc = DATASET_NODE_ENCODER[run_cfg.dataset]
    if enc_suffix is not None:
        node_enc = f"{node_enc}+{enc_suffix}"
        block = getattr(cfg, posenc_key)
        block.enable = True
        block.dim_pe = dim_pe
        if posenc_key in ("posenc_LapPE", "posenc_SignNet"):
            # max_freqs is how many eigenvectors come IN; dim_pe is how many channels go
            # OUT. They coincide at 16 for LapPE and differ for SignNet (16 in, 32 out),
            # so this must be K_LAP and not dim_pe -- using dim_pe would have asked
            # SignNet's encoder for 32 eigenvectors the cache does not have.
            block.eigen.max_freqs = K_LAP
        if posenc_key == "posenc_RWSE":
            block.kernel.times_func = f"range(1,{K_RWSE + 1})"
    cfg.dataset.node_encoder_name = node_enc

    cfg.seed = run_cfg.seed
    cfg.out_dir = os.path.join(run_cfg.results_dir, "raw", run_cfg.run_id)
    if run_cfg.epochs is not None:
        cfg.optim.max_epoch = run_cfg.epochs
    cfg.train.mode = "custom"          # GraphGPS's own loop; 'standard' needs lightning
    cfg.wandb.use = False              # the launcher owns logging

    # --- surviving a pre-emptible partition with a wall shorter than a training run ------
    # GraphGPS's custom_train already supports resume (`start_epoch = load_ckpt(...)` then
    # `for cur_epoch in range(start_epoch, max_epoch)`), but GraphGym defaults auto_resume
    # to False, so nothing here used it. On studentkillable -- 24 h hard cap, pre-emptible,
    # and the only partition a gpu-students association grants -- that combination is
    # unworkable: run_grid.slurm sets --requeue, but a requeued cell without resume simply
    # restarts from epoch 0 and can be pre-empted again, so a cell that needs longer than
    # the wall never finishes no matter how many times it is requeued.
    cfg.train.auto_resume = True
    cfg.train.enable_ckpt = True
    # The reference YAMLs set ckpt_period 100, i.e. two checkpoints across a 200-epoch run,
    # which is fine for archiving and useless for resume -- an interruption at epoch 99
    # would lose 99 epochs. 10 caps the loss at 10 epochs for a model small enough that
    # 20-30 checkpoints is tens of MB, not a quota problem.
    cfg.train.ckpt_period = 10

    # NOTE the consequence for re-runs: with auto_resume on, re-running a cell whose
    # checkpoint has reached max_epoch does NOT retrain -- custom_train logs "Checkpoint
    # found, Task already done" and skips to evaluation. That is what makes requeue work,
    # but it also means a deliberate retrain needs cfg.out_dir cleared first.
    return cfg


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def graphgps_train(run_cfg, graphgps_dir: Optional[str] = None) -> dict:
    """Train one grid cell with GraphGPS's own pipeline. Mirrors GraphGPS main.py.

    Returns {"model", "loaders", "num_params", "metric_value", "metric_name", "cfg"}.
    """
    if run_cfg.pe == "grpe":
        raise NotImplementedError(
            "GraphGPS has no native attention-bias hook, so GRPE needs the GPSLayer "
            "self-attention replaced by adapters.graphgps_adapter.GRPEBiasedAttention and "
            "the spd_bucket/edge_type tensors threaded onto the batch. That is a genuine "
            "architectural addition to GraphGPS (flagged as such in the README), not a "
            "config change, and is deliberately left for a separate pass -- the other four "
            "PE arms are drop-ins and should be validated first."
        )

    graphgps_dir = ensure_graphgps_importable(graphgps_dir)
    from torch_geometric import seed_everything
    from torch_geometric.graphgym.loader import create_loader
    from torch_geometric.graphgym.model_builder import create_model
    from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
    from torch_geometric.graphgym.register import train_dict
    from torch_geometric.graphgym.utils.comp_budget import params_count
    from torch_geometric.graphgym.utils.device import auto_select_device
    from graphgps.logger import create_logger
    from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig
    from torch_geometric.graphgym.optim import OptimizerConfig

    cfg = build_graphgym_cfg(run_cfg, graphgps_dir)
    seed_everything(cfg.seed)
    auto_select_device()

    # Point GraphGPS's loader at THIS repo's PE cache before it builds anything. Without
    # this the GPS arm trains on GraphGPS's own LapPE/RWSE/SignNet while the SAN arm
    # trains on the cache's, and the two are not the same encoding -- see
    # graphgps_pe_cache's header for the four definitions that differ. Must happen after
    # build_graphgym_cfg (which sets max_freqs to the cache's width) and before
    # create_loader (which runs the pre-transform being replaced).
    from backends.graphgps_pe_cache import install as install_pe_cache
    pe_cache = install_pe_cache(run_cfg, cfg) if run_cfg.pe != "none" else None

    loaders = create_loader()

    if pe_cache is not None and pe_cache.calls == 0:
        raise RuntimeError(
            f"PE cache patch was installed for pe={run_cfg.pe!r} but GraphGPS never "
            "called it, so the model is about to train on whatever encoding its own "
            "pipeline produced. The upstream loader no longer routes through "
            "master_loader.compute_posenc_stats under the pin -- fix the patch target in "
            "backends/graphgps_pe_cache.install before trusting this run.")
    loggers = create_logger()
    model = create_model()
    optimizer = create_optimizer(
        model.parameters(),
        OptimizerConfig(optimizer=cfg.optim.optimizer, base_lr=cfg.optim.base_lr,
                        weight_decay=cfg.optim.weight_decay, momentum=cfg.optim.momentum))
    scheduler = create_scheduler(optimizer, ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler, steps=cfg.optim.steps,
        lr_decay=cfg.optim.lr_decay, max_epoch=cfg.optim.max_epoch,
        reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode, eval_period=cfg.train.eval_period))

    n_params = params_count(model)
    if n_params > 500_000:
        # the proposal commits to a <=500k budget so the arms are comparable
        print(f"  WARNING: {n_params:,} parameters exceeds the 500k budget in the proposal")

    train_dict[cfg.train.mode](loggers, loaders, model, optimizer, scheduler)

    return {
        "model": model,
        "loaders": loaders,
        "num_params": n_params,
        "metric_name": run_cfg.metric_name,
        "metric_value": _read_best_metric(cfg, run_cfg.metric_name),
        "cfg": cfg,
    }


def _read_best_metric(cfg, metric_name) -> Optional[float]:
    """Best test-split value, read back from the stats GraphGPS's logger wrote.

    Deliberately parsed from disk rather than scraped out of the logger objects: the loop
    owns those, their internals move between versions, and the file is the artefact
    GraphGPS itself treats as the result.
    """
    import json

    path = os.path.join(cfg.out_dir, str(cfg.seed), "test", "stats.json")
    if not os.path.exists(path):
        return None
    best = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if metric_name not in rec:
                continue
            v = rec[metric_name]
            lower_better = metric_name in ("mae", "loss")
            if best is None or (v < best if lower_better else v > best):
                best = v
    return best


# ---------------------------------------------------------------------------
# the sensitivity probe wrapper
# ---------------------------------------------------------------------------
def probe_widths(model) -> dict:
    """Input widths available to the Jacobian probe, and what each one means.

    Returns {"dim_inner", "dim_pe", "n_content_feats"}.

    `n_content_feats` slices h^(0) to the atom-content channels (they come first; the PE
    encoder concatenates after). Semantically this is the right perturbation -- a PE
    channel describes graph STRUCTURE, not node u's CONTENT -- but its width differs per
    PE variant, because GraphGPS shrinks the atom encoder to make room for the PE. Raw
    Frobenius norms over different column counts are not comparable, and
    `sensitivity.assert_shared_width` will refuse them.

    `dim_inner` is identical across all five variants, so it removes the dimensional
    problem entirely, at the cost of perturbing PE channels alongside content.

    RECOMMENDATION: use `dim_inner`, and report per-channel normalised sensitivity
    (||J||_F / sqrt(q)) if the content-only view is also wanted. Neither choice is free,
    and this is a genuine limitation of comparing PEs inside a fixed-width backbone rather
    than a defect in the probe -- it should be stated in the paper, not hidden. The
    decision is deliberately NOT taken here.
    """
    from torch_geometric.graphgym.config import cfg

    dim_pe = 0
    for key in ("posenc_LapPE", "posenc_RWSE", "posenc_SignNet"):
        block = getattr(cfg, key, None)
        if block is not None and getattr(block, "enable", False):
            dim_pe += block.dim_pe
    return {"dim_inner": cfg.gnn.dim_inner, "dim_pe": dim_pe,
            "n_content_feats": cfg.gnn.dim_inner - dim_pe}


def make_gps_model_fn(model, data, device=None):
    """Wrap a trained GPSModel for `sensitivity.compute_sensitivity_curve`.

    Returns (model_fn, probe_data, meta) where:
      model_fn(x)  runs the layer stack on node representations `x`, returning final-layer
                   NODE embeddings [n, dim_hidden] -- not pooled, not task logits.
      probe_data   a stand-in with `.x` = h^(0) (the encoder output, detached),
                   `.edge_index` and `.num_nodes`, ready to hand to the probe.
      meta         probe_widths() plus the graph's node count.

    GPSModel.forward is just `for module in self.children(): batch = module(batch)`, with
    children in registration order: encoder, [pre_mp], layers, post_mp. So this runs the
    encoder once up front to obtain h^(0), and then replays everything between the encoder
    and post_mp on whatever `x` the probe hands back. Stopping before post_mp is what
    yields node embeddings rather than pooled graph logits.
    """
    model.eval()   # BatchNorm must use running stats: training-mode batch statistics would
                   # make the Jacobian depend on the rest of the batch, not just node u
    device = device or next(model.parameters()).device
    batch = data.clone().to(device)

    children = list(model.named_children())
    if not children or children[0][0] != "encoder":
        raise RuntimeError(
            f"unexpected GPSModel layout {[n for n, _ in children]}; expected 'encoder' "
            "first. The upstream model may have changed under the pin.")

    with torch.no_grad():
        encoded = model.encoder(batch)
    h0 = encoded.x.detach().clone()

    middle = [(n, m) for n, m in children if n not in ("encoder", "post_mp")]

    def model_fn(x):
        b = encoded.clone()
        b.x = x
        for _, module in middle:
            b = module(b)
        return b.x

    probe_data = types.SimpleNamespace(
        x=h0, edge_index=data.edge_index.to(device), num_nodes=int(data.num_nodes))
    meta = {**probe_widths(model), "num_nodes": int(data.num_nodes)}
    return model_fn, probe_data, meta
