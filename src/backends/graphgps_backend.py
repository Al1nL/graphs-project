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

# --- parameter budget ------------------------------------------------------------------
# The proposal commits to a ~500k budget so the arms stay comparable. Taken as a literal
# 500,000 it fires on a FAITHFUL reproduction: GraphGPS's own LRGB configs are 504,362
# (peptides-func), 504,459 (peptides-struct) and 510,453 (PascalVOC-SP) -- Table A.5 of
# arXiv:2205.12454v3. A warning that trips on the reference recipe is one you learn to
# scroll past, which costs you the case it exists for.
#
# The threshold sits just above the largest reference config, so the three arms that are
# genuinely in family stay quiet while a real outlier still speaks up: signnet comes in at
# 576,138 here (dim_pe 32 plus an 8-layer phi), ~14% above the others, which is worth
# seeing every time. The count is now always printed, so the budget is checkable even when
# nothing warns.
PARAM_BUDGET_REFERENCE_MAX = 510_453   # PascalVOC-SP, the largest in Table A.5
PARAM_BUDGET_WARN = 520_000

# --- this project's metric names -> the keys GraphGPS's logger actually writes -----------
# They agree for two of the three datasets and did not for the third. GraphGPS logs
# macro-F1 under the key 'f1' (logger.py: f1_score(..., average='macro')), while
# config.TASK_METRIC calls it 'macro_f1'. The QUANTITY is the same -- it really is
# macro-averaged -- only the name differs, so this is a translation, not a change of
# metric.
#
# Left-hand side untouched on purpose: TASK_METRIC lives in config.py, which the SAN arm
# also reads, and san_backend keys its higher_is_better test off exactly these strings
# ("ap", "macro_f1"). Renaming there would silently flip macro-F1 to lower-is-better for
# SAN and select its WORST epoch. The mismatch is GraphGPS-specific, so the translation
# belongs here.
GRAPHGPS_METRIC_KEY = {
    "ap": "ap",              # peptides-func    -- logger writes 'ap'
    "mae": "mae",            # peptides-struct  -- logger writes 'mae'
    "macro_f1": "f1",        # pascalvoc-sp     -- logger writes 'f1', macro-averaged
}

# PE -> (GraphGym encoder suffix, posenc config key, dim_pe). dim_pe is the encoder's
# OUTPUT width; see build_graphgym_cfg for why that is not the same as max_freqs.
PE_SPEC = {
    "none": (None, None, 0),
    "lappe": ("LapPE", "posenc_LapPE", 16),
    "rwse": ("RWSE", "posenc_RWSE", 20),
    "signnet": ("SignNet", "posenc_SignNet", 32),
    "grpe": (None, None, 0),   # attention bias -- no node-feature channels
}

# The PE encoder HEAD: the small network that maps raw PE values to the dim_pe channels
# concatenated onto the node features. Distinct from the PE VALUES, which now come from
# this repo's shared cache (see backends/graphgps_pe_cache) -- this is the learned part,
# and it is a property of the PE arm, not of the dataset.
#
# These MUST be set explicitly. All three of our base configs (peptides-func, -struct,
# vocsuperpixels) enable only posenc_LapPE, so every other block stays at the bare
# defaults from GraphGPS's posenc_config.py -- where `model` is the literal string
# 'none' and `post_layers` is 0. That is not a working configuration for any encoder:
# RWSE raises "Does not support 'none' encoder model", and SignNet raises "Num layers in
# rho model has to be positive". Only the lappe arm happened to work, purely because the
# base YAML it inherited from configures LapPE for its own use.
#
# Values follow GraphGPS's own reference configs for each PE (Linear+BatchNorm for RWSE,
# consistent across every *-GPS+RWSE.yaml; DeepSet for LapPE as in the peptides configs;
# the SNDS DeepSet variant for SignNet, which is the one the GPS paper reports). Pinning
# them here rather than per dataset is deliberate: the grid varies PE and dataset
# independently, so an encoder head that changed with the dataset would confound the
# comparison this project exists to make.
PE_ENCODER = {
    "lappe": {
        "model": "DeepSet",
        "layers": 2,          # layers in the DeepSet phi
        "post_layers": 0,     # LapPE allows 0 here; SignNet does not
        "raw_norm_type": "none",
    },
    "rwse": {
        "model": "Linear",    # a single nn.Linear(num_rw_steps -> dim_pe)
        "layers": 3,          # unused while model is Linear; set so it is not a surprise
        "post_layers": 0,
        "raw_norm_type": "BatchNorm",   # RWSE landing probabilities need it; they are
                                        # raw probabilities with very different scales
                                        # across walk lengths
    },
    "signnet": {
        "model": "DeepSet",   # the SNDS variant
        "layers": 8,          # layers in phi
        "post_layers": 3,     # layers in rho -- MUST be >= 1 or SignNet refuses
        "raw_norm_type": "none",
        "phi_hidden_dim": 64,
        "phi_out_dim": 64,
    },
}


def memory_safe_batch(pe: str, batch_size: int):
    """(physical batch, accumulation steps) for one PE arm, preserving the EFFECTIVE batch.

    All three signnet cells on peptides-func died at epoch 0 with

        torch.cuda.OutOfMemoryError: Tried to allocate 342.00 MiB
        (GPU 0; 10.57 GiB total capacity; 9.56 GiB already allocated)

    inside GPS's attention softmax, on an 11 GB 2080 Ti. The other three arms trained
    fine on identical data, so this is SignNet specifically: its phi network runs an
    8-layer GIN over each of the 16 eigenvector channels separately, holding roughly
    [K, N, phi_out_dim] activations for the whole batch. The 14% parameter gap
    (576,138 vs ~504,000) badly understates the ACTIVATION gap, and the attention
    allocation is simply what happens to ask for memory once SignNet has taken it.

    Halving the physical batch and accumulating twice is the fix that does NOT change
    what is being compared: cfg.optim.batch_accumulation makes custom_train step the
    optimizer every 2 iterations (custom_train.py:34), so the gradient each step is
    computed over the same 128 graphs as the other arms. A plain batch-size cut would
    have changed the optimization trajectory and confounded the very comparison this
    study exists to make.

    NOT identical, and this has to be disclosed rather than glossed: BatchNorm statistics
    are computed over the physical batch, so signnet's normalisation sees 64 graphs where
    the other arms see 128. That is a second-order difference on top of one already being
    disclosed for this arm -- it is not parameter-matched either -- and the alternative
    was no signnet arm at all.

    Applied by ratio rather than to a fixed number so it carries to VOC, whose reference
    batch is 32 rather than 128.
    """
    if pe != "signnet" or batch_size < 2:
        return batch_size, 1
    return batch_size // 2, 2


def build_graphgym_cfg(run_cfg, graphgps_dir: str):
    """Populate GraphGym's global cfg for one grid cell.

    GraphGym's `cfg` is a process-global singleton, so this mutates shared state and two
    cells cannot be configured concurrently in one process. The launcher runs cells
    sequentially, which is why that is acceptable here.
    """
    from torch_geometric.graphgym.config import assert_cfg, cfg, set_cfg

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
            # Set BOTH. master_loader derives `times` from `times_func` during
            # create_loader(), so times_func is what ultimately governs -- but anything
            # inspecting the config before then (graphgps_pe_cache.install's width check)
            # sees an empty `times` unless it is filled in here too. Same value either way.
            block.kernel.times_func = f"range(1,{K_RWSE + 1})"
            block.kernel.times = list(range(1, K_RWSE + 1))

        # The encoder head. hasattr-checked rather than set blindly so that a field
        # renamed upstream fails here, naming the field, instead of silently leaving the
        # default in place -- which for `model` means a crash deep in the encoder and for
        # `raw_norm_type` means quietly training without the normalisation.
        for field, value in PE_ENCODER[run_cfg.pe].items():
            if not hasattr(block, field):
                raise AttributeError(
                    f"{posenc_key} has no field {field!r}; GraphGPS's posenc_config.py "
                    f"may have changed under config.PINNED_COMMITS['gps']")
            setattr(block, field, value)
    cfg.dataset.node_encoder_name = node_enc

    cfg.seed = run_cfg.seed
    cfg.out_dir = os.path.join(run_cfg.results_dir, "raw", run_cfg.run_id)
    if run_cfg.epochs is not None:
        cfg.optim.max_epoch = run_cfg.epochs
    cfg.train.mode = "custom"          # GraphGPS's own loop; 'standard' needs lightning

    # SignNet needs a smaller physical batch to fit an 11 GB card; accumulation keeps the
    # effective batch equal to the other arms. See memory_safe_batch for what this does
    # and does not preserve.
    physical, accumulation = memory_safe_batch(run_cfg.pe, cfg.train.batch_size)
    if accumulation > 1:
        print(f"  batch: {cfg.train.batch_size} -> {physical} x {accumulation} "
              f"accumulation steps (same effective batch; SignNet's per-eigenvector phi "
              f"does not fit otherwise)")
        cfg.train.batch_size = physical
        cfg.optim.batch_accumulation = accumulation
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

    # --- smoke test: one epoch, its own directory, no resume -----------------------------
    # Placed after the checkpoint settings above so it overrides them, and after the
    # --epochs override so it wins over that too (matching san_backend).
    if getattr(run_cfg, "smoke_test", False):
        cfg.optim.max_epoch = 1
        # The reference schedule is cosine_with_warmup over 5-10 warmup epochs. At
        # max_epoch 1 the warmup never completes, so the LR stays at 0 for the whole run
        # and the loss cannot move -- a flat curve for a reason that has nothing to do
        # with the model. Shape-checking needs no schedule; train the one epoch at base LR.
        cfg.optim.num_warmup_epochs = 0

        # auto_resume OFF. custom_train does `for cur_epoch in range(start_epoch,
        # max_epoch)`, so ANY existing checkpoint makes range(start, 1) empty and the
        # smoke test silently trains nothing at all -- it logs "Task done", reports
        # "Avg time per epoch: nan", and hands the probe a model it never touched. A
        # smoke test that skips the thing it is testing is worse than no smoke test.
        cfg.train.auto_resume = False
        # enable_ckpt OFF, and a directory of its own. Both protect the REAL run: with
        # ckpt_period 10 the last epoch is always checkpointed, so a smoke test would
        # otherwise leave an epoch-0 checkpoint in the cell's own run_dir, and the next
        # real run -- which does resume -- would silently continue from a model trained
        # on two batches instead of starting clean.
        cfg.train.enable_ckpt = False
        cfg.out_dir += "_smoke"

    # NOTE the consequence for re-runs: with auto_resume on, re-running a cell whose
    # checkpoint has reached max_epoch does NOT retrain -- custom_train logs "Checkpoint
    # found, Task already done" and skips to evaluation. That is what makes requeue work,
    # but it also means a deliberate retrain needs cfg.out_dir cleared first.

    # --- the post-processing main.py gets for free from load_cfg --------------------------
    # main.py reads its config through load_cfg(), which is merge_from_file +
    # merge_from_list + assert_cfg. We replicate the merge and, until this call, skipped
    # the rest. Despite the name assert_cfg does not only assert -- it REWRITES values,
    # and one of those rewrites is load-bearing:
    #
    #   gnn.head 'default' -> cfg.dataset.task
    #       'default' is a SENTINEL, not a registered head. Nothing registers it, in PyG
    #       or in GraphGPS, so leaving it in place fails at model construction with
    #       `KeyError: 'default'` from gps_model's register.head_dict lookup. 22 of
    #       GraphGPS's own configs use it, including all three of ours.
    #   loss_fun  coerced to cross_entropy for classification / mse for regression
    #   layers_post_mp  raised to >= 1
    #   dataset.transductive  forced False for graph-level tasks
    #
    # Called last so it sees the PE and encoder edits above, mirroring load_cfg's order
    # (assert_cfg runs after the command-line overrides, which is what those edits are the
    # analogue of). It also sets cfg.run_dir = cfg.out_dir; graphgps_train overwrites that
    # with the per-seed directory immediately afterwards, which is what main.py does too.
    #
    # Worth knowing that the sentinel would be WRONG for pascalvoc-sp: its config declares
    # `task: graph` even though VOC is node-level, and pins `head: inductive_node`
    # explicitly. So this rewrite is correct for the two peptides datasets precisely
    # because VOC opts out of it.
    assert_cfg(cfg)
    return cfg


def assert_gpu_if_slurm_allocated_one() -> None:
    """Fail loudly when a GPU was allocated but torch cannot see it.

    Nodes on a shared cluster turn up with broken CUDA -- a stale context from a previous
    job, a driver that needs a reset -- and torch reports it as a warning, not an error:

        UserWarning: CUDA initialization: CUDA unknown error ...
        Setting the available devices to be zero.

    auto_select_device() then quietly selects `cpu`, and the job runs. That is the bad
    outcome, not a crash: training and the Jacobian sweep are 10-50x slower on CPU, so the
    allocation burns its entire wall clock and is killed having produced nothing. The
    numbers would have been correct; they just never arrive.

    Only fires when Slurm actually gave us a GPU, so CPU-only work (build_cache.slurm
    requests none) and laptop runs are unaffected.
    """
    import torch

    allocated = None
    for var in ("SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE", "CUDA_VISIBLE_DEVICES"):
        value = os.environ.get(var, "").strip()
        if value and value not in ("NoDevFiles", "0" * 0):
            allocated = f"{var}={value}"
            break
    if allocated is None or torch.cuda.is_available():
        return

    raise RuntimeError(
        f"Slurm allocated a GPU ({allocated}) but torch.cuda.is_available() is False, "
        f"so this job would silently run on CPU -- 10-50x slower, which on a bounded "
        f"wall clock means killed before finishing rather than merely slow. Look "
        f"above for torch's 'CUDA initialization: CUDA unknown error' warning; it "
        f"is usually a bad node rather than anything wrong here, so cancelling and "
        f"resubmitting normally lands elsewhere. Check a node with: srun --gpus=1 "
        f"--time=5 python -c 'import torch; print(torch.cuda.is_available())'. "
        f"If you genuinely want CPU, submit without --gpus.")


def patch_sklearn_squared_kwarg() -> bool:
    """Restore mean_squared_error(..., squared=False) for GraphGPS's regression logger.

    scikit-learn deprecated `squared` in 1.4 and REMOVED it in 1.6. GraphGPS is pinned at
    a Feb 2023 commit and still calls it, in exactly one place -- logger.py's regression
    branch, for the 'rmse' stat. So the classification datasets are unaffected and
    peptides-struct dies at the end of its first epoch with
    `TypeError: got an unexpected keyword argument 'squared'`.

    Patched here rather than fixed in either of the two places it looks like it belongs:

      * NOT in the GraphGPS clone. It is a pinned fork -- config.PINNED_COMMITS -- and an
        edit there is invisible to git, lost on a re-clone, and silently makes "we ran
        upstream at commit X" false.
      * NOT by downgrading the installed env. That would be the root fix and it IS pinned
        in envs/graphgps_env.yml now for fresh builds, but the existing environment took
        six rounds of conflicting pins to converge and a downgrade can drag numpy or
        scipy with it. Repairing a working env mid-grid is a worse risk than a six-line
        shim.

    Semantics are exact, not approximate. Old `squared=False` took the square root PER
    OUTPUT and then averaged; sqrt of the averaged MSE is a different number whenever
    there is more than one target, and peptides-struct has eleven. sklearn >= 1.4 ships
    root_mean_squared_error, which is precisely the old behaviour, so that is used when
    present and the per-output computation is reproduced by hand otherwise.

    Returns True if it patched anything, so the caller can say so rather than leaving a
    silent monkeypatch in the run.
    """
    import inspect

    import numpy as np

    import graphgps.logger as gl

    try:
        if "squared" in inspect.signature(gl.mean_squared_error).parameters:
            return False   # old enough sklearn: nothing to do
    except (TypeError, ValueError):
        return False       # unintrospectable; leave it alone rather than guess

    original = gl.mean_squared_error
    try:
        from sklearn.metrics import root_mean_squared_error as _rmse
    except ImportError:
        _rmse = None

    def mean_squared_error(y_true, y_pred, *, squared=True,
                           multioutput="uniform_average", **kwargs):
        if squared:
            return original(y_true, y_pred, multioutput=multioutput, **kwargs)
        if _rmse is not None:
            return _rmse(y_true, y_pred, multioutput=multioutput, **kwargs)
        per_output = original(y_true, y_pred, multioutput="raw_values", **kwargs)
        rooted = np.sqrt(per_output)
        if isinstance(multioutput, str):
            return rooted if multioutput == "raw_values" else np.average(rooted)
        return np.average(rooted, weights=multioutput)

    gl.mean_squared_error = mean_squared_error
    return True


# ---------------------------------------------------------------------------
# smoke testing
# ---------------------------------------------------------------------------
SMOKE_TEST_BATCHES = 2


class _TruncatedLoader:
    """A DataLoader that stops after n batches, for --smoke-test.

    Wraps rather than rebuilds so the batches are byte-identical to a real run's -- same
    dataset, sampler, collate and PE tensors. The point of a smoke test is to exercise the
    real path cheaply, so anything that made these batches special would defeat it.

    Forwards unknown attributes because GraphGPS reaches past the iterator: custom_train
    calls len(loader) to detect the last batch for gradient accumulation, and logs
    len(loader.dataset). len() reports the truncated count (the accumulation boundary must
    match what is actually iterated); .dataset falls through to the real one, so the log
    line still states the true split size rather than implying the dataset shrank.
    """

    def __init__(self, loader, n_batches: int):
        self._loader = loader
        self._n_batches = n_batches

    def __iter__(self):
        for i, batch in enumerate(self._loader):
            if i >= self._n_batches:
                return
            yield batch

    def __len__(self):
        return min(self._n_batches, len(self._loader))

    def __getattr__(self, name):
        return getattr(self._loader, name)


# ---------------------------------------------------------------------------
# run directory layout
# ---------------------------------------------------------------------------
def run_dir_for(out_dir: str, seed) -> str:
    """The directory GraphGPS writes ONE cell's stats and checkpoints into.

    Mirrors main.py's custom_set_run_dir: out_dir/<run_id>, with run_id == seed in the
    multi-seed mode this launcher reproduces.

    Exists as a function because three places have to agree on it and cannot check each
    other: graphgps_train sets cfg.run_dir from it, _read_best_metric reads the trained
    cell's score back out of it, and GraphGPS itself derives the checkpoint directory
    (run_dir/ckpt) that auto_resume depends on. Spelled out separately, a change to one
    would not fail -- _read_best_metric would just find no stats.json and return None, so
    the cell reports a null metric after training perfectly well.
    """
    return os.path.join(out_dir, str(seed))


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
    from torch_geometric.graphgym.logger import set_printing
    from graphgps.logger import create_logger

    if patch_sklearn_squared_kwarg():
        print("  note: patched graphgps.logger.mean_squared_error for scikit-learn >= 1.6 "
              "(the `squared` kwarg was removed); RMSE semantics are unchanged")
    from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig
    from torch_geometric.graphgym.optim import OptimizerConfig

    cfg = build_graphgym_cfg(run_cfg, graphgps_dir)

    # Slurm gives a task a CPU allocation; torch otherwise sizes its thread pool from the
    # machine's total core count and oversubscribes it, which on a shared cluster slows
    # down both this job and its neighbours. main.py sets this for the same reason.
    import torch
    torch.set_num_threads(cfg.num_threads)

    # --- per-run state main.py sets inside its run loop ---------------------------------
    # build_graphgym_cfg cannot set these: they belong to one iteration of the loop over
    # seeds, not to the config, and main.py accordingly assigns them per iteration
    # (main.py:127-135). This launcher runs exactly one cell per process, so the "loop" is
    # one pass -- but the assignments are still required, and skipping them fails late and
    # obscurely: `AttributeError: run_dir` raised by yacs from inside create_logger, after
    # the ~2 min PE pre-transform has already been paid for.
    #
    # run_id == seed: main.py's run_loop_settings() uses the seed as the run id in its
    # multi-seed mode (run_ids = seeds), which is the mode this launcher reproduces. That
    # is what makes cfg.run_dir == out_dir/<seed>, the layout _read_best_metric and the
    # `results/raw/<cell>/<seed>/ckpt/` checkpoint path both already assume.
    cfg.run_id = cfg.seed
    cfg.run_dir = run_dir_for(cfg.out_dir, cfg.seed)
    # main.py's custom_set_run_dir branches here on cfg.train.auto_resume and calls
    # makedirs_rm_exist -- i.e. DELETES the run directory -- when it is False. Inlining
    # only the exist_ok branch keeps that destructive path out of the code entirely. That
    # is no longer merely defensive: --smoke-test deliberately sets auto_resume False, so
    # upstream's branch WOULD now be reached, and a smoke test would wipe a directory.
    os.makedirs(cfg.run_dir, exist_ok=True)
    # Without this, GraphGPS is SILENT: it reports epoch stats through logging.info, and
    # an unconfigured root logger defaults to WARNING, so every epoch line is discarded.
    # It also writes cfg.run_dir/logging.log. Called before create_loader so the loader's
    # own dataset/PE messages are captured too, matching main.py's order.
    set_printing()

    # Provenance: the fully resolved config this cell actually trained on, post-merge and
    # post-assert_cfg. main.py dumps this to cfg.out_dir; we write it to cfg.run_dir
    # instead, because out_dir is shared by every seed of a cell and the Slurm array runs
    # several of them at once -- concurrent writes to one config.yaml would interleave.
    # Per-seed is also the more useful record, since cfg.seed is part of what it captures.
    with open(os.path.join(cfg.run_dir, "config.yaml"), "w") as f:
        cfg.dump(stream=f)

    seed_everything(cfg.seed)
    auto_select_device()
    assert_gpu_if_slurm_allocated_one()

    # Point GraphGPS's loader at THIS repo's PE cache before it builds anything. Without
    # this the GPS arm trains on GraphGPS's own LapPE/RWSE/SignNet while the SAN arm
    # trains on the cache's, and the two are not the same encoding -- see
    # graphgps_pe_cache's header for the four definitions that differ. Must happen after
    # build_graphgym_cfg (which sets max_freqs to the cache's width) and before
    # create_loader (which runs the pre-transform being replaced).
    from backends.graphgps_pe_cache import install as install_pe_cache
    pe_cache = install_pe_cache(run_cfg, cfg) if run_cfg.pe != "none" else None

    loaders = create_loader()

    if getattr(run_cfg, "smoke_test", False):
        # Applied AFTER create_loader so the dataset build and the PE pre-transform still
        # run in full -- those are the parts a smoke test most needs to exercise, and they
        # are where every failure on this branch so far has actually been.
        loaders = [_TruncatedLoader(dl, SMOKE_TEST_BATCHES) for dl in loaders]
        print(f"  SMOKE TEST: 1 epoch, {SMOKE_TEST_BATCHES} batches per split")

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

    # custom_train reads cfg.params directly when it logs each epoch
    # (custom_train.py:45,74), so this is load-bearing, not just bookkeeping.
    n_params = cfg.params = params_count(model)
    print(f"  parameters: {n_params:,}")
    if n_params > PARAM_BUDGET_WARN:
        print(f"  WARNING: {n_params:,} parameters is well above the ~500k budget and "
              f"above every GraphGPS LRGB reference config (max "
              f"{PARAM_BUDGET_REFERENCE_MAX:,}). This arm is NOT parameter-matched to the "
              f"others, which weakens any cross-PE comparison drawn from it.")

    train_dict[cfg.train.mode](loggers, loaders, model, optimizer, scheduler)

    return {
        "model": model,
        "loaders": loaders,
        "num_params": n_params,
        "metric_name": run_cfg.metric_name,
        "metric_value": _read_best_metric(cfg, run_cfg.metric_name),
        "cfg": cfg,
    }


def _read_split_series(run_dir, split, key):
    """{epoch: value} for one metric on one split, plus every key seen in that file."""
    import json

    path = os.path.join(run_dir, split, "stats.json")
    series, seen = {}, set()
    if not os.path.exists(path):
        return series, seen, path
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            seen.update(rec)
            if key in rec and "epoch" in rec:
                # a requeued cell re-logs epochs it already wrote; last write wins
                series[int(rec["epoch"])] = rec[key]
    return series, seen, path


def _read_best_metric(cfg, metric_name) -> Optional[float]:
    """Test-split value AT THE EPOCH SELECTED ON VALIDATION.

    Not the best test value. That distinction is the whole point of this function, and it
    used to get it wrong: it took the max over the test split across every epoch, which is
    model selection on the test set. On the first real cell that reported a number drawn
    from wherever test AP happened to peak across 200 epochs, rather than 0.6496, the test
    AP at epoch 151 -- the epoch validation actually chose. Small in magnitude, and
    exactly the kind of thing a reviewer is entitled to reject a table over.

    GraphGPS's own "Best so far" line has always done this correctly; only our read-back
    did not. Parsed from disk rather than scraped off the logger objects because the
    training loop owns those, their internals move between versions, and the stats file is
    the artefact GraphGPS itself treats as the result.
    """
    run_dir = run_dir_for(cfg.out_dir, cfg.seed)
    key = GRAPHGPS_METRIC_KEY.get(metric_name, metric_name)
    lower_better = metric_name in ("mae", "loss")

    val, val_keys, val_path = _read_split_series(run_dir, "val", key)
    test, test_keys, test_path = _read_split_series(run_dir, "test", key)

    # Checked BEFORE the empty-series return: a file that exists with records but no such
    # metric yields an empty SERIES too, so returning None first would swallow exactly the
    # naming bug this is here to catch.
    for seen, path in ((val_keys, val_path), (test_keys, test_path)):
        if seen and key not in seen:
            # File exists with records but no such metric: a naming bug, not a missing
            # score. Returning None for it is how the macro_f1/f1 mismatch stayed
            # invisible -- the cell trained, the probe ran, and the result was written
            # with status "ok" and metric_value null. auto_resume makes the re-run cheap.
            raise RuntimeError(
                f"metric {metric_name!r} (GraphGPS key {key!r}) appears in no record of "
                f"{path}. Available keys: {sorted(seen)}. Fix the mapping in "
                f"graphgps_backend.GRAPHGPS_METRIC_KEY rather than config.TASK_METRIC, "
                f"which the SAN arm also reads.")

    if not val and not test:
        # Legitimately absent: a cell pre-empted before its first eval. "No score yet" is
        # the honest answer, and the requeue will fill it in.
        return None

    if not val:
        raise RuntimeError(
            f"{os.path.join(run_dir, 'val', 'stats.json')} has no usable records while "
            f"the test split does. The epoch must be chosen on validation; falling back "
            f"to the best TEST value would be selection on the test set, which is the "
            f"bug this function exists to avoid.")

    best_epoch = (min if lower_better else max)(val, key=lambda e: val[e])
    if best_epoch not in test:
        raise RuntimeError(
            f"validation selected epoch {best_epoch} but the test split has no record for "
            f"it (test epochs: {sorted(test)[:5]}...). Reporting a different epoch's test "
            f"score would silently break the selection protocol.")

    print(f"  metric: {metric_name}={test[best_epoch]:.4f} at epoch {best_epoch} "
          f"(selected on val {metric_name}={val[best_epoch]:.4f}; best test seen was "
          f"{(min if lower_better else max)(test.values()):.4f} -- NOT reported, that "
          f"would be selection on test)")
    return test[best_epoch]


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


def unwrap_graphgym_module(model):
    """Return the GPSModel inside whatever create_model() handed back.

    torch_geometric.graphgym.create_model returns a GraphGymModule -- a Lightning wrapper
    holding the real network as `.model` -- not the network itself. GraphGPS's own
    training loop never notices, because it only ever calls the wrapper's forward. The
    probe does notice: it reaches into named_children() to run the encoder separately and
    replay the layer stack, and on the wrapper that yields ['model'] and nothing else.

    Decided on named_children(), NOT hasattr/getattr. GraphGymModule forwards `encoder`,
    `mp` and `post_mp` as @property to the network inside it, so every attribute-level
    test for "is this the real model?" answers yes on the wrapper too -- that forwarding
    is the whole point of the wrapper. Registered submodules do not lie the same way: a
    property is not a submodule, so named_children() on the wrapper is exactly ['model']
    while the real network's is ['encoder', ..., 'post_mp']. This is also what the probe
    itself walks, so the check tests the same thing the caller depends on.

    Uses no isinstance check, so it needs no Lightning import, and is a no-op on an
    already-unwrapped GPSModel. Bounded rather than `while True` -- a wrapper whose
    `.model` is itself would otherwise hang instead of raising.
    """
    for _ in range(4):
        children = dict(model.named_children())
        if "encoder" in children:
            return model
        inner = children.get("model")
        if inner is None or inner is model:
            return model
        model = inner
    return model


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
    model = unwrap_graphgym_module(model)
    model.eval()   # BatchNorm must use running stats: training-mode batch statistics would
                   # make the Jacobian depend on the rest of the batch, not just node u
    device = device or next(model.parameters()).device
    batch = data.clone().to(device)

    # The probe hands over ONE graph, so `.batch` is None -- there was no DataLoader to
    # build it. Most of the stack tolerates that: to_dense_batch, which GPS's attention
    # uses, creates the zeros itself when handed None. SignNet's sign-invariant net does
    # not; batched_n_nodes calls batch_index.max() and dies with `'NoneType' object has
    # no attribute 'max'`. It needs the node counts to mask eigenvector columns beyond
    # each graph's size, the same padding concern lap_to_graphgps handles on our side.
    #
    # Setting it explicitly is not a SignNet workaround: a single graph genuinely IS a
    # batch of one, and saying so beats depending on every downstream op to guess. For
    # the arms that already worked this changes nothing -- they were getting these same
    # zeros, just constructed further down.
    if getattr(batch, "batch", None) is None:
        batch.batch = torch.zeros(int(batch.num_nodes), dtype=torch.long, device=device)

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
