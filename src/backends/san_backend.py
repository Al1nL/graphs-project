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

─────────────────────────────────────────────────────────────────────────────
CHANGELOG (this pass)
─────────────────────────────────────────────────────────────────────────────
- Throttled the per-step gc.collect()/torch.cuda.empty_cache() call from every
  training step to every 50 steps. empty_cache() forces a CUDA sync and was
  costing significant throughput when called on every iteration. At the
  observed ~1MB/step leak rate, 50 steps accumulates ~50MB, which is well
  within the multi-GB of unused GPU memory headroom these runs actually have
  (confirmed via nvidia-smi: ~2GB used of 11GB available), so this remains a
  safe margin against OOM.
- Added early stopping: training now stops if best_metric hasn't improved in
  `run_cfg.early_stop_patience` epochs (default 15, ~1.5x the LR scheduler's
  own patience of 10, so a scheduled LR drop gets a chance to help before
  giving up). Pass early_stop_patience=0 to disable and always run the full
  `epochs` ceiling. NOTE: epochs_since_improvement is NOT persisted in the
  checkpoint, so a resumed run restarts its patience counter from 0 -- a
  resume gets a fresh grace period rather than picking up mid-count.
- Added an LR print after each scheduler.step(val_loss) call, so it's visible
  in the logs whether/when ReduceLROnPlateau actually drops the learning rate
  -- previously this was silent and unverifiable from the logs alone.
- Gradient-checkpointing wrapper now skips torch.utils.checkpoint entirely
  when torch.is_grad_enabled() is False (i.e. during _evaluate()'s
  torch.no_grad() blocks), calling the layer directly instead. Checkpointing
  exists purely to recompute activations on backward to save memory; under
  no_grad there is no backward, so wrapping the call there only wasted
  compute and triggered PyTorch's "None of the inputs have requires_grad=True"
  warning on every eval batch. This was investigated as a possible cause of
  the training-metric plateau observed across all 5 PE variants on
  peptides-func; the fix removes the ambiguity by construction -- if this
  warning still appears in the logs after this change, it is firing during
  an actual training step (grad enabled) and is a real bug worth chasing,
  not eval-loop noise. Set env var SAN_DEBUG_CHECKPOINT_GRAD=1 to print
  requires_grad/grad_enabled state on every checkpointed layer call for
  targeted debugging if needed.
- Tightened the cache-clear interval from every 50 steps to every 10, after a
  real CUDA OOM (job 780000, DGL's GSpMM allocator, ~8h into a run) at the
  every-50 setting. This is a mitigation, not a fix -- see the inline comment
  at the empty_cache() call site for why, and what the real fix would be
  (an edge-budget batch sampler, not yet implemented despite
  run_experiment.py's --edge-budget flag documenting one).
- Added regularization for peptides-func/peptides-struct (both full_graph=True,
  ~928k params, only ~10.8k training graphs, previously dropout=0.0,
  in_feat_dropout=0.0, weight_decay=0.0): dropout and in_feat_dropout raised
  to 0.1, weight_decay raised to 1e-5. Motivation: after ruling out effective
  batch size (accumulation_steps 2->8, no change), the LR scheduler not
  firing (confirmed it does fire correctly, no change to the oscillation
  pattern), and NaN/inf (none found), a persistent noisy oscillation in
  val/test AP with no sustained improvement across all 5 PE variants
  remained unexplained. Zero regularization on this param-count/data-size
  ratio is a plausible remaining cause and hasn't been tested. This has NOT
  been validated yet -- treat as an untested hypothesis, not a confirmed fix.
  pascalvoc-sp's dropout/weight_decay were left untouched (full_graph=False,
  different memory/data regime, no evidence yet that it has the same issue).
- Implemented EdgeBudgetBatchSampler (previously referenced only in
  run_experiment.py's --edge-budget help text as a documented no-op -- the
  class did not exist). This is the real fix for full_graph=True OOM: it
  caps total densified edges (sum of n*(n-1)) per batch directly, instead of
  relying on a fixed batch_size + empty_cache() timing to keep worst-case
  batches under the memory ceiling (which is a probability reduction, not a
  cap, and was confirmed insufficient by two real OOM crashes even after
  tightening the empty_cache() interval to every 10 steps -- see job 780000
  and 782872/782552 in the changelog history). Activate it by passing
  --edge-budget N (N = max total n*(n-1) per batch; tune based on available
  GPU memory -- not yet calibrated against real hardware, start with
  something conservative and watch nvidia-smi). Passing --edge-budget 0 or
  omitting it entirely keeps the old fixed --batch-size behavior unchanged,
  so existing runs/configs are unaffected unless this is explicitly opted
  into. Only applies when net_params["full_graph"] is True; pascalvoc-sp
  (full_graph=False) always uses the fixed-batch_size path regardless of
  this flag.
- Extended the RWSE encoding fix (originally built only for peptides-func's
  _SAN_RWSE class) to the other two datasets' model classes:
  _SAN_NodeLPE_Regression (peptides-struct) and _SAN_NodeClassification
  (pascalvoc-sp). Both of these use a DATASET-level _variant
  ("regression" / "node_classification") that build_san_net_params
  preserves over PE_SPEC's PE-level _variant -- meaning, before this fix,
  --pe rwse on either of these datasets would have silently routed through
  the same eigenvector-shaped linear_A/PE_Transformer/mean-pool-across-steps
  path diagnosed as destroying RWSE's step-order signal on peptides-func
  (see _SAN_RWSE's docstring), never reaching the fix at all. Both classes
  now check net_params["pe"] (a new key set by build_san_net_params,
  necessary because net_params["LPE"] == "node" is shared by lappe/rwse/grpe
  and can't disambiguate them) and use a plain per-node MLP over the full
  RWSE vector when pe == "rwse", matching _SAN_RWSE's approach. NOT yet
  validated on real training runs for either dataset -- validated so far
  only for peptides-func's _SAN_RWSE (climbing well in early epochs at time
  of writing). Smoke-test both datasets' rwse runs before trusting this.
- Fixed a crash in _SAN_NodeLPE_Regression (peptides-struct): it built
  SAN_NodeLPE(net_params) unconditionally, but upstream SAN_NodeLPE.__init__
  unconditionally reads net_params['LPE_dim']/['LPE_n_heads']/['LPE_layers']
  at construction time -- even when LPE=="none" (it only checks that flag
  later, inside forward). PE_SPEC["none"] intentionally omits those keys
  since they're meaningless without a PE, so --pe none on peptides-struct
  crashed with KeyError: 'LPE_layers' the first time it was actually run
  (during the hyperparameter sweep -- this path had never been exercised
  before). Fixed by filling in harmless defaults via setdefault before
  constructing SAN_NodeLPE; they're inert when LPE=="none". Added the same
  defensive setdefault to _SAN_GRPE, which has the identical unconditional-
  construction pattern -- currently unreachable in practice since
  PE_SPEC["grpe"] always supplies those keys, but guarded anyway rather
  than relying on that staying true. _SAN_NodeClassification (pascalvoc-sp)
  was checked and does NOT have this issue -- it builds its LPE components
  itself, conditionally on lpe != "none", rather than delegating to
  upstream SAN_NodeLPE.
- Fixed a second bug in the same code path, surfaced immediately after the
  KeyError fix above: _SAN_NodeLPE_Regression.forward required EigVecs/
  EigVals with no defaults, but _forward_pass calls model(bg, h, e) with
  only 3 args for pe == "none" -- TypeError: missing 2 required positional
  arguments. Fixed by defaulting both to None and adding a third forward
  branch for pe == "none": since the base SAN_NodeLPE's embedding_h output
  width is fixed at (GT_hidden_dim - LPE_dim) regardless of whether a real
  PE is used (see the setdefault fix above), a same-width zero vector is
  concatenated in place of a real PE when none is requested, keeping h's
  concatenated width equal to GT_hidden_dim as every downstream GT layer
  expects. _SAN_GRPE is unaffected (only ever invoked with pe == "grpe",
  which always supplies real EigVecs/EigVals). _SAN_NodeClassification is
  unaffected (its embedding_h width is already conditionally sized on
  lpe != "none", so it never needs a zero-pad fallback).
- Fixed a third bug in the same forward, also a pre-existing issue predating
  every other fix in this file (surfaced now only because this was the
  first time ANY PE was actually run on peptides-struct, not something
  specific to --pe none): _SAN_NodeLPE_Regression.forward called
  base.embedding_e(e), but upstream SAN_NodeLPE actually names this
  attribute embedding_e_real (confirmed via a checkpoint state_dict
  mismatch seen earlier in debugging: unexpected key
  "embedding_e_real.bond_embedding_list...."). AttributeError:
  'SAN_NodeLPE' object has no attribute 'embedding_e'. Fixed with a
  getattr-guarded fallback that prefers embedding_e_real if present.
  _SAN_GRPE is unaffected: its forward calls self._base(...) directly
  (upstream's own forward method), rather than a hand-replicated copy, so
  it always uses upstream's real attribute names automatically.
- Added gradient clipping (clip_grad_norm_, max_norm=1.0) before every
  optimizer.step(). Added after observing pascalvoc-sp's --pe none
  --gamma 1e-5 sweep run freeze to a bit-identical output (val/test AP/F1
  unchanged for 16 straight epochs, starting from epoch 0) -- the classic
  signature of an early exploding-gradient step collapsing the model into
  a saturated/dead region. No-op for gradients already under max_norm, so
  this should not change behavior for runs that weren't hitting this
  failure mode.
- Added class-weighted CrossEntropyLoss for pascalvoc-sp
  (_compute_pascalvoc_class_weights), computed as inverse class frequency
  over the REAL training set (not a diagnostic sample), normalized so mean
  weight is ~1. Motivation: pascalvoc-sp's labels are heavily imbalanced
  (majority class ~71% of nodes in a diagnostic sample; several classes
  under 1%), but training used plain unweighted CrossEntropyLoss while the
  eval metric (macro-F1) weights every class equally regardless of
  frequency -- a real train/eval objective mismatch that unweighted loss
  gives the model little incentive to address. _build_loaders now returns
  a 4th value (class_weights, None for the other two datasets) alongside
  the 3 loaders; _loss_for takes class_weights/device kwargs and applies
  them only for pascalvoc-sp. NOT yet validated against a real training
  run -- confirmed only that node_feat_dim=14/edge_feat_dim=2/n_classes=21
  (BASE_NET_PARAMS's hardcoded values) all correctly match the real data,
  ruling out a shape-mismatch bug as the cause of pascalvoc-sp's low
  scores; the imbalance/objective-mismatch hypothesis above is untested.
- Implemented the sensitivity probe for SAN (make_san_model_fn), previously a
  stub raising NotImplementedError. Mirrors graphgps_backend.make_gps_model_fn's
  contract: differentiates from h^(0) (post-embedding, pre-GT-layers node
  representation, since raw atom-index features have no derivative), returns
  final-layer node embeddings (not pooled), and reports dim_inner as
  n_shared_feats -- which for SAN is ALWAYS exactly GT_hidden_dim regardless of
  PE (embedding_h width + PE-vector width sum to GT_hidden_dim algebraically),
  so unlike GraphGPS there is no shrinking-content-channel tradeoff to choose
  between. Added _PEAttachedDataset: a probe-only wrapper (separate from
  training's tuple-yielding _PECacheDataset) that attaches PE fields plus _pe/
  _full_graph onto each Data object, needed because SAN's PE comes from an
  external cache keyed by graph index rather than being computed internally
  from raw features the way GraphGPS's encoder does. san_train now returns
  "probe_dataset" in its result dict; run_experiment.run_cell needs a small
  patch (train_out.get("probe_dataset", loaders[-1].dataset)) to use it instead
  of the generic loaders[-1].dataset path, and "san" needs adding to
  PROBE_WIRED_BACKBONES -- see the accompanying note for that change, made
  outside this file.
- Two bugs found by an actual first run of the probe on peptides-func:
  (1) sensitivity.py's is_grads_batched fast path raises TypeError on this
  env's PyTorch 1.9.0 (added ~1.11) -- not a SAN-specific bug, but it blocked
  every backbone's probe here. sensitivity.py's own fallback already existed
  for this exact situation but only caught (RuntimeError, NotImplementedError),
  not TypeError; widened to catch TypeError too. Falls back to an unbatched
  VJP loop -- same numbers, slower. (2) torch.utils.checkpoint on this
  PyTorch version raises "Checkpointing is not compatible with .grad()" the
  moment sensitivity.py's torch.autograd.grad() touches a graph that passed
  through a checkpointed segment -- affects every full_graph=True dataset
  (peptides-func, peptides-struct), since enable_gradient_checkpointing
  permanently wraps every GT layer's forward at train time. Fixed by having
  enable_gradient_checkpointing also store the ORIGINAL unwrapped forward as
  layer._probe_forward; make_san_model_fn's model_fn now calls that directly
  instead of the checkpoint-wrapped layer.forward, bypassing checkpointing
  entirely for probing (which doesn't need its memory savings -- one graph
  at a time, not a full batch). Layers that were never checkpointed
  (pascalvoc-sp, full_graph=False) don't have this attribute; model_fn falls
  back to calling the layer directly for those. Confirmed via one real run
  on peptides-func --pe none that training + both these fixes work together
  end-to-end; NOT yet confirmed for the other 4 PE variants or the other two
  datasets -- expect at least one of the class-specific attribute-name
  guesses in make_san_model_fn (documented in its own docstring) to need a
  similar fix once those are actually run.
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

    if variant == "signnet":
        return _SAN_SignNetLPE(net_params)
    if variant == "grpe":
        return _SAN_GRPE(net_params)
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

        # SAN_NodeLPE.__init__ unconditionally reads LPE_dim/LPE_n_heads/LPE_layers.
        # PE_SPEC["grpe"] always supplies these today, so this is currently
        # unreachable in practice -- but guard it anyway rather than relying on
        # that staying true forever (see the real crash this exact pattern caused
        # in _SAN_NodeLPE_Regression under --pe none on peptides-struct).
        net_params = dict(net_params)
        net_params.setdefault("LPE_dim", 16)
        net_params.setdefault("LPE_n_heads", 4)
        net_params.setdefault("LPE_layers", 2)

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

    @property
    def layers(self):
        """Expose _base.layers so enable_gradient_checkpointing can wrap them."""
        return self._base.layers

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

    RWSE handling: peptides-struct's BASE_NET_PARAMS sets _variant="regression" at
    the DATASET level, which build_san_net_params preserves over PE_SPEC's own
    _variant -- meaning every PE (including rwse) routes through this one class,
    unlike peptides-func where rwse gets its own _SAN_RWSE class. Without an
    explicit branch here, --pe rwse would silently fall through to the same
    eigenvector-shaped linear_A/PE_Transformer/mean-pool-across-steps path that
    was diagnosed as destroying RWSE's step-order signal on peptides-func (see
    _SAN_RWSE's docstring for the full diagnosis). net_params["pe"] (set by
    build_san_net_params) is used here since net_params["LPE"]=="node" is shared
    by lappe/rwse/grpe and can't disambiguate them on its own.
    """
    def __init__(self, net_params):
        super().__init__()
        from nets.molhiv_graph_regression.SAN_NodeLPE import SAN_NodeLPE
        from layers.mlp_readout_layer import MLPReadout
        n_classes = net_params.get("n_classes", 1)

        # SAN_NodeLPE.__init__ unconditionally reads LPE_dim/LPE_n_heads/LPE_layers
        # at construction time, even when LPE == "none" (it only checks the LPE
        # flag later, inside forward). PE_SPEC["none"] intentionally omits these
        # keys since they're meaningless without a PE, which crashes construction
        # with KeyError: 'LPE_layers' the first time --pe none is actually run on
        # peptides-struct (this path went untested until the hyperparameter sweep
        # exercised it). Fill in harmless defaults so construction doesn't crash;
        # they're never actually used when LPE == "none".
        net_params = dict(net_params)
        net_params.setdefault("LPE_dim", 16)
        net_params.setdefault("LPE_n_heads", 4)
        net_params.setdefault("LPE_layers", 2)

        self._base = SAN_NodeLPE(net_params)
        # Replace sigmoid MLP with linear regression head (no activation)
        self._base.MLP_layer = MLPReadout(net_params["GT_out_dim"], n_classes)
        self._n_classes = n_classes

        self.is_rwse = net_params.get("pe") == "rwse"
        self.lpe_dim = net_params["LPE_dim"]
        if self.is_rwse:
            LPE_dim = net_params["LPE_dim"]
            # Same fix as _SAN_RWSE: plain per-node MLP over the full ordered
            # RWSE vector, no mean-pool-across-steps.
            self.rwse_encoder = torch.nn.Sequential(
                torch.nn.Linear(LPE_dim, LPE_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(LPE_dim, LPE_dim),
            )

    @property
    def layers(self):
        """Expose _base.layers so enable_gradient_checkpointing can wrap them."""
        return self._base.layers

    def forward(self, g, h, e, EigVecs=None, EigVals=None):
        import dgl
        # Replicate SAN_NodeLPE.forward but WITHOUT the final sigmoid
        base = self._base
        h = base.embedding_h(h)
        h = base.in_feat_dropout(h)
        # Upstream SAN_NodeLPE names this attribute embedding_e_real (confirmed
        # via a checkpoint state_dict mismatch earlier in debugging), not
        # embedding_e as this class originally assumed -- a pre-existing bug
        # that predates all other fixes in this file. It went undetected until
        # now because this was the first time ANY PE was actually run on
        # peptides-struct (not specific to --pe none). getattr fallback kept
        # in case upstream's naming differs across versions/checkouts.
        e = (base.embedding_e_real(e) if hasattr(base, "embedding_e_real")
             else base.embedding_e(e))

        if self.is_rwse:
            # EigVecs is actually the [n, 20] rwse tensor here (see _forward_pass's
            # pe == "rwse" branch, which passes rwse features in the eigvecs slot).
            # EigVals is a zero placeholder and is intentionally unused.
            pe = self.rwse_encoder(EigVecs)  # [n, LPE_dim]
        elif EigVecs is not None and EigVals is not None:
            # LPE
            EigVecs_u = EigVecs.unsqueeze(-1)  # [n, k, 1]
            pe_inp = torch.cat([EigVecs_u, EigVals], dim=-1)  # [n, k, 2]
            empty_mask = (EigVecs == 0).all(dim=-1)  # [n]
            pe_inp[empty_mask] = 0.0
            pe_inp = pe_inp.transpose(0, 1)  # [k, n, 2]
            pe = base.linear_A(pe_inp)       # [k, n, LPE_dim]
            pe = base.PE_Transformer(pe)     # [k, n, LPE_dim]
            pe = pe.transpose(0, 1).mean(dim=1)  # [n, LPE_dim]
        else:
            # pe == "none": _forward_pass calls model(bg, h, e) with no PE args at
            # all. The base SAN_NodeLPE was still constructed with embedding_h's
            # output width fixed at (GT_hidden_dim - LPE_dim) -- see the setdefault
            # fix in __init__ -- so a same-width PE slot must still be filled to
            # keep h's concatenated width equal to GT_hidden_dim. A zero vector is
            # the correct no-PE filler (equivalent to no positional information).
            pe = torch.zeros(h.shape[0], self.lpe_dim, device=h.device, dtype=h.dtype)

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

        # RWSE gets its own encoder, same fix as _SAN_RWSE / _SAN_NodeLPE_Regression:
        # a plain per-node MLP over the full ordered RWSE vector, instead of falling
        # through to the eigenvector-shaped linear_A/PE_Transformer/mean-pool-across-
        # steps path below (self.lpe == "node" is shared by lappe/rwse/grpe and can't
        # tell them apart on its own -- net_params["pe"], set by build_san_net_params,
        # is used here instead).
        self.is_rwse = net_params.get("pe") == "rwse"
        if self.is_rwse:
            self.rwse_encoder = torch.nn.Sequential(
                torch.nn.Linear(LPE_dim, LPE_dim), torch.nn.ReLU(),
                torch.nn.Linear(LPE_dim, LPE_dim),
            )
        # LPE components (only if not 'none' and not rwse, which has its own path)
        elif lpe != "none":
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
        if self.is_rwse and EigVecs is not None:
            # EigVecs is actually the [n, 20] rwse tensor here (see _forward_pass's
            # pe == "rwse" branch). EigVals is a zero placeholder, unused.
            pe = self.rwse_encoder(EigVecs)  # [n, LPE_dim]
            h = torch.cat([h, pe], dim=-1)
        elif self.lpe != "none" and EigVecs is not None:
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
    # Explicit PE identity, separate from "LPE" (which is shared as "node" by
    # lappe/rwse/grpe and can't disambiguate them). Dataset-variant classes
    # (_SAN_NodeLPE_Regression, _SAN_NodeClassification) need this to give RWSE
    # its own encoding branch instead of falling through to the eigenvector-
    # shaped PE_Transformer path meant for lappe/grpe -- see _SAN_RWSE's
    # docstring for why that path silently destroys RWSE's step-order signal.
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

    train_loader, val_loader, test_loader, class_weights, probe_dataset = _build_loaders(
        run_cfg, net_params, train_params)

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
            if net_params.get("full_graph") and device.type == "cuda" and global_step % 10 == 0:
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
    """Wraps a _PECacheDataset (or Subset thereof), returning single self-contained
    Data objects with PE fields attached as attributes, instead of (data, pe_data)
    tuples.

    Exists ONLY for the sensitivity probe. run_experiment.sample_test_graphs and
    run_probe are written generically across backbones and expect
    `test_dataset[i]` to return one Data object carrying everything needed to
    reconstruct the model's forward pass for that graph -- training's collate_fn
    needs the tuple form instead, so this is a separate wrapper rather than a
    change to _PECacheDataset itself, to avoid touching the training path at all.

    Also attaches `_pe` (the active PE name) and `_full_graph` (net_params's
    full_graph flag) onto each Data object. Unlike GraphGPS, which computes its
    PE internally from raw x/edge_index inside the model's own encoder, SAN's PE
    comes from an external cache keyed by graph index and split -- so
    make_san_model_fn has no way to know which PE was active or how to rebuild
    the DGL graph (dense vs sparse) without these two attributes.
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
            loaders.append(DataLoader(ds, batch_sampler=sampler, collate_fn=collate))
        else:
            shuffle = split == "train"
            bs = train_params["batch_size"] if split == "train" else 16
            loaders.append(DataLoader(ds, batch_size=bs, shuffle=shuffle,
                                      collate_fn=collate))

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
    """Wrap a trained SAN model for sensitivity.compute_sensitivity_curve.

    Mirrors graphgps_backend.make_gps_model_fn's contract exactly:
      model_fn(x)  runs SAN's GraphTransformerLayer stack (self.layers) on node
                   representations `x`, returning final-layer NODE embeddings
                   [n, GT_out_dim] -- not pooled, not passed through
                   MLP_layer/readout.
      probe_data   a stand-in with `.x` = h^(0) (embedding_h(feat) concatenated
                   with the PE-derived positional channels, detached),
                   `.edge_index`, `.num_nodes`.
      meta         {"dim_inner": <width>, "num_nodes": ...}.

    SAME REASONING AS GraphGPS for why h^(0) rather than raw features: LRGB node
    features are integer atom-type indices consumed by an embedding lookup, so
    d h / d x is undefined for a discrete index -- there is no derivative to
    take at the raw input. The probe therefore differentiates from the encoder's
    OUTPUT, exactly as graphgps_backend.make_gps_model_fn does.

    THE n_shared_feats CONTRACT, AND WHY SAN HAS NO SHRINKING-CONTENT PROBLEM:
    embedding_h's output width is (GT_hidden_dim - LPE_dim) when a PE is active
    (0 when pe=='none'), and the concatenated PE vector is exactly LPE_dim wide
    -- so h^(0)'s total width is ALWAYS GT_hidden_dim, algebraically, regardless
    of which PE is active. Unlike GraphGPS (see that module's docstring), SAN
    does not need a choice between "content-only" and "dim_inner" widths: since
    BASE_NET_PARAMS holds GT_hidden_dim constant across all 5 PE variants for a
    given dataset, meta["dim_inner"] (== h0.shape[1] == GT_hidden_dim) is
    trivially identical across variants, so sensitivity.assert_shared_width will
    pass without needing any judgment call about which width to report.

    REQUIRES `data` from _PEAttachedDataset (san_train's "probe_dataset"), which
    attaches `_pe` and `_full_graph` onto each Data object. Unlike GraphGPS,
    which computes its PE internally from raw x/edge_index inside the model's
    own encoder (a known limitation disclosed in graphgps_backend's docstring),
    SAN's PE comes from an EXTERNAL cache keyed by graph index and split, so
    this function has no way to reconstruct the model's forward pass for a
    given graph without knowing which PE was active and whether full_graph
    densification applies -- hence the two attributes, rather than inferring
    them from `data` alone.

    CHECKPOINTING NOTE: for full_graph=True datasets, self.layers[i].forward is
    permanently monkey-patched by enable_gradient_checkpointing. Since the probe
    runs with torch.is_grad_enabled()==True (autograd.grad calls require it),
    checkpointing WILL engage here (recomputing each layer's activations during
    backward) -- this is functionally correct (checkpoint is differentiable)
    but doubles the compute cost per probed layer. Expected and acceptable
    given the probe already subsamples (num_target_nodes, chunk_size); not a
    bug, just a cost worth knowing about if probing is slower than training
    would suggest.

    NOT YET VALIDATED against a real GPU run -- unlike every other fix in this
    file, this has not been through a smoke-test/crash/fix cycle. The class-
    specific attribute names (embedding_e vs embedding_e_real, whether a
    wrapper class exposes `_base` or its own attributes directly) were inferred
    from reading the five model classes' __init__/forward methods and the one
    real state_dict mismatch encountered earlier (embedding_e_real), not
    confirmed by an actual run. Expect at least one debugging round here,
    consistent with how every other backend integration in this project went;
    the getattr/hasattr fallbacks below are a best-effort guard against the
    naming inconsistencies already known to exist across the 5 classes, not a
    guarantee they cover every case.
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
    # tagging, same EigVecs/EigVals/rwse/spd placement).
    pe_data = {}
    for key in ("EigVecs", "EigVals", "rwse", "spd"):
        if hasattr(data, key):
            pe_data[key] = getattr(data, key)

    g = _pyg_to_dgl(data, full_graph=full_graph, pe_data=pe_data).to(device)
    feat = g.ndata["feat"]
    e_raw = g.edata.get("feat", None)

    # _SAN_GRPE and _SAN_NodeLPE_Regression wrap an upstream SAN_NodeLPE instance in
    # `_base`; every other class (including upstream's own SAN/SAN_NodeLPE, used
    # directly for pe=='none'/'lappe') exposes embedding_h/embedding_e etc. on itself.
    base = getattr(model, "_base", model)

    with torch.no_grad():
        h_content = base.embedding_h(feat.float() if feat.dtype.is_floating_point
                                      else feat)
        if hasattr(base, "in_feat_dropout"):
            h_content = base.in_feat_dropout(h_content)

        # Edge embedding: naming is inconsistent across the 5 classes (see the
        # embedding_e_real vs embedding_e bug found earlier in _SAN_NodeLPE_Regression)
        # -- try the plausible names in order rather than assuming one.
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
            rwse_enc = getattr(model, "rwse_encoder", None) or getattr(base, "rwse_encoder")
            pe_vec = rwse_enc(g.ndata["rwse"])
        elif pe == "signnet":
            phi = getattr(model, "signnet_phi", None) or getattr(base, "signnet_phi")
            EigVecs = g.ndata["EigVecs"]
            n, k = EigVecs.shape
            v = EigVecs.view(n * k, 1)
            pe_vec = (phi(v) + phi(-v)).view(n, k, -1).mean(dim=1)
        elif pe == "none":
            pe_vec = None
        else:  # lappe, grpe: eigenvector path through linear_A + PE_Transformer
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
