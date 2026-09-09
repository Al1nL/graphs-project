"""
graphormer_backend.py
======================
Real integration with the Graphormer fork at ../Graphormer (config.UPSTREAM_PATHS["graphormer"]).

Two entry points, replacing the stubs in run_experiment.py:

    graphormer_train(run_cfg)              -> trains a cell of the grid, returns model + metrics
    make_graphormer_model_fn(model, item)  -> wraps a trained model for the Jacobian probe

Both drive Graphormer's own code (fairseq's `Trainer`, invoked the same way the
`fairseq-train` console script does) rather than reimplementing it -- mirrors
graphgps_backend.py's philosophy exactly: the point of the study is to vary the PE inside a
fixed backbone, so anything we reimplement is a confound. Training args are built from
Graphormer's own official example recipe (examples/property_prediction/zinc.sh -- the
closest official reference: small-molecule graph regression/classification), the same way
graphgps_train starts from GraphGPS's tuned reference YAML, with only the PE-related pieces
overridden per cell.

--------------------------------------------------------------------------------------
WHY GRAPHORMER NEEDS A GENUINE HOOK, NOT JUST CLI FLAGS -- READ BEFORE TRUSTING A PE ARM
--------------------------------------------------------------------------------------
Two of the five PE variants really are drop-in CLI flags, once one non-obvious problem is
fixed:

  * grpe: GRPE (Park et al., 2022) is explicitly a member of Graphormer's own family --
    Graphormer already has a `spatial_pos` / `edge_input` attention-bias slot. BUT stock
    Graphormer computes spatial_pos/edge features ITSELF, per item, via its own
    floyd_warshall (`graphormer/data/wrapper.py:preprocess_item`, backed by
    `data/algos.pyx`) -- independently of our shared `src/pe/compute_pe.py` cache. Left
    alone, that defeats the entire point of the harness (every backbone must see the SAME
    PE definition). `_CachedPEGraphormerDataset` below substitutes the cached, T5-bucketed
    `spd` (`dataset_meta.spd_bucket_id`) for Graphormer's own raw-distance embedding, and
    switches `--edge-type standard` so the bias comes from our single categorical
    `edge_type_id` rather than Graphormer's multi-hop path-feature machinery (which LRGB's
    scalar edge weights don't support anyway) -- matching what
    `adapters.graphormer_adapter.collate_spatial_and_edge` already documents as the design.

  * none: Graphormer's centrality encoding (in/out-degree, always on) is left untouched;
    "no PE" means literally nothing else is added. No hook needed.

The other three (lappe/rwse/signnet) are a genuine architectural addition, not a config
change -- Graphormer has no slot for a dense per-node PE, only its own degree-based
centrality term. They are summed into the node embedding via
`adapters.graphormer_adapter.ExtraNodePEProjection` (already written for exactly this),
wired in here through `_ExtraPEGraphNodeFeature` / `_ExtraPEGraphormerGraphEncoder` /
`_ExtraPEGraphormerEncoder` -- each a thin subclass that swaps in ONE component and leaves
the rest of the stock forward pass (attention bias, transformer layers) untouched, so the
comparison against the other four arms stays apples-to-apples. Registered as a new
`--arch graphormer_extra_pe_slim` rather than monkeypatching the stock `graphormer` model,
so nothing about the other four arms' code path changes.

--------------------------------------------------------------------------------------
WHERE THE JACOBIAN IS TAKEN, AND WHY (mirrors graphgps_backend.py's reasoning exactly)
--------------------------------------------------------------------------------------
LRGB node features are integer atom-type indices consumed by an nn.Embedding
(`GraphNodeFeature.atom_encoder`). d h / d x is undefined for a discrete index, so the probe
differentiates h^(0) = GraphNodeFeature's output (centrality encoding + PE already applied,
BEFORE the graph token is attended to by any transformer layer) -- the standard reading of
Di Giovanni et al., same as GraphGPS's "first representation the network actually starts
from".

--------------------------------------------------------------------------------------
THE INPUT-SPACE WIDTH STORY -- SIMPLER THAN GRAPHGPS's, BUT NOT FREE OF CAVEATS
--------------------------------------------------------------------------------------
GraphGPS CONCATENATES its PE, so content width shrinks per variant (80/76/64/96/96) and
`n_shared_feats` has to slice columns -- see that module's long docstring. Graphormer's
extra_node_pe instead ADDS the projected PE into the SAME `embedding_dim` channels
(`ExtraNodePEProjection`), so h^(0) has IDENTICAL width across all five arms with no
slicing needed: `n_shared_feats = embedding_dim` always, and `sensitivity.assert_shared_width`
is satisfied trivially.

The honest caveat this trades in: differentiating the FULL h^(0) for the extra_node_pe arms
means the measured sensitivity reflects centrality-encoding AND the added PE mixed into the
same directions -- there is no column-level way to isolate "sensitivity attributable to the
PE channels alone" the way there arguably almost is for GraphGPS's concatenation (mode (a)
of the input-space contract in sensitivity.py). This is Graphormer's version of "not free",
stated here rather than hidden, same as `probe_widths`'s docstring does for GraphGPS.

--------------------------------------------------------------------------------------
WHAT THIS FILE DOES NOT DO
--------------------------------------------------------------------------------------
PascalVOC-SP (node classification) is NOT wired up. Graphormer's stock `graph_prediction`
task assumes ONE label per graph: `TargetDataset` returns `item.y` as a single vector, the
collator concatenates one `y` per graph, and every criterion in `graphormer/criterions/`
reads out only the graph-token position (`logits[:, 0, :]`). Making Graphormer predict a
label per NODE needs a different task (masking/labels per node, dropping the graph-token
readout, a different collator) -- a real architectural project of its own, not a PE-arm
detail, and orthogonal to which PE is active. `graphormer_train` raises `NotImplementedError`
for it rather than silently training something that isn't the node-classification task LRGB
actually specifies. Peptides-func and Peptides-struct (both graph-level tasks) are fully
wired.
"""

import os
import resource
import sys
import time
import types
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# this file lives at <repo>/src/backends/graphormer_backend.py
_SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # <repo>/src
_REPO_ROOT = os.path.dirname(_SRC_ROOT)                                    # <repo>


# ---------------------------------------------------------------------------
# importing Graphormer (+ its vendored fairseq submodule)
# ---------------------------------------------------------------------------
def ensure_graphormer_importable(graphormer_dir: Optional[str] = None) -> str:
    """Put the Graphormer clone on sys.path and import it so its arch/task register.

    Two distinct failure modes, and each needs its own message:

    1. `fairseq` is a git SUBMODULE of the Graphormer repo (path "fairseq"), not a pip
       package fetched by a plain `git clone` (what setup_upstream.sh does -- it does not
       fetch submodules). An uninitialized submodule directory is EMPTY, but Python still
       lets `import fairseq` succeed silently as an implicit namespace package -- the
       failure only shows up later as a confusing `ModuleNotFoundError: No module named
       'fairseq.tasks'` deep inside training. This was hit and confirmed while wiring up
       this file: `import fairseq` returned cleanly with `fairseq.__file__ is None`, and
       only `import fairseq.tasks` exposed the real problem. So this checks for a real,
       non-empty package, not mere importability.
    2. `graphormer` itself -- registers "graphormer"/"graphormer_slim"/... architectures and
       the "graph_prediction" task into fairseq's registries. NOT a side effect of a plain
       `import graphormer` (verified empirically: graphormer/__init__.py only eagerly
       imports graphormer.criterions, leaving TASK_REGISTRY/ARCH_MODEL_REGISTRY untouched)
       -- done via `fairseq.utils.import_user_module`, the same call `--user-dir` on the
       CLI triggers, since that is what actually walks tasks/ and models/. Calling it here
       ourselves (rather than only passing --user-dir, as Graphormer's own examples do) is
       required because `_ensure_extra_pe_arch_registered` needs graphormer's model/module
       classes importABLE before fairseq's argument parser runs -- and doing both (this
       call, and --user-dir on the CLI) is safe: import_user_module memoizes by path, so
       the CLI's later pass over the same directory is a silent no-op. Doing the reverse
       (only --user-dir, plus a bare `import graphormer` anywhere first) is NOT safe: any
       import of `graphormer` or a submodule of it runs graphormer/__init__.py first,
       which claims the module NAME in sys.modules without performing this registration
       walk, and import_user_module then refuses to run at all ("not globally unique") --
       the exact crash this ordering avoids.
    """
    if graphormer_dir is None:
        sys.path.insert(0, _SRC_ROOT)
        from config import UPSTREAM_PATHS
        graphormer_dir = UPSTREAM_PATHS["graphormer"]
    graphormer_dir = os.path.abspath(graphormer_dir)
    if not os.path.isdir(os.path.join(graphormer_dir, "graphormer")):
        raise FileNotFoundError(
            f"no Graphormer clone at {graphormer_dir}. Run "
            f"`bash scripts/setup_upstream.sh graphormer`"
        )

    fairseq_pkg = os.path.join(graphormer_dir, "fairseq", "fairseq")
    if not os.path.isdir(fairseq_pkg) or not os.listdir(fairseq_pkg):
        raise FileNotFoundError(
            f"{graphormer_dir}/fairseq is empty -- the fairseq git submodule was never "
            f"initialized (a plain `git clone` does not fetch submodules). Run "
            f"`git -C {graphormer_dir} submodule update --init --recursive`, then install "
            "it into the `graphormer` conda env from within that directory: "
            "`LD_LIBRARY_PATH=$CONDA_PREFIX/lib pip install --no-build-isolation -e .` "
            "(needs Cython and a host compiler Graphormer's pinned torch==1.9.1+cu111 can "
            "actually compile against -- see envs/graphormer_env.yml's tail comment)."
        )

    try:
        # import fairseq BEFORE graphormer_dir touches sys.path: `Graphormer/fairseq` (the
        # submodule ROOT, containing setup.py, not the `fairseq/fairseq` package inside it)
        # has no __init__.py, so once graphormer_dir is on sys.path Python happily treats
        # it as an empty implicit namespace package -- and since sys.path.insert(0, ...)
        # puts it ahead of site-packages, it can SHADOW the real, pip-installed fairseq
        # (found this the same way as the earlier bug: `import fairseq.tasks` here failed
        # with "cannot import name 'metrics' from 'fairseq' (unknown location)" -- the
        # "unknown location" was the tell). Importing fairseq first, while only site-packages
        # is on the path, sidesteps it entirely; only `graphormer` itself needs graphormer_dir.
        import fairseq.tasks  # noqa: F401 -- forces the real package, not a namespace stub
        import argparse
        from fairseq.utils import import_user_module

        # NOT a plain `import graphormer` -- verified empirically that alone leaves
        # fairseq.tasks.TASK_REGISTRY and fairseq.models.ARCH_MODEL_REGISTRY untouched.
        # graphormer/__init__.py only eagerly imports graphormer.criterions; the
        # "graph_prediction" task and "graphormer"/"graphormer_slim" archs are registered
        # by fairseq.utils.import_user_module walking the tasks/ and models/ directories
        # itself (see fairseq/utils.py) -- exactly what passing --user-dir on the CLI
        # triggers. Calling it here, ourselves, does the SAME real work up front (needed
        # because _ensure_extra_pe_arch_registered has to import graphormer's model/module
        # classes before fairseq's own argument parser runs), while staying safe for
        # build_graphormer_args to ALSO pass --user-dir: import_user_module memoizes by
        # path (`import_user_module.memo`), so fairseq's own later call is a silent no-op
        # -- not calling it here and relying on --user-dir alone doesn't work in the other
        # direction, because a bare `import graphormer` (or any `graphormer.xxx` import,
        # which always runs graphormer/__init__.py first) claims the module NAME in
        # sys.modules without doing this registration walk, and import_user_module refuses
        # to run at all once that name is taken ("not globally unique") -- which is
        # exactly the crash this replaces.
        import_user_module(argparse.Namespace(user_dir=os.path.join(graphormer_dir, "graphormer")))
    except ImportError as exc:
        raise ImportError(
            f"Graphormer/fairseq at {graphormer_dir} could not be imported ({exc}). It "
            "needs its OWN environment -- torch==1.9.1+cu111, PyG==1.7.2, dgl==0.7.2, a "
            "compiled fairseq. See README 'Environment setup'; do not share one env across "
            "the three backbones."
        ) from exc
    return graphormer_dir


_EXTRA_PE_ARCH_REGISTERED = False


def _ensure_extra_pe_arch_registered():
    """Register `graphormer_extra_pe` / `graphormer_extra_pe_slim` into fairseq's model
    registry, once. Deferred to call time (rather than module import time) because it needs
    `graphormer` already imported (ensure_graphormer_importable), and fairseq's
    `register_model` raises on a duplicate registration -- calling this twice in one
    process must be a no-op, not a crash.
    """
    global _EXTRA_PE_ARCH_REGISTERED
    if _EXTRA_PE_ARCH_REGISTERED:
        return
    from fairseq.models import register_model, register_model_architecture
    from graphormer.models.graphormer import (
        GraphormerModel, GraphormerEncoder, base_architecture, graphormer_slim_architecture,
    )
    from graphormer.modules import GraphormerGraphEncoder
    from graphormer.modules.graphormer_layers import GraphNodeFeature
    from adapters.graphormer_adapter import ExtraNodePEProjection

    class _ExtraPEGraphNodeFeature(GraphNodeFeature):
        """GraphNodeFeature, plus a linear-projected shared-cache PE summed into the node
        embedding (see module docstring: "WHY GRAPHORMER NEEDS A GENUINE HOOK")."""

        def __init__(self, *args, extra_pe_dim, **kwargs):
            super().__init__(*args, **kwargs)
            self.pe_proj = ExtraNodePEProjection(extra_pe_dim, kwargs["hidden_dim"])

        def forward(self, batched_data):
            graph_node_feature = super().forward(batched_data)  # [B, T+1, C], incl. token
            pe = batched_data["extra_pe"]  # [B, T, pe_dim], padded, no graph-token row
            pe = F.pad(pe, (0, 0, 1, 0))  # zero row for the graph token -- it has no PE
            return self.pe_proj(graph_node_feature, pe)

    class _ExtraPEGraphormerGraphEncoder(GraphormerGraphEncoder):
        def __init__(self, *args, extra_pe_dim, **kwargs):
            super().__init__(*args, **kwargs)
            self.graph_node_feature = _ExtraPEGraphNodeFeature(
                num_heads=kwargs["num_attention_heads"],
                num_atoms=kwargs["num_atoms"],
                num_in_degree=kwargs["num_in_degree"],
                num_out_degree=kwargs["num_out_degree"],
                hidden_dim=kwargs["embedding_dim"],
                n_layers=kwargs["num_encoder_layers"],
                extra_pe_dim=extra_pe_dim,
            )

    class _ExtraPEGraphormerEncoder(GraphormerEncoder):
        def __init__(self, args):
            super().__init__(args)
            stock_kwargs = dict(
                num_atoms=args.num_atoms, num_in_degree=args.num_in_degree,
                num_out_degree=args.num_out_degree, num_edges=args.num_edges,
                num_spatial=args.num_spatial, num_edge_dis=args.num_edge_dis,
                edge_type=args.edge_type, multi_hop_max_dist=args.multi_hop_max_dist,
                num_encoder_layers=args.encoder_layers, embedding_dim=args.encoder_embed_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads, dropout=args.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.act_dropout,
                encoder_normalize_before=args.encoder_normalize_before,
                pre_layernorm=args.pre_layernorm,
                apply_graphormer_init=args.apply_graphormer_init,
                activation_fn=args.activation_fn,
            )
            self.graph_encoder = _ExtraPEGraphormerGraphEncoder(
                extra_pe_dim=args.extra_pe_dim, **stock_kwargs
            )

    @register_model("graphormer_extra_pe")
    class _ExtraPEGraphormerModel(GraphormerModel):
        @staticmethod
        def add_args(parser):
            GraphormerModel.add_args(parser)
            parser.add_argument(
                "--extra-pe-dim", type=int,
                help="width of the concatenated-as-node-feature PE (lappe=16, rwse=20, "
                     "signnet=16 -- see src/pe/compute_pe.py K_LAP/K_RWSE)",
            )

        @classmethod
        def build_model(cls, args, task):
            base_architecture(args)
            if not hasattr(args, "max_nodes"):
                args.max_nodes = args.tokens_per_sample
            encoder = _ExtraPEGraphormerEncoder(args)
            return cls(args, encoder)

    @register_model_architecture("graphormer_extra_pe", "graphormer_extra_pe_slim")
    def _extra_pe_slim_architecture(args):
        graphormer_slim_architecture(args)

    _EXTRA_PE_ARCH_REGISTERED = True


# ---------------------------------------------------------------------------
# dataset wiring: substitute the shared PE cache for Graphormer's own on-the-fly PE
# ---------------------------------------------------------------------------
_PE_CACHE_FIELD = {"lappe": "lap_pe", "rwse": "rwse", "signnet": "signnet_in"}

# How often _CachedPEGraphormerDataset.__getitem__ prints a progress/memory line while
# filling its cache. Existing purely because the run that motivated this file's memory
# comment (see __getitem__ below) died silently 11.5h in with zero output between "training
# X for real" and either a fairseq log line or a crash -- there was no way to tell, after
# the fact, whether it was still crawling through preprocessing or already deadlocked/OOM.
# 200, not something finer: ~55 lines over a ~10.9k-graph split is enough to see the rate
# and RSS trend without flooding slurm-<jobid>.out.
_PE_CACHE_PROGRESS_EVERY = 200


def _log_pe_cache_progress(split_name: str, n_cached: int, n_total: int, t_start: float) -> None:
    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)  # KB -> GB
    elapsed_min = (time.time() - t_start) / 60
    print(
        f"[graphormer_backend] {split_name}: preprocessed {n_cached}/{n_total} graphs "
        f"-- peak RSS so far: {peak_rss_gb:.1f} GB, elapsed: {elapsed_min:.1f} min",
        flush=True,
    )


def _make_cached_pe_dataset_cls():
    """Built lazily (needs `graphormer` importable first) -- returns the
    `_CachedPEGraphormerDataset` class, a `GraphormerPYGDataset` whose `spatial_pos` /
    `attn_edge_type` (for grpe) or `.extra_pe` (for lappe/rwse/signnet) come from the shared
    cache instead of Graphormer's own `preprocess_item`.

    Subclassing rather than monkeypatching `graphormer.data.wrapper.preprocess_item`
    deliberately: `pyg_dataset.py` does `from ..wrapper import preprocess_item` at import
    time, a direct name binding that a later monkeypatch of `wrapper.preprocess_item` would
    NOT see -- subclassing is the mechanism that actually reaches the call site.
    """
    from graphormer.data.pyg_datasets.pyg_dataset import GraphormerPYGDataset
    from graphormer.data.wrapper import convert_to_single_emb
    from graphormer.data import algos
    from dataset_meta import spd_bucket_id  # noqa: F401 -- documents where spd_bucket comes from

    def _fast_preprocess_item(item):
        """`graphormer.data.wrapper.preprocess_item`, minus the one call that made every
        epoch take hours: `algos.gen_edge_input(max_dist, path, ...)`, where `max_dist` is
        the graph's ACTUAL diameter (not the 5-hop window the model ever looks at -- see
        `_CachedPEGraphormerDataset.__getitem__`, which is the only caller of this). Every
        other field (x, spatial_pos, attn_bias, degrees) is computed identically to stock;
        `edge_input` becomes a cheap placeholder shaped like collator.py expects, since
        GraphAttnBias's standard/non-multi_hop branch (this harness's --edge-type for
        every PE arm) never reads it.
        """
        edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
        n = x.size(0)
        x = convert_to_single_emb(x)

        adj = torch.zeros([n, n], dtype=torch.bool)
        adj[edge_index[0, :], edge_index[1, :]] = True

        if len(edge_attr.size()) == 1:
            edge_attr = edge_attr[:, None]
        attn_edge_type = torch.zeros([n, n, edge_attr.size(-1)], dtype=torch.long)
        attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
            convert_to_single_emb(edge_attr) + 1
        )

        shortest_path_result, _path = algos.floyd_warshall(adj.numpy())  # fast: ~0.3s even at n=434

        item.x = x
        item.attn_bias = torch.zeros([n + 1, n + 1], dtype=torch.float)
        item.attn_edge_type = attn_edge_type
        item.spatial_pos = torch.from_numpy(shortest_path_result).long()
        item.in_degree = adj.long().sum(dim=1).view(-1)
        item.out_degree = item.in_degree
        item.edge_input = torch.zeros([n, n, 1, edge_attr.size(-1)], dtype=torch.long)
        return item

    class _CachedPEGraphormerDataset(GraphormerPYGDataset):
        def __init__(self, *args, pe_cache=None, pe_name="none", **kwargs):
            # Set BEFORE super().__init__(), not after: with the train_set/valid_set/
            # test_set constructor path, GraphormerPYGDataset.__init__ itself calls
            # create_subset() -> copy.copy(self) to build .train_data/.valid_data/
            # .test_data -- if _pe_cache/_pe_name were only set on `self` afterward (as
            # they were originally here), the shallow copies made during super().__init__()
            # would predate them and end up without either attribute at all.
            self._pe_cache = pe_cache
            self._pe_name = pe_name
            self._item_cache = {}
            self._split_name = "?"  # overwritten per-subset by _register_dataset below
            self._t0 = time.time()
            super().__init__(*args, **kwargs)

        def __getitem__(self, idx):
            idx = int(idx)
            # Persistent (unbounded, whole-lifetime-of-this-object) cache -- NOT the stock
            # @lru_cache(maxsize=16) that GraphormerPYGDataset.__getitem__ (called via
            # super() below) already has. maxsize=16 is fine for a single batch, but
            # fairseq's Trainer iterates the FULL split once per epoch, so with e.g. 10,873
            # training graphs and --max-epoch 20, a 16-slot cache buys almost nothing:
            # every item is preprocessed from scratch roughly once PER EPOCH -- 20x more
            # Floyd-Warshall (Cython, O(n^3), some Peptides-func graphs run to 444 nodes)
            # and PECache disk reads than necessary. Measured directly on this cluster: one
            # epoch took ~4.6 hours on a dedicated GPU node with zero other load, and it
            # was entirely CPU-bound preprocessing (0% GPU utilization throughout) -- this
            # cache is what turns "recompute every epoch" into "recompute once, reuse 19x".
            # Costs real memory instead (a full preprocessed item, dominated by
            # `edge_input` at up to ~24 MB for a 444-node graph): budget accordingly in
            # whatever launches training (see scripts/slurm_graphormer_calibrate.sh's
            # --mem).
            cached = self._item_cache.get(idx)
            if cached is not None:
                return cached
            # NOT super().__getitem__(idx): that calls stock preprocess_item
            # (graphormer/data/wrapper.py), which unconditionally calls
            # algos.gen_edge_input(max_dist, path, ...) with max_dist = the graph's ACTUAL
            # diameter (np.amax(shortest_path_result)) -- not multi_hop_max_dist (5), which
            # only trims the result at BATCH time (collator.py:77). Measured directly on
            # this cluster: on a 434-node Peptides-func graph with diameter 159,
            # gen_edge_input alone took 18.7s and produced a 719 MB array, versus 0.28s for
            # floyd_warshall itself -- gen_edge_input's naive recursive path
            # reconstruction (get_all_edges in algos.pyx) is what actually burned every
            # multi-hour epoch across five separate SLURM attempts, not Floyd-Warshall and
            # not a lack of cross-epoch caching (that fix was real but nowhere near the
            # dominant cost). Every PE arm in this harness runs `--edge-type standard`
            # (see build_graphormer_args), and GraphAttnBias's standard/non-multi_hop
            # branch never reads `edge_input` at all -- so this result was computed, at
            # enormous cost, for a value nothing downstream uses. `_fast_preprocess_item`
            # below is preprocess_item with exactly that one call removed, replaced with a
            # cheap placeholder shaped like collator.py expects.
            raw = self.dataset[idx]
            raw.idx = idx
            raw.y = raw.y.reshape(-1)
            item = _fast_preprocess_item(raw)
            if self._pe_cache is not None:
                rec = self._pe_cache[idx]
                if self._pe_name == "grpe":
                    n = item.spatial_pos.size(0)
                    spd_bucket = np.asarray(rec["spd_bucket"])[:n, :n]
                    edge_type = np.asarray(rec["edge_type_id"])[:n, :n]
                    item.spatial_pos = torch.from_numpy(spd_bucket.astype(np.int64))
                    item.attn_edge_type = torch.from_numpy(
                        edge_type.astype(np.int64)
                    ).unsqueeze(-1)
                elif self._pe_name in _PE_CACHE_FIELD:
                    pe = np.asarray(rec[_PE_CACHE_FIELD[self._pe_name]])
                    item.extra_pe = torch.from_numpy(pe.astype(np.float32))
            self._item_cache[idx] = item
            n_cached = len(self._item_cache)
            if n_cached == 1 or n_cached % _PE_CACHE_PROGRESS_EVERY == 0:
                _log_pe_cache_progress(self._split_name, n_cached, len(self), self._t0)
            return item

        # index_select/create_subset (inherited) return plain GraphormerPYGDataset copies
        # via `copy.copy(self)` -- a SHALLOW copy, so without help every subset (train_data/
        # valid_data/test_data) would share the SAME `_item_cache` dict by reference, and
        # idx 5 in train_data would collide with idx 5 in test_data (different graphs, same
        # key). _register_dataset gives each subset its own fresh dict explicitly, the same
        # way it already does for `_pe_cache` (a different PECache per split).

    return _CachedPEGraphormerDataset


def _make_cached_pe_batched_dataset_cls():
    """`BatchedDataDataset`, extended to also batch `.extra_pe` when present. Stock
    `collator.pad_2d_unsqueeze` adds +1 before padding (it is designed for embedding
    INDICES, where 0 must be free for the padding token) -- wrong for a continuous PE
    tensor, so this pads with plain zeros instead.
    """
    from graphormer.data.dataset import BatchedDataDataset
    from graphormer.data.collator import collator as _stock_collator

    def _pad_float2d_unsqueeze(x, padlen):
        xlen, xdim = x.size()
        if xlen < padlen:
            new_x = x.new_zeros([padlen, xdim])
            new_x[:xlen, :] = x
            x = new_x
        return x.unsqueeze(0)

    class _CachedPEBatchedDataset(BatchedDataDataset):
        def collater(self, samples):
            batched = _stock_collator(
                samples, max_node=self.max_node,
                multi_hop_max_dist=self.multi_hop_max_dist,
                spatial_pos_max=self.spatial_pos_max,
            )
            # Mirror _stock_collator's OWN oversized-graph filter exactly (it silently
            # drops any item with item.x.size(0) > self.max_node before computing its
            # batch, via the same condition below) -- without this, a graph the stock
            # collator dropped still had its (un-truncated, larger) extra_pe padded
            # against `max_node_num` computed from the SMALLER, already-filtered batch,
            # which raises "Sizes of tensors must match ... in dimension 1" the first time
            # a batch actually contains one (hit on a real Peptides-func run: default
            # --max-nodes=128 drops plenty of its graphs, up to 444 nodes).
            kept = [s for s in samples if s.x.size(0) <= self.max_node]
            if kept and hasattr(kept[0], "extra_pe"):
                max_node_num = batched["x"].size(1)
                batched["extra_pe"] = torch.cat(
                    [_pad_float2d_unsqueeze(s.extra_pe, max_node_num) for s in kept]
                )
            return batched

    return _CachedPEBatchedDataset


def _register_dataset(run_cfg, graphormer_dir, max_graphs_per_split=None):
    """Load the three LRGB splits (same `LRGBDataset` call as src/pe/compute_pe.py, so
    graph order -- and therefore the index the PE cache was written under -- matches
    exactly) and make `--dataset-name <run_cfg.dataset> --dataset-source pyg` resolve to
    them, wrapped with the cached-PE dataset class above.

    NOT wired through `graphormer.data.DATASET_REGISTRY` / `--user-data-dir`, despite that
    being the officially documented "customized dataset" mechanism (see
    examples/customized_dataset) -- that path only reaches
    `GraphormerDataset(dataset=..., train_idx=..., valid_idx=..., test_idx=...)`, i.e. ONE
    combined PyG dataset plus three index arrays, not three already-separate LRGBDataset
    splits. Forcing our data into that shape means either concatenating three PyG datasets
    (uncertain whether LRGBDataset's `Dataset.index_select` supports it cleanly) or fighting
    the framework. Instead this patches `PYGDatasetLookupTable.GetPYGDataset` -- the
    function `GraphormerDataset.__init__` itself calls, one level below the registry, which
    (per graphormer/data/pyg_datasets/pyg_dataset.py) takes train_set/valid_set/test_set
    directly. Same effect, no `--user-data-dir` needed, and no shape mismatch.
    """
    from torch_geometric.datasets import LRGBDataset
    from graphormer.data.pyg_datasets.pyg_dataset_lookup_table import PYGDatasetLookupTable
    from pe.cache import PECache

    dataset_cls = _make_cached_pe_dataset_cls()
    pyg_name = {"peptides-func": "Peptides-func", "peptides-struct": "Peptides-struct"}[
        run_cfg.dataset
    ]
    cache_dir = run_cfg.resolved_cache_dir

    # Load the three raw LRGB splits with the SAME call src/pe/compute_pe.py used, so graph
    # order -- and therefore the index the PE cache was written under -- lines up exactly.
    raw_by_split = {
        pyg_split: LRGBDataset(root=f"raw_data/{pyg_name}", name=pyg_name, split=pyg_split)
        for pyg_split in ("train", "val", "test")
    }
    for pyg_split, ds in raw_by_split.items():
        print(f"[graphormer_backend] loaded {pyg_split} split: {len(ds)} graphs", flush=True)
    if max_graphs_per_split is not None:
        # Debugging/smoke-test knob ONLY (None in every real run) -- truncates to a
        # PREFIX of each split, not a random sample, so index i in the truncated set is
        # still index i in the full split and the PE cache (indexed positionally, see
        # src/pe/cache.py) stays aligned without any remapping.
        raw_by_split = {
            split: ds.index_select(list(range(min(max_graphs_per_split, len(ds)))))
            for split, ds in raw_by_split.items()
        }
    pe_cache_by_split = {
        pyg_split: (PECache(cache_dir, pyg_split) if run_cfg.pe != "none" else None)
        for pyg_split in ("train", "val", "test")
    }

    # GraphormerPYGDataset's train_set/valid_set/test_set constructor path builds all three
    # subsets in one call via `copy.copy(self)` (see pyg_datasets/pyg_dataset.py), which
    # would leave all three sharing one `_pe_cache` -- train/valid/test each need their OWN
    # PECache (a different split's files), so it is set per-subset right after construction.
    wrapped = dataset_cls(
        dataset=None,
        train_set=raw_by_split["train"],
        valid_set=raw_by_split["val"],
        test_set=raw_by_split["test"],
        pe_cache=None, pe_name=run_cfg.pe,
    )
    wrapped.train_data._pe_cache = pe_cache_by_split["train"]
    wrapped.valid_data._pe_cache = pe_cache_by_split["val"]
    wrapped.test_data._pe_cache = pe_cache_by_split["test"]
    # Fresh per-split caches -- see _CachedPEGraphormerDataset's index_select/create_subset
    # note above: without this, the three splits would share one dict by reference and
    # collide on index.
    wrapped.train_data._item_cache = {}
    wrapped.valid_data._item_cache = {}
    wrapped.test_data._item_cache = {}
    # Labels for _log_pe_cache_progress's output -- otherwise every split's progress line
    # would print the same unhelpful "?" and there'd be no way to tell from the log which
    # split (train, 20x more epochs of exposure, vs. one-off val/test) is being built.
    wrapped.train_data._split_name = "train"
    wrapped.valid_data._split_name = "val"
    wrapped.test_data._split_name = "test"

    # GraphormerDataset.__init__ (data/dataset.py), when dataset_source="pyg" and no
    # `dataset=` object is given directly, resolves the CLI's `--dataset-name` through
    # `PYGDatasetLookupTable.GetPYGDataset(dataset_spec, seed)`. Patching the staticmethod
    # on the class itself (not a name in some importing module's namespace) is safe here
    # because every caller holds a reference to this SAME class object -- unlike
    # BatchedDataDataset just below, there is no separate rebinding to chase. Falls through
    # to the original lookup for any other dataset name (there is none, in this harness,
    # but this keeps the patch narrowly scoped rather than replacing the whole table).
    # GetPYGDataset is a @staticmethod -- accessed via the class it is already the plain
    # function (unlike a regular method, a staticmethod descriptor has no bound-method
    # wrapper to unwrap with .__func__; that was the bug here originally).
    _orig_get_pyg_dataset = PYGDatasetLookupTable.GetPYGDataset
    target_name = run_cfg.dataset

    def _patched_get_pyg_dataset(dataset_spec, seed):
        if dataset_spec == target_name:
            return wrapped
        return _orig_get_pyg_dataset(dataset_spec, seed)

    PYGDatasetLookupTable.GetPYGDataset = staticmethod(_patched_get_pyg_dataset)

    # GraphPredictionTask.load_dataset hardcodes `BatchedDataDataset(...)` -- it imports
    # that name directly into graphormer.tasks.graph_prediction's namespace, and looks it
    # up there at call time, so patching the name IN THAT NAMESPACE (not the defining
    # module) is what actually reaches the call site. Applied unconditionally: the
    # replacement's collater is a strict superset of stock behaviour (only adds `extra_pe`
    # when the batch's items actually carry it), so it is a no-op for the other PE arms.
    import graphormer.tasks.graph_prediction as _gp_task_module
    _gp_task_module.BatchedDataDataset = _make_cached_pe_batched_dataset_cls()

    return wrapped


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
# Optimizer/scheduler/LR hyperparameters below are carried from Graphormer's OWN reference
# recipe for a small-molecule graph task, examples/property_prediction/zinc.sh -- the same
# "start from the backbone's own tuned config, override only the PE-relevant pieces" policy
# graphgps_backend.py follows for its GraphGym YAML. UNLIKE that YAML, this has not been
# re-tuned against LRGB Peptides specifically (no GPU here to do so yet -- see README
# "Status note"); treat --lr/--warmup-updates/--max-epoch as a documented starting point,
# not a validated choice.
TASK_CRITERION = {"peptides-func": "binary_logloss", "peptides-struct": "l1_loss"}
# LRGB's own task definition: Peptides-func is 10-task multi-label classification,
# Peptides-struct is 11-target regression (Dwivedi et al. 2022, LRGB paper, Table 1).
TASK_NUM_CLASSES = {"peptides-func": 10, "peptides-struct": 11}

PE_EXTRA_ARCH = {"lappe", "rwse", "signnet"}
# must match src/pe/compute_pe.py K_LAP/K_RWSE. signnet is 16, NOT the 32 this said until
# it caused "mat1 and mat2 shapes cannot be multiplied (Nx16 and 32x80)" on the very first
# training step of every signnet cell: cache.py's PECache derives "signnet_in" as
# node[:, :k_lap] -- the exact same K_LAP=16 slice as lap_pe (SignNet's raw eigenvector
# input, see compute_pe.py's docstring: "signnet_in ... = lap_pe"), not the unrelated
# out_dim=32 of compute_pe.py's SignNetEncoder (a different, already-learned-through
# encoder that this harness's Graphormer path never uses -- it feeds the raw eigenvectors
# straight into ExtraNodePEProjection's own Linear instead).
PE_DIM = {"lappe": 16, "rwse": 20, "signnet": 16}


def build_graphormer_args(run_cfg, graphormer_dir):
    """Build the fairseq-cli argv list for one grid cell -- the Graphormer analogue of
    graphgps_backend.build_graphgym_cfg. Returns List[str], consumed by
    `options.parse_args_and_arch` exactly as `fairseq-train`'s own argv would be.
    """
    import json

    cfg_path = os.path.join(
        _REPO_ROOT, "configs", "graphormer",
        f"graphormer_{run_cfg.pe}_{run_cfg.dataset}.json",
    )
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"no reference config at {cfg_path}")
    with open(cfg_path) as f:
        base = json.load(f)
    # base["pe_cache"] names a stale per-split-blob path from before cache.py's fix-3
    # rewrite to one-mmap-file-per-graph; NOT used here -- _register_dataset resolves the
    # real cache location from run_cfg.resolved_cache_dir via pe.cache.PECache instead.

    save_dir = os.path.join(run_cfg.results_dir, "raw", run_cfg.run_id)
    arch = "graphormer_extra_pe_slim" if run_cfg.pe in PE_EXTRA_ARCH else "graphormer_slim"

    args = [
        "--user-dir", os.path.join(graphormer_dir, "graphormer"),
        # 0, not fairseq's typical 4+: this dataset (like every torch_geometric
        # InMemoryDataset) materialises the WHOLE split as one big in-memory tensor blob at
        # construction time -- for Peptides-func train, ~11k graphs' worth. Each
        # --num-workers process is a fork() of the main process, and forked children start
        # as copy-on-write over that blob, but CPython's reference counting touches nearly
        # every object it accesses (incrementing/decrementing refcounts on Python objects
        # wrapping the tensors), which dirties the underlying pages one by one and defeats
        # COW sharing in practice -- so each worker's RSS creeps toward a FULL separate
        # copy of the dataset, not a fraction of it. Measured directly on this cluster:
        # --num-workers 4 pushed one run to ~33 GB RSS combined (worker + main), which
        # filled the shared machine's swap outright (58 concurrent users; not a private
        # GPU box) and looked like a stalled/crashed SSH session from the outside. 0 keeps
        # everything in the main process -- slower per-batch, but the only setting that
        # doesn't risk taking the shared machine down. Raise this only on a machine you
        # are not sharing, and watch `free -h` while you do.
        "--num-workers", "0",
        # Explicit, not left to fairseq's default (torch.cuda.device_count()): this
        # cluster has 4 GPUs, and distributed_utils.call_main only takes the plain
        # single-process path (`main(cfg)` directly) when distributed_world_size == 1.
        # Otherwise it spawns one process PER GPU via torch.multiprocessing.spawn -- and
        # spawned (not forked) child processes re-run every import from scratch, which
        # re-triggers fairseq's own eager import of every built-in model (hubert,
        # wav2vec2, ...) in a fresh interpreter and reproduces the exact
        # "cannot import name 'metrics' from 'fairseq'" shadowing failure this file
        # already works around for the parent process -- in the child, differently.
        # Nothing about this harness needs multi-GPU training, so the simplest fix is to
        # never take that path at all.
        "--distributed-world-size", "1",
        "--ddp-backend", "legacy_ddp",
        "--dataset-name", run_cfg.dataset,
        "--dataset-source", "pyg",
        "--task", "graph_prediction",
        "--criterion", TASK_CRITERION[run_cfg.dataset],
        "--arch", arch,
        "--num-classes", str(TASK_NUM_CLASSES[run_cfg.dataset]),
        "--attention-dropout", str(base["attention_dropout"]),
        "--act-dropout", "0.1",
        "--dropout", str(base["dropout"]),
        "--optimizer", "adam", "--adam-betas", "(0.9, 0.999)", "--adam-eps", "1e-8",
        "--clip-norm", "5.0", "--weight-decay", "0.01",
        "--lr-scheduler", "inverse_sqrt", "--warmup-updates", str(base["warmup_updates"]),
        "--lr", str(base["lr"]),
        # 8, not the reference recipe's 32 (zinc.sh, molecules averaging ~23 atoms): LRGB
        # Peptides graphs run up to 444 nodes, and attention/edge-bias tensors scale with
        # batch_size x n^2 -- 32 x 444^2 was part of what OOM'd a 12 GB GPU here (see the
        # --edge-type note above for the other, larger part of it).
        "--batch-size", "8",
        "--encoder-layers", str(base["num_layers"]),
        "--encoder-embed-dim", str(base["embed_dim"]),
        "--encoder-ffn-embed-dim", str(base["ffn_embed_dim"]),
        "--encoder-attention-heads", str(base["attention_heads"]),
        "--max-epoch", str(run_cfg.epochs if run_cfg.epochs is not None else base["max_epoch"]),
        # Graphormer's own default (128, GraphPredictionConfig.max_nodes) silently DROPS
        # any graph bigger than that from every batch (see _CachedPEBatchedDataset.collater
        # above) rather than erroring -- LRGB Peptides-func's largest graph measured 444
        # nodes across all three splits, so the stock default would quietly train on a
        # biased subset (small graphs only) without any warning. 512 is round headroom
        # above the measured max for every dataset this harness uses.
        "--max-nodes", "512",
        "--best-checkpoint-metric", "loss",
        "--save-dir", save_dir,
        "--seed", str(run_cfg.seed),
    ]

    # --edge-type standard for EVERY pe, not just grpe: "multi_hop" (Graphormer's default,
    # and what the reference zinc.sh recipe uses) decomposes the shortest path between every
    # node pair into a sequence of up to `multi_hop_max_dist` bond-level edge features --
    # meaningful for molecular graphs with real bond-type paths, but adapters.graphormer_
    # adapter.collate_spatial_and_edge already documents that LRGB has no such thing ("LRGB
    # edge features are scalar weights, not multi-hop bond paths"). It is also what actually
    # OOM'd a 12 GB GPU here: GraphAttnBias's multi_hop branch materialises several
    # [batch, n, n, multi_hop_max_dist, num_heads] tensors, and at n up to 444 (Peptides-
    # func's largest graph) those run into the multiple-GB range EACH, well before
    # accounting for the rest of the model's activations. "standard" uses the single
    # categorical edge_type_id path instead (GraphAttnBias's non-multi_hop branch) -- both
    # cheaper and the mode collate_spatial_and_edge was actually designed around.
    args += ["--edge-type", "standard"]
    if run_cfg.pe == "grpe":
        from dataset_meta import SPD_NUM_BUCKETS
        # +1: the stock collator's pad_spatial_pos_unsqueeze adds 1 to every real distance
        # before padding, so index 0 stays free for padding -- num_spatial must cover that
        # shift, or a real bucket at the top of the range collides with padding_idx=0.
        args += ["--num-spatial", str(SPD_NUM_BUCKETS + 1), "--num-edges", "2"]

    if run_cfg.pe in PE_EXTRA_ARCH:
        args += ["--extra-pe-dim", str(PE_DIM[run_cfg.pe])]

    return args


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def graphormer_train(run_cfg, graphormer_dir: Optional[str] = None,
                      max_graphs_per_split: Optional[int] = None) -> dict:
    """Train one grid cell with Graphormer's own pipeline (fairseq's Trainer).

    Returns {"model", "task", "test_dataset", "num_params", "metric_name", "metric_value",
    "cfg"}. `test_dataset` is the cached-PE-wrapped test split (indexable: `test_dataset[i]`
    yields one Graphormer-preprocessed item, ready for `make_graphormer_model_fn`) -- handed
    back directly rather than making a caller dig it out of `task`/fairseq's dataset
    machinery, which is exactly what `scripts/calibrate_target_nodes.py`'s `load_real`
    needs to sample calibration graphs.

    `max_graphs_per_split` is a smoke-test knob (None in every real run): truncates each
    split to its first N graphs so the whole train -> checkpoint -> evaluate chain can be
    validated in minutes instead of however long a real epoch over the full split takes on
    a busy shared machine. See `_register_dataset`.
    """
    if run_cfg.dataset == "pascalvoc-sp":
        raise NotImplementedError(
            "PascalVOC-SP is node classification; Graphormer's stock `graph_prediction` "
            "task and every shipped criterion assume ONE label per graph (they all read "
            "out logits[:, 0, :], the graph-token position only). Predicting a label per "
            "node needs a different task/collator, not a PE-arm change -- see this file's "
            "module docstring, 'WHAT THIS FILE DOES NOT DO'. Left as a separate pass, "
            "deliberately, the same way graphgps_backend.py declines GraphGPS+GRPE."
        )

    graphormer_dir = ensure_graphormer_importable(graphormer_dir)
    if run_cfg.pe in PE_EXTRA_ARCH:
        _ensure_extra_pe_arch_registered()

    from fairseq import options
    from fairseq.dataclass.utils import convert_namespace_to_omegaconf
    from fairseq_cli import train as fairseq_train
    from fairseq.distributed import utils as distributed_utils
    from fairseq.utils import set_torch_seed
    from torch_geometric import seed_everything

    seed_everything(run_cfg.seed)
    set_torch_seed(run_cfg.seed)

    # Force OFF, regardless of what the caller set beforehand: scripts/launch.py's
    # seed_everything() sets cudnn.deterministic=True AND os.environ["CUBLAS_WORKSPACE_
    # CONFIG"] = ":4096:8" (for reproducibility) before dispatching to ANY backbone.
    #
    # Resetting only the cudnn flag (an earlier version of this fix) did NOT resolve the
    # hang -- SLURM job 818716 ran launch.py end-to-end with that fix in place and still
    # froze at the identical point after 39+ minutes. A faulthandler/SIGUSR1 stack trace
    # taken directly from the frozen process (job 819320, 2026-08-30) showed it stuck
    # inside a single call in fairseq's Adam.step() (adam.py:225, a plain elementwise
    # exp_avg.mul_/add_) -- and a SECOND SIGUSR1 sent minutes later produced no new dump
    # at all, meaning the interpreter never returned to the bytecode loop to service it.
    # That is a genuine low-level CUDA/cuBLAS freeze inside one C call, not merely a slow
    # step. The same freeze reproduced on two different nodes (s-003, s-006), ruling out
    # a single bad GPU, and scripts/graphormer_base_check.py -- the one script that never
    # calls seed_everything at all, so CUBLAS_WORKSPACE_CONFIG is never set -- is the one
    # run that has always completed cleanly.
    #
    # CUBLAS_WORKSPACE_CONFIG=":4096:8" restricts cuBLAS to a ~32KB workspace pool; it is
    # read directly by the CUDA runtime the first time it lazily creates a GEMM
    # workspace, independent of whether torch.use_deterministic_algorithms itself
    # actually engaged (on this torch==1.9.1 pin that call fails outright with
    # "unexpected keyword argument 'warn_only'", so PyTorch's own deterministic-algorithms
    # flag was never really turned on here -- only the raw env var was left behind).  A
    # workspace that small is a known trigger for exactly this kind of silent freeze on
    # large GEMMs (e.g. this dataset's attention matrices for its biggest graphs) on old
    # cuBLAS/driver combinations. Removing it before any CUDA op in this process has
    # created a cuBLAS handle is the fix.
    #
    # Seeds (torch_geometric.seed_everything/set_torch_seed above) still make this run
    # reproducible in the sense that matters for this project -- weight init and data
    # order -- just not bit-exact GPU kernel selection, which was never being achieved
    # anyway.
    import torch as _torch
    _torch.backends.cudnn.deterministic = False
    _torch.backends.cudnn.benchmark = False
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    try:
        _torch.use_deterministic_algorithms(False)
    except Exception:  # noqa: BLE001 - best-effort; absence of the flag is the goal
        pass

    wrapped_dataset = _register_dataset(run_cfg, graphormer_dir, max_graphs_per_split)

    argv = build_graphormer_args(run_cfg, graphormer_dir)
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, input_args=argv)
    cfg = convert_namespace_to_omegaconf(args)

    # NOT fairseq_train.main(cfg) directly -- the `fairseq-train` console script's
    # cli_main() never calls main() by itself either; it always goes through
    # distributed_utils.call_main(cfg, main), which sets up the (possibly trivial,
    # single-process) torch.distributed process group first. Skipping that -- as this did
    # until it was caught by an actual run -- fails deep inside Trainer with
    # "RuntimeError: Default process group has not been initialized", since fairseq's
    # Trainer unconditionally queries the distributed group even for one process.
    distributed_utils.call_main(cfg, fairseq_train.main)

    from fairseq import checkpoint_utils, tasks
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    ckpt_path = os.path.join(cfg.checkpoint.save_dir, "checkpoint_best.pt")
    state = checkpoint_utils.load_checkpoint_to_cpu(ckpt_path)
    model.load_state_dict(state["model"], strict=True, model_cfg=cfg.model)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    if n_params > 500_000:
        print(f"  WARNING: {n_params:,} parameters exceeds the 500k budget in the proposal")

    metric_value = _evaluate_metric(cfg, task, model, "test", run_cfg.metric_name)

    return {
        "model": model, "task": task, "test_dataset": wrapped_dataset.test_data,
        "num_params": n_params, "metric_name": run_cfg.metric_name,
        "metric_value": metric_value, "cfg": cfg,
    }


def _evaluate_metric(cfg, task, model, split: str, metric_name: str) -> Optional[float]:
    """Compute the PAPER's metric (AP or MAE) post-hoc over a split, the same way
    `graphormer/evaluate/evaluate.py` does -- fairseq's stock criteria only log training
    loss (and, for binary_logloss, a THRESHOLDED accuracy, not average precision), so AP
    specifically cannot be read off a fairseq log/checkpoint the way GraphGPS's stats.json
    is read in `graphgps_backend._read_best_metric`. This re-derives it from raw
    predictions instead, mirroring evaluate.py's `eval()` loop exactly.
    """
    from fairseq import utils
    from sklearn.metrics import average_precision_score

    task.load_dataset(split)
    itr = task.get_batch_iterator(
        dataset=task.dataset(split), max_sentences=32,
        max_positions=utils.resolve_max_positions(task.max_positions(), model.max_positions()),
    ).next_epoch_itr(shuffle=False)

    y_pred, y_true = [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for sample in itr:
            sample = utils.move_to_cuda(sample) if device.type == "cuda" else sample
            logits = model(**sample["net_input"])[:, 0, :]
            y_pred.append(logits.cpu())
            y_true.append(sample["target"].cpu().reshape(logits.shape))
    if not y_pred:
        return None
    y_pred, y_true = torch.cat(y_pred), torch.cat(y_true)

    if metric_name == "ap":
        mask = ~torch.isnan(y_true)
        aps = []
        for t in range(y_true.shape[1]):
            col = mask[:, t]
            if col.any() and y_true[col, t].unique().numel() > 1:
                aps.append(average_precision_score(y_true[col, t], y_pred[col, t]))
        return float(np.mean(aps)) if aps else None
    if metric_name == "mae":
        return float((y_true - y_pred).abs().mean())
    raise ValueError(f"_evaluate_metric does not support metric_name={metric_name!r}")


# ---------------------------------------------------------------------------
# the sensitivity probe wrapper
# ---------------------------------------------------------------------------
def probe_widths(run_cfg) -> dict:
    """Input width available to the Jacobian probe. Unlike GraphGPS's `probe_widths`,
    there is only one number: extra_node_pe is ADDED into `embedding_dim`, not
    concatenated, so h^(0) has the same width for all five PE variants. See the module
    docstring's "INPUT-SPACE WIDTH STORY" for the caveat this trades in exchange.
    """
    import json

    cfg_path = os.path.join(
        _REPO_ROOT, "configs", "graphormer",
        f"graphormer_{run_cfg.pe}_{run_cfg.dataset}.json",
    )
    with open(cfg_path) as f:
        embed_dim = json.load(f)["embed_dim"]
    return {"embedding_dim": embed_dim, "n_shared_feats": embed_dim}


def make_graphormer_model_fn(model, item, device=None):
    """Wrap a trained Graphormer model for `sensitivity.compute_sensitivity_curve`.

    Unlike `make_gps_model_fn`, `item` is NOT a raw PyG graph -- it is one already
    preprocessed by `_CachedPEGraphormerDataset.__getitem__` (has `.x`, `.in_degree`,
    `.out_degree`, `.spatial_pos`, `.attn_edge_type`, `.edge_input`, `.attn_bias`, and
    `.extra_pe` if applicable). Re-deriving those fields from a raw graph here would
    duplicate that class's cached-PE substitution logic; batching a preprocessed item is a
    one-item `collator([item], ...)` call instead.

    Returns (model_fn, probe_data, meta):
      model_fn(x)  runs the frozen attention-bias + transformer stack + output projection on
                   node representations `x` (attention bias, degrees, etc. are closed over
                   from `item`, fixed for the whole call), returning final-layer NODE
                   embeddings [n, p] with the graph token dropped.
      probe_data   `.x` = h^(0) (GraphNodeFeature's output, node rows only, detached),
                   `.edge_index`, `.num_nodes` -- ready for the probe.
      meta         probe_widths() plus the graph's node count.
    """
    from graphormer.data.collator import collator

    model.eval()
    device = device or next(model.parameters()).device
    encoder = model.encoder
    graph_encoder = encoder.graph_encoder

    batch = collator(
        [item], max_node=10_000, multi_hop_max_dist=5, spatial_pos_max=10_000,
    )
    if hasattr(item, "extra_pe"):
        # batch size 1, and item.x.size(0) <= max_node=10_000 guarantees the stock
        # collator above did not pad x beyond its own length, so extra_pe needs no padding
        # either -- unlike _CachedPEBatchedDataset.collater, which pads across a real batch.
        batch["extra_pe"] = item.extra_pe.unsqueeze(0)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    n = int(item.x.size(0))

    with torch.no_grad():
        node_feature_full = graph_encoder.graph_node_feature(batch)  # [1, n+1, C]
        attn_bias = graph_encoder.graph_attn_bias(batch)  # [1, H, n+1, n+1], fixed
        graph_token_row = node_feature_full[:, :1, :].clone()  # [1, 1, C], PE-independent

    h0 = node_feature_full[0, 1:, :].detach().clone()  # [n, C], node rows only

    def model_fn(x):
        full = torch.cat([graph_token_row, x.unsqueeze(0)], dim=1)  # [1, n+1, C]
        if graph_encoder.embed_scale is not None:
            full = full * graph_encoder.embed_scale
        if graph_encoder.quant_noise is not None:
            full = graph_encoder.quant_noise(full)
        if graph_encoder.emb_layer_norm is not None:
            full = graph_encoder.emb_layer_norm(full)
        full = full.transpose(0, 1)  # [n+1, 1, C]
        for layer in graph_encoder.layers:
            full, _ = layer(full, self_attn_padding_mask=None, self_attn_bias=attn_bias)
        full = full.transpose(0, 1)  # [1, n+1, C]
        out = encoder.layer_norm(encoder.activation_fn(encoder.lm_head_transform_weight(full)))
        return out[0, 1:, :]  # drop the graph token -- node embeddings only

    probe_data = types.SimpleNamespace(
        x=h0, edge_index=item.edge_index if hasattr(item, "edge_index") else None,
        num_nodes=n,
    )
    meta = {"embedding_dim": h0.shape[1], "n_shared_feats": h0.shape[1], "num_nodes": n}
    return model_fn, probe_data, meta
