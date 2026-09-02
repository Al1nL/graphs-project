"""
graphgps_pe_cache.py
====================
Make GraphGPS read THIS repo's precomputed PE cache instead of computing its own.

WHY THIS FILE EXISTS
--------------------
The whole study rests on one claim: every backbone sees the identical positional encoding,
so a difference in long-range sensitivity is attributable to the ARCHITECTURE and not to
two libraries' disagreeing definitions of "LapPE". SAN honours that -- san_backend reads
src/pe/cache.py directly. GraphGPS did not: `graphgps/loader/master_loader.py` runs its own
`compute_posenc_stats` as a pre-transform and the encoders consume THAT. So the GPS arm was
comparing GraphGPS's LapPE against SAN's cache LapPE, which is exactly the confound the
project exists to remove.

This module closes that gap by replacing the pre-transform, not by patching the encoders:
GraphGPS's own encoder architecture is left completely untouched, only the numbers going
into it change.

THE FOUR DEFINITIONS THAT DID NOT MATCH
---------------------------------------
Naively handing our arrays over would have produced a DIFFERENT wrong answer rather than
the right one. Measured against the pinned clone's reference configs:

                    ours (src/pe/compute_pe.py)      GraphGPS reference YAML
  Laplacian         sym-normalised                   laplacian_norm: none (combinatorial)
  eigenvectors      indices 1..k, trivial DROPPED    argsort()[:max_freqs], trivial KEPT
  count             K_LAP = 16 non-trivial           max_freqs: 10
  eigvec norm       none                             eigvec_norm: L2
  padding           ZEROS                            NaN

The last one is a silent-corruption hazard and the reason this is a module and not a
three-line lambda. `laplace_pos_encoder.py` and `signnet_pos_encoder.py` both do
`empty_mask = torch.isnan(pos_enc)` to mask frequencies a small graph does not have. Hand
them our zero padding and nothing raises -- the padding is simply read as a real
eigenvector of all zeros, on every graph with fewer than k+1 nodes.

Resolution: OUR definition wins, because it is the one SAN already consumes and the one
recorded in the analysis plan. We convert layout (zeros -> NaN, eigenvalues broadcast per
node) but never values. build_graphgym_cfg already sets max_freqs to K_LAP so the encoder
is built to the cache's width. `laplacian_norm` and `eigvec_norm` in the reference YAML
become INERT once this patch is installed -- nothing reads them, because nothing recomputes.

That is a real deviation from upstream's tuned config and it may move GPS's task metric.
It is the intended trade: a tuned-but-incomparable number answers no question this study
asks.

INDEX ALIGNMENT
---------------
The cache is keyed (split, index-within-split); GraphGPS pre-transforms ONE dataset. The
mapping between them is read off the dataset, not assumed -- assuming it is how this
module was first wrong, and the assumption looked entirely reasonable.

GraphGPS uses TWO different layouts:

  * `join_dataset_splits` (VOC, GNNBenchmark) concatenates train, then val, then test.
    Position alone is the mapping.
  * `preformat_Peptides` loads ONE dataset in its own file order and attaches
    `split_idxs = [s_dict['train'], s_dict['val'], s_dict['test']]` -- arbitrary index
    lists into it. Joined index 0 is the first molecule in FILE order, which is not the
    first TRAIN molecule.

Assuming concatenation for both paired a 119-node graph with a 338-node cache record on
the very first Peptides graph. bind_dataset() now reads `split_idxs` and builds the
mapping from it, which covers both layouts -- in the concatenated case it degenerates to
exactly the positional mapping.

`pre_transform_in_memory` applies the transform over `for i in range(len(dataset))`,
strictly ascending, so the call counter still identifies WHICH joined index we are on;
split_idxs then says where that index lives in the cache.

Because a derived mapping is still a thing to bet an experiment on, every call
independently verifies the record it fetched against the graph handed over, on both node
count AND edge count (spd == 1 gives the cached edge count). Any drift in upstream's
layout surfaces as a loud error on the first mismatched graph rather than as quietly
wrong encodings.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pe.cache import PECache  # noqa: E402

# The split order join_dataset_splits concatenates in. Not alphabetical -- do not "tidy".
SPLIT_ORDER = ("train", "val", "test")


def n_valid_lap_columns(num_nodes: int, k_lap: int) -> int:
    """How many of the k_lap eigenvector columns hold a real value rather than padding.

    compute_pe.compute_lap_pe asks LAPACK for eigenpairs 0..k_eff where
    k_eff = min(k_lap, num_nodes - 1), then keeps indices 1..k_eff -- dropping the trivial
    constant eigenvector -- and zero-pads the remainder out to k_lap.

    Padding is identified by POSITION, never by testing for zero: a genuine eigenvector
    entry can be exactly 0.0 (any node outside the support of that mode), and masking those
    would delete real signal from the encoder.
    """
    return max(0, min(k_lap, num_nodes - 1))


def lap_to_graphgps(lap_pe, eigvals, num_nodes: int, k_lap: int):
    """Convert cached (eigenvectors, eigenvalues) into GraphGPS's (EigVals, EigVecs).

    Returns EigVals of shape [n, k, 1] and EigVecs of shape [n, k], matching
    posenc_stats.get_lap_decomp_stats' documented contract exactly, including its NaN
    padding convention. Values are passed through untouched; only layout changes.
    """
    n_valid = n_valid_lap_columns(num_nodes, k_lap)

    vecs = torch.from_numpy(np.array(lap_pe, dtype=np.float32))
    if vecs.shape != (num_nodes, k_lap):
        raise ValueError(
            f"cached lap_pe has shape {tuple(vecs.shape)}, expected ({num_nodes}, {k_lap})")
    vecs[:, n_valid:] = float("nan")

    vals = torch.from_numpy(np.array(eigvals, dtype=np.float32))
    if vals.numel() != k_lap:
        raise ValueError(
            f"cached eigvals has {vals.numel()} entries, expected {k_lap}")
    vals[n_valid:] = float("nan")
    # GraphGPS repeats the graph's eigenvalues on every node and keeps a trailing 1-dim,
    # because the encoder concatenates them with the eigenvectors along a new last axis.
    vals = vals.unsqueeze(0).repeat(num_nodes, 1).unsqueeze(2)

    return vals, vecs


class CachedPosencStats:
    """Drop-in replacement for graphgps.transform.posenc_stats.compute_posenc_stats.

    Signature matches how master_loader binds it:
        partial(compute_posenc_stats, pe_types=..., is_undirected=..., cfg=cfg)(data)

    `is_undirected` is accepted and ignored -- it only ever steered GraphGPS's own
    eigendecomposition, and we are not doing one.
    """

    def __init__(self, cache_dir: str):
        self.caches = {s: PECache(cache_dir, s) for s in SPLIT_ORDER}
        self.sizes = {s: len(self.caches[s]) for s in SPLIT_ORDER}
        # Widths come from the manifest, never from a constant duplicated here: the cache
        # on disk is the authority on how wide the cache on disk is. A K_LAP edited in
        # compute_pe.py without rebuilding would otherwise show up as a shape error deep
        # in an encoder instead of as the stale cache it actually is.
        manifest = self.caches["train"].manifest
        self.k_lap = manifest["k_lap"]
        self.k_rwse = manifest["k_rwse"]
        self.calls = 0
        self._rev = None   # joined index -> (split, pos); set by bind_dataset

    def total(self) -> int:
        return sum(self.sizes.values())

    def bind_dataset(self, dataset) -> None:
        """Learn the joined-index -> (split, position) mapping from the dataset itself.

        GraphGPS assembles the three datasets in TWO different layouts, and assuming
        either one is how this module was first wrong:

          * join_dataset_splits (VOC, GNNBenchmark) concatenates train, then val, then
            test, so position alone IS the mapping.
          * preformat_Peptides loads ONE dataset in its own file order and records
            `split_idxs = [s_dict['train'], s_dict['val'], s_dict['test']]` -- arbitrary
            index LISTS into that dataset, not a concatenation. Joined index 0 is the
            first molecule in file order, which is not the first TRAIN molecule. Assuming
            otherwise paired a 119-node graph with a 338-node cache record on the very
            first graph.

        Reading split_idxs covers both, because in the concatenated case it degenerates
        to exactly the positional mapping. Called from the patched
        pre_transform_in_memory in install(), which is the only point where the dataset
        object is visible -- the transform callable itself receives just `data`.
        """
        idxs = getattr(dataset, "split_idxs", None)
        if idxs is None:
            # No split_idxs: fall back to positional. Left as a fallback rather than an
            # error because a dataset assembled some third way should still work if its
            # order happens to match; the per-graph node/edge checks below are what
            # actually protect correctness.
            self._rev = None
            return

        if len(idxs) != len(SPLIT_ORDER):
            raise ValueError(
                f"dataset.split_idxs has {len(idxs)} entries, expected "
                f"{len(SPLIT_ORDER)} ({', '.join(SPLIT_ORDER)}).")

        rev = {}
        for split, positions in zip(SPLIT_ORDER, idxs):
            positions = list(positions)
            if len(positions) != self.sizes[split]:
                raise ValueError(
                    f"split '{split}' has {len(positions)} graphs in GraphGPS's dataset "
                    f"but {self.sizes[split]} in the PE cache. The cache was built from a "
                    "different version or a different split definition -- rebuild it with "
                    "src/pe/compute_pe.py.")
            for pos, joined in enumerate(positions):
                rev[int(joined)] = (split, pos)

        if len(rev) != self.total():
            raise ValueError(
                f"dataset.split_idxs covers {len(rev)} distinct indices but the cache "
                f"holds {self.total()} graphs -- the split lists overlap.")
        self._rev = rev

    def locate(self, i: int):
        """Map a joined-dataset index onto (split, index within that split)."""
        if self._rev is not None:
            try:
                return self._rev[i]
            except KeyError:
                raise IndexError(
                    f"joined index {i} is not in any split of dataset.split_idxs, which "
                    f"covers {self.total()} graphs. GraphGPS is pre-transforming a graph "
                    "the cache has no entry for.") from None

        for split in SPLIT_ORDER:
            n = self.sizes[split]
            if i < n:
                return split, i
            i -= n
        raise IndexError(
            f"joined index {self.calls} is past the end of the cache ({self.total()} "
            "graphs). GraphGPS is pre-transforming more graphs than were cached, so the "
            "cache was built for a different dataset version -- rebuild it with "
            "src/pe/compute_pe.py rather than trusting this alignment.")

    def __call__(self, data, pe_types, is_undirected=None, cfg=None):
        split, local = self.locate(self.calls)
        self.calls += 1
        rec = self.caches[split][local]

        num_nodes = int(data.num_nodes)
        cached_nodes = int(np.asarray(rec["lap_pe"]).shape[0])

        # Edge count as a second, independent check on top of node count. spd == 1 marks
        # adjacent ordered pairs, so it equals edge_index.shape[1] for PyG's undirected
        # representation. Node count alone is a weak fingerprint -- molecules of equal
        # size are common -- so a mapping that is subtly rather than grossly wrong could
        # slip past it.
        cached_edges = int((np.asarray(rec["spd"]) == 1).sum())
        num_edges = int(data.edge_index.shape[1]) if hasattr(data, "edge_index") else -1

        if cached_nodes != num_nodes or (num_edges >= 0 and cached_edges != num_edges):
            raise ValueError(
                f"PE cache misalignment at joined index {self.calls - 1} "
                f"({split}[{local}]): cache has {cached_nodes} nodes / {cached_edges} "
                f"edges, the graph GraphGPS handed over has {num_nodes} nodes / "
                f"{num_edges} edges. The loader's ordering does not match the order "
                "compute_pe.py cached in -- see this module's INDEX ALIGNMENT note. Do "
                "NOT relax this check; silently misaligned PEs would invalidate every "
                "number in the run.")

        if "LapPE" in pe_types or "EquivStableLapPE" in pe_types:
            data.EigVals, data.EigVecs = lap_to_graphgps(
                rec["lap_pe"], rec["lap_eigvals"], num_nodes, self.k_lap)

        if "SignNet" in pe_types:
            # SignNet consumes the same raw eigenvectors; its encoder is what makes the
            # result sign-invariant, so there is nothing different to feed it.
            data.eigvals_sn, data.eigvecs_sn = lap_to_graphgps(
                rec["signnet_in"], rec["lap_eigvals"], num_nodes, self.k_lap)

        if "RWSE" in pe_types:
            rwse = torch.from_numpy(np.array(rec["rwse"], dtype=np.float32))
            if rwse.shape != (num_nodes, self.k_rwse):
                raise ValueError(
                    f"cached rwse has shape {tuple(rwse.shape)}, expected "
                    f"({num_nodes}, {self.k_rwse})")
            # No padding: the k-step return probability is defined for every node of every
            # graph, however small, so unlike LapPE there is nothing to mask.
            data.pestat_RWSE = rwse

        unsupported = set(pe_types) - {"LapPE", "EquivStableLapPE", "SignNet", "RWSE"}
        if unsupported:
            raise NotImplementedError(
                f"PE types {sorted(unsupported)} are enabled in the GraphGPS config but "
                "are not in this repo's cache. Either add them to src/pe/compute_pe.py "
                "(so every backbone still sees one definition) or disable them in "
                "build_graphgym_cfg -- but do not fall back to GraphGPS's own "
                "computation for some PEs and the cache for others.")
        return data


def install(run_cfg, cfg=None) -> CachedPosencStats:
    """Patch GraphGPS's loader to read the cache. Call BEFORE create_loader().

    Patches the name bound inside `graphgps.loader.master_loader`, not the one in
    `graphgps.transform.posenc_stats`: master_loader does `from ... import
    compute_posenc_stats` at import time, so rebinding the source module afterwards would
    have no effect on the reference it already holds.

    Pass GraphGym's `cfg` to have the encoder widths checked against the cache's before a
    run starts. They are set independently -- build_graphgym_cfg from PE_SPEC, the cache
    from whatever compute_pe.py was run with -- and a mismatch is otherwise a shape error
    thrown several hundred graphs into a pre-transform.
    """
    from graphgps.loader import master_loader

    # No parentheses: resolved_cache_dir is a @property on RunConfig, unlike its
    # resolved_max_dist()/resolved_num_probe_graphs() neighbours, which are plain methods.
    stats = CachedPosencStats(run_cfg.resolved_cache_dir)

    if cfg is not None:
        for key in ("posenc_LapPE", "posenc_SignNet"):
            block = getattr(cfg, key, None)
            if block is not None and block.enable and block.eigen.max_freqs != stats.k_lap:
                raise ValueError(
                    f"{key}.eigen.max_freqs is {block.eigen.max_freqs} but the PE cache "
                    f"holds {stats.k_lap} eigenvectors per graph. build_graphgym_cfg and "
                    "the cache disagree on width -- fix PE_SPEC or rebuild the cache; do "
                    "not pad or truncate to bridge them.")
        rwse = getattr(cfg, "posenc_RWSE", None)
        if rwse is not None and rwse.enable:
            # `times` is NOT populated yet at this point. master_loader fills it in from
            # `times_func` inside load_dataset_master --
            #     if pecfg.kernel.times_func:
            #         pecfg.kernel.times = list(eval(pecfg.kernel.times_func))
            # -- which runs during create_loader(), AFTER install(). Reading `times` alone
            # here saw an empty list and reported "0 steps but the cache holds 20" against
            # a perfectly correct config. Resolve it the same way upstream does, so the
            # guard does not depend on which side of create_loader() it is called from.
            times = list(rwse.kernel.times)
            if not times and getattr(rwse.kernel, "times_func", ""):
                times = list(eval(rwse.kernel.times_func))
            if len(times) != stats.k_rwse:
                raise ValueError(
                    f"posenc_RWSE resolves to {len(times)} kernel steps but the PE cache "
                    f"holds {stats.k_rwse}. (times={rwse.kernel.times!r}, "
                    f"times_func={getattr(rwse.kernel, 'times_func', None)!r})")

    master_loader.compute_posenc_stats = stats

    # Also wrap pre_transform_in_memory, purely to get a look at the dataset OBJECT before
    # the per-graph calls start. The transform callable receives only `data`, so this is
    # the sole point where `dataset.split_idxs` -- which says how GraphGPS laid the three
    # splits out -- is reachable. master_loader deletes that attribute after the posenc
    # pass (see its `delattr(dataset, 'split_idxs')`), so it must be read here, not later.
    original = master_loader.pre_transform_in_memory

    def _capture_dataset(dataset, transform_func, show_progress=False):
        # master_loader calls this several times (task-specific preprocessing, posenc,
        # clipping). Bind only on the posenc pass, identified by our own callable sitting
        # inside the functools.partial it builds.
        if getattr(transform_func, "func", transform_func) is stats:
            stats.bind_dataset(dataset)
        return original(dataset, transform_func, show_progress)

    master_loader.pre_transform_in_memory = _capture_dataset
    return stats
