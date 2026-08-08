"""
cache.py
========
On-disk format for the precomputed PEs: one set of files per graph, memory-mapped on read,
never all resident at once.

The size problem
----------------
PascalVOC-SP is 11,355 graphs averaging 479 nodes. A dense all-pairs array is n^2 per
graph:

    479^2 x 11,355 = 2.6 GB   as uint8, for ONE dense field

The previous format did `torch.save(split_records, "<split>_pe.pt")` -- a single blob built
by accumulating every graph in a Python list first. That materialises the whole split in
RAM to write it, and again to read it. On VOC-SP that is fatal, and on Peptides it is
merely wasteful.

Three decisions follow.

1. STORE ONLY `spd`. The old format also stored `spd_bucket` and `edge_type_id`, but both
   are pure functions of `spd` -- the bucket via dataset_meta.spd_bucket_id, the edge type
   via (spd == 1). Storing them tripled the footprint to ~7.8 GB for nothing, and it was
   also what made the cache go stale when fix 3 changed the bucketing scheme. Deriving
   them on read costs a 256-entry lookup-table index (no Python loop) and means the
   bucketing scheme can now change WITHOUT recomputing the cache.

2. uint8, with 255 reserved for "unreachable". Real distances are far below the ceiling
   (Peptides diameter ~57, VOC-SP ~27), so nothing is capped -- this keeps fix 3's
   requirement that raw distances are stored uncapped, while still fitting one byte. The
   writer asserts rather than silently wrapping.

3. One file per graph per field, written as the graph is processed and read back with
   `mmap_mode="r"`. Nothing accumulates in memory in either direction; the OS pages in
   only the graphs actually touched.

Layout:

    cache/<dataset>/
      manifest.json          format version, counts, dims, observed max diameter
      <split>/node/0000123.npy   float32 [n, K_LAP + K_RWSE]   LapPE || RWSE
      <split>/eig/0000123.npy    float32 [K_LAP]               Laplacian eigenvalues
      <split>/spd/0000123.npy    uint8   [n, n]                255 = unreachable

SignNet consumes the raw eigenvectors, i.e. the LapPE block of the node file -- it is not
stored separately, because it is the same tensor (the sign-invariance is applied inside the
learned encoder, not baked into the cache).
"""

import json
import os

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PE_CACHE_VERSION  # noqa: E402
from dataset_meta import SPD_NUM_BUCKETS, spd_bucket_id  # noqa: E402

UNREACHABLE_U8 = 255
SPLITS = ("train", "val", "test")

# Lookup table for uint8 distance -> GRPE bucket. Built once; applying it is a single
# numpy fancy-index over the [n, n] array rather than a per-element Python call.
_BUCKET_LUT = np.array(
    [spd_bucket_id(d) for d in range(UNREACHABLE_U8)] + [spd_bucket_id(-1)],
    dtype=np.uint8,
)


def _dirs(root, split):
    return {k: os.path.join(root, split, k) for k in ("node", "eig", "spd")}


def encode_spd(spd: np.ndarray) -> np.ndarray:
    """Pack a raw int distance matrix (-1 = unreachable) into uint8.

    Raises rather than wrapping: a diameter at or above 255 would silently alias onto the
    unreachable sentinel, which is exactly the class of bug fix 3 removed from the old
    format (far and disconnected sharing a value).
    """
    finite = spd[spd >= 0]
    if finite.size and finite.max() >= UNREACHABLE_U8:
        raise ValueError(
            f"graph diameter {int(finite.max())} >= {UNREACHABLE_U8}; the uint8 encoding "
            "would alias it onto the unreachable sentinel. Widen the dtype in cache.py "
            "before caching this dataset."
        )
    out = np.full(spd.shape, UNREACHABLE_U8, dtype=np.uint8)
    mask = spd >= 0
    out[mask] = spd[mask].astype(np.uint8)
    return out


class PECacheWriter:
    """Streams one graph at a time to disk. Nothing is accumulated."""

    def __init__(self, root, dataset, k_lap, k_rwse):
        self.root, self.dataset = root, dataset
        self.k_lap, self.k_rwse = k_lap, k_rwse
        self.counts, self.max_diameter, self.bytes = {}, 0, 0

    def write_split(self, split, records_iter, total=None):
        for d in _dirs(self.root, split).values():
            os.makedirs(d, exist_ok=True)
        paths = _dirs(self.root, split)
        n_written = 0
        for i, (lap_pe, lap_eigvals, rwse, spd) in enumerate(records_iter):
            node = np.concatenate(
                [np.asarray(lap_pe, dtype=np.float32),
                 np.asarray(rwse, dtype=np.float32)], axis=1)
            spd_u8 = encode_spd(np.asarray(spd))
            finite = spd_u8[spd_u8 != UNREACHABLE_U8]
            if finite.size:
                self.max_diameter = max(self.max_diameter, int(finite.max()))
            for arr, key in ((node, "node"),
                             (np.asarray(lap_eigvals, dtype=np.float32), "eig"),
                             (spd_u8, "spd")):
                p = os.path.join(paths[key], f"{i:07d}.npy")
                np.save(p, arr)
                self.bytes += arr.nbytes
            n_written = i + 1
            if n_written % 500 == 0:
                # flush: stdout is block-buffered when redirected to a file, so without
                # this a 30-minute PascalVOC-SP build shows no progress at all until it
                # exits -- exactly when progress output is most wanted.
                print(f"[{self.dataset}/{split}] {n_written}"
                      f"{'/' + str(total) if total else ''} graphs -> "
                      f"{self.bytes / 1e9:.2f} GB", flush=True)
        self.counts[split] = n_written
        print(f"[{self.dataset}/{split}] done: {n_written} graphs", flush=True)

    def finalize(self):
        manifest = {
            "pe_cache_version": PE_CACHE_VERSION,
            "dataset": self.dataset,
            "counts": self.counts,
            "k_lap": self.k_lap,
            "k_rwse": self.k_rwse,
            "node_dim": self.k_lap + self.k_rwse,
            "spd_dtype": "uint8",
            "spd_unreachable": UNREACHABLE_U8,
            "max_observed_diameter": self.max_diameter,
            "total_bytes": self.bytes,
            "note": "spd_bucket and edge_type_id are DERIVED on read, not stored -- both "
                    "are pure functions of spd, and storing them tripled the footprint "
                    "and made the cache stale whenever the bucketing scheme changed.",
        }
        with open(os.path.join(self.root, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Wrote {self.root}/manifest.json  "
              f"({self.bytes / 1e9:.2f} GB total, max diameter "
              f"{self.max_diameter})")
        return manifest


class PECache:
    """Random-access, memory-mapped reader for one split.

    `cache[i]` returns a dict of arrays for graph i. `spd`/`node`/`eig` are memmaps (the
    OS pages in only what is touched); `spd_bucket` and `edge_type_id` are derived on
    demand and are ordinary arrays, so take them one graph at a time.
    """

    def __init__(self, root, split, expect_version=PE_CACHE_VERSION):
        self.root, self.split = root, split
        mpath = os.path.join(root, "manifest.json")
        if not os.path.exists(mpath):
            raise FileNotFoundError(
                f"no manifest at {mpath}; run `python src/pe/compute_pe.py --dataset ... "
                f"--out {root}` first"
            )
        with open(mpath) as f:
            self.manifest = json.load(f)
        got = self.manifest.get("pe_cache_version")
        if expect_version is not None and got != expect_version:
            raise RuntimeError(
                f"PE cache at {root} was written by version {got}, this code expects "
                f"{expect_version}. The cache is structurally different -- reusing it "
                "would mix two definitions of the same PE across cells of one grid. "
                "Recompute it."
            )
        self.k_lap = self.manifest["k_lap"]
        self._paths = _dirs(root, split)
        self._n = self.manifest["counts"].get(split, 0)

    def __len__(self):
        return self._n

    def _load(self, kind, i):
        return np.load(os.path.join(self._paths[kind], f"{i:07d}.npy"), mmap_mode="r")

    def __getitem__(self, i):
        if not 0 <= i < self._n:
            raise IndexError(f"graph {i} out of range for split '{self.split}' ({self._n})")
        node = self._load("node", i)
        spd = self._load("spd", i)
        return {
            "lap_pe": node[:, : self.k_lap],
            "rwse": node[:, self.k_lap:],
            "signnet_in": node[:, : self.k_lap],   # SignNet consumes the raw eigenvectors
            "lap_eigvals": self._load("eig", i),
            "spd": spd,
            "spd_bucket": _BUCKET_LUT[np.asarray(spd)],
            "edge_type_id": (np.asarray(spd) == 1).astype(np.uint8),
        }

    def __iter__(self):
        for i in range(self._n):
            yield self[i]


def derive_spd_bucket(spd_u8: np.ndarray) -> np.ndarray:
    """GRPE bias-table index from a uint8 distance matrix. Exposed for tests and adapters."""
    return _BUCKET_LUT[np.asarray(spd_u8)]


def estimate_cache_bytes(n_graphs: int, avg_nodes: int, k_lap: int, k_rwse: int) -> dict:
    """Budget a dataset before computing it. The dense term dominates and is quadratic."""
    dense = n_graphs * avg_nodes ** 2                      # uint8
    node = n_graphs * avg_nodes * (k_lap + k_rwse) * 4     # float32
    eig = n_graphs * k_lap * 4
    return {"dense_bytes": dense, "node_bytes": node, "eig_bytes": eig,
            "total_gb": (dense + node + eig) / 1e9,
            "note": f"{SPD_NUM_BUCKETS} GRPE buckets derived on read, not stored"}
