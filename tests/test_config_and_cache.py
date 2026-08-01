"""Tests for the run-config schema, version locking, and the on-disk PE cache.

    python -m pytest tests/test_config_and_cache.py

The cache tests do a real write/read round-trip through PECacheWriter and PECache,
including the memory-map path, because the failure modes here are all I/O-shaped: a
silently stale cache, a dtype that wraps, a derived field that disagrees with its
definition.
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "pe"))
from cache import (  # noqa: E402
    UNREACHABLE_U8,
    PECache,
    PECacheWriter,
    derive_spd_bucket,
    encode_spd,
    estimate_cache_bytes,
)
from config import (  # noqa: E402
    BACKBONES,
    DATASETS,
    PES,
    PE_CACHE_VERSION,
    RunConfig,
    check_pinned,
    grid,
)
from dataset_meta import spd_bucket_id  # noqa: E402

K_LAP, K_RWSE = 16, 20


# --------------------------------------------------------------------------- config axes
def test_grid_is_the_full_product_of_explicit_axes():
    cells = list(grid())
    assert len(cells) == len(BACKBONES) * len(PES) * len(DATASETS) * 3 == 135
    assert len({c.run_id for c in cells}) == len(cells), "run_ids must be unique"


def test_grid_filters_and_axis_order():
    cells = list(grid(backbones=("gps",), pes=("rwse",), datasets=("peptides-func",),
                      seeds=(0, 1)))
    assert [c.run_id for c in cells] == [
        "gps_rwse_peptides-func_seed0", "gps_rwse_peptides-func_seed1"]


def test_run_config_rejects_unknown_axis_values():
    for bad in ({"backbone": "gcn"}, {"pe": "sinusoidal"}, {"dataset": "coco-sp"},
                {"seed": -1}):
        kw = {"backbone": "gps", "pe": "rwse", "dataset": "peptides-func", "seed": 0}
        kw.update(bad)
        try:
            RunConfig(**kw)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should have been rejected")


def test_run_config_is_frozen_and_hashes_its_content():
    c = RunConfig("gps", "rwse", "peptides-func", 0)
    try:
        c.seed = 1
    except Exception:
        pass
    else:
        raise AssertionError("RunConfig must be immutable")
    assert c.config_hash() == RunConfig("gps", "rwse", "peptides-func", 0).config_hash()
    assert c.config_hash() != RunConfig("gps", "rwse", "peptides-func", 1).config_hash()


def test_run_config_derives_paths_and_per_dataset_max_dist():
    c = RunConfig("san", "grpe", "pascalvoc-sp", 2, results_dir="out")
    assert c.run_id == "san_grpe_pascalvoc-sp_seed2"
    assert c.result_path == os.path.join("out", "san_grpe_pascalvoc-sp_seed2.json")
    assert c.resolved_cache_dir == os.path.join("cache", "pascalvoc-sp")
    assert c.metric_name == "macro_f1"
    assert c.resolved_max_dist() == 28                     # from dataset_meta, not 20
    assert RunConfig("gps", "none", "peptides-func", 0).resolved_max_dist() == 40


def test_unpinned_upstream_is_loud_not_silent():
    """PINNED_COMMITS holds None, not 'main'. A None is a checkable 'nobody pinned this';
    a 'main' is a lie that looks like a pin."""
    rep = check_pinned("gps", strict=False)
    assert rep["status"] in ("unpinned", "missing", "mismatch", "ok")
    if rep["status"] != "ok":
        assert "warning" in rep
        try:
            check_pinned("gps", strict=True)
        except RuntimeError:
            return
        raise AssertionError("strict mode must raise on an unpinned upstream")


# --------------------------------------------------------------------------- cache
def _fake_graphs(sizes, seed=0):
    rng = np.random.default_rng(seed)
    for n in sizes:
        lap = rng.normal(size=(n, K_LAP)).astype(np.float32)
        eig = np.sort(rng.uniform(0, 2, K_LAP)).astype(np.float32)
        rwse = rng.uniform(0, 1, (n, K_RWSE)).astype(np.float32)
        spd = np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(np.int64)
        spd[0, -1] = spd[-1, 0] = -1          # one unreachable pair
        yield lap, eig, rwse, spd


def test_cache_roundtrip_preserves_every_pe():
    root = tempfile.mkdtemp()
    try:
        sizes = [7, 11, 5]
        w = PECacheWriter(root, "peptides-func", K_LAP, K_RWSE)
        originals = list(_fake_graphs(sizes))
        w.write_split("test", iter(originals), total=len(sizes))
        w.finalize()

        cache = PECache(root, "test")
        assert len(cache) == 3
        for i, (lap, eig, rwse, spd) in enumerate(originals):
            got = cache[i]
            assert np.allclose(got["lap_pe"], lap, atol=1e-6)
            assert np.allclose(got["rwse"], rwse, atol=1e-6)
            assert np.allclose(got["lap_eigvals"], eig, atol=1e-6)
            assert np.allclose(got["signnet_in"], lap, atol=1e-6)   # same tensor as LapPE
            assert got["spd"].shape == (sizes[i], sizes[i])
            assert got["spd"][0, -1] == UNREACHABLE_U8
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_cache_reads_are_memory_mapped():
    """The whole point: 2.6 GB of dense arrays must not be resident to touch one graph."""
    root = tempfile.mkdtemp()
    try:
        w = PECacheWriter(root, "peptides-func", K_LAP, K_RWSE)
        w.write_split("test", _fake_graphs([9, 9]), total=2)
        w.finalize()
        got = PECache(root, "test")[0]
        assert isinstance(got["spd"], np.memmap), type(got["spd"])
        assert isinstance(got["lap_eigvals"], np.memmap)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_cache_refuses_a_stale_version():
    """A cache from an older compute_pe is structurally different but looks identical on
    disk. Reusing it would mix two definitions of the same PE across one grid."""
    root = tempfile.mkdtemp()
    try:
        w = PECacheWriter(root, "peptides-func", K_LAP, K_RWSE)
        w.write_split("test", _fake_graphs([6]), total=1)
        w.finalize()
        mpath = os.path.join(root, "manifest.json")
        m = json.load(open(mpath))
        m["pe_cache_version"] = PE_CACHE_VERSION - 1
        json.dump(m, open(mpath, "w"))
        try:
            PECache(root, "test")
        except RuntimeError as exc:
            assert "Recompute" in str(exc)
            return
        raise AssertionError("a stale cache version must be refused")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_cache_reports_a_missing_manifest_actionably():
    root = tempfile.mkdtemp()
    try:
        PECache(root, "test")
    except FileNotFoundError as exc:
        assert "compute_pe.py" in str(exc)
    else:
        raise AssertionError("missing manifest must raise")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_derived_fields_are_not_stored_and_match_their_definitions():
    """spd_bucket and edge_type_id are pure functions of spd. Storing them tripled the
    footprint and made the cache go stale whenever the bucketing scheme changed."""
    root = tempfile.mkdtemp()
    try:
        w = PECacheWriter(root, "peptides-func", K_LAP, K_RWSE)
        w.write_split("test", _fake_graphs([12]), total=1)
        w.finalize()
        assert sorted(os.listdir(os.path.join(root, "test"))) == ["eig", "node", "spd"]

        got = PECache(root, "test")[0]
        spd = np.asarray(got["spd"])
        for i in range(spd.shape[0]):
            for j in range(spd.shape[1]):
                d = -1 if spd[i, j] == UNREACHABLE_U8 else int(spd[i, j])
                assert got["spd_bucket"][i, j] == spd_bucket_id(d), (i, j)
        assert (got["edge_type_id"] == (spd == 1)).all()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_encode_spd_uses_one_byte_and_separates_unreachable():
    spd = np.array([[0, 1, 5], [1, 0, -1], [5, -1, 0]], dtype=np.int64)
    u8 = encode_spd(spd)
    assert u8.dtype == np.uint8
    assert u8[0, 2] == 5 and u8[1, 2] == UNREACHABLE_U8
    assert derive_spd_bucket(u8)[1, 2] == spd_bucket_id(-1)


def test_size_estimate_matches_the_quoted_voc_figure():
    """VOC-SP: 479 nodes x 11,355 graphs as uint8 is the 2.6 GB that forced this format."""
    est = estimate_cache_bytes(n_graphs=11355, avg_nodes=479, k_lap=16, k_rwse=20)
    assert 2.5e9 < est["dense_bytes"] < 2.7e9, est["dense_bytes"]
    # storing the two derived fields as well would have tripled it
    assert est["dense_bytes"] * 3 > 7.5e9
    peptides = estimate_cache_bytes(15535, 151, 16, 20)
    assert peptides["dense_bytes"] < est["dense_bytes"] / 5   # quadratic in node count


# --------------------------------------------------------------------------- end to end
def test_process_dataset_writes_a_readable_cache():
    """The glue between compute_pe and the cache writer, which nothing else covered.

    compute_lap_pe / compute_rwse / compute_spd are checked in test_pe_sanity.py and
    PECacheWriter / PECache in this file, but process_dataset -- the function that wires
    them together and is the ONLY thing a real run calls -- had no test at all. It cannot
    be exercised against real LRGB here (no download), so LRGBDataset is stubbed with
    small path graphs; everything downstream of it is the real code path.
    """
    import compute_pe

    class _FakeData:
        def __init__(self, n):
            und = [(i, i + 1) for i in range(n - 1)]
            self.edge_index = __import__("torch").tensor(
                und + [(b, a) for a, b in und]).t()
            self.num_nodes = n

    sizes = {"train": [8, 12], "val": [9], "test": [7, 11, 6]}
    real_loader = compute_pe.LRGBDataset
    compute_pe.LRGBDataset = lambda root, name, split: [
        _FakeData(n) for n in sizes[split]]
    out = tempfile.mkdtemp()
    try:
        manifest = compute_pe.process_dataset("peptides-func", out)
        assert manifest["counts"] == {k: len(v) for k, v in sizes.items()}
        assert manifest["node_dim"] == compute_pe.K_LAP + compute_pe.K_RWSE
        assert manifest["spd_dtype"] == "uint8"
        for split, want in sizes.items():
            c = PECache(out, split)
            assert len(c) == len(want)
            g = c[0]
            assert g["lap_pe"].shape == (want[0], compute_pe.K_LAP)
            assert g["rwse"].shape == (want[0], compute_pe.K_RWSE)
            assert g["spd"].shape == (want[0], want[0])
            assert isinstance(g["spd"], np.memmap), "reads must stay memory-mapped"
            assert g["spd_bucket"].max() < 24 and g["edge_type_id"].max() <= 1
    finally:
        compute_pe.LRGBDataset = real_loader
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nall tests passed")
