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
import config  # noqa: E402
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
    # from dataset_meta, set from MEASURED diameter percentiles (not the paper average)
    assert c.resolved_max_dist() == 54
    assert RunConfig("gps", "none", "peptides-func", 0).resolved_max_dist() == 159


class _FakeRepo:
    """Pin config.check_pinned's view of the world so the test does not depend on which
    repos happen to be cloned on this machine.

    The previous version of this test read the real git state and asserted only inside an
    `if status != "ok"` branch. The moment `gps` was actually pinned, that branch stopped
    running and the test passed while checking nothing. Retargeting it at an
    as-yet-unpinned backbone would just relocate the same time bomb, so the fix is to make
    the test independent of ambient state entirely.
    """

    def __init__(self, sha=None, origin=None, pinned=None, fork=None):
        self.sha, self.origin, self.pinned, self.fork = sha, origin, pinned, fork

    def __enter__(self):
        self._saved = (config._git_sha, config._git_origin,
                       dict(config.PINNED_COMMITS), dict(config.FORK_URLS))
        config._git_sha = lambda _p: self.sha
        config._git_origin = lambda _p: self.origin
        config.PINNED_COMMITS["gps"] = self.pinned
        config.FORK_URLS["gps"] = self.fork
        return self

    def __exit__(self, *exc):
        config._git_sha, config._git_origin = self._saved[0], self._saved[1]
        config.PINNED_COMMITS.clear(); config.PINNED_COMMITS.update(self._saved[2])
        config.FORK_URLS.clear(); config.FORK_URLS.update(self._saved[3])


FORK = "https://github.com/pazflashner/GraphGPS.git"
UPSTREAM = "https://github.com/rampasek/GraphGPS.git"
SHA = "a" * 40


def _status(**kw):
    with _FakeRepo(**kw):
        return check_pinned("gps", strict=False)["status"]


def test_check_pinned_covers_every_status():
    assert _status(sha=SHA, origin=FORK, pinned=SHA, fork=FORK) == "ok"
    assert _status(sha=SHA, origin=FORK, pinned=None, fork=FORK) == "unpinned"
    assert _status(sha="b" * 40, origin=FORK, pinned=SHA, fork=FORK) == "mismatch"
    assert _status(sha=None, origin=None, pinned=SHA, fork=None) == "missing"


def test_unpinned_upstream_is_loud_not_silent():
    """PINNED_COMMITS holds None, not 'main'. A None is a checkable 'nobody pinned this';
    a 'main' is a lie that looks like a pin. Strict mode must refuse to proceed."""
    for kw in ({"sha": SHA, "origin": FORK, "pinned": None, "fork": FORK},      # unpinned
               {"sha": "b" * 40, "origin": FORK, "pinned": SHA, "fork": FORK},  # mismatch
               {"sha": None, "origin": None, "pinned": SHA, "fork": None}):     # missing
        with _FakeRepo(**kw):
            assert "warning" in check_pinned("gps", strict=False)
            try:
                check_pinned("gps", strict=True)
            except RuntimeError:
                continue
            raise AssertionError(f"strict mode must raise for {kw}")


def test_clone_pointed_at_upstream_instead_of_the_fork_is_rejected():
    """The subtle one: origin=upstream carries the SAME sha today, so it looks correct --
    and loses it the moment upstream force-pushes, with nowhere for our GRPE adaptations
    to live. It must fail loudly rather than pass."""
    with _FakeRepo(sha=SHA, origin=UPSTREAM, pinned=SHA, fork=FORK):
        rep = check_pinned("gps", strict=False)
        assert rep["status"] == "wrong_origin"
        assert rep["origin"] == UPSTREAM and rep["expected_origin"] == FORK
        assert "force-push" in rep["warning"]
        try:
            check_pinned("gps", strict=True)
        except RuntimeError:
            return
        raise AssertionError("a clone of upstream rather than the fork must be rejected")


def test_same_repo_normalisation():
    """Remote URLs for one repo differ in ways that must not read as different repos."""
    same = [
        ("https://github.com/x/Y.git", "https://github.com/x/Y"),
        ("https://github.com/x/Y/", "https://github.com/x/Y"),
        ("git@github.com:x/Y.git", "https://github.com/x/Y"),
        ("https://GitHub.com/X/y", "https://github.com/x/Y"),
    ]
    for a, b in same:
        assert config._same_repo(a, b), (a, b)
    assert not config._same_repo("https://github.com/x/Y", "https://github.com/z/Y")
    assert not config._same_repo("https://github.com/x/Y", "https://github.com/x/Z")


def test_no_fork_url_recorded_skips_the_origin_check():
    """san/graphormer have no fork yet; absence of a fork URL must not be a hard failure
    on its own -- the unpinned check is what catches them."""
    with _FakeRepo(sha=SHA, origin=UPSTREAM, pinned=SHA, fork=None):
        assert check_pinned("gps", strict=False)["status"] == "ok"


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


def test_raw_dir_is_passed_through_to_the_dataset_loader():
    """--raw-dir exists so a cluster can keep ~5 GB of LRGB downloads off a quota-limited
    home directory (scripts/slurm/build_cache.slurm passes it). An option that silently
    ignored its value would send the download to the wrong filesystem and fail a job
    hours in, so check the path actually reaches LRGBDataset's root.
    """
    import compute_pe

    class _FakeData:
        def __init__(self, n):
            und = [(i, i + 1) for i in range(n - 1)]
            self.edge_index = __import__("torch").tensor(
                und + [(b, a) for a, b in und]).t()
            self.num_nodes = n

    seen_roots = []
    real_loader = compute_pe.LRGBDataset

    def _spy(root, name, split):
        seen_roots.append(root)
        return [_FakeData(6)]

    compute_pe.LRGBDataset = _spy
    out = tempfile.mkdtemp()
    try:
        compute_pe.process_dataset("peptides-func", out, raw_dir="/scratch/somewhere")
        assert seen_roots, "the loader was never called"
        for root in seen_roots:
            assert root.replace("\\", "/").startswith("/scratch/somewhere"), (
                f"--raw-dir was ignored; loader got root={root!r}")

        # And the default still points at the repo-relative path it always did.
        seen_roots.clear()
        out2 = tempfile.mkdtemp()
        try:
            compute_pe.process_dataset("peptides-func", out2)
            assert all("raw_data" in r for r in seen_roots), seen_roots
        finally:
            shutil.rmtree(out2, ignore_errors=True)
    finally:
        compute_pe.LRGBDataset = real_loader
        shutil.rmtree(out, ignore_errors=True)


def test_smoke_test_shrinks_the_probe_not_just_the_training():
    """--smoke-test truncated training but left the probe at PROBE_N_GRAPHS = 256.

    Regression test for a run that looked hung. Training finished in 1.8s and the probe
    then ran silently for an unbounded time: the probe costs
    len(graphs) x num_target_nodes x dim_inner backward passes through the whole layer
    stack, so 256 graphs x 32 targets x 96 is on the order of 10^5-10^6 of them. A smoke
    test whose cheap part is 2 batches and whose expensive part is untouched is not a
    smoke test.
    """
    from config import PROBE_N_GRAPHS, SMOKE_TEST_PROBE_GRAPHS, RunConfig

    assert SMOKE_TEST_PROBE_GRAPHS < PROBE_N_GRAPHS

    normal = RunConfig("gps", "rwse", "peptides-func", 0)
    assert normal.resolved_num_probe_graphs() == PROBE_N_GRAPHS

    smoke = RunConfig("gps", "rwse", "peptides-func", 0, smoke_test=True)
    assert smoke.resolved_num_probe_graphs() == SMOKE_TEST_PROBE_GRAPHS

    # smoke_test wins over an explicit count, as it does for epochs -- otherwise a
    # --num-probe-graphs left over from another command silently un-smokes the run
    explicit = RunConfig("gps", "rwse", "peptides-func", 0,
                         num_probe_graphs=256, smoke_test=True)
    assert explicit.resolved_num_probe_graphs() == SMOKE_TEST_PROBE_GRAPHS

    # and without the flag an explicit count is still honoured
    assert RunConfig("gps", "rwse", "peptides-func", 0,
                     num_probe_graphs=8).resolved_num_probe_graphs() == 8


def test_smoke_results_never_land_on_the_real_cells_path():
    """A smoke run wrote results/gps_rwse_peptides-func_seed0.json -- the real cell's file.

    This is the dangerous one, because nothing about the CONTENT gives it away: same
    schema, status "ok", a real metric_value and a real sensitivity_curve. They just come
    from 1 epoch and 2 probe graphs. Overwriting the real cell with it is invisible.
    """
    from config import RunConfig

    real = RunConfig("gps", "rwse", "peptides-func", 0)
    smoke = RunConfig("gps", "rwse", "peptides-func", 0, smoke_test=True)

    assert real.result_path != smoke.result_path
    assert smoke.result_path.endswith("_smoke.json")
    assert real.result_path.endswith("gps_rwse_peptides-func_seed0.json")


def test_aggregation_excludes_smoke_records_by_field_not_filename():
    """Both halves matter. The field is what actually protects the numbers: results_dir
    is globbed as *.json, so a _smoke.json file is picked up by the filename glob, and a
    copied or renamed file would defeat any name-based rule."""
    import json
    import sys
    import tempfile

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import aggregate_results

    with tempfile.TemporaryDirectory() as tmp:
        def write(name, record):
            with open(os.path.join(tmp, name), "w") as f:
                json.dump(record, f)

        write("real.json", {"backbone": "gps", "smoke_test": False, "metric_value": 0.65})
        write("cell_smoke.json", {"backbone": "gps", "smoke_test": True,
                                  "metric_value": 0.19})
        # renamed to look real -- the field must still exclude it
        write("looks_real.json", {"backbone": "gps", "smoke_test": True,
                                  "metric_value": 0.19})
        # written before the field existed: kept, since absence is not proof of smokiness
        write("legacy.json", {"backbone": "gps", "metric_value": 0.64})

        loaded = aggregate_results.load_all(tmp)

    values = sorted(r["metric_value"] for r in loaded)
    assert values == [0.64, 0.65], f"smoke records leaked into aggregation: {values}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nall tests passed")
