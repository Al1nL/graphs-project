"""
Tests for backends/graphgps_pe_cache.py -- the adapter that makes GraphGPS consume this
repo's PE cache instead of computing its own.

These run WITHOUT GraphGPS installed. What they can check is the conversion contract and
the alignment guards, which is where the silent-corruption risk lives; what they cannot
check is that GraphGPS's loader still routes through master_loader.compute_posenc_stats
under the pin. That one is guarded at runtime instead, by graphgps_train raising if the
patched callable was never invoked.
"""

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from backends.graphgps_pe_cache import (  # noqa: E402
    CachedPosencStats, lap_to_graphgps, n_valid_lap_columns, SPLIT_ORDER)
from pe.cache import PECacheWriter  # noqa: E402

K_LAP, K_RWSE = 16, 20


class _Data:
    """Stand-in for a PyG Data object -- the adapter only touches num_nodes and setattr."""
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes


def _write_cache(root, sizes, node_counts):
    """Build a small real cache via the real writer, so these tests bind to the actual
    on-disk format rather than to a mock of it."""
    w = PECacheWriter(root, "peptides-func", K_LAP, K_RWSE)
    rng = np.random.default_rng(0)
    for split in SPLIT_ORDER:
        recs = []
        for j in range(sizes[split]):
            n = node_counts[split][j]
            k_eff = max(0, min(K_LAP, n - 1))
            # Zero-pad past k_eff exactly as compute_pe.compute_lap_pe does -- the whole
            # point is to exercise the real padding convention, not a tidied one.
            lap_pe = np.zeros((n, K_LAP), dtype=np.float32)
            lap_pe[:, :k_eff] = rng.normal(size=(n, k_eff))
            eig = np.zeros(K_LAP, dtype=np.float32)
            eig[:k_eff] = np.sort(rng.random(k_eff))
            rwse = rng.random((n, K_RWSE)).astype(np.float32)
            # spd for a PATH graph: d(i,j) = |i-j|. Not arbitrary -- the adapter derives
            # the cached edge count from (spd == 1) and checks it against the graph handed
            # over, so the fixture has to describe a real graph. _FakeDataset builds the
            # matching path edge_index, giving 2*(n-1) in both places.
            ii = np.arange(n)
            spd = np.abs(ii[:, None] - ii[None, :]).astype(np.uint8)
            recs.append((lap_pe, eig, rwse, spd))
        w.write_split(split, iter(recs), total=len(recs))
    w.finalize()


def test_padding_columns_are_identified_by_position_not_by_value():
    # A graph with 5 nodes supports only min(16, 4) = 4 non-trivial eigenvectors.
    assert n_valid_lap_columns(5, K_LAP) == 4
    assert n_valid_lap_columns(100, K_LAP) == K_LAP
    assert n_valid_lap_columns(1, K_LAP) == 0
    print("PASS  test_padding_columns_are_identified_by_position_not_by_value")


def test_zero_padding_becomes_nan_because_the_encoders_mask_on_nan():
    """The whole reason this module exists: GraphGPS masks with torch.isnan, our cache
    pads with zeros, and zeros would be read as a real all-zero eigenvector."""
    n = 5
    lap = np.ones((n, K_LAP), dtype=np.float32)   # all-ones INCLUDING the pad columns
    vals = np.ones(K_LAP, dtype=np.float32)
    eigvals, eigvecs = lap_to_graphgps(lap, vals, n, K_LAP)

    valid = n_valid_lap_columns(n, K_LAP)
    assert not torch.isnan(eigvecs[:, :valid]).any(), "real columns must survive"
    assert torch.isnan(eigvecs[:, valid:]).all(), "pad columns must become NaN"
    assert torch.isnan(eigvals[:, valid:, 0]).all()
    print("PASS  test_zero_padding_becomes_nan_because_the_encoders_mask_on_nan")


def test_a_genuine_zero_in_a_real_column_is_not_masked():
    """Padding must be found by position. An eigenvector entry can legitimately be 0.0,
    and masking those would delete real signal."""
    n = 50                       # >> K_LAP, so every column is real
    lap = np.zeros((n, K_LAP), dtype=np.float32)   # every value is exactly 0.0
    _, eigvecs = lap_to_graphgps(lap, np.zeros(K_LAP, dtype=np.float32), n, K_LAP)
    assert not torch.isnan(eigvecs).any(), (
        "a real column of zeros was masked as padding; padding must be positional")
    print("PASS  test_a_genuine_zero_in_a_real_column_is_not_masked")


def test_shapes_match_graphgps_get_lap_decomp_stats_contract():
    """posenc_stats.get_lap_decomp_stats documents (N, max_freqs, 1) and (N, max_freqs)."""
    n = 40
    eigvals, eigvecs = lap_to_graphgps(
        np.zeros((n, K_LAP), np.float32), np.zeros(K_LAP, np.float32), n, K_LAP)
    assert eigvecs.shape == (n, K_LAP), eigvecs.shape
    assert eigvals.shape == (n, K_LAP, 1), eigvals.shape
    print("PASS  test_shapes_match_graphgps_get_lap_decomp_stats_contract")


def test_eigenvalues_are_broadcast_identically_to_every_node():
    n = 12
    vals = np.arange(K_LAP, dtype=np.float32)
    eigvals, _ = lap_to_graphgps(np.zeros((n, K_LAP), np.float32), vals, n, K_LAP)
    valid = n_valid_lap_columns(n, K_LAP)
    for node in range(n):
        assert torch.equal(eigvals[node, :valid, 0], eigvals[0, :valid, 0])
    print("PASS  test_eigenvalues_are_broadcast_identically_to_every_node")


def test_values_pass_through_unchanged():
    """Layout is converted; numbers are not. If this ever fails, the GPS arm has stopped
    seeing the same PE as the SAN arm."""
    n = 60
    rng = np.random.default_rng(1)
    lap = rng.normal(size=(n, K_LAP)).astype(np.float32)
    _, eigvecs = lap_to_graphgps(lap, np.zeros(K_LAP, np.float32), n, K_LAP)
    assert np.allclose(eigvecs.numpy(), lap), "eigenvector values were altered in transit"
    print("PASS  test_values_pass_through_unchanged")


# ignore_cleanup_errors: PECache memory-maps every array it reads, and on Windows a
# mapped file cannot be unlinked while the mapping is alive. The mappings are released
# when the PECache objects are collected, which is not deterministic, so teardown can
# race the GC. Nothing here depends on the directory actually being removed.
def test_joined_index_maps_onto_the_right_split_in_train_val_test_order():
    sizes = {"train": 4, "val": 2, "test": 3}
    counts = {s: [20 + i for i in range(sizes[s])] for s in SPLIT_ORDER}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
        _write_cache(root, sizes, counts)
        stats = CachedPosencStats(root)
        assert stats.total() == 9
        assert stats.locate(0) == ("train", 0)
        assert stats.locate(3) == ("train", 3)
        assert stats.locate(4) == ("val", 0)
        assert stats.locate(6) == ("test", 0)
        assert stats.locate(8) == ("test", 2)
    print("PASS  test_joined_index_maps_onto_the_right_split_in_train_val_test_order")


def test_widths_are_read_from_the_manifest_not_hardcoded():
    sizes = {"train": 2, "val": 1, "test": 1}
    counts = {s: [30] * sizes[s] for s in SPLIT_ORDER}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
        _write_cache(root, sizes, counts)
        stats = CachedPosencStats(root)
        assert stats.k_lap == K_LAP and stats.k_rwse == K_RWSE
    print("PASS  test_widths_are_read_from_the_manifest_not_hardcoded")


def test_a_node_count_mismatch_raises_instead_of_encoding_the_wrong_graph():
    """The alignment guard. If upstream's loader ordering ever drifts, this must fail
    loudly rather than attach graph i's PE to graph j."""
    sizes = {"train": 2, "val": 1, "test": 1}
    counts = {"train": [30, 31], "val": [32], "test": [33]}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
        _write_cache(root, sizes, counts)
        stats = CachedPosencStats(root)
        try:
            stats(_Data(num_nodes=999), pe_types=["LapPE"])
        except ValueError as exc:
            assert "misalignment" in str(exc)
            print("PASS  test_a_node_count_mismatch_raises_instead_of_encoding_the_wrong_graph")
            return
    raise AssertionError("a node-count mismatch was accepted silently")


def test_running_off_the_end_of_the_cache_raises():
    sizes = {"train": 1, "val": 1, "test": 1}
    counts = {s: [25] for s in SPLIT_ORDER}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
        _write_cache(root, sizes, counts)
        stats = CachedPosencStats(root)
        for _ in range(3):
            stats(_Data(num_nodes=25), pe_types=["RWSE"])
        try:
            stats(_Data(num_nodes=25), pe_types=["RWSE"])
        except IndexError as exc:
            assert "past the end" in str(exc)
            print("PASS  test_running_off_the_end_of_the_cache_raises")
            return
    raise AssertionError("pre-transforming more graphs than were cached went unnoticed")


def test_each_pe_type_attaches_the_attributes_graphgps_encoders_read():
    sizes = {"train": 3, "val": 1, "test": 1}
    counts = {s: [40] * sizes[s] for s in SPLIT_ORDER}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
        _write_cache(root, sizes, counts)
        stats = CachedPosencStats(root)

        d = stats(_Data(40), pe_types=["LapPE"])
        assert hasattr(d, "EigVals") and hasattr(d, "EigVecs")
        assert not hasattr(d, "pestat_RWSE")

        d = stats(_Data(40), pe_types=["RWSE"])
        assert d.pestat_RWSE.shape == (40, K_RWSE)

        d = stats(_Data(40), pe_types=["SignNet"])
        assert hasattr(d, "eigvecs_sn") and hasattr(d, "eigvals_sn")
    print("PASS  test_each_pe_type_attaches_the_attributes_graphgps_encoders_read")


def test_an_uncached_pe_type_refuses_rather_than_falling_back():
    """Mixing cache-fed and GraphGPS-computed PEs in one run would reintroduce exactly the
    inconsistency this module removes, so an unknown PE must be an error."""
    sizes = {"train": 1, "val": 1, "test": 1}
    counts = {s: [22] for s in SPLIT_ORDER}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
        _write_cache(root, sizes, counts)
        stats = CachedPosencStats(root)
        try:
            stats(_Data(22), pe_types=["HKdiagSE"])
        except NotImplementedError as exc:
            assert "HKdiagSE" in str(exc)
            print("PASS  test_an_uncached_pe_type_refuses_rather_than_falling_back")
            return
    raise AssertionError("an uncached PE type was silently ignored")


def _stub_graphgps_loader():
    """Put a fake graphgps.loader.master_loader into sys.modules.

    install() imports it, which is why nothing here reached install() at all -- and that
    gap is exactly how the bug below survived to a live run. The import is upstream's; the
    PATCHING is ours, and ours is testable without GraphGPS installed.
    """
    import types
    mods = {}
    for name in ("graphgps", "graphgps.loader", "graphgps.loader.master_loader"):
        mods[name] = sys.modules.get(name)
        sys.modules[name] = types.ModuleType(name)
    ml = sys.modules["graphgps.loader.master_loader"]
    sys.modules["graphgps.loader"].master_loader = ml
    ml.compute_posenc_stats = lambda *a, **k: None

    # Mirrors transforms.pre_transform_in_memory: ascending, no shuffling. install()
    # wraps this to get at the dataset object, so the stub must provide it.
    def _ptim(dataset, transform_func, show_progress=False):
        return [transform_func(dataset.get(i)) for i in range(len(dataset))]
    ml.pre_transform_in_memory = _ptim
    return mods


def _unstub(saved):
    for name, mod in saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


def test_install_patches_the_loader_and_reads_cache_dir_as_a_property():
    """REGRESSION: install() called `run_cfg.resolved_cache_dir()` with parentheses.

    That attribute is a @property on RunConfig -- unlike its resolved_max_dist() and
    resolved_num_probe_graphs() neighbours, which are plain methods -- so the call raised

        TypeError: 'str' object is not callable

    on the first real run, after a conda env had been built and a GPU allocated. Uses the
    REAL RunConfig rather than a stand-in, because the whole point is which side of that
    property/method split the attribute falls on.
    """
    from backends.graphgps_pe_cache import install
    from config import RunConfig

    sizes = {"train": 2, "val": 1, "test": 1}
    counts = {s: [24] * sizes[s] for s in SPLIT_ORDER}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
        _write_cache(root, sizes, counts)
        saved = _stub_graphgps_loader()
        try:
            run_cfg = RunConfig(backbone="gps", pe="rwse", dataset="peptides-func",
                                seed=0, cache_dir=root)
            stats = install(run_cfg)
            patched = sys.modules["graphgps.loader.master_loader"].compute_posenc_stats
            assert patched is stats, "install() did not patch master_loader"
            assert isinstance(patched, CachedPosencStats)
            assert stats.total() == 4
        finally:
            _unstub(saved)
    print("PASS  test_install_patches_the_loader_and_reads_cache_dir_as_a_property")


def test_install_refuses_a_config_whose_width_disagrees_with_the_cache():
    """The guard that turns a width mismatch into an error before training, rather than a
    shape failure hundreds of graphs into a pre-transform."""
    import types
    from backends.graphgps_pe_cache import install
    from config import RunConfig

    sizes = {"train": 1, "val": 1, "test": 1}
    counts = {s: [24] for s in SPLIT_ORDER}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
        _write_cache(root, sizes, counts)
        saved = _stub_graphgps_loader()
        try:
            run_cfg = RunConfig(backbone="gps", pe="lappe", dataset="peptides-func",
                                seed=0, cache_dir=root)
            bad = types.SimpleNamespace(
                posenc_LapPE=types.SimpleNamespace(
                    enable=True, eigen=types.SimpleNamespace(max_freqs=10)),
                posenc_SignNet=types.SimpleNamespace(
                    enable=False, eigen=types.SimpleNamespace(max_freqs=10)),
                posenc_RWSE=types.SimpleNamespace(
                    enable=False, kernel=types.SimpleNamespace(times=[])))
            try:
                install(run_cfg, bad)
            except ValueError as exc:
                assert "max_freqs" in str(exc) and str(K_LAP) in str(exc)
                print("PASS  test_install_refuses_a_config_whose_width_disagrees_with_the_cache")
                return
            raise AssertionError("a max_freqs/cache width mismatch was accepted")
        finally:
            _unstub(saved)


def test_rwse_width_check_resolves_times_from_times_func():
    """REGRESSION: the guard read cfg.posenc_RWSE.kernel.times, which is EMPTY when
    install() runs.

    master_loader fills `times` in from `times_func` inside load_dataset_master, during
    create_loader() -- after install(). So a correct config was rejected with
    "0 steps but the PE cache holds 20". The guard must resolve times_func the same way
    upstream will.
    """
    import types
    from backends.graphgps_pe_cache import install
    from config import RunConfig

    sizes = {"train": 1, "val": 1, "test": 1}
    counts = {s: [24] for s in SPLIT_ORDER}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
        _write_cache(root, sizes, counts)
        saved = _stub_graphgps_loader()
        try:
            run_cfg = RunConfig(backbone="gps", pe="rwse", dataset="peptides-func",
                                seed=0, cache_dir=root)

            # exactly what build_graphgym_cfg produces BEFORE create_loader() runs
            pre_loader = types.SimpleNamespace(
                posenc_LapPE=types.SimpleNamespace(
                    enable=False, eigen=types.SimpleNamespace(max_freqs=K_LAP)),
                posenc_SignNet=types.SimpleNamespace(
                    enable=False, eigen=types.SimpleNamespace(max_freqs=K_LAP)),
                posenc_RWSE=types.SimpleNamespace(
                    enable=True,
                    kernel=types.SimpleNamespace(times=[],
                                                 times_func=f"range(1,{K_RWSE + 1})")))
            install(run_cfg, pre_loader)   # must NOT raise

            # and after the loader has expanded it, the same check still holds
            post_loader = types.SimpleNamespace(
                posenc_LapPE=pre_loader.posenc_LapPE,
                posenc_SignNet=pre_loader.posenc_SignNet,
                posenc_RWSE=types.SimpleNamespace(
                    enable=True,
                    kernel=types.SimpleNamespace(times=list(range(1, K_RWSE + 1)),
                                                 times_func=f"range(1,{K_RWSE + 1})")))
            install(run_cfg, post_loader)  # must NOT raise

            # a genuine mismatch must still be caught
            wrong = types.SimpleNamespace(
                posenc_LapPE=pre_loader.posenc_LapPE,
                posenc_SignNet=pre_loader.posenc_SignNet,
                posenc_RWSE=types.SimpleNamespace(
                    enable=True,
                    kernel=types.SimpleNamespace(times=[], times_func="range(1,8)")))
            try:
                install(run_cfg, wrong)
            except ValueError as exc:
                assert "kernel steps" in str(exc)
            else:
                raise AssertionError("a real RWSE width mismatch was accepted")
        finally:
            _unstub(saved)
    print("PASS  test_rwse_width_check_resolves_times_from_times_func")


class _FakeDataset:
    """Stands in for GraphGPS's dataset: node counts in FILE order, plus split_idxs."""
    def __init__(self, node_counts, split_idxs):
        self.node_counts = node_counts
        self.split_idxs = split_idxs

    def __len__(self):
        return len(self.node_counts)

    def get(self, i):
        import torch as _t
        n = self.node_counts[i]
        d = _Data(n)
        # a path graph's undirected edge_index: 2*(n-1) columns, matching the cache's
        # spd==1 count for the same structure written by _write_cache
        und = [(a, a + 1) for a in range(n - 1)]
        d.edge_index = _t.tensor(und + [(b, a) for a, b in und]).t()
        return d


def test_peptides_layout_maps_through_split_idxs_not_position():
    """REGRESSION: the real failure on the cluster.

        ValueError: PE cache misalignment at joined index 0 (train[0]): the cache has
        338 nodes, the graph GraphGPS handed over has 119.

    preformat_Peptides loads ONE dataset in file order and attaches split_idxs as index
    LISTS into it, so joined index 0 is the first molecule in the file, not the first
    TRAIN molecule. This module assumed join_dataset_splits' train/val/test concatenation
    for every dataset, which is true for VOC and false for Peptides.
    """
    from backends.graphgps_pe_cache import install
    from config import RunConfig
    import functools

    # cache: train has 3 graphs, val 1, test 1 -- with DISTINCT node counts so a wrong
    # mapping cannot accidentally pass the node-count check
    sizes = {"train": 3, "val": 1, "test": 1}
    counts = {"train": [30, 31, 32], "val": [40], "test": [50]}

    # GraphGPS's file order interleaves them; split_idxs says where each one went
    file_order = [40, 30, 50, 32, 31]           # val, train0, test, train2, train1
    split_idxs = [[1, 4, 3], [0], [2]]          # train, val, test

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
        _write_cache(root, sizes, counts)
        saved = _stub_graphgps_loader()
        try:
            run_cfg = RunConfig(backbone="gps", pe="rwse", dataset="peptides-func",
                                seed=0, cache_dir=root)
            stats = install(run_cfg)
            ml = sys.modules["graphgps.loader.master_loader"]

            ds = _FakeDataset(file_order, split_idxs)
            # exactly how master_loader invokes it
            ml.pre_transform_in_memory(ds, functools.partial(stats, pe_types=["RWSE"]))

            assert stats.calls == 5, stats.calls
            assert stats.locate(0) == ("val", 0)
            assert stats.locate(1) == ("train", 0)
            assert stats.locate(2) == ("test", 0)
            assert stats.locate(3) == ("train", 2)
            assert stats.locate(4) == ("train", 1)
        finally:
            _unstub(saved)
    print("PASS  test_peptides_layout_maps_through_split_idxs_not_position")


def test_split_idxs_that_disagree_with_the_cache_size_are_rejected():
    """A cache built from a different split definition must fail at bind time, before
    any graph is encoded."""
    from backends.graphgps_pe_cache import install
    from config import RunConfig
    import functools

    sizes = {"train": 2, "val": 1, "test": 1}
    counts = {"train": [30, 31], "val": [40], "test": [50]}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
        _write_cache(root, sizes, counts)
        saved = _stub_graphgps_loader()
        try:
            run_cfg = RunConfig(backbone="gps", pe="rwse", dataset="peptides-func",
                                seed=0, cache_dir=root)
            stats = install(run_cfg)
            ml = sys.modules["graphgps.loader.master_loader"]
            # three train graphs in the dataset, two in the cache
            ds = _FakeDataset([30, 31, 32, 40, 50], [[0, 1, 2], [3], [4]])
            try:
                ml.pre_transform_in_memory(ds, functools.partial(stats, pe_types=["RWSE"]))
            except ValueError as exc:
                assert "train" in str(exc) and "PE cache" in str(exc)
                print("PASS  test_split_idxs_that_disagree_with_the_cache_size_are_rejected")
                return
            raise AssertionError("a split-size disagreement was accepted")
        finally:
            _unstub(saved)


def test_fast_cache_serves_the_same_records_as_the_per_graph_reader():
    """The consolidated reader must be a pure optimisation: same numbers, fewer reads.

    Checked against PECache itself rather than against expected values, because the point
    is equivalence with the reference reader, not agreement with a second transcription
    of the fixture.
    """
    import numpy as np
    import tempfile

    from backends.graphgps_pe_cache import CachedPosencStats, FastPECache
    from pe.cache import PECache

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache_dir = os.path.join(tmp, "cache")
        _write_cache(cache_dir, {"train": 2, "val": 1, "test": 2},
                     {"train": [4, 6], "val": [5], "test": [7, 3]})
        fast_dir = os.path.join(tmp, "fast")

        for split in ("train", "val", "test"):
            slow = PECache(cache_dir, split)
            fast = FastPECache(cache_dir, split, fast_dir)
            assert len(fast) == len(slow)

            for i in range(len(slow)):
                a, b = slow[i], fast[i]
                for key in ("lap_pe", "rwse", "signnet_in", "lap_eigvals"):
                    assert np.allclose(np.asarray(a[key]), np.asarray(b[key])), (
                        f"{split}[{i}].{key} differs between the two readers")
                # the edge count the alignment check uses, precomputed instead of
                # recomputed from the [n, n] spd matrix
                assert b["num_edges"] == int((np.asarray(a["spd"]) == 1).sum())
                assert b["num_nodes"] == np.asarray(a["lap_pe"]).shape[0]

    print("PASS  test_fast_cache_serves_the_same_records_as_the_per_graph_reader")


def test_fast_cache_persists_and_is_reused():
    """Written once, read thereafter -- that reuse IS the optimisation. Also checks the
    build does not silently re-run, which would leave the cost exactly where it was."""
    import tempfile

    from backends.graphgps_pe_cache import FAST_CACHE_LAYOUT_VERSION, FastPECache

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache_dir = os.path.join(tmp, "cache")
        _write_cache(cache_dir, {"train": 2, "val": 1, "test": 1},
                     {"train": [4, 6], "val": [5], "test": [7]})
        fast_dir = os.path.join(tmp, "fast")

        first = FastPECache(cache_dir, "train", fast_dir)
        _ = first[0]
        path = os.path.join(fast_dir, f"train_pe_v{FAST_CACHE_LAYOUT_VERSION}.pt")
        assert os.path.exists(path), "the consolidated file was never written"

        # a second reader must load that file rather than walk the per-graph cache again
        import pe.cache as pe_cache_mod

        original = pe_cache_mod.PECache._load
        pe_cache_mod.PECache._load = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("fast cache rebuilt instead of loading the persisted file"))
        try:
            second = FastPECache(cache_dir, "train", fast_dir)
            assert second[0]["num_nodes"] == first[0]["num_nodes"]
        finally:
            pe_cache_mod.PECache._load = original

    print("PASS  test_fast_cache_persists_and_is_reused")


def test_a_stale_fast_cache_is_rebuilt_not_trusted():
    """A consolidated file disagreeing with the cache it came from would reintroduce the
    misalignment class of bug one level further from view, so mismatched metadata must
    rebuild rather than adapt."""
    import tempfile

    import torch

    from backends.graphgps_pe_cache import FAST_CACHE_LAYOUT_VERSION, FastPECache

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache_dir = os.path.join(tmp, "cache")
        _write_cache(cache_dir, {"train": 2, "val": 1, "test": 1},
                     {"train": [4, 6], "val": [5], "test": [7]})
        fast_dir = os.path.join(tmp, "fast")

        _ = FastPECache(cache_dir, "train", fast_dir)[0]
        path = os.path.join(fast_dir, f"train_pe_v{FAST_CACHE_LAYOUT_VERSION}.pt")

        blob = torch.load(path, map_location="cpu")
        blob["meta"]["k_lap"] = blob["meta"]["k_lap"] + 1     # pretend the cache moved
        blob["node"] = blob["node"][:1] * 0                   # and corrupt the payload
        torch.save(blob, path)

        rebuilt = FastPECache(cache_dir, "train", fast_dir)[0]
        assert rebuilt["num_nodes"] == 4, (
            "a stale consolidated file was trusted instead of rebuilt")

    print("PASS  test_a_stale_fast_cache_is_rebuilt_not_trusted")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all tests passed")
