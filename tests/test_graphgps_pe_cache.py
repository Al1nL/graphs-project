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
            spd = rng.integers(0, 5, size=(n, n)).astype(np.uint8)
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


def test_joined_index_maps_onto_the_right_split_in_train_val_test_order():
    sizes = {"train": 4, "val": 2, "test": 3}
    counts = {s: [20 + i for i in range(sizes[s])] for s in SPLIT_ORDER}
    with tempfile.TemporaryDirectory() as root:
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
    with tempfile.TemporaryDirectory() as root:
        _write_cache(root, sizes, counts)
        stats = CachedPosencStats(root)
        assert stats.k_lap == K_LAP and stats.k_rwse == K_RWSE
    print("PASS  test_widths_are_read_from_the_manifest_not_hardcoded")


def test_a_node_count_mismatch_raises_instead_of_encoding_the_wrong_graph():
    """The alignment guard. If upstream's loader ordering ever drifts, this must fail
    loudly rather than attach graph i's PE to graph j."""
    sizes = {"train": 2, "val": 1, "test": 1}
    counts = {"train": [30, 31], "val": [32], "test": [33]}
    with tempfile.TemporaryDirectory() as root:
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
    with tempfile.TemporaryDirectory() as root:
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
    with tempfile.TemporaryDirectory() as root:
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
    with tempfile.TemporaryDirectory() as root:
        _write_cache(root, sizes, counts)
        stats = CachedPosencStats(root)
        try:
            stats(_Data(22), pe_types=["HKdiagSE"])
        except NotImplementedError as exc:
            assert "HKdiagSE" in str(exc)
            print("PASS  test_an_uncached_pe_type_refuses_rather_than_falling_back")
            return
    raise AssertionError("an uncached PE type was silently ignored")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all tests passed")
