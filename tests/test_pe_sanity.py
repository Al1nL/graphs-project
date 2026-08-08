"""Hand-calculated sanity checks for every PE, on a 10-node graph.

    python -m pytest tests/test_pe_sanity.py   (or: python tests/test_pe_sanity.py)

Every expected value below is derived ANALYTICALLY and written as a closed form, not copied
from a previous run of this code. That distinction is the entire point: a golden-value test
seeded from the implementation passes forever, including when the implementation is wrong.

The fixture is the 10-cycle C10, chosen because all four PEs have closed forms on it:

  * Laplacian spectrum: C10 is 2-regular, so the symmetric normalised Laplacian is
    L = I - A/2, and eig(A) = 2cos(2*pi*k/10). Hence eig(L) = 1 - cos(2*pi*k/10), i.e.
    {0, 0.190983 x2, 0.690983 x2, 1.309017 x2, 1.809017 x2, 2}.
  * RWSE: the walk is +-1 with probability 1/2, so it returns at step 2m with probability
    C(2m, m) / 2^(2m) and never at odd steps. Wrap-around needs 10 net steps, so the
    binomial form is exact for every step we check.
  * APSP: d(i, j) = min(|i-j|, 10-|i-j|), maximum 5.

A path graph and a disconnected graph cover the cases C10 cannot: distinct degrees, and
unreachable pairs.
"""

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "pe"))
from compute_pe import compute_lap_pe, compute_rwse, compute_spd  # noqa: E402
from cache import UNREACHABLE_U8, derive_spd_bucket, encode_spd  # noqa: E402
from dataset_meta import SPD_UNREACHABLE, spd_bucket_id  # noqa: E402

N = 10


def _cycle():
    und = [(i, (i + 1) % N) for i in range(N)]
    return torch.tensor(und + [(b, a) for a, b in und]).t()


def _path():
    und = [(i, i + 1) for i in range(N - 1)]
    return torch.tensor(und + [(b, a) for a, b in und]).t()


def _two_components():
    """0-1-2-3-4  and  5-6-7-8-9: pairs across the split are unreachable."""
    und = [(i, i + 1) for i in (0, 1, 2, 3)] + [(i, i + 1) for i in (5, 6, 7, 8)]
    return torch.tensor(und + [(b, a) for a, b in und]).t()


# --------------------------------------------------------------------------- LapPE
def test_lappe_eigenvalues_match_the_closed_form_for_a_cycle():
    """eig(L) = 1 - cos(2*pi*k/n) for an n-cycle. compute_lap_pe drops the trivial 0."""
    _, vals = compute_lap_pe(_cycle(), N, k=8)
    expected = sorted(1 - math.cos(2 * math.pi * k / N) for k in range(N))[1:9]
    assert np.allclose(vals.numpy(), expected, atol=1e-5), (vals.numpy(), expected)
    # the two lowest non-trivial eigenvalues are a degenerate pair (k = 1 and k = 9)
    assert abs(vals[0] - vals[1]) < 1e-6
    assert abs(vals[0].item() - 0.1909830) < 1e-5


def test_lappe_drops_the_trivial_eigenvector_and_returns_orthonormal_columns():
    vecs, vals = compute_lap_pe(_cycle(), N, k=8)
    assert vecs.shape == (N, 8) and vals.shape == (8,)
    assert vals.min() > 1e-6, "the trivial lambda=0 eigenvector must be dropped"
    for j in range(8):
        assert abs(np.linalg.norm(vecs[:, j].numpy()) - 1.0) < 1e-5, j
    # a constant vector is the lambda=0 eigenvector of a regular graph's Laplacian, so
    # every retained eigenvector must be orthogonal to it, i.e. sum to zero
    for j in range(8):
        assert abs(vecs[:, j].sum().item()) < 1e-4, j


def test_lappe_pads_when_the_graph_is_smaller_than_k():
    vecs, vals = compute_lap_pe(_cycle(), N, k=16)     # only 9 non-trivial exist
    assert vecs.shape == (N, 16) and vals.shape == (16,)
    assert np.allclose(vals[9:].numpy(), 0.0)          # padding, not real eigenvalues
    assert np.allclose(vecs[:, 9:].numpy(), 0.0)


def test_lappe_detects_disconnection_via_a_second_zero_eigenvalue():
    """Two components => multiplicity-2 zero eigenvalue, so the first RETAINED value is 0."""
    _, vals = compute_lap_pe(_two_components(), N, k=4)
    assert vals[0].item() < 1e-6
    _, connected = compute_lap_pe(_cycle(), N, k=4)
    assert connected[0].item() > 0.1


# --------------------------------------------------------------------------- RWSE
def test_rwse_matches_the_binomial_return_probability_on_a_cycle():
    """A cycle walk is +-1 w.p. 1/2, so P(return at step 2m) = C(2m, m) / 2^(2m), and
    P(return at an odd step) = 0. Wrap-around needs 10 net steps, so this is exact here."""
    rwse = compute_rwse(_cycle(), N, k=8).numpy()
    assert rwse.shape == (N, 8)
    for step in range(1, 9):
        expected = 0.0 if step % 2 else math.comb(step, step // 2) / 2 ** step
        got = rwse[:, step - 1]
        assert np.allclose(got, expected, atol=1e-6), (step, got[0], expected)
    # spot values, written out: step 2 -> 1/2, step 4 -> 6/16, step 6 -> 20/64
    assert abs(rwse[0, 1] - 0.5) < 1e-6
    assert abs(rwse[0, 3] - 0.375) < 1e-6
    assert abs(rwse[0, 5] - 20 / 64) < 1e-6


def test_rwse_is_vertex_transitive_on_a_cycle_but_not_on_a_path():
    """Every node of C10 is equivalent, so all rows are identical. A path breaks that.

    Two-step return probability, from first principles:

        P(return at 2 from i) = sum_{j ~ i} P(i->j) P(j->i)
                              = sum_{j ~ i} 1 / (deg(i) * deg(j))

    On P10 the endpoints have degree 1 and the interior degree 2, so:
      node 0  -> 1/(1*2)               = 1/2     (its only neighbour has degree 2, so the
                                                  walk is forced out but returns only half
                                                  the time -- NOT 1)
      node 1  -> 1/(2*1) + 1/(2*2)     = 3/4     (adjacent to a degree-1 endpoint)
      node 5  -> 1/(2*2) + 1/(2*2)     = 1/2
    """
    cyc = compute_rwse(_cycle(), N, k=6).numpy()
    assert np.allclose(cyc, cyc[0], atol=1e-6)

    pth = compute_rwse(_path(), N, k=6).numpy()
    deg = {i: (1 if i in (0, N - 1) else 2) for i in range(N)}
    nbrs = {i: [j for j in (i - 1, i + 1) if 0 <= j < N] for i in range(N)}
    expected2 = [sum(1.0 / (deg[i] * deg[j]) for j in nbrs[i]) for i in range(N)]
    assert np.allclose(pth[:, 1], expected2, atol=1e-6), (pth[:, 1], expected2)

    # spelled out, so the closed form above is checkable by eye
    assert abs(pth[0, 1] - 0.5) < 1e-6
    assert abs(pth[1, 1] - 0.75) < 1e-6
    assert abs(pth[5, 1] - 0.5) < 1e-6
    # the degree-1 endpoints make the path non-vertex-transitive, unlike the cycle
    assert not np.allclose(pth, pth[0], atol=1e-6)


def test_rwse_rows_are_probabilities():
    for ei in (_cycle(), _path(), _two_components()):
        r = compute_rwse(ei, N, k=8).numpy()
        assert (r >= -1e-9).all() and (r <= 1 + 1e-9).all()


# --------------------------------------------------------------------------- SignNet input
def test_signnet_input_is_the_raw_eigenvectors_and_is_sign_ambiguous():
    """SignNet consumes the raw eigenvectors; the sign-invariance lives in its learned
    encoder, not in the cache. So the cached tensor is deliberately sign-ambiguous, and
    phi(v) + phi(-v) is what removes the ambiguity downstream."""
    from compute_pe import SignNetEncoder

    vecs, _ = compute_lap_pe(_cycle(), N, k=4)
    enc = SignNetEncoder(k=4, hidden=8, out_dim=4).eval()
    with torch.no_grad():
        a = enc(vecs)
        b = enc(-vecs)              # flipping every eigenvector's sign
        c = enc(vecs * torch.tensor([1.0, -1.0, 1.0, -1.0]))   # per-column flips
    assert torch.allclose(a, b, atol=1e-6), "encoder must be invariant to a global flip"
    assert torch.allclose(a, c, atol=1e-6), "and to per-eigenvector flips"


# --------------------------------------------------------------------------- APSP / GRPE
def test_apsp_matches_the_closed_form_for_a_cycle():
    """d(i, j) = min(|i-j|, n-|i-j|)."""
    spd = compute_spd(_cycle(), N)
    for i in range(N):
        for j in range(N):
            expected = min(abs(i - j), N - abs(i - j))
            assert spd[i, j] == expected, (i, j, spd[i, j], expected)
    assert spd.max() == N // 2 == 5
    assert (np.diag(spd) == 0).all()


def test_apsp_on_a_path_and_symmetry():
    spd = compute_spd(_path(), N)
    for i in range(N):
        for j in range(N):
            assert spd[i, j] == abs(i - j), (i, j)
    assert (spd == spd.T).all()
    assert spd.max() == N - 1 == 9


def test_apsp_marks_unreachable_pairs_distinctly_from_far_ones():
    """The bug fix 3 removed: far and disconnected must not share a value."""
    spd = compute_spd(_two_components(), N)
    assert spd[0, 4] == 4                      # far, but reachable
    assert spd[0, 5] == -1                     # different component
    assert spd[9, 0] == -1
    u8 = encode_spd(spd)
    assert u8[0, 4] == 4
    assert u8[0, 5] == UNREACHABLE_U8
    assert derive_spd_bucket(u8)[0, 5] == SPD_UNREACHABLE
    assert derive_spd_bucket(u8)[0, 4] != SPD_UNREACHABLE


def test_grpe_buckets_derived_from_cache_match_the_direct_function():
    """The read-time lookup table must agree with dataset_meta.spd_bucket_id exactly --
    it is an optimisation, not a second definition."""
    spd = compute_spd(_cycle(), N)
    got = derive_spd_bucket(encode_spd(spd))
    for i in range(N):
        for j in range(N):
            assert got[i, j] == spd_bucket_id(int(spd[i, j])), (i, j)
    # C10's diameter is 5, inside the exact-bucket range, so buckets equal distances
    assert (got == spd).all()


def test_edge_type_id_is_exactly_the_adjacency():
    spd = compute_spd(_cycle(), N)
    edge_type = (encode_spd(spd) == 1).astype(np.uint8)
    assert edge_type.sum() == 2 * N            # each node has exactly 2 neighbours
    assert (np.diag(edge_type) == 0).all()


def test_encode_spd_refuses_to_alias_a_large_diameter():
    """A diameter >= 255 would silently collide with the unreachable sentinel."""
    big = np.arange(300, dtype=np.int64).reshape(1, 300)
    try:
        encode_spd(big)
    except ValueError as exc:
        assert "alias" in str(exc)
        return
    raise AssertionError("must refuse rather than wrap")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nall tests passed")
