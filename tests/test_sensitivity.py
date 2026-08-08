"""Regression tests for src/sensitivity.py.

    python -m pytest tests/test_sensitivity.py      (or: python tests/test_sensitivity.py)

s_bar(d) is the quantity the whole project rests on, and its two failure modes are silent:
a wrong-but-plausible norm, and a Jacobian taken over an input space whose width depends
on the PE. Both produce numbers that look fine and rank PEs incorrectly. These tests pin
the probe against a brute-force full Jacobian so a regression cannot pass unnoticed.

Needs only torch + networkx (no GPU, no dataset, no backbone repo).
"""

import os
import sys
import types

import networkx as nx
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sensitivity import (  # noqa: E402
    _frobenius_per_source,
    assert_shared_width,
    average_curves,
    compute_sensitivity_curve,
)

N, Q_SHARED, Q_PE, P = 9, 5, 4, 8  # nodes, shared feats, PE feats, hidden width
Q = Q_SHARED + Q_PE
MAX_D = 8


def _fixture():
    """Path graph 0-1-...-8 (so distances 1..8 are all populated) + a toy backbone."""
    torch.manual_seed(0)
    edges = [[i, i + 1] for i in range(N - 1)] + [[i + 1, i] for i in range(N - 1)]
    edge_index = torch.tensor(edges).t()
    adj = torch.zeros(N, N)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj = adj / adj.sum(1, keepdim=True)
    data = types.SimpleNamespace(x=torch.randn(N, Q), edge_index=edge_index, num_nodes=N)

    class Toy(nn.Module):
        """Local mixing + global attention: every node pair is genuinely coupled."""

        def __init__(self, use_ln=False):
            super().__init__()
            self.inp, self.qkv, self.out = nn.Linear(Q, P), nn.Linear(P, 3 * P), nn.Linear(P, P)
            self.norm = nn.LayerNorm(P) if use_ln else nn.Identity()

        def forward(self, x):
            h = torch.tanh(self.inp(x))
            h = torch.tanh(adj @ h) + h
            q, k, v = self.qkv(h).chunk(3, dim=-1)
            attn = torch.softmax(q @ k.t() / P**0.5, dim=-1)
            return self.norm(self.out(attn @ v + h))

    return data, Toy


def _brute_force(model_fn, data, n_shared):
    """Ground truth via the full [n, p, n, q] Jacobian, bucketed by hop distance."""
    full = torch.autograd.functional.jacobian(model_fn, data.x)
    fro = full[:, :, :, :n_shared].pow(2).sum(dim=(1, 3)).sqrt()  # [v, u]
    g = nx.Graph()
    g.add_nodes_from(range(N))
    g.add_edges_from(data.edge_index.t().tolist())
    dist = dict(nx.all_pairs_shortest_path_length(g))
    acc = {}
    for v in range(N):
        for u in range(N):
            d = dist[v].get(u, -1)
            if 1 <= d <= MAX_D:
                acc.setdefault(d, []).append(fro[v, u].item())
    return {d: {"mean": sum(x) / len(x), "count": len(x)} for d, x in sorted(acc.items())}


def test_probe_matches_true_frobenius_norm():
    """The probe must return ||dh_v/dx_u||_F, not the norm of the Jacobian's column sum."""
    data, Toy = _fixture()
    model_fn = Toy().eval()
    for n_shared in (Q_SHARED, Q):
        got = compute_sensitivity_curve(
            model_fn, data, n_shared_feats=n_shared, max_dist=MAX_D, num_target_nodes=N
        )
        want = _brute_force(model_fn, data, n_shared)
        assert got.keys() == want.keys()
        for d in want:
            assert abs(got[d]["mean"] - want[d]["mean"]) < 1e-5, d
            assert got[d]["count"] == want[d]["count"], d


def test_slice_excludes_pe_channels():
    """Including PE columns inflates ||.||_F mechanically -- the effect fix #1 removes.

    Guards against a refactor that silently drops the slice: without it, a PE that adds
    input channels wins on sensitivity purely by adding non-negative terms to the sum.
    """
    data, Toy = _fixture()
    model_fn = Toy().eval()
    kw = dict(max_dist=MAX_D, num_target_nodes=N)
    sliced = compute_sensitivity_curve(model_fn, data, n_shared_feats=Q_SHARED, **kw)
    unsliced = compute_sensitivity_curve(model_fn, data, n_shared_feats=Q, **kw)
    for d in sliced:
        assert unsliced[d]["mean"] > sliced[d]["mean"] * 1.05, d


def test_fallback_and_chunking_are_exact():
    """||J||_F^2 is additive over output basis directions, so chunk size cannot change it;
    the unbatched fallback (for backbones with ops lacking a vmap batching rule) must
    agree with the is_grads_batched path to numerical precision."""
    data, Toy = _fixture()
    model_fn = Toy().eval()
    x = data.x.clone().detach().requires_grad_(True)
    h = model_fn(x)
    batched, _ = _frobenius_per_source(h[4], x, Q_SHARED, chunk_size=3, batched_ok=True)
    looped, _ = _frobenius_per_source(h[4], x, Q_SHARED, chunk_size=3, batched_ok=False)
    single, _ = _frobenius_per_source(h[4], x, Q_SHARED, chunk_size=1, batched_ok=True)
    assert torch.allclose(batched, looped, atol=1e-6)
    assert torch.allclose(batched, single, atol=1e-6)


def test_sum_readout_is_degenerate_under_layernorm():
    """Documents WHY the old estimator was replaced, not just that it differed.

    LayerNorm zero-means across channels, so 1^T h retains only the non-uniform part of
    gamma. At init gamma == 1, making the gradient of `h[v].sum()` identically zero while
    the true Jacobian is large. The size of that error is therefore backbone-dependent
    (Graphormer is pre-LN; GPS/SAN use BatchNorm), which would confound the PE comparison.
    """
    data, Toy = _fixture()
    model = Toy(use_ln=True).eval()
    x = data.x.clone().detach().requires_grad_(True)
    old = torch.autograd.grad(model(x)[4].sum(), x)[0][:, :Q_SHARED].norm(dim=1)
    true = torch.autograd.functional.jacobian(model, data.x)[4][:, :, :Q_SHARED]
    true = true.pow(2).sum(dim=(0, 2)).sqrt()
    assert old.max() < 1e-6          # old estimator: identically zero
    assert true.max() > 1e-2         # true sensitivity: emphatically not


def test_average_curves_weights_by_pair_count():
    """Far pairs concentrate in large graphs, so an unweighted mean over graphs is biased."""
    pooled = average_curves([{3: {"mean": 1.0, "count": 100}}, {3: {"mean": 5.0, "count": 2}}])
    assert abs(pooled[3]["mean"] - (1.0 * 100 + 5.0 * 2) / 102) < 1e-12  # not 3.0
    assert pooled[3]["count"] == 102


def test_shared_width_guard():
    assert assert_shared_width({p: 9 for p in ("none", "lappe", "rwse", "signnet", "grpe")}) == 9
    try:
        assert_shared_width({"none": 9, "lappe": 25, "rwse": 29, "signnet": 41, "grpe": 9})
    except ValueError:
        return
    raise AssertionError("mismatched input widths must be rejected")


def test_rejects_bad_shapes():
    data, Toy = _fixture()
    model_fn = Toy().eval()
    for bad in (0, Q + 1):
        try:
            # num_target_nodes passed explicitly: without it the required-arg check below
            # fires first and this test would pass for the wrong reason.
            compute_sensitivity_curve(model_fn, data, n_shared_feats=bad,
                                      max_dist=MAX_D, num_target_nodes=N)
        except ValueError as exc:
            assert "n_shared_feats" in str(exc)
            continue
        raise AssertionError(f"n_shared_feats={bad} should have been rejected")


def test_num_target_nodes_is_required():
    """It has no default on purpose -- it must be calibrated per (backbone, dataset), and
    a default would reinstate the arbitrary constant scripts/calibrate_target_nodes.py
    exists to remove."""
    data, Toy = _fixture()
    model_fn = Toy().eval()
    try:
        compute_sensitivity_curve(model_fn, data, n_shared_feats=Q_SHARED, max_dist=MAX_D)
    except ValueError as exc:
        assert "calibrate_target_nodes" in str(exc)
    else:
        raise AssertionError("omitting num_target_nodes must raise")
    for bad in (0, -1):
        try:
            compute_sensitivity_curve(model_fn, data, n_shared_feats=Q_SHARED,
                                      max_dist=MAX_D, num_target_nodes=bad)
        except ValueError:
            continue
        raise AssertionError(f"num_target_nodes={bad} should have been rejected")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nall tests passed")
