"""
calibrate_target_nodes.py
=========================
One-off convergence check for the probe's `num_target_nodes` (T).

    # runnable today, no trained model needed -- verifies the machinery end to end
    python scripts/calibrate_target_nodes.py --demo

    # the real thing, once a backbone is wired up (see run_experiment.make_model_fn)
    python scripts/calibrate_target_nodes.py --backbone gps --pe rwse \
        --dataset peptides-func --checkpoint path/to/ckpt.pt

Sweeps T over a ladder, computes rho at each rung with a graph-clustered bootstrap CI, and
reports the smallest T past which rho has stopped moving relative to that CI. Writes:

  results/calibration_target_nodes[_<tag>].csv   one row per rung
  results/calibration_target_nodes[_<tag>].png   rho vs T, CI bars, convergence band

Run it ONCE per (backbone, dataset) -- it is a property of the probe and the graph regime,
not of the PE -- then put the chosen T in your run config and quote the printed sentence.

Cost: the ladder totals sum(ladder) target-probes per graph, ~2x a single probe at the
largest rung. Ten graphs is plenty; this is a convergence check, not an estimate of rho.
"""

import argparse
import os
import sys
import types

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from calibration import (  # noqa: E402
    DEFAULT_LADDER,
    recommend_target_nodes,
    report_sentence,
    sweep_target_nodes,
)


# ---------------------------------------------------------------------------
# Demo mode: synthetic peptide-like graphs + an untrained toy backbone.
#
# This exists so the calibration machinery can be exercised and tested before any backbone
# repo is cloned. The T it recommends is NOT transferable to a real run -- an untrained
# model has a different Jacobian structure than a trained one, and these graphs are not
# Peptides. Use it to check the pipeline works, then re-run for real.
# ---------------------------------------------------------------------------
Q_SHARED, Q_PE, P_HIDDEN = 9, 16, 96


def _demo_graphs(n_graphs=10, seed=0):
    """Chain-dominated graphs with a few chords -- long diameter, like a peptide.

    Sized 160-320 nodes so the default ladder's top rung (T=128) does not saturate: once
    T >= n the rung samples every node, making it a different estimator rather than denser
    sampling, and therefore a poor reference to measure convergence against. Peptides
    averages ~151 nodes, so this is also the right regime.
    """
    g = torch.Generator().manual_seed(seed)
    graphs = []
    for _ in range(n_graphs):
        n = int(torch.randint(160, 320, (1,), generator=g))
        edges = [[i, i + 1] for i in range(n - 1)]
        for _ in range(n // 12):  # sparse chords, keeps the diameter long
            a = int(torch.randint(0, n, (1,), generator=g))
            b = int(torch.randint(0, n, (1,), generator=g))
            if a != b:
                edges.append([a, b])
        ei = torch.tensor(edges + [[b, a] for a, b in edges]).t()
        x = torch.randn(n, Q_SHARED + Q_PE, generator=g)
        graphs.append(types.SimpleNamespace(x=x, edge_index=ei, num_nodes=n))
    return graphs


class _ToyBackbone(nn.Module):
    """Local mixing + global attention, i.e. a GPS-shaped Jacobian. Untrained."""

    def __init__(self, q_in, p=P_HIDDEN):
        super().__init__()
        self.inp = nn.Linear(q_in, p)
        self.qkv = nn.Linear(p, 3 * p)
        self.out = nn.Linear(p, p)
        self.p = p

    def forward(self, x, adj):
        h = torch.tanh(self.inp(x))
        h = torch.tanh(adj @ h) + h
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        attn = torch.softmax(q @ k.t() / self.p**0.5, dim=-1)
        return self.out(attn @ v + h)


def _demo_factory(seed=0):
    torch.manual_seed(seed)
    model = _ToyBackbone(Q_SHARED + Q_PE).eval()

    def factory(data):
        adj = torch.zeros(data.num_nodes, data.num_nodes)
        adj[data.edge_index[0], data.edge_index[1]] = 1.0
        adj = adj / adj.sum(1, keepdim=True).clamp(min=1)
        return lambda x: model(x, adj)

    return factory


def load_real(backbone, pe, dataset, checkpoint, n_graphs):
    """Load a trained checkpoint and return (model_fn_factory, graphs, n_shared_feats).

    THE CONTRACT sweep_target_nodes ACTUALLY NEEDS, READ THIS BEFORE CHANGING IT: it calls
    `compute_sensitivity_curve(model_fn_factory(data), data, ...)` -- the SAME `data` object
    is used both to build model_fn AND as the thing whose `.x` gets differentiated. Demo
    mode satisfies this because its synthetic graphs are already continuous (`data.x =
    torch.randn(...)`). GraphGPS's real input is NOT: LRGB node features are integer
    atom-type indices consumed by an nn.Embedding lookup, so d h / d x is undefined for
    them (see backends/graphgps_backend.py's header) -- the probe must instead differentiate
    h^(0), the encoder's OUTPUT. `graphs` below is therefore a list of h^(0)-space
    `probe_data` objects (built once via run_experiment.make_model_fn), each with its
    matching `model_fn` attached as an attribute, so `factory(data)` can just read it back
    off `data` rather than needing the original raw graph in scope.

    Only "gps" is wired, matching run_experiment.PROBE_WIRED_BACKBONES. `checkpoint` must be
    a state_dict saved by GraphGPS's own training loop; `build_graphgym_cfg` reconstructs
    the exact architecture that state_dict was trained under from (pe, dataset) -- passing
    the wrong `--pe` for a given checkpoint loads weights into a differently-shaped model
    and fails loudly (via `load_state_dict(strict=False)`'s missing/unexpected keys), not
    silently.
    """
    if backbone != "gps":
        raise NotImplementedError(
            f"load_real is wired for 'gps' only (matches run_experiment.PROBE_WIRED_"
            f"BACKBONES); '{backbone}' still needs its own probe wrapper. Use --demo to "
            "exercise the calibration pipeline itself in the meantime."
        )
    if not checkpoint:
        raise ValueError(
            "--checkpoint is required for backbone=gps: this loads a TRAINED model's "
            "weights, it does not train one. Point it at the checkpoint GraphGPS's own "
            "training loop wrote for this exact (pe, dataset) -- see cfg.out_dir in "
            "backends.graphgps_backend.build_graphgym_cfg."
        )

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    from config import RunConfig
    from run_experiment import make_model_fn, sample_test_graphs
    from backends.graphgps_backend import build_graphgym_cfg, ensure_graphgps_importable

    graphgps_dir = ensure_graphgps_importable()
    from torch_geometric.graphgym.loader import create_loader
    from torch_geometric.graphgym.model_builder import create_model

    # seed=0 here reconstructs the ARCHITECTURE only -- calibration is a property of the
    # probe and the graph regime, not of the training seed, so which seed's checkpoint you
    # point at should not matter, and it is not recorded as part of what this returns.
    run_cfg = RunConfig(backbone=backbone, pe=pe, dataset=dataset, seed=0)
    build_graphgym_cfg(run_cfg, graphgps_dir)
    loaders = create_loader()
    model = create_model()

    import torch
    state = torch.load(checkpoint, map_location="cpu")
    state_dict = state.get("model_state", state) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint at {checkpoint} does not match the architecture built for "
            f"(pe={pe!r}, dataset={dataset!r}): missing={missing}, unexpected={unexpected}. "
            "This usually means --pe doesn't match what the checkpoint was trained with."
        )
    model.eval()

    test_dataset = loaders[-1].dataset
    raw_graphs = [data for _, data in sample_test_graphs(test_dataset, n_graphs, seed=0)]

    graphs = []
    n_shared_feats = None
    for raw in raw_graphs:
        model_fn, probe_data, meta = make_model_fn(model, backbone, raw)
        probe_data.model_fn = model_fn   # stashed so factory() below needs no extra state
        graphs.append(probe_data)
        if n_shared_feats is None:
            n_shared_feats = meta["dim_inner"]

    def factory(data):
        return data.model_fn

    return factory, graphs, n_shared_feats


def load_real_san(pe, dataset, seed, results_dir, n_graphs):
    """SAN equivalent of load_real, above. Same contract: returns (factory, graphs,
    n_shared_feats), where `graphs` are h^(0)-space probe_data objects with their
    matching model_fn stashed on them as an attribute, so factory(data) just reads it
    back off data -- see load_real's own docstring for why that shape is required by
    sweep_target_nodes.

    Unlike GPS's path, this does NOT take an explicit --checkpoint argument: SAN's
    san_train persists final weights itself, at a predictable path
    (results/model_san_<pe>_<dataset>_seed<seed>.pt -- see san_backend.py's
    changelog for why this didn't exist until now), so this just needs (pe, dataset,
    seed) to find it, the same way `run_experiment.py` finds a run's result JSON.

    Loads net_params from the SAME file the weights were saved with (also written by
    san_train), so the reconstructed architecture is guaranteed consistent with the
    trained weights -- no separate "pass --pe and hope it matches" step like GPS's
    load_real needs, since that information is bundled with the checkpoint here
    rather than being reconstructed from scratch via build_graphgym_cfg.
    """
    model_path = os.path.join(
        results_dir, f"model_san_{pe}_{dataset}_seed{seed}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"no saved model at {model_path}. This is written by san_train on "
            f"completion (see san_backend.py's changelog) -- run "
            f"`python src/run_experiment.py --backbone san --pe {pe} --dataset "
            f"{dataset} --seed {seed} ...` to completion first, or check --results-dir "
            "if that run used a non-default one."
        )

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    from config import RunConfig
    from run_experiment import make_model_fn, sample_test_graphs
    from backends.san_backend import (
        _build_san_model, ensure_san_importable, san_train as _san_train_module,
    )
    ensure_san_importable()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(model_path, map_location=device)
    net_params = saved["net_params"]
    net_params["device"] = device  # stripped before saving (not picklable-safe
                                    # across processes/machines); re-add here

    model = _build_san_model(net_params).to(device)
    missing, unexpected = model.load_state_dict(saved["model_state"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"saved model at {model_path} does not match the architecture "
            f"_build_san_model reconstructs from its own bundled net_params: "
            f"missing={missing}, unexpected={unexpected}. This would mean the "
            "architecture-building code (_build_san_model / build_san_net_params) "
            "changed since this model was saved -- retrain, or check that this "
            "checkpoint's net_params still round-trips through the current code."
        )
    model.eval()

    # run_cfg is needed only to build a _PEAttachedDataset-backed test split matching
    # what this model was trained on (max_nodes filtering, full_graph flag, etc.) --
    # NOT to retrain anything. seed=0 here mirrors load_real's own note: calibration
    # is a property of the probe/graph regime, not the training seed, so which
    # seed's saved model is used for calibration doesn't need to match run_cfg.seed
    # -- but the ACTUAL trained seed IS used (not hardcoded to 0) since a real
    # calibration run should reflect a real trained model, unlike GPS's demo-adjacent
    # note about seed being irrelevant to architecture reconstruction specifically.
    run_cfg = RunConfig(backbone="san", pe=pe, dataset=dataset, seed=seed)
    from backends.san_backend import build_san_net_params, build_san_train_params, _build_loaders
    built_net_params = build_san_net_params(run_cfg)
    train_params = build_san_train_params(run_cfg)
    _, _, _, _, probe_dataset = _build_loaders(run_cfg, built_net_params, train_params)

    raw_graphs = [probe_dataset[i] for i in
                  torch.randperm(len(probe_dataset),
                                 generator=torch.Generator().manual_seed(0))[:n_graphs].tolist()]

    graphs = []
    n_shared_feats = None
    for raw in raw_graphs:
        model_fn, probe_data, meta = make_model_fn(model, "san", raw)
        probe_data.model_fn = model_fn
        graphs.append(probe_data)
        if n_shared_feats is None:
            n_shared_feats = meta["dim_inner"]

    def factory(data):
        return data.model_fn

    return factory, graphs, n_shared_feats


def plot(rows, rec, out_png, d_min, d_max, title_extra=""):
    rows = sorted(rows, key=lambda r: r["T"])
    ts = [r["T"] for r in rows]
    rhos = [r["rho"] for r in rows]
    lo = [r["rho"] - r["rho_ci_lo"] for r in rows]
    hi = [r["rho_ci_hi"] - r["rho"] for r in rows]

    plt.figure(figsize=(7.5, 4.8))
    ref_rho = rows[-1]["rho"]
    if rec["band"] == rec["band"]:
        plt.axhspan(ref_rho - rec["band"], ref_rho + rec["band"], color="tab:green",
                    alpha=0.15, label=f"$\\rho(T_{{max}}) \\pm {rec['tol']:g}\\times$ CI half-width")
    plt.axhline(ref_rho, color="tab:green", lw=1, ls="--", alpha=0.7)
    plt.errorbar(ts, rhos, yerr=[lo, hi], marker="o", ms=5, capsize=4, lw=1.5,
                 color="tab:blue", label=r"$\rho(T)$ with graph-clustered 95% CI")
    if rec["converged"]:
        plt.axvline(rec["recommended_T"], color="tab:red", lw=1.5, ls=":",
                    label=f"recommended $T = {rec['recommended_T']}$")
    plt.xscale("log", base=2)
    plt.xticks(ts, [str(t) for t in ts])
    plt.xlabel("target nodes sampled per graph, $T$")
    plt.ylabel(rf"$\rho$   (window $d \in [{d_min}, {d_max}]$)")
    plt.title(f"Convergence of $\\rho$ in target-node budget{title_extra}")
    plt.grid(alpha=0.25, lw=0.5)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="run on synthetic graphs with an untrained toy backbone")
    ap.add_argument("--backbone"), ap.add_argument("--pe"), ap.add_argument("--dataset")
    ap.add_argument("--checkpoint",
                    help="required for --backbone gps: path to a GraphGPS-written "
                         "checkpoint. NOT used for --backbone san, which instead "
                         "loads results/model_san_<pe>_<dataset>_seed<seed>.pt, "
                         "written automatically by a completed san_train run -- "
                         "see --seed / --results-dir below.")
    ap.add_argument("--results-dir", default="results",
                    help="--backbone san only: directory to look for the saved "
                         "model_san_*.pt file in, matching whatever --results-dir "
                         "the original training run used (default: results).")
    ap.add_argument("--n-graphs", type=int, default=10)
    ap.add_argument("--ladder", type=int, nargs="+", default=list(DEFAULT_LADDER))
    ap.add_argument("--max-dist", type=int, default=20)
    ap.add_argument("--d-min", type=int, default=5)
    ap.add_argument("--d-max", type=int, default=20)
    ap.add_argument("--tol", type=float, default=0.5,
                    help="accept T when rho is within tol x the reference CI half-width")
    ap.add_argument("--max-ci-inflation", type=float, default=0.15,
                    help="reject a rung whose own bootstrap CI is more than this fraction "
                         "wider than the reference rung's -- unbiased but underpowered")
    ap.add_argument("--min-bucket", type=int, default=5,
                    help="reject a rung whose sparsest distance bucket holds fewer than "
                         "this many pairs, however stable rho looks there")
    ap.add_argument("--n-boot", type=int, default=500)
    # --seed: for --demo, seeds the synthetic-graph RNG. For --backbone san,
    # ALSO selects which seed's saved model to load
    # (results/model_san_<pe>_<dataset>_seed<SEED>.pt) -- reuses this single
    # flag rather than adding a second one, since argparse rejects duplicate
    # --seed registrations (this WAS a duplicate before being merged in).
    # Ignored for --backbone gps, which takes an explicit --checkpoint path.
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    if args.demo:
        factory = _demo_factory(args.seed)
        graphs, n_shared, tag = _demo_graphs(args.n_graphs, args.seed), Q_SHARED, "demo"
        print(f"DEMO MODE: {len(graphs)} synthetic graphs, untrained toy backbone.\n"
              f"The recommended T is NOT transferable to a real run -- this only checks "
              f"that the calibration pipeline works.\n")
    else:
        missing = [f for f in ("backbone", "pe", "dataset") if not getattr(args, f)]
        if missing:
            ap.error(f"--{', --'.join(missing)} required (or pass --demo)")
        if args.backbone == "san":
            factory, graphs, n_shared = load_real_san(
                args.pe, args.dataset, args.seed, args.results_dir, args.n_graphs
            )
        else:
            factory, graphs, n_shared = load_real(
                args.backbone, args.pe, args.dataset, args.checkpoint, args.n_graphs
            )
        tag = f"{args.backbone}_{args.pe}_{args.dataset}"

    print(f"Sweeping T over {args.ladder} on {len(graphs)} graphs "
          f"(rho window d in [{args.d_min}, {args.d_max}], max_dist={args.max_dist})")
    rows = sweep_target_nodes(
        factory, graphs, n_shared_feats=n_shared, ladder=args.ladder,
        max_dist=args.max_dist, d_min=args.d_min, d_max=args.d_max,
        n_boot=args.n_boot, seed=args.seed,
    )
    rec = recommend_target_nodes(rows, tol=args.tol, min_bucket=args.min_bucket,
                                 max_ci_inflation=args.max_ci_inflation)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"calibration_target_nodes_{tag}")
    pd.DataFrame(rows).to_csv(f"{stem}.csv", index=False)
    print(f"\nWrote {stem}.csv")
    plot(rows, rec, f"{stem}.png", args.d_min, args.d_max,
         title_extra=f"  ({tag})" if tag != "demo" else "  (demo)")

    print(f"\n{'=' * 78}")
    print(f"Recommended T = {rec['recommended_T']}   "
          f"(reference rung T={rec['reference_T']}, converged={rec['converged']}, "
          f"binding constraint: {rec['limited_by']})")
    print(f"  {rec['reason']}")
    print(f"\nFor the paper:\n  {report_sentence(rec, rows, args.d_min, args.d_max)}")
    print("=" * 78)

    sat = [r for r in rows if r["graphs_saturated"] == r["n_graphs"]]
    if sat:
        print(f"\nNOTE: rungs {[r['T'] for r in sat]} exceed every calibration graph's node "
              "count, so they sample all nodes and add no information. Treat any apparent "
              "convergence there as an artefact of saturation, not of stability.")
    thin = [r for r in rows if r["min_bucket_count"] < 5]
    if thin:
        print(f"\nNOTE: rungs {[r['T'] for r in thin]} leave at least one distance bucket "
              "with <5 pairs. Sparse tail buckets are exactly where the paper's claims "
              "live -- check `min_bucket_count` in the CSV before choosing a small T.")


if __name__ == "__main__":
    main()
