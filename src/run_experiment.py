"""
run_experiment.py
==================
Single entry point for one cell of the (backbone x PE x dataset x seed) grid.

    python run_experiment.py --backbone gps --pe rwse --dataset peptides-func --seed 0

This script is deliberately a thin orchestrator: the actual model code lives in each
backbone's own official repository (see README "Environment setup" -- clone GraphGPS/SAN/
Graphormer as siblings of this repo). What this script owns:
  1. picking the right adapter (src/adapters/*) to translate the shared PE cache into that
     backbone's expected input format,
  2. calling out to that backbone's training entry point with the resulting config,
  3. after training, running the shared sensitivity probe (src/sensitivity.py) on a sample
     of test graphs,
  4. writing one JSON result file to results/<backbone>_<pe>_<dataset>_seed<seed>.json

NOTE: the calls to each backbone's own train/eval functions (`graphgps_train`,
`san_train`, `graphormer_train`) are import stubs -- point them at the actual entry points
in the cloned repos (e.g. GraphGPS's `main.py:run_loop_settings`, SAN's `main_SAN.py`,
Graphormer's `graphormer/train.py`) once those repos are on disk. Left as stubs here
because those repos are not vendored into this harness.
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from adapters.graphgps_adapter import build_posenc_config
from adapters.san_adapter import build_san_config
from adapters.graphormer_adapter import build_graphormer_config
import sensitivity
from dataset_meta import abs_rho_window, REL_BINS, REL_RHO_WINDOW

DATASETS = ["peptides-func", "peptides-struct", "pascalvoc-sp"]
PES = ["none", "lappe", "rwse", "signnet", "grpe"]
BACKBONES = ["gps", "san", "graphormer"]

TASK_METRIC = {
    "peptides-func": "ap",       # Average Precision (multi-label graph classification)
    "peptides-struct": "mae",    # Mean Absolute Error (graph regression)
    "pascalvoc-sp": "macro_f1",  # macro-F1 (node classification)
}


def build_config(backbone: str, pe: str, dataset: str, cache_dir: str) -> dict:
    if backbone == "gps":
        return build_posenc_config(pe, cache_dir)
    if backbone == "san":
        return build_san_config(pe, cache_dir)
    if backbone == "graphormer":
        return build_graphormer_config(pe, cache_dir)
    raise ValueError(backbone)


def graphgps_train(run_cfg, dataset=None, seed=None):
    """Train one grid cell with GraphGPS. Delegates to backends/graphgps_backend.py.

    Imported lazily: GraphGPS needs its own environment (yacs, pytorch_lightning, its
    pinned PyG), so importing at module scope would break the launcher's --dry-run and the
    whole test suite on any machine that has not set that env up.
    """
    from backends.graphgps_backend import graphgps_train as _train
    return _train(run_cfg)


def san_train(config, dataset, seed):
    raise NotImplementedError(
        "Point this at SAN's main_SAN.py training entry point once SAN is cloned locally."
    )


def graphormer_train(run_cfg, dataset=None, seed=None):
    """Train one grid cell with Graphormer. Delegates to backends/graphormer_backend.py.

    Imported lazily: Graphormer needs its own environment (fairseq, torch==1.9.1+cu111,
    PyG==2.2.0 -- see envs/graphormer_env.yml), so importing at module scope would break
    the launcher's --dry-run and the whole test suite on any machine that has not set that
    env up. Mirrors graphgps_train's signature exactly: launch.py's run_one() calls
    TRAIN_FN[cfg.backbone](cfg, cfg.dataset, cfg.seed), so dataset/seed are accepted but
    unused -- both already live inside run_cfg.
    """
    from backends.graphormer_backend import graphormer_train as _train
    return _train(run_cfg)


TRAIN_FN = {"gps": graphgps_train, "san": san_train, "graphormer": graphormer_train}


def make_model_fn(trained_model, backbone: str, data, pe_record):
    """Wrap a trained backbone into the `model_fn(x) -> [n, p]` callable that
    src/sensitivity.py's probe expects.

    Two requirements, both load-bearing for the PE comparison (see the input-space
    contract at the top of sensitivity.py):

    1. `x` must be laid out as [shared_original_features | PE channels], with the shared
       channels FIRST, or the PE must not be in `x` at all -- reach it via closure over
       `pe_record` instead. The probe differentiates only the leading `n_shared_feats`
       columns so that all five PE variants are measured on an identical input space;
       that slice is meaningless if PE channels are interleaved.
    2. `n_shared_feats` must be the SAME integer for all five PE variants on a given
       dataset. Derive it from the raw (un-augmented) dataset's feature width rather than
       hardcoding it, and run `sensitivity.assert_shared_width` over the five variants
       once before launching the grid.

    Return the final-layer NODE embeddings [n, p] -- not pooled graph embeddings, and not
    task logits: s_bar(d) is defined on h_v^(L).

    Returns (model_fn, probe_data, meta). `probe_data` is what to hand the probe: its `.x`
    is h^(0), the node representation AFTER the feature encoder, because LRGB node features
    are integer atom indices and d h / d x is undefined for a discrete index. `meta` carries
    the candidate input widths -- see graphgps_backend.probe_widths for why there is more
    than one and why the choice is not free.
    """
    if backbone == "gps":
        from backends.graphgps_backend import make_gps_model_fn
        return make_gps_model_fn(trained_model, data)
    if backbone == "graphormer":
        # CAVEAT not present for "gps": `data` here must already be a Graphormer-
        # preprocessed item (whatever _CachedPEGraphormerDataset.__getitem__ returns --
        # has .x, .spatial_pos, .attn_edge_type, .extra_pe if applicable), NOT a raw PyG
        # graph. make_gps_model_fn's `data` is closer to raw because GraphGPS's own
        # encoder does that preprocessing internally; Graphormer's cached-PE substitution
        # happens at the DATASET level (see graphormer_backend.py's module docstring), so
        # whatever samples test graphs to call make_model_fn on must pull them from that
        # backend's `test_dataset` (graphormer_train's return value), not from a raw
        # LRGBDataset split directly -- see sample_test_graphs below, which does exactly
        # that dispatch.
        from backends.graphormer_backend import make_graphormer_model_fn
        return make_graphormer_model_fn(trained_model, data)
    raise NotImplementedError(
        f"make_model_fn is implemented for 'gps' and 'graphormer' only; '{backbone}' "
        "still needs its repo cloned and forked. For SAN this is the output of the final "
        "SAN layer before readout. Must satisfy the two constraints above."
    )


def sample_test_graphs(train_result: dict, backbone: str, n_graphs: int, seed: int = 0):
    """Pull up to `n_graphs` individual PyG-Data-like objects from a trained cell's TEST
    split, in whatever shape `make_model_fn` expects for that backbone (see its docstring
    for why that shape differs between backbones).

    Returns (graph_ids, graphs): `graph_ids` are the TEST-SPLIT indices (stable across
    seeds -- see `main()`'s result-dict schema for why that stability matters for the
    bootstrap), parallel to `graphs`.
    """
    if backbone == "gps":
        # GraphGym's create_loader() convention (graphgps_backend.graphgps_train's
        # "loaders"): [train_loader, val_loader, test_loader]. Each loader wraps a PyG
        # Dataset directly accessible via .dataset, the same indexing style as
        # graphormer's test_dataset below -- NOT validated against a real GraphGPS run by
        # this author (no GraphGPS env here); flagged for the GraphGPS owner to confirm
        # index 2 is really the test split before trusting graph_id stability across seeds.
        test_dataset = train_result["loaders"][2].dataset
    elif backbone == "graphormer":
        test_dataset = train_result["test_dataset"]
    else:
        raise NotImplementedError(f"sample_test_graphs: backbone={backbone!r} not wired")

    n = len(test_dataset)
    rng = torch.Generator().manual_seed(seed)
    graph_ids = torch.randperm(n, generator=rng)[: min(n_graphs, n)].tolist()
    return graph_ids, [test_dataset[i] for i in graph_ids]


def run_probe(train_result: dict, backbone: str, run_cfg, n_graphs: int = 10):
    """Run the shared sensitivity probe on a sample of the trained cell's test graphs, at
    the ALREADY-CALIBRATED `run_cfg.num_target_nodes` (no sweep -- that is
    scripts/calibrate_target_nodes.py's job, run once per (backbone, dataset) beforehand).

    This is what launch.py's run_one() was missing entirely: it trained a real model but
    never called this, so `rho`/`n_shared_feats` came back empty even on a successful
    training run. Mirrors calibrate_target_nodes.py's `load_real` sampling logic, minus
    the T-sweep.

    Returns (pooled_curve, per_graph_records, n_shared_feats):
      pooled_curve       average_curves(...) over the sampled graphs -- what `long_range_
                         fraction` needs for the absolute-d rho.
      per_graph_records  one {"curve", "diameter", "num_nodes", "graph_id"} dict per
                         sampled graph -- exactly the schema main()'s result JSON
                         documents as required (bootstrap clustering, the relative-d axis,
                         re-deriving rho at analysis time for a different window). This
                         CANNOT be reconstructed after the fact, so callers should persist
                         it, not just the pooled point estimate.
      n_shared_feats     from the wrapped model_fn's meta; identical across PE variants by
                         construction (see make_model_fn's docstring) -- callers should
                         still run `sensitivity.assert_shared_width` across a dataset's
                         five PE arms once before trusting cross-PE comparisons.
    """
    if run_cfg.num_target_nodes is None:
        raise ValueError(
            "run_cfg.num_target_nodes is required to run the probe -- calibrate it first "
            "with scripts/calibrate_target_nodes.py and pass the value it reports."
        )
    graph_ids, graphs_raw = sample_test_graphs(train_result, backbone, n_graphs, run_cfg.seed)
    model = train_result["model"]

    per_graph = []
    n_shared_feats = None
    for graph_id, raw in zip(graph_ids, graphs_raw):
        model_fn, probe_data, meta = make_model_fn(model, backbone, raw, pe_record=None)
        n_shared_feats = meta["n_shared_feats"]
        curve = sensitivity.compute_sensitivity_curve(
            model_fn, probe_data, n_shared_feats=n_shared_feats,
            max_dist=run_cfg.resolved_max_dist(), num_target_nodes=run_cfg.num_target_nodes,
        )
        diameter = sensitivity.graph_diameter(probe_data.edge_index, probe_data.num_nodes)
        per_graph.append({
            "curve": curve, "diameter": diameter,
            "num_nodes": probe_data.num_nodes, "graph_id": graph_id,
        })

    pooled = sensitivity.average_curves([r["curve"] for r in per_graph])
    return pooled, per_graph, n_shared_feats


def run_cell(run_cfg, n_graphs: int = 10) -> dict:
    """Train one grid cell AND run the shared probe on it, returning the full result-JSON
    schema (see the field-by-field rationale in this function's body -- graph_id/diameter
    per graph, rho on both axes, ...).

    Shared by run_experiment.py's own CLI (`main`, below) and scripts/launch.py's
    `run_one`, so the two entry points for "run one cell" cannot drift into writing two
    different result shapes -- which is exactly what happened before this fix: `main()`
    always wrote an empty placeholder, and launch.py's run_one() trained for real but
    never called the probe at all, so neither one actually produced this schema.

    Does NOT catch exceptions -- callers decide how to handle a NotImplementedError (a
    stub backbone/dataset combination, e.g. Graphormer+pascalvoc-sp) vs. any other failure.
    """
    t0 = time.time()
    train_result = TRAIN_FN[run_cfg.backbone](run_cfg)
    result = {
        "backbone": run_cfg.backbone, "pe": run_cfg.pe, "dataset": run_cfg.dataset,
        "seed": run_cfg.seed, "metric_name": run_cfg.metric_name,
        "metric_value": train_result["metric_value"],
        "num_params": train_result["num_params"],
        "num_target_nodes": run_cfg.num_target_nodes,
        "train_time_seconds": None,  # filled below, after the probe -- see the note there
    }

    # `graph_id` (the TEST-SPLIT index) and `diameter` are recorded per graph -- see
    # run_probe's docstring for why both are load-bearing for the bootstrap and the
    # relative-distance axis, and CANNOT be reconstructed from the pooled curve alone
    # after the fact (record them on the FIRST run, or re-train to recover them).
    pooled, per_graph, n_shared = run_probe(train_result, run_cfg.backbone, run_cfg, n_graphs)
    result["n_shared_feats"] = n_shared
    result["sensitivity_curve"] = pooled
    result["sensitivity_curves_per_graph"] = per_graph

    d_min, d_max = abs_rho_window(run_cfg.dataset)
    result["rho"] = sensitivity.long_range_fraction(pooled, d_min, d_max)

    rel_curves = [
        sensitivity.to_relative_curve(r["curve"], r["diameter"], n_bins=REL_BINS)
        for r in per_graph if r["diameter"] > 0
    ]
    result["rho_rel"] = (
        sensitivity.long_range_fraction(sensitivity.average_curves(rel_curves), *REL_RHO_WINDOW)
        if rel_curves else None
    )
    # train_time_seconds intentionally covers training AND the probe: whoever reads this
    # field (e.g. launch.py's CSV) wants "how long did this cell tie up the GPU", not just
    # the training fraction of it.
    result["train_time_seconds"] = round(time.time() - t0, 2)
    result["status"] = "ok"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=BACKBONES)
    parser.add_argument("--pe", required=True, choices=PES)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", default=None, help="defaults to cache/<dataset>/")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--num-target-nodes", type=int, required=True,
                        help="from scripts/calibrate_target_nodes.py; no default by design")
    parser.add_argument("--n-graphs", type=int, default=10,
                        help="test graphs to sample for the sensitivity probe")
    args = parser.parse_args()

    from config import RunConfig  # noqa: E402 -- local import, config.py already sys.path'd

    cache_dir = args.cache_dir or f"cache/{args.dataset}"
    config = build_config(args.backbone, args.pe, args.dataset, cache_dir)

    print(f"[run_experiment] backbone={args.backbone} pe={args.pe} dataset={args.dataset} "
          f"seed={args.seed}")
    print(f"[run_experiment] resolved config: {json.dumps(config, indent=2, default=str)}")

    run_cfg = RunConfig(backbone=args.backbone, pe=args.pe, dataset=args.dataset,
                        seed=args.seed, cache_dir=args.cache_dir,
                        results_dir=args.results_dir,
                        num_target_nodes=args.num_target_nodes)

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = run_cfg.result_path
    try:
        result = run_cell(run_cfg, n_graphs=args.n_graphs)
    except NotImplementedError as exc:
        # the training entry point (or the probe wiring) is a stub for this backbone --
        # say so rather than writing a placeholder that looks like a real result
        result = {
            "backbone": args.backbone, "pe": args.pe, "dataset": args.dataset,
            "seed": args.seed, "metric_name": TASK_METRIC[args.dataset],
            "metric_value": None, "status": f"not_implemented: {exc}",
        }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[run_experiment] wrote {result['status']} result to {out_path}")


if __name__ == "__main__":
    main()
