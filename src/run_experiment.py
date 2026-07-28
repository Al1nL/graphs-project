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

sys.path.insert(0, os.path.dirname(__file__))
from adapters.graphgps_adapter import build_posenc_config
from adapters.san_adapter import build_san_config
from adapters.graphormer_adapter import build_graphormer_config

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


def graphgps_train(config, dataset, seed):
    raise NotImplementedError(
        "Point this at GraphGPS's own training entry point (e.g. adapt "
        "GraphGPS/main.py's run_loop_settings) once GraphGPS is cloned locally. "
        "This stub exists so run_experiment.py's CLI/orchestration logic can be reviewed "
        "and unit-tested independently of any single backbone repo being present."
    )


def san_train(config, dataset, seed):
    raise NotImplementedError(
        "Point this at SAN's main_SAN.py training entry point once SAN is cloned locally."
    )


def graphormer_train(config, dataset, seed):
    raise NotImplementedError(
        "Point this at Graphormer's graphormer/train.py (fairseq-cli based) once "
        "Graphormer is cloned locally."
    )


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
    """
    raise NotImplementedError(
        "Implement per backbone once its repo is cloned: run the trained model's forward "
        "pass up to (and excluding) the task head, returning node embeddings. For GraphGPS "
        "this is the GPSModel layer stack before `post_mp`; for SAN, the output of the "
        "final SAN layer before readout; for Graphormer, the last encoder layer's token "
        "states with the virtual/graph token dropped. See make_model_fn's docstring for "
        "the two constraints the wrapper must satisfy."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=BACKBONES)
    parser.add_argument("--pe", required=True, choices=PES)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", default=None, help="defaults to cache/<dataset>/")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    cache_dir = args.cache_dir or f"cache/{args.dataset}"
    config = build_config(args.backbone, args.pe, args.dataset, cache_dir)

    print(f"[run_experiment] backbone={args.backbone} pe={args.pe} dataset={args.dataset} "
          f"seed={args.seed}")
    print(f"[run_experiment] resolved config: {json.dumps(config, indent=2, default=str)}")

    train_fn = TRAIN_FN[args.backbone]
    # metrics, sensitivity_curve = train_fn(config, args.dataset, args.seed)
    # -- disabled until a real backbone repo is wired in; see stub NotImplementedError above

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(
        args.results_dir,
        f"{args.backbone}_{args.pe}_{args.dataset}_seed{args.seed}.json",
    )
    result = {
        "backbone": args.backbone,
        "pe": args.pe,
        "dataset": args.dataset,
        "seed": args.seed,
        "metric_name": TASK_METRIC[args.dataset],
        "metric_value": None,           # <-- FILL AFTER RUN: primary task metric (AP/MAE/F1)
        "num_params": None,             # <-- FILL AFTER RUN: trainable parameter count
        "train_time_seconds": None,     # <-- FILL AFTER RUN
        "peak_gpu_mem_mb": None,        # <-- FILL AFTER RUN
        "n_shared_feats": None,         # <-- FILL AFTER RUN: input width the Jacobian was
                                        #     taken over; must match across all 5 PE variants
        "sensitivity_curve": {},        # <-- FILL AFTER RUN: {hop_distance: {"mean":, "count":}}
                                        #     pooled over sampled graphs via average_curves
        "sensitivity_curves_per_graph": [],
        # ^ FILL AFTER RUN: one entry per sampled test graph, shaped
        #     {"curve": {d: {"mean":, "count":}}, "diameter": int, "num_nodes": int}
        #   `diameter` (sensitivity.graph_diameter) is REQUIRED for the relative-distance
        #   axis: it rebins each graph onto d/diam(G) so rho is comparable ACROSS datasets
        #   whose diameters differ ~2x, and so far buckets are not dominated by whichever
        #   graphs happen to be large enough to have them. Without it only absolute rho
        #   can be computed, and that is within-dataset only.
        #   REQUIRED for error bars: rho's confidence interval is a bootstrap that
        #   resamples whole GRAPHS, because node pairs within a graph are not independent.
        #   Without this list, aggregate_results.py can only report rho as a point estimate
        #   with no way to distinguish a real gap from sampling noise. It also lets the rho
        #   window (d_min, d_max) be varied at analysis time without re-running the probe --
        #   which matters, since that window is still provisional (see docs/analysis-plan.md).
        "status": "NOT_RUN — training stub not wired to a cloned backbone repo yet",
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[run_experiment] wrote placeholder result to {out_path}")


if __name__ == "__main__":
    main()
