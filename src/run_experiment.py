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
        "sensitivity_curve": {},        # <-- FILL AFTER RUN: {hop_distance: mean_jacobian_norm}
        "status": "NOT_RUN — training stub not wired to a cloned backbone repo yet",
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[run_experiment] wrote placeholder result to {out_path}")


if __name__ == "__main__":
    main()
