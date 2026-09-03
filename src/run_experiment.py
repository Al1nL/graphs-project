"""
run_experiment.py
==================
Single entry point for one cell of the (backbone x PE x dataset x seed) grid.

    python run_experiment.py --backbone gps --pe rwse --dataset peptides-func --seed 0 \
        --num-target-nodes 32

This script is deliberately a thin orchestrator: the actual model code lives in each
backbone's own official repository (see README "Environment setup" -- clone GraphGPS/SAN/
Graphormer as siblings of this repo). What this script owns:
  1. picking the right adapter (src/adapters/*) to translate the shared PE cache into that
     backbone's expected input format,
  2. calling out to that backbone's training entry point with the resulting config,
  3. after training, running the shared sensitivity probe (src/sensitivity.py) on a sample
     of test graphs,
  4. writing one JSON result file to results/<backbone>_<pe>_<dataset>_seed<seed>.json

--------------------------------------------------------------------------------------
FIX (this pass): run_cell() was missing -- train_fn was never actually called
--------------------------------------------------------------------------------------
Before this fix, `main()` built a config, printed it, and then had the one line that would
call `train_fn` commented out -- it unconditionally wrote a JSON with every metric set to
None, regardless of what backbone was requested or whether training was even attempted.
Separately, `scripts/launch.py`'s `run_one()` called `TRAIN_FN[cfg.backbone](...)` directly
(bypassing this file's `main()` entirely) but threw away the returned dict, and never
invoked the sensitivity probe at all. Net effect: a real grid run would train GraphGPS
models correctly and then silently discard every metric and curve -- `results/*.json`
would stay empty (or full of `NOT_RUN` placeholders), so `aggregate_results.py` had nothing
to read and `launch.py --resume` could never see a cell as complete.

`run_cell()` below is the fix: it actually calls `train_fn`, and -- when a probe wrapper
exists for the backbone (today: "gps" only) -- samples `num_probe_graphs` test graphs from
the trained model's own test loader, runs `sensitivity.compute_sensitivity_curve` on each,
pools them, and writes the fully populated result. Both `main()` and
`scripts/launch.py:run_one()` now call this one function, so there is a single code path
that produces a result file instead of two half-implementations that silently diverged.

For backbones without a probe wrapper yet (san, graphormer), the task metric and parameter
count are still recorded for real; only `sensitivity_curve` stays empty, with `status`
saying exactly why, so a partially-wired backbone still gives you a real task-metric number
rather than nothing.

--------------------------------------------------------------------------------------
EVERY BACKBONE NOW SEES THE SAME PE (this used to be false for the GraphGPS arm)
--------------------------------------------------------------------------------------
GraphGPS's `posenc_LapPE`/`RWSE`/`SignNet` encoders compute the PE internally from the raw
graph, so for a while `graphgps_train` trained on GraphGPS's OWN LapPE while san_backend
trained on `src/pe/cache.py`'s -- and those are different encodings, not two spellings of
one. The comparison the whole study rests on was confounded at its root.

`backends/graphgps_pe_cache.py` closes it by replacing GraphGPS's PE PRE-TRANSFORM (not
its encoders, which are untouched) with a reader over this repo's cache. Four conventions
had to be reconciled to do that honestly -- Laplacian normalisation, whether the trivial
eigenvector is kept, eigenvector count, and zero- vs NaN-padding, the last of which would
have corrupted every small graph silently. That file's header is the authority on what was
reconciled and why our definition is the one that wins.

Two consequences worth knowing before reading a result:
  * `laplacian_norm` and `eigvec_norm` in GraphGPS's reference YAMLs are now INERT. They
    configured a computation that no longer runs.
  * The GPS arm no longer trains on the PE its tuned hyperparameters were tuned against,
    so its task metric may move relative to published GraphGPS numbers. That is the
    intended trade -- a tuned but incomparable number answers no question this study asks.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from adapters.graphgps_adapter import build_posenc_config
from adapters.san_adapter import build_san_config
from adapters.graphormer_adapter import build_graphormer_config
from config import PROBE_N_GRAPHS, RunConfig
from sensitivity import average_curves, compute_sensitivity_curve, graph_diameter

DATASETS = ["peptides-func", "peptides-struct", "pascalvoc-sp"]
PES = ["none", "lappe", "rwse", "signnet", "grpe"]
BACKBONES = ["gps", "san", "graphormer"]

TASK_METRIC = {
    "peptides-func": "ap",       # Average Precision (multi-label graph classification)
    "peptides-struct": "mae",    # Mean Absolute Error (graph regression)
    "pascalvoc-sp": "macro_f1",  # macro-F1 (node classification)
}

# Backbones with a working sensitivity-probe wrapper (run_experiment.make_model_fn). Kept
# as an explicit set rather than a try/except around make_model_fn, so a backbone that is
# wired for training but NOT yet for the probe (see san_train once implemented) can still
# report a real task metric while sensitivity_curve honestly stays empty.
PROBE_WIRED_BACKBONES = {"gps"}


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


def san_train(run_cfg, dataset=None, seed=None):
    """Train one grid cell with SAN. Delegates to backends/san_backend.py.

    Imported lazily: SAN needs its own environment (DGL + its pinned PyTorch/CUDA combo),
    so importing at module scope would break the launcher's --dry-run and the whole test
    suite on any machine that has not set that env up -- same reasoning as graphgps_train.
    """
    from backends.san_backend import san_train as _train
    return _train(run_cfg)


def graphormer_train(run_cfg, dataset=None, seed=None):
    raise NotImplementedError(
        "Point this at Graphormer's graphormer/train.py (fairseq-cli based) once "
        "Graphormer is cloned locally."
    )


TRAIN_FN = {"gps": graphgps_train, "san": san_train, "graphormer": graphormer_train}


def make_model_fn(trained_model, backbone: str, data, pe_record=None):
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
    if backbone == "san":
        from backends.san_backend import make_san_model_fn
        return make_san_model_fn(trained_model, data)   # raises NotImplementedError itself
    raise NotImplementedError(
        f"make_model_fn is implemented for 'gps' only; '{backbone}' still needs its repo "
        "cloned and forked. For Graphormer this is the last encoder layer's token states "
        "with the virtual/graph token dropped. It must satisfy the two constraints above."
    )


def sample_test_graphs(test_dataset, n_graphs: int, seed: int):
    """Deterministically sample up to `n_graphs` individual PyG Data objects from a
    backbone's own test-split dataset object.

    Deliberately drawn from the SAME dataset object the trained model's own loader used
    (`train_result["loaders"][-1].dataset` for GraphGPS/GraphGym), not from a fresh
    `LRGBDataset(...)` call -- the model was trained against whatever pre_transform that
    loader applied (e.g. GraphGPS's own PE computation, see the module docstring's
    disclosed limitation), and probing against a differently-transformed copy of the same
    graphs would silently reintroduce exactly the kind of PE-definition mismatch this
    project exists to avoid.

    Returns a list of (graph_id, data) pairs. `graph_id` is the plain index into the test
    dataset and MUST be recorded alongside each curve: the same graphs are probed under
    every training seed, so it is what lets aggregate_results.py's bootstrap cluster on
    "this molecule", not "this molecule at this seed" (see run_experiment's per-graph
    schema below, and sensitivity.bootstrap_over_graphs).
    """
    import torch

    n = len(test_dataset)
    k = min(n_graphs, n)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:k].tolist()
    return [(i, test_dataset[i]) for i in idx]


def run_probe(trained_model, backbone: str, test_dataset, run_cfg) -> dict:
    """Run the shared sensitivity probe over a sample of test graphs for one trained model.

    Returns {"pooled_curve", "per_graph", "n_shared_feats_used", "n_shared_feats_note"}.
    Callers that only have a task metric to report (backbone not in PROBE_WIRED_BACKBONES)
    should skip this entirely rather than call it -- it raises NotImplementedError via
    make_model_fn otherwise, which is correct but not a useful way to find that out.
    """
    import time

    graphs = sample_test_graphs(test_dataset, run_cfg.resolved_num_probe_graphs(),
                                 run_cfg.seed)
    max_dist = run_cfg.resolved_max_dist()
    per_graph = []
    n_shared_feats_used = None

    # The probe is by far the most expensive part of a cell, and it used to run silently:
    # one Jacobian per (graph, target node) means len(graphs) x num_target_nodes x
    # dim_inner backward passes through the full layer stack, which is tens of minutes on
    # a real grid cell. Without output that is indistinguishable from a hang -- and the
    # natural response to an apparent hang is to kill the job, losing the training too.
    print(f"  probe: {len(graphs)} graphs x {run_cfg.num_target_nodes} target nodes, "
          f"max_dist={max_dist}", flush=True)
    t0 = time.time()
    report_every = max(1, len(graphs) // 20)

    for i, (graph_id, data) in enumerate(graphs):
        model_fn, probe_data, meta = make_model_fn(trained_model, backbone, data)
        if n_shared_feats_used is None:
            # RECOMMENDATION from graphgps_backend.probe_widths(): use dim_inner, the width
            # that is IDENTICAL across all five PE variants for this backbone/dataset, so
            # raw Frobenius norms stay comparable. Recorded in the result so the choice is
            # visible rather than buried in this call.
            n_shared_feats_used = meta["dim_inner"]
        curve = compute_sensitivity_curve(
            model_fn, probe_data, n_shared_feats=n_shared_feats_used, max_dist=max_dist,
            num_target_nodes=run_cfg.num_target_nodes, seed=run_cfg.seed,
        )
        diam = graph_diameter(probe_data.edge_index, probe_data.num_nodes)
        per_graph.append({"graph_id": int(graph_id), "curve": curve,
                          "diameter": diam, "num_nodes": int(probe_data.num_nodes)})

        done = i + 1
        if done % report_every == 0 or done == len(graphs):
            elapsed = time.time() - t0
            remaining = elapsed / done * (len(graphs) - done)
            print(f"  probe: {done}/{len(graphs)} graphs, {elapsed:.0f}s elapsed, "
                  f"~{remaining:.0f}s remaining", flush=True)
    return {
        "pooled_curve": average_curves([r["curve"] for r in per_graph]),
        "per_graph": per_graph,
        "n_shared_feats_used": n_shared_feats_used,
        "n_shared_feats_note": (
            "dim_inner (full backbone hidden width, identical across all 5 PE variants); "
            "see backends/graphgps_backend.probe_widths for the content-only alternative "
            "this run did NOT take"
        ),
    }


def run_cell(run_cfg: RunConfig) -> dict:
    """Train one grid cell, probe it if the backbone supports the probe, and write the
    full result JSON to run_cfg.result_path. Returns the same dict it wrote.

    THIS is the function that was missing before this fix (see module docstring): both
    `main()` below and `scripts/launch.py:run_one()` now call it, so there is exactly one
    code path that turns a RunConfig into a result file instead of two that silently
    diverged (one that never called train_fn, one that called it and threw the answer
    away). A failed or not-yet-implemented backbone still raises -- this function does not
    swallow exceptions; that is `launch.py`'s job, so one failing cell doesn't kill a grid.
    """
    os.makedirs(run_cfg.results_dir, exist_ok=True)
    result = {
        "backbone": run_cfg.backbone, "pe": run_cfg.pe, "dataset": run_cfg.dataset,
        "seed": run_cfg.seed, "metric_name": run_cfg.metric_name,
        "config_hash": run_cfg.config_hash(),
        # Stamped on EVERY record, not just smoke ones, so "no field" means "written
        # before this existed" rather than "definitely a real run". A filename suffix
        # alone would not survive a copy or a rename; this travels with the data.
        "smoke_test": bool(run_cfg.smoke_test),
    }

    train_out = TRAIN_FN[run_cfg.backbone](run_cfg, run_cfg.dataset, run_cfg.seed)
    result["metric_value"] = train_out.get("metric_value")
    result["num_params"] = train_out.get("num_params")

    if run_cfg.backbone in PROBE_WIRED_BACKBONES:
        loaders = train_out["loaders"]
        test_dataset = loaders[-1].dataset  # GraphGym's create_loader(): [train, val, test]
        probe_out = run_probe(train_out["model"], run_cfg.backbone, test_dataset, run_cfg)
        result["n_shared_feats"] = probe_out["n_shared_feats_used"]
        result["n_shared_feats_note"] = probe_out["n_shared_feats_note"]
        result["sensitivity_curve"] = probe_out["pooled_curve"]
        result["sensitivity_curves_per_graph"] = probe_out["per_graph"]
        result["num_target_nodes"] = run_cfg.num_target_nodes
        result["max_dist"] = run_cfg.resolved_max_dist()
        result["status"] = "ok"
    else:
        result["n_shared_feats"] = None
        result["sensitivity_curve"] = {}
        result["sensitivity_curves_per_graph"] = []
        result["status"] = (
            f"trained_but_not_probed: '{run_cfg.backbone}' has no sensitivity-probe "
            "wrapper yet (see run_experiment.PROBE_WIRED_BACKBONES / make_model_fn). "
            "metric_value and num_params are real; sensitivity_curve is intentionally "
            "empty rather than fabricated."
        )

    with open(run_cfg.result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[run_experiment] wrote {run_cfg.result_path}  status={result['status']}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=BACKBONES)
    parser.add_argument("--pe", required=True, choices=PES)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", default=None, help="defaults to cache/<dataset>/")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--num-target-nodes", type=int, default=None,
                        help="required for a real (non-dry) run; see "
                             "scripts/calibrate_target_nodes.py")
    parser.add_argument("--num-probe-graphs", type=int, default=None,
                        help=f"default {PROBE_N_GRAPHS} (config.PROBE_N_GRAPHS)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="override the backbone's own per-dataset default (e.g. "
                             "san_backend.BASE_NET_PARAMS); useful for probing a card's "
                             "OOM ceiling without editing source. Not persisted anywhere "
                             "but this run's own config_hash/provenance.")
    parser.add_argument("--edge-budget", type=int, default=None,
                        help="SAN full_graph=True only: override the per-batch edge-count "
                             "cap (see san_backend.EdgeBudgetBatchSampler). Pass 0 to "
                             "disable edge-budget batching and fall back to fixed "
                             "--batch-size.")
    parser.add_argument("--max-nodes", type=int, default=None,
                        help="SAN full_graph=True only: exclude graphs with more nodes "
                             "than this from every split (a disclosed compromise -- "
                             "logs excluded count/fraction). Pass 0 to explicitly "
                             "disable filtering.")
    parser.add_argument("--accumulation-steps", type=int, default=None,
                        help="SAN only: accumulate gradients over this many physical "
                             "mini-batches per optimizer step, to recover a larger "
                             "EFFECTIVE batch's training statistics without raising peak "
                             "memory. Does not affect --batch-size/--edge-budget.")
    parser.add_argument("--no-grad-checkpointing", action="store_true",
                        help="disable SAN's gradient checkpointing on full_graph=True "
                             "datasets (on by default there -- see "
                             "san_backend.enable_gradient_checkpointing). Only meaningful "
                             "for --backbone san.")
    parser.add_argument("--amp", action="store_true",
                        help="enable SAN's mixed-precision (AMP) training. OFF by "
                             "default: this SAN env's pinned DGL has no fp16-capable "
                             "compiled CUDA kernel (confirmed: DGLError 'Data type not "
                             "recognized with bits 16' from a real training crash -- see "
                             "san_backend.py's use_amp comment). Only enable if you've "
                             "confirmed your DGL build supports it. Only meaningful for "
                             "--backbone san.")
    parser.add_argument("--lr", type=float, default=None,
                        help="override TRAIN_PARAMS['init_lr']. SAN only.")
    parser.add_argument("--gamma", type=float, default=None,
                        help="override net_params['gamma']. SAN only.")
    parser.add_argument("--dropout", type=float, default=None,
                        help="override net_params['dropout'] and in_feat_dropout. SAN only.")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="override TRAIN_PARAMS['weight_decay']. SAN only.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override TRAIN_PARAMS['epochs'] for this run only. "
                             "Useful for quick smoke tests (--epochs 1) without editing "
                             "source. None means use the backend's own default.")
    parser.add_argument("--early-stop-patience", type=int, default=15,
                        help="stop training if best_metric hasn't improved in this many "
                             "epochs. Pass 0 to disable early stopping entirely. Default "
                             "15 (~1.5x the LR scheduler's own patience of 10, so a LR "
                             "drop gets a chance to help before giving up).")
    parser.add_argument("--smoke-test", action="store_true",
                        help="run just 2 batches of train+val+test to verify shapes, "
                             "then exit without saving results. CPU-only friendly. "
                             "Implies --epochs 1.")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved adapter config and exit; train nothing")
    args = parser.parse_args()

    cache_dir = args.cache_dir or f"cache/{args.dataset}"
    adapter_config = build_config(args.backbone, args.pe, args.dataset, cache_dir)
    print(f"[run_experiment] backbone={args.backbone} pe={args.pe} dataset={args.dataset} "
          f"seed={args.seed}")
    print(f"[run_experiment] resolved adapter config: "
          f"{json.dumps(adapter_config, indent=2, default=str)}")

    if args.dry_run:
        return

    if args.num_target_nodes is None:
        raise SystemExit(
            "--num-target-nodes is required for a real run (no default by design -- see "
            "sensitivity.compute_sensitivity_curve). Calibrate it first with "
            "scripts/calibrate_target_nodes.py and pass the value it recommends."
        )

    run_cfg = RunConfig(
        backbone=args.backbone, pe=args.pe, dataset=args.dataset, seed=args.seed,
        cache_dir=args.cache_dir, results_dir=args.results_dir,
        num_target_nodes=args.num_target_nodes, num_probe_graphs=args.num_probe_graphs,
        batch_size=args.batch_size, grad_checkpointing=not args.no_grad_checkpointing,
        edge_budget=args.edge_budget, use_amp=args.amp, lr=args.lr, gamma=args.gamma, dropout=args.dropout,
        weight_decay=args.weight_decay,
        max_nodes=args.max_nodes, accumulation_steps=args.accumulation_steps,
        epochs=args.epochs,
        early_stop_patience=args.early_stop_patience,
        smoke_test=args.smoke_test,
    )
    run_cell(run_cfg)


if __name__ == "__main__":
    main()
