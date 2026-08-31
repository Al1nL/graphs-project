"""
launch.py
=========
Single entry point for the (backbone x pe x dataset x seed) grid.

    python scripts/launch.py --dry-run                       # show the grid, run nothing
    python scripts/launch.py --dataset peptides-func --backbone gps
    python scripts/launch.py --preset reduced --wandb
    python scripts/launch.py --resume                        # skip cells already done

Replaces scripts/run_all.sh, which hardcoded the axes in nested bash loops with no logging,
no resume, no seeding and no provenance. Everything it decides comes from src/config.py, so
the grid is defined in exactly one place.

What this owns
--------------
  * enumerating the grid from explicit axes, with filters
  * deterministic seeding, applied before anything touches an RNG
  * version-lock enforcement (upstream commits, PE cache version) BEFORE burning GPU hours
  * one CSV row per run, appended as it completes, plus optional Weights & Biases
  * resume: a completed cell is skipped rather than silently recomputed

What it does not own: training. That is each backbone's own code, reached through
run_experiment.make_model_fn / the *_train stubs. This script will refuse to pretend a run
happened when those are still stubs.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from config import (  # noqa: E402
    BACKBONES,
    DATASETS,
    DEFAULT_SEEDS,
    PES,
    PE_CACHE_VERSION,
    RunConfig,
    check_pinned,
    grid,
    repo_sha,
)

CSV_FIELDS = [
    "run_id", "backbone", "pe", "dataset", "seed", "status",
    "metric_name", "metric_value", "num_params", "train_time_seconds",
    "peak_gpu_mem_mb", "rho", "rho_rel", "n_shared_feats", "num_target_nodes",
    "max_dist", "config_hash", "code_sha", "upstream_sha", "pe_cache_version",
    "started_at", "finished_at", "error",
]


def seed_everything(seed: int, deterministic: bool = True):
    """Seed every RNG this project can reach, and optionally force deterministic kernels.

    Order matters: PYTHONHASHSEED only takes effect for a fresh interpreter, so it is set
    for CHILD processes rather than pretended to apply here. cuDNN autotuning is disabled
    because benchmark mode picks algorithms by timing, which varies run to run.

    torch.use_deterministic_algorithms raises on ops with no deterministic implementation
    -- scatter-add on CUDA, used by most message-passing layers, is the usual offender. We
    surface that as a warning rather than a crash, because a non-deterministic scatter is a
    real but bounded reproducibility cost, whereas refusing to run at all is not a
    trade-off anyone would choose mid-grid.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed by cuBLAS
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:  # noqa: BLE001 - torch raises several types here
            print(f"  [seed] deterministic algorithms unavailable: {exc}")


def preflight(configs, strict_pins=True, require_cache=True):
    """Check everything that would invalidate a whole grid, BEFORE running any of it."""
    problems = []
    for backbone in sorted({c.backbone for c in configs}):
        rep = check_pinned(backbone, strict=False)
        if rep["status"] != "ok":
            line = f"  upstream {backbone}: {rep['status']} -- {rep.get('warning', '')}"
            (problems if strict_pins else []).append(line)
            print(line)
    if require_cache:
        from pe.cache import PECache
        for dataset in sorted({c.dataset for c in configs}):
            root = os.path.join("cache", dataset)
            try:
                PECache(root, "test")
            except (FileNotFoundError, RuntimeError) as exc:
                problems.append(f"  PE cache {root}: {exc}")
    if problems:
        raise SystemExit(
            "Pre-flight failed:\n" + "\n".join(problems) +
            "\n\nFix these before running -- each one silently invalidates the whole grid "
            "rather than one cell. Use --no-strict-pins / --no-require-cache to override "
            "for exploratory runs whose numbers will not be reported."
        )
    print("Pre-flight OK")


def already_done(cfg) -> bool:
    if not os.path.exists(cfg.result_path):
        return False
    try:
        with open(cfg.result_path) as f:
            return json.load(f).get("metric_value") is not None
    except (OSError, json.JSONDecodeError):
        return False


def append_csv(path, row):
    exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


def run_one(cfg, wandb_run=None, strict_pins=True):
    """Run one grid cell end to end and append a CSV row.

    FIX: previously called `TRAIN_FN[cfg.backbone](cfg, ...)` directly and discarded the
    returned dict entirely -- metric_value, num_params and the sensitivity curve never
    reached the CSV, and no JSON was written to `cfg.result_path` (see
    run_experiment.py's module docstring for the full account). This now calls
    `run_experiment.run_cell(cfg)`, the single function that trains, probes, and writes
    that JSON, and reads the real numbers back out of what it returns.
    """
    from dataset_meta import REL_BINS, REL_RHO_WINDOW, abs_rho_window
    from run_experiment import run_cell
    from sensitivity import long_range_fraction, to_relative_curve

    seed_everything(cfg.seed, cfg.deterministic)
    prov = cfg.provenance(strict_pins=strict_pins)
    row = {
        **{k: v for k, v in cfg.to_dict().items() if k in CSV_FIELDS},
        "config_hash": prov["config_hash"],
        "code_sha": prov["code_sha"],
        "upstream_sha": (prov["upstream"] or {}).get("actual"),
        "pe_cache_version": PE_CACHE_VERSION,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    t0 = time.time()
    try:
        result = run_cell(cfg)
        row["status"] = result["status"]
        row["metric_value"] = result.get("metric_value")
        row["num_params"] = result.get("num_params")
        row["n_shared_feats"] = result.get("n_shared_feats")
        curve = result.get("sensitivity_curve")
        if curve:
            d_min, d_max = abs_rho_window(cfg.dataset)
            row["rho"] = long_range_fraction(curve, d_min, d_max)
            # Relative rho needs per-graph diameters to rebin each curve onto d/diam(G)
            # BEFORE pooling -- pooling first and rebinning the pooled curve would not be
            # the same computation, since diameter varies per graph.
            per_graph = result.get("sensitivity_curves_per_graph") or []
            rel_curves = [to_relative_curve(g["curve"], g["diameter"], n_bins=REL_BINS)
                          for g in per_graph if g.get("diameter", 0) > 0]
            row["rho_rel"] = (
                long_range_fraction(_pool_rel(rel_curves), *REL_RHO_WINDOW)
                if rel_curves else None
            )
    except NotImplementedError as exc:
        # the training entry points are stubs until the backbone repos are cloned; say so
        # rather than writing a placeholder that looks like a result
        row["status"] = "not_implemented"
        row["error"] = str(exc).split("\n")[0][:200]
    except Exception:  # noqa: BLE001 - one failing cell must not kill the grid
        row["status"] = "failed"
        row["error"] = traceback.format_exc(limit=3).replace("\n", " | ")[:500]
    row["train_time_seconds"] = round(time.time() - t0, 2)
    row["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if wandb_run is not None:
        wandb_run.log({k: v for k, v in row.items() if isinstance(v, (int, float))})
    return row


def _pool_rel(rel_curves):
    """Pool a list of per-graph relative-distance curves the same way
    sensitivity.average_curves pools absolute ones (it is generic over integer-keyed
    curves, so this is exactly that call, factored out for readability at the call site)."""
    from sensitivity import average_curves as _avg
    return _avg(rel_curves)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", nargs="+", choices=BACKBONES, default=list(BACKBONES))
    ap.add_argument("--pe", nargs="+", choices=PES, default=list(PES))
    ap.add_argument("--dataset", nargs="+", choices=DATASETS, default=list(DATASETS))
    ap.add_argument("--seed", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    ap.add_argument("--preset", choices=["full", "reduced"], default="full",
                    help="reduced: 3 seeds for gps (primary backbone), 1 for san/graphormer")
    ap.add_argument("--num-target-nodes", type=int, default=None,
                    help="from scripts/calibrate_target_nodes.py; required for real runs")
    ap.add_argument("--num-probe-graphs", type=int, default=None,
                    help="test graphs sampled per cell for the sensitivity probe; "
                         "default config.PROBE_N_GRAPHS")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--csv", default="results/runs.csv")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip cells already completed")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="graph-pe-sensitivity")
    ap.add_argument("--no-strict-pins", dest="strict_pins", action="store_false")
    ap.add_argument("--no-require-cache", dest="require_cache", action="store_false")
    args = ap.parse_args()

    kw = dict(results_dir=args.results_dir, num_target_nodes=args.num_target_nodes,
              num_probe_graphs=args.num_probe_graphs)
    configs = list(grid(args.backbone, args.pe, args.dataset, args.seed, **kw))
    if args.preset == "reduced":
        # README fallback: publication-quality seed variance on the primary backbone only
        configs = [c for c in configs if c.backbone == "gps" or c.seed == args.seed[0]]

    print(f"Grid: {len(configs)} cells  "
          f"({len(args.dataset)} datasets x {len(args.backbone)} backbones x "
          f"{len(args.pe)} PEs x {len(args.seed)} seeds, preset={args.preset})")
    print(f"code_sha={repo_sha()}  pe_cache_version={PE_CACHE_VERSION}")

    if args.resume:
        before = len(configs)
        configs = [c for c in configs if not already_done(c)]
        print(f"Resume: skipping {before - len(configs)} completed, {len(configs)} to run")

    if args.dry_run:
        for c in configs:
            print(f"  {c.run_id:48s} max_dist={c.resolved_max_dist():3d} "
                  f"hash={c.config_hash()}")
        return

    preflight(configs, strict_pins=args.strict_pins, require_cache=args.require_cache)
    if args.num_target_nodes is None:
        raise SystemExit(
            "--num-target-nodes is required: it has no default by design. Calibrate it "
            "with `python scripts/calibrate_target_nodes.py --backbone ... --dataset ...` "
            "and pass the value it reports (docs/analysis-plan.md, Amendment 4)."
        )

    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project,
                                   config={"grid_size": len(configs),
                                           "code_sha": repo_sha(),
                                           "pe_cache_version": PE_CACHE_VERSION})
        except ImportError:
            print("wandb not installed; CSV logging only (pip install wandb)")

    counts = {}
    for i, cfg in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {cfg.run_id}")
        row = run_one(cfg, wandb_run, strict_pins=args.strict_pins)
        append_csv(args.csv, row)
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        print(f"    -> {row['status']}  ({row['train_time_seconds']}s)")

    print(f"\nDone. {counts}")
    print(f"CSV: {args.csv}")
    if counts.get("not_implemented"):
        print(f"\n{counts['not_implemented']} cell(s) hit a training stub. Wire the "
              "backbone entry points in src/run_experiment.py (see make_model_fn) once "
              "the upstream repos are cloned -- nothing was trained.")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
