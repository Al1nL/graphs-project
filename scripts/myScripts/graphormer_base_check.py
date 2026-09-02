"""
graphormer_base_check.py
=========================
One-off sanity check for the SHARED hyperparameters underneath every PE arm (num_layers,
embed_dim, lr, warmup_updates, ... -- see configs/graphormer/*.json), run BEFORE the full
grid (5 PE x 3 datasets x 3 seeds), not instead of it.

Why this exists: the calibration run (scripts/calibrate_target_nodes.py) deliberately
trains for only a handful of epochs (--epochs 20 there) to find num_target_nodes quickly --
it says nothing about whether these hyperparameters, trained PROPERLY (the reference
config's own max_epoch, 200), produce a competitive result on Peptides-func at all. They
were transcribed from Graphormer's own reference recipe for a DIFFERENT dataset (ZINC,
examples/property_prediction/zinc.sh) and never re-tuned for LRGB -- see
src/backends/graphormer_backend.py's build_graphormer_args docstring. Running the full
45-cell Graphormer grid on an untuned base and only noticing afterward (from bad numbers
across every PE) is exactly what happened on the SAN side of this project; this script is
the check that avoids repeating it here.

    python scripts/myScripts/graphormer_base_check.py --pe none --dataset peptides-func

Runs `graphormer_train` with the reference config's own max_epoch (200) -- no --epochs
override -- and prints the final task metric (AP for peptides-func). Compare against
published Peptides-func baselines (LRGB paper, Dwivedi et al. 2022): most GNN/GT baselines
there land in roughly 0.55-0.65 AP. A number well below that suggests the base
hyperparameters (lr, warmup_updates, encoder-layers, encoder-embed-dim) need tuning before
trusting ANY cell of the grid that shares them -- which is every Graphormer cell.

--pe none (the default here) isolates the check from any PE-specific effect: it is the
plain architecture + hyperparameters with nothing added, so a bad number here cannot be
blamed on the PE.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from config import RunConfig  # noqa: E402
from backends.graphormer_backend import graphormer_train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pe", default="none", choices=["none", "lappe", "rwse", "signnet", "grpe"])
    ap.add_argument("--dataset", default="peptides-func",
                    choices=["peptides-func", "peptides-struct"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_cfg = RunConfig(backbone="graphormer", pe=args.pe, dataset=args.dataset,
                        seed=args.seed, epochs=None)  # None -> the reference config's own max_epoch
    print(f"[base_check] training graphormer/{args.pe}/{args.dataset} with the reference "
          f"config's OWN epoch count (not a calibration shortcut) -- this is the real check.")
    result = graphormer_train(run_cfg)
    # Published LRGB-paper (Dwivedi et al. 2022) ballpark per dataset -- AP higher-better,
    # MAE lower-better. Different metrics/datasets need different sanity ranges; printing
    # the peptides-func AP range unconditionally would be actively misleading for
    # peptides-struct's MAE.
    baseline_hint = {
        "peptides-func": "~0.55-0.65 AP (higher is better)",
        "peptides-struct": "~0.25-0.30 MAE (lower is better)",
    }[args.dataset]
    print(f"\n{'=' * 70}")
    print(f"[base_check] {result['num_params']:,} params")
    print(f"[base_check] {result['metric_name']} = {result['metric_value']}")
    print(f"[base_check] compare against published {args.dataset} baselines ({baseline_hint}, "
          "LRGB paper) before trusting any cell of the grid that shares these hyperparameters.")
    print("=" * 70)


if __name__ == "__main__":
    main()
