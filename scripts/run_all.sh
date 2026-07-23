#!/bin/bash
# Runs the full (backbone x PE x dataset x seed) grid.
# Full grid: 3 backbones x 5 PEs x 3 datasets x 3 seeds = 135 runs.
#
# Usage:
#   bash scripts/run_all.sh full        # everything, 3 seeds
#   bash scripts/run_all.sh reduced      # fallback grid if compute is tight (see below)
#
# The "reduced" mode implements the fallback described in README.md / docs/rationale.docx:
#   - GraphGPS (primary backbone): full 5 PEs x 3 datasets x 3 seeds
#   - SAN, Graphormer (secondary backbones): 5 PEs x 3 datasets x 1 seed
# This keeps the primary backbone's numbers publication-quality (seed variance reported)
# while still getting every backbone x PE x dataset cell filled in at least once, which is
# the minimum needed to answer "is this PE effect backbone-specific?".

set -e
MODE=${1:-full}
SEEDS_PRIMARY=(0 1 2)
SEEDS_SECONDARY=(0 1 2)
if [ "$MODE" = "reduced" ]; then
  SEEDS_SECONDARY=(0)
fi

DATASETS=("peptides-func" "peptides-struct" "pascalvoc-sp")
PES=("none" "lappe" "rwse" "signnet" "grpe")

for ds in "${DATASETS[@]}"; do
  for pe in "${PES[@]}"; do
    for seed in "${SEEDS_PRIMARY[@]}"; do
      python src/run_experiment.py --backbone gps --pe "$pe" --dataset "$ds" --seed "$seed"
    done
    for seed in "${SEEDS_SECONDARY[@]}"; do
      python src/run_experiment.py --backbone san --pe "$pe" --dataset "$ds" --seed "$seed"
      python src/run_experiment.py --backbone graphormer --pe "$pe" --dataset "$ds" --seed "$seed"
    done
  done
done

echo "Done. Aggregate results/*.json once training is wired in (see run_experiment.py stubs)."
