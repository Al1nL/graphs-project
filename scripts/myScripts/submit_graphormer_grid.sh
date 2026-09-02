#!/bin/bash
# submit_graphormer_grid.sh
# ==========================
# Submits one scripts/myScripts/slurm_graphormer_cell.sh job per (PE, dataset) pair for a single
# SEED -- i.e. one person's share of the 30-cell Graphormer grid (5 PE x 2 datasets;
# pascalvoc-sp excluded for Graphormer by team decision). Splitting by seed this way
# lets teammates run their own seed in parallel, each writing to their own RESULTS_DIR.
#
# SLURM .out/.err logs for this batch go under
#   <results_dir>/slurm_logs/graphormer-seed<seed>/<pe>_<dataset>-<jobid>.out
# -- a dedicated, seed-named folder per submission, kept apart from older ad-hoc
# debugging logs (launch-test-*.out, diag-stack-*.out, ...) sitting in the repo root, and
# apart from other people's/other seeds' logs since it lives under YOUR results_dir.
#
# Why SUBMIT_DELAY exists
# -----------------------
# The first real 10-cell run (seed=0, 2026-08-31) submitted all 10 `sbatch` calls back to
# back with no gap. SLURM happily packed several onto the SAME node at once (e.g. 3 of
# them landed on s-005 within the same minute) -- each only asks for 4 CPUs, well within
# the node's total, but each ALSO briefly bursts far past that (~10-13 cores measured
# directly) during its CPU-bound preprocessing (Floyd-Warshall etc. over the whole
# 10,873-graph training split). This cluster does not appear to hard-cap a job at its
# requested CPU count, so several such bursts landing on one node at once fight over the
# same physical cores: cells that normally preprocess in ~9-10 minutes took 90-104
# minutes instead, and one cell's probe stage (also CPU-heavy) took 5h42m and still
# hadn't finished when its 6h budget ran out -- confirmed NOT a code bug: the exact same
# probe, run in isolation with nothing else contending, finished 4 graphs in under 25
# seconds each. A fixed delay between submissions doesn't guarantee different nodes, but
# it keeps each new job's CPU-heavy preprocessing burst from starting while a just-
# submitted one is still mid-burst, which is what actually caused the pile-up.
#
# Usage:
#   ./scripts/myScripts/submit_graphormer_grid.sh <seed> [results_dir] [submit_delay_seconds]
#
# Examples:
#   ./scripts/myScripts/submit_graphormer_grid.sh 0
#   ./scripts/myScripts/submit_graphormer_grid.sh 1 /home/yandex/MLWG2026/<you>/graphormer_results
#   ./scripts/myScripts/submit_graphormer_grid.sh 0 "" 120   # shorter 2-minute gap instead of the default 3

set -e

SEED="${1:?Usage: submit_graphormer_grid.sh <seed> [results_dir] [submit_delay_seconds]}"
RESULTS_DIR="${2:-$(pwd)/results}"
SUBMIT_DELAY="${3:-180}"

PES=(none lappe rwse signnet grpe)
DATASETS=(peptides-func peptides-struct)

LOG_DIR="$RESULTS_DIR/slurm_logs/graphormer-seed${SEED}"
mkdir -p "$LOG_DIR"
echo "Logs for this batch: $LOG_DIR"
echo "Spacing submissions ${SUBMIT_DELAY}s apart to avoid piling onto the same node"

first=true
for PE in "${PES[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    if [ "$first" = true ]; then
      first=false
    else
      sleep "$SUBMIT_DELAY"
    fi
    PE=$PE DATASET=$DATASET SEED=$SEED RESULTS_DIR=$RESULTS_DIR \
      sbatch --job-name="g_${PE}_${DATASET}_s${SEED}" \
             --output="$LOG_DIR/${PE}_${DATASET}-%j.out" \
             --error="$LOG_DIR/${PE}_${DATASET}-%j.out" \
             scripts/myScripts/slurm_graphormer_cell.sh
  done
done
