#!/bin/bash
# slurm_graphormer_grid.sh
# =========================
# Runs the Graphormer slice of the --preset reduced grid (5 PE x 3 datasets, seed 0 -- 15
# cells, see scripts/myScripts/gen_grid_cells.py) as a SLURM JOB ARRAY: one array task per cell,
# each its own independent job, instead of one process looping over all 15 sequentially.
#
# Why an array and not one `launch.py --backbone graphormer --preset reduced` job: cells
# run sequentially inside launch.py's own process, and real per-cell cost measured on this
# cluster is up to ~8-10h (a cold 200-epoch training run) -- 15 of those in one process
# could exceed a week, while this partition caps any SINGLE job at 24h
# (studentkillable: MaxTime=1-00:00:00). An array lets SLURM run multiple cells IN
# PARALLEL across the partition's nodes (subject to what's free), and each task stays
# safely inside the 24h cap on its own.
#
# Each task writes its OWN csv (results/csvs/graphormer_run_<task_id>.csv) rather than
# all appending to one shared file -- avoids a real race condition (concurrent
# open(...,"a") + header-write from multiple tasks finishing near-simultaneously on a
# networked filesystem). Merge them after the array finishes:
#   python -c "
#     import pandas as pd, glob
#     pd.concat([pd.read_csv(f) for f in glob.glob('results/csvs/graphormer_run_*.csv')]) \
#       .to_csv('results/csvs/graphormer_merged.csv', index=False)"
#
# The real scientific output -- one JSON per cell with the full sensitivity curves -- is
# NOT this CSV; run_experiment.run_cell() already writes those separately to
# results/<run_id>.json regardless (see run_experiment.py), one file per cell by design,
# meant to be combined later by scripts/aggregate_results.py.
#
# Submit with:
#   sbatch scripts/myScripts/slurm_graphormer_grid.sh
# Check status:
#   squeue -u $USER                          # one row per array task (RUNNING/PENDING)
#   sacct -j <jobid> --format=JobID,State,Elapsed
#   tail -f slurm_logs/graphormer_grid-<jobid>_<taskid>.out
# Cancel the WHOLE array:
#   scancel <jobid>
# Cancel just one task:
#   scancel <jobid>_<taskid>

#SBATCH --job-name=graphormer_grid
#SBATCH --partition=studentkillable
#SBATCH --array=1-15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/graphormer_grid-%A_%a.out
#SBATCH --error=slurm_logs/graphormer_grid-%A_%a.out

set -e

REPO_ROOT="/home/yandex/MLWG2026/liorayacob/graphs-project"
ENV="/home/yandex/MLWG2026/liorayacob/anaconda3/envs/graphormer"
CELLS_FILE="$REPO_ROOT/results/grid_cells_graphormer.txt"
NUM_TARGET_NODES=16   # calibrated for graphormer on peptides-func AND peptides-struct

cd "$REPO_ROOT"

export LD_LIBRARY_PATH="${ENV}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_JIT=0  # see slurm_graphormer_calibrate.sh's comment -- needed on RTX 2080 Ti nodes

# SLURM_ARRAY_TASK_ID is 1-indexed here (--array=1-15); the cells file is 0-indexed lines.
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CELLS_FILE")
read -r BACKBONE PE DATASET SEED <<< "$LINE"

if [ -z "$BACKBONE" ]; then
    echo "No cell found for array task $SLURM_ARRAY_TASK_ID in $CELLS_FILE -- aborting."
    exit 1
fi

echo "Task $SLURM_ARRAY_TASK_ID: backbone=$BACKBONE pe=$PE dataset=$DATASET seed=$SEED"
echo "Running on node: $(hostname), GPU(s): $CUDA_VISIBLE_DEVICES"

"${ENV}/bin/python" -u scripts/launch.py \
    --backbone "$BACKBONE" --pe "$PE" --dataset "$DATASET" --seed "$SEED" \
    --num-target-nodes "$NUM_TARGET_NODES" \
    --csv "results/csvs/graphormer_run_${SLURM_ARRAY_TASK_ID}.csv" \
    --no-strict-pins

echo "Done."
