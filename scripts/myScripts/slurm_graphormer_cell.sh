#!/bin/bash
# slurm_graphormer_cell.sh
# =========================
# One SLURM job = one grid cell (backbone=graphormer, one PE, one dataset, one seed).
# Used for the real 30-cell grid (5 PE x 2 datasets x 3 seeds -- pascalvoc-sp excluded
# for Graphormer by team decision), one job per cell, matching how SAN is being run
# (each seed as its own job).
#
# Reads the CODE, the conda env, the LRGB processed-dataset cache, and the PE cache from
# THIS repo (world-readable) -- but writes results (checkpoints, JSON, CSV) to
# $RESULTS_DIR, which defaults to this repo's own results/ and MUST be overridden to a
# directory you own if you are not liorayacob (this repo's files are read-only for
# everyone else: r-x, no w, for group and other).
#
# Submit one cell:
#   PE=rwse DATASET=peptides-func SEED=0 sbatch scripts/myScripts/slurm_graphormer_cell.sh
#
# Submit as someone else, writing to your own space:
#   PE=rwse DATASET=peptides-func SEED=1 \
#     RESULTS_DIR=/home/yandex/MLWG2026/<you>/graphormer_results \
#     sbatch scripts/myScripts/slurm_graphormer_cell.sh
#
# (loop over PE x DATASET for a fixed SEED to submit a whole seed's worth of cells --
# see scripts/myScripts/submit_graphormer_grid.sh, which also redirects --output/--error into a
# dedicated results_dir/slurm_logs/graphormer-seed<seed>/ folder per batch. The
# --output=graphormer-%x-%j.out below is only what you get submitting this file directly
# instead of through that wrapper -- it lands in whatever directory you ran sbatch from.)
#
# --exclude=s-006: /usr/bin/git is missing on that node entirely (confirmed directly,
# 2026-08-31 -- `git -C ../Graphormer rev-parse HEAD` fails with "command not found",
# no git anywhere in the conda env either as a fallback). check_pinned() in config.py
# treats a failed git call as "pin missing" and launch.py's preflight() then refuses to
# run at all (strict_pins=True by default here, deliberately -- this is the real grid,
# not an exploratory smoke test). Six of the first ten seed=0 cells landed on s-006 and
# died instantly with exactly this error before this fix.

#SBATCH --job-name=graphormer_cell
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=s-002,s-006
#SBATCH --output=graphormer-%x-%j.out
#SBATCH --error=graphormer-%x-%j.out

set -e

: "${PE:?Set PE (none|lappe|rwse|signnet|grpe)}"
: "${DATASET:?Set DATASET (peptides-func|peptides-struct)}"
: "${SEED:?Set SEED (0|1|2)}"

REPO_ROOT="/home/yandex/MLWG2026/liorayacob/graphs-project"
ENV="/home/yandex/MLWG2026/liorayacob/anaconda3/envs/graphormer"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results}"

mkdir -p "$RESULTS_DIR"
cd "$REPO_ROOT"

export LD_LIBRARY_PATH="${ENV}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_JIT=0

RUN_ID="graphormer_${PE}_${DATASET}_seed${SEED}"

# job_ids.csv maps SLURM job id -> which grid cell it is, so "which job was cell X"
# or "which cell was job Y" is a grep away instead of cross-referencing squeue history.
# Written under flock since up to 10 of these jobs can start around the same moment and
# all append to the SAME shared file -- flock serialises the read-modify-write so no two
# jobs interleave mid-line and corrupt it.
JOBID_FILE="$RESULTS_DIR/job_ids.csv"
(
    flock -x 200
    [ -s "$JOBID_FILE" ] || echo "job_id,run_id,backbone,pe,dataset,seed,node,started_at" > "$JOBID_FILE"
    echo "${SLURM_JOB_ID},${RUN_ID},graphormer,${PE},${DATASET},${SEED},$(hostname),$(date +%Y-%m-%dT%H:%M:%S)" >> "$JOBID_FILE"
) 200>"$JOBID_FILE.lock"

echo "Running on node: $(hostname)"
echo "job_id=$SLURM_JOB_ID  run_id=$RUN_ID  RESULTS_DIR=$RESULTS_DIR"

"${ENV}/bin/python" -u scripts/launch.py \
    --backbone graphormer --pe "$PE" --dataset "$DATASET" --seed "$SEED" \
    --num-target-nodes 16 \
    --results-dir "$RESULTS_DIR" \
    --csv "$RESULTS_DIR/runs_graphormer_${PE}_${DATASET}_seed${SEED}.csv"

echo "Done."
