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
# --exclude=s-005,s-006: s-005 excluded 2026-09-0X for the reason below (our OWN cells
# piling up on it). s-006 added 2026-09-08 for the SAME underlying problem but a
# different cause: `sinfo -N -o "%N %C %O"` showed CPU_LOAD=64.18 on s-006 (vs 5-8 on
# s-003/s-004/s-005) from OTHER students' jobs sharing it -- studentkillable does not
# give exclusive CPU access, so a heavy neighbor slows every job on the node, not just
# ours. Confirmed directly: graphormer_grpe_peptides-func_seed1 and
# graphormer_none_peptides-struct_seed1 both sat frozen at the SAME checkpoint count
# (96/200, 53/200) across three separate restarts over three days on s-006/s-002 before
# being cancelled and resubmitted excluding both.
#
# This list is NOT a permanent fix -- contention rotates to whichever node other
# students' jobs land on. Before a large submission, check `sinfo -N -p studentkillable
# -o "%N %C %O"` and add whichever node has CPU_LOAD far above the others (single digits
# is normal; two digits or more means a heavy neighbor).
#
# (history: s-002/s-006 were ALSO excluded briefly for an unrelated reason -- a missing
# `git` binary -- fixed 2026-09-01 in config.py with a pure-Python .git/ fallback, see
# _dotgit_head_sha/_dotgit_origin_url.)
#
# The rest of this comment is the ORIGINAL s-005 finding, unchanged: s-005 turned out to
# have its own problem: it's where our OWN concurrently-
# submitted cells kept landing together (confirmed via sinfo: consistently the least
# idle CPU of the 5 nodes, e.g. 8/40 idle vs. 16-20/40 elsewhere, despite having the same
# --cpus-per-task=4 request as everywhere else), and each cell's CPU-bound preprocessing
# bursts far past its requested 4 CPUs (~10-13 cores measured directly) -- several such
# bursts on one node fight over the same physical cores. Measured directly on
# lappe/peptides-func seed0: ~55 min/epoch while sharing s-005 with 5 other cells of
# ours, dropping to ~2 min/epoch once resubmitted excluding s-005.

#SBATCH --job-name=graphormer_cell
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=s-005,s-006
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
