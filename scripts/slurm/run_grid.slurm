#!/bin/bash
# run_grid.slurm
# ===============
# One SBATCH submission running all 45 cells (5 PE x 3 seeds x 3 datasets) for ONE backbone
# as a Slurm JOB ARRAY, on the TAU CS cluster (slurm-client.cs.tau.ac.il).
#
# Do not submit 45 separate `sbatch` calls instead of this. The `studentbatch` partition
# caps a user at 6 BATCH JOBS -- a job array is ONE submission (one `sbatch` call, one
# job ID family), so this fits that cap; 45 individual submissions would not. See
# scripts/slurm/README.md for the full explanation and for what to do if your account is
# on a different partition with different limits.
#
# Usage (see scripts/slurm/submit_grid.sh for the wrapper that fills these in for you):
#
#   sbatch --partition=studentbatch \
#          --export=BACKBONE=gps,NUM_TARGET_NODES=32,RESULTS_DIR=results \
#          --array=0-44%4 \
#          scripts/slurm/run_grid.slurm
#
# BACKBONE must be "gps" or "san" (Graphormer has no training integration yet -- see
# README "Implementation status" -- so there is nothing for this script to run for it).
# NUM_TARGET_NODES has no default anywhere in this project by design (see
# sensitivity.compute_sensitivity_curve's docstring) -- calibrate it FIRST with
# scripts/calibrate_target_nodes.py and pass the value it recommends. This script refuses
# to guess one for you.
#
# The `%4` in `--array=0-44%4` throttles to at most 4 tasks running concurrently -- a
# courtesy default on a shared cluster, not a hard requirement. Raise it if you have a
# priority `gpu-<research-group>` allocation; lower it (or drop to `%1`) if you are running
# under `studentkillable`/`killable`, where your job can be pre-empted by higher-priority
# work and a lower concurrency reduces how much gets pre-empted at once.

#SBATCH --job-name=pe-grid
# TAU requires an account as well as a partition; a submission without one is rejected
# outright. submit_grid.sh passes --account explicitly (default gpu-students) and a
# command-line flag overrides this directive, so this is only the fallback for someone
# running `sbatch run_grid.slurm` by hand. Change it if your group's account differs.
#SBATCH --account=gpu-students
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --gpus=1
# 1440 min = 24 h, the hard cap on studentkillable:
#     $ sinfo -p studentkillable -o "%P %l"
#     PARTITION       TIMELIMIT
#     studentkillable 1-00:00:00
# This used to request 4320 (3 days), which that partition rejects outright -- the value
# was copied from studentbatch's 3-day limit. Raise it only if you submit to a partition
# your account can actually use AND whose limit is higher; check both, they are separate
# questions (see submit_grid.sh's PARTITION comment).
#SBATCH --time=1440
#SBATCH --signal=USR1@120
# --requeue: cells on a killable partition can be pre-empted; this asks Slurm to requeue
# rather than mark the array task failed.
#
# A requeued GPS cell RESUMES; it does not restart. graphgps_backend sets
# cfg.train.auto_resume with enable_ckpt and ckpt_period 10, so a pre-empted cell reloads
# its last checkpoint and continues, losing at most 10 epochs. That is what makes a
# 200-epoch cell viable on a partition with a 24 h wall and no protection from
# pre-emption -- without it, a cell needing longer than the wall would never finish no
# matter how many times Slurm requeued it. Note the flip side: re-running a cell that
# already reached max_epoch will NOT retrain it (custom_train logs "Task already done"),
# so a deliberate retrain needs its results/raw/<cell>/<seed>/ directory cleared first.
#
# The SAN arm has no equivalent yet, so a pre-empted SAN cell does restart from scratch.
#SBATCH --requeue

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root, regardless of where sbatch was invoked from
mkdir -p slurm_logs

# ---------------------------------------------------------------------------
# Required env vars (passed via `sbatch --export=...`, see submit_grid.sh)
# ---------------------------------------------------------------------------
: "${BACKBONE:?set BACKBONE=gps or BACKBONE=san (--export=BACKBONE=...)}"
: "${NUM_TARGET_NODES:?set NUM_TARGET_NODES (calibrate first: scripts/calibrate_target_nodes.py)}"
RESULTS_DIR="${RESULTS_DIR:-results}"
NUM_PROBE_GRAPHS="${NUM_PROBE_GRAPHS:-}"   # empty -> run_experiment.py's own default

case "$BACKBONE" in
  gps|san) ;;
  *) echo "BACKBONE must be 'gps' or 'san', got '$BACKBONE'" >&2; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# Map SLURM_ARRAY_TASK_ID (0..44) -> (pe, seed, dataset). Order matches
# src/config.py:grid()'s own axis order (dataset outermost, then pe, then seed) so a
# resumed/partial array and a `launch.py --dry-run` listing describe the same cells in the
# same order -- useful when cross-checking which array index corresponds to which cell.
# ---------------------------------------------------------------------------
PES=(none lappe rwse signnet grpe)
SEEDS=(0 1 2)
DATASETS=(peptides-func peptides-struct pascalvoc-sp)

: "${SLURM_ARRAY_TASK_ID:?this script must be run as a Slurm array (--array=0-44), not directly}"
idx="$SLURM_ARRAY_TASK_ID"
n_pe=${#PES[@]}; n_seed=${#SEEDS[@]}
pe_i=$(( idx % n_pe ))
seed_i=$(( (idx / n_pe) % n_seed ))
ds_i=$(( idx / (n_pe * n_seed) ))
PE="${PES[$pe_i]}"
SEED="${SEEDS[$seed_i]}"
DATASET="${DATASETS[$ds_i]}"

if [ "$ds_i" -ge "${#DATASETS[@]}" ]; then
  echo "SLURM_ARRAY_TASK_ID=$idx out of range for a 45-cell grid (0-44)" >&2
  exit 1
fi

echo "=== cell: backbone=$BACKBONE pe=$PE dataset=$DATASET seed=$SEED (array idx $idx) ==="
echo "node=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-unset}  started=$(date -Is)"

# ---------------------------------------------------------------------------
# Environment: GraphGPS and SAN need DIFFERENT conda envs (see README "Environment
# setup" -- their PyG/DGL/CUDA pins conflict). One array runs ONE backbone, so one env.
#
# CONDA_ENV overrides the per-backbone default and accepts either a NAME or a full PATH
# prefix, because `conda activate` takes both. The path form is the one that matters here:
# the CS cluster warns that home quotas cannot hold an env ("do NOT install in your
# HOME-DIR as you don't have enough quota for it"), so a pytorch+CUDA env usually lives on
# netapp/scratch and was created with `conda env create -p /path/to/envs/graphgps_env`.
# A -p env is NOT findable by name unless its parent is registered in envs_dirs, so
# passing the prefix explicitly avoids depending on ~/.condarc state that the job record
# does not capture:
#
#   sbatch --export=ALL,BACKBONE=gps,CONDA_ENV=/path/to/envs/graphgps_env,... run_grid.slurm
#   scripts/slurm/submit_grid.sh gps 32 --conda-env=/path/to/envs/graphgps_env
# ---------------------------------------------------------------------------
# Locating conda: `conda` is NOT on PATH on this cluster by default (`which conda` ->
# "Command not found"), and a batch shell does not source your login rc, so
# `conda info --base` can fail here even when conda works in an interactive session.
# CONDA_BASE names the installation root explicitly; it falls back to asking conda only
# when conda happens to be on PATH.
CONDA_BASE="${CONDA_BASE:-}"
if [ -z "$CONDA_BASE" ] && command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
fi
if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  echo "ERROR: no conda installation found." >&2
  echo "  conda is not on PATH here and CONDA_BASE is unset or wrong." >&2
  echo "  Pass the installation root, e.g.:" >&2
  echo "    --export=ALL,CONDA_BASE=/home/yandex/<course>/<user>/miniforge3,..." >&2
  echo "  (CONDA_BASE=$CONDA_BASE)" >&2
  exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [ -z "${CONDA_ENV:-}" ]; then
  if [ "$BACKBONE" = "gps" ]; then CONDA_ENV="graphgps_env"; else CONDA_ENV="san_env"; fi
fi

# `set -e` does not catch this: conda activate is a shell function, and on a missing env it
# prints "EnvironmentNameNotFound" and returns non-zero WITHOUT terminating the script under
# every conda version. Unchecked, the cell would then run against whatever python is on
# PATH -- most likely the base env, which has no torch -- and fail later with a confusing
# ImportError instead of the real cause. Check explicitly.
conda activate "$CONDA_ENV" || {
  echo "ERROR: conda activate '$CONDA_ENV' failed." >&2
  echo "  If this is a name, it must exist in one of: $(conda config --show envs_dirs 2>/dev/null | tr '\n' ' ')" >&2
  echo "  If the env lives outside those (e.g. on netapp because of the home quota), pass" >&2
  echo "  the full prefix instead: --export=ALL,CONDA_ENV=/path/to/envs/<name>,..." >&2
  exit 1
}
if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "ERROR: conda activate '$CONDA_ENV' reported success but CONDA_PREFIX is unset." >&2
  exit 1
fi

# Logged so the job record shows which interpreter actually ran the cell. Provenance:
# "which env produced this number" is otherwise unanswerable from results/*.json alone.
echo "env: $CONDA_PREFIX"
echo "python: $(command -v python)"

# GRPE is not yet a drop-in on either backbone (see README "Implementation status") --
# skip it here rather than let it fail loudly 9 times across the array (3 datasets x 3
# seeds) for a cell we already know raises NotImplementedError. Remove this guard once
# the GRPE attention-bias hook lands for the backbone you are running.
if [ "$PE" = "grpe" ]; then
  echo "SKIP: PE=grpe is not yet a drop-in for backbone=$BACKBONE (see README); exiting 0" >&2
  exit 0
fi

EXTRA_ARGS=()
if [ -n "$NUM_PROBE_GRAPHS" ]; then
  EXTRA_ARGS+=(--num-probe-graphs "$NUM_PROBE_GRAPHS")
fi

python src/run_experiment.py \
  --backbone "$BACKBONE" \
  --pe "$PE" \
  --dataset "$DATASET" \
  --seed "$SEED" \
  --num-target-nodes "$NUM_TARGET_NODES" \
  --results-dir "$RESULTS_DIR" \
  "${EXTRA_ARGS[@]}"

echo "finished=$(date -Is)"
