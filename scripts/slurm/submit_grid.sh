#!/bin/bash
# submit_grid.sh
# ===============
# Submit the 45-cell (5 PE x 3 seeds x 3 datasets) grid for one backbone as a single Slurm
# array job on the TAU CS cluster. Run this FROM slurm-client.cs.tau.ac.il (or a node you
# ssh'd into from there), from the repo root.
#
# Usage:
#   scripts/slurm/submit_grid.sh gps 32                       # backbone=gps, T=32
#   scripts/slurm/submit_grid.sh san 32 --partition=studentkillable --throttle=2
#   scripts/slurm/submit_grid.sh gps 32 --pe=none,lappe        # only these PEs (fewer cells)
#   scripts/slurm/submit_grid.sh gps 32 --dataset=peptides-func  # only this dataset
#   scripts/slurm/submit_grid.sh gps 32 --dry-run              # print the sbatch command, submit nothing
#
# --pe and --dataset INTERSECT, so `--dataset=peptides-func --pe=none,lappe,rwse,signnet`
# is the 12 cells of that dataset runnable today (grpe is not a drop-in on either backbone).
#
# Why --dataset exists, beyond convenience: one array submission carries ONE
# NUM_TARGET_NODES, and T is calibrated per (backbone, dataset) -- whether a given T
# populates the rho window depends on graph topology, which differs sharply between
# 150-node peptide chains and 479-node superpixel graphs. Honouring that means one
# submission per dataset. The index blocks happen to be contiguous (dataset is the
# outermost axis), so --array-range could do it by hand; the point of the flag is that a
# mistyped index is INVISIBLE -- the wrong cell trains perfectly well and records a T it
# was never calibrated for.
#
# Positional args: BACKBONE (gps|san), NUM_TARGET_NODES (from calibrate_target_nodes.py).
# Everything else is an optional flag with a TAU-appropriate default (see below).

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

BACKBONE="${1:?usage: submit_grid.sh <gps|san> <num_target_nodes> [flags]}"
NUM_TARGET_NODES="${2:?usage: submit_grid.sh <gps|san> <num_target_nodes> [flags]}"
shift 2

# ---------------------------------------------------------------------------
# Defaults. See scripts/slurm/README.md for what each partition actually means on this
# cluster and why studentbatch is the default here.
# ---------------------------------------------------------------------------
PARTITION="studentkillable"   # pre-emptible student partition. The comment here used to
                              # describe studentbatch (3-day, capped at 6 jobs) while the
                              # value said studentkillable -- two different partitions with
                              # different limits. studentkillable is what a gpu-students
                              # association actually grants; verify yours with
                              # `sacctmgr -P show assoc user=$USER format=account,partition,qos`
                              # and check its wall limit with `sinfo -p studentkillable -o "%P %l"`
                              # before trusting run_grid.slurm's --time.
THROTTLE=4                   # max concurrent array tasks (`--array=0-44%THROTTLE`)
ARRAY_RANGE="0-44"            # override with --array-range if running a partial grid
CONSTRAINT=""                 # e.g. "a5000|a6000|l40s" -- empty = any GPU node
RESULTS_DIR="results"
NUM_PROBE_GRAPHS=""
PE_LIST=""                    # non-empty -> filters run_grid.slurm's PES array client-side
DS_LIST=""                    # non-empty -> filters its DATASETS array the same way
DRY_RUN=0
ACCOUNT="gpu-students"
CONDA_BASE=""                 # conda installation root; empty -> job asks `conda info
                              # --base`, which only works if conda is on PATH there
CONDA_ENV=""                  # empty -> run_grid.slurm picks per backbone
                              # (graphgps_env / san_env). Accepts a NAME or a full
                              # PATH prefix; use the path when the env lives off
                              # home, which the cluster's quota warning forces.

while [ $# -gt 0 ]; do
  case "$1" in
    --partition=*) PARTITION="${1#*=}" ;;
    --account=*) ACCOUNT="${1#*=}" ;;
    --conda-env=*) CONDA_ENV="${1#*=}" ;;
    --conda-base=*) CONDA_BASE="${1#*=}" ;;
    --throttle=*) THROTTLE="${1#*=}" ;;
    --array-range=*) ARRAY_RANGE="${1#*=}" ;;
    --constraint=*) CONSTRAINT="${1#*=}" ;;
    --results-dir=*) RESULTS_DIR="${1#*=}" ;;
    --num-probe-graphs=*) NUM_PROBE_GRAPHS="${1#*=}" ;;
    --pe=*) PE_LIST="${1#*=}" ;;
    --dataset=*) DS_LIST="${1#*=}" ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
  shift
done

case "$BACKBONE" in
  gps|san) ;;
  *) echo "BACKBONE must be 'gps' or 'san', got '$BACKBONE'" >&2; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# Soft pre-flight: warn (don't block) if this account cannot see the requested partition.
# `sacctmgr` is TAU's own documented way to check this (see scripts/slurm/README.md).
# ---------------------------------------------------------------------------
if command -v sacctmgr >/dev/null 2>&1; then
  me="$(whoami)"
  if ! sacctmgr -P -i show user -s "$me" 2>/dev/null | grep -q "$PARTITION"; then
    echo "WARNING: 'sacctmgr -P -i show user -s $me' did not list partition '$PARTITION'." >&2
    echo "  You may need --account=<your-group> (see scripts/slurm/README.md), or this" >&2
    echo "  partition may not be enabled for your account yet. Continuing anyway." >&2
  fi
fi

if [ ! -d "cache/peptides-func" ] || [ ! -d "cache/peptides-struct" ] || [ ! -d "cache/pascalvoc-sp" ]; then
  echo "WARNING: cache/{peptides-func,peptides-struct,pascalvoc-sp} not all present." >&2
  echo "  Run src/pe/compute_pe.py for each dataset first (see README step 4) -- every" >&2
  echo "  array task will fail pre-flight otherwise." >&2
fi

_in_list() {   # _in_list <needle> <item>...
  local needle="$1"; shift
  local x
  for x in "$@"; do [ "$x" = "$needle" ] && return 0; done
  return 1
}

if [ -n "$PE_LIST" ] || [ -n "$DS_LIST" ]; then
  # Recompute the same index mapping run_grid.slurm uses (PES x SEEDS x DATASETS, dataset
  # outermost) to turn name lists into Slurm array indices, e.g.
  # --dataset=peptides-func --pe=none,lappe -> "0,1,5,6,10,11". Kept as a literal
  # re-derivation rather than shared with run_grid.slurm: it is a dozen lines of
  # arithmetic, and importing bash logic across the two would be more fragile than keeping
  # them in sync by eye. Change PES or DATASETS there and change ALL_PES / ALL_DS here.
  ALL_PES=(none lappe rwse signnet grpe)
  ALL_DS=(peptides-func peptides-struct pascalvoc-sp)
  N_PE=${#ALL_PES[@]}; N_SEED=3; N_DS=${#ALL_DS[@]}

  if [ "$ARRAY_RANGE" != "0-44" ]; then
    # Refuse rather than silently pick one. --pe used to overwrite --array-range without a
    # word, so a submission carrying both ran something the command line did not say.
    echo "--array-range cannot be combined with --pe/--dataset: they set the same thing." >&2
    echo "  Drop --array-range, or drop the filters and pass explicit indices." >&2
    exit 1
  fi

  # An unset filter means "all of that axis", which is what makes the two intersect.
  if [ -n "$PE_LIST" ]; then IFS=',' read -ra WANT_PE <<< "$PE_LIST"
  else WANT_PE=("${ALL_PES[@]}"); fi
  if [ -n "$DS_LIST" ]; then IFS=',' read -ra WANT_DS <<< "$DS_LIST"
  else WANT_DS=("${ALL_DS[@]}"); fi

  # Validate every name up front. The previous version matched names by scanning, so a
  # typo in a list simply contributed nothing and ran a SMALLER grid than asked for,
  # silently -- which on a 45-cell submission is not something you notice.
  for w in "${WANT_PE[@]}"; do
    _in_list "$w" "${ALL_PES[@]}" || {
      echo "--pe: unknown PE '$w' (known: ${ALL_PES[*]})" >&2; exit 1; }
  done
  for w in "${WANT_DS[@]}"; do
    _in_list "$w" "${ALL_DS[@]}" || {
      echo "--dataset: unknown dataset '$w' (known: ${ALL_DS[*]})" >&2; exit 1; }
  done

  indices=()
  for ds_i in $(seq 0 $((N_DS - 1))); do
    _in_list "${ALL_DS[$ds_i]}" "${WANT_DS[@]}" || continue
    for seed_i in $(seq 0 $((N_SEED - 1))); do
      for pe_i in "${!ALL_PES[@]}"; do
        _in_list "${ALL_PES[$pe_i]}" "${WANT_PE[@]}" || continue
        indices+=($(( ds_i * N_PE * N_SEED + seed_i * N_PE + pe_i )))
      done
    done
  done

  ARRAY_RANGE="$(IFS=,; echo "${indices[*]}")"
  echo "filter: --pe=${PE_LIST:-<all>}  --dataset=${DS_LIST:-<all>}"
  echo "        -> ${#indices[@]} cells -> --array=$ARRAY_RANGE"
  if [ "${#WANT_DS[@]}" -gt 1 ]; then
    echo "        NOTE: spans ${#WANT_DS[@]} datasets on ONE NUM_TARGET_NODES"          "($NUM_TARGET_NODES)." >&2
    echo "        T is calibrated per (backbone, dataset); submit one array per dataset" >&2
    echo "        to give each its own." >&2
  fi
fi

EXPORT_VARS="BACKBONE=$BACKBONE,NUM_TARGET_NODES=$NUM_TARGET_NODES,RESULTS_DIR=$RESULTS_DIR"
[ -n "$NUM_PROBE_GRAPHS" ] && EXPORT_VARS="$EXPORT_VARS,NUM_PROBE_GRAPHS=$NUM_PROBE_GRAPHS"
[ -n "$CONDA_ENV" ] && EXPORT_VARS="$EXPORT_VARS,CONDA_ENV=$CONDA_ENV"
[ -n "$CONDA_BASE" ] && EXPORT_VARS="$EXPORT_VARS,CONDA_BASE=$CONDA_BASE"

SBATCH_ARGS=(
  --partition="$PARTITION"
  --account="$ACCOUNT"
  --export="ALL,$EXPORT_VARS"
  --array="${ARRAY_RANGE}%${THROTTLE}"
)
[ -n "$CONSTRAINT" ] && SBATCH_ARGS+=(--constraint="$CONSTRAINT")

CMD=(sbatch "${SBATCH_ARGS[@]}" scripts/slurm/run_grid.slurm)

echo "Grid: backbone=$BACKBONE  T=$NUM_TARGET_NODES  partition=$PARTITION"
echo "      account=$ACCOUNT  array=${ARRAY_RANGE}%${THROTTLE}  constraint=${CONSTRAINT:-<any>}"
echo "      env=${CONDA_ENV:-<per-backbone default: graphgps_env/san_env>}"
echo "      conda_base=${CONDA_BASE:-<from PATH>}"
echo "Command: ${CMD[*]}"

if [ "$DRY_RUN" = "1" ]; then
  echo "(--dry-run: not submitting)"
  exit 0
fi

"${CMD[@]}"
