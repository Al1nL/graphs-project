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
#   scripts/slurm/submit_grid.sh gps 32 --pe none,lappe        # only these PEs (fewer cells)
#   scripts/slurm/submit_grid.sh gps 32 --dry-run              # print the sbatch command, submit nothing
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
DRY_RUN=0
ACCOUNT="gpu-students"

while [ $# -gt 0 ]; do
  case "$1" in
    --partition=*) PARTITION="${1#*=}" ;;
    --account=*) ACCOUNT="${1#*=}" ;;
    --throttle=*) THROTTLE="${1#*=}" ;;
    --array-range=*) ARRAY_RANGE="${1#*=}" ;;
    --constraint=*) CONSTRAINT="${1#*=}" ;;
    --results-dir=*) RESULTS_DIR="${1#*=}" ;;
    --num-probe-graphs=*) NUM_PROBE_GRAPHS="${1#*=}" ;;
    --pe=*) PE_LIST="${1#*=}" ;;
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

if [ -n "$PE_LIST" ]; then
  # Recompute the same index mapping run_grid.slurm uses (PES x SEEDS x DATASETS, dataset
  # outermost) to turn a PE name list into the matching comma-separated Slurm array
  # indices, e.g. --pe=none,lappe -> "0,1,5,6,10,11,...". Kept as a literal re-derivation
  # (not a shared script) because it is 6 lines of arithmetic and importing bash logic
  # from run_grid.slurm here would be more fragile than just keeping the two in sync by
  # eye -- if you change PES there, change ALL_PES here too.
  ALL_PES=(none lappe rwse signnet grpe)
  N_PE=${#ALL_PES[@]}; N_SEED=3; N_DS=3
  IFS=',' read -ra WANTED <<< "$PE_LIST"
  indices=()
  for ds_i in $(seq 0 $((N_DS - 1))); do
    for seed_i in $(seq 0 $((N_SEED - 1))); do
      for pe_i in "${!ALL_PES[@]}"; do
        for w in "${WANTED[@]}"; do
          if [ "${ALL_PES[$pe_i]}" = "$w" ]; then
            indices+=($(( ds_i * N_PE * N_SEED + seed_i * N_PE + pe_i )))
          fi
        done
      done
    done
  done
  if [ "${#indices[@]}" -eq 0 ]; then
    echo "--pe=$PE_LIST matched no known PE name (known: ${ALL_PES[*]})" >&2
    exit 1
  fi
  ARRAY_RANGE="$(IFS=,; echo "${indices[*]}")"
  echo "--pe=$PE_LIST -> ${#indices[@]} cells -> --array=$ARRAY_RANGE"
fi

EXPORT_VARS="BACKBONE=$BACKBONE,NUM_TARGET_NODES=$NUM_TARGET_NODES,RESULTS_DIR=$RESULTS_DIR"
[ -n "$NUM_PROBE_GRAPHS" ] && EXPORT_VARS="$EXPORT_VARS,NUM_PROBE_GRAPHS=$NUM_PROBE_GRAPHS"

SBATCH_ARGS=(
  --partition="$PARTITION"
  --account="$ACCOUNT"
  --export="ALL,$EXPORT_VARS"
  --array="${ARRAY_RANGE}%${THROTTLE}"
)
[ -n "$CONSTRAINT" ] && SBATCH_ARGS+=(--constraint="$CONSTRAINT")

CMD=(sbatch "${SBATCH_ARGS[@]}" scripts/slurm/run_grid.slurm)

echo "Grid: backbone=$BACKBONE  T=$NUM_TARGET_NODES  partition=$PARTITION  "\
"account=$ACCOUNT  array=${ARRAY_RANGE}%${THROTTLE}  constraint=${CONSTRAINT:-<any>}"
echo "Command: ${CMD[*]}"

if [ "$DRY_RUN" = "1" ]; then
  echo "(--dry-run: not submitting)"
  exit 0
fi

"${CMD[@]}"
