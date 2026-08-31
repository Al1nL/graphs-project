#!/bin/bash
# run_until_done.sh
# ==================
# Resubmits `sbatch run_job.sh PE DATASET [extra args...]` until the corresponding
# result JSON exists (or MAX_RESUBMITS is hit), same as before -- but now takes an
# explicit SEED as its 3rd positional argument, since the original version
# hardcoded RESULT_PATH to seed0 and had no way to watch for any other seed's
# result file. Needed once seeds 1/2 are added on top of the existing seed-0 grid.
#
# Usage:
#   bash scripts/run_until_done.sh <pe> <dataset> <seed> [extra --flags for run_job.sh]
#
# Examples:
#   bash scripts/run_until_done.sh none peptides-func 0
#   bash scripts/run_until_done.sh lappe peptides-struct 1 --accumulation-steps 8 --epochs 150
#
# NOTE: run_job.sh's own python call always passes --seed 0 hardcoded today
# (`--seed 0` is baked into its argument list). This script's SEED argument is
# used only to compute the correct RESULT_PATH to watch for -- it does NOT by
# itself make run_job.sh train with that seed. Pass `--seed N` as one of the
# extra args too, or edit run_job.sh to forward $SEED, or this will watch for
# seed1's result file while actually training seed 0 forever. See the paired
# note in the deployment instructions.
set -uo pipefail

PE="${1:-rwse}"
DATASET="${2:-peptides-func}"
SEED="${3:-0}"
shift 3 2>/dev/null || true

RESULT_PATH="results/san_${PE}_${DATASET}_seed${SEED}.json"
MAX_RESUBMITS=50

echo "[run_until_done] PE=$PE DATASET=$DATASET SEED=$SEED | watching for $RESULT_PATH"

if [ -f "$RESULT_PATH" ]; then
    echo "[run_until_done] already done."
    exit 0
fi

for i in $(seq 1 "$MAX_RESUBMITS"); do
    echo "[run_until_done] attempt $i/$MAX_RESUBMITS..."
    sbatch --wait run_job.sh "$PE" "$DATASET" --seed "$SEED" "$@"
    sleep 5
    if [ -f "$RESULT_PATH" ]; then
        echo "[run_until_done] done after $i attempt(s)."
        exit 0
    fi
    echo "[run_until_done] no result yet, resubmitting..."
done

echo "[run_until_done] gave up after $MAX_RESUBMITS attempts."
exit 1
