#!/bin/bash
set -uo pipefail

PE="${1:-rwse}"
RESULT_PATH="results/san_${PE}_peptides-func_seed0.json"
MAX_RESUBMITS=50

echo "[run_until_done] PE=$PE | watching for $RESULT_PATH"

if [ -f "$RESULT_PATH" ]; then
    echo "[run_until_done] already done."
    exit 0
fi

for i in $(seq 1 "$MAX_RESUBMITS"); do
    echo "[run_until_done] attempt $i/$MAX_RESUBMITS..."
    sbatch --wait run_job.sh "$PE"
    sleep 5
    if [ -f "$RESULT_PATH" ]; then
        echo "[run_until_done] done after $i attempt(s)."
        exit 0
    fi
    echo "[run_until_done] no result yet, resubmitting..."
done

echo "[run_until_done] gave up after $MAX_RESUBMITS attempts."
exit 1
