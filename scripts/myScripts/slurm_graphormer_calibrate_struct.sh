#!/bin/bash
# slurm_graphormer_calibrate_struct.sh
# ======================================
# Same calibration run as slurm_graphormer_calibrate.sh, for peptides-struct instead of
# peptides-func. See that script for the full reasoning (time budget, ladder/n-graphs
# trim, PYTORCH_JIT fix).
#
# peptides-struct shares Peptides-func's graphs (same nodes/edges/diameters), just a
# different task head -- max-dist/d-min/d-max are the SAME values from src/dataset_meta.py
# (both peptides-func and peptides-struct use abs_rho_window=(26, 80), max_dist=159).
#
# Submit with:
#   sbatch scripts/myScripts/slurm_graphormer_calibrate_struct.sh

#SBATCH --job-name=graphormer_rwse_struct
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=calibrate_struct-%j.out
#SBATCH --error=calibrate_struct-%j.out

set -e

REPO_ROOT="/home/yandex/MLWG2026/liorayacob/graphs-project"
ENV="/home/yandex/MLWG2026/liorayacob/anaconda3/envs/graphormer"

cd "$REPO_ROOT"

export LD_LIBRARY_PATH="${ENV}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_JIT=0

echo "Running on node: $(hostname), GPU(s): $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

"${ENV}/bin/python" -u scripts/calibrate_target_nodes.py \
    --backbone graphormer \
    --pe rwse \
    --dataset peptides-struct \
    --epochs 20 \
    --max-dist 159 --d-min 26 --d-max 80 \
    --n-graphs 5 --ladder 4 8 16 32

echo "Done."
