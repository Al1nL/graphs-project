#!/bin/bash
# slurm_graphormer_base_check.sh
# ================================
# Runs scripts/myScripts/graphormer_base_check.py -- the "is the BASE actually good" check a
# teammate on SAN recommended doing BEFORE the full grid (see that script's docstring for
# the full reasoning). Trains with the reference config's OWN max_epoch (200), not the
# --epochs 20 shortcut the calibration run used, so the resulting AP is trustworthy enough
# to compare against published Peptides-func baselines.
#
# Submit with:
#   sbatch scripts/myScripts/slurm_graphormer_base_check.sh
# Check on it with:
#   squeue -u $USER
#   tail -f slurm-<jobid>.out
# Cancel with:
#   scancel <jobid>

#SBATCH --job-name=graphormer_base_check
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# 12h: job 785429 measured 20 epochs at ~57min of pure training time (~2.85min/epoch) on
# the full Peptides-func training split, plus ~41min total preprocessing (train+val+test,
# cached once via _CachedPEGraphormerDataset -- see graphormer_backend.py). 200 epochs at
# that rate is ~9.5h; 12h leaves real margin without needing to guess at a tighter number.
#SBATCH --time=12:00:00
#SBATCH --output=basecheck-%j.out
#SBATCH --error=basecheck-%j.out

set -e

REPO_ROOT="/home/yandex/MLWG2026/liorayacob/graphs-project"
ENV="/home/yandex/MLWG2026/liorayacob/anaconda3/envs/graphormer"

cd "$REPO_ROOT"

export LD_LIBRARY_PATH="${ENV}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_JIT=0  # see slurm_graphormer_calibrate.sh's comment -- needed on RTX 2080 Ti nodes

echo "Running on node: $(hostname), GPU(s): $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

"${ENV}/bin/python" -u scripts/myScripts/graphormer_base_check.py --pe none --dataset peptides-func

echo "Done."
