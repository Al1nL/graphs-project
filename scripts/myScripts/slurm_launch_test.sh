#!/bin/bash
# slurm_launch_test.sh
# ======================
# One-cell validation of the launch.py run_cell()/run_probe() fix, as a real SLURM job --
# not a plain background shell process. Two separate background-process attempts at this
# same check died silently (no error, no traceback, just gone) when the launching session
# ended, exactly like the peptides-struct smoke test earlier; SLURM jobs are immune to
# that, which is the whole reason this project moved everything else to SLURM already.
#
# Reuses the existing graphormer_rwse_peptides-func_seed0 checkpoint (already trained to
# 20/20 epochs) -- fairseq will skip straight to the probe.
#
# Submit with:
#   sbatch scripts/myScripts/slurm_launch_test.sh
# Check on it with:
#   squeue -u $USER
#   tail -f launch-test-<jobid>.out

#SBATCH --job-name=launch_test
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --exclude=s-002
#SBATCH --output=launch-test-%j.out
#SBATCH --error=launch-test-%j.out

set -e

REPO_ROOT="/home/yandex/MLWG2026/liorayacob/graphs-project"
ENV="/home/yandex/MLWG2026/liorayacob/anaconda3/envs/graphormer"

cd "$REPO_ROOT"

export LD_LIBRARY_PATH="${ENV}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_JIT=0

echo "Running on node: $(hostname)"

"${ENV}/bin/python" -u scripts/launch.py \
    --backbone graphormer --pe rwse --dataset peptides-func --seed 0 \
    --num-target-nodes 16 --csv results/launch_test.csv \
    --no-require-cache --no-strict-pins

echo "--- CSV row ---"
cat results/launch_test.csv
echo "--- JSON result ---"
cat results/graphormer_rwse_peptides-func_seed0.json

echo "Done."
