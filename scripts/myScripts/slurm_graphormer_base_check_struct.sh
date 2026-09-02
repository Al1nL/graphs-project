#!/bin/bash
# slurm_graphormer_base_check_struct.sh
# =======================================
# Same "is the BASE actually good" check as slurm_graphormer_base_check.sh, for
# peptides-struct instead of peptides-func. See that script and
# scripts/myScripts/graphormer_base_check.py for the full reasoning.
#
# peptides-struct is a graph-REGRESSION task (11 real-valued targets, MAE -- lower is
# better, unlike AP), scored via `l1_loss` (see graphormer_backend.TASK_CRITERION). It has
# never been run through this harness before this script -- unlike peptides-func, there is
# no prior successful run to point to, only the 2-epoch smoke test done right before
# submitting this for real.
#
# Submit with:
#   sbatch scripts/myScripts/slurm_graphormer_base_check_struct.sh
# Check on it with:
#   squeue -u $USER
#   tail -f basecheck-struct-<jobid>.out
# Cancel with:
#   scancel <jobid>

#SBATCH --job-name=graphormer_base_check_struct
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# 12h: same budget as slurm_graphormer_base_check.sh -- peptides-struct is the SAME graphs
# as peptides-func (same node/edge counts, same preprocessing cost), just a different task
# head, so there is no reason to expect per-epoch cost to differ meaningfully.
#SBATCH --time=12:00:00
#SBATCH --output=basecheck-struct-%j.out
#SBATCH --error=basecheck-struct-%j.out

set -e

REPO_ROOT="/home/yandex/MLWG2026/liorayacob/graphs-project"
ENV="/home/yandex/MLWG2026/liorayacob/anaconda3/envs/graphormer"

cd "$REPO_ROOT"

export LD_LIBRARY_PATH="${ENV}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_JIT=0  # see slurm_graphormer_calibrate.sh's comment -- needed on RTX 2080 Ti nodes

echo "Running on node: $(hostname), GPU(s): $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

"${ENV}/bin/python" -u scripts/myScripts/graphormer_base_check.py --pe none --dataset peptides-struct

echo "Done."
