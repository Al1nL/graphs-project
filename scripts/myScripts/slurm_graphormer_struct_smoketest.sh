#!/bin/bash
# slurm_graphormer_struct_smoketest.sh
# ======================================
# Quick (2-epoch) validation that peptides-struct works at all through this harness before
# committing to the full 12h base-check (slurm_graphormer_base_check_struct.sh). Run as a
# real SLURM job, not a disowned background shell process -- the previous attempt at this
# same check was a plain background process outside SLURM, and it silently died without
# completing (no error, just gone) when the session that launched it ended overnight.
# SLURM jobs are unaffected by that; this is the fix for HOW the check runs, not the check
# itself.
#
# Submit with:
#   sbatch scripts/myScripts/slurm_graphormer_struct_smoketest.sh
# Check on it with:
#   squeue -u $USER
#   tail -f struct-smoketest-<jobid>.out

#SBATCH --job-name=graphormer_struct_smoketest
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=struct-smoketest-%j.out
#SBATCH --error=struct-smoketest-%j.out

set -e

REPO_ROOT="/home/yandex/MLWG2026/liorayacob/graphs-project"
ENV="/home/yandex/MLWG2026/liorayacob/anaconda3/envs/graphormer"

cd "$REPO_ROOT"

export LD_LIBRARY_PATH="${ENV}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_JIT=0

echo "Running on node: $(hostname), GPU(s): $CUDA_VISIBLE_DEVICES"

"${ENV}/bin/python" -u -c "
import sys; sys.path.insert(0, 'src')
from config import RunConfig
from backends.graphormer_backend import graphormer_train, make_graphormer_model_fn

run_cfg = RunConfig(backbone='graphormer', pe='none', dataset='peptides-struct', seed=0, epochs=2)
result = graphormer_train(run_cfg)
print('TRAIN OK:', result['num_params'], 'params,', result['metric_name'], '=', result['metric_value'])
item = result['test_dataset'][0]
model_fn, probe_data, meta = make_graphormer_model_fn(result['model'], item)
print('PROBE WRAP OK:', meta)
"

echo "Done."
