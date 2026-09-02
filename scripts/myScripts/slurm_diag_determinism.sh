#!/bin/bash
#SBATCH --job-name=diag_determinism
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=diag-det-%j.out
#SBATCH --error=diag-det-%j.out

set -e
REPO_ROOT="/home/yandex/MLWG2026/liorayacob/graphs-project"
ENV="/home/yandex/MLWG2026/liorayacob/anaconda3/envs/graphormer"
cd "$REPO_ROOT"
export LD_LIBRARY_PATH="${ENV}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_JIT=0

echo "Running on node: $(hostname)"

"${ENV}/bin/python" -u -c "
import sys, torch; sys.path.insert(0, 'src')
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception as exc:
    print('  [seed] deterministic algorithms unavailable:', exc)

from config import RunConfig
from backends.graphormer_backend import graphormer_train

run_cfg = RunConfig(backbone='graphormer', pe='rwse', dataset='peptides-func', seed=0)
result = graphormer_train(run_cfg)
print('SUCCESS:', result['metric_name'], '=', result['metric_value'])
"
echo "Done."
