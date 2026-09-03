#!/bin/bash
#SBATCH --job-name=san_test
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --output=/home/yandex/MLWG2026/liorpernik/graphs-project/logs/job_%j.out
#SBATCH --error=/home/yandex/MLWG2026/liorpernik/graphs-project/logs/job_%j.err

PE="${1:-rwse}"
DATASET="${2:-peptides-func}"
shift 2 2>/dev/null || true

mkdir -p /home/yandex/MLWG2026/liorpernik/graphs-project/logs

export DGLBACKEND=pytorch
export DGL_HOME=/home/yandex/MLWG2026/liorpernik/tmp/.dgl
mkdir -p /home/yandex/MLWG2026/liorpernik/tmp/.dgl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONPATH=/home/yandex/MLWG2026/alinl/SAN:/home/yandex/MLWG2026/liorpernik/graphs-project:$PYTHONPATH

cd /home/yandex/MLWG2026/liorpernik/graphs-project

ENV_DIR=/home/yandex/MLWG2026/alinl/anaconda3/envs/san_env
PYTHON=$ENV_DIR/bin/python

NVIDIA_LIBS=$(find $ENV_DIR/lib/python3.8/site-packages/nvidia -name lib -type d 2>/dev/null | tr '\n' ':')
PYLIBS_DIR=/home/yandex/MLWG2026/liorpernik/pylibs
PYLIBS_NVIDIA=$(find $PYLIBS_DIR/nvidia -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH=$NVIDIA_LIBS$PYLIBS_NVIDIA$ENV_DIR/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "[run_job] PE=$PE DATASET=$DATASET"
$PYTHON src/run_experiment.py --backbone san --pe $PE --dataset $DATASET --seed 0 --num-target-nodes 300 --accumulation-steps 8 "$@"
