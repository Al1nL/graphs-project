#!/bin/bash
#SBATCH --job-name=san_test
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

PE="${1:-rwse}"
DATASET="${2:-peptides-func}"

# Derive project root from script location -- works regardless of which user runs it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

export DGLBACKEND=pytorch
export DGL_HOME="$SCRIPT_DIR/../tmp/.dgl"
mkdir -p "$SCRIPT_DIR/../tmp/.dgl"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# SAN clone and project root on PYTHONPATH
# SAN lives as a sibling of graphs-project (../SAN relative to project root)
export PYTHONPATH="/home/yandex/MLWG2026/alinl/SAN:$SCRIPT_DIR:$PYTHONPATH"

# Conda env -- shared, always lives under alinl
ENV_DIR=/home/yandex/MLWG2026/alinl/anaconda3/envs/san_env
PYTHON=$ENV_DIR/bin/python

# CUDA libs: san_env has cublas/cusparse; pylibs (sibling of project) has curand/cufft/cudnn
NVIDIA_LIBS=$(find $ENV_DIR/lib/python3.8/site-packages/nvidia -name lib -type d 2>/dev/null | tr '\n' ':')
PYLIBS_DIR="$SCRIPT_DIR/../pylibs"
PYLIBS_NVIDIA=$(find "$PYLIBS_DIR/nvidia" -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH=$NVIDIA_LIBS$PYLIBS_NVIDIA$ENV_DIR/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "[run_job] PE=$PE DATASET=$DATASET from $SCRIPT_DIR"
$PYTHON src/run_experiment.py --backbone san --pe $PE --dataset $DATASET --seed 0 --num-target-nodes 300
