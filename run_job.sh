#!/bin/bash
#SBATCH --job-name=san_test
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

# Redirect DGL config/cache away from quota-restricted home directory
export DGLBACKEND=pytorch
export DGL_HOME=/home/yandex/MLWG2026/alinl/tmp/.dgl
mkdir -p /home/yandex/MLWG2026/alinl/tmp/.dgl

# Add SAN repository and project root to Python import path
export PYTHONPATH=/home/yandex/MLWG2026/alinl/SAN:/home/yandex/MLWG2026/alinl/graphs-project:$PYTHONPATH

# Navigate to project directory
cd /home/yandex/MLWG2026/alinl/graphs-project
# Activate environment
source /home/yandex/MLWG2026/alinl/anaconda3/etc/profile.d/conda.sh
conda activate san_env

# Dynamically add all NVIDIA CUDA pip library subdirectories to LD_LIBRARY_PATH
ENV_DIR=/home/yandex/MLWG2026/alinl/anaconda3/envs/san_env
NVIDIA_LIBS=$(find $ENV_DIR/lib/python3.8/site-packages/nvidia -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH=$NVIDIA_LIBS$ENV_DIR/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Run smoke test
python src/run_experiment.py --backbone san --pe rwse --dataset peptides-func --seed 0 --num-target-nodes 300
