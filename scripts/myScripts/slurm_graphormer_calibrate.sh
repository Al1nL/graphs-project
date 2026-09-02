#!/bin/bash
# slurm_graphormer_calibrate.sh
# ==============================
# Submit the Graphormer calibration/training run to a DEDICATED SLURM node instead of
# running it on the shared login node (c-003 etc.) -- see the lecturer's message: this
# cluster has real GPU nodes behind SLURM (studentkillable: s-002..s-006, up to 8 GPUs
# each) precisely so students stop competing for RAM/CPU/GPU on the login node the way we
# were doing until now, which is what made every run here take hours instead of minutes.
#
# Submit with:
#   sbatch scripts/myScripts/slurm_graphormer_calibrate.sh
#
# Check on it with:
#   squeue -u $USER          # PENDING (queued) or RUNNING, and which node
#   tail -f slurm-<jobid>.out   # live output, once RUNNING (filename SLURM prints after sbatch)
#
# Cancel with:
#   scancel <jobid>
#
# Adjust --time / --pe / --dataset / --epochs below as needed. This does NOT need tmux,
# ssh, or the terminal to stay open at all -- SLURM keeps the job running on its own node
# regardless of what happens to your login session.

#SBATCH --job-name=graphormer_rwse
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
# 32G: src/backends/graphormer_backend.py caches every preprocessed training graph for the
# life of the run (_CachedPEGraphormerDataset), and -- since the fix that made this run
# feasible at all -- no longer computes the (unused) full-diameter edge_input tensor that
# used to dominate both time AND memory per graph (see the --time comment below for the
# time side of that fix). Per-item cost is now small enough that the whole ~10.9k-graph
# training split's cache measured at well under 32G; kept as a real, generous margin, not
# tuned to the wire.
#SBATCH --mem=32G
# 4h, not 2h: job 785429 measured the pre-sweep cost directly -- train preprocessing
# ~25min, val ~3min, 20 training epochs ~57min, test preprocessing ~16min, ~101min total --
# and then got CANCELLED (TIMEOUT) two hours in, mid-way through the FIRST rung of the
# calibration sweep itself, having produced zero rho output. The sweep is the one part
# whose cost couldn't be measured directly ahead of time: torch==1.9.1 (Graphormer's pin)
# predates `is_grads_batched`, so sensitivity.py's fallback runs one plain
# torch.autograd.grad call per output-basis-vector per target node per graph -- for the
# default ladder+n_graphs that's up to 252 (sum of DEFAULT_LADDER) x ~80 (embed_dim) x 10
# graphs = ~200k individual backward passes, an amount this pin never lets run batched.
# Trimmed the ladder/n_graphs below specifically to bring that number down for THIS run
# rather than gambling on a still-longer time limit; 4h leaves real margin either way.
#SBATCH --time=04:00:00
#SBATCH --output=calibrate_func-%j.out
#SBATCH --error=calibrate_func-%j.out

set -e

REPO_ROOT="/home/yandex/MLWG2026/liorayacob/graphs-project"
ENV="/home/yandex/MLWG2026/liorayacob/anaconda3/envs/graphormer"

cd "$REPO_ROOT"

# Not `conda activate` -- sbatch scripts run non-interactively (no login shell, no .tcshrc/
# .bashrc sourced), so activation hooks are unreliable. Calling the env's own python/lib
# directly is exactly what was validated by hand all day today; it needs nothing else.
export LD_LIBRARY_PATH="${ENV}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Graphormer's own wrapper.py decorates convert_to_single_emb with @torch.jit.script.
# On job 785420 (node s-004, RTX 2080 Ti) that crashed at runtime trying to JIT-compile a
# fused CPU kernel ("sh: 1: : Permission denied" -- some environment variable the fuser
# expects to hold a compiler path is empty/inaccessible on this node's compute allocation;
# the same job ran fine through this point on the TITAN Xp nodes tried earlier). PYTORCH_JIT=0
# is torch's own documented escape hatch: every @torch.jit.script function just runs as
# plain eager Python instead. Must be set before `python` starts -- torch reads it at
# import time, and by the time our own code could set it, `import torch` already happened.
# convert_to_single_emb is a few elementwise ops; the eager-mode cost is negligible.
export PYTORCH_JIT=0

echo "Running on node: $(hostname), GPU(s): $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# --max-dist/--d-min/--d-max: NOT the script's own defaults (20/5/20) -- those are generic
# placeholders. src/dataset_meta.py's real, dataset-specific values for Peptides (both
# func and struct) are max_dist=159 (== the measured max diameter, so the relative tail
# is complete for every graph) and abs_rho_window=(26, 80). Passing the wrong window here
# calibrates T against a DIFFERENT rho statistic than the one aggregate_results.py will
# actually report -- see README: "Must match what aggregate_results.py will use, or the
# calibration is for a different statistic than the one you report."
# --n-graphs 5, --ladder 4 8 16 32 (not the defaults, 10 and up to 128): cuts the sweep's
# total backward-pass count roughly 8x (5/10 graphs x 60/252 summed-ladder) relative to
# job 785429's attempt, which ran out of time inside the FIRST (cheapest) rung. Once this
# completes and the actual sweep-only wall time is known, both can be raised back toward
# the defaults for a more thorough calibration if there's time to spare.
"${ENV}/bin/python" -u scripts/calibrate_target_nodes.py \
    --backbone graphormer \
    --pe rwse \
    --dataset peptides-func \
    --epochs 20 \
    --max-dist 159 --d-min 26 --d-max 80 \
    --n-graphs 5 --ladder 4 8 16 32

echo "Done."
