#!/bin/bash
# build_cache.slurm
# =================
# Rebuild the shared PE cache on the cluster, as a 3-task job ARRAY -- one task per
# dataset, running concurrently.
#
# The cache is the ~5 GB of precomputed Laplacian eigenvectors, RWSE return probabilities
# and all-pairs shortest paths that EVERY backbone reads (src/pe/cache.py). It is
# gitignored, so cloning the repo on the cluster does not bring it: it has to be rebuilt
# there, or copied there. This script is the rebuild.
#
# Usage:
#
#   sbatch --partition=studentbatch --array=0-2 scripts/slurm/build_cache.slurm
#
#   # raw downloads on scratch (recommended -- see RAW_DIR below):
#   sbatch --partition=studentbatch --array=0-2 \
#          --export=RAW_DIR=/vol/scratch/$USER/raw_data \
#          scripts/slurm/build_cache.slurm
#
#   # just one dataset (VOC is index 2 and by far the longest):
#   sbatch --partition=studentbatch --array=2 scripts/slurm/build_cache.slurm
#
# RUN THIS BEFORE submit_grid.sh. Every training cell reads the cache, and since the
# GraphGPS arm started reading it too (backends/graphgps_pe_cache.py), a missing or
# half-built cache now fails the GPS arm as well as the SAN one.
#
# ---------------------------------------------------------------------------------------
# NO GPU. This is deliberate and is not an oversight.
# ---------------------------------------------------------------------------------------
# The whole precompute is scipy/LAPACK eigendecomposition and BFS on the CPU -- there is no
# CUDA in this path at all. Requesting a GPU would queue the job behind GPU demand for
# hours to leave a card idle for the entire build. If your account only has a GPU
# partition, it will still run; it just wastes the allocation.
#
# ---------------------------------------------------------------------------------------
# TIME AND SIZE (measured, not estimated -- from the Windows build of this same cache)
# ---------------------------------------------------------------------------------------
#   peptides-func     0.80 GB   15,535 graphs
#   peptides-struct   0.80 GB   15,535 graphs
#   pascalvoc-sp      3.40 GB   11,355 graphs   <- dominates: cost is quadratic in node
#                                                  count and VOC graphs are ~480 nodes
#   TOTAL             5.00 GB
#
# VOC took ~45 min single-threaded after the LapPE partial-solver speedup (11x -- see
# compute_pe.compute_lap_pe); it was ~2 h before. The 4 h limit below is deliberate
# headroom over that, because a cache that dies at 90% has to start over: PECacheWriter
# streams per-graph files but only writes manifest.json at the END, and PECache refuses to
# open a directory with no manifest. There is no resume.
#
# Disk: make sure the filesystem you point --out at has 5 GB free BEYOND the ~5 GB of raw
# LRGB downloads. Home quotas on shared clusters are routinely smaller than 10 GB.

#SBATCH --job-name=pe-cache
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=240

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root, regardless of where sbatch was invoked from
mkdir -p slurm_logs

DATASETS=(peptides-func peptides-struct pascalvoc-sp)

idx="${SLURM_ARRAY_TASK_ID:?this script must be submitted as an array, e.g. --array=0-2}"
if [ "$idx" -lt 0 ] || [ "$idx" -ge "${#DATASETS[@]}" ]; then
  echo "array index $idx is out of range; valid indices are 0-$(( ${#DATASETS[@]} - 1 ))" >&2
  exit 1
fi
DATASET="${DATASETS[$idx]}"

# Where raw LRGB downloads live. Defaults to the repo-relative path compute_pe.py has
# always used; override with --export=RAW_DIR=... to put ~5 GB of downloads on scratch.
RAW_DIR="${RAW_DIR:-./raw_data}"
CACHE_ROOT="${CACHE_ROOT:-cache}"
OUT="$CACHE_ROOT/$DATASET"

# The cache build needs torch, torch_geometric, scipy and networkx -- all present in the
# GraphGPS env, which is why it is reused here rather than adding a fourth env for one
# script. Nothing GPU-specific is touched.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate graphgps_env

# Refuse to overwrite a COMPLETE cache. A finished build has a manifest; a crashed one does
# not, and re-running over that is exactly what you want. Silently rebuilding a good cache
# would be a 45-minute no-op at best, and at worst would leave a half-written directory if
# THIS run dies -- destroying a cache that was fine.
if [ -f "$OUT/manifest.json" ]; then
  echo "SKIP: $OUT/manifest.json already exists -- this cache is complete." >&2
  echo "Delete the directory first if you really mean to rebuild it (e.g. after changing" >&2
  echo "K_LAP/K_RWSE in compute_pe.py, which requires a rebuild -- PECache refuses to load" >&2
  echo "a cache whose pe_cache_version does not match)." >&2
  exit 0
fi

echo "=== building PE cache: dataset=$DATASET out=$OUT raw=$RAW_DIR ==="
echo "started=$(date -Is)  host=$(hostname)"

python src/pe/compute_pe.py --dataset "$DATASET" --out "$OUT" --raw-dir "$RAW_DIR"

echo "finished=$(date -Is)"

# Fail loudly if the manifest is missing: compute_pe.py writes it last, so its absence
# means the build did not actually complete even if python exited 0.
if [ ! -f "$OUT/manifest.json" ]; then
  echo "ERROR: $OUT/manifest.json was not written -- the cache is INCOMPLETE and every" >&2
  echo "backbone will refuse to read it. Do not submit the grid against this." >&2
  exit 1
fi

echo "OK: $(du -sh "$OUT" | cut -f1) written to $OUT"
