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
# Usage (partition and account are already set as directives below -- you only need
# --array, which selects which datasets to build):
#
#   sbatch --array=0-2 scripts/slurm/build_cache.slurm
#
#   # raw downloads somewhere other than ./raw_data (see RAW_DIR below):
#   sbatch --array=0-2 --export=RAW_DIR=/path/with/room/raw_data \
#          scripts/slurm/build_cache.slurm
#
#   # just one dataset (VOC is index 2 and by far the longest):
#   sbatch --array=2 scripts/slurm/build_cache.slurm
#
# BEFORE RUNNING THIS AT ALL: check whether the cache you need already exists. On a shared
# cluster another account may have built it, and reading theirs makes "every backbone sees
# the identical PE" true by construction rather than by two builds happening to agree --
# LAPACK fixes no eigenvector sign convention, so two independent builds under different
# conda envs can legitimately differ in sign, and in a degenerate eigenspace in basis.
# Symlink theirs (`ln -s <their-repo>/cache cache`) in preference to rebuilding.
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
#SBATCH --account=gpu-students
# studentkillable, NOT cpu-killable. The CS Slurm guide advertises cpu-killable (CPU-only,
# 5-day) and it is the better queue for a job that asks for no GPU -- but entitlement is
# per account, and a student account may hold exactly one pairing:
#
#     $ sacctmgr -P show assoc user=$USER format=account,partition,qos
#     Account|Partition|QOS
#     gpu-students|studentkillable|normal
#
# Anything else fails at submission with "Invalid account or account/partition combination
# specified". Check your own associations before changing this; if you do have cpu-killable,
# prefer it, since this job requests no GPU and should not sit in a GPU queue.
#SBATCH --partition=studentkillable
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=240
# "killable" means pre-emptible. A pre-empted build restarts from scratch (there is no
# resume -- see the TIME AND SIZE note above), so ask Slurm to requeue rather than fail.
# That is safe here specifically because of the manifest guard below: a requeued task over
# a COMPLETED cache exits immediately instead of rebuilding it.
#SBATCH --requeue

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
# script. Nothing GPU-specific is touched. Override with --export=CONDA_ENV=..., which
# takes a name or a full path prefix (see run_grid.slurm for why the path form matters).
# Locating conda: `conda` is NOT on PATH on this cluster by default (`which conda` ->
# "Command not found"), and a batch shell does not source your login rc, so
# `conda info --base` can fail here even when conda works in an interactive session.
# CONDA_BASE names the installation root explicitly; it falls back to asking conda only
# when conda happens to be on PATH.
CONDA_BASE="${CONDA_BASE:-}"
if [ -z "$CONDA_BASE" ] && command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
fi
if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  echo "ERROR: no conda installation found." >&2
  echo "  conda is not on PATH here and CONDA_BASE is unset or wrong." >&2
  echo "  Pass the installation root, e.g.:" >&2
  echo "    --export=ALL,CONDA_BASE=/home/yandex/<course>/<user>/miniforge3,..." >&2
  echo "  (CONDA_BASE=$CONDA_BASE)" >&2
  exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-graphgps_env}"
conda activate "$CONDA_ENV" || {
  echo "ERROR: conda activate '$CONDA_ENV' failed. If the env lives outside your" >&2
  echo "envs_dirs (e.g. on netapp because of the home quota), pass the full prefix:" >&2
  echo "  sbatch --array=0-2 --export=ALL,CONDA_ENV=/path/to/envs/graphgps_env $0" >&2
  exit 1
}
[ -n "${CONDA_PREFIX:-}" ] || { echo "ERROR: CONDA_PREFIX unset after activate" >&2; exit 1; }
echo "env: $CONDA_PREFIX"

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
