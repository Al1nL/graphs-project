# Running the grid on the TAU CS cluster

Three files: `run_grid.slurm` (the actual array job Slurm executes), `submit_grid.sh` (a
convenience wrapper around `sbatch` with TAU-appropriate defaults), and
`build_cache.slurm` (rebuilds the PE cache, which has to exist before any grid runs). You
almost always want `submit_grid.sh`; read `run_grid.slurm` if you need to change what a
cell actually runs.

`build_cache.slurm` is the one people miss. The ~5 GB PE cache is gitignored, so a fresh
clone on the cluster does not have it, and **every** backbone reads it — GraphGPS included,
since `backends/graphgps_pe_cache.py` landed. It is CPU-only work and requests no GPU.

## Why one job array, not 45 `sbatch` calls

The 5 PE x 3 seeds x 3 datasets grid is 45 cells. TAU's `studentbatch` partition caps a user
at **6 batch jobs** (see the cluster's own Slurm documentation). A Slurm **job array**
(`sbatch --array=0-44 ...`) is a single `sbatch` invocation — one job-ID family — so it
counts as one of those six, however many array *tasks* it expands to. Forty-five separate
`sbatch` calls would not fit that cap at all. This is the whole reason `run_grid.slurm` maps
`$SLURM_ARRAY_TASK_ID` to a (pe, seed, dataset) triple internally instead of taking them as
arguments to 45 separate submissions.

If your account's limits differ from what's described here (check with `sacctmgr -P -i show
user -s <you>` and `sinfo`), the array-job approach is still correct; only the partition
name and `--time` budget need to change.

## Quick start

```bash
# 1. ssh in
ssh <you>@slurm-client.cs.tau.ac.il

# 2. from the repo root, with your conda envs already set up (README "Environment setup")
cd /path/to/graphs-project

# 3. build the PE cache. It is gitignored, so a fresh clone does NOT have it, and every
#    backbone reads it -- including GraphGPS since backends/graphgps_pe_cache.py landed.
#    Three datasets, one array task each, CPU-only, ~45 min for the slowest (VOC).
sbatch --partition=studentbatch --array=0-2 \
       --export=RAW_DIR=/vol/scratch/$USER/raw_data \
       scripts/slurm/build_cache.slurm

# 4. calibrate T once per backbone (NOT per PE/dataset -- it's a property of the probe and
#    the graph regime; see calibration.py's own docstring). This needs a trained checkpoint,
#    so do at least one manual training run first, or use --demo to sanity-check the
#    calibration pipeline itself while a real checkpoint isn't ready yet:
python scripts/calibrate_target_nodes.py --backbone gps --pe rwse --dataset peptides-func \
    --checkpoint results/raw/gps_rwse_peptides-func/0/ckpt/best.pt

# 5. dry-run the submission (prints the sbatch command, submits nothing)
scripts/slurm/submit_grid.sh gps 32 --dry-run

# 6. submit for real
scripts/slurm/submit_grid.sh gps 32

# 7. watch it
squeue --me
tail -f slurm_logs/pe-grid_<jobid>_<taskid>.out
```

Run the SAN grid the same way once its fork is cloned and pinned:
```bash
scripts/slurm/submit_grid.sh san 32
```

Both can be queued at once (two array-job submissions, each countable separately toward
the 6-job cap — two jobs used, four left for anything else you need that month).

## What each partition on this cluster actually buys you

(Condensed from the cluster's own Slurm documentation; verify against `sinfo` yourself,
since limits and partition names can change.)

| Partition | Max time | What it's for | Relevant here |
|---|---|---|---|
| `studentbatch` | 3 days | Batch jobs, **max 6 per user** | Default in `submit_grid.sh`. One array submission = one of your six. |
| `studentkillable` | 1 day | Low-priority, pre-emptible | Faster to get scheduled, but your job can be stopped and requeued by higher-priority work mid-training. `run_grid.slurm` sets `--requeue` so a pre-empted cell restarts rather than dying, at the cost of re-training that one cell from scratch. |
| `studentrun` | 3 hours | Interactive testing only | Use this to smoke-test ONE cell interactively before submitting the full array — see "Smoke test" below. Too short for a real training run. |
| `killable` | 1 day | Default research partition, pre-emptible | Available if you have research-account access rather than a student one. |
| `gpu-<research-group>` | 1–5 days | Priority partition | Needs explicit permission; ask your supervisor/TA if your project has one. Longest runway if you have it. |

`--partition` is mandatory on every `sbatch` call on this cluster; `submit_grid.sh` always
sets it (default `studentbatch`), so you never need to pass it yourself unless overriding.

## GPU selection

The cluster exposes GPU type as a `--constraint` feature, not a partition. Available
features (from the cluster's hardware table): `tesla_v100`, `quadro_rtx_8000`,
`geforce_rtx_3090`, `titan_xp`, `geforce_rtx_2080`, `a100`, `a5000`, `a6000`, `l40s`.

`submit_grid.sh` leaves `--constraint` empty by default (any available GPU). If a cell is
failing with an out-of-memory error on an older/smaller card, or you want to avoid the
oldest hardware (`titan_xp`, `geforce_rtx_2080`) for consistency across seeds of the same
cell, request newer cards explicitly:

```bash
scripts/slurm/submit_grid.sh gps 32 --constraint="a5000|a6000|l40s|geforce_rtx_3090"
```

Do NOT vary `--constraint` across seeds of the *same* (backbone, pe, dataset) cell if you
care about wall-clock comparisons between them — different GPU generations have different
throughput, which would confound a "how long did training take" comparison, though not the
task metric or sensitivity curve themselves.

## Smoke test before submitting 45 cells

`studentrun` (3-hour interactive) is for exactly this. Get one node:

```bash
srun --partition=studentrun --gpus=1 --cpus-per-task=4 --mem=16000 --pty bash
# now on a compute node:
conda activate graphgps_env
python src/run_experiment.py --backbone gps --pe none --dataset peptides-func --seed 0 \
    --num-target-nodes 32 --dry-run     # sanity-checks the adapter config, trains nothing
```

Confirm the adapter config prints what you expect before spending real GPU time. Drop
`--dry-run` to actually train inside the interactive session if you want to watch the first
few epochs before trusting the batch array with the full run.

## Concurrency (`--throttle`)

`submit_grid.sh`'s `--throttle` (default 4) sets `--array=0-44%N` — at most `N` array tasks
running at once. This is a courtesy on a shared cluster, not a correctness requirement:
every cell is independent, so throttling only changes how fast the grid drains, not what it
produces. Turn it up if you have a priority allocation; turn it down (or to `%1`) on a
pre-emptible partition if you'd rather have fewer cells stopped and requeued at once when
higher-priority work arrives.

## Logs and results

- `slurm_logs/pe-grid_<jobid>_<taskid>.out` / `.err` — one pair per array task, written by
  `run_grid.slurm`'s `#SBATCH --output`/`--error` directives.
- `results/<backbone>_<pe>_<dataset>_seed<seed>.json` — the real per-cell result, written
  by `run_experiment.run_cell()` (see the top-level README's "Implementation status" for
  why this used to be silently empty and no longer is).
- `results/runs.csv` is only written by `scripts/launch.py`, which these Slurm scripts do
  **not** go through (they call `run_experiment.py` directly per cell, since Slurm is
  already doing the grid iteration `launch.py` would otherwise do). If you want the CSV
  rollup too, run `scripts/aggregate_results.py` afterward — it reads the same
  `results/*.json` files regardless of how they were produced.

## GRPE

`run_grid.slurm` skips any array task whose PE is `grpe` (exits 0 immediately, logging why)
for both backbones, because GRPE is not yet a drop-in on either (see the top-level README's
"Implementation status" — it needs an attention-bias hook that hasn't landed). This means a
default `submit_grid.sh gps 32` submission "runs" 45 array tasks but only 36 of them
actually train anything; the other 9 (3 datasets x 3 seeds of PE=grpe) exit immediately.
Once the GRPE hook lands for a backbone, delete that guard from `run_grid.slurm`.
