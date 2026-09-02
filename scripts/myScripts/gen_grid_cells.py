"""
gen_grid_cells.py
==================
Write one line per grid cell ("backbone pe dataset seed") to a plain text file, for a
SLURM job array to read by line number ($SLURM_ARRAY_TASK_ID). Exists because running the
whole --preset reduced grid (75 cells) as ONE launch.py process is not feasible on this
cluster: cells run sequentially in one Python process, Graphormer alone measured ~8-10h
per real cell, and the SLURM partition here caps a single job at 24h
(studentkillable: MaxTime=1-00:00:00) -- one giant job would TIMEOUT having finished maybe
one or two cells. Many small per-cell jobs, submitted as a SLURM array, can run in
PARALLEL across the partition's nodes instead of queued behind each other in one process,
and each stays safely under the 24h cap.

Reuses config.grid()'s SAME --preset reduced filter launch.py applies (config.py x
launch.py's own logic), so this cannot silently drift from what `launch.py --dry-run
--preset reduced` would show you.

SAN is excluded entirely, not just left to fail per-cell: san_train is a 100%
NotImplementedError stub (see run_experiment.py) for every single cell, so including its
15 cells here would only burn 15 SLURM array slots on guaranteed, instant failures.
GraphGPS's 9 GRPE cells (real NotImplementedError, see graphgps_backend.py) and
Graphormer's 5 pascalvoc-sp cells per PE are LEFT IN deliberately -- they still fail fast
and cheap (no wasted GPU time), and excluding them here would let this file drift from
what config.grid() actually defines as "the reduced grid" without a specific reason tied
to THIS script's purpose (avoiding a guaranteed-100%-stub backbone, not a partial gap).

Usage:
    python scripts/myScripts/gen_grid_cells.py > results/grid_cells_reduced.txt
    wc -l results/grid_cells_reduced.txt   # sanity-check the count before submitting an array
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from config import DEFAULT_SEEDS, grid  # noqa: E402

if __name__ == "__main__":
    configs = [c for c in grid() if c.backbone != "san"]
    # Mirror launch.py's --preset reduced filter exactly (see launch.py's own comment):
    # all seeds for gps, only the first seed for the other backbones.
    configs = [c for c in configs if c.backbone == "gps" or c.seed == DEFAULT_SEEDS[0]]
    for c in configs:
        print(f"{c.backbone} {c.pe} {c.dataset} {c.seed}")
