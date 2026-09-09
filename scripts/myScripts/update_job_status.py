#!/usr/bin/env python3
"""
update_job_status.py
=====================
Enriches results/job_ids.csv with two snapshot-in-time columns:

    epochs_progress   "<checkpoints on disk>/<max_epoch for that PE+dataset>"
    run_status        ok | failed | <SLURM state, e.g. running/pending/timeout/cancelled>

Not live/continuous -- this is a snapshot at the moment you run it (matches how status
has been checked throughout this project: on demand, not via a background daemon).
Re-run any time you want a fresh read. Rewrites job_ids.csv in place, under the SAME
flock used when a SLURM job appends a new row (scripts/myScripts/slurm_graphormer_cell.sh),
so a job starting mid-update can't interleave and corrupt the file.

Usage:
    python scripts/myScripts/update_job_status.py [results_dir]

(results_dir defaults to "results" -- pass your own RESULTS_DIR if it lives elsewhere,
e.g. for a teammate's own results directory.)
"""
import csv
import fcntl
import json
import os
import subprocess
import sys

REPO_ROOT = "/home/yandex/MLWG2026/liorayacob/graphs-project"
CONFIG_DIR = os.path.join(REPO_ROOT, "configs", "graphormer")


def max_epoch_for(pe: str, dataset: str) -> int:
    """Reads the real configured max_epoch rather than assuming 200 -- see
    src/backends/graphormer_backend.py's build_graphormer_args, which sources this same
    per-(pe, dataset) JSON file."""
    path = os.path.join(CONFIG_DIR, f"graphormer_{pe}_{dataset}.json")
    try:
        with open(path) as f:
            return json.load(f).get("max_epoch", 200)
    except (OSError, json.JSONDecodeError):
        return 200


def epochs_done(results_dir: str, run_id: str) -> int:
    """fairseq names one checkpoint file per completed epoch (checkpoint<N>.pt) --
    counting them is a direct, non-invasive read of real progress, no need to touch the
    running job. checkpoint_best.pt/checkpoint_last.pt are extra copies, not new epochs."""
    d = os.path.join(results_dir, "raw", run_id)
    try:
        names = os.listdir(d)
    except OSError:
        return 0
    return sum(
        1 for n in names
        if n.startswith("checkpoint") and n.endswith(".pt")
        and n not in ("checkpoint_best.pt", "checkpoint_last.pt")
    )


def result_status(results_dir: str, run_id: str):
    """None if the cell hasn't produced a final result JSON yet."""
    path = os.path.join(results_dir, f"{run_id}.json")
    try:
        with open(path) as f:
            return json.load(f).get("status")
    except (OSError, json.JSONDecodeError):
        return None


def squeue_states() -> dict:
    """job_id -> SLURM state, for every job currently queued under this user."""
    out = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i %T"],
        capture_output=True, text=True,
    )
    states = {}
    for line in out.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) == 2:
            states[parts[0]] = parts[1]
    return states


def sacct_state(job_id: str) -> str:
    """Fallback for a job no longer in squeue (finished, timed out, cancelled, ...)."""
    out = subprocess.run(
        ["sacct", "-j", job_id, "-X", "-n", "-o", "State"],
        capture_output=True, text=True,
    )
    first_line = out.stdout.strip().splitlines()[0] if out.stdout.strip() else "unknown"
    return first_line.split()[0] if first_line.split() else "unknown"


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, "results")
    job_ids_csv = os.path.join(results_dir, "job_ids.csv")
    lock_path = job_ids_csv + ".lock"

    queued = squeue_states()

    with open(lock_path, "a+") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        try:
            with open(job_ids_csv, newline="") as f:
                rows = list(csv.DictReader(f))

            for row in rows:
                run_id = row["run_id"]
                total = max_epoch_for(row["pe"], row["dataset"])
                done = epochs_done(results_dir, run_id)
                row["epochs_progress"] = f"{done}/{total}"

                status = result_status(results_dir, run_id)
                if status is not None:
                    row["run_status"] = status
                elif row["job_id"] in queued:
                    row["run_status"] = queued[row["job_id"]].lower()
                else:
                    row["run_status"] = sacct_state(row["job_id"]).lower()

            fieldnames = list(rows[0].keys()) if rows else []
            tmp_path = job_ids_csv + ".tmp"
            with open(tmp_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows)
            os.replace(tmp_path, job_ids_csv)
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)

    print(f"Updated {job_ids_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
