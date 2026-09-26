"""
collect_cross_backbone_results.py
==================================
Pulls every real result JSON from all three backbone branches (graphGPS,
graphormer, san-transformer) via `git show <branch>:<path>`, without checking
those branches out, and writes them into one flat directory so
aggregate_results.py can be run once across the whole cross-backbone grid.

Each branch only ever committed its OWN backbone's results/*.json (never
merged into master), so this is the one place that combines them. Filenames
already carry the backbone prefix (gps_/graphormer_/san_), so there is no
collision.

Usage:
    python scripts/collect_cross_backbone_results.py [--out results_all]
"""
import argparse
import os
import subprocess

BRANCHES = ["origin/graphGPS", "origin/graphormer", "origin/san-transformer"]


def list_result_jsons(branch):
    out = subprocess.run(
        ["git", "ls-tree", "-r", branch, "--name-only"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    return [p for p in out if p.startswith("results/") and p.endswith(".json")]


def show(branch, path):
    return subprocess.run(
        ["git", "show", f"{branch}:{path}"],
        capture_output=True, text=True, check=True,
    ).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_all")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    total = 0
    for branch in BRANCHES:
        paths = list_result_jsons(branch)
        for p in paths:
            content = show(branch, p)
            dest = os.path.join(args.out, os.path.basename(p))
            with open(dest, "w") as f:
                f.write(content)
            total += 1
        print(f"{branch}: {len(paths)} result file(s)")
    print(f"Wrote {total} result file(s) to {args.out}/")


if __name__ == "__main__":
    main()
