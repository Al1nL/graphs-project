"""What .gitignore actually does to run outputs.

    python tests/test_gitignore.py

Asserted against `git check-ignore` rather than by reading the file, because the rules
here are order-dependent and involve two negations and a re-exclusion -- exactly the shape
that reads correctly and behaves otherwise. Precedent: this file previously carried
`results/` plus `!results/.gitkeep`, and the negation had never once fired, because git
does not descend into an excluded DIRECTORY to reconsider what is inside it. That was
invisible for as long as nobody checked what git actually did.

Needs git and a work tree. It fails rather than skips if either is missing: a silent skip
here would restore precisely the condition being guarded against.
"""

import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (path, must_be_ignored, why)
CASES = [
    ("results/gps_rwse_peptides-func_seed0.json", False,
     "the per-cell result is the scientific artefact and the only off-cluster copy"),
    ("results/san_lappe_pascalvoc-sp_seed1.json", False,
     "the rule must hold for the SAN arm's results too, not just this branch's"),

    ("results/gps_rwse_peptides-func_seed0_smoke.json", True,
     "a smoke result carries the full schema and status ok from 1 epoch and 2 graphs; it "
     "must not be shareable as a cell's real output"),

    ("results/raw/gps_rwse_peptides-func_seed0/0/ckpt/199.ckpt", True,
     "checkpoints are ~6 MB each, ~4 GB across the grid, and reproducible from the "
     "result plus the seed"),
    ("results/raw/gps_rwse_peptides-func_seed0/0/test/stats.json", True,
     "a single * does not match /, so the !results/*.json negation must not reach into "
     "results/raw/ and start tracking per-epoch logs"),

    ("results/runs.csv", True, "launch.py's rollup is derived from the JSONs"),
    ("results/calibration_target_nodes_gps_rwse_peptides-func.csv", True,
     "calibration output is not a cell result"),

    ("results/.gitkeep", False, "the directory itself must survive an empty results/"),

    ("cache/peptides-func/manifest.json", True,
     "the PE cache is ~5 GB and shared on the cluster; !results/*.json must not leak "
     "outside results/"),
]


def _check_ignored(path):
    """True if git ignores `path`. Asks git rather than reinterpreting the rules."""
    proc = subprocess.run(["git", "check-ignore", "-q", path],
                          cwd=REPO, capture_output=True)
    if proc.returncode not in (0, 1):
        raise RuntimeError(
            f"`git check-ignore {path}` returned {proc.returncode}; this test needs git "
            f"and a work tree. stderr: {proc.stderr.decode(errors='replace').strip()}")
    return proc.returncode == 0


def test_run_outputs_are_tracked_or_ignored_as_intended():
    wrong = []
    for path, want_ignored, why in CASES:
        got = _check_ignored(path)
        if got != want_ignored:
            wrong.append(f"  {path}\n      expected {'ignored' if want_ignored else 'tracked'}"
                         f", got {'ignored' if got else 'tracked'}\n      ({why})")
    assert not wrong, "gitignore does not do what it says:\n" + "\n".join(wrong)


def test_results_rule_is_not_a_bare_directory_exclude():
    """`results/` would make every negation below it dead, silently. It has to be
    `results/*` so git evaluates each entry and the negations can fire."""
    with open(os.path.join(REPO, ".gitignore"), encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    assert "results/*" in lines
    assert "results/" not in lines, (
        "a bare `results/` exclude makes !results/*.json and !results/.gitkeep dead "
        "rules -- git never descends into an excluded directory to reconsider them")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nall tests passed")
