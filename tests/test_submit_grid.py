"""Tests for scripts/slurm/submit_grid.sh's cell filters.

    python tests/test_submit_grid.py

The wrapper turns `--pe`/`--dataset` name lists into Slurm array indices by re-deriving
run_grid.slurm's mapping in bash. Two scripts computing the same arithmetic from two
hand-maintained copies of the axis lists is a drift hazard, and submit_grid.sh says so in
a comment ("if you change PES there, change ALL_PES here too"). A comment is not a check.

Drift here is close to undetectable by eye: a wrong index submits a real cell, which
trains perfectly well and records a T calibrated for a different dataset.

Runs the script with --dry-run, which prints the sbatch command and submits nothing.
Needs bash; fails rather than skips if it is missing, since a silent skip would leave the
hazard unguarded on exactly the machine where someone edits these files.
"""

import os
import re
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMIT = os.path.join(REPO, "scripts", "slurm", "submit_grid.sh")
RUN_GRID = os.path.join(REPO, "scripts", "slurm", "run_grid.slurm")

N_SEED = 3


def _bash(*args):
    proc = subprocess.run(["bash", SUBMIT, *args], cwd=REPO, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def _array_from(output):
    m = re.search(r"-> --array=([0-9,]+)", output)
    assert m, f"no array line in output:\n{output}"
    return [int(x) for x in m.group(1).split(",")]


def _axis(path, name):
    """Read a bash array literal, e.g. `PES=(none lappe ...)`, out of a script."""
    with open(path, encoding="utf-8") as f:
        src = f.read()
    m = re.search(rf"^\s*{name}=\(([^)]*)\)", src, re.M)
    assert m, f"{name}=(...) not found in {path}"
    return m.group(1).split()


def test_axis_lists_agree_with_run_grid_slurm():
    """The two scripts must enumerate the same axes in the same ORDER.

    Order is what makes the index arithmetic mean anything: submit_grid.sh computes
    ds_i * N_PE * N_SEED + seed_i * N_PE + pe_i, and run_grid.slurm inverts it. Reorder
    one list and every index still resolves to a valid cell -- just the wrong one.
    """
    assert _axis(SUBMIT, "ALL_PES") == _axis(RUN_GRID, "PES")
    assert _axis(SUBMIT, "ALL_DS") == _axis(RUN_GRID, "DATASETS")
    assert _axis(RUN_GRID, "SEEDS") == ["0", "1", "2"], (
        "N_SEED is hardcoded to 3 in submit_grid.sh; a different seed axis breaks the "
        "index arithmetic silently")


def test_filters_produce_the_indices_the_mapping_implies():
    """Compare the shell's answer to an independent re-derivation of the same formula."""
    pes = _axis(SUBMIT, "ALL_PES")
    dss = _axis(SUBMIT, "ALL_DS")
    n_pe = len(pes)

    def expected(want_pe, want_ds):
        return sorted(
            ds_i * n_pe * N_SEED + seed_i * n_pe + pe_i
            for ds_i, ds in enumerate(dss) if ds in want_ds
            for seed_i in range(N_SEED)
            for pe_i, pe in enumerate(pes) if pe in want_pe
        )

    cases = [
        (["--dataset=peptides-func"], pes, ["peptides-func"]),
        (["--pe=rwse"], ["rwse"], dss),
        (["--dataset=pascalvoc-sp", "--pe=none,lappe,rwse,signnet"],
         ["none", "lappe", "rwse", "signnet"], ["pascalvoc-sp"]),
        (["--dataset=peptides-func,peptides-struct", "--pe=signnet"],
         ["signnet"], ["peptides-func", "peptides-struct"]),
    ]
    for flags, want_pe, want_ds in cases:
        code, out = _bash("gps", "32", *flags, "--dry-run")
        assert code == 0, f"{flags} failed:\n{out}"
        assert _array_from(out) == expected(want_pe, want_ds), (
            f"{flags} produced the wrong cells\n{out}")


def test_an_unfiltered_submission_is_still_the_whole_grid():
    """The filters are opt-in; without them nothing about the default submission moves."""
    code, out = _bash("gps", "32", "--dry-run")
    assert code == 0, out
    assert "--array=0-44%" in out, f"default range changed:\n{out}"


def test_a_misspelled_name_is_rejected_rather_than_quietly_dropped():
    """The previous matcher scanned for names, so a typo contributed nothing and ran a
    SMALLER grid than asked for -- invisible on a 45-cell submission."""
    for flag, bad in (("--pe", "rwsee"), ("--dataset", "peptides_func")):
        code, out = _bash("gps", "32", f"{flag}={bad}", "--dry-run")
        assert code != 0, f"{flag}={bad} was accepted:\n{out}"
        assert bad in out and "known:" in out, (
            f"the error must name the bad value and the valid ones:\n{out}")


def test_array_range_and_the_filters_refuse_to_be_combined():
    """They set the same variable. --pe used to overwrite --array-range silently, so a
    submission carrying both ran something the command line did not say."""
    code, out = _bash("gps", "32", "--dataset=peptides-func", "--array-range=0-14",
                      "--dry-run")
    assert code != 0, f"conflicting flags were accepted:\n{out}"
    assert "array-range" in out


def test_spanning_datasets_warns_about_the_single_T():
    """One array carries one NUM_TARGET_NODES, and T is calibrated per (backbone,
    dataset). Spanning datasets is allowed -- sometimes one T is adequate for both -- but
    it must not pass without saying so."""
    code, out = _bash("gps", "32", "--dataset=peptides-func,pascalvoc-sp", "--dry-run")
    assert code == 0, out
    assert "NUM_TARGET_NODES" in out and "per dataset" in out, (
        f"no warning about one T across several datasets:\n{out}")

    code, out = _bash("gps", "32", "--dataset=peptides-func", "--dry-run")
    assert "per dataset" not in out, f"warned on a single-dataset submission:\n{out}"


def test_slurm_scripts_find_the_repo_via_slurm_submit_dir():
    """Regression test for twelve array tasks that died in under six seconds each.

    Both scripts navigated to the repo with `cd "$(dirname "$0")/../.."`, whose comment
    claimed it worked "regardless of where sbatch was invoked from". The opposite is true:
    sbatch copies the batch script to the compute node's spool directory and runs the
    COPY, so $0 is /var/spool/slurmd/job<id>/slurm_script and the cd lands in /var/spool.
    The first `mkdir -p slurm_logs` there is denied, `set -e` aborts, and the whole array
    fails with empty stdout and a 67-byte stderr.

    It had never been caught because it only breaks under sbatch -- which is the only way
    these scripts are meant to be run, but every prior invocation had been srun calling
    python directly.
    """
    for path in (RUN_GRID, os.path.join(os.path.dirname(RUN_GRID), "build_cache.slurm")):
        with open(path, encoding="utf-8") as f:
            src = f.read()
        name = os.path.basename(path)

        assert "SLURM_SUBMIT_DIR" in src, (
            f"{name} does not use SLURM_SUBMIT_DIR; under sbatch it will cd into the "
            f"spool directory instead of the repo")
        # the $0 form may remain ONLY as the fallback inside the parameter expansion
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("cd ") and "dirname" in stripped:
                assert "SLURM_SUBMIT_DIR" in stripped, (
                    f"{name} still navigates by $0 alone: {stripped}")

        assert "not the graphs-project root" in src, (
            f"{name} does not verify it landed in the repo; without that the next failure "
            f"is whatever breaks first, several steps from the cause")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nall tests passed")
