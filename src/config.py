"""
config.py
=========
One config schema with (backbone x pe x dataset x seed) as explicit axes, plus the version
locking that makes a run reproducible six months later.

Why a schema at all
-------------------
Before this, a "run" was whatever `run_experiment.py --backbone gps --pe rwse ...` happened
to assemble from three adapter modules and a YAML/JSON file per backbone, with no single
object naming what varied. That is fine for one run and unmanageable for 135: nothing
records which axis combination produced which result file, nothing validates that a
requested cell exists, and nothing detects that the PE cache on disk was written by an
older, incompatible version of `compute_pe.py`.

Why version locking
-------------------
Three things drift underneath this project and each would silently invalidate results:

  1. The three upstream backbones are CLONED, not vendored (see README). GraphGPS, SAN and
     Graphormer all move. A result produced against an unpinned `main` cannot be
     reproduced, and worse, half a grid run before an upstream change and half after is
     not a controlled comparison at all.
  2. Our own PE computation changes. Fix 3 altered the GRPE bucketing scheme; any cache
     written before that is structurally different. Silently reusing it would mix two
     definitions of the same PE across cells.
  3. Our own analysis code changes. Recorded so a result file can be traced to the commit
     that produced it.

PINNED_COMMITS is deliberately populated with None rather than "main". A None is a loud,
checkable "nobody has pinned this yet"; a "main" is a lie that looks like a pin.
"""

import dataclasses
import hashlib
import json
import os
import subprocess
from typing import Optional

# --------------------------------------------------------------------------- axes
BACKBONES = ("gps", "san", "graphormer")
PES = ("none", "lappe", "rwse", "signnet", "grpe")
DATASETS = ("peptides-func", "peptides-struct", "pascalvoc-sp")
DEFAULT_SEEDS = (0, 1, 2)

TASK_METRIC = {
    "peptides-func": "ap",       # Average Precision, higher better
    "peptides-struct": "mae",    # Mean Absolute Error, LOWER better
    "pascalvoc-sp": "macro_f1",  # macro-F1, higher better
}

# --------------------------------------------------------------------------- versions
# Bump whenever src/pe/compute_pe.py changes what it writes. The cache manifest records
# this, and PECache refuses to load a cache whose version differs -- a stale cache is
# indistinguishable from a fresh one on disk, and mixing two PE definitions across cells of
# the same grid is the kind of error that produces a plausible, wrong table.
PE_CACHE_VERSION = 2   # v2: uint8 spd with 255=unreachable, per-graph files, derived buckets

# Upstream repos are cloned as siblings (see README "Environment setup"). Fill these in
# once, from `git -C ../GraphGPS rev-parse HEAD`, and never run a grid with any of them
# None for a backbone you are actually using.
PINNED_COMMITS = {
    "gps": None,          # rampasek/GraphGPS
    "san": None,          # DevinKreuzer/SAN
    "graphormer": None,   # microsoft/Graphormer
}

UPSTREAM_PATHS = {
    "gps": "../GraphGPS",
    "san": "../SAN",
    "graphormer": "../Graphormer",
}


def _git_sha(path: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def repo_sha() -> Optional[str]:
    return _git_sha(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_pinned(backbone: str, strict: bool = True) -> dict:
    """Compare the checked-out upstream commit against the pin. Returns a report dict.

    `strict` raises on a missing pin or a mismatch. Turn it off only for exploratory runs
    whose numbers will not appear anywhere.
    """
    pinned = PINNED_COMMITS.get(backbone)
    actual = _git_sha(UPSTREAM_PATHS.get(backbone, ""))
    report = {"backbone": backbone, "pinned": pinned, "actual": actual,
              "status": "ok" if (pinned and actual == pinned) else None}
    if pinned is None:
        report["status"] = "unpinned"
        msg = (f"{backbone} has no pinned commit in config.PINNED_COMMITS. Record "
               f"`git -C {UPSTREAM_PATHS.get(backbone)} rev-parse HEAD` before producing "
               "results: an unpinned upstream cannot be reproduced, and a grid run across "
               "an upstream change is not a controlled comparison.")
    elif actual is None:
        report["status"] = "missing"
        msg = f"{backbone} repo not found at {UPSTREAM_PATHS.get(backbone)}"
    elif actual != pinned:
        report["status"] = "mismatch"
        msg = (f"{backbone} is at {actual[:12]} but config pins {pinned[:12]}. Check out "
               "the pinned commit, or update the pin deliberately and re-run every cell "
               "for this backbone -- not just the new ones.")
    else:
        return report
    if strict:
        raise RuntimeError(msg)
    report["warning"] = msg
    return report


# --------------------------------------------------------------------------- run config
@dataclasses.dataclass(frozen=True)
class RunConfig:
    """One cell of the grid. Frozen: a run's identity must not change under it."""

    backbone: str
    pe: str
    dataset: str
    seed: int
    cache_dir: Optional[str] = None
    results_dir: str = "results"
    num_target_nodes: Optional[int] = None   # calibrate: scripts/calibrate_target_nodes.py
    max_dist: Optional[int] = None           # defaults per dataset from dataset_meta
    epochs: Optional[int] = None             # None -> backbone config's own value
    deterministic: bool = True
    notes: str = ""

    def __post_init__(self):
        for field, allowed in (("backbone", BACKBONES), ("pe", PES), ("dataset", DATASETS)):
            val = getattr(self, field)
            if val not in allowed:
                raise ValueError(f"{field}={val!r} not in {allowed}")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")

    @property
    def run_id(self) -> str:
        return f"{self.backbone}_{self.pe}_{self.dataset}_seed{self.seed}"

    @property
    def resolved_cache_dir(self) -> str:
        return self.cache_dir or os.path.join("cache", self.dataset)

    @property
    def metric_name(self) -> str:
        return TASK_METRIC[self.dataset]

    @property
    def result_path(self) -> str:
        return os.path.join(self.results_dir, f"{self.run_id}.json")

    def resolved_max_dist(self) -> int:
        if self.max_dist is not None:
            return self.max_dist
        from dataset_meta import max_dist
        return max_dist(self.dataset)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["run_id"] = self.run_id
        d["metric_name"] = self.metric_name
        d["max_dist"] = self.resolved_max_dist()
        return d

    def provenance(self, strict_pins: bool = True) -> dict:
        """Everything needed to reproduce this run, recorded into the result file."""
        return {
            "pe_cache_version": PE_CACHE_VERSION,
            "code_sha": repo_sha(),
            "upstream": check_pinned(self.backbone, strict=strict_pins),
            "config_hash": self.config_hash(),
        }

    def config_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def grid(backbones=BACKBONES, pes=PES, datasets=DATASETS, seeds=DEFAULT_SEEDS, **kw):
    """Every cell of the (backbone x pe x dataset x seed) product, as RunConfigs.

    Axis order is deliberate: dataset outermost so a run sweep reuses one PE cache before
    moving on, then backbone (one conda env at a time), then pe, then seed.
    """
    for dataset in datasets:
        for backbone in backbones:
            for pe in pes:
                for seed in seeds:
                    yield RunConfig(backbone=backbone, pe=pe, dataset=dataset,
                                    seed=seed, **kw)
