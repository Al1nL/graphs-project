#!/bin/bash
# Clone the forked backbones to the sibling paths src/config.py expects, wire up an
# `upstream` remote for syncing, and check out the pinned commit.
#
#   bash scripts/setup_upstream.sh              # all backbones with a fork URL recorded
#   bash scripts/setup_upstream.sh gps          # just one
#
# Why forks and not upstream directly: a commit SHA is a REFERENCE. It assumes the object
# still exists on someone else's server, so pinning alone survives ordinary drift but not a
# force-push, a rename, or a deletion. The fork preserves the objects; the pin says which
# one. They are complementary.
#
# The forks are also the only sane home for our architectural adaptations -- SAN+GRPE and
# GraphGPS's GRPE attention-bias hook are genuine additions to the published models, and
# uncommitted edits in an unversioned clone is the most fragile place they could sit.
#
# After forking a new backbone on GitHub, add its URL to config.FORK_URLS and re-run this.

set -e
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

python - "$@" <<'PYEOF'
import os, subprocess, sys
sys.path.insert(0, "src")
from config import FORK_URLS, PINNED_COMMITS, UPSTREAM_PATHS, UPSTREAM_URLS

wanted = sys.argv[1:] or list(FORK_URLS)
for backbone in wanted:
    fork = FORK_URLS.get(backbone)
    path = UPSTREAM_PATHS[backbone]
    if not fork:
        print(f"[{backbone}] no fork URL in config.FORK_URLS -- fork "
              f"{UPSTREAM_URLS[backbone]} on GitHub, then add the URL there and re-run.")
        continue

    if not os.path.isdir(os.path.join(path, ".git")):
        print(f"[{backbone}] cloning {fork} -> {path}")
        subprocess.run(["git", "clone", fork, path], check=True)
    else:
        print(f"[{backbone}] already cloned at {path}")

    remotes = subprocess.run(["git", "-C", path, "remote"], capture_output=True,
                             text=True).stdout.split()
    if "upstream" not in remotes:
        subprocess.run(["git", "-C", path, "remote", "add", "upstream",
                        UPSTREAM_URLS[backbone]], check=True)
        print(f"[{backbone}] added upstream remote {UPSTREAM_URLS[backbone]}")

    pin = PINNED_COMMITS.get(backbone)
    if pin:
        head = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        if head != pin:
            print(f"[{backbone}] checking out pinned {pin[:12]} (was {head[:12]})")
            subprocess.run(["git", "-C", path, "checkout", "--quiet", pin], check=True)
        else:
            print(f"[{backbone}] at pinned commit {pin[:12]}")
    else:
        head = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        print(f"[{backbone}] NOT PINNED. Record this in config.PINNED_COMMITS before "
              f"producing any results:\n    \"{backbone}\": \"{head}\",")

    # Cloning is NOT enough to make the package importable. GraphGPS registers its custom
    # modules into GraphGym as an import side effect (`import graphgps`), which only
    # resolves once the clone is installed into the active env. This script cannot do it:
    # each backbone needs its OWN env (their PyG/DGL/CUDA pins conflict), and the right one
    # may not be active -- installing into the wrong env is worse than not installing.
    print(f"[{backbone}] NEXT, in this backbone's conda env:  pip install -e {path}")
PYEOF

echo
echo "Verify with: python -c \"import sys; sys.path.insert(0,'src'); from config import check_pinned; print(check_pinned('gps', strict=False))\""
