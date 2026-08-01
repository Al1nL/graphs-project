"""Tests for the GraphGPS integration that run WITHOUT GraphGPS installed.

    python -m pytest tests/test_graphgps_backend.py

GraphGPS needs its own environment (yacs, pytorch_lightning, its pinned PyG, and the
compiled torch_scatter), so the rest of this suite must keep passing on a machine that has
none of that. What is checked here is exactly the part that has to hold regardless:

  * the imports are LAZY -- `import run_experiment` and `--dry-run` must not drag GraphGPS
    in, or the launcher and the whole test suite break wherever it is absent,
  * the unsupported paths fail with a message that says what to do,
  * the PE spec stays consistent with src/pe/compute_pe.py.

The parts that genuinely need GraphGPS -- config construction, model wiring, the Jacobian
probe against a live GPSModel -- were verified separately against the real fork and real
Peptides graphs; see the notes in graphgps_backend.py. They are not asserted here because a
test that silently skips is worse than one that is absent.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import run_experiment  # noqa: E402
from backends import graphgps_backend  # noqa: E402


def test_importing_the_orchestrator_does_not_import_graphgps():
    """The launcher's --dry-run, and this whole suite, must work with no GraphGPS env."""
    assert "graphgps" not in sys.modules, (
        "importing run_experiment pulled in GraphGPS; the import must stay lazy or the "
        "launcher cannot enumerate a grid on a machine without GraphGPS's environment")


def test_make_model_fn_rejects_backbones_that_are_not_wired():
    for backbone in ("san", "graphormer"):
        try:
            run_experiment.make_model_fn(None, backbone, None, None)
        except NotImplementedError as exc:
            assert backbone in str(exc)
            continue
        raise AssertionError(f"{backbone} should not claim to be implemented")


def test_grpe_is_refused_with_the_reason():
    """GraphGPS has no attention-bias hook; GRPE needs a real architectural change, not a
    config flag. Refusing loudly beats silently training something that is not GRPE."""
    from config import RunConfig

    try:
        graphgps_backend.graphgps_train(RunConfig("gps", "grpe", "peptides-func", 0))
    except NotImplementedError as exc:
        msg = str(exc)
        assert "GRPEBiasedAttention" in msg and "architectural" in msg
        return
    raise AssertionError("GRPE must be refused until the GPSLayer hook exists")


def test_missing_clone_is_reported_actionably():
    try:
        graphgps_backend.ensure_graphgps_importable("/definitely/not/a/clone")
    except FileNotFoundError as exc:
        assert "setup_upstream.sh" in str(exc)
        return
    raise AssertionError("a missing GraphGPS clone must raise")


def test_pe_spec_matches_compute_pe_dimensions():
    """dim_pe here and K_LAP / K_RWSE there are two statements of the same quantity. If
    they drift, GraphGPS trains on a PE of a different width than the shared cache holds."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "pe"))
    import compute_pe

    assert graphgps_backend.PE_SPEC["lappe"][2] == compute_pe.K_LAP
    assert graphgps_backend.PE_SPEC["rwse"][2] == compute_pe.K_RWSE
    # PEs that carry no node-feature channels must declare dim_pe 0, or the content-width
    # bookkeeping in probe_widths silently shifts
    assert graphgps_backend.PE_SPEC["none"][2] == 0
    assert graphgps_backend.PE_SPEC["grpe"][2] == 0


def test_every_pe_and_dataset_axis_is_covered():
    from config import DATASETS, PES

    assert set(graphgps_backend.PE_SPEC) == set(PES)
    assert set(graphgps_backend.BASE_CONFIG) == set(DATASETS)
    assert set(graphgps_backend.DATASET_NODE_ENCODER) == set(DATASETS)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nall tests passed")
