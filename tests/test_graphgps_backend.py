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
    """Unwired backbones must refuse, and the refusal must say WHICH one.

    Case-insensitive on purpose: 'san' now dispatches to san_backend.make_san_model_fn,
    which raises its own message naming the backbone as 'SAN'. That is the right error --
    it explains that SAN's TRAINING is real and only the probe is missing, which the
    generic fallback here cannot say -- so the assertion checks the message names the
    backbone, not that it spells it in the config's lowercase.
    """
    for backbone in ("san", "graphormer"):
        try:
            run_experiment.make_model_fn(None, backbone, None, None)
        except NotImplementedError as exc:
            assert backbone.lower() in str(exc).lower(), (
                f"{backbone}'s NotImplementedError does not name it: {exc}")
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


def test_run_dir_and_metric_readback_agree_on_the_layout():
    """The trainer's output directory and the score reader's input path must be the same
    place. This is the regression test for a real cluster failure: graphgps_train never set
    cfg.run_dir at all, and GraphGPS raised `AttributeError: run_dir` from create_logger --
    but only AFTER the dataset build and the PE pre-transform had run, so a config mistake
    surfaced minutes in and pointed at yacs rather than at us.

    The quieter half is what this actually guards. Had run_dir been set to a directory that
    merely DISAGREED with _read_best_metric, nothing would have raised: training would
    finish, stats.json would be written somewhere else, the reader would find no file and
    return None, and the cell would record a null metric for a run that went fine.
    """
    import json
    import tempfile
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory() as tmp:
        cfg = SimpleNamespace(out_dir=os.path.join(tmp, "gps_rwse_peptides-func"), seed=3)
        run_dir = graphgps_backend.run_dir_for(cfg.out_dir, cfg.seed)

        # the SEED names the directory, not a 0-based run index: re-running a cell has to
        # land on the same directory or auto_resume cannot find the checkpoint it left
        assert os.path.basename(run_dir) == "3"

        # write what GraphGPS's logger writes, where it writes it
        test_dir = os.path.join(run_dir, "test")
        os.makedirs(test_dir)
        with open(os.path.join(test_dir, "stats.json"), "w") as f:
            for ap in (0.10, 0.42, 0.31):
                f.write(json.dumps({"ap": ap, "loss": 1.0}) + "\n")

        assert graphgps_backend._read_best_metric(cfg, "ap") == 0.42, (
            "the score reader did not find the stats file the trainer's run_dir points at")


def test_best_metric_direction_depends_on_the_metric():
    """`ap` is better high, `mae` is better low. Reading both as 'max' would silently
    report the worst epoch as the headline number on the regression datasets."""
    import json
    import tempfile
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory() as tmp:
        cfg = SimpleNamespace(out_dir=tmp, seed=0)
        test_dir = os.path.join(graphgps_backend.run_dir_for(tmp, 0), "test")
        os.makedirs(test_dir)
        with open(os.path.join(test_dir, "stats.json"), "w") as f:
            for v in (0.5, 0.2, 0.9):
                f.write(json.dumps({"mae": v, "ap": v}) + "\n")

        assert graphgps_backend._read_best_metric(cfg, "mae") == 0.2
        assert graphgps_backend._read_best_metric(cfg, "ap") == 0.9


def test_missing_stats_file_reports_none_rather_than_raising():
    """A cell can be pre-empted before its first eval lands. That must come back as
    'no score', not an exception that loses the rest of the grid's bookkeeping."""
    import tempfile
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory() as tmp:
        assert graphgps_backend._read_best_metric(
            SimpleNamespace(out_dir=tmp, seed=0), "ap") is None


def test_graphgps_train_sets_the_per_run_cfg_fields_before_they_are_read():
    """graphgps_train must assign the state main.py assigns inside its run loop.

    Read off the SOURCE, not by calling it, because graphgps_train needs the whole GraphGPS
    environment and this suite deliberately runs without it (see the module docstring). A
    structural assertion is a weak test in general; it is the right one here because the
    failure it guards is structural -- a missing assignment -- and because every one of
    these fields is read by code we do not own:

      run_dir  GraphGPS's create_logger, and get_ckpt_dir() == run_dir/ckpt, which is what
               auto_resume needs to survive pre-emption on studentkillable
      params   custom_train reads cfg.params directly when logging each epoch
      run_id   utils.py's wandb run naming

    Omitting run_dir is not hypothetical: it raised `AttributeError: run_dir` out of yacs
    on the cluster, from inside create_logger, after the dataset build and the ~2 min PE
    pre-transform had already been paid for. Nothing cheaper than a real GPU allocation
    caught it, which is exactly the situation worth a source-level test.
    """
    import ast

    src = os.path.join(os.path.dirname(__file__), "..", "src", "backends",
                       "graphgps_backend.py")
    with open(src, encoding="utf-8") as f:
        tree = ast.parse(f.read())

    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "graphgps_train")

    assigned = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "cfg"):
                assigned.setdefault(tgt.attr, node.lineno)

    for field in ("run_dir", "run_id", "params"):
        assert field in assigned, (
            f"graphgps_train never assigns cfg.{field}; GraphGPS reads it and will raise "
            f"AttributeError partway into the run")

    calls = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.setdefault(node.func.id, node.lineno)

    assert "set_printing" in calls, (
        "graphgps_train does not call set_printing(); GraphGPS reports every epoch through "
        "logging.info and an unconfigured root logger defaults to WARNING, so training "
        "would run completely silently")

    # ordering: both of these read cfg.run_dir, so it has to exist by the time they run
    for consumer in ("create_logger", "create_loader"):
        assert assigned["run_dir"] < calls[consumer], (
            f"cfg.run_dir is assigned after {consumer}() is called")
    assert assigned["run_dir"] < calls["set_printing"], (
        "set_printing() writes to cfg.run_dir/logging.log and must follow the assignment")


def test_every_pe_with_an_encoder_declares_a_working_head():
    """Each PE that contributes node channels must pin its own encoder head.

    Regression test for a live cluster failure: `ValueError: RWSENodeEncoder: Does not
    support 'none' encoder model`, raised from create_model after the dataset build and
    the ~2 min PE pre-transform. All three base configs enable only posenc_LapPE, so every
    other PE block sat at GraphGPS's defaults, where model is the literal string 'none'.
    The lappe arm worked only because it inherited a block the base YAML configures for
    its own use -- so 'it ran once' was never evidence the other arms would.
    """
    for pe, (enc_suffix, _key, _dim) in graphgps_backend.PE_SPEC.items():
        if enc_suffix is None:          # 'none' and 'grpe' add no node channels
            assert pe not in graphgps_backend.PE_ENCODER, (
                f"{pe} has no PE encoder but declares an encoder head")
            continue
        assert pe in graphgps_backend.PE_ENCODER, (
            f"{pe} has encoder {enc_suffix} but no entry in PE_ENCODER, so it would run "
            f"with GraphGPS's defaults -- model='none', which no encoder accepts")
        head = graphgps_backend.PE_ENCODER[pe]
        assert head.get("model", "none") != "none"
        assert "raw_norm_type" in head, (
            f"{pe} does not pin raw_norm_type; the default 'none' would silently train "
            f"without normalisation rather than failing")


def test_pe_encoder_models_are_spelled_as_each_encoder_expects():
    """The three encoders validate `model` differently, and two of them are
    case-sensitive, so the string has to match exactly:

      RWSE      lowercases first, then requires 'linear' or 'mlp' -- else ValueError
      SignNet   `if model_type not in ['MLP', 'DeepSet']` -- case-sensitive ValueError
      LapPE     `if model_type == 'Transformer': ... else: <DeepSet>` -- accepts ANYTHING
                as DeepSet, so a typo here would build a silently different encoder
                rather than raising. That last one is why this test checks LapPE too.
    """
    assert graphgps_backend.PE_ENCODER["rwse"]["model"].lower() in ("linear", "mlp")
    assert graphgps_backend.PE_ENCODER["signnet"]["model"] in ("MLP", "DeepSet")
    assert graphgps_backend.PE_ENCODER["lappe"]["model"] in ("Transformer", "DeepSet")


def test_signnet_rho_depth_is_positive():
    """SignNet raises "Num layers in rho model has to be positive" for post_layers < 1,
    and the GraphGPS default is 0 -- so the signnet arm was broken too, just with a
    different exception than the one RWSE hit first."""
    assert graphgps_backend.PE_ENCODER["signnet"]["post_layers"] >= 1


def test_build_graphgym_cfg_actually_applies_the_encoder_head():
    """Declaring PE_ENCODER is only half of it -- build_graphgym_cfg has to read it.

    Checked on the source because build_graphgym_cfg needs the GraphGPS clone (for the
    base YAML, and because cfg.posenc_RWSE only exists once GraphGPS registers it), which
    this suite runs without by design. Without this, PE_ENCODER could be dropped from
    build_graphgym_cfg entirely and every other test in this file would still pass while
    the RWSE arm went back to crashing on the cluster.
    """
    import ast

    src = os.path.join(os.path.dirname(__file__), "..", "src", "backends",
                       "graphgps_backend.py")
    with open(src, encoding="utf-8") as f:
        tree = ast.parse(f.read())

    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "build_graphgym_cfg")
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    assert "PE_ENCODER" in names, (
        "build_graphgym_cfg never reads PE_ENCODER, so the PE blocks keep GraphGPS's "
        "defaults (model='none') no matter what PE_ENCODER says")


def test_encoder_heads_do_not_vary_by_dataset():
    """PE_ENCODER is keyed by PE alone, and must stay that way. The grid varies PE and
    dataset independently; an encoder head that changed with the dataset would confound
    exactly the cross-dataset comparison this project is built to make."""
    from config import PES

    assert set(graphgps_backend.PE_ENCODER) <= set(PES)
    for head in graphgps_backend.PE_ENCODER.values():
        assert not any(d in head for d in graphgps_backend.BASE_CONFIG), (
            "a PE encoder head is being specialised per dataset")


def test_build_graphgym_cfg_runs_graphgyms_config_post_processing():
    """build_graphgym_cfg must call assert_cfg, which is not optional validation.

    main.py never calls it directly -- it arrives via load_cfg(), which is
    merge_from_file + merge_from_list + assert_cfg. Replicating only the merge, as this
    backend did, silently skips a set of REWRITES that despite the function's name are
    part of building a usable config:

        gnn.head 'default' -> cfg.dataset.task   (a sentinel; nothing registers 'default',
                                                  so it fails at model construction with
                                                  KeyError: 'default')
        model.loss_fun      coerced by task_type
        gnn.layers_post_mp  raised to >= 1
        dataset.transductive forced False for graph tasks

    Only the first one raises. The others would have quietly trained a slightly different
    model than the reference config describes, which is worse.
    """
    import ast

    src = os.path.join(os.path.dirname(__file__), "..", "src", "backends",
                       "graphgps_backend.py")
    with open(src, encoding="utf-8") as f:
        tree = ast.parse(f.read())

    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "build_graphgym_cfg")
    called = {n.func.id for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}

    assert "assert_cfg" in called, (
        "build_graphgym_cfg merges the base config but never calls assert_cfg, so "
        "GraphGym's post-processing never runs -- starting with gnn.head, which stays at "
        "the unregistered sentinel 'default'")
    assert "set_cfg" in called, (
        "build_graphgym_cfg must reset the global cfg before merging; it is a singleton "
        "and a previous cell's values would otherwise leak into this one")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nall tests passed")
