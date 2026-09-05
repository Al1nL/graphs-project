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

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import run_experiment  # noqa: E402
from backends import graphgps_backend  # noqa: E402



def _write_stats(run_dir, split, epoch_values, key, extra=None, also=None):
    """Write a GraphGPS-shaped stats.json: one JSON object per epoch, per split."""
    import json

    d = os.path.join(run_dir, split)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "stats.json"), "w") as f:
        for epoch, value in epoch_values:
            rec = {"epoch": epoch, key: value, "loss": 1.0}
            if also:
                rec[also] = value
            if extra:
                rec.update(extra)
            f.write(json.dumps(rec) + "\n")


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

        # write what GraphGPS's logger writes, where it writes it. val peaks at epoch 1;
        # test peaks at epoch 2. The reported number must be test AT EPOCH 1 -- 0.42, not
        # the larger 0.51, which would be selection on the test set.
        _write_stats(run_dir, "val",  [(0, 0.20), (1, 0.55), (2, 0.40)], "ap")
        _write_stats(run_dir, "test", [(0, 0.10), (1, 0.42), (2, 0.51)], "ap")

        assert graphgps_backend._read_best_metric(cfg, "ap") == 0.42, (
            "reported the best TEST value instead of test at the val-selected epoch")


def test_best_metric_direction_depends_on_the_metric():
    """`ap` is better high, `mae` is better low. Reading both as 'max' would silently
    report the worst epoch as the headline number on the regression datasets."""
    import json
    import tempfile
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory() as tmp:
        cfg = SimpleNamespace(out_dir=tmp, seed=0)
        run_dir = graphgps_backend.run_dir_for(tmp, 0)
        # val and test agree here, so the direction of selection is the only thing tested
        for split in ("val", "test"):
            _write_stats(run_dir, split, [(0, 0.5), (1, 0.2), (2, 0.9)], "mae", also="ap")

        assert graphgps_backend._read_best_metric(cfg, "mae") == 0.2   # lower better
        assert graphgps_backend._read_best_metric(cfg, "ap") == 0.9    # higher better


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


def test_truncated_loader_stops_early_but_stays_a_loader():
    """--smoke-test on the gps arm was a no-op: only san_backend honoured it, so a
    "smoke test" on GraphGPS silently ran the base YAML's full 200 epochs.

    The wrapper has to keep behaving like a DataLoader, because custom_train reaches past
    the iterator: it calls len(loader) to find the last batch for gradient accumulation,
    and logs len(loader.dataset).
    """
    class _FakeLoader:
        dataset = list(range(500))          # the real split, unshrunk
        batch_size = 128

        def __iter__(self):
            return iter(range(10))

        def __len__(self):
            return 10

    wrapped = graphgps_backend._TruncatedLoader(_FakeLoader(), 2)

    assert list(wrapped) == [0, 1], "the wrapper did not stop after n batches"
    assert len(wrapped) == 2, (
        "len() must report the TRUNCATED count -- custom_train uses `iter + 1 == "
        "len(loader)` as the gradient-accumulation boundary, so a full-length len() "
        "would mean the final step never fires")
    # unknown attributes fall through to the real loader
    assert len(wrapped.dataset) == 500, (
        "the wrapper shadowed .dataset; the split-size log line would understate it")
    assert wrapped.batch_size == 128

    # truncating to more batches than exist must not invent any
    assert len(graphgps_backend._TruncatedLoader(_FakeLoader(), 99)) == 10
    assert list(graphgps_backend._TruncatedLoader(_FakeLoader(), 99)) == list(range(10))


def test_truncated_loader_is_re_iterable():
    """custom_train iterates the train loader once per epoch. A generator-based wrapper
    that could only be consumed once would give an empty second epoch -- silently, since
    an empty loop raises nothing."""
    class _FakeLoader:
        def __iter__(self):
            return iter(range(10))

        def __len__(self):
            return 10

    wrapped = graphgps_backend._TruncatedLoader(_FakeLoader(), 2)
    assert list(wrapped) == [0, 1]
    assert list(wrapped) == [0, 1], "second pass over the loader came back empty"


def test_smoke_test_forces_one_epoch_and_drops_the_warmup():
    """Checked on the source: build_graphgym_cfg needs the GraphGPS clone.

    Both assignments matter. max_epoch 1 is the point of the flag. Dropping the warmup is
    what makes that one epoch informative: the reference schedule is cosine_with_warmup
    over 10 warmup epochs, so at max_epoch 1 the LR would stay at 0 for the entire run and
    the loss could not move -- a flat curve for a reason unrelated to the model.
    """
    import ast

    src = os.path.join(os.path.dirname(__file__), "..", "src", "backends",
                       "graphgps_backend.py")
    with open(src, encoding="utf-8") as f:
        tree = ast.parse(f.read())

    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "build_graphgym_cfg")

    guarded = [n for n in ast.walk(fn)
               if isinstance(n, ast.If) and "smoke_test" in ast.dump(n.test)]
    assert guarded, "build_graphgym_cfg does not branch on smoke_test"

    assigned = {t.attr for node in guarded for n in ast.walk(node)
                if isinstance(n, ast.Assign) for t in n.targets
                if isinstance(t, ast.Attribute)}
    assert "max_epoch" in assigned, "smoke_test does not force max_epoch"
    assert "num_warmup_epochs" in assigned, (
        "smoke_test does not drop the warmup, so its single epoch would train at LR 0")


def test_unwrap_finds_the_network_inside_the_lightning_wrapper():
    """create_model() returns a GraphGymModule, not the GPSModel.

    Regression test for a live failure: `unexpected GPSModel layout ['model']; expected
    'encoder' first`. GraphGPS's own training loop never notices the wrapper, because it
    only calls forward. The probe does -- it walks named_children() to run the encoder
    separately and replay the layer stack -- so it was handed a module whose only child
    is the network it actually wanted.
    """
    import torch

    class _GPSModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.post_mp = torch.nn.Linear(2, 2)

    class _GraphGymModule(torch.nn.Module):
        """Mirrors torch_geometric.graphgym.model_builder.GraphGymModule.

        The forwarding properties are the point of this fixture, not decoration. The
        first version of unwrap_graphgym_module tested `hasattr(model, "encoder")` and
        this test passed, because the fake wrapper did not have them -- while on the real
        wrapper hasattr answered True and the unwrap returned the wrapper unchanged. A
        wrapper that is transparent at the attribute level is exactly what has to be seen
        through here, so the fixture has to be transparent too.
        """

        def __init__(self, inner):
            super().__init__()
            self.model = inner

        @property
        def encoder(self):
            return self.model.encoder

        @property
        def post_mp(self):
            return self.model.post_mp

    net = _GPSModel()
    wrapper = _GraphGymModule(net)

    # the wrapper is transparent at the attribute level -- this is why hasattr cannot be
    # used to tell the two apart, and the assertion that pins the actual bug
    assert hasattr(wrapper, "encoder")
    assert [n for n, _ in wrapper.named_children()] == ["model"]

    assert graphgps_backend.unwrap_graphgym_module(wrapper) is net

    # idempotent: an already-unwrapped network passes straight through
    assert graphgps_backend.unwrap_graphgym_module(net) is net

    # nested wrappers unwrap all the way down
    assert graphgps_backend.unwrap_graphgym_module(
        _GraphGymModule(_GraphGymModule(net))) is net


def test_unwrap_terminates_on_a_self_referential_wrapper():
    """Bounded rather than `while True`: a module whose .model is itself must return,
    not hang. A test process that hangs gives far less information than one that fails."""
    import torch

    class _Loop(torch.nn.Module):
        @property
        def model(self):
            return self

    loop = _Loop()
    assert graphgps_backend.unwrap_graphgym_module(loop) is loop


def test_smoke_test_does_not_resume_or_checkpoint_into_the_real_cell():
    """Checked on the source; build_graphgym_cfg needs the GraphGPS clone.

    Regression test for a smoke test that silently trained NOTHING. custom_train does
    `for cur_epoch in range(start_epoch, max_epoch)`, and with auto_resume on, a stale
    checkpoint gave start_epoch 170 against the smoke test's max_epoch 1 -- an empty
    range. It logged "Task done", reported "Avg time per epoch: nan", and handed the
    probe a model it had never trained.

    The other two settings protect the real run rather than the smoke test: ckpt_period
    always checkpoints the final epoch, so a smoke test sharing the cell's run_dir would
    leave an epoch-0 checkpoint that the next real run would silently resume from -- a
    200-epoch cell quietly continuing from a model trained on two batches.
    """
    import ast

    src = os.path.join(os.path.dirname(__file__), "..", "src", "backends",
                       "graphgps_backend.py")
    with open(src, encoding="utf-8") as f:
        tree = ast.parse(f.read())

    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "build_graphgym_cfg")
    guarded = [n for n in ast.walk(fn)
               if isinstance(n, ast.If) and "smoke_test" in ast.dump(n.test)]
    assert guarded, "build_graphgym_cfg does not branch on smoke_test"

    body = "\n".join(ast.dump(n) for n in guarded)
    for field, why in (
            ("auto_resume", "a stale checkpoint would make the smoke test train nothing"),
            ("enable_ckpt", "the smoke test would leave a checkpoint the real run resumes"),
            ("out_dir", "the smoke test would write into the real cell's directory")):
        assert field in body, f"smoke_test does not override {field}: {why}"


def test_probe_gives_the_encoder_a_batch_vector_for_its_single_graph():
    """The probe passes one Data object, so `.batch` is None until we set it.

    Regression test for `AttributeError: 'NoneType' object has no attribute 'max'` from
    SignNet's batched_n_nodes during the probe. Only the signnet arm hit it: GPS's
    attention goes through to_dense_batch, which builds the zeros itself when handed
    None, so rwse/lappe/none all passed while signnet crashed on the same input.

    Verified functionally rather than by AST, with a stub model, since the assertion is
    about what the encoder actually receives.
    """
    import torch

    seen = []

    class _Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 3)

        def forward(self, b):
            seen.append(b.batch)
            b.x = self.lin(b.x)
            return b

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _Encoder()
            self.layers = torch.nn.Identity()
            self.post_mp = torch.nn.Linear(3, 3)

    class _Data:
        """Minimal stand-in for a PyG Data: .batch reads as None when never assigned."""

        def __init__(self, x, edge_index):
            self.x = x
            self.edge_index = edge_index
            self.num_nodes = x.shape[0]
            self.batch = None

        def clone(self):
            d = _Data(self.x.clone(), self.edge_index.clone())
            d.batch = None if self.batch is None else self.batch.clone()
            return d

        def to(self, device):
            return self

    data = _Data(torch.randn(5, 3), torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]))
    assert data.batch is None

    original = graphgps_backend.probe_widths
    graphgps_backend.probe_widths = lambda model: {"dim_inner": 3}
    try:
        graphgps_backend.make_gps_model_fn(_Model(), data)
    finally:
        graphgps_backend.probe_widths = original

    assert seen and seen[0] is not None, (
        "the encoder was handed batch=None; SignNet's batched_n_nodes calls .max() on it")
    assert torch.equal(seen[0], torch.zeros(5, dtype=torch.long)), (
        "one graph is a batch of one -- every node must map to graph 0")


def test_adapter_summary_is_derived_from_the_tables_that_actually_run():
    """The printed "resolved adapter config" must not drift from what the run uses.

    It had. build_posenc_config restated PE_SPEC/PE_ENCODER by hand and advertised
    phi_out_dim 32 while runs used 64, omitted raw_norm_type (BatchNorm for RWSE, which
    materially changes what the encoder sees), and named a "<cache>/{split}_pe.pt" file
    the cache stopped using. It is printed at the head of every run, so it read as a
    record of the configuration while describing something untrue.

    This asserts derivation, not equality with a second hardcoded copy -- another literal
    table here would be the same bug wearing a test.
    """
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from adapters.graphgps_adapter import build_posenc_config

    for pe, (_enc, posenc_key, dim_pe) in graphgps_backend.PE_SPEC.items():
        summary = build_posenc_config(pe, "cache/peptides-func")

        # nothing may still advertise the retired per-split .pt layout
        assert "_pe.pt" not in json.dumps(summary), (
            f"{pe} summary still names the retired <cache>/{{split}}_pe.pt layout")

        if posenc_key is None:
            for key in ("posenc_LapPE", "posenc_RWSE", "posenc_SignNet"):
                assert summary[key]["enable"] is False
            continue

        block = summary[posenc_key]
        assert block["enable"] is True
        assert block["dim_pe"] == dim_pe

        # every encoder-head field must match the table build_graphgym_cfg applies
        for field, value in graphgps_backend.PE_ENCODER[pe].items():
            assert block[field] == value, (
                f"{pe}.{field}: summary says {block[field]!r}, the run applies {value!r}")

    # the two specific drifts that prompted this
    assert build_posenc_config("signnet", "c")["posenc_SignNet"]["phi_out_dim"] == \
        graphgps_backend.PE_ENCODER["signnet"]["phi_out_dim"]
    assert build_posenc_config("rwse", "c")["posenc_RWSE"]["raw_norm_type"] == "BatchNorm"


def test_param_budget_warning_does_not_fire_on_the_reference_configs():
    """GraphGPS's own LRGB configs are 504,362 / 504,459 / 510,453 parameters (Table A.5,
    arXiv:2205.12454v3) -- all above a literal 500,000. The old threshold warned on a
    faithful reproduction, which teaches you to ignore the warning and so costs you the
    case it exists for: signnet at 576,138 is genuinely not parameter-matched."""
    assert graphgps_backend.PARAM_BUDGET_WARN > graphgps_backend.PARAM_BUDGET_REFERENCE_MAX
    assert graphgps_backend.PARAM_BUDGET_WARN < 576_138, (
        "the threshold must still catch the signnet arm, the one that is actually "
        "out of family")


def test_macro_f1_is_read_from_graphgpss_own_key():
    """config.TASK_METRIC calls it 'macro_f1'; GraphGPS's logger writes 'f1'.

    Regression test for a bug that would have hit EVERY pascalvoc-sp cell: the metric
    lookup found no 'macro_f1' key in any record, returned None, and the cell was written
    with status "ok" and metric_value null after training and probing perfectly well.

    The quantity was never wrong -- GraphGPS's 'f1' is macro-averaged
    (f1_score(..., average='macro')). Only the name differed.
    """
    import json
    import tempfile
    from types import SimpleNamespace

    assert graphgps_backend.GRAPHGPS_METRIC_KEY["macro_f1"] == "f1"

    with tempfile.TemporaryDirectory() as tmp:
        cfg = SimpleNamespace(out_dir=tmp, seed=0)
        run_dir = graphgps_backend.run_dir_for(tmp, 0)
        for split in ("val", "test"):
            # exactly the keys the VOC smoke run logged
            _write_stats(run_dir, split, [(0, 0.10), (1, 0.37), (2, 0.22)], "f1",
                         extra={"accuracy": 0.03, "auc": 0.51})

        assert graphgps_backend._read_best_metric(cfg, "macro_f1") == 0.37


def test_a_metric_absent_from_every_record_fails_instead_of_returning_none():
    """The silent None is what let the macro_f1/f1 mismatch hide. A missing FILE is a
    real 'no score yet' (a cell pre-empted before its first eval); a present file with no
    such key is a naming bug, and must be told apart from it."""
    import json
    import tempfile
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory() as tmp:
        cfg = SimpleNamespace(out_dir=tmp, seed=0)

        # no file at all -> None, not an error
        assert graphgps_backend._read_best_metric(cfg, "ap") is None

        _write_stats(graphgps_backend.run_dir_for(tmp, 0), "test",
                     [(0, 0.2)], "f1", extra={"accuracy": 0.03})

        try:
            graphgps_backend._read_best_metric(cfg, "ap")
        except RuntimeError as exc:
            assert "ap" in str(exc) and "f1" in str(exc), (
                f"the error must name both the metric sought and what was available: {exc}")
        else:
            raise AssertionError("a metric present in no record returned quietly")


def test_every_dataset_metric_has_a_graphgps_key():
    """A dataset whose metric is missing from the table falls through to its own name,
    which is how pascalvoc-sp failed. Make the table cover the axis explicitly."""
    from config import DATASETS, TASK_METRIC

    for dataset in DATASETS:
        metric = TASK_METRIC[dataset]
        assert metric in graphgps_backend.GRAPHGPS_METRIC_KEY, (
            f"{dataset}'s metric {metric!r} has no GraphGPS key mapping; if GraphGPS "
            f"logs it under a different name the cell will report a null metric")


def test_the_reported_score_is_never_the_best_test_score():
    """The distinction this function exists for, isolated.

    _read_best_metric used to take the max over the TEST split across every epoch, which
    is model selection on the test set. On the first real cell that would have reported
    whatever test AP peaked across 200 epochs instead of 0.6496, the test AP at epoch 151
    -- the epoch validation chose. Small in magnitude, and exactly the kind of thing a
    reviewer is entitled to throw out a table over.

    Here val and test peak at DIFFERENT epochs, so the two protocols give different
    numbers and only the correct one passes.
    """
    import tempfile
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory() as tmp:
        cfg = SimpleNamespace(out_dir=tmp, seed=0)
        run_dir = graphgps_backend.run_dir_for(tmp, 0)

        _write_stats(run_dir, "val",  [(0, 0.30), (1, 0.90), (2, 0.40)], "ap")
        _write_stats(run_dir, "test", [(0, 0.20), (1, 0.50), (2, 0.99)], "ap")

        got = graphgps_backend._read_best_metric(cfg, "ap")
        assert got == 0.50, f"expected test-at-val-selected-epoch 0.50, got {got}"
        assert got != 0.99, "reported the best test score -- selection on the test set"


def test_selection_direction_is_taken_from_the_metric_on_the_val_split():
    """For a lower-is-better metric the val ARGMIN selects the epoch. Getting the
    direction right on test but wrong on val would pick the worst validation epoch and
    still look plausible, since the returned number is a real test score either way."""
    import tempfile
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory() as tmp:
        cfg = SimpleNamespace(out_dir=tmp, seed=0)
        run_dir = graphgps_backend.run_dir_for(tmp, 0)

        _write_stats(run_dir, "val",  [(0, 0.90), (1, 0.10), (2, 0.50)], "mae")
        _write_stats(run_dir, "test", [(0, 0.80), (1, 0.15), (2, 0.60)], "mae")

        assert graphgps_backend._read_best_metric(cfg, "mae") == 0.15


def test_a_test_split_without_a_val_split_refuses_to_guess():
    """Falling back to the best test value when val is missing would reintroduce the bug
    exactly where it is least visible, so it raises instead."""
    import tempfile
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory() as tmp:
        cfg = SimpleNamespace(out_dir=tmp, seed=0)
        _write_stats(graphgps_backend.run_dir_for(tmp, 0), "test",
                     [(0, 0.2), (1, 0.9)], "ap")
        try:
            graphgps_backend._read_best_metric(cfg, "ap")
        except RuntimeError as exc:
            assert "validation" in str(exc).lower()
        else:
            raise AssertionError("selected an epoch without a validation split")


def test_sklearn_squared_shim_reproduces_the_old_semantics():
    """scikit-learn removed mean_squared_error's `squared` kwarg in 1.6; GraphGPS still
    calls it, and peptides-struct died at the end of epoch 0 with
    `TypeError: got an unexpected keyword argument 'squared'`.

    The exactness matters and is easy to get wrong. Old squared=False took the square
    root PER OUTPUT and then averaged. sqrt of the averaged MSE is a different number as
    soon as there is more than one target -- and peptides-struct has eleven. A shim that
    did the easy thing would run fine and report a subtly wrong RMSE every epoch.
    """
    import sys
    import types

    import numpy as np

    # a new-style sklearn: no `squared` parameter
    def modern_mse(y_true, y_pred, *, multioutput="uniform_average", sample_weight=None):
        per_output = np.average((np.asarray(y_true) - np.asarray(y_pred)) ** 2, axis=0,
                                weights=sample_weight)
        if isinstance(multioutput, str):
            return per_output if multioutput == "raw_values" else np.average(per_output)
        return np.average(per_output, weights=multioutput)

    fake = types.ModuleType("graphgps.logger")
    fake.mean_squared_error = modern_mse
    saved = {k: sys.modules.get(k) for k in ("graphgps", "graphgps.logger")}
    pkg = sys.modules.get("graphgps") or types.ModuleType("graphgps")
    pkg.logger = fake
    sys.modules["graphgps"] = pkg
    sys.modules["graphgps.logger"] = fake
    try:
        assert graphgps_backend.patch_sklearn_squared_kwarg() is True

        # eleven targets with deliberately UNEQUAL per-output errors, so the two
        # orderings of sqrt and average disagree
        rng = np.random.default_rng(0)
        y_true = rng.normal(size=(64, 11))
        y_pred = y_true + rng.normal(size=(64, 11)) * np.linspace(0.1, 3.0, 11)

        got = fake.mean_squared_error(y_true, y_pred, squared=False)

        per_output = modern_mse(y_true, y_pred, multioutput="raw_values")
        want = np.average(np.sqrt(per_output))          # sqrt first, then average
        naive = np.sqrt(np.average(per_output))         # the tempting wrong one

        assert abs(got - want) < 1e-12, f"got {got}, old sklearn would give {want}"
        assert abs(want - naive) > 1e-3, (
            "fixture is degenerate: the two orderings must differ for this to test "
            "anything")

        # squared=True must be untouched
        assert abs(fake.mean_squared_error(y_true, y_pred)
                   - modern_mse(y_true, y_pred)) < 1e-12
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_the_shim_is_a_no_op_on_an_sklearn_that_still_has_squared():
    """An older env needs no patching, and patching it anyway would replace a tested
    implementation with ours for no reason."""
    import sys
    import types

    def old_mse(y_true, y_pred, *, squared=True, multioutput="uniform_average"):
        return 0.0

    fake = types.ModuleType("graphgps.logger")
    fake.mean_squared_error = old_mse
    saved = {k: sys.modules.get(k) for k in ("graphgps", "graphgps.logger")}
    pkg = sys.modules.get("graphgps") or types.ModuleType("graphgps")
    pkg.logger = fake
    sys.modules["graphgps"] = pkg
    sys.modules["graphgps.logger"] = fake
    try:
        assert graphgps_backend.patch_sklearn_squared_kwarg() is False
        assert fake.mean_squared_error is old_mse
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_a_gpu_allocation_torch_cannot_see_is_an_error_not_a_warning():
    """Regression test for a wasted allocation.

    A node came up with broken CUDA. torch reported it as a WARNING -- "CUDA
    initialization: CUDA unknown error ... Setting the available devices to be zero" --
    auto_select_device() quietly chose cpu, and the job ran on. That is worse than a
    crash: the sweep is 10-50x slower on CPU, so a bounded wall clock kills it having
    produced nothing, and the failure looks like a timeout rather than a bad node.
    """
    import os

    import torch

    saved = {v: os.environ.get(v) for v in
             ("SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE", "CUDA_VISIBLE_DEVICES")}
    try:
        for v in saved:
            os.environ.pop(v, None)

        # nothing allocated -> silent, whatever torch reports. CPU-only jobs are normal:
        # build_cache.slurm requests no GPU at all.
        graphgps_backend.assert_gpu_if_slurm_allocated_one()

        os.environ["SLURM_JOB_GPUS"] = "0"
        if torch.cuda.is_available():
            # a machine with a working GPU must NOT be told off
            graphgps_backend.assert_gpu_if_slurm_allocated_one()
        else:
            try:
                graphgps_backend.assert_gpu_if_slurm_allocated_one()
            except RuntimeError as exc:
                msg = str(exc)
                assert "SLURM_JOB_GPUS=0" in msg, "the error must name what was allocated"
                assert "CPU" in msg and "resubmit" in msg.lower(), (
                    f"the error must say what went wrong and what to do: {msg}")
            else:
                raise AssertionError("a GPU allocation torch cannot see passed silently")

        # "NoDevFiles" is Slurm's way of saying no devices were actually attached, and
        # must not be read as an allocation
        os.environ["CUDA_VISIBLE_DEVICES"] = "NoDevFiles"
        os.environ.pop("SLURM_JOB_GPUS")
        graphgps_backend.assert_gpu_if_slurm_allocated_one()
    finally:
        for v, val in saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val


def test_both_gpu_entry_points_check_the_allocation():
    """graphgps_train and calibrate's load_real both call auto_select_device(); both must
    then verify it. Checked on the source, since neither runs without GraphGPS."""
    import ast

    for rel, funcs in (
        (("src", "backends", "graphgps_backend.py"), ("graphgps_train",)),
        (("scripts", "calibrate_target_nodes.py"), ("load_real",)),
    ):
        path = os.path.join(os.path.dirname(__file__), "..", *rel)
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for name in funcs:
            fn = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == name)
            called = {n.func.id for n in ast.walk(fn)
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
            assert "assert_gpu_if_slurm_allocated_one" in called, (
                f"{rel[-1]}:{name} selects a device without checking it is the one "
                f"Slurm allocated")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nall tests passed")
