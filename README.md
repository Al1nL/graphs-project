# How Do Positional Encodings Affect Long-Range Sensitivity in Graph Transformers?

Experiment harness implementing the expanded design (3 backbones × 5 PEs × 3 datasets)
requested after proposal review. This is a **research harness**, not a from-scratch
reimplementation of GraphGPS / SAN / Graphormer: it wraps the three official codebases and
adds (a) a shared PE-precomputation module so every backbone sees the *same* PE definitions,
and (b) a shared, backbone-agnostic long-range sensitivity probe.

> **Status note:** the real LRGB data is downloaded and all three PE caches are built and
> verified; the probe, the calibration tool and the GraphGPS integration have been run
> against real graphs. What has *not* happened is a training run — no model has been
> trained, `results/` is empty, and there is no GPU here, so every number in the paper is
> still ahead of us. See "Implementation status" for what is wired and what is a stub.
> Import paths for the not-yet-cloned backbones were checked by hand against each library's
> documented API as of early 2026 — re-verify against the exact commit you clone, since
> upstream repos drift.

## Why this structure

The professor's comment was: don't let a PE's apparent effect be a GraphGPS-specific
artifact, and don't let a dataset's result be a Peptides-func-specific artifact. So we now
vary two axes independently:

- **Backbone axis** (architecturally distinct, see `docs/rationale.docx` for the reasoning):
  1. **GraphGPS** (hybrid MPNN + Transformer) — primary backbone, unchanged from the proposal.
  2. **SAN** (Spectral Attention Network) — full/sparse attention with a dedicated *learned*
     spectral PE module; no message-passing branch.
  3. **Graphormer** — pure attention, no message passing at all; PEs enter as *attention
     biases* (centrality/spatial/edge), not as node features.
- **Dataset axis** (all within LRGB, so splits/metrics/eval protocol stay comparable):
  1. **Peptides-func** (graph classification, AP) — from the original proposal.
  2. **Peptides-struct** (graph regression, MAE) — same graphs as Peptides-func, different
     task → isolates task-generalization from graph-regime generalization.
  3. **PascalVOC-SP** (node classification, macro-F1) — superpixel graphs, different diameter/
     degree distribution → isolates graph-regime generalization.
  - COCO-SP was deliberately **excluded**: the original LRGB paper reports SAN failing to
    converge on COCO-SP within a 60-hour budget, which would make the SAN arm of the grid
    infeasible on typical course compute.

## Repository layout

```
graphs-project/
├── README.md                     <- you are here
├── requirements.txt
├── src/
│   ├── pe/
│   │   ├── compute_pe.py         <- backbone-agnostic PE computation (the shared "ground truth")
│   │   └── cache.py              <- streaming writer + memory-mapped reader, versioned
│   ├── adapters/
│   │   ├── graphgps_adapter.py   <- maps PE tensors -> GraphGPS posenc_* config/format
│   │   ├── san_adapter.py        <- maps PE tensors -> SAN's LPE input format
│   │   └── graphormer_adapter.py <- maps PE tensors -> Graphormer spatial_pos/edge_input/attn-bias
│   ├── backends/
│   │   └── graphgps_backend.py   <- REAL integration: drives GraphGPS's own train loop, and
│   │                                wraps a trained GPSModel for the Jacobian probe
│   ├── config.py                 <- run schema (backbone x pe x dataset x seed) + version locking
│   ├── dataset_meta.py           <- per-dataset caps, ρ windows, GRPE bucketing
│   ├── calibration.py            <- target-node budget sweep + decision rule
│   ├── sensitivity.py            <- backbone-agnostic Jacobian long-range sensitivity s̄(d),
│   │                                plus the scale-free summaries (s̃(d), ρ) and the
│   │                                graph-clustered bootstrap
│   └── run_experiment.py         <- single entry point: --backbone --pe --dataset --seed
├── configs/
│   ├── graphgps/                 <- 15 GraphGym YAML configs (5 PE x 3 datasets)
│   ├── san/                      <- 15 JSON configs (SAN's own config format)
│   └── graphormer/               <- 15 JSON configs (fairseq-style args)
├── docs/
│   └── analysis-plan.md          <- amended success criteria + pre-registered ρ windows.
│                                    Dated BEFORE any results exist; read this first.
├── tests/                        <- pins the probe against a brute-force Jacobian.
│                                    torch + networkx only; no GPU, dataset, or backbone.
├── scripts/
│   ├── launch.py                 <- THE entry point: grid, seeding, pre-flight, CSV/W&B
│   ├── calibrate_target_nodes.py <- one-off convergence check for the probe's T
│   ├── run_all.sh                <- superseded by launch.py; kept for reference
│   └── aggregate_results.py      <- Table 1 + figures; ρ is the primary statistic
├── raw_data/                     <- gitignored, 5.2 GB. LRGB downloads; see setup step 4.
├── cache/                        <- gitignored, 5.0 GB. Built PE caches, one file per graph.
└── results/                      <- EMPTY, and not in a fresh clone; run_experiment.py
                                     creates it. Filled after real runs.
```

## Implementation status

The shared machinery — PE computation and cache, the sensitivity probe, calibration,
aggregation, the launcher — is complete and tested. The per-backbone training integrations
are not, and that is the critical path:

| backbone | training | probe wrapper | notes |
|---|---|---|---|
| GraphGPS | **wired** (4 of 5 PEs) | **wired** | `src/backends/graphgps_backend.py`; GRPE refused, see below |
| SAN | stub | stub | repo not cloned or forked yet |
| Graphormer | stub | stub | repo not cloned or forked yet |

`graphgps_train` drives GraphGPS's **own** run loop, starting from its tuned reference YAML
for the dataset and overriding only the PE block, so every arm differs in exactly one thing.
`make_gps_model_fn` runs a trained `GPSModel` up to — but not including — the task head and
returns node embeddings. Both import GraphGPS lazily, so `--dry-run` and the whole test
suite work on a machine with no GraphGPS environment.

Two things to know before trusting cross-PE numbers from the GraphGPS arm:

- **GraphGPS+GRPE raises rather than running.** GraphGPS has no native attention-bias hook,
  so GRPE needs `GPSLayer`'s self-attention replaced by
  `adapters.graphgps_adapter.GRPEBiasedAttention` and the `spd_bucket`/`edge_type` tensors
  threaded onto the batch. That is an architectural addition, not a config change, and is
  left for a separate pass; the other four arms are drop-ins.
- **The content width differs per PE, and this is not fixable in the probe.** GraphGPS holds
  `dim_inner` constant and makes room for the PE by *shrinking* the atom encoder, so at
  `dim_inner=96` the content channels measure 96 / 80 / 76 / 64 / 96 for
  No-PE / LapPE / RWSE / SignNet / GRPE. Slicing to content compares ‖J‖_F over different
  column counts (which `assert_shared_width` correctly refuses); using the full `dim_inner`
  is identical across arms but perturbs PE channels too. `probe_widths` returns both and
  takes no side — the choice belongs in the paper, not hidden in a wrapper.

## Reporting the sensitivity results

Raw s̄(d) is **not** comparable across backbones — it conflates decay *shape* with overall
*gain*, and gain is set by LayerNorm placement, residual scaling, depth and width rather
than by the PE. Rank cells on **ρ**, the long-range mass fraction; it is a ratio of sums,
so the gain cancels exactly. Raw curves stay in the appendix. The full argument and the
amended proposal success criteria are in `docs/analysis-plan.md`.

ρ is reported on two axes, with distinct jobs (`src/dataset_meta.py`):

- **absolute `d`**, per-dataset windows — primary *within* a dataset; the axis
  over-squashing theory is stated in. Windows are **(26, 80)** for Peptides and
  **(14, 36)** for PascalVOC-SP.
- **`max_dist` = the dataset's full diameter** — **159** for Peptides, **54** for
  PascalVOC-SP. This is a *measurement* cap, not a reporting one: measuring wider fills the
  relative tail bins of the largest graphs, while `long_range_fraction` ignores every bucket
  past the window's `d_max`, so absolute ρ is unchanged. Anything smaller leaves the biggest
  graphs with partially sampled tails, which biases relative ρ **downward** precisely where
  the effect is expected to be strongest (see `docs/analysis-plan.md`, Amendment 5).
- **relative `d/diam(G)`**, one shared window — primary *across* datasets, and largely
  immune to the population shift whereby far buckets can only draw from graphs big enough
  to have them. Needs each graph's diameter in the result JSON.

GRPE's spatial-bias table is a **model** parameter and moves with this: it uses T5-style
bucketing (exact to d=8, log-spaced to 128, plus a dedicated *unreachable* bucket) rather
than a hard cap, so measuring out to `max_dist` (159 on Peptides) does not make GRPE's tail
an artefact of the cap. Changing it requires re-training the GRPE cells.

```bash
python tests/test_sensitivity.py            # each tests/test_*.py is a standalone script;
                                            # pytest also collects them, but it is not a
                                            # dependency and is not in requirements.txt
python scripts/aggregate_results.py         # --d-min/--d-max to test window sensitivity
```

### Calibrate the probe's target-node budget first

`compute_sensitivity_curve`'s `num_target_nodes` (T) has **no default** — it controls how
much of each graph's distance profile is observed, and the paper's claims live in the
sparse far buckets, so it is calibrated rather than guessed. Run once per (backbone,
dataset), then put the reported T in your run config and quote the printed sentence:

```bash
python scripts/calibrate_target_nodes.py --demo      # exercises the pipeline, no model needed
python scripts/calibrate_target_nodes.py --backbone gps --pe rwse --dataset peptides-func
```

It sweeps T and picks the smallest rung that is unbiased in ρ, dense enough in the tail,
and no wider in its bootstrap CI than the densest rung. See `docs/analysis-plan.md`
(Amendment 4) for why all three criteria are needed — ρ-stability alone accepts almost any
T, because the CI is dominated by between-graph variance that does not shrink with T.

## The 5 PE variants (unchanged from proposal, now computed once and shared)

| PE | Level | Native fit | Adaptation needed |
|---|---|---|---|
| No-PE | — | all three | none |
| LapPE | node feature | GraphGPS, SAN | Graphormer: concatenated as extra node feature (not its usual mode) |
| RWSE | node feature | GraphGPS | SAN: concatenated alongside LPE; Graphormer: extra node feature |
| SignNet-PE | node feature | GraphGPS (via custom encoder) | SAN: replaces its own LPE module; Graphormer: extra node feature |
| GRPE | attention bias | Graphormer (native family) | GraphGPS: custom attention-bias hook (see `graphgps_adapter.py`); SAN: added as an additive bias term next to its edge-existence bias (an explicit, documented extension of SAN, not part of the original SAN paper) |

Every backbone gets all 5 PEs so the grid is fully crossed, but two cells (SAN+GRPE,
GraphGPS-attention-bias-GRPE) required a genuine architectural adaptation rather than a
drop-in. This is called out explicitly in the paper draft and in `docs/rationale.docx` —
it's a source of confound we can't fully remove, only document.

## Environment setup (on your own GPU machine)

```bash
# 1. FORK each backbone on GitHub, add the fork URL to src/config.py FORK_URLS, then:
bash scripts/setup_upstream.sh          # clones your forks as siblings, adds an
                                        # `upstream` remote, checks out the pinned commit
# Do NOT `git clone` upstream directly -- pre-flight rejects it (see "Version locking").
# Status today: gps forked and pinned; san and graphormer still need forking.

# 2. Create one env per backbone (their pinned dependency sets conflict with each other -
#    GraphGPS wants PyG>=2.0 + torch 1.9-2.x, SAN pins an older PyG/DGL combo, Graphormer
#    pins fairseq + its own CUDA ops). Do NOT try to share one env across all three.
conda env create -f envs/graphgps_env.yml
conda env create -f envs/san_env.yml
conda env create -f envs/graphormer_env.yml

# 3. Pin the commit. setup_upstream.sh prints the exact line to paste into
#    config.PINNED_COMMITS for any backbone that is cloned but not yet pinned.

# 4. Precompute the shared PEs once per dataset (cached to disk, reused by all backbones)
python src/pe/compute_pe.py --dataset peptides-func   --out cache/peptides-func/
python src/pe/compute_pe.py --dataset peptides-struct --out cache/peptides-struct/
python src/pe/compute_pe.py --dataset pascalvoc-sp    --out cache/pascalvoc-sp/

# 5. Calibrate the probe's target-node budget (once per backbone x dataset)
python scripts/calibrate_target_nodes.py --backbone gps --dataset peptides-func --pe rwse

# 6. Launch. --dry-run first; --resume to continue an interrupted grid.
python scripts/launch.py --dry-run
python scripts/launch.py --num-target-nodes 32 --preset reduced --wandb
```

## Version locking

Three things drift underneath this project, and each would silently invalidate results
rather than break loudly:

| Drifts | Locked by | Detected by |
|---|---|---|
| the three cloned upstream backbones | `config.PINNED_COMMITS` | `launch.py` pre-flight |
| our PE computation | `config.PE_CACHE_VERSION` + cache manifest | `PECache` refuses a stale cache |
| our analysis code | `config.repo_sha()` | recorded in every result row |

`PINNED_COMMITS` starts as `None`, not `"main"` — a `None` is a checkable "nobody pinned
this yet", whereas `"main"` is a lie that looks like a pin. A grid run half before and half
after an upstream change is not a controlled comparison, so `launch.py` refuses to start
until the pins are filled (override with `--no-strict-pins` for throwaway runs only).

### Why forks, given we already pin commits

A SHA is a *reference*: it assumes the object still exists on someone else's server. Pinning
survives ordinary upstream drift, but **not** a force-push, a rename, or a deletion — in all
three the pin dangles and there is no way back to the pinned state. The fork preserves the
objects; the pin identifies which one. They are complementary, not alternatives.

The fork is also the only sane home for our architectural adaptations: SAN+GRPE and
GraphGPS's GRPE attention-bias hook are genuine additions to the published models, and
uncommitted edits inside an unversioned clone is the most fragile place they could sit.

`check_pinned` therefore verifies that each clone's `origin` is *your fork*, not upstream —
a clone of upstream carries the same SHA today and loses it the moment upstream rewrites
history. HTTPS and SSH remote forms are treated as equivalent.

| backbone | forked | pinned |
|---|---|---|
| GraphGPS | `pazflashner/GraphGPS` | `28015707` |
| SAN | not yet | — |
| Graphormer | not yet | — |

**All teammates must pin the same forks.** If two people pin different ones, their results
are not comparable and the grid silently stops being a controlled experiment.

## PE cache format

`compute_pe.py` streams one graph at a time to disk; nothing accumulates in memory, in
either direction. This is not optional at LRGB scale:

```
PascalVOC-SP:  479 nodes^2 x 11,355 graphs = 2.6 GB   for ONE dense field, as uint8
```

Three decisions follow, all in `src/pe/cache.py`:

- **Only `spd` is stored.** `spd_bucket` and `edge_type_id` are pure functions of it, so
  storing them tripled the footprint to ~7.8 GB for nothing — and baking the bucket into
  the cache is what made it go stale when the GRPE scheme changed. Both are derived on read
  via a 256-entry lookup table, so the bucketing can now change without recomputing.
- **uint8, 255 = unreachable.** The number that matters here is the *maximum* diameter, not
  the average: measured over every split from the built caches it is **159** (Peptides) and
  **54** (VOC-SP), both far below the sentinel, so nothing is capped. The writer raises
  rather than aliasing a large diameter onto 255.
- **One file per graph, `mmap_mode="r"` on read.** The OS pages in only the graphs touched.

## Sanity checks

Every PE is checked against a **hand-calculated 10-node graph** (`tests/test_pe_sanity.py`)
— closed forms, not values copied from a previous run of this code, since a golden-value
test seeded from the implementation passes forever including when the implementation is
wrong. The fixture is the 10-cycle, where all four have closed forms: Laplacian eigenvalues
`1 - cos(2πk/10)`, RWSE return probability `C(2m,m)/2^{2m}` at even steps and 0 at odd ones,
and `d(i,j) = min(|i-j|, 10-|i-j|)`. A path and a disconnected graph cover distinct degrees
and unreachable pairs.

## Compute budget reality check

Full grid = 3 backbones × 5 PEs × 3 datasets × 3 seeds = **135 runs**. If that's not
feasible before the deadline, the fallback (documented in `docs/rationale.docx`) is:
drop to 1 seed for the two new backbones and keep 3 seeds only for GraphGPS (the primary
backbone), and/or drop PascalVOC-SP to a 20% node-subsampled variant for the SAN arm only
(SAN's full attention is O(n²) and PascalVOC-SP graphs average ~480 nodes).
