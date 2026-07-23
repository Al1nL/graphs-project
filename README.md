# How Do Positional Encodings Affect Long-Range Sensitivity in Graph Transformers?

Experiment harness implementing the expanded design (3 backbones × 5 PEs × 3 datasets)
requested after proposal review. This is a **research harness**, not a from-scratch
reimplementation of GraphGPS / SAN / Graphormer: it wraps the three official codebases and
adds (a) a shared PE-precomputation module so every backbone sees the *same* PE definitions,
and (b) a shared, backbone-agnostic long-range sensitivity probe.

> **Sandbox note:** this was written and organized in an environment with no GPU and no
> network egress, so it has not been executed end-to-end here. Treat it as a ready-to-run
> scaffold for your own machine (see "Environment setup"). Syntax/import paths were checked
> by hand against each library's documented API as of early 2026 — re-verify against the
> exact commit you clone before a real run, since upstream repos do drift.

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
lrgb_pe_project/
├── README.md                     <- you are here
├── requirements.txt
├── src/
│   ├── pe/
│   │   └── compute_pe.py         <- backbone-agnostic PE computation (the shared "ground truth")
│   ├── adapters/
│   │   ├── graphgps_adapter.py   <- maps PE tensors -> GraphGPS posenc_* config/format
│   │   ├── san_adapter.py        <- maps PE tensors -> SAN's LPE input format
│   │   └── graphormer_adapter.py <- maps PE tensors -> Graphormer spatial_pos/edge_input/attn-bias
│   ├── sensitivity.py            <- backbone-agnostic Jacobian long-range sensitivity s̄(d)
│   └── run_experiment.py         <- single entry point: --backbone --pe --dataset --seed
├── configs/
│   ├── graphgps/                 <- 15 GraphGym YAML configs (5 PE x 3 datasets)
│   ├── san/                      <- 15 JSON configs (SAN's own config format)
│   └── graphormer/               <- 15 JSON configs (fairseq-style args)
├── scripts/
│   └── run_all.sh                <- loops over the full 3 x 5 x 3 grid, 3 seeds each
└── results/                      <- EMPTY. Filled by run_experiment.py after real runs.
```

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
# 1. Clone the three official backbones as siblings of this repo
git clone https://github.com/rampasek/GraphGPS.git
git clone https://github.com/DevinKreuzer/SAN.git
git clone https://github.com/microsoft/Graphormer.git

# 2. Create one env per backbone (their pinned dependency sets conflict with each other -
#    GraphGPS wants PyG>=2.0 + torch 1.9-2.x, SAN pins an older PyG/DGL combo, Graphormer
#    pins fairseq + its own CUDA ops). Do NOT try to share one env across all three.
conda env create -f envs/graphgps_env.yml
conda env create -f envs/san_env.yml
conda env create -f envs/graphormer_env.yml

# 3. Precompute the shared PEs once per dataset (cached to disk, reused by all backbones)
python src/pe/compute_pe.py --dataset peptides-func   --out cache/peptides-func/
python src/pe/compute_pe.py --dataset peptides-struct --out cache/peptides-struct/
python src/pe/compute_pe.py --dataset pascalvoc-sp    --out cache/pascalvoc-sp/

# 4. Run one cell of the grid
python src/run_experiment.py --backbone gps --pe rwse --dataset peptides-func --seed 0

# 5. Or run everything (45 cells x 3 seeds = 135 runs — budget compute accordingly;
#    see docs/rationale.docx for a fallback reduced grid if compute is tight)
bash scripts/run_all.sh
```

## Compute budget reality check

Full grid = 3 backbones × 5 PEs × 3 datasets × 3 seeds = **135 runs**. If that's not
feasible before the deadline, the fallback (documented in `docs/rationale.docx`) is:
drop to 1 seed for the two new backbones and keep 3 seeds only for GraphGPS (the primary
backbone), and/or drop PascalVOC-SP to a 20% node-subsampled variant for the SAN arm only
(SAN's full attention is O(n²) and PascalVOC-SP graphs average ~480 nodes).
