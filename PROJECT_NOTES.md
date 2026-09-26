# Project Notes — LRGB PE Sensitivity Study (for picking this up again cheaply)

## What this project is
Course project (ML with Graphs, TAU). Question: does positional-encoding (PE) choice affect a
Graph Transformer's long-range sensitivity, and is any such effect specific to one backbone/dataset?
Design: 3 backbones (GraphGPS, SAN, Graphormer) × 5 PEs (No-PE, LapPE, RWSE, SignNet-PE, GRPE) ×
3 LRGB datasets (Peptides-func, Peptides-struct, PascalVOC-SP).

## Repo / branch structure — READ THIS BEFORE TOUCHING GIT
- Remote: `https://github.com/Al1nL/graphs-project.git`. Main branch is **`master`**.
- Each backbone has its **own branch with its own `results/*.json`**, never merged into master:
  `graphGPS` (36 cells), `graphormer` (30 cells), `san-transformer` (45 cells after it filled in
  5 previously-2-seed cells — **always `git fetch` and re-check branch heads before trusting a
  local `results_all/` copy**, it goes stale silently).
- **`results-all`** branch (created from `origin/master`, pushed) holds the aggregation snapshot:
  `results_all/*.json` (pulled via `git show branch:path`, no checkout/merge), the generated
  `summary_table.csv`/`criterion_b_*.csv`/`kendall_w.csv`/`paired_bootstrap.csv`, `figures/`,
  `paper_ACL_format.tex`/`.pdf`, and the 6 `results_*.tex` fragments the paper `\input`s.
- To refresh after a backbone branch changes: re-run in order (from `results-all`):
  ```
  python scripts/collect_cross_backbone_results.py --out results_all
  python scripts/aggregate_results.py --results-dir results_all
  python scripts/make_paper_tables.py --in-dir results_all --out-dir .
  python scripts/make_ranking_figures.py --in-dir results_all --out-dir figures
  python scripts/make_combined_curves_figure.py --in-dir results_all --out figures/sensitivity_norm_combined.png
  python scripts/paired_bootstrap.py --in-dir results_all --out results_all/paired_bootstrap.csv
  ```
  Then **recompile the tex and manually re-check every hardcoded number in the prose** — the
  tables/figures are auto-generated, the prose interpreting them is not, and a data refresh can
  flip a qualitative claim (it has, twice: SAN's mechanism story, and see the open issues below).

## Core metric (src/sensitivity.py, src/dataset_meta.py)
- `s̄(d)` = mean Frobenius norm of Jacobian ∂h_v/∂x_u over sampled pairs at hop distance d.
- `ρ` = (Σ s̄(d) for d≥d_min) / (Σ s̄(d) for d≥1) over an **absolute** window (per-dataset,
  primary within-dataset) — a scale-free ratio so backbone "gain" cancels.
- `ρ_rel` = same ratio over the **relative** window d/diam(G) > 0.5 (bins 6-10 of 10) — the
  cross-dataset-comparable axis, what §5/all cross-backbone claims in the paper rank on.
- Bootstrap CIs are graph-clustered (resample graph identity, keep seed-copies together).
- **Known bug (found in review, not yet fixed as of last edit — check before trusting stars in
  Table 3/`results_rho.tex`)**: `aggregate_results.py`'s `rho_seed_std` is the seed-std of
  *absolute* ρ, not ρ_rel. `make_paper_tables.py`'s `significant()` uses it as the noise floor
  for ρ_rel's two-part significance rule (CI-disjoint AND exceeds seed-std) — wrong statistic.
  Fix: compute a real `rho_rel_seed_std` in `aggregate_results.py` (per-seed ρ_rel, not ρ).

## CRITICAL DATA FACT — Graphormer's probe sample size is 10, not 256
Verified directly from JSONs (`sensitivity_curves_per_graph` length):
GraphGPS = 256 graphs/seed, SAN = 256 graphs/seed, **Graphormer = 10 graphs/seed** (all
Graphormer files, no exceptions). `num_target_nodes` (T): gps=8, san=128, graphormer=16.
Consequence: Graphormer's ρ_rel bootstrap CI half-width (~0.036) and seed-to-seed sd (~0.024)
are both several times LARGER than the entire spread across its 4-5 PEs on Peptides-struct
(~0.007-0.009 range). Any claim of the form "Graphormer's PEs don't separate" or "Graphormer's
sign-flip / r=+1.00" is confounded with 25× less data than GraphGPS/SAN and should be stated
as "not measured precisely enough to distinguish" rather than as a property of Graphormer's
architecture, unless/until it's re-probed at 256 graphs (cheap: reuse the existing probe/resume
path, no retraining needed — this is a probe-only rerun).

## Other verified-true facts from the Sep 2026 review pass
- Tables 3/4 (`results_tables.tex`/`results_rho.tex`/`results_critb.tex`) numerically match
  `summary_table.csv`/`criterion_b_*.csv` to the last digit. Paired-bootstrap numbers in the
  prose match `paired_bootstrap.csv`. 37/45 cell arithmetic (12+15+10=37) and 111 JSONs=37×3
  seeds both check out. `collect_cross_backbone_results.py` really does use `git show` without
  checkout, as documented.
- **SAN mechanism claim was FALSE as written**: paper said LapPE uses SAN's "native learned-PE
  module" while RWSE/SignNet-PE are "concatenated features bypassing" it. Checked
  `src/backends/san_backend.py`'s `PE_SPEC`: lappe/rwse/signnet/grpe **all** set `LPE: "node"`
  (identical dispatch slot); they differ only in an internal `_variant` selecting the encoder
  class (default/rwse/signnet/grpe), and GRPE alone adds an attention-bias term on top. There is
  no "concatenated feature vs native module" distinction in the code. The *empirical* pattern
  (LapPE/GRPE collapse, RWSE elevated, SignNet inconsistent across the two Peptides sets) is
  real and still worth reporting — just not with that (wrong) mechanistic story attached.
- Real, undisclosed confounds: SignNet-PE on GraphGPS is not parameter-matched (576,138 vs
  ~503-507K for the other 3 PEs, +14%; also off-budget on VOC). On SAN's Peptides-func, the
  collapsed-vs-elevated PE split is *perfectly confounded* with parameter count (lappe/grpe
  ≈928K vs none/rwse/signnet ≈791-793K) — but this does NOT hold on Peptides-struct (lappe and
  none are both ≈928K yet ρ_rel is 0.020 vs 0.232), so it's a partial alternative explanation for
  func only, not a rebuttal of the struct finding. SAN's calibrated T=128 saturates (enumerates
  rather than samples) for graphs with ≤128 nodes — true for a large fraction of its probe pool.
- Table/criterion-b p-values from `scipy.stats.spearmanr` are the *asymptotic* approximation,
  which is invalid at n=4-5 (can report impossible values like p=0 or p=1.4e-24 that are below
  the exact-test floor of 2/n!). An exact permutation test is needed at this n. Same issue for
  Kendall's W's chi-square p-value (reviewer's manual exact enumeration at n=4 items, m=3 raters
  got p≈0.033 vs the asymptotic 0.060).
- VOC's pooled Spearman r=0.84 (criterion_b_pascalvoc-sp.csv "ALL (pooled)") is a two-cluster
  artifact: SAN's 5 points all sit at ρ_rel=0 (degenerate, per the paper's own Limitations) with
  uniformly worse F1 than every GraphGPS point — the "correlation" is just "SAN is worse at both
  metrics than GraphGPS," not a within-backbone relationship, and it's built from ρ_rel numbers
  the paper itself calls invalid.
- Table 5 (headline table)/criterion-b caption said "higher r means more local ⇒ better task"
  for the MAE-flip convention — this is backwards. Checked by hand: GraphGPS/Peptides-struct has
  r=-0.80 and its most-local PE (SignNet) has the best (lowest) MAE — i.e. more-local-⇒-better is
  the NEGATIVE-r case, not positive. The prose elsewhere ("strongly negative", §6's "pulling mass
  toward the near half") already reads this correctly; only the table caption states the decode
  rule inverted.
- Internal contradiction: §5.1 said "on Peptides-func nothing separates" (GraphGPS) while §5.3
  says paired bootstrap "resolves GraphGPS's Peptides-func null" — and Table 3 already stars that
  exact cell, resting on a marginal CI gap of ~0.000062 (≈0.5% of the interval's own width).
- Page limit: PDF was 6 pages with body content (Figure 2) spilling onto page 6, before the
  bibliography — violates "5 pages excluding references." Needs Figure 2 moved back onto page 5
  or content trimmed elsewhere to compensate for whatever text the fixes above add.

## Scripts (all in `scripts/`, all committed on `results-all`)
- `collect_cross_backbone_results.py` — pulls every branch's `results/*.json` via `git show`.
- `aggregate_results.py` — pre-existing repo script; computes `summary_table.csv`, plots raw/norm
  curves, criterion-b Spearman correlations. **Has the rho_seed_std bug noted above.**
- `make_paper_tables.py` — generates the `results_*.tex` fragments the paper `\input`s
  (`results_tables`=metric, `results_rho`=ρ_rel, `results_critb`=criterion b, `results_headline`
  =9-row W/sign summary). Contains `significant()`, the two-part star rule (has the bug above).
- `make_ranking_figures.py` — combined 3-panel rank heatmap (`figures/ranking_heatmap_combined.png`).
  Cell text: bold rank number (large) + ρ_rel as % (small, non-bold) below it — this exact sizing
  was tuned empirically (`get_window_extent`) because 5-PE-wide columns are only ~0.275in each;
  don't casually bump font sizes without re-measuring, text silently overlaps.
- `make_combined_curves_figure.py` — combined 3-panel normalized sensitivity curves, shared
  legend (color=backbone, linestyle=PE) instead of one per-panel legend (was illegible with up to
  14 series/panel).
- `paired_bootstrap.py` — per-graph-identity paired bootstrap of Δρ_rel(PE) vs that backbone's
  own No-PE, matched by graph id (tighter than the marginal CIs in Table 3 when between-graph
  variance dominates; does NOT resample seeds, so it can reproduce a seed-variance false-positive
  on cells where that dominates instead — flagged in the paper's Limitations).

## Status: the Sep 2026 review issues above are FIXED (as of this edit)
All items in the "Other verified-true facts" section above and the CRITICAL Graphormer fact have
been addressed in `paper_ACL_format.tex` and the scripts, except the one item that needs GPU
compute we don't have (re-probing Graphormer at 256 graphs — instead, every Graphormer-dependent
claim is now explicitly downgraded/disclosed rather than fixed at the source). Specifically:

- `scripts/aggregate_results.py`: added `rho_rel_seed_std` (was missing; `rho_seed_std` is the
  ABSOLUTE-rho seed spread, a different number) and `_exact_spearman_p` (exact permutation test,
  vectorized/fast, replacing the invalid asymptotic p at n=4-9; falls back to asymptotic only at
  n>9, flagged via a `p_exact` column). Also fixed a constant-input edge case (SAN's degenerate
  VOC arm) that used to silently produce p=0.0 instead of NaN.
- `scripts/make_paper_tables.py`: `significant()` now reads `rho_rel_seed_std` (fixes the star
  rule; SAN/Peptides-func/LapPE gained a star as a result — verified this is the *only* cell that
  changed). Criterion-b and headline-table captions rewritten with the correct sign convention
  (negative r = "more local ⇒ better task", was stated backwards) and exact-vs-asymptotic p
  labelling. Graphormer's rows in the headline table carry a ‡ marking them as built from the
  10-graph probe.
- `scripts/make_ranking_figures.py`: `kendalls_w` now returns an exact permutation p-value
  (enumerating all `(n!)^m` null rank-matrices) instead of the asymptotic chi-square, which had
  given p=0.060 (read as "marginal") for Peptides-struct when the true exact value is p=0.033
  (significant at α=0.05) — but that significance depends on including Graphormer as a rater;
  GraphGPS+SAN alone give W=0.800, p=0.208 (not significant). The paper states both numbers.
- `paper_ACL_format.tex`: abstract/intro/§5/§6/Limitations/conclusion all rewritten to (a)
  disclose Graphormer's 10-graph probe explicitly and downgrade every claim built on it, (b)
  correct §3.2's channel-width description (all backbones probe the FULL hidden width including
  PE channels, constant per backbone across its own 5 PEs — not "content-only, PE excluded" as
  previously stated, and not GraphGPS's 96/80/76/64 per-PE claim, which was never what was run:
  GraphGPS actually used 96 for all four of its PE arms, verified via `n_shared_feats` in every
  result file), (c) correct the SAN "mechanism" story (LapPE/RWSE/SignNet-PE/GRPE all share the
  identical `LPE="node"` dispatch slot in `san_backend.py`'s `PE_SPEC` — there is no
  "concatenated feature vs. native module" distinction in the code; the empirical split stands,
  the mechanistic explanation for it does not), (d) flag the VOC pooled r=0.84 as a two-cluster
  artifact built partly from SAN's invalid degenerate ρ_rel values, (e) disclose the SignNet-PE
  parameter-budget confound and SAN's param/collapse confound (holds on func, not struct) and
  T=128 saturation, (f) resolve the §5.1/§5.3 internal contradiction over GraphGPS's
  Peptides-func "null" (it was a marginal star, not a null — both sections now say so
  consistently), (g) fix Table 1's stale "SAN GRPE: partial seeds" (now full 3×3).
- Page budget: after all the above additions the body grew to ~8 pages before compression;
  retightened margins/spacing/figure heights and cut duplicated explanations (state each
  correction fully ONCE, point to it briefly elsewhere) back down to body-ends-on-page-6,
  references-starting-same-page-6 — matches what was previously agreed acceptable ("references
  can be on page 6 if needed"). If asked to shrink further, the two full-width figures
  (`figures/ranking_heatmap_combined.png`, `figures/sensitivity_norm_combined.png`) are the
  biggest remaining single-item costs; their font sizes are tuned to exact per-cell pixel widths
  (see the comments in `make_ranking_figures.py`) so re-shrink figsize height only, not width,
  and re-view the PNG directly before trusting it isn't overlapping again.
- NOT fixed (needs compute, not text): Graphormer's probe is still 10 graphs, not re-run at 256.
  This is now the single highest-value next action for this project — it would let the headline
  claim either be confirmed with real backbone-3 data or retired, instead of staying unresolved.
