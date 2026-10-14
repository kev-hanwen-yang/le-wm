# Readout repair feasibility protocol — R0 and R1 only

Written before the new audit/fitting outcomes, 2026-10-15 (Australia/Melbourne).
The latest approval authorizes R0 followed by bounded R1 if the pipeline is
correct and the old instruments remain inadequate. R2, formal Step 3, patches,
H2/H3, commits/pushes and automatic instrument promotion remain unauthorized.
Exact historical provenance stays UNRESOLVED. Preserve every original artifact.

## R0: implementation and measurement sanity

Audit all 32 exposed v2 diagnostic families at their four initial agent doses,
plus the 21 already exposed historical source initial images. Recreate fresh
initial scenes with the unchanged Step 2 goal/marker `(350,120,0)`; verify full
quiescent body readbacks, public state, goal and pixel correspondence against
saved artifacts. Source comparisons are render-relevant/stored-label matching,
not complete historical dynamic-state restoration or MP4 reproduction.

Block target means **body.position / local origin**, not its world center of
gravity or silhouette centroid. Independently compare analytic rotations and
translations of local T vertices against engine local_to_world results. Compare
non-learned circle/T masks on the actual 512 canvas with analytic geometry;
boundary differences are expected from integer rounding, polygon rasterization
and the renderer's 2-world-px polygon stroke. Report interior/exterior mismatches
outside derived boundary bands (circle2px, polygon4px), not a pixel-RMSE scientific
compatibility threshold. Check canvas RGB orientation and exact cv2 resize to224;
do not substitute a new renderer. Record body-origin, mass-center and area-center
definitions explicitly, with independent geometry visualizations.

Check target builders, 512-world units versus224-image grid, RGB/HWC→B1CHW and
training/evaluation/manual preprocessing parity. Check actual original MLP shapes,
ReLU/dropout/eval mode, head dimensions, stored target mean/std, strict state_dict
loading and independent layer-by-layer output/unnormalization. Re-encode sampled
saved source/prospective images under the original cache tolerance. Report source
and fresh-domain absolute/bias/differential errors, including pose/orientation/
stratum concentration. Never correct a probe using pilot-validation labels.

If an actual pipeline/target bug is found: save failed audit, minimally correct
it without changing scientific contracts, and rerun affected prior diagnostics
before deciding on R1. No speculative correction or goal/style change for PASS.

## R1: natural-state feasibility population

Use the exact unchanged Step 2 prospective domain and frozen LeWM encoder.
No predicted or patched latents enter training/evaluation. All observations are
fresh quiescent renders with independently checked body/public labels. Histories,
future actions and predictor experiments are not required for this readout fit.

Training: reuse all32 exposed v2 families as development, add16 predetermined
fresh development families. Pilot-validation:16 new independent families.
New positions/orientations use the previous outcome-independent geometry ranges
x=[220,290], y=[205,290], theta=[0,2pi), alternating free/near-contact local
offsets(-150,-60)/(-107,15). New seeds2026101501/2026101502; rename family IDs
to r1_train/r1_pilot to distinguish this experiment. All related variants remain
in one highest-level family. Exclude shared/near-duplicate variant poses across
families using the previous1px agent+block/.01rad rule, and against185 historical
initial poses and8 mechanics fixtures. No outcome-based family replacement.

Each family has base, agent+10/+20/+30 along local+x, block±1/±2/±5/±10 along
the local perpendicular direction, angle±.01/±.025/±.05/±.10rad (wrapped initial
angles in[0,2pi)), and an independently reconstructed zero-repeat. Agent/block/
angle changes are independent variants, not compound triplets. All other state
and goals match. Invalid variants invalidate the whole family and are logged,
not replaced. Later R2/final families are not generated or inspected now.

Exclude the duplicate zero control from fitting. Each family's20 distinct
natural scenes get equal weight within an equally weighted family. These scenes
are repeated measurements, not20 independent samples. For nested training-size
curves use16/32/48 families; each size has equal free/near counts, ordered by
fixed seeded permutations within strata (seed2026101503). Pilot-validation
families are used for bounded model/epoch selection and become exposed
development evidence, **not untouched confirmatory testing**.

## Predeclared candidate set and training

- Original frozen agent/block/scalar-angle MLPs remain baselines.
- Ridge in the original192-D latent coordinates, unpenalized intercept and
  equally weighted mean squared loss. Four lambdas only:1e-4,1e-2,1,100.
  Fit positions, scalar angle and separately sin/cos angle targets. Select per
  head by pilot-validation natural absolute error; numerical ties within1e-12
  choose stronger regularization. Weighted sklearn equivalence is unit-tested.
- Existing MLP:192→256→ReLU→Dropout(.1)→output, one hidden layer only. Positions,
  original scalar angle and separately versioned sin/cos angle head. AdamW,
  lr=.001, weight_decay=.0001, full-batch weighted MSE, maximum600epochs,
  patience100, evaluate every10epochs. Seeds42 and43 both reported, never pick
  the more favorable seed. Restore the best checkpoint by pilot-validation
  absolute error; curves store training and validation errors/losses.
- Target mean/std are fit from training families only (explicit weighted
  population variance, reject constant targets rather than epsilon/clipping).
  No encoder/latent whitening or model-weight updates. Scalar angles use raw
  wrapped initial labels; sin/cos heads train sine and cosine and decode atan2.
  Predicted sin/cos norm<=1e-8 is declared degenerate and excluded/count-logged,
  not repaired or treated as zero angular error. Original scalar instrument
  and all angle candidates remain in the report.

Saved candidate artifacts are R1 instruments only, separately versioned under
`.stable-wm/phase1/<run_id>/readouts/`; no official checkpoint replacement or P.
Record every candidate/lambda/epoch/seed, selected metadata, training lineage,
normalization and exact commands. No post-outcome expansion of the search.

## Metrics, uncertainty and scale-dependent resolution

Report coordinate-x/y RMSE, coordinate-pooled RMSE, Euclidean position RMSE and
coordinate R² separately; scalar wrapped-angle error in radians. Absolute
metrics average squared errors within family before equally weighting families.
Per paired dose, compare predicted changes with actual body-coordinate changes;
wrap angle effects and then their residual. Report signed bias, vector error,
direction cosine, magnitude error, non-target preservation and duplicate-zero
leakage. Nonzero-dose zero decoded motion makes direction undefined, never PASS.

10,000 whole-family paired bootstrap draws, seed2026101504; two-sided95% CIs,
one-sided95% bounds for proposed practical/resolution comparisons. These CIs are
conditional on pilot-selected instruments and exploratory finite-family evidence,
not confidence accounting for model selection or a deployment population.

Candidate broad caps remain5worldpx position/.10rad angle. Proposed dose-resolving
margin is min(cap,half true dose), checked with an upper bound; FAIL if lower
bound exceeds the margin, otherwise INCONCLUSIVE. This half-effect allocation
leaves room to distinguish the physical signal from measurement error. It is a
predeclared feasibility proposal, not activated confirmatory acceptance criteria.
For block doses1/2/5/10, report respective margins.5/1/2.5/5px and the smallest
tested scale satisfying its error bound, positive-direction evidence and matched
non-target control. Angle margins similarly use half the tested dose and .10cap.

The1px/.5px block objective is **high-resolution only**, not a universal Phase I
prerequisite. Do not change the existing Step3>=1px feasibility criterion.
Larger measurable effects can motivate a separately approved scale-specific H2
protocol, but no final outcomes or criteria are available/modified here.

H1 is separate: inventory existing source affine probes (AdamW, not Ridge) and
report R1 Ridge validation accuracy/rank/row-space stability across training
counts. A fitted R1 Ridge is not the official H1 projector; create no P, patch
or selectivity result. Nonlinear instrument success cannot certify linear H1
adequacy, source-subspace transfer, predictor dynamics or predicted/patched OOD
readout validity. Further honest H1 and independent selectivity remain required.

## Outputs and stop

R0 audit manifest/tables/geometry evidence and corrections if any; R1 family
assignments/attempts/natural arrays, every versioned candidate, learning curves,
absolute/dose/control/bias tables, family CIs, reproducibility manifests and memo.
Compact reports in `diagnostic_results/phase1/<run_id>`, large arrays/models in
ignored `.stable-wm/phase1/<run_id>`. Existing97tests remain passing; add direct
tests for geometry/labels/preprocessing, group isolation, weighted fit algebra,
angle conventions, metric definitions, scale-dependent criteria and lineage.

Answer all eight user go/no-go questions using measured facts, distinguish R2
recommendation from authorization, update `readout_refit_proposal_v2.md`, stop.
The academic-research workflow informs family-level uncertainty and the separation
of feasibility from confirmatory/causal validity. Primary implementation references:
[Pymunk7.2](https://www.pymunk.org/en/7.2.0/pymunk.html),
[Ridge1.7](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.Ridge.html).
