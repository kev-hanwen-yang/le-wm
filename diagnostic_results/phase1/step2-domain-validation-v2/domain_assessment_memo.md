# Step 2 observation-domain assessment v2

## Material Passport

- Origin: academic-research-suite / experiment-agent, validation.
- Date: 2026-10-14. Version: natural-domain-diagnostic-v2.
- Verification: VERIFIED for executed numerical reproducibility; ANALYZED for interpretation.
- Scope: natural observations and natural predictions only. Not H1/H2 or formal Step 3.

## Outcome

**The unchanged fresh-start rendering domain is reproducible, but the original
MLP readouts are not adequate instruments for this Phase I experiment.** This
is a measured resolution problem, not an indefinite historical-records blocker.
Neither compatibility-qualified Step 3 development collection nor a confirmatory
observation-contract freeze is supported with the current observation/readout
combination. Exact historical provenance remains **UNRESOLVED**.

## Design and independence

`observation_domain_protocol_v2.md` and code/config were written before these
outcomes. Development ran separately on 16 new families, then saved calibrated
margins. Validation used 16 reserved independent families, eight free-space and
eight near-contact, with no subsequent criterion/codec/readout selection.
All doses, histories and nuisance variants stay within their family. The old
16+5 source families are now exposed development evidence only; none is virgin
validation. The 19 protected natural-source families were not queried.

All 32 fresh families and all four doses were physically valid; none was
replaced. Actual shape clearances were positive (near-contact dose30 starts
at a 2-world-px gap). Labels were independently checked against simulator bodies
and returned observations, including zero velocities/forces and reset side
effects. Duplicate zero scenes reproduce every raw/substep state, contact, RNG
and pixel exactly. This establishes current-backend fresh replay, **not** a full
historical snapshot or arbitrary moving-state restore.

The prospective diagnostic domain is the **unchanged Step 2** `SceneSpec` goal
`(70,420,350,120,0,0,0)`, visible matching marker `(350,120,0)`, direct RGB uint8
224 output and original frozen preprocessing. The previous matched-source
candidate used a central marker `(256,256,pi/4)`; these are different conditions.
We did not move goals or select a display/codec to improve results. Source/fresh
comparisons cannot identify a domain-only effect when goals/poses/actions differ.

## 1. Source-versus-reconstruction diagnosis

For the previous 21 render-relevant matched initial observations, all true
labels, source/render decoded vectors, signed differences, spatial residuals and
encoder distances are in development `signed_source_diagnosis.json`. Historical
block velocities/contact caches and pre-quantization positions remain unknown;
public-state matching is **not complete dynamic-state equivalence**.

Across all 21 exposed cases, agent cross-domain drift RMS is **11.660 px**.
The bias vector estimated only on the original 16 development cases is
`(+0.504,-1.265)` px. Removing that constant *for diagnosis only* leaves
**11.690 px** residual RMS; no correction was applied to any instrument.
Signed x drift versus true agent x has descriptive correlation **−0.825**;
y drift versus true agent y **−0.775**. These are associations in selected
development cases, not a fitted corrective model or causal identification.

This favors **pose-dependent decoded distortion**, not a single translation
bias. Fixed color-mask agent-centroid shifts have median **0.198**, RMS **0.972**
and maximum **4.241 world px**. They are much smaller than many decoded drifts,
but occlusion/chroma-dependent segmentation and unknown subpixel positions mean
we cannot exclude every state/image mismatch. Prior spatial evidence places
most foreground residual at boundaries with systematic interior color shifts.
Offline codec perturbations at exactly unchanged physical origin establish
appearance sensitivity. Rasterization, decoder/chroma and subpixel causes remain
plausible; none uniquely reproduces the original MP4 pipeline.

The earlier five-case 17.658-px drift remains unchanged; the new 11.660 value
uses **all 21 exposed cases**, not a repaired or improved validation result.

## 2. Within-domain physical readout resolution

Family-balanced RMS with two-sided 95% percentile family-bootstrap intervals,
10,000 draws, seed2026101403. Positions use Euclidean world px; angle uses
wrapped radians. No pooling of units, epsilon stabilization or clipping.

| Initial dose0 absolute error | RMS | 95% interval | Candidate cap |
|---|---:|---:|---:|
| Agent position, px | 57.251 | 40.067–73.616 | 5 |
| Block position, px | 14.909 | 11.036–18.641 | 5 |
| Block angle, rad | 0.501 | 0.273–0.707 | 0.10 |

| Dose, world px | Agent differential RMS (95% interval), px | Block false-motion RMS, px | Angle false-change RMS, rad |
|---|---:|---:|---:|
| 10 | 8.981 (4.957–12.522) | 1.431 | 0.0420 |
| 20 | 10.253 (6.715–13.774) | 2.194 | 0.0617 |
| 30 | 15.957 (11.129–20.652) | 2.490 | 0.1118 |

The 10-px agent **one-sided** 95% lower bound is 5.305 px, exceeding the
5-px cap; its two-sided lower endpoint is slightly below five. This distinction
and small N should not be concealed. Development agent differential upper
calibration margins were **13.747/14.734/15.240 px**, already too large to be
acceptable. Validation can meet an inflated empirical noise envelope while
still failing the unchanged physical cap. The harness checks both separately.

Direction cosines have medians **0.967/0.982/0.975** with positive directions in
**14/15/15 of 16** families, but decoded displacement magnitudes have medians
**10.664/15.618/21.209 px**. Magnitude-error RMS is 4.275/7.142/12.335 px.
Good median direction does not rescue large vector errors/outliers. Free-space
agent differential RMS is 11.418/11.930/18.203; near-contact is
5.562/8.241/13.337 px, so failure is not confined to collisions.

Block leakage passes the broad 5-px cap, but fails some tighter development-
calibrated checks; 1.431–2.490 px spurious movement cannot certify 1-px physical
effects. Angle leakage at dose30 is INCONCLUSIVE against .10 rad, not PASS.
Absolute angle and position errors are clear measurement failures.

## 3. Encoder stability and nuisance

Every fresh same-state replay, repeated encoding and repeated prediction has
maximum difference **zero** on this audited CPU backend. Exact RGB FFV1 likewise
has zero latent/readout change. Sixteen independent validation physical-motion
references exist **per dose**, in addition to sixteen development references
per dose; no longer just three qualifying 10-px source references.

Latent dose RMS (95% family intervals):
10px **0.791 (0.684–0.888)**; 20px **1.492 (1.245–1.714)**;
30px **2.566 (2.030–3.012)**. Medians are 0.786/1.406/2.335.
Thus controlled physical changes are visible and stable in the representation;
this is **not** proof of accurate decoding or selective latent coordinates.

Predefined same-origin CRF23 stress controls, never production:

| Offline codec | Latent RMS | Agent drift RMS, px | Block drift RMS, px | Angle drift RMS, rad |
|---|---:|---:|---:|---:|
| 4:2:0 | 0.819 | 16.431 | 1.901 | 0.0563 |
| 4:4:4 | 1.549 | 23.688 | 2.925 | 0.0660 |

Compression nuisance can rival 10/20-px motion and severely alter the old
instrument. This does not invalidate deterministic **uncompressed** generation
by itself; it does prohibit treating codec conditions as interchangeable or
choosing a codec because its probe output improves.

Median latent norm is **13.215** on prospective initial scenes versus **13.783**
in the fixed 592-row source-development reference bank. Prospective norm range
10.504–14.877 lies within source-bank range10.499–15.827; nearest-bank median
distance is **12.264**. Norm overlap and this sparse nearest-reference statistic
do not establish observation compatibility or manifold membership. The bank's
rows/families and identity are retained by the immutable parent reference.

## 4. Natural predictor baseline, not H2

All scenes use genuine zero-raw-action histories h=0/5/10, then the same fixed
relative command within family. Seven training-standardized five-action tokens
cover [0,35). Teacher forcing slides natural frames; open loop feeds predictions
back. First predictions agree exactly. Goals and goal encoding are unchanged.

Family-RMS absolute readout errors at h=15/20/25/30/35:

| Instrument/input | Agent px | Block px | Wrapped angle rad |
|---|---|---|---|
| Natural future encoding | 46.12/41.05/35.41/35.36/35.36 | 15.57/15.51/14.85/14.73/14.73 | .413/.344/.357/.349/.349 |
| Teacher forced prediction | 47.39/42.15/37.66/35.09/35.19 | 18.67/19.72/19.66/15.73/15.87 | .554/.510/.487/.480/.470 |
| Open-loop prediction | 47.39/43.44/40.07/40.44/41.35 | 18.67/22.86/29.78/31.14/32.70 | .554/.620/.675/.701/.717 |

Open-loop **latent** forecast error grows from2.115 at h15 to5.937 at h35;
teacher-forced h35 error is approximately1.26 (full precise curves saved).
At h35, decoded open-loop minus natural-future differences have RMS
21.879px agent,30.095px block,.508rad angle; teacher-forced differences are
5.182px,5.371px,.156rad. Thus feedback adds a measurable forecast discrepancy
even relative to the same imperfect instrument on natural future latents.
This decomposition is **not** an exact physical separation of predictor and
probe error on predicted OOD latents. The instrument remains a major limitation.

Natural dose-versus-zero physical-effect errors are reported, with true and
predicted vectors saved—not patched H2. Real block-effect RMS across families
and three doses is8.570/17.318/16.924/16.882/16.882px. Encoded-natural readout
effect error is3.800/5.306/5.566/5.391/5.391px; open-loop effect error is
6.689/12.137/10.655/10.322/10.223px. Teacher forced h35 error is6.146px.
These are sufficiently excited diagnostic trajectories, not proof of good
contact prediction. Agent/angle effect curves and all confidence intervals are
in `domain_manifest.json` and `supplemental_summary.json`.

Correctly standardized source-domain windows were recomputed for all21 exposed
families. Source open-loop h35 errors are8.172px agent,8.796px block,.211rad
angle; their natural-encoded errors are4.324px,2.677px,.187rad. Fresh-domain
errors are substantially larger, **but domain-only attribution is INCONCLUSIVE**:
source and fresh samples differ in state/goal/action populations. Old raw-action
rollout reports were not reused as equivalent baselines. Neither a perfect
predictor requirement nor the prior DICTA cohort's under-updating ratio was
imposed on this different diagnostic regime.

## Gates and readiness

| Gate | Decision | Meaning / remaining requirement |
|---|---|---|
| A Physical simulation | PASS | Old mechanics/source/binary contracts unchanged; all new duplicate raw/substep replays exact. Fresh quiescent scope only. |
| B Observation generation | INCONCLUSIVE | Explicit controlled generation/reproducibility subchecks PASS. Empirical scientific compatibility remains instrument-limited; do not freeze solely from repeatability. Historical provenance separately UNRESOLVED. |
| C Measurement | FAIL | Absolute and differential agent errors, absolute block/angle errors exceed fixed physical budgets; new valid instrument requires independent validation. |
| D Representation/prediction | INCONCLUSIVE | Raw stability and natural forecast ceiling measured; nuisance sensitivity, predicted-latent readout validity and domain-only degradation remain unresolved. No perfect-model requirement. |
| E Independent validation/freeze | FAIL | New families and criteria freeze were independent, but relevant scientific criteria did not pass. These families are now exposed, not reusable final validation for a revised readout. |

**Step 3 development-only dataset readiness: FALSE** for compatibility-qualified
data with the present frozen readouts. Bounded simulator geometry/measurement
diagnostics are defensible and were performed under this separate task; they must
not be relabeled a passing Step 3 pilot or supply physical causal claims.
**Step 3 confirmatory observation-contract readiness: FALSE.** No unlock occurred.

Next proposal is a separately versioned natural-state readout refit/calibration,
with LeWM and all original probes/results preserved, fresh disjoint families and
effect-scale measurement tests. See `readout_refit_proposal_v1.md`.
It requires human approval **before any fitting**. If new natural readouts pass
but forecast errors persist, retain them as intrinsic/OOD baseline limitations
and assess a diagnostic negative H2 only after later selectivity gates—not H3.
Exact historical MP4 recovery is useful, not a mandatory path to operational
compatibility. Changing a prospective goal/render contract also needs explicit
approval and fresh validation; it was not done here.

## Verification and limitations

97 direct tests across 13 scripts pass (85 retained +12 new). Independent-process
validation replay agrees exactly on 2,614 HDF5 datasets and 163 attributes;
HDF5 file SHA itself is identical, as are family metrics and frozen validation
records. Replay is a reproducibility check, **not16 more independent families**.
All original checkpoint/readout hashes and Step1/2 source contracts checked.
Large artifacts stay in `.stable-wm/phase1/<run_id>`; compact manifests, CSVs,
four diagnostic figures and this memo stay in `diagnostic_results/phase1/<run_id>`.
The four figures are diagnostics, not the four completed PhaseI H2 deliverables.

No weights, physics, controller, preprocessing, task goals, action statistics,
costs, official metrics, caches, latent patches or Ridge fits changed. No formal
Step3, H1, H2, H3, commit or push occurred. Existing native SDL/deprecation
warnings persist; logs are retained. Generalization is limited by16 families
per partition, prescribed geometry ranges, fixed goal and one action template.
No population rare-tail, deployment, or causal-coordinate claim is made.

Fallacy scan 11/11: strata reported (Simpson); family scope, not population/individual
causality (ecological); predetermined geometry/no outcome replacements
(Berkson/collider/survivorship); no prevalence or failure-rate claim (base rate);
no best-case/codec retest improvement (regression to mean/look-elsewhere);
protocol/calibration frozen and supplemental summaries explicitly descriptive
(forking paths); readout/latent association not physical causality (correlation);
no reverse-direction inference. Small N, selected historical matching,
instrument OOD and unmatched source/fresh dynamics remain CAUTION.

Methodological grounding: [Lakens (2017)](https://journals.sagepub.com/doi/10.1177/1948550617697177)
supports meaningful prespecified bounds, not nonsignificance-as-equivalence;
[Saravanan et al. (2020)](https://arxiv.org/abs/2007.07797) supports respecting
nested dependence. Our intervals are family percentile-bootstrap estimates,
not signed-mean TOST or guaranteed small-sample coverage.
