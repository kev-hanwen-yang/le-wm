# Observation-domain diagnostic protocol v2 — before new outcomes

This is a bounded readout-calibration and **natural, unpatched** predictor
diagnostic, authorized by the latest task. It is not the formal Step 3 dataset,
H1, H2, or an activated gate. Exact historical provenance stays UNRESOLVED.
The previously exposed 16+5 source families are development evidence only.

## Unchanged prospective domain

Use the existing `SceneSpec` goal `(70,420,350,120,0,0,0)` and
`SimulatorSettings(224,True,1e-10)`. The visible marker is `(350,120,0)`, matching
that physical goal. Do not substitute the previous central-marker comparison
candidate `(256,256,pi/4)`. No goal image/encoding, renderer, controller, physics,
weights, preprocessing, action statistics, or official metrics are modified.
Images are direct RGB uint8 render output; no production codec. Operational
assessment of this explicitly labeled domain does not recover historical goals.

## Sampling and independence

Generate 32 predetermined families, 16 development and 16 independent validation,
with separate NumPy seeds 2026101401 and 2026101402. Each has its own block
position uniformly in x=[220,290], y=[205,290], angle uniformly in [0,2pi).
Within each partition alternate eight free-space and eight near-contact families.
Rotate local agent offsets by the block angle: free `(-150,-60)`, near
`(-107,15)`; positive doses move along local +x. Verify actual shape clearances,
reset readbacks, zero dynamics, goals and exactly two reset substeps; do not
assume this analytic geometry is sufficient. Invalid whole families are logged,
not replaced; missing strata/counts make validation inconclusive.

All 0/10/20/30-world-px doses and duplicate zero controls stay in one family.
Reject shared/near-duplicate (within 1 world px in agent and block, .01 rad angle)
initial poses across partitions and against historical initial poses and Step 2
fixtures. Reserve validation poses before running development; do not inspect
validation pixels/latents/readouts until development calibration is saved.
These diagnostic families are permanently exposed afterward, not future H2 test.

Every scene starts fresh and uses ten zero raw actions, then amplitude .10 along
local +x for fifteen raw steps, then ten zero actions. This fixed action is not
selected for a favorable model/contact effect. Keep all valid outcomes, including
zero effects and wall contact. Independently repeat the zero scene's full replay;
compare all raw/substep states, contacts, RNG and pixels. Retain all raw readbacks
and RGB; encode retained h=0,5,10,15,20,25,30,35 with frozen preprocessing.

## Measurements and development calibration

Use original frozen root MLPs, correctly unnormalized; no correction/refit is
applied. Per family, report absolute initial agent/block Euclidean errors and
wrapped angle error; differential error against true dose, block/angle leakage,
direction cosine (undefined for zero decoded change), magnitude error, latent
L2/norms and distances to the SAME frozen development-source reference bank.
Use all 16 independent physical-dose references per partition per dose, not the
previous three natural-source 10-px references. No nearest-neighbor statistic
alone certifies a manifold. Report original natural-source same-state/dose
references and spatial residuals separately, never as exact dynamic matching.

Same-state production nuisance is duplicate fresh replay + repeated encoding.
Stress-test two *predefined offline* same-origin codecs: libx264 CRF23 at yuv420p
and yuv444p, plus FFV1 bgr0 lossless. They are diagnostic perturbations, not a
production pipeline, selected repair, or independent samples. Compare their
latent/readout drift to every dose; no arbitrary pixel threshold.

Fixed candidate physical caps: 5 world px each for absolute position, differential
agent error and block leakage; .10 rad for absolute angle and angle leakage.
Five px is half the smallest agent dose. It is NOT adequate evidence for resolving
1-px block effects. Report that stricter limitation separately and require
effect-scale calibration before future downstream-block claims.

For each dose and non-target metric, set proposed calibrated margin to the
development one-sided 95% upper family-RMS bootstrap bound. Never enlarge the
5/.10 caps. Independent validation must satisfy BOTH its physical cap and that
saved calibration margin; if development already exceeds a cap it is a measured
measurement limitation, not a new acceptable margin. Cap-only and calibrated
checks are separately reported. No favorable codec/readout/dose selection.
Direction/magnitude are descriptive checks in addition to vector error, not
alternate ways to pass a failed vector-error criterion.

Average repeats/frames within family before RMS, resample whole families 10,000
times seed 2026101403; report two-sided 95% intervals and one-sided 95% bounds.
PASS requires upper bound <= margin, FAIL lower bound > margin, otherwise
INCONCLUSIVE. Also report point-outside-margin; point failure alone is not
population-certainty. All-required checks form a conservative intersection;
secondary strata/horizons/codecs are descriptive, no significance hunting.
Sixteen deliberately sampled families do not establish rare-tail/population
equivalence or deployment generalization.

## Natural predictor baseline (not H2)

Reuse seven correctly standardized five-action tokens over raw [0,35). Open loop
uses natural h=0,5,10, then predicted feedback. Teacher forcing uses the sliding
three natural retained frames and corresponding sliding three action tokens.
Both predict h=15,20,25,30,35; first predictions must agree. Report absolute
physical readout errors, predicted-minus-natural-encoded readout differences,
latent forecast errors, teacher-forced transition magnitude versus true encoded
transition, and natural dose-versus-zero physical-effect errors (including
wrapped-angle effects/residuals). Save real/predicted vectors, not just norms.
No patch, Ridge or causal claim. Probe errors on predicted OOD latents cannot be
separated exactly from model errors by these instruments.

Source comparison: use first 36 frames/actions of the previously exposed matched
families if available. Recompute with the correct action standardization; old
raw-action rollout reports are not like-for-like. Historical labels are source
physical observations, NOT reconstructed complete snapshots. Different states,
goals and action populations mean source/prospective aggregate differences cannot
uniquely attribute degradation to domain. Report attribution INCONCLUSIVE unless
additional matched evidence identifies it; no covert dynamic restore.

## Freeze and decisions

Development and validation are separate CLI executions. Development persists
family reservation, source/reference/readout/model identities, measurements,
calibration and protocol/code/config hashes. Validation rejects changes to those
choices, freezes its parent hashes before encoding, and runs once. New criteria
after validation need new families. Existing 19 protected natural families stay
unqueried. No automatic collection unlock, even if checks pass.

A: existing mechanics + new duplicate replay. B: explicit reproducible unchanged
production contract (operational generation only, historical provenance separate).
C: physical caps AND calibrated independent measurement checks. D: production
stability, stress sensitivity and natural predictor ceiling measured; unresolved
domain attribution/OOD instrument error is not a PASS or evidence of no physics.
E: independence, complete coverage, prespecified criteria and relevant passes.
Report PASS/FAIL/INCONCLUSIVE separately, plus development-only and confirmatory
readiness booleans with scope. No perfect predictor requirement; no fabricated
measurement PASS. Obtain human approval before any refit or formal Step 3.

Methodological basis: [Lakens (2017)](https://journals.sagepub.com/doi/10.1177/1948550617697177)
for meaningful prespecified bounds; [Saravanan et al. (2020)](https://arxiv.org/abs/2007.07797)
for respecting hierarchical dependence. These family-RMS bounds are percentile
bootstrap diagnostic estimates, not signed-mean TOST or certified error coverage.
