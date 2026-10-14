# Readout repair proposal v2 — measured R0/R1, R2 approval required

2026-10-15, Australia/Melbourne. This supersedes v1's proposed allocation,
not any historical report or the authoritative scientific specification.
**R0/R1 are complete. R2 below is a proposal, NOT authorized or implemented.**

## Evidence and decision

R0 found no original input/label/checkpoint/preprocessing bug. At 21 matched
stored poses, legacy agent RMSE is 5.66 px on source RGB versus 55.56 px on
prospective RGB. Historical goal/style/subpixel/dynamics/codec provenance is
still **UNRESOLVED**; this is not historical-pipeline reproduction.

R1 used 48 training and 16 independent pilot-validation families, all natural
unpatched static inputs, the unchanged operational Step 2 domain and frozen
LeWM. At 48 families, Ridge agent/block Euclidean RMSE is 26.48/7.83 px;
MLP seeds 42/43 give 25.55/26.70 agent and 8.34/8.19 block px. These exceed
the candidate 5-px cap. Sin/cos MLP absolute angle errors .0650/.0754 rad
meet the candidate .10-rad cap on selected pilot data, but small-dose angle
resolution is not qualified. No tested block scale 1/2/5/10 px passes the
combined resolution/preservation checks; agent-10 is inconclusive for MLP,
20/30 fail. Increasing families helps, but does not establish saturation or
guarantee success. Neither positions nor the H1 subspace can be promoted.

Recommendation: **conditional GO for further bounded R2 development**, if
approved; **NO-GO for immediate final-instrument freeze or confirmatory H2**.
Do not execute v1's 80/24/60 recipe unchanged on the assumption that refitting
will solve the measurement problem. R1 supports investigating coverage and
family scaling while keeping a meaningful stop before fresh final evidence.

Measured evidence:
`diagnostic_results/phase1/readout-r0-sanity-verified-v1/r0_audit_memo.md` and
`diagnostic_results/phase1/readout-r1-feasibility-v1/readout_repair_memo.md`.
All original reports/weights and the unsuccessful first R0 attempt are retained.

## Proposed R2 design, to freeze before execution

1. **Preserve the domain and model.** Same Step 2 safe goal/marker, RGB renderer,
   frozen checkpoint/preprocessing, physics and body-origin/world-px labels.
   No codec/goal selection, encoder fine-tuning or predictor rescue. New
   readouts get a new schema/run path, never overwrite legacy or R1 files.
2. **Develop before confirming.** Suggested bounded resource targets: nested
   96/192 training families, 32 new model-selection families, 32 different
   calibration families, and 60 later fresh final measurement families.
   The existing 64 R1 families are exposed and may enter development only.
   Counts are planning targets, not a power guarantee; prior family-error
   distributions should assess expected interval precision before approval.
   Do not generate/query final-family outputs unless development/calibration
   meets the frozen qualification rules. No adaptive final-test expansion.
3. **Improve identifiable coverage, not just frames.** Predeclare geometry-only
   ranges and clearance rules covering more independently varied agent and
   block positions, relative offsets and orientations. Retain balanced
   free-space/near-contact strata. R1's two fixed local offsets constrain the
   population and associate agent pose with block pose; broaden valid offsets
   before outcomes, rather than overgeneralizing from that narrow support.
   Include both-axis directions and signed doses, grouped under the highest
   physical-family ID. Reject overlap with exposed/final families before any
   outcome. Invalid families are recorded, never replaced for poor accuracy.
4. **Keep search bounded.** Ridge in raw 192-D coordinates with unpenalized
   bias and family-normalized mean loss; existing MLP256 only; separately
   versioned sin/cos angle head. Predeclare a small penalty/optimizer schedule
   and two training seeds; report all candidates, scalar baseline and both
   seeds. Training-only target statistics, no hidden whitening. Selection
   families choose candidate/epoch; separate calibration families assess its
   errors. If qualification fails, stop rather than extending search until PASS.
5. **Natural calibration only.** No patched/predicted latent, H2 result or final
   family enters fitting/selection. Static R1 evidence does not validate
   moving states. Any approved extension to natural moving observations must
   separately specify coherent trajectories, audited replay, exact full labels
   and family grouping; it cannot be introduced implicitly into a static-fit
   claim. No new action/controller contract or formal Step 3 collection follows.

The suggested family counts and coverage extension require explicit approval
and a versioned executable protocol; no new family has been generated for them.

## Measurement-scale requirements: proposal, not gate activation

Keep candidate broad practical caps **5 world px** for position and **0.10 rad**
for wrapped angle. Report coordinate RMSE, Euclidean RMSE, signed bias,
direction and displacement magnitude separately. Never combine px and rad.
Natural paired errors must use the same dose and geometry support as the
subsequent claim, with equal-family weighting and family-level intervals.

Proposed resolution allocation for true effect magnitude `d`:

- Position: one-sided 95% upper differential-error bound <= `min(5 px,d/2)`.
- Angle: corresponding bound <= `min(.10 rad,d/2)`.
- Require positive-direction evidence, exact zero-repeat checks, and matched
  non-target errors within the fixed practical caps. Record undefined directions
  or angle decodes; do not epsilon-repair or silently remove their failures.
- Absolute state recovery must separately meet its practical cap. Differential
  cancellation does not certify absolute recovery or out-of-distribution use.

The half-effect allocation reserves a measurement-error budget relative to a
physical signal. It is not a theorem that causal fidelity follows from this
budget. Any activated H2 acceptance rule still needs independent calibration,
pretest freeze, measurement sensitivity analysis and human approval.
Bounds spanning a margin are **INCONCLUSIVE**; lower bounds above it are FAIL.
Final family evaluation is once under the frozen protocol, not for retuning.
Predeclare either a primary scale/head bundle with hierarchical gates or
simultaneous bounds across claimed scales; R1's many selected-pilot intervals
are not a family-wise confirmatory success criterion.

The **1-px effect / <=0.5-px differential budget is an optional high-resolution
objective**, reported separately. It is not a universal Phase I prerequisite.
The original **Step 3 >=1-px physical-effect feasibility threshold is unchanged**;
changing it would require another explicit protocol revision. Larger effects
may be useful only after their natural-state resolution is demonstrated.
Consider adding 20/30-px block calibration doses, if geometrically valid and
approved, alongside the unchanged 1/2/5/10 grid; R1 gives no evidence that
these untested larger scales will pass or that their non-target leakage is small.

Before future H2, freeze which scales/supports are instrument-qualified and
which conclusions are measurement-limited. Preserve all valid final families
and small/null effects in the analysis; do not select favorable large-effect
test cases, discard cases below instrument resolution or enlarge error caps
after seeing final outcomes. A larger-effect qualification is not a claim to
accurately measure 1-px effects. No H2 scale amendment is activated here.

## H1 suitability remains separate

The existing source affine head is AdamW, not Ridge; no official frozen H1/P
has been implemented. R1 Ridge rank two/high R² does not certify a stable or
selective agent-position subspace. Position errors are large and fitted
row-spaces shift substantially with family count/penalty selection.

Keep the planned H1 family split, raw-coordinate Ridge fit, row-space
construction and natural prospective-domain transfer assessment explicit and
separate from nonlinear readout calibration. Report learning/penalty dependence
and independent non-target selectivity before interpreting a P. Do not install
R1's prospective Ridge as the official projector, expand the H1 feature space,
or use future H2 outcomes to choose it. Any approved change to H1 fitting
population/protocol needs its own version and freeze. MLP improvement alone
does not remove this requirement.

## Proposed stop points and later evidence

1. **R2 development/calibration:** assess absolute/dose/preservation error,
   family generalization, selected-model uncertainty and coverage. Failure or
   inadequate precision stops work before untouched final evidence. New
   architectures/domain changes need a new proposal, not a continuation until
   PASS. No guarantee that 192 training families will be enough.
2. **Final measurement validation:** only after approved pretest freeze and
   acceptable development/calibration. Keep failures and all assigned families;
   do not refit on these outcomes or relabel them independent after inspection.
3. **Reassess observation/predictor limits, separately authorized:** natural
   readout success does not establish accurate decoding on predicted/patched
   OOD latents or correct dynamics. Preserve original and new readouts, latent
   forecast errors and natural-C controls. Do not recycle old predictor/readout
   gate outcomes as though they were measured with a newly trained instrument.
4. **Human checkpoint before formal Step 3/H1/H2:** calibration cannot unlock
   the existing gate automatically. Historical provenance stays UNRESOLVED;
   empirical operational-domain qualification and H1/selectivity remain distinct.

Family-level resampling and practical error margins follow the distinction
between precision and negligible error discussed in
[Saravanan et al.](https://arxiv.org/abs/2007.07797) and
[Lakens](https://journals.sagepub.com/doi/10.1177/1948550617697177).
These methods do not supply a LeWM-specific 5-px/.10-rad acceptance guarantee.

**Requested next authorization:** a bounded R2 development/calibration protocol
with the above stop points, not full automatic refit→Step 3→H2. Nothing in this
document authorizes R2, H3, closed-loop planning, commits or pushes.
