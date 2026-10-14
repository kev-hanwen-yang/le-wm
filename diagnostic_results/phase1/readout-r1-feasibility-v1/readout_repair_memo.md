# R0/R1 readout repair — measured assessment and stop

2026-10-15, Australia/Melbourne. **R0 found no original pipeline bug. R1
improves natural-state readouts, but does not qualify a position instrument
or a reliably resolved block-effect scale.** Do not promote these weights,
freeze the observation contract or collect formal Step 3 data. Historical
provenance remains **UNRESOLVED**. R2 requires approval.

## Material Passport and methods

Protocol: `readout_repair_protocol_r0_r1.md`, written before these outcomes;
passing parent `readout-r0-sanity-verified-v1`. Frozen 192-D LeWM CLS latents,
CPU/float32, unchanged Step 2 goal/render/preprocessing. Only readout heads
were trained on independently checked natural quiescent simulator images.

Training: 48 families (32 exposed earlier diagnostics plus 16 new), nested
balanced counts 16/32/48. Independent pilot-validation: 16 new families,
eight free-space and eight near-contact. Each has 20 distinct natural scenes
and one zero-repeat: 1,344 images total, **not 1,344 independent samples**.
All 64 families were valid; no exclusions or replacements. Pilot families
selected Ridge penalty/MLP epoch and are now exposed development evidence.
No future final-test family was generated. Seeds/assignments/labels/checks,
checkpoint identities, environment versions and commands are in the manifest.

Bounded candidates: original three MLPs; four Ridge penalties; existing
MLP256, fixed optimizer/schedule, seeds 42/43; scalar and separately versioned
sin/cos angle heads. There are 75 comparison rows and 72 saved new heads,
including all penalties/counts and both seeds. Seed averaging below averages
**squared errors**, not predictions and not an ensemble selected for deployment.
Family-bootstrap intervals use 10,000 paired draws. They are conditional on
pilot selection, not multiplicity-adjusted or final confirmatory intervals.

## Absolute recovery on pilot families

Position is Euclidean world-px RMSE; angle is wrapped-radian RMSE. Brackets
are two-sided 95% family intervals. All new rows use 48 training families.

| Head family | Agent px | Block px | Wrapped angle rad |
|---|---|---|---|
| Original MLP | 44.01 [32.12,55.19] | 16.53 [14.31,18.39] | scalar 0.358 [0.272,0.439] |
| Selected Ridge | 26.48 [20.81,31.72] | 7.83 [4.87,10.89] | sin/cos 0.0951 [0.0670,0.1231] |
| MLP seed 42 | 25.55 [17.24,33.75] | 8.34 [5.20,11.18] | sin/cos 0.0650 [0.0466,0.0848] |
| MLP seed 43 | 26.70 [16.26,36.13] | 8.19 [4.87,11.16] | sin/cos 0.0754 [0.0518,0.0964] |

Position heads exceed the candidate 5-px absolute cap. Sin/cos MLP one-sided
95% upper bounds, 0.0816/0.0930 rad, meet the candidate 0.10-rad **absolute**
cap on this selected pilot; Ridge's 0.1182 does not. New scalar-angle errors
are worse than the legacy head (0.698/0.718/0.720 rad); these failures remain
reported. No near-zero sin/cos decode occurred.

Coordinate RMSE is distinct: Ridge agent x/y = 19.55/17.85 px; MLP42 =
20.38/15.42 px. Ridge agent mean coordinate R² = 0.9628 (one-sided lower
0.9375), yet its Euclidean error is 26.48 px: high R² is not sufficient
measurement resolution. Absolute agent errors remain worse in free-space
than near-contact (MLP42 31.20 versus 18.23 px). Systematic biases, coordinate
errors, R² and stratum intervals are saved in JSON/CSV.

## Differential resolution and non-target controls

Predeclared proposed target margins are `min(5 px, dose/2)` for positions
and `min(0.10 rad, dose/2)` for angles. Require a one-sided 95% error upper
bound within that margin, positive-direction evidence, and matched non-target
preservation. Bounds crossing a margin are **INCONCLUSIVE**, not PASS.
The half-effect allocation is a proposed instrument budget, not a validated
universal physical threshold. The 1-px/0.5-px block objective is separately
high-resolution; the original Step 3 >=1-px feasibility threshold is unchanged.

| Target differential RMSE | Ridge | MLP42 | MLP43 |
|---|---:|---:|---:|
| Agent 10/20/30 px | 7.89/13.62/18.79 | 5.93/9.57/10.68 | 6.10/9.90/10.78 |
| Block 1/2/5/10 px | 1.28/2.25/3.72/6.06 | 0.97/1.48/2.43/4.44 | 1.07/1.67/2.70/4.80 |
| Angle .01/.025/.05/.10 rad, sin/cos | .0239/.0274/.0415/.0718 | .0169/.0230/.0298/.0444 | .0184/.0233/.0302/.0457 |

MLP agent-10 is inconclusive against 5 px; agent-20/30 fail. Positive
direction alone is insufficient: at agent-10, MLP42 mean cosine is 0.802,
lower bound 0.695, but magnitude RMSE is 3.93 px and vector error is 5.93 px.
Block-1/2 fail; MLP target-only block-5/10 are inconclusive (MLP42 upper
bounds 2.780/5.289 px, margins 2.5/5). Neither qualifies a measurable scale.

When only the block moves 5/10 px, MLP42 reports spurious agent changes
6.78/9.55 px, exceeding the non-target cap. Thus all combined instrument
bundles fail those block scales. **No tested 1–10-px block scale passes the
combined criteria.** This is instrument leakage on natural inputs, not actual
agent motion or proof of causal encoder entanglement. Angle .01/.025 fail;
.05/.10 MLP target errors are inconclusive, with additional agent leakage
at .10. Absolute-angle accuracy does not certify small angle-effect accuracy.

All zero-repeat readout differences are exactly zero. Every tested nonzero
pilot dose changed RGB and frozen latents; no dose collapsed to an identical
image or zero latent signal. Therefore readout failures cannot simply be
explained by complete rasterization/encoding aliasing. Bias, direction,
magnitude, per-family vectors and all non-target errors are retained.

## Eight requested decisions

1. **Original bug?** None found in labels, geometry, units, preprocessing,
   checkpoint architecture/transforms or image identity. The new audit's
   floating-equality fixture failure is preserved and explained in the R0 memo;
   no original pipeline correction was needed.
2. **Absolute physical recovery?** Improved, but agent/block are inadequate
   against 5 px. Sin/cos angle is promising against 0.10 rad on selected pilot
   data, not an official or confirmatory instrument.
3. **Actual agent doses?** No reliable qualification at 10/20/30 px; 10 is
   inconclusive for MLP and 20/30 fail the proposed vector-error budget.
4. **Useful non-target resolution?** Not yet demonstrated. Larger block and
   angle changes induce false agent readout changes; fine angle doses also
   fail or are inconclusive. Exact zero leakage is not enough.
5. **Smallest block scale?** None qualified among 1/2/5/10 px. Do not assume
   that untested larger scales pass. The optional 1-px high-resolution
   objective fails without becoming a universal Phase I prerequisite.
6. **Scaling/saturation?** 16→48-family gains are positive for positions and
   sin/cos angle. At 32→48, MLP block gain is 1.93 px [1.03,3.07], sin/cos
   gain .0447 rad [.0236,.0639]; agent gain 3.87 px [-.50,8.89] is less
   certain. Ridge agent/block gains are about .51/.50 px with intervals
   spanning zero. Training/pilot gaps remain large (MLP42 agent 3.95/25.55
   px). Three sizes do not establish saturation or guarantee eventual success.
7. **Ridge versus MLP?** No clear material absolute-position advantage:
   paired Ridge→MLP seed-error-mean gains are +.34 px [-4.84,6.91] agent and
   -.44 px [-2.82,1.49] block. Sin/cos gain +.0247 rad [-.0028,.0502] is
   suggestive, not definitive. MLP differential errors are lower descriptively,
   but still insufficient. Neither model qualifies the position instrument.
8. **R2 justified?** Evidence supports a further bounded development-only
   scaling/coverage study, **not immediate instrument freeze or blind full
   confirmatory refitting**. `readout_refit_proposal_v2.md` proposes staged
   stopping rules; approval is required before implementation/execution.

## H1 is a separate unresolved requirement

No official frozen Ridge/P currently exists; the source affine head was
trained with AdamW, not Ridge, and gives 60.53-px pilot agent error. R1 Ridge
rank is two, but row-space angles change 71.96/73.50 degrees from 16→32
families and 64.23/67.90 from 32→48; source-affine→R1 angles are
81.42/86.16 degrees. These descriptive comparisons combine changed data and
penalty selection, so they do not prove the absence of a causal subspace.
They also do not establish prospective-domain H1 suitability. No projector
was created/replaced and no patched latent or H2 outcome was used.

## Verification, evidence and limits

All **118 direct tests in 15 scripts** passed, including unchanged Step 1/2
regressions. Independent complete execution reproduced 142 HDF5 datasets,
259 attributes and all 72 model files exactly, plus assignments, learning
curves and feasibility decisions. This repeat is not additional independent N.
Original MLPs, LeWM weights and audited source contracts are unchanged; all
world-model parameters remain frozen and gradient-free.

Files: `readout_comparison.csv/json`, `dose_error_comparison.csv`,
`epoch_learning_curves.csv/json`, `supplemental_summary.json`,
`reproducibility_checks.json`, manifests and family records. Figures:
`family_learning_curves.png`, `epoch_learning_curves.png`,
`dose_resolution.png`, `non_target_preservation.png`; these are R1 diagnostics,
not the four completed H2 figures. Arrays/weights/logs/source snapshots live
under ignored `.stable-wm/phase1/<run_id>/`.

Statistical integrity scan **11/11 checked**: strata are reported (Simpson);
inference stays family-level (ecological); restricted geometric population and
validity selection are explicit (Berkson/collider); no diagnostic prevalence
claim (base rate); no outcome-selected families (regression to mean); zero
attrition, failures retained (survivorship); all 75 candidates and both seeds
reported, selection-conditional intervals (look-elsewhere); prewritten bounded
protocol and labeled post-fit summaries (forking paths); no latent/H2 causal
claim (correlation/causation or reverse causality). Remaining caution: only 16
pilot families, model selection and many exploratory comparisons. No global
error-controlled PASS is asserted.

The academic-research workflow informed family-level uncertainty, preserved
failure records and separation of instrument feasibility from causal validity.
**Stopped after R1:** no R2, formal Step 3, H2/H3, commits or pushes.
