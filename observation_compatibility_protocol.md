# Proposed Step 2 observation-compatibility assessment — version 1

Written before this assessment's encoder/readout results. Historical provenance
remains UNRESOLVED. This protocol does not authorize full Step 3 collection or
change the existing `encoder_dataset_ready=false` gate.

Use saved source/reconstruction initial images only. Exclude moving middle/final
pose illustrations. Match positions, block angle and zero public agent velocity
to the source's stored precision and previous fresh readbacks. This certifies
**render-relevant/public-state matching**, not complete historical dynamics:
block velocity/contact caches, true pre-quantization pose and source style/goal
configuration are unavailable. Consequently HDF5 comparisons cannot be called
exact full-state restoration or exact original rendering. A codec roundtrip of
one saved image has an exactly unchanged originating physical state; saved
Step 2 replay images also have complete audited current-backend body readbacks.

No environment is reset, stepped or edited in this assessment. No goal, cached
goal image, physics, model, preprocessing, action or official metric is changed.
All codecs are offline diagnostic conditions, never replacement production data.

## Partition and references

Apply the existing implementation plan's rule: sort all 185 initial-observation
family IDs, shuffle with NumPy `default_rng(42)`, allocate 148 development / 18
validation / 19 protected test. Do not encode/read out protected test families.
Within the prior predetermined sample, only 21 valid first-initial families are
available: 16 development, 5 validation. All were exposed in the previous pixel
audit: validation is new to latent/readout assessment, not virgin confirmatory
data. Do not resample convenient families after seeing measurements.

Natural same-render-state references: the first nine episode-initial observations
per query family (first versus eight others), requiring identical stored 7-D
state. Codec/appearance variation is not independent physical evidence; aggregate
within family. Do not infer the original codec from this reference.

Natural latent reference bank: four evenly spaced saved frames in the first
episode of each development family, excluding each query's entire family for
nearest-neighbor distances. Protected test rows are not references.

Intervention-scale references: from each query family's first saved trajectory,
find the earliest equally good frame whose agent displacement is within 1 world
px of 10, 20 or 30 px, with **exactly unchanged stored block position and angle**.
Selection uses states only, not latent/readout outcomes. Log missing doses. These
are genuine observed pusher-motion scales, not a Ridge patch or causal H2 test;
public agent velocity can differ, but the single-image renderer does not encode
velocity as a separate channel. Actual future rank-two injected patch norms
are not yet available and must be checked before confirmatory intervention work.

## Measurements

- Raw spatial residuals: background, agent, block, goal; interior/edge regions,
  signed color bias, RGB RMSE and error-energy shares in the saved 224-pixel grid.
- Plausible codec mechanisms: fixed libx264 CRF 18/23/28 at 4:2:0 and 4:4:4,
  one intra-coded image, single-threaded; include an exact RGB lossless control.
  Compare all conditions. Do not select a winning codec/pipeline or claim MP4
  generation equivalence. Temporal compression effects remain untested.
- Frozen preprocessing tensor differences, 192-D latent Euclidean differences,
  norm change and same-family-excluded natural-reference distances.
- Frozen, existing agent/block/angle MLP readouts, correctly unnormalized to
  world px and wrapped radians. Report paired readout drift **and** absolute
  error against stored source labels. They are independent of a future Ridge
  projector's algebra, not independent ground-truth sensors or proven family-
  held-out instruments. No probe fitting/selection is permitted here.
- Ratios to same-state natural variation and available dose references; missing
  or zero denominators stay undefined, without epsilon or clipping.

## Proposed decision criteria (not an automatic PASS)

Use equally weighted families. For each family, average squared reference
differences over its repeated observations before computing population RMS.
10,000 seeded family-bootstrap draws give descriptive one-sided 95% bounds;
five validation families cannot establish population equivalence or rare tails.

Calibrate natural-nuisance margins **on development only**: one-sided 95% upper
RMS bounds for same-state latent/readout differences. Save these margins before
evaluating validation aggregates. Validation reconstruction/source differences
should lie within the corresponding natural-nuisance envelopes. A larger source
probe error is not permission to enlarge a nuisance tolerance.

Separately require readout drift upper bounds below 1.25 world px for each
position and 0.025 rad for angle. These are proposed error-budget allocations:
1/4 of the plan's practical 5-px / 0.10-rad preservation caps per observation,
so two independent observation shifts consume at most half of that contrast's
budget by the triangle inequality. They are design choices requiring review,
not empirically justified universal constants. Report sensitivity at half/double
these proposed allocations, without changing the primary result afterward.
For confirmatory effects at the plan's minimum informative **1-world-px block
motion**, the preservation budget alone is too permissive: require block nuisance
drift at most **0.25 world px per observation**, so a two-observation contrast
consumes at most half that effect scale. Also calibrate actual block-effect
readout resolution; a low observation drift cannot rescue a noisy readout.

Limited **development-only** collection can be recommended, with caveats, if
mechanics remain valid, preprocessing/checkpoint identity holds, the observed
readout nuisance stays within the proposed practical budgets, and reconstruction
latent shifts are smaller than the development median 10-px natural-motion
signal. Failure of natural-nuisance equivalence must still be reported. A pilot
recommendation does not establish H1/H2, authorize collection now, or certify
probe accuracy; its purpose is bounded development/calibration.

A confirmatory observation-contract freeze needs practical and natural-nuisance
checks on independent, sufficiently diverse validation states; resolution of
systematic spatial/state bias; readout measurement validation; coverage of the
planned contact/orientation/history strata; nuisance margins small against the
**actual** injected patch norm and physical effects; and explicit prospective
renderer/goal semantics shared across all branches. Current five-family,
previously pixel-exposed evidence alone cannot satisfy those conditions, even
if a bootstrap comparison passes. This requirement is independent of recovering
exact historical MP4 provenance. No arbitrary pixel-RMSE acceptance threshold
is used. Proposed gate revision separates provenance, pilot feasibility,
observation compatibility and confirmatory freeze; none silently replaces the
existing gate until human review.
