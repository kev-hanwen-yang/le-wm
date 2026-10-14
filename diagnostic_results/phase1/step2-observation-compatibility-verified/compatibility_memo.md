# Step 2 observation compatibility — measured assessment

## Material Passport

academic-research-suite / experiment-agent; validation; 2026-10-14.
VERIFIED for executed deterministic measurements/regressions and source/cache
identity; ANALYZED for scientific interpretation. Historical provenance remains
UNRESOLVED. This is development evidence, not confirmatory H1/H2/H3.

## Design and matching

The protocol and numeric budgets were written before measuring latent/readout
outcomes. Use the implementation plan's sorted-family, NumPy seed-42 148/18/19
partition. Of the previous fixed sample, 21 first-initial observations meet
quiescent/public-state/readback criteria: 16 development and 5 validation.
No protected-test query family was encoded or used as a reference. The old
episode-split cache filenames do not define this new family partition.

HDF5 versus reconstruction matching is **render-relevant/public-state matching
to stored precision**, not proof of the complete historical physical state.
Missing block dynamics/contact caches and pre-quantization poses cannot be
recovered. Moving middle/final illustrations are excluded. Codec roundtrips use
one unchanged originating image: the physical state is exactly unchanged by
that pixel-only operation. No environment, goal, physics or readout was edited.

All 21 source encodings match their existing frozen cached embeddings under
the original tolerance, maximum absolute difference 4.053116e−6. This excludes
an accidental checkpoint/preprocessing mismatch as the explanation for the
large reconstruction/source differences. Repeated encoding difference is zero.
The three MLP readouts are reused frozen and correctly unnormalized to world
px/wrapped radians; their independent role is relative to future Ridge algebra,
not to all data provenance. No probe or world model is trained.

The natural reference uses eight additional identical-public-state initial
images per family, aggregated within family. The 592-row latent bank contains
development families only and excludes each query's entire family for nearest
neighbors. The sparse bank is descriptive, not a manifold certificate.

## Primary measurements

RMS across validation families; development margins are one-sided 95% upper
family-bootstrap bounds on natural same-state nuisance, frozen before validation
aggregate evaluation. The validation upper bounds are descriptive at n=5.

| Metric | Development natural RMS | Development upper margin | Validation source/reconstruction RMS | Validation upper bound |
|---|---:|---:|---:|---:|
| Latent Euclidean L2 | 0.068802 | 0.087971 | 0.686561 | 0.879722 |
| Agent MLP drift, world px | 0.525765 | 0.731627 | 17.658217 | 24.736982 |
| Block MLP drift, world px | 0.191314 | 0.242617 | 2.582461 | 3.354508 |
| Wrapped-angle MLP drift, rad | 0.032923 | 0.054719 | 0.060704 | 0.079153 |

All empirical-envelope checks fail. The proposed practical pilot caps
1.25 px / 1.25 px / 0.025 rad also fail; halving or doubling those allocations
does not change that aggregate outcome. The stricter 0.25-px block-effect
resolution cap fails as well. These failures were not converted to a PASS.

Absolute Euclidean readout RMS against stored labels on validation:
agent **7.860849 → 18.969322 px**, block **2.424900 → 2.863012 px**, wrapped angle
**0.029127 → 0.067427 rad** for source → raw reconstruction. Thus a drift check
is not being substituted for absolute physical accuracy. Legacy readout
generalization and measurement resolution remain limitations.

The development median latent change for near-10-px observed pusher motion is
**0.496154**, from only **3** qualifying families; actual displacements are
9.846/10.221/10.674 world px with unchanged stored block pose. Readout agent
changes are 5.421/6.718/9.747 px. Four development 20-px references and one
30-px reference are available; validation has one 10-px, two 20-px and no
30-px references. All missing doses are retained in case records.
The validation reconstruction nuisance RMS is 1.384 times the development
10-px median latent signal (upper-bound ratio 1.773). These sparse references
are contextual scales, not proof about the future rank-two patch norm or a
population comparison. No latent intervention or prediction was performed.

Nearest-bank RMS distance is **11.027338 source versus 10.906460 reconstruction**
on validation: reconstruction can look slightly nearer to this sparse bank
while its physical readout worsens. A nearest-neighbor score alone cannot pass
the observation gate.

## Spatial residual and codec evidence

Validation mean region RGB RMSE: background **1.933888**, agent **17.395052**,
block **8.324010**, goal **8.227999**, in raw 0..255 channel units. Respective
mean squared-residual energy shares are **44.00%, 10.23%, 25.32%, 20.45%**.
"Background" includes walls. Object differences concentrate at boundaries:
agent/block/goal edge bands contribute 10.05%/23.35%/19.05% of total energy.
The background interior RMSE is only 0.602559; agent/block/goal interiors are
3.761810/3.064710/2.960538. All masks use fixed saved-render segmentation and
a two-image-pixel erosion, not output-tuned geometry fitting.

There are systematic interior color residuals: source minus reconstruction
mean blue channel is about −3.568 for agent and −2.420 for goal; block mean
green-channel residual is −1.519. This is consistent with photometric/chroma
effects being plausible, while edge localization also leaves rasterization and
subpixel alignment plausible. Neither mechanism is uniquely identified.

Fixed offline libx264 controls, one intra frame, CRF 18/23/28, yuv420p/yuv444p,
single thread, FFmpeg 7.1, do not change the originating scene or production
pipeline. Same-state validation latent RMS shifts range **0.370560–0.936133**;
agent MLP RMS shifts **4.320096–13.293413 world px**. The exact RGB FFV1 control
has zero pixel, latent and readout change. This directly demonstrates nuisance
sensitivity at unchanged physical state, not an actual agent movement.

Codec effects resemble some residuals but do not reproduce the source uniquely.
Different conditions alter background/foreground behavior differently; all
conditions are in `codec_summary.csv`. Some reduce one discrepancy but remain
outside practical readout budgets. No winner is selected or applied. Temporal
MP4 prediction, original decoder/range/matrix settings, library rasterization
versions and true subpixel poses remain untested/unidentified. Exact historical
provenance remains UNRESOLVED and is not inferred from these experiments.

## Decisions and next requirement

**Compatibility-qualified limited Step 3 development pilot:** NOT SUPPORTED
under the current candidate plus existing frozen readouts. The problem is now
quantified nuisance sensitivity, not merely incomplete historical records.
A separately scoped observation/readout-calibration diagnostic could be useful;
it would remain explicitly incompatible/OOD development, not passing causal data.

**Freeze observation contract for confirmatory testing:** NOT SUPPORTED.
The prospective observation/goal semantics have not been approved; only five
previously pixel-exposed validation families exist; empirical/practical criteria
fail; readout/dose/history/contact coverage is insufficient. Exact historical
records need not be recovered if an independent prospective contract eventually
passes meaningful calibrated criteria. See `observation_compatibility_gate_revision.md`.

The prior central-marker comparison candidate is not automatically the actual
mechanics fixture's `(350,120,0)` marker/physical-goal contract. No goal is changed
to reconcile them. Validate the **actual** prospective pipeline and independent
readout resolution before collecting compatibility-qualified Step 3 groups.
Actual Ridge patch-scale validation belongs to the later intervention gate;
it must not create a circular prerequisite for initial observation/data work.

## Verification, scope and artifacts

85 direct tests across 12 scripts pass, including all Step 1/2 mechanics tests.
Independent executions agree exactly on all 7 HDF5 datasets, 2 root attributes,
case/spatial metrics, family partition and development calibration. No physics,
task goal, goal encoding, preprocessing, checkpoint, action/time contract, cost
or official metric was changed. No Step 3 collection, Ridge training, H2/H3,
commit or push occurred. Native SDL/deprecation warnings are preserved in logs.

`compatibility_manifest.json` contains source/checkpoint/readout/FFmpeg hashes,
versions, actual commands, immutable parents, measured outputs and failed checks.
Large RGB/latent/readout/reference arrays, codec files/commands, source snapshots
and regression logs live under `.stable-wm/phase1/step2-observation-compatibility-verified/`.
Figures and tables are generated from saved numeric outputs, not fresh model calls.

Statistical fallacy scan **11/11**: development/validation are separate (Simpson);
no individual/population claims from aggregates (ecological); geometry filtering
limits scope (Berkson/collider); no prevalence/rare-failure claim (base rate);
no extreme-score selection/retest claim (regression to mean); all prior
exclusions disclosed (survivorship); entire fixed codec grid reported (look-
elsewhere); protocol/criteria prewritten and no threshold revision for PASS
(forking paths); codec input dependence is not physical causality (correlation);
no reverse-direction claim (reverse causality). Selection, tiny top-level N,
legacy probe provenance and partial historical-state observability remain CAUTION.
