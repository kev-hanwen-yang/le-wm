# R0 input/label sanity audit — completed

2026-10-15, Australia/Melbourne. **No concrete bug was found in the original
measurement pipeline.** R0 passed its implementation checks; that does not mean
the original readouts are accurate in the prospective observation domain.
Exact historical provenance remains **UNRESOLVED**.

## Material Passport and scope

- Evidence: `repair_manifest.json`, `audit_measurements.json`,
  `absolute_domain_comparison.json`, `geometry_audit.json`,
  `pose_concentration.json`, `position_concentration_supplement.json`,
  `r0_domain_comparison.csv` and `r0_geometry.png` in this directory.
- Inputs: all 32 previously exposed v2 families at four initial agent doses,
  plus 21 previously exposed source initial images and 21 fresh scenes with
  those same stored poses: 170 images. These are development evidence, not
  new confirmatory families. Source images retain their original HDF5 IDs.
- Frozen LeWM: `lewm_object.ckpt`, SHA-256
  `100d6459b9198697c1c26ef47e3eef4d7548946401b7501213caf761495a2553`;
  CPU/float32 inference. Original MLP identities and package versions are in
  the manifest. Large arrays, source snapshots and test logs are under
  `.stable-wm/phase1/readout-r0-sanity-verified-v1/`.
- No original physics, goals, preprocessing, checkpoint, target definition,
  actions, temporal alignment, CEM or official metrics were modified.

## Verified contracts

1. **Physical labels:** agent target is `agent.position`; block target is
   `block.position`, the transformed local origin. Body readback indices
   `[0,1,9,10,11]` agree with public-state position/angle labels to the existing
   `1e-10` world-state tolerance. Labels and RGB are read from the same fresh
   environment; rendering leaves body values unchanged. Existing target
   builders use state `0:2`, `2:4`, `4:5` as expected.
2. **Block origin is not its center:** the local center of gravity is `(0,45)`;
   the union-of-polygons area centroid is approximately `(0,40.714286)`.
   Neither is the position label. Rotation changes these centers relative to
   the origin. The independent polygon/rigid-transform audit avoids diagnosing
   this expected offset as a mislabeled block.
3. **Geometry and units:** physics coordinates and the native canvas are
   512-world-px, with image Y increasing downwards. The RGB frame is resized
   to 224×224 by the existing OpenCV path. Positions must not be interpreted
   as 224-image-px. Analytic transformed T vertices agree with engine
   `local_to_world` to a maximum `5.6843e-14` world px over 149 fresh renders.
   Independent circle/polygon masks have zero missing deep-interior or
   out-of-boundary-band pixels. Boundary bands derive from stroke, rounding
   and rasterization, not a tuned pixel-RMSE compatibility threshold.
4. **Preprocessing:** RGB uint8 HWC → `(1,T,3,224,224)` float32; scale by
   `1/255`, ImageNet normalization and the original bilinear antialiased
   resize. Training, official evaluation and an independent formula agree
   exactly. RGB orientation and exact native-canvas-to-frame resize agree.
5. **Original MLPs:** strict checkpoint loading verifies 192→256→output
   architecture, ReLU, dropout 0.1 disabled in eval, two-output position
   heads and one-output scalar-angle head. Stored target mean/std invert
   normalization as `output*std+mean`. Independent layer-by-layer forward
   evaluation agrees exactly. There is no head swap or extra normalization.
6. **Image/label/cache identity:** repeated encoding differs by zero. Maximum
   difference from sampled saved latent identity is `2.26498e-6`, within the
   unchanged cache-check tolerance. Fresh physical readbacks and saved images
   agree; no cache was rebuilt and no label was corrected.

## Measured source/prospective comparison

Position entries are equally family-weighted **Euclidean** world-px RMSE;
angle is wrapped-radian RMSE. Coordinate-level errors and family intervals
are saved separately, not mixed with Euclidean errors.

| Population | Independent families | Agent px | Block px | Angle rad |
|---|---:|---:|---:|---:|
| Original source images | 21 | 5.663 | 3.098 | 0.0748 |
| Prospective renders at those stored poses | 21 | 55.563 | 13.336 | 0.5189 |
| Prospective v2 initial-dose scenes | 32 | 46.546 | 16.461 | 0.4519 |

The source agent 95% family interval is `[4.003,7.261]`; for prospective
renders at the same stored poses it is `[29.768,78.257]`. This comparison
controls stored rendering-relevant poses more directly than an unmatched
source/fresh population comparison. It does **not** certify original hidden
dynamic state, unquantized positions, goals, rasterization or MP4 processing.
It cannot identify a unique rendering mechanism or reproduce historical
generation. The prospective marker remains the Step 2 `(350,120,0)` marker.

Errors depend on pose and appearance condition, not just a constant offset.
In the exposed fresh sample, agent RMSE is 58.531 px in free-space versus
30.118 px near-contact; orientation-quadrant RMSEs are
20.638/34.761/65.943/52.730 px. Angle error is largest in quadrant 3
(0.7624 rad). These are small-sample descriptive strata, not subgroup discovery
tests or independent evidence that one appearance feature causes failure.
Position-bin errors and signed-error/position correlations are retained in
the supplemental JSON. No subgroup was excluded or used for correction.

## Corrections, failed attempt and decision

**Original implementation corrections: none.** The new audit helper required
tuple inputs for Pymunk, and its new unit-test fixture initially asserted
bitwise equality for floating body positions. The unsuccessful
`readout-r0-sanity-v1` attempt is preserved, including its failed regression
record. The fixture was corrected to the already established `1e-10`, `rtol=0`
physical-state tolerance; no simulator or scientific tolerance was loosened.
The successful new run ID records the corrected test/source identity.

All original regressions plus the new seven audit tests passed. Because the
contracts are valid and the original readouts remain inadequate, bounded R1
was scientifically warranted under the user's approval. R0 does not repair
the observation gate or authorize Step 3. Failure of these particular probes
does not prove that LeWM contains no useful physical-state information.
