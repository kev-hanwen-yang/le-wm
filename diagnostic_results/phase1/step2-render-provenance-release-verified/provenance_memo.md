# Step 2 goal/render provenance decision

Material Passport: academic-research-suite / experiment-agent; validation;
2026-10-14; VERIFIED for executed file/replay/preprocessing checks;
INCONCLUSIVE for original observation-generation provenance.

The original HDF5 is now proven byte-identical to the pinned official release:
archive SHA-256 `7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318`,
HDF5 SHA-256 `b6ebd9ac94bbe9e383f6e7a9cd92d74e9aa665ea57b758ed3717b0ee7df8d4fb`.
The published artifact does not contain original goal/style/seed/codec metadata.

Original DINO-WM renderer defaults establish visible light-green display marker
`(256,256,pi/4)`, white 512-pixel canvas, blue circle, gray T, OpenCV 224 RGB
resize. Available historical SWM conversion code reads decoded MP4 frames,
but targets Minari rather than documenting this HDF5's full conversion run.
The installed renderer separates its displayed marker from the physical task
goal and cached goal image. Those must not be conflated.

128 images were predetermined from 32 of 185 initial-observation families.
54 admit approximate quiescent pose reconstruction; 74 are explicitly rejected
without corrections/resampling. The 21 valid first-initial families have mean
raw-channel RMSE 2.794267 visible, 13.617667 hidden, 16.716039 shifted and
15.655947 rotated. No exact image match was obtained. This is compatibility
evidence, not recovery of the original configuration or a causal result.

76 direct tests pass. The new full mechanics suite passes 24 case/seed groups;
1,918 replay datasets and 151 attributes exactly match the historical run.
The independent image audits agree on 580 datasets and 183 attributes.
Training, official evaluation and Step 1 preprocessing have max delta 0,
including the independent uint8/ImageNet formula. The CPU frozen recheck has
repeated encode/predict/rollout deltas 0 and all 48 cache samples match;
CEM was intentionally skipped, so that recheck is not a new full Step 1 gate.

| Gate | Status |
|---|---|
| Fresh-start mechanics | PASS |
| Goal visibility | Source-default confirmed; dataset empirically supported |
| Actual goal-pose / variation convention | UNRESOLVED |
| Raw-image generation compatibility | INCONCLUSIVE |
| Frozen preprocessing compatibility | PASS |
| Encoder-facing Step 3 readiness | FALSE |

Next decision requires human review: obtain original generation/codec/conversion
records, or approve a clearly labeled compatibility protocol with the explicit
default display marker and an acknowledged source-image domain shift. Neither
approval nor dataset collection is implied by this memo. The full source/line
audit, reasons and protocol safeguards are in the new section of `step2_review.md`;
exact run commands, identities, per-case metrics and regression logs are in the
manifests and `reproduce_phase1.md`. No physics/controller/model/task-goal
encoding/normalization/cost/official metric was changed. No Step 3/H1/H2/H3,
commit or push was performed.
