## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run, with deterministic reproducibility verification
- Origin Date: 2026-10-14 (Australia/Melbourne)
- Verification Status: VERIFIED for the specified fresh-start mechanics fixtures only
- Version Label: phase1_step2_fresh_start_v1

## Outcome and scope

Step 2 fresh-start simulator replay validation is implemented and passed on the
current CPU/Pymunk backend. Two separate processes each completed eight fixed
mechanics fixtures under three seeds: 24 case–seed checks, each with two audited
fresh starts and one unmodified Push-T reference. These are not 24 sampled
independent research families and do not support any H1/H2 statistical claim.

The gate is deliberately narrower than a full-state-restore claim:

- `step2_gate_passed=true`: verified fresh quiescent initialization, unchanged
  dynamics, repeated physical trajectories and rendering under stated settings.
- `arbitrary_dynamic_restore=false`: no incomplete public-observation restore
  or undocumented space-copy approximation is supported.
- `encoder_dataset_ready=false`: training goal/render convention still needs
  resolution before encoder-facing Step 3 collection.
- No world-model inference, probe fitting, AB/ABC research dataset, H2, H3 or
  model-based closed-loop planning was run. The installed PD servo remains unchanged.

## Files and design choices

### `counterfactual/simulator.py`

This is the reusable simulator boundary, not a new simulator.

- `SceneSpec` stores the explicit scene seed, agent/block positions, block angle
  and shared quiescent goal in world coordinates. It copies mutable input arrays,
  checks dimensions/finiteness and generates explicit reset options with variation
  sampling disabled. Its default goal is a stated safe mechanics-fixture choice,
  not a recovered training goal; Step 3 must supply its approved goal contract.
- `SimulatorSettings` controls resolution, goal display and numerical tolerance,
  leaving all physics arguments at installed defaults.
- `body_vector` reads 18 fields: position, angle, linear/angular velocity, force
  and torque for both bodies. `physical_readback` adds public observations,
  proprioception, goal state/pose, latest action, contacts and RNG identity.
  `simulator_contract` records actual geometry, masses/inertia/CoG, solver,
  controller, static walls/body, style, native versions and hidden-state limitations.
- `AuditedPushT` inherits the installed implementation. Its `_setup` observes
  `space.step` without changing its arguments/order. Its post-solve callback
  delegates the original handler before copying typed contact points, normals and
  impulses. Callback exceptions are retained because CFFI may otherwise swallow
  them. Live arbiter objects are never held after the callback.
- `create_fresh_scene` calls reset once, validates the result and closes rejected
  environments. It never corrects a pose, zeroes a returned velocity, steps to
  settle a scene or edits physics to obtain a passing result. Both reset physics
  steps are audited, including the goal initialization step.
- `geometry_clearances` uses the actual circle radius, polygon point queries and
  shape/wall bounding boxes. Center distance alone cannot detect T-shape overlap.
  Starts touching/penetrating a body or wall, outside supported variation bounds,
  or contaminated by goal-reset contact are rejected.
- `replay_fixed_actions` records all h=0..35 states/images plus all audited
  substeps. It preserves returned reward/termination values, validates raw
  commands without clipping, uses the released float32 action-space dtype and
  saves the derived relative controller targets. Same relative commands do not
  imply the same absolute controller targets in different scenes.
- `compare_replays` compares each physical field independently with circular
  angle differences, rather than combining px, radians and force into one score.
  Pixel, action, contact, event, RNG and metadata comparisons are exact.
- `restore_dynamic_state` raises explicitly: readback is not a complete snapshot.

The observer-neutral comparison against a separate unmodified `PushT` instance
is important: two equally changed simulators could otherwise repeat perfectly.

### `counterfactual/simulator_validation.py`

This contains stage-specific fixtures, evidence persistence and the acceptance gate.

- `validation_cases` supplies stationary, free-motion, near-contact hold,
  agent–block push, push/release, block–wall collision, rotated hold and angle-branch
  hold cases. The first ten commands are zero raw actions, so h=0/5/10 comes from
  independent physical evolution, not copied/teleported history frames.
- `validate_case_behavior` requires that fixtures actually exercise their stated
  mechanics and that all first-ten-step histories remain quiescent. Direct
  accumulated contact-point counts are not unique collision counts or percentages.
- `write_replay` saves the numeric/image/RNG/contact evidence to HDF5 and labels
  it as incomplete for arbitrary restoration. Seven standardized action tokens
  use the exact parent training statistics; no model is loaded for prediction.
- `audit_unsafe_public_restore` is a negative diagnostic, not an alternative API.
  It restores h=20 public observations in a new environment and compares both
  initial readback and all remaining states with the original continuation.
- `validate_baseline_parent` requires a completed Step 1 gate, compatible full
  baseline config/contract/checkpoint/runtime, unchanged existing inference/physics
  sources and source dataset file metadata. Only the CLI/config extensions are
  exempted; new code is independently snapshotted. Dataset size/mtime plus prior
  sampled identity is stated accurately, not described as a full-data hash.
- `audit_dataset_rendering` saves 16 source images/labels, hashes and metadata
  gaps. Descriptive color counts are not promoted to renderer identity.
- `run_simulator_stage` writes early/partial manifests, a CSV and summary, refuses
  output reuse, saves source and Chipmunk binary identities, checks deadlines,
  compares duplicate and unmodified traces, and verifies identities again before
  passing. Failures remain saved. The deadline is cooperative: a native call that
  stalls cannot be interrupted by an in-process check.

No third-party dependency was added. Existing HDF5 compression plugin registration
is necessary when inspecting the source data. Physics images/tensors remain under
the ignored `.stable-wm/phase1` directory.

### `scripts/run_phase1.py`

Adds `--stage simulator` and an optional `--baseline-manifest` override. It lazily
dispatches to the new stage, retaining existing baseline behavior. Later scientific
stages are not offered, so this CLI cannot accidentally run H2 or planning.

### `config/phase1/pusht.yaml`

Adds a simulator section with the actual passing parent path, seeds
20261014/20261015/20261016, 35 commands, `atol=1e-10`, `rtol=0`, explicit default
goal display and a 300-second cooperative deadline. Step 1 values are unchanged.
Fixture goal/geometry/action constants are recorded in source snapshots and every
case specification; they are not implied to be approved Step 3 generator settings.

### `test_counterfactual_simulator.py`

Eleven direct-execution tests cover input validation/copying, exact quiescent reset,
overlap/tangency/walls/goal contamination, no corrective settling, bounded action
semantics, full mechanics replay, observer neutrality, goal-display modes, related
scene invariants, deliberate intermediate-field corruption, dynamic-restore rejection
and deadline/tolerance failures. Tests use genuine installed Push-T physics.

### `test_counterfactual_simulator_stage.py`

Five direct-execution tests cover partial/incompatible parent rejection, unknown
renderer metadata, HDF5 serialization, preserved early failures/immutable run IDs,
and a two-case end-to-end stage fixture. Its parent/checkpoint stubs are explicitly
synthetic and temporary: that integration test is not the scientific baseline.

### `reproduce_phase1.md`

Retains Step 1 instructions and adds commands, output schema, numerical/time/action
conventions, supported/unsupported state handling, environment warnings and the
renderer-readiness boundary for Step 2.

### `step2_review.md` and saved results

This document explains implementation, verification and limitations. Each saved
run has `simulator_manifest.json`, `simulator_summary.md` and `simulator_cases.csv`.
The first also has `reproducibility.json`, generated by comparing both completed
run records and every saved replay HDF5 dataset/attribute after execution. Existing
completed manifests were not edited for this post-run comparison.

The pre-existing uncommitted `CODEX_REAL_PLAN.md` review was preserved; this step
did not rewrite either research plan, the original simulator or model/planner code.

## Executed verification and measurements

All ten direct-execution test scripts passed: 66 top-level test functions, including
16 new Step 2 functions. Assertions/checks within those functions are additional;
their number is not an experimental sample size.

Actual saved commands (both exit code 0):

```bash
MPLCONFIGDIR=.matplotlib-cache .venv/bin/python scripts/run_phase1.py \
  --stage simulator --run-id step2-fresh-start-replay-reviewed
MPLCONFIGDIR=.matplotlib-cache .venv/bin/python scripts/run_phase1.py \
  --stage simulator --run-id step2-fresh-start-replay-repeat
```

Both runs contain 72 fresh trajectories: 24 first, 24 repeats and 24 unmodified
references. Each trajectory contains 36 raw states/images; each audited trajectory
contains 352 physical states (two reset plus 350 rollout). All repeat/observer
comparisons and intended fixture behaviors passed. The independent processes
have identical case summaries, source/config/parent identities and unsafe-restore
results; **1,918 replay datasets and all attributes matched exactly**. Timestamps,
run IDs, paths and container byte layout were deliberately not equated.

To repeat the saved numeric/attribute comparison without re-running physics:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import h5py
import numpy as np
root = Path('diagnostic_results/phase1')
a = json.loads((root/'step2-fresh-start-replay-reviewed/simulator_manifest.json').read_text())
b = json.loads((root/'step2-fresh-start-replay-repeat/simulator_manifest.json').read_text())
assert a['step2_gate_passed'] and b['step2_gate_passed']
for key in ('cases', 'source_sha256', 'config', 'baseline_parent',
            'unsafe_restore_audit', 'training_render_audit'):
    assert a[key] == b[key], key
with h5py.File(a['output_locations']['replay_hdf5']) as x, h5py.File(b['output_locations']['replay_hdf5']) as y:
    px, py = [], []
    x.visit(px.append)
    y.visit(py.append)
    assert px == py
    count = 0
    for key in ['', *px]:
        u, v = (x[key], y[key]) if key else (x, y)
        assert set(u.attrs) == set(v.attrs)
        assert all(np.array_equal(u.attrs[k], v.attrs[k]) for k in u.attrs)
        if isinstance(u, h5py.Dataset):
            assert u.dtype == v.dtype and u.shape == v.shape
            assert np.array_equal(u[()], v[()]), key
            count += 1
print(f'PASS: {count} datasets and all attributes exactly equal')
PY
```

Representative measured cases at seed 20261014:

| Fixture | Agent–block points | Block–wall points | Final block displacement (world px) |
|---|---:|---:|---:|
| Stationary / free motion / near-contact hold | 0 | 0 | 0 |
| Contact push | 145 | 0 | 120.83335562119473 |
| Contact release | 90 | 0 | 80.68285532520954 |
| Wall-contact stress | 100 | 193 | 114.78140574115619 |

These displacements establish fixture excitation, not patch/causal fidelity. The
wall case stresses installed dynamics; initial validity is checked, but the
kinematic agent is not promised to remain within visual workspace after that stress.
Future dataset action/visibility constraints still require their own frozen rules.

Negative restoration at h=20 adds exactly one extra physics step in both tested
cases and fails to match the original continuation. In the contact case, source
block velocity is `[55.65513876231677, 25.601702187754917]` world px/s and angular
velocity is `0.3580491063298553` rad/s; neither is present in the public 7-D state.

An initial unit test incorrectly required post-push coasting. The installed
`space.damping=0` makes that an inappropriate assumption. It was replaced by an
explicit push-then-withdraw/post-contact fixture **without changing damping or
any physics**. This was a mechanics-test correction, not tuning a held-out research
outcome. The initial failure log and corrected test/run logs are retained.

## Rendering follow-up and next gate

Visual inspection of the saved training preview shows a green T-shaped goal.
An exploratory same-pose reconstruction of source rows 0/109/218/327 used the
installed default rendered goal pose `(256,256,pi/4)`. With the goal visible,
pixel RMSE was 2.145–2.172 on the raw 0..255 channel scale; with it hidden, RMSE
was 11.921–11.931. These four rows share an initial-observation family and are
not independent replicates. This supports visible-goal rendering and contradicts
silently inferring `with_target=False`; it does not establish exact renderer identity.

The exploratory image-only check is reproducible with this command; it does not
generate causal data or approve a goal convention:

```bash
MPLCONFIGDIR=.matplotlib-cache .venv/bin/python - <<'PY'
import h5py, hdf5plugin, numpy as np
from counterfactual.simulator import SceneSpec, SimulatorSettings, create_fresh_scene
with h5py.File('.stable-wm/pusht_expert_train.h5') as f:
    for row in f['ep_offset'][:4]:
        s = f['state'][int(row)].astype(np.float64)
        target = f['pixels'][int(row)].astype(np.float64)
        scene = SceneSpec('render_only_audit', 20261014, tuple(s[:2]), tuple(s[2:4]), float(s[4]),
                          goal_state=(70.,420.,256.,256.,float(np.pi/4),0.,0.))
        for visible in (True, False):
            env, _ = create_fresh_scene(scene, SimulatorSettings(with_target=visible))
            try:
                image = env.render().astype(np.float64)
                print(int(row), visible, 'pixel_RMSE', np.sqrt(np.mean((image-target)**2)))
            finally:
                env.close()
PY
```

The source HDF5 lacks physical goal state, goal pose, seed and variation metadata.
The sampled images and imperfect image-only reconstruction cannot recover the
entire task/goal convention. The suite's explicit safe fixture goal is not claimed
to be the training goal. **Keep `encoder_dataset_ready=false`** until source goal/
render provenance is recovered or an explicit, separately documented protocol is
approved. Do not tune an intervention against these image comparisons.

Fresh-start construction remains validated only on the tested cases/backend.
Every new dataset family must still pass initialization/geometry/history checks
and its own replay validation. Matching readbacks cannot certify hidden caches
for arbitrary dynamic snapshots or cross-version/hardware reproducibility.

The installed native SDL duplicate-class and package deprecation warnings were
observed and preserved. No completed gate run crashed or mismatched; installed
dependencies were not altered to conceal warnings. `MPLCONFIGDIR` only provides
a writable ignored cache. Native hangs remain a limitation of the cooperative deadline.

Pymunk explicitly warns that copied/pickled spaces omit some internal collision
cache data. This supports refusing to call a space copy an exact dynamic restore:
[Pymunk 7.2 Space documentation](https://www.pymunk.org/en/7.2.0/pymunk.html#pymunk.Space).
Contact data is copied within callbacks, consistent with the API's arbiter lifetime
guidance: [Pymunk 7.2 Arbiter documentation](https://www.pymunk.org/en/7.2.0/pymunk.html#pymunk.Arbiter).

**Checkpoint:** fresh-start mechanics PASS; renderer/domain provenance unresolved;
H1/H2/H3 NOT_RUN. Next safe action is resolving the source goal/render contract,
then implementing the separately gated family audit/dataset stage—not planning.

## Goal/render provenance audit — 2026-10-14

The investigation is complete, but the strict observation-generation gate is
**not passed**. File identity, renderer defaults and preprocessing are now much
better established; the original generation/conversion configuration is not.
`encoder_dataset_ready=false` remains intentional. No Step 3 data, Ridge probe,
intervention, H2/H3 result, commit or push was produced.

### Confirmed artifact identity and source facts

The retained `.stable-wm/pusht_expert_train.h5.zst` exactly matches the official
Hugging Face release at revision `655cd446b9929369d7d406001da85c15d1457850`:

- Compressed size: 13,136,247,974 bytes; SHA-256
  `7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318`.
- Its entire decompressed stream was byte-compared against the local HDF5,
  without creating another large file: all 46,300,921,856 bytes match.
- HDF5 SHA-256:
  `b6ebd9ac94bbe9e383f6e7a9cd92d74e9aa665ea57b758ed3717b0ee7df8d4fb`.

This resolves **published artifact identity**, not the pre-release generation
run. The [pinned dataset card](https://huggingface.co/datasets/quentinll/lewm-pusht/blob/655cd446b9929369d7d406001da85c15d1457850/README.md)
identifies the official task dataset but supplies no renderer/config/codec run
manifest. The [LeWM paper, Appendix E](https://arxiv.org/html/2603.19312v1#A5)
states that Push-T follows DINO-WM's setup/data; its summarized episode count and
length do not identify the precise 18,685-episode local artifact.

The nine public source snapshots, exact URLs, SHA-256 hashes and relevant line
numbers are recorded in `render_provenance_manifest.json` and in
`.stable-wm/phase1/step2-render-provenance-release-verified/sources/index.json`.
Key source-grounded findings:

| Source / immutable revision | Relevant lines | What it establishes |
|---|---|---|
| DINO-WM `env/pusht/pusht_env.py`, `0a9492fa12044b852ae9e001cc74604b79c8bb0c` | 369–379, 595–648, 727–735, 771–818 | Default 224 RGB output from a white 512-world Pygame canvas; light-green marker at `(256,256,π/4)`; RoyalBlue circle and LightSlateGray T; OpenCV resize; optional action overlay. |
| Early SWM `scripts/data/pusht_dataset.py`, `7124d26abef3b38afdd738e7f9f676900b46d8b8` | 286–298 | Decord decodes `obses/episode_<id>.mp4`; CHW permutation; optional transform. This implementation preserves decoded uint8 pixel scale before conversion. |
| Early SWM `scripts/data/pusht_converter.py`, same revision | 27–44, 118–123 | Copies decoded source pixels/state/proprio into **Minari**. It does not re-render those observations, and is **not proof of the later HDF5 conversion chain**. Its default action-normalization setting also prevents assuming this script alone produced the current raw-action HDF5. |
| SWM Push-T `env.py`, March revision `a00659833e13b2148663b5bfaf5e549146c6d22b` | 14–18, 155–195, 420–480, 577–582 | Default variation excludes goal/style factors; displayed goal pose comes from its separate variation values. |
| Installed SWM 0.0.6 `envs/pusht/env.py` | 248–280, 403–478, 527–528, 575–580 | Physical `goal_state`, cached task-goal image `_goal`, and displayed `goal_pose` are different quantities. `_set_goal_state` does not set `goal_pose`. |
| Installed SWM `data/dataset.py`; local `utils.py`, `eval.py`, `probe/embedding_cache.py` | 164–191; 7–11; 27–35; 24–54 respectively | Saved HWC pixels become TCHW; training, evaluation and Step 1 use float32 uint8 scaling, ImageNet normalization and bilinear/antialiased resize. |

The fetched March renderer differs from the installed file only in observation
space velocity bounds/comments; rendering code is unchanged in that comparison.
This does not identify the library versions used for original MP4 generation.
Today's weak-policy collection scripts write other dataset names and have no
traceable link to this expert release. They were not substituted as its provenance.

Installed renderer SHA-256:
`d8d0de35aaab5b846db4e79b0fbfd6b17375178cce40a25df5301c8030ca6d68`;
installed drawing helpers:
`38314dcabc556c954d64b1c2a1b558e32b48a0db092657dd745dcb960c9cf6f0`.
Runtime: SWM 0.0.6, Pymunk 7.2.0, Pygame 2.6.1, OpenCV 4.13.0.92,
stable-pretraining 0.1.6, PyTorch 2.11.0, torchvision 0.26.0, NumPy 2.2.6,
Python 3.10.20, macOS arm64. The optional release check uses already-installed
zstandard 0.25.0; no dependency was installed or changed.

### HDF5 and empirical compatibility

All root and column attributes are empty. The file contains 2,336,736 saved
224×224 RGB uint8 frames, action/state/proprio and episode/step indices, but no
goal, seed, style, variation or codec metadata. Blosc is lossless **container**
compression; it does not imply that original images were never video-compressed.

Grouping exact initial stored 7-D observations gives 185 families, each with
101 episodes. These are public-observation families, not certified original
demonstration IDs or complete physical snapshots. Repeated images from one
family are not independent physical evidence, and differing pixel hashes do
not establish deliberate appearance randomization.

The audit selects 32 families using spatial extrema and evenly spaced block
orientation order, independently of image scores. It retains the first/last
episode's initial image plus the first episode's middle/final image: 128 frames.
All 128 map uniquely to existing train/val/test cache rows with exact state and
proprio label agreement; caches are memory-mapped and never written.

Only 54 frames admit the existing contact-free, quiescent fresh-start constructor:
21 first-initial, 21 repeated-initial, 10 middle and 2 final images. The other 74
are recorded as rejected, principally due to variation bounds or initial contacts;
they are neither corrected nor resampled. Comparisons are **approximate stored-
pose illustrations**: source float32 positions/angle are reused, source velocities
and hidden block dynamics are not restored. No source trajectory replay is claimed.

The main aggregation uses only the 21 distinct valid first-initial families:

| Display condition | Mean raw-channel RMSE | Median | Range |
|---|---:|---:|---:|
| Visible, `(256,256,π/4)` | 2.794267 | 2.798026 | 1.964546–3.346019 |
| Hidden | 13.617667 | 14.137762 | 8.557390–14.317095 |
| Visible, shifted by `(48,−32)` world px | 16.716039 | 17.254213 | 13.603443–17.418662 |
| Visible, rotated by `π/4` | 15.655947 | 16.148608 | 11.733531–16.314120 |

Units are **0..255 channel intensity**, not normalized tensors or world-position
error. Pixel masks/centroids/bounding boxes use the saved 224-pixel image grid;
world coordinates belong to a 512-pixel workspace. Fixed descriptive color-mask
mean IoUs are agent 0.979462, block 0.976733 and goal 0.959041. They are not exact
segmentation labels or calibrated pose estimates. Goal-region mean RMSE is
8.295007; background mean RMSE is 1.523235. Zero green-mask area can mean occlusion,
not necessarily a hidden goal. Per-case masks, color medians, spatial residuals
and exclusions are saved. No valid comparison is byte-identical.

These results strongly support the default visible marker at the default pose
in the tested subset, but cannot uniquely attribute residuals to MP4 encoding,
rasterization/backend versions, precision or another unrecorded transformation.
No codec, histogram matching, renderer tuning or latent-similarity shortcut was
introduced to make the images appear exact.

### Changes, safeguards and measured regression results

- `counterfactual/render_provenance.py`: isolated source snapshots, full archive
  identity, deterministic family/frame selection, guarded display-only controls,
  raw/spatial comparisons, exact preprocessing parity, cache mappings, regression
  logs and conservative manifests. It never authorizes Step 3.
- `scripts/run_phase1.py`: adds only `--stage render-provenance`, parent/source
  evidence arguments and lazy dispatch; existing baseline/simulator paths remain.
- `config/phase1/pusht.yaml`: adds `simulator.provenance` with a source-default
  **comparison candidate**, explicitly `candidate_only: true`. No frozen Step 1
  or mechanics setting is changed.
- `test_counterfactual_render_provenance.py`: 10 focused offline tests for
  incompatible settings, deterministic pose/visibility, exception restoration,
  future mechanics neutrality, exact preprocessing, raw metric edge cases,
  non-pseudoreplicated selection, pinned source hashes, archive identity and
  conservative readiness. The existing simulator/model/normalizer code is untouched.

The display helper temporarily changes only `with_target`/`goal_pose`, then
restores both. It checks all physical readbacks, RNG states, physics-step count
and the original `_goal` image. Tests also compare all subsequent 35 raw actions,
substep bodies, rewards and official returned observations/info against an
environment that never ran the image audit.

All **76 direct tests across 11 scripts pass**. The full additional Step 2 suite
passes all 24 case/seed groups (72 fresh trajectories). Its replay HDF5 has
**1,918 datasets and 151 attributes exactly equal** to the reviewed historical
run. Independently executed image audits agree on all **580 HDF5 datasets and
183 attributes**, and on sample records and numeric audit summaries.

The additional CPU Step 1 recheck has finite repeated encode, prediction and
five-step rollout deltas of 0.0. All three caches match 16 sampled rows each,
maximum absolute differences 3.337861e−6 / 3.337861e−6 / 3.576279e−6 for
train/val/test. It intentionally skips CEM and remains `verification_only`,
`step1_gate_passed=false`; the original full passing Step 1 remains the parent.
There is no new official CEM success-rate claim.

An initial independent preprocessing test oracle divided by 255 and differed
from torchvision by 7.152557e−7. The oracle was corrected to torchvision's actual
float32 multiplication by the reciprocal; the frozen transform and tolerances
were not changed. This test-development failure is recorded separately in
`test_oracle_correction.json`, rather than mislabeled as a physics/model regression.

Final evidence: `diagnostic_results/phase1/step2-render-provenance-release-verified/`;
earlier audit: `step2-render-provenance-audit/`; mechanics rerun:
`step2-render-regression-replay/`; frozen recheck:
`step1-render-provenance-recheck/`. Exact commands are in the manifests and
`reproduce_phase1.md`. Large comparisons, source snapshots and logs stay under
`.stable-wm/phase1/<run_id>/`; source images/HDF5/caches were not modified.

### Final gate and conservative alternative requiring review

| Gate | Status |
|---|---|
| Fresh-start mechanics | **PASS**, no regression |
| Goal visibility | Source defaults confirmed; local dataset **empirically supported** |
| Goal pose / episode variations | Default fixed pose confirmed in source; actual generation convention **UNRESOLVED** |
| Raw-image generation compatibility | **INCONCLUSIVE**; approximate agreement, no exact generation/codec chain |
| Frozen-encoder preprocessing compatibility | **PASS**, training/eval/Step 1 parity and independent formula delta 0 |
| Encoder-facing Step 3 readiness | **FALSE** |

Strict resolution needs the original generation/configuration revision, render
dependency/MP4 encoding settings and conversion chain, or source-author records
that establish them. These are not recoverable from this HDF5's omitted fields
alone. Published artifact identity does not fill those omissions.

If those records cannot be obtained, a separately approved compatibility protocol
could freeze the installed renderer's explicit visible default marker, colors,
geometry, 512→224 OpenCV raw resize and unchanged Step 1 preprocessing, while
acknowledging a training-image domain shift. It must distinguish display marker
from task goal, preserve validated physical initialization and official goal
encoding, declare held-out family splits and pre-register compatibility checks
before observing H1/H2 outcomes. Do not fit a codec/renderer to causal results or
silently call this exact reproduction of training observations. The protocol and
its goal/render boundary require human review; **it is not activated here**.

Material Passport: academic-research-suite / experiment-agent; validation;
2026-10-14; VERIFIED for the reported file identity and executed regressions,
ANALYZED/INCONCLUSIVE for original image-generation provenance. Descriptive
diagnostics only; no inferential causal statistics or H1/H2/H3 claims.

## Independent observation compatibility — 2026-10-14 follow-up

Historical provenance remains **UNRESOLVED**, but its incompleteness is no longer
the sole gate argument. The new independent assessment finds substantial
encoder/readout sensitivity in the current raw comparison candidate. This is a
measured obstacle; no historical-record requirement is silently reinstated or
compatibility PASS manufactured.

`scripts/run_phase1.py --stage observation-compatibility` reuses saved initial
source/reconstruction images, the unchanged Step 1 encoder/preprocessing and
three frozen existing MLP readouts. It constructs no environments, changes no
goals and trains no probe. Moving middle/final pose illustrations are excluded.
The actual code is `counterfactual/observation_compatibility.py`; saved-output
plots/tables are generated by `counterfactual/compatibility_report.py`. Nine new
direct tests live in `test_counterfactual_observation_compatibility.py`.

The prewritten `observation_compatibility_protocol.md` specifies family roles,
codec grid, natural/dose references and proposed budgets. The plan's deterministic
148/18/19 family split yields 16 development and 5 validation eligible initial
families from the previous fixed sample. Protected-test query families are not
encoded or used as references. These validation families were already pixel-
audited; they are not virgin confirmatory evidence. Original cache split filenames
do not override the new family roles.

Matching is explicit: stored positions, block angle and zero public agent velocity
match the previous fresh readbacks to stored precision; full historical dynamics,
hidden contact state and pre-quantization poses are not certified. Codec variants
of one saved originating image have exactly unchanged physical state, permitting
a genuine same-state nuisance test independent of historical dynamics. No partial
7-D state is restored or claimed complete.

On validation, reconstruction/source RMS differences are latent L2 **0.686561**,
agent MLP **17.658217 world px**, block MLP **2.582461 world px**, wrapped angle
MLP **0.060704 rad**. Their descriptive one-sided 95% family-bootstrap upper
bounds are **0.879722 / 24.736982 / 3.354508 / 0.079153**. Development-only natural
same-state upper margins are **0.087971 / 0.731627 / 0.242617 / 0.054719**.
All envelope checks fail. Proposed pilot budgets 1.25 px / 1.25 px / 0.025 rad
also fail, even doubled. For confirmatory 1-px block effects the stricter proposed
0.25-px per-observation budget fails too. These allocations are linked to the
plan's physical error/effect budgets, not an arbitrary pixel RMSE threshold.

The development median latent signal for near-10-px fixed-block source motion
is **0.496154**, from only three qualifying families: reconstruction nuisance
is not small against that contextual scale. Missing 10/20/30-px references are
logged, not replaced. Actual Ridge injection norms are not measured here; they
belong to the later intervention gate, not a circular prerequisite for initial
observation-contract work.

Original source encodings still match their frozen caches (21 samples, maximum
absolute difference **4.053116e−6**); repeated encoding delta is zero. Absolute
agent readout RMS against source labels worsens **7.860849 → 18.969322 world px**.
The observed drift is not an accidental checkpoint/preprocessing mismatch, but
legacy probe provenance/domain sensitivity limits semantic interpretation. These
are readout-conditioned results, not proof that LeWM lacks physical information.

Residuals are spatially structured. On validation, background/agent/block/goal
mean region RGB RMSE is **1.933888 / 17.395052 / 8.324010 / 8.227999**; respective
squared-residual energy shares are **44.00% / 10.23% / 25.32% / 20.45%**. Much
object error lies on edges, with additional signed interior color biases.
Background includes walls. Fixed libx264 CRF 18/23/28 at yuv420p/yuv444p produces
same-state latent RMS shifts **0.370560–0.936133** and agent MLP shifts
**4.320096–13.293413 px**; exact RGB FFV1 controls produce zero pixel/latent/readout
drift. Codec/color/chroma effects are therefore testable plausible contributors,
but do not uniquely explain or reproduce historical residuals. All conditions
are reported; no codec or production correction is selected. Original temporal
MP4/decoder/rasterization/subpixel effects remain unresolved.

**Decisions:** the current combination does not support a compatibility-qualified
limited Step 3 pilot, and does not support confirmatory observation-contract
freeze. A separately approved observation/readout-calibration diagnostic could
be useful, but must remain labeled incompatible/OOD development. The candidate's
central marker is not automatically the current mechanics fixture's safe-goal
marker; prospective semantics require review, not a hidden goal change.

The proposed independent route, remaining empirical requirements and non-circular
gate sequence are in `observation_compatibility_gate_revision.md`. It removes
exact historical reconstruction as a mandatory condition in principle; it is
**proposed, not activated**. Existing `encoder_dataset_ready=false` remains.

Evidence: `diagnostic_results/phase1/step2-observation-compatibility-verified/`
contains manifest, memo, calibrated margins, per-case/spatial/codec tables and two
diagnostic figures. Arrays/codec files/commands/source snapshots/logs are under
the corresponding `.stable-wm/phase1/` directory. Independent executions agree
exactly on seven HDF5 datasets, two root attributes and numeric reports.
All **85 tests across 12 direct scripts pass**, including unchanged Step 1/2
mechanics regressions. No Step 3 data, Ridge intervention, H2/H3, commit or push.

## Natural observation-domain follow-up v2 — completed, scientific gate failed

The latest task authorized a bounded natural readout/predictor diagnostic, not
formal Step 3 collection. The unchanged Step 2 safe goal/marker `(350,120,0)`
was tested, separately from the previous central-marker comparison candidate.
Sixteen new development families calibrated margins before sixteen reserved
validation families were evaluated. All four doses and duplicate controls stay
within their physical family; no invalid family or poor outcome was replaced.

All mechanics/repeated inference checks pass exactly. Independent validation
agent differential RMS is **8.981/10.253/15.957 world px** at doses 10/20/30,
exceeding the unchanged 5-px cap. Dose-zero absolute RMS is **57.251 px agent,
14.909 px block, 0.501 rad angle**. Stable dose latents and repeatable RGB do not
make these instruments sufficiently accurate. Natural teacher-forced/open-loop
ceilings were measured under the correct seven-token contract, with source
comparisons labeled unmatched in state/goal/action population.

**A PASS; B INCONCLUSIVE; C FAIL; D INCONCLUSIVE; E FAIL.** Both compatibility-
qualified Step 3 development readiness and confirmatory contract readiness are
false. Exact historical provenance remains UNRESOLVED; missing records are not
the sole blocker. A separate versioned natural-state readout refit is proposed
in `readout_refit_proposal_v1.md`, requiring approval before fitting.

97 tests across 13 scripts pass. Independent-process replay matches all 2,614
HDF5 datasets and 163 attributes, with identical file hash and numerical reports;
this is reproducibility, not more independent samples. Detailed evidence and
four diagnostic figures: `diagnostic_results/phase1/step2-domain-validation-v2/`.
See its `domain_assessment_memo.md` and `observation_domain_gate_revision_v2.md`.
No model/readout/physics/goal/preprocessing/action/CEM changes, formal Step 3,
latent patch, Ridge fit, H2/H3, commit or push were performed.

## Authorized readout repair R0/R1 — completed, no gate promotion

The subsequent 2026-10-15 task authorized an input/label audit followed by a
bounded natural-state fit. R0 verifies geometry/body-origin targets, RGB/resize,
preprocessing, legacy MLP architecture/normalization and image/label identity;
no original implementation bug or scientific-contract correction was found.
At 21 matched stored poses, source versus prospective agent RMSE is
5.663 versus 55.563 world px. Full historical dynamics/render provenance is
not certified and remains UNRESOLVED. A new audit-test fixture failure was
preserved, corrected to the existing physical tolerance, and rerun successfully.

R1 uses 48 training and 16 independent pilot-validation families; every related
dose/zero control stays grouped. The pilot selected penalties/epochs and is now
exposed development evidence. New Ridge and MLP256 heads are versioned under
ignored `.stable-wm/phase1/readout-r1-feasibility-v1/readouts/`, with original
weights untouched. Position absolute/differential errors remain inadequate;
sin/cos MLP absolute-angle upper bounds meet the candidate .10-rad cap but do
not establish small-angle resolution. No tested block scale1/2/5/10 qualifies
combined resolution and non-target preservation. No official H1 Ridge/P was
created/replaced. The optional .5-px high-resolution objective is not universal;
the original Step3>=1px feasibility threshold remains unchanged.

118 tests across15 scripts pass. An independent full execution matches142 HDF5
datasets,259 attributes and all72 new model artifacts exactly. Readout accuracy
does not validate predicted/patched latents or dynamics. R2 development is
recommended conditionally for approval, not implemented. Formal Step3/H2/H3,
commits/pushes and automatic instrument promotion remain unauthorized.
Evidence: the R0 audit memo and R1 research memo in their respective
`diagnostic_results/phase1/readout-r0-sanity-verified-v1/` and
`readout-r1-feasibility-v1/` directories; proposed next protocol:
`readout_refit_proposal_v2.md`.
