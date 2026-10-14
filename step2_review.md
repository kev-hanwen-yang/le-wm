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
