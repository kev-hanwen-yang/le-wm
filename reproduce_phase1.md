# Reproducing Phase I — Frozen Baseline and Fresh-start Replay

Step 1 verifies the frozen released LeWM checkpoint, the exact Push-T action
contract used during training, the three-frame time convention, and the identity
of the natural-latent caches. It does not generate counterfactual scenes, patch
latents, run H2, or evaluate planning interventions.

Run from the repository root with the project environment active:

```bash
.venv/bin/python test_counterfactual_contracts.py
.venv/bin/python test_counterfactual_baseline.py

.venv/bin/python scripts/run_phase1.py \
  --stage baseline \
  --run-id step1-reviewed-baseline-seed42
```

For a fast contract/cache check that skips the official CEM smoke evaluation:

```bash
.venv/bin/python scripts/run_phase1.py \
  --stage baseline \
  --run-id step1-reviewed-contracts-seed42 \
  --skip-cem
```

The baseline creates:

- `diagnostic_results/phase1/<run_id>/baseline_manifest.json`: configuration,
  source revision/status and source hashes, pinned checkpoint hash, deterministic
  settings, versions, action statistics, model shapes, cache checks, and gate status.
- `diagnostic_results/phase1/<run_id>/baseline_summary.md`: concise human-readable
  gate result.
- `.stable-wm/phase1/<run_id>/cem_smoke.txt`: output from the existing official
  evaluator, plus its Hydra run directory under the same artifact root.
- `.stable-wm/phase1/<run_id>/frozen_rollout.h5`: the raw smoke window, standardized
  tokens, embeddings, one-step predictions, and both five-step rollouts.
- `.stable-wm/phase1/<run_id>/source_snapshot.h5`: exact source/config bytes with
  paths and SHA-256 attributes, including uncommitted code and inspected dependency
  implementations. This complements the git SHA and package-version record.
- `diagnostic_results/phase1/<run_id>/cem_metrics.json`: actual episode IDs/start
  steps, per-episode outcomes, solver/device/config observations, evaluation time,
  and official evaluator action statistics. stdout/stderr are saved beside it.
- `diagnostic_results/phase1/<run_id>/failure_traceback.txt`: diagnostic traceback
  when a started run fails. Partial manifests remain available during execution.

The command refuses any run ID already present in either output root. There is
no overwrite switch. Run IDs are single path components. `--skip-cem` reports
`verification_only` and `step1_gate_passed=false`; a full gate requires valid
official metrics for all ten episodes. A zero child-process exit code alone is
insufficient. CEM has a recorded 1800-second timeout and writes live logs.

The checkpoint must match the SHA-256 frozen in `config/phase1/pusht.yaml`. The
loaded architecture must agree with the 3-frame, 10-action-coordinate, 192-latent
contract. Both inference passes must be finite and equal through every predicted
step at h=15,20,25,30,35. Inputs/outputs are persisted in HDF5 for inspection.

Cache validation checks all row shapes, valid episode/step indices, unique raw
rows and complete membership in the existing 80/10/10 episode split. It samples
16 rows per split with endpoints and fixed-seed stratified interior rows,
checks their physical labels, then compares fresh frozen embeddings using
`atol=1e-5`, `rtol=1e-4`. This establishes sampled identity; it does not certify
every cached latent. Existing large `.pt` caches are memory-mapped.

An embedding mismatch stops the default run. If repair is explicitly requested:

```bash
.venv/bin/python scripts/run_phase1.py --stage baseline \
  --run-id step1-reviewed-repair-seed42 --rebuild-mismatched-caches
```

Only caches with valid metadata/indices/labels and mismatching embeddings are
re-encoded. Repair streams all rows into `.stable-wm/phase1/<run_id>/<split>_encoded.h5`,
preserves the original cache, revalidates the result, and records both validations.
Missing or structurally invalid caches halt; rebuilding them requires resolving
their schema/split provenance first. Matching caches are reused without writes.

The CEM smoke invokes the repository's existing `eval.py` with its cost, goal
handling, physics, and metrics. The configured cache/dataset/image size are
forwarded as quoted Hydra values, so paths containing spaces work. The chosen
device is supplied through `eval.py`'s actual `eval_device()` hook because its
call to `hydra.utils.instantiate(..., device=device)` overrides `solver.device`.
Default device selection reuses the existing CPU/CUDA/MPS detection; `--device`
sets both the contracts check and the official model/solver placement.

The wrapper records evaluation inputs and returned metrics without changing
them. Python/NumPy/PyTorch RNGs and the CEM generator are seeded, and PyTorch
deterministic algorithms are enabled. The HDF5 dataset has no `seed` column:
the official evaluator calls `World.reset(seed=None)`, which can use OS entropy.
Consequently, this is a seeded model/CEM functionality smoke, not proof of
deterministic simulator replay. `_set_state` also omits block velocities and
steps physics. Step 2 must validate physical initialization separately.

Phase I action tokens reproduce `utils.get_column_normalizer`: `torch.std`
uses the sample standard deviation. Official CEM retains its existing sklearn
`StandardScaler` population deviation. Their measured statistics are recorded
separately; the official preprocessing is preserved for baseline comparability.

The original 9/10 smoke at `step1-baseline-seed42-cpu` remains a historical record.
New runs use manifest schema version 2. See `step1_review.md` for review findings,
changes, verification, and remaining scientific limitations.

## Step 2 — Fresh-start simulator replay validation

Step 2 runs actual Push-T physics, not model prediction or planning. It requires
a completed passing Step 1 manifest, the same checkpoint/runtime and unchanged
frozen model/physics sources. Only the staged CLI and addition of the simulator
config section are permitted extensions of the parent protocol. The pre-existing
Step 1 settings remain unchanged. Cache identity remains the sampled guarantee
documented above; no stronger whole-dataset identity claim is introduced here.

```bash
MPLCONFIGDIR=.matplotlib-cache .venv/bin/python test_counterfactual_simulator.py
MPLCONFIGDIR=.matplotlib-cache .venv/bin/python test_counterfactual_simulator_stage.py

MPLCONFIGDIR=.matplotlib-cache .venv/bin/python scripts/run_phase1.py \
  --stage simulator --run-id step2-fresh-start-replay-reviewed

# Independent process, same scientific configuration, new output identifier:
MPLCONFIGDIR=.matplotlib-cache .venv/bin/python scripts/run_phase1.py \
  --stage simulator --run-id step2-fresh-start-replay-repeat
```

Use a new run ID if those records already exist. An alternative parent must be
explicitly supplied with `--baseline-manifest <path>`. Partial/verification-only
parents, changed frozen sources/checkpoint/config/runtime, unsafe IDs, reused
output roots and baseline-only flags are rejected. A failed started run retains
its manifest, completed/failed case records, partial HDF5 and traceback. The
configured 300-second timeout is checked during raw-step replay; no failed case
is automatically resampled. Native-library stalls inside an individual call
cannot be preempted by this cooperative in-process deadline.

The eight fixed mechanics fixtures run with three explicit seeds: stationary,
free motion, near-contact hold, contact push, push-then-withdraw/post-contact,
block–wall collision, rotated stationary block and an angle near the 2π branch.
They are validation cases, **not** a sampled AB/ABC dataset. Each executes 35
recorded raw commands, with ten initial zero raw commands. Every fixture is run
twice in fresh audited environments and once in the unmodified `PushT` class.
The third trajectory checks that logging does not change physics, RNGs, rewards,
goal/controller fields or images. Goal-display on/off is separately covered by
unit tests; the recorded suite uses the installed default `with_target=True`.

Initialization calls the installed reset exactly once, with no random variation
sampling, an explicit goal and matched goal-render pose. That reset performs two
physics steps. Both are logged and checked; the experiment defines h=0 **after**
reset. No correction, velocity-zeroing, settling step, physics override or
planner/metric modification follows reset. Invalid starts are rejected.

Each trajectory records 36 raw readbacks/images (h=0..35) and, for audited runs,
352 physical readbacks (two reset plus 350 rollout substeps). The 18 body fields
include each body's position, angle, velocity, angular velocity, force and
torque. Other records include goal state/pose, public observation/proprio,
latest raw command, derived relative controller target, raw/physics timestamps,
solver/geometry/style constants, RNG identities, copied typed contact events,
original direct contact counts, rewards, termination flags and image hashes.
Float64 physical fields must agree with `atol=1e-10`, `rtol=0`; integer/event/RNG
and pixel records must agree exactly on this audited backend. No incompatible
units are combined into a physical score. Seven standardized five-action tokens
are saved alongside the commands using the parent training statistics.

The collision observer delegates the installed post-solve handler first, then
copies event data rather than retaining live arbiter references. Agent–block
events are distinguished from block–wall events. Kinematic pusher versus static
wall collisions need not generate post-solve events in this backend; geometric
initial-wall clearance is checked separately. Official `info['n_contacts']`
normalization is not used or altered.

Output locations:

- `diagnostic_results/phase1/<run_id>/simulator_manifest.json`: gate, immutable
  parent hash, versions, source/binary/checkpoint identities, per-case comparisons,
  physical scope, unsafe-restore evidence and unresolved training-render status.
- `simulator_summary.md` and `simulator_cases.csv` in that same directory:
  human-readable and numeric case results, including failures.
- `.stable-wm/phase1/<run_id>/simulator_replay.h5`: both audited trajectories,
  unmodified reference, actions/tokens, substeps, contacts and dynamic-restore
  counterexamples. These are **audited readbacks, not complete snapshots**.
- `source_snapshot.h5`: exact harness/config/test/plan and inspected dependency
  source bytes. The compiled Chipmunk binary is hashed, not copied into Git.
- `dataset_render_audit.h5`, `training_render_sample.png` and
  `fresh_render_sample.png`: sampled source-image evidence and fresh-render preview.

Two negative diagnostics take moving/free-space and post-push observations at
h=20, apply the installed `_set_state` in a fresh environment and compare the
readback and remaining raw-step trajectory with its original continuation. The
extra physics step and missing block dynamics make public 7-D restoration
unsuitable. The production `restore_dynamic_state` entry point deliberately
raises an error. Pymunk also documents that copied spaces omit some collision
cache data; copying/pickling is not presented as exact restoration:
[Pymunk 7.2 Space documentation](https://www.pymunk.org/en/7.2.0/pymunk.html#pymunk.Space).

`step2_gate_passed=true` means **fresh-start physical replay and rendering under
the declared configuration** passed. It does not prove renderer identity with
the training data, arbitrary moving/contact restoration, or any causal H2 claim.
The source HDF5 lacks goal/style/seed provenance; sampled color counts do not
resolve it. Until the source rendering/goal convention is recovered or a
separately specified protocol is explicitly approved, `encoder_dataset_ready`
stays false. Do not silently disable goals or advance to encoder-facing Step 3.

The package's native SDL duplicate-class/deprecation warnings are preserved in
logs; this implementation does not modify installed dependencies to hide them.
`MPLCONFIGDIR` selects a writable, ignored Matplotlib cache, not simulator physics.
Reproducibility compares numeric trajectories/pixel hashes and protocol identity,
not run IDs, output paths, timestamps or elapsed time.
