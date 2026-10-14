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

## Step 2 — Goal/render provenance audit (no Step 3 progression)

Run the focused tests, then the audit through the same staged CLI:

```bash
MPLCONFIGDIR=.matplotlib-cache .venv/bin/python test_counterfactual_render_provenance.py

# First source investigation: fetch nine small, pinned public source snapshots.
MPLCONFIGDIR=.matplotlib-cache .venv/bin/python scripts/run_phase1.py \
  --stage render-provenance --run-id step2-render-provenance-audit

# Offline reuse of the exact source snapshots, with hash/URL checks:
MPLCONFIGDIR=.matplotlib-cache .venv/bin/python scripts/run_phase1.py \
  --stage render-provenance --run-id step2-render-provenance-release-verified \
  --source-evidence .stable-wm/phase1/step2-render-provenance-audit/sources/index.json

# Additional full mechanics and frozen model/cache regression checks:
MPLCONFIGDIR=.matplotlib-cache .venv/bin/python scripts/run_phase1.py \
  --stage simulator --run-id step2-render-regression-replay
MPLCONFIGDIR=.matplotlib-cache .venv/bin/python scripts/run_phase1.py \
  --stage baseline --run-id step1-render-provenance-recheck --device cpu --skip-cem
```

Those IDs are existing immutable evidence; choose new IDs when reproducing.
`--simulator-manifest <path>` selects a different passing Step 2 parent;
`--baseline-manifest <path>` selects a different full passing Step 1 parent.
The audit validates unchanged parent inference/physics sources and checkpoint,
records the dirty Git tree plus exact source snapshots, and refuses reused roots.
It runs all 11 direct-execution regression scripts and saves commands/exit codes
and complete stdout/stderr under its ignored artifact directory.

The optional retained `.h5.zst` archive is checked against the pinned official
HF LFS SHA-256 and completely decompressed/byte-compared with the local HDF5.
This uses installed zstandard 0.25.0, streams data, writes no second HDF5 and
does not download the 13-GB archive. If the archive is absent, the identity check
reports UNAVAILABLE; it does not infer identity from the filename. The full
comparison reads roughly 59 GB and can take a few minutes on slower storage.

Audit outputs under `diagnostic_results/phase1/<run_id>/`:

- `render_provenance_manifest.json`: parent/source/release identities, actual
  candidate config, runtime, audit metrics, regressions, blockers and separate
  gate statuses. Completed audit means the investigation ran successfully,
  **not** that the observation-generation gate passed.
- `initial_families.json`: all 185 exact initial-public-observation families and
  episode membership; no claim of complete physical state or original demo IDs.
- `sample_records.json`: 128 predetermined frames from 32 families, raw-row
  hashes, stored states, all reconstruction exclusions and per-condition metrics.
- `cache_frame_mappings.json`: exact existing cache row/split mappings and
  state/proprio label checks for all 128 sampled observations.
- `image_metrics.csv`: raw-channel RMSE/MAE, goal/non-goal/background regions
  and descriptive color-mask IoUs; repeated initial images are not independent
  families in the main aggregate.
- `comparison_<raw_row>.png`: source, visible, hidden, shifted and rotated
  candidates plus an explicitly amplified absolute-error panel. These are
  provenance diagnostics, **not** the four future H1/H2 research figures.

Large/readback/source artifacts under `.stable-wm/phase1/<run_id>/`:
`render_comparisons.h5`, `source_snapshot.h5`, `sources/*.txt`/`index.json`, and
`regression_logs/`. Render comparisons include saved original images/states,
candidate images, goal-region masks, RGB residuals and numeric metrics.
No original HDF5 or `.pt` cache is edited, rebuilt or overwritten.

The source-grounded display candidate is explicitly `candidate_only: true`
under `simulator.provenance`. The mechanics fixtures still use their original
safe physical goal. Comparison controls temporarily affect display only and
restore it; they do not change task goals, cached goal images, reset/physics,
reward, goal encoding, model, action normalization or time indexing.
Middle/final source poses are only attempted as independent quiescent image
illustrations; invalid/contact/bounds cases are rejected without resampling.
There is no arbitrary dynamic-state restore or source-trajectory replay.

The complete release identity check now passes, and training/evaluation/Step 1
preprocessing agrees exactly. The original generation/MP4 encoding/conversion
run and per-episode goal/style provenance remain unknown. Therefore all audits
keep `verified_generation_contract=false` and `encoder_dataset_ready=false`.
See the new provenance section of `step2_review.md` for the measured results,
source lines and a conservative compatibility protocol requiring human review.
**Stop here; do not start Step 3 or infer approval from an audit exit code of 0.**

The independent-process numeric/attribute comparisons recorded in
`step2-render-provenance-release-verified/reproducibility_checks.json` can be
rerun read-only:

```bash
.venv/bin/python - <<'PY'
import h5py, numpy as np
pairs = [
    ('step2-fresh-start-replay-reviewed', 'step2-render-regression-replay', 'simulator_replay.h5'),
    ('step2-render-provenance-audit', 'step2-render-provenance-release-verified', 'render_comparisons.h5'),
]
for left, right, filename in pairs:
    with h5py.File(f'.stable-wm/phase1/{left}/{filename}') as a, h5py.File(f'.stable-wm/phase1/{right}/{filename}') as b:
        names, other = [], []
        a.visit(names.append); b.visit(other.append)
        assert names == other
        datasets = attributes = 0
        for name in ['', *names]:
            x, y = (a[name], b[name]) if name else (a, b)
            assert set(x.attrs) == set(y.attrs)
            for key in x.attrs:
                assert np.array_equal(x.attrs[key], y.attrs[key]), (name, key)
            attributes += len(x.attrs)
            if isinstance(x, h5py.Dataset):
                assert x.shape == y.shape and x.dtype == y.dtype
                assert np.array_equal(x[()], y[()]), name
                datasets += 1
        print(filename, 'PASS', datasets, 'datasets', attributes, 'attributes')
PY
```

## Step 2 — Independent encoder-facing compatibility assessment

This assessment is separate from historical provenance and never activates a
collection gate. It reads saved initial-image comparisons only, applies the
unchanged frozen encoder/preprocessing and existing frozen MLP readouts, and
runs fixed **offline** codec controls. No environment or goal is modified.

Required parents: the passing original Step 1, passing Step 2 and completed
`step2-render-provenance-release-verified` audit, including its ignored comparison
HDF5 plus compact family/sample/cache-mapping records. The released checkpoint,
three root MLP checkpoints, original HDF5 and existing caches must be available.
No model/probe/cache is retrained or overwritten. Bundled imageio-ffmpeg 7.1
supplies libx264/FFV1; no system FFmpeg installation or new dependency was needed.

```bash
MPLCONFIGDIR=.matplotlib-cache .venv/bin/python test_counterfactual_observation_compatibility.py
MPLCONFIGDIR=.matplotlib-cache .venv/bin/python scripts/run_phase1.py \
  --stage observation-compatibility --run-id step2-observation-compatibility-v1 --device cpu
MPLCONFIGDIR=.matplotlib-cache .venv/bin/python scripts/run_phase1.py \
  --stage observation-compatibility --run-id step2-observation-compatibility-verified --device cpu
```

Those IDs are saved evidence; choose new IDs for reproduction. An alternative
completed audit can be supplied with `--provenance-manifest <path>`. The original
protocol and numerical criteria were written before the first measurement; the
verified run adds cache-identity/end-of-run guards and reporting, with unchanged
case measurements/criteria. Runs agree exactly on saved images, embeddings,
readouts, reference bank and calibrated numerical reports.

Compact outputs: `compatibility_manifest.json`, `family_partition.json`,
`development_calibration.json`, `case_metrics.json`/`.csv`, `spatial_metrics.json`,
`spatial_summary.csv`, `codec_summary.csv`, `spatial_residuals.png`,
`encoder_readout_drift.png` and `compatibility_memo.md`. Large arrays, the 147
codec diagnostic files, exact codec commands, source snapshots and regression
logs live under `.stable-wm/phase1/<run_id>/`.

Regenerate figures/tables from saved numeric records only, into a new temporary
output directory so immutable evidence is not overwritten:

```bash
MPLCONFIGDIR=.matplotlib-cache .venv/bin/python - <<'PY'
from pathlib import Path
from tempfile import mkdtemp
import shutil
from counterfactual.compatibility_report import write_compatibility_plots
source = Path('diagnostic_results/phase1/step2-observation-compatibility-verified')
destination = Path(mkdtemp(prefix='lewm-compatibility-plots-'))
for name in ['case_metrics.json', 'spatial_metrics.json',
             'development_calibration.json', 'compatibility_manifest.json']:
    shutil.copy2(source / name, destination / name)
write_compatibility_plots(destination)
print(destination)
PY
```

The historical HDF5 comparison has render-relevant/public-state matching, not
full historical-state proof. Middle/final pose illustrations and protected-test
query families are excluded. Natural repeats are aggregated within physical
family, and all missing dose references are reported. The existing test cache
filename is not synonymous with the new protected family partition.

Process completion/regression success does not mean scientific compatibility.
The measured natural-nuisance and proposed practical checks **fail**. Neither a
compatibility-qualified Step 3 pilot nor confirmatory freeze is supported for
the present candidate/readout combination. Read the separate, review-required
`observation_compatibility_gate_revision.md`; it proposes an independent empirical
route without making unrecoverable historical records a permanent prerequisite.
No full Step 3 collection or gate activation follows this assessment.
# Natural observation-domain diagnostic v2 (not formal Step 3)

The task authorized a bounded calibration diagnostic and natural predictor
baseline. Original goals/physics/weights/contracts and all old artifacts remain
unchanged. See `observation_domain_protocol_v2.md`,
`observation_domain_gate_revision_v2.md`, and
`diagnostic_results/phase1/step2-domain-validation-v2/domain_assessment_memo.md`.
No readout refit, formal Step3, H1/H2/H3, commit or push was performed.

Executed from repository root (macOS arm64 CPU). The two cache environment
variables only place Matplotlib/font caches in writable temporary directories:

```bash
MPLCONFIGDIR=/private/tmp/lewm-domain-mpl XDG_CACHE_HOME=/private/tmp/lewm-domain-cache .venv/bin/python test_counterfactual_domain_diagnostics.py
MPLCONFIGDIR=/private/tmp/lewm-domain-mpl XDG_CACHE_HOME=/private/tmp/lewm-domain-cache .venv/bin/python scripts/run_phase1.py --stage domain-diagnostics --partition development --run-id step2-domain-calibration-dev-v2 --device cpu
MPLCONFIGDIR=/private/tmp/lewm-domain-mpl XDG_CACHE_HOME=/private/tmp/lewm-domain-cache .venv/bin/python scripts/run_phase1.py --stage domain-diagnostics --partition validation --calibration-manifest diagnostic_results/phase1/step2-domain-calibration-dev-v2/domain_manifest.json --run-id step2-domain-validation-v2 --device cpu
MPLCONFIGDIR=/private/tmp/lewm-domain-mpl XDG_CACHE_HOME=/private/tmp/lewm-domain-cache .venv/bin/python scripts/run_phase1.py --stage domain-diagnostics --partition validation --calibration-manifest diagnostic_results/phase1/step2-domain-calibration-dev-v2/domain_manifest.json --run-id step2-domain-validation-replay-v2 --device cpu
MPLCONFIGDIR=/private/tmp/lewm-domain-mpl XDG_CACHE_HOME=/private/tmp/lewm-domain-cache .venv/bin/python diagnostic_results/phase1/step2-domain-validation-v2/summarize_saved.py
```

Outputs are immutable; do not rerun these exact IDs. To reproduce, choose new
development/validation/replay IDs and pass the new development parent. Validation
rejects code/config/protocol/readout/runtime changes after calibration. Saved
source snapshots reproduce the exact historical run; later source additions
require a new development parent, even when the scientific protocol is unchanged.
The saved-summary script is post-assessment descriptive analysis, not criterion
selection. Adjust its input paths for a newly named reproduction series.

Each stage runs 97 direct tests in13 scripts. The frozen production path passes
repeatability; GateC fails physical readout budgets. A successful process exit
means evidence collection completed, **not a scientific PASS**. GateA PASS,
B INCONCLUSIVE, C FAIL, D INCONCLUSIVE, E FAIL. Both Step3 readiness decisions
are false. Independent-process replay matches2,614 datasets/163 attributes;
it is not additional independent sample size.

Saved numerical files: `domain_manifest.json`, `reserved_families.json`,
`family_metrics.json`, `attempts.json`, calibration/freeze records, CSVs and four
diagnostic PNGs. Development also saves signed historical-domain diagnosis and
21 correctly standardized source predictor references. Large full raw/substep
trajectories/RGB/latents/predictions, codec stress artifacts, source snapshots and
regression logs are in `.stable-wm/phase1/<run_id>`. This is a32-family readout
diagnostic, not a formal AB/ABC Step 3 dataset or the four completed H2 figures.

## Readout repair R0/R1 — authorized natural-state feasibility only

`readout_repair_protocol_r0_r1.md` predeclares the audit, family allocation,
candidate set, seeds and proposed measurement-scale criteria. Exact historical
provenance remains UNRESOLVED. No original LeWM/physics/goal/action/temporal/CEM
contract was changed; only separately versioned readout heads were fitted.
R0 must complete and pass before R1 accepts its `--audit-manifest` parent.
Source/config/protocol/runtime identities are checked; original MLPs are frozen.

Executed commands from the repository root:

```bash
MPLCONFIGDIR=/private/tmp/lewm-domain-mpl XDG_CACHE_HOME=/private/tmp/lewm-domain-cache .venv/bin/python test_counterfactual_readout_audit.py
MPLCONFIGDIR=/private/tmp/lewm-domain-mpl XDG_CACHE_HOME=/private/tmp/lewm-domain-cache .venv/bin/python test_counterfactual_readout_feasibility.py
MPLCONFIGDIR=/private/tmp/lewm-domain-mpl XDG_CACHE_HOME=/private/tmp/lewm-domain-cache .venv/bin/python scripts/run_phase1.py --stage readout-audit --run-id readout-r0-sanity-v1 --device cpu
MPLCONFIGDIR=/private/tmp/lewm-domain-mpl XDG_CACHE_HOME=/private/tmp/lewm-domain-cache .venv/bin/python scripts/run_phase1.py --stage readout-audit --run-id readout-r0-sanity-verified-v1 --device cpu
MPLCONFIGDIR=/private/tmp/lewm-domain-mpl XDG_CACHE_HOME=/private/tmp/lewm-domain-cache .venv/bin/python -c "from counterfactual.readout_repair_report import write_r0_report; write_r0_report('diagnostic_results/phase1/readout-r0-sanity-verified-v1')"
MPLCONFIGDIR=/private/tmp/lewm-domain-mpl XDG_CACHE_HOME=/private/tmp/lewm-domain-cache .venv/bin/python scripts/run_phase1.py --stage readout-feasibility --audit-manifest diagnostic_results/phase1/readout-r0-sanity-verified-v1/repair_manifest.json --run-id readout-r1-feasibility-v1 --device cpu
MPLCONFIGDIR=/private/tmp/lewm-domain-mpl XDG_CACHE_HOME=/private/tmp/lewm-domain-cache .venv/bin/python scripts/run_phase1.py --stage readout-feasibility --audit-manifest diagnostic_results/phase1/readout-r0-sanity-verified-v1/repair_manifest.json --run-id readout-r1-feasibility-replay-v1 --device cpu
MPLCONFIGDIR=/private/tmp/lewm-domain-mpl XDG_CACHE_HOME=/private/tmp/lewm-domain-cache .venv/bin/python diagnostic_results/phase1/readout-r1-feasibility-v1/summarize_saved.py
```

The first R0 run preserves a regression failure in the **new** floating-position
test fixture. The fixture was corrected to the existing `1e-10` state tolerance;
no original pipeline bug or physics change was found. The passing R0 uses the
next immutable ID. Do not reuse any saved ID; choose new audit/fit/replay IDs.
For a new series, update the summary script's input paths. To replay the exact
historical execution after future code edits, use its saved source/config
snapshots in a separate checkout; the current guards intentionally reject
changed parents, rather than relabel changed code as the same experiment.

R0 records170 images and149 independent-geometry checks, exact preprocessing/
MLP-forward parity and source/prospective pose-matched comparisons. R1 collects
64 families,21 scenes per family (zero-repeat excluded from fitting):48 training,
16 pilot-validation;1344 images are not1344 independent samples. The pilot is
used for bounded selection and becomes exposed development evidence. Search:
nested16/32/48 counts,4 Ridge penalties,4 target heads,2 MLP seeds with fixed
architecture/schedule. All75 comparison rows and72 new artifacts are retained.
No later final families have been generated.

Implementation files: `counterfactual/readout_audit.py` (contracts/geometry/legacy
pipeline audit), `readout_feasibility.py` (natural-family collection, grouped fits,
versioned loaders/metrics/criteria), `readout_repair_report.py` (saved-number
tables/plots), two direct test scripts, CLI stages and nested config. No new
third-party dependency. Target statistics are explicitly training-only weighted
population statistics for new heads, not a rewrite of legacy transformations.

Compact outputs are in `diagnostic_results/phase1/readout-r0-sanity-verified-v1/`
and `readout-r1-feasibility-v1/`: manifests, lineage/attempts, complete JSON/CSV,
geometry figure, four R1 diagnostic figures, audit/research memos and numerical
reproducibility checks. Large HDF5/model files and regression/source snapshots
remain ignored under `.stable-wm/phase1/<run_id>/`. The saved-summary script
uses existing arrays/predictions only; it adds descriptive strata/bias summaries,
the fourth plot and replay checks, without changing criteria or model selection.

All118 tests in15 scripts pass, including existing Step1/2 regressions. Independent
R1 executions agree exactly on142 HDF5 datasets,259 attributes,72 model files,
assignments/curves and feasibility decisions. Original checkpoint/readout hashes
and frozen model status are checked. Repetition is verification, not more N.

**Scientific outcome:** R0 pipeline valid; R1 position accuracy/preservation
inadequate and no tested block scale qualified. Some selected-pilot absolute
angle checks pass, not an overall gate. Read `readout_refit_proposal_v2.md`:
R2 is proposed, not authorized. No official readout/P promotion, formal Step3,
patches, H2/H3, commits or pushes follow these diagnostic stages.
