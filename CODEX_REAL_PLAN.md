# Phase I LeWM/Push-T Counterfactual Study

## Summary and repository audit

Phase I is feasible with a small, isolated evaluation package. LeWM, Push-T physics, probes, encoder caches, rollout alignment, manifold diagnostics, and CEM can largely be reused. The frozen model and simulator must be wrapped more carefully before causal data are collected.

| Area | Verified state | Reuse / required change |
|---|---|---|
| Model | [jepa.py](jepa.py) exposes `encode`, `predict`, `rollout`, and final-latent squared-L2 planning cost. [module.py](module.py) provides the six-layer conditional `ARPredictor`. Latents are 192-D with three-frame predictor context. | Reuse unchanged and frozen. |
| Checkpoint | `.stable-wm/pusht/lewm_object.ckpt`, SHA-256 `100d6459…a2553`; git revision `2b15172c…e27`, branch `main`. Repeated CPU encoding/prediction was bit-identical. | Record hashes and versions in every manifest. |
| Time alignment | [probe/rollout_windows.py](probe/rollout_windows.py) correctly maps context frames to `h=0,5,10`, action blocks of five, and predictions to `h=15…35`. | Reuse indexing and tests. |
| Action preprocessing | Training standardizes each raw 2-D action before flattening five actions into a 10-D token. Existing `probe/rollout_ablation.py` instead sends raw actions directly to `action_encoder`. On one real window this changed the final predicted latent by L2 `3.48` and max coordinate `0.74`. | New Phase I code must use exact training statistics: mean `[-0.007812565, 0.006860687]`, std `[0.208467484, 0.206748664]`. Do not reuse old rollout artifacts for Phase I. Preserve the historical diagnostic path and flag its mismatch separately. |
| Simulator | Installed Push-T’s public 7-D state omits block velocity and angular velocity; `_set_state` also advances physics. Fresh, quiescent reconstruction produced exact state readback and identical 35-step trajectories in repeated environments. | Use fresh environments and zero-velocity initial scenes. Explicitly reject arbitrary 7-D dynamic restoration. Limit Phase I claims to verified quiescent starts. |
| Contact reporting | `PushT._get_info()` divides collision counts using an inconsistent 1,000-step denominator although each action executes 10 substeps. | Record the environment’s direct collision callback count and physical block motion, not `info["n_contacts"]`. |
| Probes | Existing episode-wise split and encoded caches are reusable. Existing “linear” probes are AdamW-trained affine layers, not Ridge. Linear test errors are roughly 21 px agent, 10.5 px block, and 0.80 rad angle; MLP probes are substantially better. | Train the required Ridge agent probe. Revalidate frozen MLP block/angle probes as primary non-target readouts; retain linear readouts as sensitivity checks. |
| Diagnostics | Exact chunked nearest-neighbor/manifold calculations, rollout metrics, plotting helpers, and frozen encoder preprocessing are reusable. | Extend rather than replace. |
| Baseline | Saved official run reports 48/50 success in 296 s with CEM: 300 samples, 30 iterations, top-30, horizon 5, action block 5. Current environment is CPU-only. | Run a deterministic 10-episode smoke test and preserve the historical 50-episode result as a comparison, not a newly reproduced result. |
| Repository state | User changes already exist in training config and two diagnostic modules; large artifacts are untracked. No dependency lock or pytest installation exists. | Avoid those modified files, add isolated Phase I code, use existing direct-execution test style, and never overwrite prior artifacts. |

No new third-party runtime dependency is required. The run manifest will capture the installed direct versions, including `stable-worldmodel 0.0.6`, `stable-pretraining 0.1.6`, PyTorch `2.11.0`, scikit-learn `1.7.2`, HDF5, Matplotlib, Pymunk, and Gymnasium.

## Implementation changes

Add an isolated `counterfactual/` package, a single staged CLI at `scripts/run_phase1.py`, configuration at `config/phase1/pusht.yaml`, direct-execution `test_counterfactual_*.py` tests, and `reproduce_phase1.md`. Large HDF5 artifacts go under `.stable-wm/phase1/<run_id>`; manifests, tables, four figures, and the memo go under `diagnostic_results/phase1/<run_id>`.

1. **Frozen baseline and contracts**

   - Centralize checkpoint loading, ImageNet image preprocessing, exact action standardization, five-action tokenization, and `h/k` indexing.
   - Validate checkpoint/cache identity by re-encoding sampled cached frames; rebuild caches only if they do not match.
   - Run repeated encoding and prediction checks plus the fixed-seed 10-episode CEM smoke test without changing cost, physics, goal encoding, or official metrics.
   - Save git status/SHA, checkpoint hashes, package versions, device, precision, action statistics, shapes, commands, and measured outputs.

2. **Simulator fidelity**

   - Implement full physical-state capture for audit purposes: agent and block positions, velocities, angles/angular velocities, goal, controller-visible state, contacts, physics parameters, and seed.
   - Construct every experimental rollout in a newly instantiated environment with the same family seed, shared goal/style, explicit state, and zero velocities.
   - Require exact initial readback and duplicate-trajectory agreement within `atol=1e-10` for all recorded physical components.
   - Stress-test moving/contact restoration, but if hidden contact state cannot be reconstructed exactly, mark it unsupported and retain the predeclared quiescent-start restriction.

3. **AB-only and ABC datasets**

   - Use master seed `20261014`. Per cohort generate 30 pilot groups—18 development and 12 validation—followed, after freezing choices, by 50 untouched test groups. Balance free-space and near-contact groups; cycle agent displacements of 10, 20, and 30 px.
   - Define the displacement toward block `b1`, so C/B’s agent is closer than A’s. For ABC, move donor block `b2` 60 px perpendicular to the approach direction. Reject and resample complete families for overlap, wall proximity, drift, mismatched goal/style, or nonzero initial velocity.
   - Histories use zero raw actions through `h=10`, producing coherent frames at `h=0,5,10`. Future actions run from `h=10…35`.
   - Select one global scripted push template on development groups only from amplitudes `{0.10,0.20,0.30}` and active durations `{10,15,25}` raw steps. Choose the lowest-action-energy template yielding at least 80% near-contact trajectories and 0% free-space contacts; stop if none qualifies.
   - Save every raw physical step for A/B/C, direct contact counts, raw actions, standardized tokens, images, full readbacks, validation flags, group IDs, and split labels in HDF5 plus a hashed JSON manifest.

4. **H1 Ridge probe**

   - Reuse natural encoder caches and episode split seed 42. Fit Ridge in float64 from chunked sufficient statistics, avoiding a multi-gigabyte sklearn copy.
   - Tune normalized regularization `λ ∈ 10^{[-8,-7,…,2]}` on validation data only; fit an intercept without per-coordinate latent rescaling; freeze `W`, `b`, rank, and config before causal test data.
   - Report coordinate R²/RMSE and 2-D Euclidean RMSE with 10,000 episode-level bootstrap replicates. H1 passes when the lower 95% CI for aggregate R² exceeds `0.90`; errors relative to the 10–30 px interventions remain explicitly reported.
   - Revalidate existing frozen MLP agent/block/angle probes on natural validation/test latents. Use MLP readouts for independent physical evaluation; the defining Ridge readout is only an algebraic implementation check.

5. **Intervention and Step 6 gate**

   - Compute a rank-aware SVD row-space basis `Q` from Ridge `W`, with tolerance `max(W.shape) × eps × largest_singular_value`; patch all three history latents using `A + Q(Qᵀ(B-A))`.
   - Compute `r_P` per frame and group. Mark denominators undefined using a scale-aware floating-point tolerance rather than adding epsilon.
   - Evaluate zero, full-swap, natural C, and 32 deterministic random rank-matched controls per group. Each random projected difference is rescaled to the targeted patch’s actual injected norm; degenerate draws are resampled and logged.
   - Calibrate equivalence margins from natural validation probe errors, then freeze them. On untouched test groups, require:
     - independent MLP agent readout improves over A and is equivalent to natural C within one validation-calibrated error margin;
     - block-position and wrapped-angle excess errors remain within their calibrated margins;
     - targeted agent error is better than matched-norm random controls;
     - at most 5% of patched tokens exceed the 99th-percentile natural-validation nearest-neighbor distance.
   - Use paired group bootstrap intervals for these decisions. If the gate fails, emit Figure 1, Figure 2, and a STOP memo; do not execute or interpret H2.

6. **H2a and H2b**

   - Roll out `k=1…5`, corresponding exactly to `h=15,20,25,30,35`, with identical standardized model actions and identical raw simulator actions.
   - H2a uses Euclidean latent distance for `Phi_out`, records all undefined near-zero denominators, and reports `r_P` jointly without clipping.
   - H2b uses frozen independent probes and reports separate units:
     - agent and block position effect error in pixels;
     - wrapped block-angle effect error in radians;
     - natural-C absolute and effect baselines;
     - matched random, zero, and full controls.
   - Stratify by free-space/near-contact, actual contact, and real effect strength. Aggregate only across independent groups with 10,000 paired bootstrap replicates.

7. **Reporting**

   - Generate the four specified figures exclusively from saved numeric results.
   - Produce per-group/per-horizon CSV or JSON tables, exclusion and OOD counts, frozen-config hashes, commands, and a concise H1/H2 memo.
   - End after the Phase I go/no-go recommendation. No action ranking, H3 implementation, first-action execution, or closed-loop planning is included.

## Interfaces and artifacts

The internal API will expose typed records for `SceneSpec`, `ActionContract`, `GroupRecord`, `FrozenProbe`, and `InterventionResult`, plus functions for fresh-scene construction, replay, encoding, Ridge fitting, row-space patching, controls, H2 metrics, bootstrap aggregation, and plotting.

The CLI stages will be:

`baseline → simulator → pilot-data → probes → pilot-intervention/freeze → test-data → test-selectivity → h2 → report`

A frozen configuration hash is required before test generation; test stages refuse to run if probe, action template, thresholds, checkpoint, or data contract changes.

## Validation tests and acceptance

- Existing six diagnostic test scripts continue to pass.
- Exact training-action normalization, raw-versus-normalized regression, image preprocessing, cache/checkpoint identity, and repeated model inference.
- Fresh-environment replay across stationary, moving, near-contact, and collision-producing actions; explicit rejection of incomplete 7-D dynamic restore.
- AB/ABC scene identities, zero velocities, bounds, no initial contacts, shared goals, group split isolation, and trajectory/action equality.
- Context/action alignment at every raw and model step.
- Ridge closed-form checks against sklearn on a small fixture; alpha selection and episode bootstrap tests.
- `QᵀQ`, projector symmetry/idempotence, rank, patch formula, `r_P`, zero identity, full swap, and matched-norm random controls.
- Synthetic `Phi_out`, `E_effect`, angle wrapping, denominator flags, and incompatible-unit separation.
- A two-group end-to-end integration run that regenerates identical manifests and metrics from the same seed.
- No H2 execution unless the frozen held-out selectivity gate passes.

## Scientific risks and boundaries

- The public simulator state cannot support arbitrary post-contact restoration; conclusions are restricted to verified quiescent initializations.
- Prior long-horizon artifacts used raw model actions and cannot be treated as compatible Phase I baselines without regeneration.
- The defining Ridge probe necessarily reads its own row-space patch as donor-like; independent MLP readouts, natural C, random controls, and manifold checks provide the non-tautological evidence.
- Existing linear-probe error is comparable to the smaller interventions, and block-angle linear error is large. Dose-stratified results and calibrated nonlinear readouts are mandatory.
- Near-contact dynamics are discontinuous and may have high variance; contact/effect-strength stratification and group-level intervals prevent zero-effect cases from appearing artificially successful.
- Static zero-action histories validate the initial causal protocol but do not establish validity for moving histories or deployment.
- Negative H2 results may reflect LeWM’s known one-step under-updating rather than failure of the latent edit; natural-C prediction establishes that ceiling.
