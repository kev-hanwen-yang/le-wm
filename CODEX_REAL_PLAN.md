# Phase I LeWM/Push-T Counterfactual Study — Implementation Plan

Reviewed on 2026-10-14 against [AGENTS.md](AGENTS.md), the full [CODEX_EXPERIMENT_PLAN.md](CODEX_EXPERIMENT_PLAN.md), the current repository and the primary sources below.

## Goal, authority and status

The experimental specification remains authoritative. This document operationalizes **Steps 1–7 only**; it does not replace the hypotheses or authorize Phase II.

- **H1:** held-out agent-position decodability using a frozen Ridge probe.
- **H2a:** A/B-only predictor dependence, measured by `r_P` and `Phi_out`; not physical causality.
- **H2b, primary:** selective A/B-donor editing toward real hybrid C, followed by physical-effect fidelity under matched actions.
- AB-only: A=`(a1,b1)`, B=`(a2,b1)`. ABC: A=`(a1,b1)`, B=`(a2,b2)`, C=`(a2,b1)`. **B is the donor; C is the ABC physical target.** Full swap produces B, not C.

The core design is sound. Improvements are needed for family isolation, measurement resolution, precommitment and honest gates. Step 1 is implemented with saved passing evidence. Steps 2–7 below are **planned, not completed**. Negative or measurement-limited findings are valid Phase I outcomes; improved planning success is not its objective.

No LeWM retraining/fine-tuning, physics/reward/cost changes, goal-encoding changes, official-metric changes, H3 action ranking or new closed-loop experiment is included. The already authorized official CEM smoke remains a Step 1 functionality check, not Phase II research.

## Repository audit and reuse boundaries

| Area | Verified state | Reuse / required work |
|---|---|---|
| Model | [jepa.py](jepa.py): `encode`, `predict`, `rollout`, final-latent squared-L2 planner cost. [module.py](module.py): six-layer `ARPredictor`. Latents are 192-D; predictor context is three frames. | Freeze and reuse through [counterfactual/model.py](counterfactual/model.py). |
| Checkpoint/revision | `.stable-wm/pusht/lewm_object.ckpt`, SHA-256 `100d6459b9198697c1c26ef47e3eef4d7548946401b7501213caf761495a2553`. Review started on `phase1-counterfactual`, HEAD `b2bbdea`; reviewed baseline records its own run revision `597cc334…`. | Do not substitute review HEAD for a saved run's actual revision. Hash checkpoint, code and resolved config in each new run. |
| Step 1 evidence | [Reviewed baseline](diagnostic_results/phase1/step1-reviewed-baseline-seed42/baseline_summary.md): repeated CPU encoding, prediction and five-step rollout max difference `0.0`; sampled cache checks pass; official CEM **9/10**. | Reuse immutable evidence. Cache checks sampled 16 frames per split, not every row. Official evaluation resets without dataset seeds; process/CEM seeding does not prove physical replay. Historical 48/50 remains historical. |
| Time/actions | [counterfactual/contracts.py](counterfactual/contracts.py): `TimeActionContract`, `ActionNormalizer`; [probe/rollout_windows.py](probe/rollout_windows.py) has alignment helpers. Context `h=0,5,10`; targets `h=15,20,25,30,35`. | Reuse existing contracts, not a second `ActionContract`. Five predictions require **seven** five-action tokens covering raw intervals `[0,35)`. |
| Preprocessing | Training standardizes raw 2-D actions before flattening five rows. [probe/rollout_ablation.py](probe/rollout_ablation.py) contains a historical raw-action path. Step 1 centralizes the correct path. | Compute statistics from source HDF5: approximately mean `[-0.007812565,0.006860687]`, std `[0.208467484,0.206748664]`. Preserve historical code/results but do not reuse incompatible rollout predictions as Phase I baselines. |
| Simulator | Installed `stable_worldmodel/envs/pusht/env.py`: public 7-D observation omits block velocity/angular velocity; `_set_state` steps physics. Preliminary quiescent checks are encouraging, but formal Step 2 evidence is missing. | Fresh verified quiescent initialization only. Capturing exposed fields does not guarantee hidden solver/contact state can be restored. |
| Controls/contacts | Relative control targets current agent position plus `100 × raw_action`; each raw action has 10 physics substeps. `_get_info()` normalizes contacts with an inconsistent denominator. | Equal relative commands are not equal absolute targets. Log agent–block and wall contacts separately while preserving original callback behavior. |
| Natural data/splits | Read-only scan of `.stable-wm/pusht_expert_train.h5`: **18,685 episodes, 2,336,736 frames, 185 distinct initial 7-D observations**. All 185 families span train/validation/test under [probe/data.py](probe/data.py): `split_by_episode(seed=42)`. First 20 episodes share one initial observation but eight initial-image hashes. | Episode isolation does not establish unseen-initial-pose isolation. Add a conservative family audit and new H1 family split; leave old splits/reports unchanged. Appearance variants are not independent physical families. |
| Caches | [probe/embedding_cache.py](probe/embedding_cache.py) and Step 1 validation expose source/frame/episode identity. | Reindex existing cache rows across old split files into new family partitions, without re-encoding when identity/coverage checks pass. Cache filenames do not define the new scientific split. |
| Probes/units | [probe/models.py](probe/models.py), [probe/train.py](probe/train.py), [probe/targets.py](probe/targets.py), [probe/metrics.py](probe/metrics.py). Existing affine probes are AdamW fits, not Ridge. Historical position RMSE pools coordinates; `r` is Pearson, not R². Positions are 512-world-coordinate px, not resized 224-image px. | Reuse architectures/loaders, not old accuracy/split claims. Multi-seed summaries give roughly 20.95/10.47 coordinate-pooled px for linear agent/block and 7.00/2.43 for MLP. Euclidean position RMSE is `sqrt(2)` times these. Recompute wrapped angle errors; old reports use raw angles. |
| Readout files | Root agent/block/angle MLP checkpoints exist in `.stable-wm/probes/`; seed-specific files referenced by historical summaries are not all present. Root files lack full family/probe-seed provenance. | Inventory/hash actual files. Historical performance is context, not fresh family-held-out calibration. Validate frozen MLPs on natural pilot scenes with known simulator labels. |
| Diagnostics/layout | [probe/latent_diagnostics.py](probe/latent_diagnostics.py): chunked nearest-reference distances, norms and rollout metrics. Six legacy and two Step 1 direct-execution test scripts exist. `counterfactual/`, [scripts/run_phase1.py](scripts/run_phase1.py), [config/phase1/pusht.yaml](config/phase1/pusht.yaml), [reproduce_phase1.md](reproduce_phase1.md) already exist; CLI implements `baseline` only. Review began with a clean tree. | Extend the isolated harness; reuse kernels with a family-filtered reference. Never overwrite prior artifacts or claim proposed stages already exist. |

No new third-party runtime dependency is presently required. Capture installed PyTorch, stable-worldmodel, stable-pretraining, NumPy, scikit-learn, HDF5/h5py/hdf5plugin, Matplotlib, Pymunk and Gymnasium versions. Import the installed HDF5 filter plugin before reading compressed data. Save a dependency inventory and restoration instructions: version logging alone is not a complete lock. Current execution is CPU; backend/precision changes require revalidation.

## Fixed scientific contracts

1. **Units/actions:** world-coordinate px and wrapped radians remain separate metrics. Use `wrap(theta)=atan2(sin(theta),cos(theta))`. Identical bounded raw relative commands go to A/B/C; exactly standardized versions go to every model condition. Zero raw action is **not** zero standardized action.
2. **Histories:** independently simulate each branch, retain `h=0,5,10`, and patch corresponding latents in all three frames using one frozen projector. Raw actions `[0,10)` construct history; `[10,35)` produce five future targets. No last-frame teleport in primary H2.
3. **Families:** related base poses, appearance variants, doses, templates and controls share a family ID/split. Related AB/ABC cases share the highest-level sampling unit; they are not extra independent replicates.
4. **Partition roles:** natural-family train/validation/test for H1; synthetic development/validation pilot for protocol/calibration; untouched synthetic test for selectivity/H2. H1 test is held out from probe fitting, **not necessarily original LeWM training**.
5. **Freeze:** checkpoints/probes/P, generator, actions, units, metrics, margins, reference sets, denominators, exclusions, primary endpoints and sample counts freeze before final test outcomes. Pilot changes create a protocol version; final-test failure never triggers tuning on that same set.
6. **Selection:** predeclared geometry/replay/contract failures can invalidate a whole family. Zero effect, poor predictions, failed selectivity, OOD flags and undefined ratios are outcomes, not permission to replace test families. Log every attempt. Invalid simulator fidelity halts collection.

## Implementation sequence and gates

### 1. Frozen baseline and contracts — implemented

Reuse [counterfactual/baseline.py](counterfactual/baseline.py), [counterfactual/model.py](counterfactual/model.py) and [counterfactual/contracts.py](counterfactual/contracts.py): exact checkpoint loading, ImageNet preprocessing, action statistics, tokenization and indexing.

The reviewed run establishes repeated inference, sampled cache identity and official ten-episode CEM functionality with unchanged cost/physics/goal/metrics. Re-run for relevant code/environment/checkpoint changes or a new fully traced protocol run. Repair only demonstrated cache mismatches, preserving originals and replacement identity.

**Gate:** saved passing baseline plus compatible source/environment identity. Not a simulator gate. Preserve failures too; skipped CEM means verification-only, not complete baseline pass.

### 2. Simulator fidelity — first implementation task

- Audit reset ordering, body/controller state, seed, physics, goal/render settings and installed source. Goal initialization followed by scene initialization can step physics; inspect final positions **and all body velocities**, not just supplied reset options.
- Create a fresh environment per branch/replay with shared family seed, explicit goal/style and zero velocities/angular velocities. Verify whether training images render a goal; legacy `with_target=False` reconstruction comments are not evidence of training rendering configuration.
- Save an audited physical readback, not a claimed complete engine snapshot. Reject incomplete 7-D dynamic restore in the primary pipeline.
- Compare duplicate fresh-start trajectories at every raw step through `h=35`, initially float64 `atol=1e-10`, `rtol=0`, on the same audited backend. Compare all physical fields, wrapped angles, typed contacts and actions; require exact image hashes on that backend. Any tolerance change needs a numerical justification before collection.
- Cover stationary, free motion, near-contact, collision-producing and post-contact trajectories. Separately stress-test moving/contact restoration; unsupported arbitrary restore does not invalidate tested fresh-start replay, but limits scope.
- Identify contact bodies while delegating original collision handlers. Do not modify response, geometry, timestep or controller gains.

**Gate:** reproducible fresh initialization and full raw-step replay, explicit image/action/goal contracts, and tests detecting `_set_state` side effects and omitted dynamics. No ground-truth dataset before this passes. Quiescent histories do not establish moving-history validity.

### 3. AB-only / ABC datasets and coherent histories

- Keep master seed `20261014`. Per cohort target 30 independent pilot families (18 development, 12 validation), then 50 untouched test families after freeze. Assign one primary group/dose per family; extra doses/templates remain repeated measurements. Balance free-space/near-contact and doses 10/20/30 world px as counts allow. These are feasibility targets, **not guaranteed statistical power**.
- Vary block pose, approach direction/angle and location. Move C/B's agent toward `b1`; place ABC donor block `b2` 60 world px perpendicular to approach, balancing both perpendicular signs. Use actual body geometry, not center distance alone, for clearances. Reject complete candidates for overlap, unsafe wall clearance, drift, nonzero velocities or changed goal/style.
- Freeze numerical clearance ranges after development-only feasibility checks. Save intended/realized poses and verify AB/ABC identities throughout the zero-action history. A history that drifts away from the intended identity fails construction, not an outcome-based filter after H2.
- Execute zero **raw** actions for ten steps and retain actual `h=0,5,10` frames. Future bank: amplitudes `{0.10,0.20,0.30}` × active durations `{10,15,25}` raw steps along the precomputed approach direction, then zero commands to fill 25 future steps.
- Choose one global primary template using **development simulator trajectories only**, never patched/model/probe outputs. It must produce A-or-C agent–block contact in at least 80% of near-contact groups, none in free-space A/C, and nontrivial real A/C block-position effect in at least 50% of near-contact groups. Define nontrivial as `max_k ||block_C(k)-block_A(k)|| >= 1` world px, a proposed feasibility threshold equal to one tenth of the minimum agent dose. B is not the ABC physical target.
- Among qualifying templates choose lowest raw squared-action energy; ties: ascending amplitude, then duration. If none qualifies, STOP for inadequate excitation. A newly versioned pilot is possible before test, not a silent test-driven bank/threshold change.
- Freeze generator/template before test. Keep every physically valid final family, including zero effects and unexpected realized contacts. The other bank templates may be a predeclared secondary robustness analysis if compute permits, not action ranking or planner optimization.
- Save raw-step physical/contact trajectories, commands, seven standardized tokens, images/latents, goal/style, intended/realized poses, seeds, family/split labels, source identities and attempt logs. Prevent duplicate/related families across splits using the frozen rule.

**Gate:** valid scene identities, exact shared actions, coherent histories, real C trajectories, deterministic replay and disjoint families. Weak test excitation is reported, not repaired by resampling.

### 4. H1 Ridge and independent readout calibration

**Natural family split.** Group at least episodes sharing initial observation/pose, regardless of rendering. Audit near-duplicate poses and repeated trajectory signatures before freezing the rule; 7-D equality cannot certify full physical-state identity. Reuse caches via global source-row indices, with coverage/uniqueness checks. Sort canonical family IDs, shuffle with NumPy `default_rng(42)` and partition with floored 80%/10% counts and the remainder for test (148/18/19 if exactly 185 families remain). Record the algorithm and IDs; preserve original DICTA splits/reports.

**Explicit Ridge objective:**

`min_(W,b) sum_i w_i ||W z_i + b - y_i||_2^2 + lambda ||W||_F^2`.

Use original latent coordinates, unpenalized intercept, no whitening. Set `w_i=1/(F*n_f)` for a frame in training family `f`, with `F` training families and `n_f` frames in that family. Weights sum to one; families contribute equally. Stream weighted centering/sufficient statistics in float64 and solve the 192-D system. Tune `lambda` over `{10^-8,...,10^2}` by family-balanced validation position squared error. Numerical ties choose larger lambda under a declared tolerance. Fit on training only, not an undocumented post-selection train+validation refit. With these normalized sample weights sklearn uses `alpha=lambda`; unweighted sum-loss Ridge corresponding to mean-loss lambda instead uses `alpha=N*lambda`.

Report coordinate R²/RMSE, coordinate-pooled RMSE and Euclidean position RMSE; family-balanced summaries are primary and frame-pooled summaries descriptive. Compute R² with family-balanced residual sums and target-mean variance across the partition, not a mean of per-family R²; constant-target coordinates remain undefined. Retain the previous implementation-plan H1 default: lower 95% family-bootstrap CI of equal-coordinate aggregate R² exceeds `0.90`. This is a declared acceptance choice, not a theoretical threshold from the authoritative specification. Report error against each dose: high R² alone does not establish intervention resolution.

**Independent readouts.** Inventory/hash actual root MLPs; calibrate frozen agent/block/angle readouts on fresh natural pilot A/B/C with simulator labels. They are independent of Ridge **algebra**, not independent ground-truth sensors. Recompute wrapped angle and differential errors; old coordinate-pooled/unwrapped summaries cannot directly define margins. Linear probes remain sensitivity checks.

Reuse MLPs if fresh calibration suffices. Otherwise stop the causal gate and report measurement limits. A separately versioned refit of the existing small probe architecture on natural-family training data can be proposed before final test, not silently introduced or trained on patched/test outcomes. LeWM stays frozen. Never select a favorable readout after final H2.

**Gate:** frozen Ridge/non-target readouts, honest H1 held-out accuracy, differential resolution, projector coordinate convention and provenance. Failure/inadequate resolution means STOP/INCONCLUSIVE, not proof that LeWM lacks position information.

### 5. Intervention, matched controls and held-out selectivity gate

Take rank-aware SVD row-space basis `Q` of `W`, tolerance `max(W.shape)*eps*sigma_max`, rank at most two. Freeze `Q`; patch every context frame with `A[t] + Q(Q^T(B[t]-A[t]))`. Reject nonfinite/rank-zero inputs. The defining Ridge's donor-like readout is an algebraic check, not independent selectivity evidence.

Per frame, `r_P=||Q Q^T delta||^2/||delta||^2`; also report a group ratio of summed numerators/denominators, distinguished from a mean of frame ratios. Freeze scale-aware denominator tolerance; log undefined cases and numerical range violations without epsilon/clipping.

**Controls:** natural A/B/C, zero, full swap and 32 deterministic random bases per family. Draw Haar-distributed orthonormal bases via seeded Gaussian QR with fixed sign convention; retain one basis across three frames. Match actual injected target norm per frame:

`random_delta[t] = (||target_delta[t]||/||U U^T delta[t]||) * U U^T delta[t]`.

This is a **scaled random-subspace perturbation**, not unscaled orthogonal interchange; do not give it the target projector's `r_P` interpretation. Zero target norm means zero perturbation. Resample a basis only when a nonzero target cannot be matched; cap attempts at 1,024 and log scales/degeneracy. Control failure is recorded, not permission to drop the family. Full swap must equal donor B history; zero must equal A; all controls share aligned actions.

**Pretest margins:** calibrate family-balanced RMS differential readout error on unpatched ABC pilot-validation C versus A, `[g_j(C)-g_j(A)]-[c_j(C)-c_j(A)]`, wrapping angle differences. Set measurement margins `M_j` to their one-sided 95% upper calibration-bootstrap bounds, recording both point errors and uncertainty; report absolute natural errors too. This accounts for small-pilot uncertainty rather than treating an estimated RMS as exact. Probe noise is not automatically acceptable physical error. Proposed practical caps: **5 world px for each position readout** (half the minimum agent dose) and **0.10 rad for angle**. These are explicit protocol choices, not measurements. Freeze them before test; if a calibrated margin exceeds its cap, return measurement-limited rather than enlarge the cap for a pass. Include a prespecified margin-sensitivity analysis.

**Input distribution:** reuse chunked nearest-reference distance and latent norms, with a frozen natural-training-family reference excluding queries/self/final-test rows. Use at most 100,000 references, allocated equally across training families (remaining slots by sorted family ID), sampled without replacement with a separately recorded seed; save every source-row ID. Compute exact distances against this bank, not an inaccurately labeled full-dataset nearest neighbor. Check bank-size sensitivity on pilot data only before freeze; all conditions use the same final bank. This bounds CPU/memory cost without changing readouts or inspecting test outcomes.

Compare patch to A/B/C and direct latent distance to C. Define `M_NN` as the one-sided 95% upper calibration bound on ABC pilot-validation family RMS of A-versus-C nearest-reference-distance differences; compare positive patch-versus-C distance excess to it using a family-level bound. Report natural-C distances and norm shifts. This is a compatibility proxy, not a manifold certificate. Keep percentiles/tails descriptive: 12 validation ABC families cannot reliably estimate a 99th-percentile tail; three frames do not triple N.

On untouched ABC families, first average squared input errors over the three frames within each family, then take RMS over equally weighted families for the error-limit gates. Recompute that statistic in each bootstrap draw; for improvement contrasts, use paired per-family RMS errors. For NN excess, average positive excess over frames within family and report its family mean/bound. Require independent MLP evidence for:

- differential agent error against true `a2-a1` within `M_agent`, and paired improvement over unpatched A and mean matched-random controls;
- block-position and wrapped-angle changes from A within `M_block`/`M_angle`;
- reported patch-versus-natural-C decoded errors and natural-C absolute/differential calibration, not just a donor-like defining-Ridge readout;
- positive patch-versus-C nearest-reference-distance excess within `M_NN`, alongside norm/domain-shift diagnostics.

Use one-sided 95% upper family-bootstrap bounds for nonnegative error limits and lower bounds greater than zero for improvements. Wide intervals are **INCONCLUSIVE**, not equivalence/preservation. Keep every case, including failures, in downstream aggregation if the aggregate gate passes. Evaluate once under the frozen protocol. Failed/inconclusive measurement, distribution or selectivity gives Figures 1–2 and a STOP/INCONCLUSIVE memo; H2 stays `NOT_RUN` under this staged implementation. Any separately authorized diagnostic rollout cannot be relabeled successful causal H2.

**Gate:** correct projector/controls, independent held-out target selectivity, non-target preservation and calibrated compatibility. Passing supports a candidate intervention, not uniquely causal coordinates.

### 6. H2a dependence and H2b physical-effect fidelity

Run the frozen predictor with three-frame histories at `k=1..5`, `h=15,20,25,30,35`. Simulator branches receive identical relative raw commands; every model condition receives exactly their standardized tokens.

**H2a, AB-only**, Euclidean latent distance:

`Phi_out(k)=1-||F^k(patched_A,u)-F^k(B,u)||/||F^k(A,u)-F^k(B,u)||`.

Save denominators/tolerance decisions, exclusion counts, all finite values including negative ones, and `r_P`. Undefined ratios do not remove cases from other valid analyses. No epsilon, clipping or ABC donor-as-physics-target substitution.

**H2b, ABC**, separately per physical variable:

`E_effect(k)=||[g(F^k(patched_A,u))-g(F^k(A,u))]-[c_C_real(k)-c_A_real(k)]||`.

Save real/predicted effects, not only error curves. For angle, wrap each predicted/real effect and then their residual difference; report absolute circular error in radians. Position errors are Euclidean world px. Include natural-C absolute error `E_C_abs`, natural-C-versus-A effect error, zero/full/random controls, real A/C trajectories and donor B diagnostics. C is real simulator ground truth.

**Predeclared primary contrast:** per-family mean block-position effect error over five horizons in the **geometrically near-contact ABC stratum**, patch versus zero and versus its mean 32-control random error. Positive lower one-sided 95% paired-gain bounds for both support improvement, not automatically absolute accuracy. Always report patch error and its gap to natural-C effect error. Downstream block motion is more informative than the directly edited agent readout; this endpoint prevents favorable-horizon selection.

Keep all valid near-contact families in that primary estimate, even with weak realized contact/effect. Agent/angle, free-space, horizon curves, realized contact classes and the fixed `>=1` world-px block-effect stratum are secondary. Interpret effect direction/magnitude only above its declared near-zero threshold. Extra bank templates are repeated measurements, not independent N or permission to choose the best held-out action.

Separate selective input construction, improvement over controls, and absolute/proximity-to-natural-C fidelity. Bad natural-C prediction establishes a frozen-predictor ceiling; a negative patch result then cannot by itself disprove the subspace. Even a positive result is readout-conditioned, quiescent-distribution evidence, not complete causal identification.

**Uncertainty:** 10,000 deterministic paired family-bootstrap replicates. Aggregate frames/horizons/random draws within family; preserve paired conditions and jointly cluster related AB/ABC cases. Report 95% two-sided descriptive curve intervals and separately specified one-sided gate/primary bounds. Few independent families (e.g. 19 natural H1 test or 25 near-contact test) limit precision; bootstrap cannot fix that. No post-test sample expansion, pooled px/radian score or significance hunt across secondary endpoints.

### 7. Numeric outputs, figures and Phase I memo

Generate exclusively from saved results:

1. AB-only versus ABC donor `r_P`, denominator flags and family/dose summaries.
2. A/B/C/patch/control agent/block/angle selectivity, margins, natural-C comparison and input-distribution diagnostics.
3. AB-only `r_P` versus horizon-specific `Phi_out`, including undefined counts.
4. ABC error curves and real effects/trajectories, natural-C absolute/effect baselines, controls, paired uncertainty and primary contrast.

If a prior gate stops, produce available figures and explicit `NOT_RUN` entries, not fabricated H2 panels. The memo distinguishes `PASS`, `FAIL`, `INCONCLUSIVE/measurement-limited` and `NOT_RUN`; states tested facts, uncertainties and Phase II recommendation. A recommendation is not authorization.

## Interfaces, CLI and reproducibility

Extend existing `TimeActionContract`/`ActionNormalizer`. Proposed records: `SceneSpec`, `PhysicalReadback`, `GroupRecord`, `FrozenProbe`, `InterventionResult`. Keep new simulator/data/Ridge/intervention/metric/statistical/report helpers inside `counterfactual/`; reuse existing kernels.

Proposed CLI sequence:

`baseline -> simulator -> family-audit -> pilot-data -> probes -> pilot-intervention/freeze -> test-data -> test-selectivity -> h2 -> report`.

`report` also works after a failed gate. Each stage validates parent manifests and writes immutable evidence. Freeze resolves every scientific choice; no placeholder threshold reaches test. Changing probe/P, checkpoint, split, generator, actions, reference, margin, exclusion or metric invalidates downstream stages and requires a versioned protocol. Evaluate natural H1 test only after probe selection; it cannot become another tuning set.

Large HDF5/tensor artifacts: `.stable-wm/phase1/<run_id>`, outside ordinary Git commits. Compact manifests/tables/figures/memo: `diagnostic_results/phase1/<run_id>`. Keep checkpoint hashes/paths and restoration instructions rather than committing large `.pt` files. Source/config/tests/docs and compact evidence may be committed separately; preserve ignore rules and user artifacts.

Record seeds/families/attempts, source HDF5/cache/checkpoint/readout/reference hashes, code SHA plus dirty diff/source identity, commands, resolved config, dependency inventory, device/precision, action statistics and simulator/controller/goal/render contracts. Freeze manifests link parents. Deterministic integration compares canonical arrays/data hashes, not identical timestamps, paths, run IDs or durations. Bitwise reproducibility is scoped to the audited backend, not promised across devices/library versions.

## Validation and incremental implementation

- Keep six legacy and two existing Step 1 test scripts passing; direct-execution style.
- **First smallest patch:** fresh initialization/readback/replay helpers plus proposed `test_counterfactual_simulator.py`, showing identical raw-step physics/contact/images and rejection of incomplete dynamic restore. Proceed only after this gate.
- Family fixtures: duplicated poses/appearance variants cannot straddle splits; global cache-row reindexing preserves coverage/labels without duplication. Where real data are available, reproduce the 185-family audit.
- Data/history: AB/ABC identities, clearance, velocity, goal/style, raw-zero versus standardized actions, all seven token intervals, shared commands, attempt logs and no outcome-based test replacement.
- Ridge: streamed weighted fit versus sklearn; intercept/alpha convention, no whitening, tie/rank handling, R² denominators, coordinate versus Euclidean RMSE and family-balanced bootstrap.
- Patch: basis/projector algebra/rank, defining-probe identity, zero/full history equality, per-frame/group ratios, coherent random basis, actual per-frame norms and zero/degenerate attempt handling.
- Metrics/gates: synthetic `Phi_out`/real-C effect and natural-C baselines, donor-B mismatch, nested angle wrapping, separate units, undefined denominators, calibration caps, inconclusive intervals and immutable freeze.
- Reference: exclude self/query/test families; streamed distance equals a small exact fixture; state L2 versus squared-L2 units; tails counted by families, not independent tokens.
- Statistics: paired highest-family resampling, no frame/horizon/control pseudoreplication, deterministic seed and outcome-independent primary endpoint.
- Two-family integration checks deterministic numeric execution, **not power or scientific gate passage**. H2 rejects absent/failed/inconclusive selectivity; report still runs.

## Risks and interpretation limits

- Exact initial-observation equality reveals split overlap but not complete physical-state identity; the family audit must be conservative. This does not retroactively alter the DICTA study's measured estimand.
- Differential probe noise may be smaller than absolute error, but patched/predicted OOD latents can invalidate calibration. Independent readouts are imperfect instruments; probe-only success cannot certify a complete hybrid state.
- Scaled random matching controls injected energy, not all geometry; disclose frame-wise scales.
- Static histories may be visually redundant/easier than moving ones; fresh-start results do not establish arbitrary restore, deployment or planning benefit.
- Contacts need not produce useful A/C effects. Near-contact discontinuities and small N can yield broad intervals; zero-effect cases must not look like successful contact physics.
- Rare-event/equivalence claims are underpowered at these counts. Even zero failures in 50 independent families gives a one-sided 95% exact binomial upper failure bound of about **5.82%**, not a confident `<=5%` population claim.
- Separate readout, intervention and predictor limitations. Good `r_P`, `Phi_out` or selective probes are not proof of uniquely causal coordinates; failures do not authorize H3.

## Primary research supporting the safeguards

These sources inform methodology, not empirical claims about LeWM. Language-model patching results are analogies; this study supplies its own real simulator counterfactual.

- [Geiger et al., *Causal Abstractions of Neural Networks* (2021)](https://papers.nips.cc/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html): compare interchange interventions with the appropriate high-level counterfactual; here C, not donor B.
- [Makelov et al., *An Interpretability Illusion for Subspace Activation Patching* (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/70b8505ac79e3e131756f793cd80eb8d-Abstract-Conference.html): successful subspace patches can mislead about naturally used features. Retain real-C tests, independent readouts/controls and cautious interpretation; do not add new representation-learning methods.
- [Zhang and Nanda, *Towards Best Practices of Activation Patching in Language Models*](https://arxiv.org/abs/2309.16042): metric/construction choices can change conclusions; freeze metrics/actions/controls before test.
- [Hewitt and Liang, *Designing and Interpreting Probes with Control Tasks* (2019)](https://aclanthology.org/D19-1275/): interpret probe accuracy beyond decodability. Their control-task selectivity differs from this study's physical preservation gate.
- [Lakens, *Equivalence Tests: A Practical Primer* (2017)](https://journals.sagepub.com/doi/10.1177/1948550617697177): explicit meaningful bounds are necessary; nonsignificance is not equivalence. Here one-sided bounds on nonnegative error are used, not mislabeled signed-mean TOST.
- [Saravanan et al., *Application of Hierarchical Bootstrap to Multi-level Data* (2020)](https://arxiv.org/abs/2007.07797): correlated observations cannot replace independent units. Resample families and preserve pairing, acknowledging small top-level N.
- [scikit-learn 1.7 Ridge documentation](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.Ridge.html): weighted residual-sum plus `alpha` penalty; explicit normalized weights avoid a regularization-scale mismatch.

## Reasons for every revision

| Revision from the previous implementation plan | Reason |
|---|---|
| Correct Step 1 status, revisions, 9/10 evidence and current layout | Old text mixed historical/preliminary results with future work and stale repository status. |
| New natural initial-pose-family split; reuse caches, preserve old reports | All 185 observed families span old episode partitions; episode isolation does not establish new-pose generalization. |
| Explicit units, RMSE/correlation definitions, relative-action semantics | Coordinate RMSE is not Euclidean RMSE, Pearson is not R², equal relative commands are not equal absolute targets. |
| Formal fresh replay, renderer/goal audit and typed contacts | Public state/preliminary checks miss dynamics; reset side effects/render assumptions/contact normalization can conceal mismatches. |
| Development-only action selection also requires real effect | Contact alone can yield uninformative zero A/C effects. Never select on patch predictions or test outcomes. |
| Weighted Ridge/objective/alpha and highest-family resampling | Reproducible efficient fit; repeated frames/appearance variants must not dominate weighting or inflate N. |
| Audit available MLPs and calibrate fresh differential error | Historical summaries do not establish available file provenance, fresh resolution or OOD validity. Ridge cannot validate its own patch. |
| Per-frame norm-matched coherent controls, capped degeneracy | Match actual injected energy across all frames and disclose scaled controls' geometric limitations. |
| Practical error caps and explicit inconclusive outcome | Broad probe noise must not manufacture selective success. Proposed 5-px/0.10-rad budgets are transparent pretest choices, not findings. |
| Replace hard 99th-percentile/5%-token gate with continuous compatibility, a fixed family-balanced reference bank and descriptive tails | Twelve validation families cannot calibrate an extreme tail; dependent tokens cannot substantiate a population failure-rate claim. A bounded, explicitly sampled reference makes CPU evaluation feasible without pretending to search the entire data manifold. |
| Predeclare downstream-block primary contrast; retain valid test cases | Avoid favorable-horizon/metric/contact selection; downstream effects are stronger evidence than the directly edited variable alone. |
| Stronger freeze chain, failure reporting, canonical numeric reproducibility | Prevent test-driven revisions; timestamps need not match, and missing H2 panels must remain explicitly not run. |
| Preserve seven-step endpoint and AB/ABC controls | Core scientific question is appropriate; H3/new model training/planner changes would broaden scope without solving Phase I validity. |

The academic-research experiment-validation workflow informed the separation of feasibility, measurement resolution, inference and go/no-go decisions. This review changes only the implementation-plan document. New numerical defaults are explicit proposals that must be resolved into a frozen config before future held-out execution.
