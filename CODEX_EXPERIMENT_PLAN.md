# From Decodability to Decision Causality
## Interventional Analysis of Physical State in Latent World Models — LeWM on Push-T

**Document type:** Research specification and implementation plan for a coding agent (Codex)
**Status:** Approved experimental design; implementation not yet verified against the repository
**Last revised:** 2026-10-14
**Owner:** Independent researcher (DICTA 2026 LeWM rollout-diagnostics follow-up)
**Primary implementation scope:** Phase I / first research week, **Steps 1–7 (H1 and H2)**, with the controls required for valid H2 interpretation
**Deferred:** Step 8 / H3 action ranking and Step 10 closed-loop evaluation (Phase II)

> **Instructions to Codex:** Read this document and `AGENTS.md` before touching code. This is a scientific experiment, not a request to maximize a benchmark number. Do not silently change the hypotheses, original LeWM checkpoint, simulator physics, cost definition, evaluation units, data splits, or comparison conditions. Inspect the actual repository first: module names, APIs, checkpoint paths, and exact configuration must be discovered, **never invented**. Where an implementation detail in this specification differs from what is actually present, document the discrepancy and ask before changing scientific assumptions.

---

# 1. Scientific context, question, and hypotheses

## 1.1 Background: what the previous study established

The prior accepted sole-author DICTA 2026 study, *Long-Horizon Rollout Diagnostics for Latent World Models: LeWM on Push-T*, evaluated the released LeWM Push-T checkpoint with frozen physical-state probes and latent-trajectory diagnostics. It examined 1,000 held-out test episodes, an 18-step open-loop latent rollout, and a frame skip of five environment steps. The major observations were:

- Predicted latents retain approximately plausible **global scale/norm**, but diverge from the encoded future corresponding to the actual episode.
- Agent-position decodability deteriorates early; block-position information deteriorates more slowly; block angle exhibits an abrupt deterioration within the rollout.
- Predicted latent transitions are **under-updated** and **overly straight**. Even with real encoded context at each step (teacher forcing), the one-step predicted transition magnitude is only roughly one-third of the true encoded transition; open-loop feedback worsens the mismatch.
- These measurements diagnose the **symptoms** of a learned transition map, but do not establish whether a **particular readable physical variable is causally used** by the predictor or planner.

**Motivation for this follow-up:** A probe that decodes agent position does not prove the predictor uses that position, still less that manipulating its representation yields physically correct effects. This experiment moves from representational readout to **selective internal intervention**, then (later) decision consequences.

## 1.2 Core research question

> If a latent world model encodes the agent's physical position, is the position-related representation actually used in future prediction? Can a targeted intervention produce a new internally represented state whose predicted consequences agree with a matched, simulator-realizable physical counterfactual? Does this matter for action selection?

**Primary physical variable:** the 2D position `(x, y)` of the Push-T circular pusher/end-effector (the “agent”).
**Model/environment:** existing **LeWM + Push-T**, with no new world-model training.
**Physical comparator:** simulator ground truth from precisely initialized and reproducible states.
**Scope restriction:** one physical variable, one environment, one frozen world-model checkpoint in Phase I.

## 1.3 Hypotheses

| ID | Name | Exact empirical question | Phase |
|---|---|---|---|
| **H1** | Representation / decodability | Can a held-out linear probe accurately recover agent position from natural encoder latents? | I |
| **H2a** | Predictor dependence | Does intervening in the probe-defined position subspace change predictor outputs in the direction of the naturally encoded A/B comparison? | I; diagnostic, **not sufficient** |
| **H2b** | Compositional physical counterfactual fidelity | Does patching position-related features from a donor B into A result in a physically selective representation of constructed state **C**, and do its predicted effects match simulator C relative to A? | I; **primary** |
| **H3** | Decision fidelity | Do action ranking and actual candidate-set regret in **real C** change appropriately when planning from patched A versus A and naturally encoded C? | II |

The correct ordering is **method validity → predictor mechanism / physics → planning**. If a patch is not selective and meaningful at the input, its downstream effect must not be presented as evidence of physical causality.

## 1.4 Non-goals (strict in Phase I)

- Do not train or fine-tune LeWM, its encoder, predictor, or goal model.
- Do not introduce an SAE, DAS, new causal model, new world-model architecture, or multiple physical target variables as an unrequested “improvement.”
- Do not expand to another environment or larger model until this protocol is validated.
- Do not optimize for a favorable H2 outcome, retrofit thresholds after inspecting the test set, or present negative results as implementation failures without evidence.
- Do not implement H3 candidate ranking, first-action execution, or full closed-loop success during the **first-week deliverable** unless explicitly authorized after H2 is reviewed.

---

# 2. Model mechanics and conventions to confirm in the repository

The prior LeWM Push-T study reports a **192-dimensional** ViT-Tiny encoder representation, a ViT-S action-conditioned predictor, **three context frames** and **frame-skip five**. The original action plan notes a released Push-T evaluation setting of `horizon=5`, `receding_horizon=5`, and `action_block=5`. These are **expected source-informed values, not permission to hard-code incompatible shapes**. Verify all details in the exact checkout/checkpoint currently provided.

**Two horizon systems must remain separate:**

| Prediction point | Predictor lookahead `k` | Absolute `h` from first context frame (in raw environment steps) |
|---|---:|---:|
| Last context frame | 0 | 10 |
| 1 predicted step | 1 | 15 |
| 2 predicted steps | 2 | 20 |
| 3 predicted steps | 3 | 25 |
| 4 predicted steps | 4 | 30 |
| 5 predicted steps | 5 | 35 |

The first three retained context frames occur at absolute `h = 0, 5, 10`. Each predictor step consumes the appropriately aligned block of **five raw environment actions**; a five-step prediction therefore covers **25 raw environment steps beyond the last context frame**. Do not confuse raw action timesteps, predictor steps, replanning intervals, frame subsampling, or the absolute indexing convention.

**Before any experiment, record:** exact checkpoint identity (path and ideally hash), git revision, software/environment versions, GPU/CPU/device and precision, encoder output shape/normalization, predictor input and context shapes, action semantics and scaling, image rendering/preprocessing, simulator state variables, goal encoding, planner cost, and coordinate units (`px`, normalized coordinates, radians, etc.).

**Important technical risk from the original plan:** a Push-T environment's public seven-dimensional observation need not contain the full simulator dynamics state. A repository's `_set_state()` may restore only part of it or take a physics step during assignment. This behavior **must be checked in code and empirically tested**. Do not assume arbitrary state restoration is exact.

---

# 3. Overall workflow and experimental design

1. Restore original LeWM/Push-T codebase and freeze the reproducible baseline.
2. Validate simulator snapshot/restore and identical-action deterministic replay.
3. Build two distinct counterfactual datasets: **A/B-only** and **A/B/C compositional**.
4. Encode natural A/B/C observations and **dynamically consistent three-frame latent histories**.
5. Train/validate a frozen agent-position linear Ridge probe (H1); reuse calibrated block probes.
6. Build a candidate position subspace from the linear probe; run low-rank latent swaps and **input/selectivity/manifold checks**.
7. Perform **H2a A/B predictor-dependence** diagnostics and **H2b A/B/C physical-fidelity** evaluation with matched actions, real simulator trajectories, and controls. **First-week completion point.**
8. **Phase II:** fixed-candidate decision ranking and regret in real physical **C**.
9. Attribution controls, held-out replication, and matched-pair statistics (core controls integrated into Phase I; extended study in Phase II).
10. **Phase II:** first-action physical evaluation, then optionally full closed-loop success with a specified patching protocol.

**Why two dataset regimes?**

- **A/B-only:** A and B differ **only in agent position**. Their encoded difference is relatively clean for detecting predictor sensitivity to the candidate position subspace. We retain the original A/B question and the `r_P` vs `Phi_out` diagnostic.
- **A/B/C compositional:** A and B differ in **both agent and block positions**, while **C is a physically constructed hybrid** containing B's agent position and A's block position. A targeted patch of A using B should approximate the naturally encoded **C**, **not B**. This tests whether the method can selectively combine representations rather than merely move all predictions toward the donor scene.

**Do not conflate `B` between the two regimes**: the B in A/B-only has A's block position, while the B in A/B/C does not. Each record must carry an explicit `experiment_type` and `group_id`.

## 3.1 Definitive scenario definitions

Use different values for agent and block positions in the A/B/C group:

| Scene | Agent position | Block position | Initial velocities | Goal / physics |
|---|---|---|---|---|
| **A** | `a1` | `b1` | zero | shared |
| **B** | `a2` | `b2` | zero | shared |
| **C** | `a2` | `b1` | zero | shared |

For **A/B-only**, use A=`(a1,b1)` and B=`(a2,b1)` with every other physical variable matched. Note that the A/B-only B is geometrically equivalent to C **for matching coordinates**, but it is a **separate, purpose-labelled cohort**. Keep the experimental conditions distinct in data and plots to prevent misleading denominators/comparisons.

Other held-constant variables should include block angle/orientation, goal, velocities, controller state, physics parameters, and all simulator attributes relevant to the subsequent dynamics. Agent and block positions should be physically valid, non-interpenetrating, within workspace limits, and initially non-contacting. Initially use quiescent scenes, then stress-test moving/contact states only after basic validity is established.

---

# 4. Detailed execution steps

## Step 1 — Restore LeWM codebase; establish a frozen baseline

**Original schedule:** Day 1, about half a day to one day. **Status:** required in Phase I.

- Discover available code for the released checkpoint, encoder, predictor, preprocessing, Push-T environment, existing probes/diagnostics, and original CEM planner. **Reuse existing implementations** before writing replacements.
- Evaluate the same observations and action blocks twice in inference mode; verify identical output under a fixed seed and deterministic settings or explicitly quantify any remaining nondeterminism. Assert correct shapes, finite values, no NaNs.
- Run original CEM on 10 fixed-seed episodes as a smoke test and, if affordable, 50 episodes following the released evaluation protocol. Compare against **appropriate** published/repository baseline without promising exact match under different versions/hardware.
- Freeze and record the time-coordinate-action contract described in Section 2. Store seed, configuration, checkpoint identity, version information, and a concise baseline record.

**Acceptance gate:** verified model load and repeated rollout behavior; explicit map between time conventions, action blocks, latent step and simulator steps; recorded immutable baseline. If the baseline cannot be reproduced, stop and report the exact blocker before introducing interventions.

## Step 2 — Deterministic Push-T simulator snapshot/restore

**Original schedule:** Days 1–2. **Status:** required in Phase I.

**Task:** Given the same full physical state `s` and the same raw action sequence `u[0:T]`, the restored simulator must produce the same per-timestep physical trajectory within measured numerical tolerance.

1. Audit existing reset and `_set_state` implementations, including side effects such as stepping the physics engine. Identify all required dynamic state: pusher and block positions, orientation, velocities, angular velocity, contact or constraint state when needed, controller state, physics parameters, goal, and stochastic state.
2. Prefer the repository's reliable full-state restore if it is exact. Otherwise implement/encapsulate a reproducible, validated fresh-environment initialization or complete snapshot/restore path; **do not covertly modify dynamics**.
3. Restore -> execute fixed actions -> save **every** simulator timestep. Restore again -> replay identical actions -> compare full physical state trajectories, not only final images.
4. Test easy stationary/no-contact states first, then moving or just-contacted states as a stress test. If dynamic restore fails, restrict the initial dataset explicitly to verified quiescent conditions.
5. Assert state readback after reset/restore before encoding, and detect unintentional position drift, collision/penetration, extra physics stepping, and goal changes.

**Acceptance gate:** reproducible trajectories at a stated physical tolerance, reliable initialization of matched quiescent scenes, and a unit/integration test that catches broken restore behavior. Do not collect “ground truth” before this passes.

## Step 3 — Counterfactual datasets: original A/B **plus A/B/C**

**Original schedule:** Days 2–3. **Status:** required in Phase I. **This step is intentionally updated.**

### 3.1 Original A/B-only experiment (retain)

Let `s_A=[p_agent,p_block,theta_block,v_agent,v_block,omega_block,...]` with initially zero velocities and no contact. Construct `s_B` so that:

\[
p_{\mathrm{agent}}^B=p_{\mathrm{agent}}^A+\delta,\qquad
p_{\mathrm{block}}^B=p_{\mathrm{block}}^A,\qquad
\theta_{\mathrm{block}}^B=\theta_{\mathrm{block}}^A.
\]

Start with **candidate** displacements around 10, 20, and 30 pixels if those units match the current environment; adapt only based on physical validity/validation data, not held-out outcome. All other simulator and goal properties must remain matched. Avoid borders, overlap, and unintended initial contacts.

For each A/B pair, restore A and B separately and execute **the exact same precommitted action sequence**. Save the entire state and contact trajectories. This provides the cleaner geometry for H2a predictor dependence and free-space vs near-contact tests.

### 3.2 Compositional A/B/C experiment (add)

Construct an independent triplet:

\[
A=(a_1,b_1),\qquad B=(a_2,b_2),\qquad C=(a_2,b_1),
\]

where ordered pairs denote `(agent position, block position)`. Use the simulator to **physically create all three natural scenes**, not latent interpolation as a substitute. Verify `a1 != a2` and `b1 != b2` within meaningful physical tolerances. Keep block angle, goals, physics settings, velocities, and other relevant state consistent.

For each triplet, save:

- complete physical snapshots of A, B, and C and their validation readbacks;
- independently valid rendered observation sequences and three-frame histories for A/B/C;
- one shared, recorded action sequence (with unambiguous raw steps / action blocks);
- **three actual simulator trajectories** obtained by replaying that same action sequence from A, B, and C;
- complete per-step physical state, optionally contact events, and any diagnostic flags.

**Core comparison:** a position-subspace patch from A using B should produce predictions whose physical consequences resemble **C**, not all of B, because C retains A's block state.

### 3.3 Scene categories and sample sizes (retain)

- **Free-space / negative-control:** pusher well away from the T-block. Moving its starting position should not spuriously move/rotate a distant T-block.
- **Near-contact / task-relevant:** pusher close to, but not touching, the block. A position change may alter subsequent contact and block motion for the same action sequence.

**Pilot:** approximately **20–30 independent A/B pairs and 20–30 A/B/C triplets** as feasible; original plan proposed ~20–30 initial matched examples followed by **50–100** after validation. Treat these as planning targets, not guaranteed statistical power. Reserve disjoint held-out **entire groups** for final evaluation, and keep sufficient diversity of block pose, agent location, displacement direction, and contact opportunity. Do not split members of a triplet across train/test.

**Critical experimental design warning:** an action sequence that never brings either pusher into contact may yield near-zero block counterfactual effect, especially in free space. Such cases are still valid negative controls but are **uninformative about downstream block-effect fidelity**. Record ground-truth effect magnitude and stratify these cases instead of counting all zero-effect cases as physical success.

**Acceptance gate:** matched and physically valid states, deterministic replay, correct A/B and A/B/C semantic labels, genuine C simulator trajectories, no train/test family leakage.

## Step 4 — Encode A/B/C and capture consistent latent histories

**Original schedule:** Day 3. **Status:** required in Phase I. **This step is intentionally updated.**

Encode each naturally rendered scene with the **frozen encoder**:

\[
z_A=E(o_A),\qquad z_B=E(o_B),\qquad z_C=E(o_C).
\]

Keep the corresponding full simulator physical-state labels; do not assume the raw difference `z_B - z_A` is an isolated “agent-position feature” because the encoder may be entangled.

The predictor requires three temporally ordered representations. For each scene `S in {A,B,C}`, save:

\[
H_S=[z_{t-2}^S,z_{t-1}^S,z_t^S]
\]

and their aligned historical action inputs. Construct each history from its **own physically consistent simulator evolution**, preferably stable, non-contact histories in the first pilot. Confirm any intended stationarity by reading back velocities/positions over the history and ensure all context frames correspond to `h=0,5,10` under the exact data/preprocessing convention.

**Do not** build the main experiment by jumping the pusher only in the final context frame. Do not splice histories whose actions, time indices, and physical motion histories disagree. The later patch must use three correspondingly indexed context frames and coherent action-history inputs.

**Acceptance gate:** natural A/B/C encodings and aligned, dynamics-consistent histories usable by the exact frozen predictor interface.

## Step 5 — Agent-position representation probe (H1)

**Original schedule:** Day 4. **Status:** required in Phase I; otherwise unchanged.

Use Ridge regression as the default linear probe:

\[
\hat p_{\mathrm{agent}} = W z + b,\qquad W\in\mathbb R^{2\times d}.
\]

The measured target is agent `(x,y)` in an explicitly documented coordinate system. The **actual LeWM latent dimension should be inspected** (prior study reported `d=192`; the original plan used `128` only as an illustrative example).

- Train on natural frozen encoder latents, not patched or predictor-generated latents.
- Split **by episode / initial state / triplet family**, not frame; the same A/B or A/B/C family must not leak across partitions. Reserve final intervention triplets completely for held-out reporting.
- Tune Ridge regularization only on training/validation sets, then freeze `W` and `b` before inspecting held-out causal outcomes.
- Report agent-position **RMSE** and **R²**, ideally coordinate-wise plus 2D Euclidean RMSE and confidence intervals as useful.
- Reuse/revalidate existing calibrated block-position and block-angle probes as **non-target readouts**. Note the probes' own error when deciding whether patching is selective.

**Scientific interpretation:** accurate probing establishes **decodability** only. It does **not** prove predictor use or a disentangled causal variable.

**Acceptance gate:** saved, frozen linear position probe; held-out accuracy/error and probe details; calibrated non-target readouts. If H1 fails, investigate coordinate scaling, split, latent shape, preprocessing, and probe capacity before proposing H2.

## Step 6 — Probe-defined position subspace and latent intervention

**Original schedule:** Days 5–7. **Status:** required in Phase I. **Original swap retained; two new explicit checks added.**

### 6.1 Candidate position subspace (retain)

For the trained Ridge weights `W`, obtain an orthonormal basis `Q` for its **row space** via an appropriate rank-aware SVD/QR (account for numerical rank), so:

\[
P=QQ^\top,\qquad P^2=P,\quad P^\top=P.
\]

`P` acts in the encoder's latent space and its rank is usually **at most two** for the 2D position probe. For efficiency, compute `Pz = Q(Q^T z)` without allocating a dense `d x d` matrix if unnecessary. Note whether latent normalization/transforms were applied and maintain consistent units. Use the same frozen `P` for all conditions.

**Important:** this is a **probe-defined candidate position subspace**, not an established disentangled causal representation.

### 6.2 Original low-rank swap (retain)

\[
\boxed{\tilde z_A=(I-P)z_A+Pz_B
=z_A+P(z_B-z_A)}
\]

Applied to the A/B-only data, the intended latent-change target is B's agent position (B has the same block state). Applied to the A/B/C triplets, the intended compositional target is **C = `(a2,b1)`**, **not** B = `(a2,b2)`.

### 6.3 Pre-predictor checks (retain and extend)

**Target variable:** `g_agent(tilde z_A)` should be close to `a2` within probe measurement limitations.
**Non-target variables:** `g_block_pos(tilde z_A)` should remain close to `b1`; `g_block_angle(tilde z_A)` should remain near A's angle. Do not accept a patch that translates or rotates the block spuriously.
**Distribution:** compare patched latents against natural encoder latents using norms, local-nearest-neighbor distance, and (optionally) regularized Mahalanobis distance. **No single scalar proves manifold membership.** Evaluate on held-out natural latents when possible; indicate selection of reference population.

### 6.4 **NEW Check 1 — Input Difference Ratio**

For a paired A/B comparison, calculate:

\[
\boxed{
r_P=\frac{\lVert P(z_B-z_A)\rVert_2^2}{\lVert z_B-z_A\rVert_2^2}
}
\]

Compute on **both datasets**, reporting their distributions **separately**:

1. `AB_only`: only the agent position changes between A and B.
2. `ABC_compositional`: both agent and block positions change between A and donor B; real target C is constructed separately.

Optionally compute per-context-frame `r_P`, not merely for the last context frame. Handle vanishing `||z_B-z_A||²` with an explicit denominator tolerance, counts of skipped/undefined cases, and **no fabricated values**. With a correct orthogonal projector and finite nonzero denominator, `r_P` should lie within `[0,1]` up to numerical tolerance.

Interpretation:

- `r_P ≈ 1`: the probe-defined subspace spans almost all of this **latent difference**.
- `r_P << 1`: it spans only a fraction of this latent difference.
- **Not** a fraction of true physical information, an estimate of causal influence, or a selectivity guarantee.

Do not equate `r_P` with an effect on prediction: that is separately measured by `Phi_out` in Step 7.

### 6.5 **NEW Check 2 — Intervention Selectivity against real C**

In **A/B/C triplets**, verify at the input (before prediction):

\[
\boxed{
g_{\mathrm{agent}}(\tilde z_A)\approx a_2,
\qquad
g_{\mathrm{block}}(\tilde z_A)\approx b_1
}
\]

Also verify block angle preservation and compare `tilde z_A` to the **natural encoded C**, `z_C` (latent distance and decoded state). Report selective-edit error **relative to probe noise/baseline**; include natural A, B and C readouts as calibration references.

**Do not infer genuine variable-selective editing solely from these probes**: their own inductive biases, distribution shift, or an out-of-manifold patched vector may obscure other physical information. Combine selective readouts, natural-C comparisons, distribution diagnostics and controls, then qualify conclusions accordingly.

**Critical first-week gate:** If selectivity fails, **stop causal interpretation of H2**, inspect probe row space, donor entanglement, patch magnitude, temporal consistency, and off-manifold behavior. Do not rush to ranking.

### 6.6 Consistent-history swapping (retain)

A single-frame patch is allowed as a **mechanical sanity check**, not primary causal evidence. For actual predictor comparisons, intervene at all three matched history indices:

\[
\tilde z^A_{t-i}=z^A_{t-i}+P(z^B_{t-i}-z^A_{t-i}),
\qquad i\in\{0,1,2\}.
\]

Build `tilde H_A` from the three patched vectors. Confirm the A/B/C natural histories are dynamically valid and that A and donor B used compatible historical action conventions. Keep the same history actions and the same future candidate action sequence across predictor branches. If A and B's past actions are physically incompatible, resolve the matched-history protocol before interpreting outputs.

### 6.7 Baselines and perturbation controls (retain)

Use at minimum: original A; patched A; original donor B; **natural C (for ABC)**; zero-subspace patch; full-latent swap from A to B; same-rank **and same injected latent-difference norm** random-subspace swap; and non-target physical readouts. Full-swap into B is a code-path control, **not** the correct compositional C target. Random-subspace baselines must be faithfully norm-matched to the actual patch, with the matching method and degenerate cases documented.

**Acceptance gate:** reliable projector/swap arithmetic, `r_P` on both datasets, agent-selectivity with block preservation on independent ABC triplets, distribution diagnostics, and controls. Do not proceed to H2 claims if the intervention fails this gate.

## Step 7 — Physical counterfactual fidelity (H2; Phase I primary endpoint)

**Original schedule:** Days 7–9, now targeted as the **first-week endpoint** after Steps 1–6 pass. **This step is intentionally updated into two complementary experiments.**

### 7.1 Shared rollout protocol (retain)

Use the same frozen encoder, predictor, future action sequence and goal/physics convention in all conditions. Use **three-frame histories**, not inconsistent one-frame teleports. Write `F^k(H,u)` for the action-conditioned predicted latent at lookahead `k`; `u` is a fully specified fixed raw-action sequence aligned to action blocks. For physical readout, `g` is a previously validated/frozen probe.

For each `k = 1..5` (mapped to absolute environment steps `h=15,20,...,35`), compute predictor outputs aligned to the corresponding **real simulator future**. Save intermediate outputs, not just the final point; optionally inspect longer horizons as secondary stress tests, distinctly marked.

### 7.2 **Experiment 1 — A/B Predictor Dependence (H2a)**

**Dataset:** A/B-only pairs, where A and B differ solely in agent position. Use natural `H_A`, natural `H_B`, and `tilde H_A` patched from A with B.

For a specified, documented latent-space distance `d` at each lookahead `k`:

\[
\boxed{
\Phi_{\mathrm{out}}(k)=1-
\frac{d(F^k(\tilde H_A,u),F^k(H_B,u))}
{d(F^k(H_A,u),F^k(H_B,u))}
}
\]

Report **jointly with `r_P`**, including a per-pair scatter plot and, if useful, horizon-specific summaries. `Phi_out` measures whether the patch moves model predictions toward the model's **natural B prediction**. It is a *predictor-dependence* diagnostic, **not proof of physical fidelity or causality**.

**Numerical rule:** if the denominator `d(F^k(H_A,u),F^k(H_B,u))` is near zero, mark `Phi_out` as undefined and report how many; do not hide it with arbitrary epsilons or clip finite values into `[0,1]`. `Phi_out` may be negative if patching moves further away. Record the metric scaling and exact norm/distance definition.

For A/B/C data `r_P` is also recorded, but **do not use donor B as the physics target of the ABC primary analysis**.

### 7.3 **Experiment 2 — A/B/C Physical Fidelity (H2b; PRIMARY)**

**Dataset:** compositional A/B/C triplets, where A=`(a1,b1)`, B=`(a2,b2)`, and physically constructed C=`(a2,b1)`.

Predict under identical actions from:

\[
F^k(H_A,u),\qquad F^k(\tilde H_A,u),\qquad F^k(H_C,u),
\]

while simulator replay supplies true physical trajectories:

\[
c_A^{\mathrm{real}}(k),\qquad c_C^{\mathrm{real}}(k).
\]

For each target physical variable (agent position, block position, and separately block orientation as appropriate), compute the **real physical effect**:

\[
\Delta c_{\mathrm{real}}(k)=c_C^{\mathrm{real}}(k)-c_A^{\mathrm{real}}(k),
\]

the **internally induced predicted effect**:

\[
\Delta\hat c_{\mathrm{patch}}(k)=
g(F^k(\tilde H_A,u))-g(F^k(H_A,u)),
\]

and the **primary error**:

\[
\boxed{
E_{\mathrm{effect}}(k)=
\left\|
\big[g(F^k(\tilde H_A,u))-g(F^k(H_A,u))\big]
-\big[c_C^{\mathrm{real}}(k)-c_A^{\mathrm{real}}(k)\big]
\right\|
}
\]

For block angle, use a stated wrapped angular-difference convention; do not subtract angle values across a branch cut and present that as a large physical effect. Do not pool incompatible units (px, normalized distances, radians) into one norm without explicit, justified normalization. Report per-variable effects/errors and physically meaningful stratifications (free-space, near-contact; low vs nonzero real effect).

**Natural-C predictor baseline is mandatory:**

\[
\boxed{
E_C^{\mathrm{abs}}(k)=
\left\|g(F^k(H_C,u))-c_C^{\mathrm{real}}(k)\right\|
}
\]

For a like-for-like effect comparison, also consider reporting:

\[
E_{\mathrm{effect},C}(k)=
\left\|\big[g(F^k(H_C,u))-g(F^k(H_A,u))\big]
-\big[c_C^{\mathrm{real}}(k)-c_A^{\mathrm{real}}(k)\big]\right\|.
\]

That additional effect baseline separates whether an error comes from **the patched input** or from **the frozen predictor's inability to predict C even with a natural C encoding**. A bad natural-C prediction limits what can legitimately be concluded about the patch. Also include same-rank, matched-norm random-subspace controls.

**Interpretation boundaries:**

- A smaller `E_effect` than controls **supports** selective physical-effect fidelity under the tested distribution and readout; it does not establish uniquely causal internal coordinates.
- Patched A resembling donor **B** is *not* success for this compositional experiment: **C** is the target.
- A low total effect error when ground-truth effect is nearly zero is **not** proof of correct contact dynamics. Stratify by effect strength; track direction and magnitude where the real effect is nontrivial.
- Probe errors on predicted/patched latents may themselves change under distribution shift; report probe calibration and latent-distribution diagnostics.
- The earlier LeWM study observed one-step under-updating even under teacher forcing; any poor physical fidelity may reflect a frozen predictor limitation rather than selective edit failure. Keep these mechanisms separate.

### 7.4 Mandatory figures / reporting for first week

| Figure | Required content | Primary question |
|---|---|---|
| **Figure 1 — Input difference ratio** | `r_P` distributions for **AB-only** vs **ABC donor A–B** (optionally per history index/displacement/group) | How much of each encoded A–B difference lies in the probe-defined subspace? |
| **Figure 2 — Intervention selectivity** | Input readouts of **agent position**, **block position** and **block angle** for natural A/B/C, patched A and controls; include patched-vs-C errors and latent-manifold diagnostics | Did we selectively construct agent=`a2`, block=`b1`, rather than accidentally recreate donor B or corrupt state? |
| **Figure 3 — A/B dependence** | `r_P` vs `Phi_out`, with horizon-specific outcomes and undefined-denominator counts | Does the candidate subspace influence predictor outputs? |
| **Figure 4 — A/B/C physical effect fidelity** | `E_effect(k)` across `k=1..5` for patched A and random controls, alongside **real C**, natural-C prediction/error baseline, and relevant real/predicted agent/block trajectories; report effect strength and uncertainty | Does the selected internal edit match the real physics of new state C, beyond what the frozen model itself can predict? |

**Statistical reporting:** show individual paired cases where useful, aggregate by triplet (not individual frames), include sample sizes and held-out group/seed identities, and use paired resampling/bootstrap **across independent groups** to estimate intervals. Avoid reporting arbitrary significance without a proper sampling unit and predeclared outcome. Save machine-readable per-group/per-horizon measurements (not just plots).

**Phase I success criterion:** The important achievement is **not** making patched A's prediction look like **B**; it is showing that the intervention selectively changes the **agent-position representation** while preserving A's block state, and testing whether that new latent state yields the same physical consequences as **real C**, under fixed actions. If selectivity fails, **halt causal interpretation** and do not advance to H3 merely to meet a deadline.

## Step 8 — Fixed-candidate ranking / decision fidelity (H3, **PHASE II ONLY**)

**Original schedule:** Days 9–11; **now deferred until H2 review**. The **method remains unchanged** except **all physical target/comparison references switch from B to C**.

- Create a fixed candidate pool `A_actions` (pilot approximately 16–32 candidate sequences per state), combining broad random candidates and late CEM-distribution candidates. For each triplet, **use identical candidate sequences** for A, patched A, and C, under the same goal `z_goal = E(o_goal)` and actual planner cost definition.
- Score candidates from `H_A`, `tilde H_A`, and `H_C` using **the repository's frozen planner score** (an endpoint latent L2 distance is illustrative, not authority to change the official objective).
- Execute each candidate sequence from the **same real simulator C state** to measure `J_real^C(action)` under the frozen official evaluation definition. Separate real cost, goal progress, and binary task success rather than conflating them.
- Compare ranking alignment of patched A versus naturally encoded C, and both against **real-C ranking**.
- Define `a*(z) = argmin_{a in A_actions} J_model(z,a)` and:

\[
\boxed{
R_C(z)=J_{\mathrm{real}}^C(a^*(z))-
\min_{a\in\mathcal A}J_{\mathrm{real}}^C(a)
}
\]

- **Required three conditions:** `R_C(z_A)`, `R_C(tilde z_A)`, `R_C(z_C)` (with notation referring to their **full consistent histories** in code).
- Analyze wide-margin vs near-tie candidate distinctions: bin by real-C cost margin `|J_real^C(a_i) - J_real^C(a_j)|`. A near-tie failure might reflect cost geometry or optimizer bias, not agent-position representation.

**Intended outcome:** determine whether patched A's choices resemble choices under natural C **and** reduce real C candidate-set regret relative to unpatched A. Physical H2 validity does not automatically imply decision improvement.

## Step 9 — Controls, attribution, and replication

**Original schedule:** Days 11–12 for extended attribution; **key controls are mandatory in Phase I** and later replicated in Phase II. **This step is intentionally updated.**

| Control / validation | What it isolates | Phase I requirement |
|---|---|---|
| **Zero-subspace (`P=0`)** | No-edit identity test: output must equal original A | Yes |
| **Full-latent (`P=I`)** | Full swap must reproduce donor B's natural path with equivalent aligned inputs; **does not represent C for ABC** | Yes |
| **Random subspace of same rank + same patch norm** | Whether changes arise from indiscriminate latent perturbation of comparable strength | Yes |
| **Non-target block position and angle probes** | Leakage or corruption of physical variables intended to be held fixed | Yes |
| **Natural, real C encoder + predictor** | Baseline for achievable physical accuracy without intervention | Yes |
| **True simulator C** | Actual physical counterfactual ground truth under exactly matched actions | Yes |
| **Held-out A/B/C families** | Generalization beyond fitting/selecting the probe or intervention | Yes |
| **Irrelevant-variable perturbation if feasible** | Whether task-irrelevant changes induce spurious effects | Desirable |

`P=0` and `P=I` validate implementation, **not scientific causal validity**. The central attribution test asks whether a probe-defined position patch is more selective and more physically faithful to **C** than suitably scaled random changes, with the same future actions and correct natural-C baseline.

Report descriptive uncertainty (paired resampling/bootstraps across groups), failures, denominator exclusions, OOD flags, and pre/post-intervention physical readout errors. Do not tune on final held-out triplets.

## Step 10 — First-action and full closed-loop evaluation (**PHASE II ONLY**)

**Original schedule:** Days 13–14 or later; unchanged conceptually except all physical decision comparisons now occur in **real C**.

1. **First-action evaluation:** Choose the first action block using mismatched `H_A`, patched `tilde H_A`, and naturally encoded `H_C`, all under the same candidate pool and goal. Execute their chosen first blocks **from the same actual C simulator state**; compare real cost change, block motion, and contact outcome up to the next replanning point.
2. **Full closed loop (optional after H3):** Specify whether the patch happens **only at initial planning** or is consistently reapplied at each replanning step. Re-encoding real observations can erase a one-time patch, so an unspecified closed-loop protocol is not interpretable. Keep initial C, goals, seeds and observation-treatment policy comparable.
3. Separate decision ranking, physical regret, binary task success, and success prediction/calibration. A success-rate change alone does not prove a physical representation is necessary.

---

# 5. Implementation contract for Codex

This section specifies **interfaces and invariants**, not assumed existing module names. Adapt to the actual repository architecture after inspection; prefer minimal wrappers around its existing primitives.

## 5.1 Audit BEFORE implementation

Find and document:

1. **Model:** checkpoint loader, encoder interface, latent shape, normalization, predictor forward/rollout input order, historical actions and goal interface.
2. **Environment:** Push-T constructor, deterministic seed path, `_set_state`/reset/physics stepping behavior, complete dynamic state available, action semantics, rendering, contact extraction.
3. **Existing work:** earlier DICTA probing splits and saved probes, trajectory evaluation code, CEM planner, baseline evaluation configs, dataset format.
4. **Output conventions:** project layout, dependency management, test style and command, existing artifact directory, source of seed/config options.

Create a short **repository audit report** with file/function paths, verified settings, observed gaps, and any blocking ambiguities. If a critical scientific requirement cannot be implemented faithfully, **do not substitute an approximation without explaining and obtaining approval**.

## 5.2 Functional components (conceptual)

Prefer to implement/reuse thin, testable functions for these operations:

- `capture_full_state(env) / restore_full_state(env, snapshot)`: exact simulator state management (or verified fresh-state constructor).
- `replay_fixed_actions(snapshot, actions, seed)`: full per-raw-step state trajectory and selected observations/events.
- `make_ab_pair(...)` and `make_abc_triplet(...)`: physically valid scenes with detailed audit checks.
- `build_consistent_history(scene, aligned_past_actions, ...)`: three natural context frames and actions matching model conventions.
- `encode_histories(...)`: frozen encoder output and physical labels.
- `train_or_load_agent_ridge_probe(...)`, `load_calibrated_block_probes(...)`: freeze before held-out inference.
- `position_basis_from_probe(W)`: rank-aware orthonormal `Q`; no unconstrained dimensional edits.
- `patch_latent(A,B,Q) = A + Q(Q^T(B-A))`; `patch_history(...)`: same position basis across three frames.
- `input_difference_ratio(A,B,Q)`; `selectivity_metrics(...)`; `natural_latent_distance(...)`.
- `predict_fixed_actions(history, action_blocks)` with precise horizon indexing.
- `predictor_dependence_phi(...)`; `physical_effect_error(...)`; `natural_c_baseline_error(...)`.
- `evaluate_matched_controls(...)`, `aggregate_paired_results(...)`, `render_figures_1_to_4(...)`.

**These are illustrative interface names, not a mandate to rename the codebase.** Maintain compatibility and avoid unrelated refactors.

## 5.3 Recommended portable data schema

Persist one record per independent **pair/triplet**, with structured arrays/tensors as appropriate:

```text
experiment_type: "AB_only" | "ABC_compositional"
group_id, split: train | validation | test, seed
code_revision, checkpoint_id, simulation_config, encoder_config
coordinate_units, angle_units, action_semantics, frame_skip, context_indices
physical_states: {A, B, [C]}    # full initial snapshots and readbacks
observations: {A, B, [C]}      # 3 context frames per scene
history_actions: {A, B, [C]}   # aligned, validated convention
latents: {A, B, [C]}           # 3 natural encoded context vectors each
fixed_future_actions: ...      # raw env actions; block form if needed
real_trajectories: {A, B, [C]} # physical states per raw timestep
predicted_latents: {A, patched_A, B, [C], random_controls}
input_metrics: {r_P, selectivity, natural_C_distance, latent_shift}
rollout_metrics: {Phi_out, denominator_flags, E_effect, E_C_abs, ...}
validation_flags: {snapshot_replay, no_contact_initially, bounds, matched_goal, ...}
```

Use a traceable serialized format already supported in the repo (`npz`, `h5`, `parquet`, etc.) plus a manifest/config; do **not** assume JSON can represent large arrays efficiently. Paths are to be decided based on real repository conventions, not imaginary source-tree names. Keep plots and summary tables reproducible from stored numeric outputs.

## 5.4 Unit and integration tests that must pass

- Deterministic full-state replay: every physically relevant state component matches under identical restoration/actions, with declared numerical tolerances.
- Physical validity: A/B-only changes **only** agent location; ABC matches positions `(a1,b1),(a2,b2),(a2,b1)` and preserves all required other state/goal properties.
- All three context histories have valid temporal/frame-skip/action alignment; no single-frame teleportations in primary H2.
- Encoder/checkpoint frozen; repeated natural encoding stable; model inputs and outputs finite and correctly shaped.
- Projector arithmetic: `Q^T Q ≈ I`, `P²≈P`, `P^T≈P`; rank ≤2; `r_P` matches reference computation and is in `[0,1]` when defined.
- Zero patch is exactly/noisily equal to A as numerically expected; full swap equals donor B (with same aligned history inputs); linear probe identity/readout checks.
- Random controls match chosen **subspace rank and injected-difference norm**; explicitly flag impossible/degenerate matches.
- `Phi_out` formula verified against small synthetic numerical cases, with near-zero denominators flagged, not silently stabilized.
- `E_effect` synthetic test with known true deltas, plus proper angle wrapping and different units handled separately.
- Held-out split isolation: no pair/triplet family straddles splits; final test set remains unseen by hyperparameter/model selection.
- Natural C ground-truth replay and natural-C predictor baseline present before scoring patched-C fidelity.

## 5.5 Reproducibility and reporting rules

- **Never invent experimental numbers, completed runs or paths.** Plot only measured results; label toy/synthetic test results explicitly.
- Use deterministic seeds wherever supported and log when nondeterminism persists. Show actual sample sizes, failed cases, reasons for exclusion, and time/compute budgets.
- Do not silently mix coordinate systems or normalize physical quantities differently across methods.
- Freeze chosen evaluation horizons, distance `d`, primary physical readouts and tolerances **before** final held-out result inspection. If changed after a pilot, record the change and re-evaluate only on untouched data.
- Aggregate uncertainty by **independent pair/triplet**; multiple frames/actions from the same state are not independent experimental replicates.
- Report both positive and negative findings, confounders, and separation of encoder/probe, intervention, and predictor limitations.
- Keep generated artifacts, scripts and tests clearly distinct from the original checkpoint and data; preserve clean git diffs.
- If asked to do “the whole plan”, still **stop after validated Phase I** unless the human explicitly authorizes H3/Phase II.

---

# 6. First-week execution order and deliverables (updated schedule)

These are **focused working-day targets**, contingent on gates; simulator fidelity and method validity take precedence over deadlines.

| Day | Work | Required deliverable / exit check |
|---|---|---|
| **1** | Audit repo, load frozen LeWM, baseline CEM/rollout checks, document time/actions/units | Baseline report + verified immutable config |
| **2** | Verify/implement full Push-T state initialization/restore and deterministic per-step replay | Repeatability tests + simulator contract |
| **3** | Generate both AB-only and ABC triplets, real C, fixed-action real trajectories, valid three-frame histories | Audited pilot datasets with group-wise splits |
| **4** | Reuse/train frozen Ridge agent-position probe and calibrated block probes | H1 held-out R²/RMSE; saved probes |
| **5** | Build `Q/P`, swap full histories, evaluate `r_P` in both cohorts and basic identity/full controls | Input-difference results + unit tests |
| **6** | Check compositional selectivity vs natural C, block preservation, manifold/nearest-neighbor diagnostics, matched-norm random controls | **Figure 1 & Figure 2**, selective-edit go/no-go |
| **7** | Run matched predictor/physical simulations and produce H2a dependence + H2b physical-fidelity analysis | **Figure 3 & Figure 4**, reproducible numeric outputs and H2 decision memo |

A necessary problem may delay any stage. It is better to complete a **validated negative or incomplete experiment with clear evidence** than to produce four visually convincing but scientifically invalid plots.

**Phase I required artifact checklist:**

- [ ] Repository audit and baseline configuration/report
- [ ] Simulator snapshot/restore and deterministic replay tests
- [ ] Matched physically valid AB-only and ABC dataset, held-out group manifest
- [ ] Frozen agent Ridge probe accuracy and non-target probe validation
- [ ] Patch code, projector/ratio/identity/random-control tests
- [ ] Dynamically consistent context histories with fixed action alignment
- [ ] Figure 1: `r_P` on AB-only and ABC
- [ ] Figure 2: selectivity (agent changes, block preserved) vs real C
- [ ] Figure 3: AB-only `r_P` vs `Phi_out`
- [ ] Figure 4: ABC physical effect fidelity vs real C / natural-C model / controls
- [ ] Machine-readable per-case results, plot regeneration commands, seeds, exclusions
- [ ] Short interpretation memo: what passed, what failed, confounders, recommended next experiment

**Codex should report per step:** (1) actual files changed; (2) commands/tests run and observed outputs; (3) verified assumptions; (4) research interpretation, if any; (5) blocking issues; (6) next recommended action. Do not claim a metric is supported until it has been computed on actual appropriate data.

---

# 7. Phase II original roadmap (preserved, with C replacing B)

After a human review of Phase I, continue the original ~14-workday program with the following revisions. Days 8+ are **not** part of the first-week requirement.

| Later day (approx.) | Work | Outcome |
|---|---|---|
| **8** | Refine intervention amplitude/history/horizon on validation data; replicate H2 | Stable compositional physical-fidelity findings |
| **9** | Fixed random/CEM candidate action pool, simulator C cost evaluator | Reusable candidate evaluator |
| **10** | H3 ranking/regret: A vs patched A vs natural C, all evaluated in **real C** | Decision-fidelity figure |
| **11** | Expanded attribution controls, cost margin / predictor failure analysis | Alternative explanations addressed |
| **12** | Replicate on held-out ABC groups; matched bootstrap intervals | Reliability analysis |
| **13** | First-action execution in **real C**; optional later closed-loop protocol | Physical decision-outcome comparison |
| **14** | Synthesize results and future experiments | 1–2 page technical memo for Tao Zhou / Ivan |

## 7.1 Decision rules (explicit go/no-go)

- **H1 fails:** check probe, encoder/data compatibility, coordinates and split; do not claim the latent lacks position information from one probe failure.
- **H1 passes, input selectivity fails:** stop causal interpretation. Inspect basis/normalization, potential block leakage, off-manifold displacement, history mismatch and random controls; H2/H3 remain unproven.
- **Input selectivity passes, H2b fails:** compare `E_C_abs`, full/zero/random controls, contact regime, probe OOD, fixed-action effects and the frozen model's intrinsic under-updating. A meaningful *negative result* is possible.
- **H2b passes, H3 fails:** examine goal/cost geometry, close-cost candidates and CEM, not automatically the absence of physical state. This separation is scientifically valuable.
- **H1/H2/H3 pass:** expand to independent scenes/checkpoints/physical variables and properly specified closed-loop validation.

The goal is **not** to prove that physical representations are important. It is to **test whether a readable physical variable can be selectively altered and whether its prediction and eventual decision effects agree with real physical counterfactuals**.

## 7.2 What to show collaborators after Phase I

A concise 1–2 page technical memo should include:

1. Main question and two dataset regimes (AB-only dependence; ABC compositional fidelity).
2. Exactly how full simulator states, matched actions, and consistent histories were built.
3. H1 probe accuracy; patch subspace rank; `r_P`; selectivity against C; distribution/OOD checks.
4. `Phi_out` as **dependence only**, not a physical proof.
5. Real A/C effects vs patched A and natural-C predictions across horizons, with controls and failures.
6. Clear limitations and whether Phase II H3 is warranted.

When Phase II is finished, add C-based action ranking and real-C regret; do not insert nonexistent planning results into the Phase I memo.

---

# 8. Source traceability and update log

**Source of original plan:** user-supplied *“最终研究行动计划：从 Representation Probing 到 Causal Intervention，再到 Planning”* (original 10-step Action Plan).
**Scientific baseline:** *Long-Horizon Rollout Diagnostics for Latent World Models: LeWM on Push-T*, DICTA 2026 (user-provided PDF and diagnostic blog).
**Approved modifications (2026-10-13):** exactly six targeted changes to the original plan:

1. Step 3: retain A/B; add physically constructed A/B/C triplets.
2. Step 4: encode natural C and its consistent three-frame history.
3. Step 6: retain original swap; add `r_P` and selectivity checks relative to **C**, plus distribution check.
4. Step 7: explicitly separate **A/B predictor dependence** (`Phi_out`) and **A/B/C physical effect fidelity** (primary H2).
5. Step 8: compare A, patched A and **C**; evaluate regret under **real C**; defer H3.
6. Step 9 / scheduling: add named controls and make **Step 7 / H2 the first-week deliverable**, with ranking/closed loop in Phase II.

Everything else in the original scientific workflow is retained in substance, with implementation-level reproducibility checks made explicit for Codex. Details such as actual module names, environment API, valid coordinates, checkpoint path, cost functions and dependency commands **remain to be verified in the repository**.
