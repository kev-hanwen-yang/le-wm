# Codex Agent Instructions — LeWM Push-T Counterfactual Study

Read `CODEX_EXPERIMENT_PLAN.md` **in full** before making changes. That document is the authoritative research specification, including updated A/B/C counterfactual design, equations, staged requirements, and scientific go/no-go criteria.

## Mission and scope

Implement the **first-week Phase I experiment (Steps 1–7, H1/H2)** in the **existing** LeWM/Push-T research repository. The primary scientific test is whether a **Ridge-probe-defined agent-position subspace** can be selectively patched from scene B into scene A to approximate **real simulator scene C**, and whether the patched model's **future physical effects** agree with those obtained from C under the **same actions**.

Preserve the original A/B-only predictor-dependence experiment. For compositional triplets, A=(agent a1, block b1), B=(agent a2, block b2), C=(agent a2, block b1). **B is the latent donor; C is the target for physical fidelity.** Do not confuse this with A/B-only, where B differs from A only in agent position.

**Do not work on Step 8 action ranking, H3, Step 10, or full closed-loop planning unless the user expressly approves Phase II.** The first-week endpoint is a validated H2 analysis and four specified figures, not a higher task success rate.

## Required work style

1. **Audit before editing.** Identify the real encoder/predictor/checkpoint API, three-frame history and action-block alignment, Push-T state/reset/step semantics, existing probe/splits, CEM cost, and relevant tests. Provide a brief audit with actual file/function paths and discrepancies from the plan. Do not invent code paths or checkpoint/config values.
2. **Reuse, don't replace.** Prefer existing implementation and a small, tested evaluation harness. Do not retrain/fine-tune LeWM; do not modify physics, planner reward/cost, goal encoding, or official metrics to obtain desirable results. Avoid unrelated refactors.
3. **Validate simulator first.** Full physical state must restore reproducibly (or use a demonstrably equivalent fresh initialization) and yield identical per-raw-step trajectories for identical actions. A public 7D observation alone is not necessarily a complete snapshot; `_set_state` may step physics. Do not trust it without inspecting and testing.
4. **Maintain physically coherent histories.** The predictor uses three retained context frames and aligned historical actions. Construct independent physically valid histories for A/B/C, then patch corresponding position-related subspaces across **all three** context frames. A last-frame-only teleport is a sanity test, not the main experiment.
5. **Maintain honest experiment splits.** Split by whole pair/triplet/initial-state family, freeze probe and P before held-out causal evaluation, and never tune with held-out outcomes.
6. **Check selectivity before physical claims.** `g_agent(patched_A) ≈ a2`, `g_block(patched_A) ≈ b1`, angle stays at A, patched latent reasonably close to natural encoded distribution, and compare against natural `z_C`. Failure here is a **STOP** for H2 causal interpretation.
7. **Perform required matched controls.** Zero/full-subspace, random subspace matched for rank and *actual injected norm*, non-target probes, natural-C encoder+predictor, ground-truth simulator C, and held-out ABC groups. The full swap yields donor B, **not** hybrid C.
8. **Separate two scientific outcomes.** A/B-only `Phi_out` plus `r_P` measures **predictor dependence**; A/B/C `E_effect` against real C (with natural-C baseline) measures **physical fidelity**. Neither a good probe nor a good `Phi_out` alone establishes causality.
9. **Respect units and numerical edge cases.** Confirm k=1..5 vs absolute h=15..35, frame-skip/action-block definitions, angle wrapping, pixel vs normalized coordinates, and near-zero denominators in `r_P`/`Phi_out`. Do not silently epsilon-stabilize, clip, or combine unlike units. Record exclusions.
10. **Make results reproducible.** Log seeds, git SHA, checkpoint identity, config, scene family IDs, per-case results, actual commands, and failures. Add tests for physics replay, dataset invariants, projector algebra, history alignment, controls, and metric formulas. Generate plots from saved numeric outputs; do not fabricate measurements.

## Expected work sequence

- Audit & baseline → exact physical replay → AB-only / ABC dataset → consistent histories → Ridge H1 + non-target probes → low-rank patch + `r_P`/selectivity/manifold checks → A/B `Phi_out` + ABC real-C physical fidelity → figures and short research memo.
- **Checkpoint after each gate:** state tested facts, code/files changed, tests run, remaining uncertainty, and whether the next phase is valid. If a scientific assumption cannot be met, stop, explain the blocker and propose alternatives **without silently switching protocols**.

## First-week deliverables

- Figure 1: `r_P` distributions for AB-only versus ABC A/B-donor latent differences.
- Figure 2: patch selectivity for agent position, block position/angle and comparison to natural C (plus off-manifold checks).
- Figure 3: AB-only `r_P` versus `Phi_out`, with near-zero-denominator exclusions.
- Figure 4: ABC physical effect error across planning horizon, real C, natural-C prediction baseline and random controls.
- Reproducible tests, dataset, configs, numerical tables, plotting commands, and an evidence-based go/no-go memo on Phase II.

## How to start

Read the plan, inspect the repository, and report:

- Verified module/function paths, checkpoint/config, environment reset/restore behavior and existing baseline functionality.
- A minimal proposed implementation sequence, tests, any required dependencies, and scientific ambiguities that must be resolved.
- First smallest useful patch/test that can be made safely.

Then proceed incrementally **only if the assumptions have been verified**. Do not claim an experiment has run unless it really ran.
