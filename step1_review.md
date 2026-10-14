# Step 1 implementation review

Review date: 2026-10-14 (Australia/Melbourne). Scope: frozen baseline only.

Material provenance: local repository/source and measured HDF5/checkpoint outputs.
The academic-research-suite reproducibility workflow informs source tracing,
explicit failure records, and the distinction between seeded inference and
simulator reproducibility. No external model or service receives research data.

## Judgment

The original implementation establishes the correct core conventions and a real
10-episode official smoke, but its acceptance gate is incomplete. The fixes target
false passes, provenance loss and configuration mismatch. They preserve the released
checkpoint and all scientific hypotheses; Steps 2–7 and Phase II remain separate.

## Verified audit

- `jepa.JEPA.encode`: flattens B*T images, takes the ViT CLS token and applies
  `projector`; no additional latent L2 normalization. `predict` combines latent
  history/action embeddings, then applies `pred_proj`. `rollout` and `criterion`
  implement the existing official planner path; final cost is squared latent
  error summed over features at the final predicted step.
- `module.ARPredictor.pos_embedding` stores the three-frame limit;
  `module.Embedder.patch_embed.in_channels` checks the 10-D action token.
  Released model: 192-D latents, 18,034,478 parameters. The new harness checks
  the loaded object rather than accepting declared shapes alone.
- `utils.get_column_normalizer` and `train.lejepa_forward`: normalize raw 2-D
  actions using Torch mean/sample std before five-row reshaping. `train.py`
  turns padded NaNs into zero; the baseline selects a complete, unpadded window
  and rejects non-finite real actions.
- `probe.embedding_cache.get_image_transform/preprocess_pixels`: shared ImageNet
  transform. `probe.rollout_ablation.open_loop_rollout_from_embeddings`: reusable
  sliding three-frame/action-window recurrence. Older raw-action diagnostic
  encoding is bypassed by explicitly standardizing Phase I tokens.
- `probe.data.split_by_episode`: existing 80/10/10 complete-episode split, seed 42.
  Cache schema/index/membership checks reproduce this rule and check sampled labels.
- `eval.run`: uses one observed frame at initial official planning, horizon=5,
  receding_horizon=5, action_block=5; CEM samples=300, iterations=30, elites=30.
  Its sklearn action normalization is preserved. It passes the auto-selected
  device explicitly to solver construction, overriding the solver YAML device.
- `stable_worldmodel.envs.pusht.env.PushT.step`: relative action target is
  current agent position + 100*raw_action, with ten physics substeps per raw
  action. Public state contains agent xy, block xy/angle, and agent velocity.
  `_set_state` does not restore block velocity/angular velocity and takes a
  physics step. This is a Step 2 replay risk, not a completed replay guarantee.
- `World.evaluate_from_dataset`: the dataset lacks a seed column, so reset seeds
  are None. Global RNG and CEM seeding do not seed `default_rng(None)`.

## Findings and fixes

| Priority | Finding | Change and reason |
|---|---|---|
| High | Only one predictor call was repeated despite reporting rollout validity. | Repeat and persist all five autoregressive outputs at h=15..35 using the existing recurrence; assert shapes/finite values and the k=1 one-step agreement. |
| High | Missing files, malformed caches or early inference errors could escape without a manifest. | Create the manifest before model/data work; save progress and preserve typed errors plus tracebacks; strict JSON disallows NaN/Infinity. |
| High | Skipped CEM was reported as completed; exit 0 without metrics could pass. | Separate verification_only from the full gate; require ten boolean outcomes, consistent percentage, actual episode/start IDs, both model/solver devices, and evaluator result file. |
| High | Reusing run IDs could overwrite manifests or append stale CEM metrics. | Remove overwrite support, reject existing result or artifact directories, and reject traversal-shaped run IDs. |
| High | CEM hard-coded cache/device assumptions diverged from CLI configuration. | Quote and forward configured cache/dataset/image size; set the actual device hook and record observed placement. The earlier solver.device CPU override was ineffective because eval.py replaces it. |
| High | Embedding-only cache checks could accept mislabelled rows or the wrong episode split. | Validate schema, all-row indices and whole-episode membership/completeness, sampled labels and finite embeddings; compare only equal-shaped outputs. |
| Medium | Three fixed rows provide weak coverage; caches were fully loaded. | Fixed-seed stratified samples (16/split) and Torch memory mapping strengthen identity checks with modest inference cost. |
| Medium | Any checkpoint at a plausible path was accepted; dirty source was not reconstructable. | Pin the known checkpoint SHA, load its exact stem, check architecture/frozen state, capture pre-output git status, hashes and exact source/config bytes in HDF5, and recheck checkpoint after execution. |
| Medium | Cache recovery was absent. | Explicit mismatch-only HDF5 repair preserves original caches, streams batches, records the original failure, and revalidates the replacement. Valid caches are reused. |
| Medium | CEM stdout was buffered and could run indefinitely. | Write logs directly, record start/finish time and timeout, retain partial metric records. |
| Scientific | Process seeding was described too broadly as fixed-seed evaluation. | Record the official seedless simulator resets and defer physical reproducibility to Step 2. |

The independent-frame cache bug in the first implementation came from selecting
`[:, 0]` from a `(1,T,D)` output and broadcasting it against `(T,D)`. The ViT
encoder itself does not add temporal positions across T: it flattens images.
Separate batch/sequence layouts and strict output-shape checks prevent recurrence.

All seven action tokens for h=0..35 are encoded together. Encoding three versus
seven tokens in separate Conv1d calls can change floating-point reduction order;
the k=1 cross-check now uses the same encoded tokens on both code paths.

## Validation and interpretation

Direct tests cover exact standardization/token order, recurrent timing/feedback,
actual cache re-encoding, corrupted labels/splits/shapes, mismatch repair and
original-cache preservation, Hydra quoting, invalid CEM metric rejection, early
failure manifests, and immutable run reuse. Synthetic tests are labelled synthetic.

Measured integration output is saved under
`diagnostic_results/phase1/step1-review-contracts-seed42-v2`. It confirms frozen
architecture, finite outputs, zero repeat deltas over h=15..35, and sampled
identity for all caches. The skipped CEM is correctly marked verification_only.

The final reviewed run is
`diagnostic_results/phase1/step1-reviewed-baseline-seed42` (manifest schema 2).
It passes the complete Step 1 gate:

- Both frozen encoding/prediction passes and all five autoregressive outputs
  agree exactly; every measured output is finite.
- Sixteen sampled rows per split match fresh encodings. Maximum absolute
  differences: train `3.337860107421875e-06`, validation
  `3.337860107421875e-06`, test `3.5762786865234375e-06`.
- Official CEM: ten outcomes, nine successes, 90.0%; evaluation time
  `126.68543699994916` seconds on CPU. The observed solver uses samples=300,
  iterations=30, topk=30; plan horizon/receding horizon/action block are all 5.
- The captured 18 source/config files match both their hashes and current file
  bytes. The persisted HDF5 repeat rollouts are exactly equal at h=15..35.
- Both new direct-execution test scripts pass (16 test functions), as do all
  six existing rollout/latent-diagnostics scripts. Compile checks and
  `git diff --check` pass. Intentional failure tests produce failure records.

Final integration command:

```bash
MPLCONFIGDIR=/private/tmp/lewm-mpl .venv/bin/python scripts/run_phase1.py \
  --stage baseline --run-id step1-reviewed-baseline-seed42 --device cpu
```

The preceding revised smoke (`step1-review-baseline-seed42-v2`) also produced
9/10 with the same ten per-episode outcomes. These two runs share the same
starting points and are repeated checks, not 20 independent observations.
Failed review runs remain explicitly labelled failed with their manifests and
tracebacks: an overly strict cross-call floating-point comparison and Hydra's
module-based configuration lookup were corrected before final verification.
The k=1 comparison now shares the same action embeddings, and the imported
official Hydra entry point has script-style module metadata for file lookup.

Methodological scan covers all 11 ARS categories. No significance tests,
pre/post effect estimates, regressions, diagnostic predictive values or causal
estimates are reported here; their corresponding fallacies are inapplicable.
The relevant risks are filtered starting-point selection, repeated cases not
being independent, discarding failed runs, and interpreting smoke success as
causal validation. All are explicitly qualified or recorded in this report.

Cache identity remains sampled; no claim is made that every latent is verified.
Full 43-GB dataset hashing is intentionally omitted; dataset metadata and hashes
of sampled pixels identify the checked inputs. The source snapshot stores exact
inference/config bytes for dirty-source runs; dependency versions identify the
remaining installed runtime. Review edits are left uncommitted for inspection.
The original 90% on ten sampled starting points and repository 96% on fifty
starting points are different sample sizes and are descriptive functionality
checks, not evidence of an improvement or a regression.

The next scientific gate is full per-raw-step physical replay. Step 1 cannot
establish selective latent editing, H1/H2, manifold membership or physical
counterfactual fidelity. No inference from benchmark success advances H3.
