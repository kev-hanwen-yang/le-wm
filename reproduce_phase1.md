# Reproducing Phase I — Step 1 Frozen Baseline

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
