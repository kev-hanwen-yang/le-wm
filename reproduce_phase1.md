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
  --run-id step1-baseline-seed42
```

For a fast contract/cache check that skips the official CEM smoke evaluation:

```bash
.venv/bin/python scripts/run_phase1.py \
  --stage baseline \
  --run-id step1-contracts-seed42 \
  --skip-cem
```

The baseline creates:

- `diagnostic_results/phase1/<run_id>/baseline_manifest.json`: configuration,
  source revision, checkpoint hash, versions, action statistics, model shapes,
  cache re-encoding comparisons, and CEM result metadata.
- `diagnostic_results/phase1/<run_id>/baseline_summary.md`: concise human-readable
  gate result.
- `.stable-wm/phase1/<run_id>/cem_smoke.txt`: output from the existing official
  evaluator, plus its Hydra run directory under the same artifact root.

The command intentionally refuses to overwrite an existing run ID. It samples
three deterministic rows from each train/validation/test latent cache and compares
fresh frozen-encoder outputs using the configured `atol=1e-5`, `rtol=1e-4`.
If any cache fails, Phase I stops; it will not silently regenerate a multi-gigabyte
cache or mix an unverified cache with counterfactual results.

The CEM smoke uses the repository's existing `eval.py`, policy cost, goal
handling, seed 42, and ten evaluation episodes. It is a baseline functionality
check only: its dataset restoration path is not used as physical ground truth for
the later quiescent counterfactual protocol. The harness overrides only the
stale machine-specific cache directory in the evaluation launcher with the
project-relative `.stable-wm` path.
