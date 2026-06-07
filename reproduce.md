# Reproducing the long-horizon rollout characterization

This file lists the exact commands that regenerate every number and figure in the
blog post. Each step writes report files (JSON/Markdown) and charts; the blog reads
its numbers from those JSON reports, not from the charts, so the reports are the
source of truth.

## Setup

Setup (environment, data, checkpoint) follows the upstream instructions in
`README.md` — same `uv` environment, same `$STABLEWM_HOME`, same Push-T data and
`pusht/lewm` checkpoint. All analysis uses **seed 42** and the test split inherited
by episode index from the probe files, so test episodes are never seen during probe
training.

Two conventions used throughout:

```bash
# Run from the repo root, with the upstream environment activated.
source .venv/bin/activate          # or call .venv/bin/python directly, as below

# Reports and charts are written under $STABLEWM_HOME (defaults to ~/.stable-wm),
# e.g. $STABLEWM_HOME/probes/reports/ and a matching *_charts/ subfolder.
```

**Device flag.** Three of the diagnostics take a `--device` flag. Pick the value
for your hardware once and reuse it:

```bash
export DEVICE=cuda     # NVIDIA GPU
# export DEVICE=mps    # Apple Silicon (Mac)
# export DEVICE=cpu    # no GPU
```

Run the steps in order: Table 1 first, then prepare-and-rollout (this caches the
predicted/true latents the diagnostics consume), then the geometry diagnostics in
any order.

---

## 1. Table 1 reproduction

Decodes the physical quantities (agent location, block location, block angle) from
real encoder latents with frozen linear and MLP probes, and reports MSE and Pearson
correlation — the horizon-zero sanity check that the probes work on real latents.

```bash
.venv/bin/python scripts/probe_report.py
```

**Output:** `probe_test_report_seed42_img224.md` and `.json`
(the Table 1 reproduction numbers).

---

## 2. Prepare test episodes and run the open-loop rollout

Picks 1000 test episodes from the Push-T dataset by `episode_idx` (requiring raw
horizon ≥ 101). For each episode it takes start timestep 0, builds the 3-frame
context (frames f0, f5, f10 at frame-skip 5), the action sequences
(a0–5, a5–10, a10–15, …, a95–100), and the ground-truth states (s15, s20, …, s100)
as probe targets. It then encodes the context, rolls the predictor forward
open-loop, and applies the frozen probes — exactly mirroring LeWM's own framing.

```bash
.venv/bin/python scripts/probe_rollout_ablation.py \
  --num-episodes 1000 \
  --raw-horizon 100
```

Then aggregate into the report tables and charts:

```bash
.venv/bin/python scripts/probe_rollout_report.py \
  --num-episodes 1000 \
  --raw-horizon 100
```

**Outputs:**

- `rollout_probe_report_split_seed42_episodes1000_horizon100_img224.json` and `.md`
- charts: `norm_mse_by_horizon.png`, `pearson_r_by_horizon.png`, `raw_rmse_by_horizon.png`

These produce the three-tier degradation curves and the Pearson-r values.

---

## 3. Latent-geometry diagnostics

These run on the rollout latents cached in Step 2. (Set `$DEVICE` as above for the
three commands that take `--device`.)

### MSE to the encoded ground-truth latent

How far the predicted latent is from the _correct_ future latent (per-dimension and
squared-L2). This diagnostic is light; `cpu` is fine.

```bash
.venv/bin/python scripts/probe_latent_mse_diagnostics.py \
  --num-episodes 1000 \
  --raw-horizon 100 \
  --episode-batch-size 16 \
  --device "$DEVICE"
```

**Outputs:** `..._latent_mse_report.json`; charts `..._per_dim_mse_by_horizon.png`,
`..._sq_l2_by_horizon.png`. (Blog: per-dim MSE 0.062 → 1.92; `sq_l2_mean` ≈ 369 at horizon 100.)

### Minimum distance to the empirical manifold

Distance from each predicted latent to its nearest real encoded latent, vs. the same
distance for the true future latent. This is the **heaviest** diagnostic; a CUDA GPU
is recommended. On `mps`/`cpu`, lower `--reference-chunk-size` (e.g. to 10000) if you
hit memory limits.

```bash
.venv/bin/python scripts/probe_manifold_distance_diagnostics.py \
  --num-episodes 1000 \
  --raw-horizon 100 \
  --device "$DEVICE" \
  --query-chunk-size 512 \
  --reference-chunk-size 50000
```

**Outputs:** charts `..._reftrain_val_manifold_drift_ratio.png`,
`..._reftrain_val_min_sq_l2_to_manifold.png`,
`..._reftrain_val_min_per_dim_mse_to_manifold.png`, and `.json` file.
(Blog: drift ratio 4.5–11×; predicted nearest-manifold distance ≈ 18 at horizon 100;
≈ 20× closer to the manifold than to the correct target.)

### Systematic bias direction

Splits total error into the shared mean error vector (systematic) and the per-episode
residual. (No `--device` flag in the command as given.)

```bash
.venv/bin/python scripts/probe_systematic_bias_diagnostics.py \
  --num-episodes 1000 \
  --raw-horizon 100
```

**Outputs:** `..._systematic_bias_report.json`; charts `..._bias_norm_by_horizon.png`,
`..._bias_fraction_by_horizon.png`, `..._bias_vs_total_per_dim_error.png`.
(Blog: bias fraction 2–5%; bias norm peaks ≈ 3.24 near horizon 75.)

### Norm trajectory

Tracks the predicted latent's L2 norm against the encoder reference shell.

```bash
.venv/bin/python scripts/probe_norm_trajectory_diagnostics.py \
  --num-episodes 1000 \
  --raw-horizon 100
```

**Outputs:** `..._reftrain_val_norm_trajectory_report.json`; charts
`..._reftrain_val_pred_norm_reference_band.png`, `..._reftrain_val_norm_ratio_by_horizon.png`,
`..._reftrain_val_norm_zscore_by_horizon.png`, `..._reftrain_val_pred_vs_true_norm_by_horizon.png`.
(Blog: reference mean 13.89, std 0.82, n ≈ 2,105,102; predicted z-score in [−0.33, +0.49].)

### Temporal straightness and velocity norm

Step-to-step velocity norm, plus mean cosine between consecutive velocity vectors.

```bash
.venv/bin/python scripts/probe_temporal_straightness_diagnostics.py \
  --num-episodes 1000 \
  --raw-horizon 100
```

**Outputs:** `..._temporal_straightness_report.json` (`velocity_rows`, `straightness_rows`);
charts `..._velocity_norm_by_horizon.png`, `..._pred_vs_true_straightness_by_horizon.png`,
`..._straightness_cosine_histogram.png`.
(Blog: predicted velocity ≈ 1.0 vs true 4–6; predicted straightness ≈ 0.70 vs true ≈ 0.53.)

### One-step teacher-forced velocity (the control)

Feeds real encoded context at every horizon and predicts one step, isolating the
one-step map from compounding feedback.

```bash
.venv/bin/python scripts/probe_teacher_forced_velocity_diagnostics.py \
  --num-episodes 1000 \
  --raw-horizon 100 \
  --device "$DEVICE" \
  --batch-size 256
```

**Outputs:** `teacher_forced_velocity_report_split_seed42_episodes1000_horizon100_img224.json`
and `.md`; charts `..._tf_velocity_ratio_by_horizon.png`, `..._tf_one_step_error_by_horizon.png`,
`..._tf_velocity_by_horizon.png`.
(Blog: teacher-forced velocity ratio ≈ 0.33 flat; open-loop decays to ≈ 0.16;
per-dim TF error 0.06 → 0.14; along/perp split from horizon-50 magnitudes.)

---

## Outputs → blog claims

| Blog section                        | Script                                                  | Primary report (source of truth)                                        |
| ----------------------------------- | ------------------------------------------------------- | ----------------------------------------------------------------------- |
| Table 1 reproduction                | `probe_report.py`                                       | `probe_test_report_split_seed42_probe_seeds0_1_2_3_4_img224.md`         |
| Three-tier degradation              | `probe_rollout_ablation.py` → `probe_rollout_report.py` | `rollout_probe_report_split_seed42_episodes1000_horizon100_img224.json` |
| Off-target (MSE to true latent)     | `probe_latent_mse_diagnostics.py`                       | `..._latent_mse_report.json`                                            |
| Manifold drift                      | `probe_manifold_distance_diagnostics.py`                | `<MANIFOLD_JSON_PLACEHOLDER>`                                           |
| Systematic vs per-episode error     | `probe_systematic_bias_diagnostics.py`                  | `..._systematic_bias_report.json`                                       |
| Stable norm shell                   | `probe_norm_trajectory_diagnostics.py`                  | `..._reftrain_val_norm_trajectory_report.json`                          |
| Slow + over-straight                | `probe_temporal_straightness_diagnostics.py`            | `..._temporal_straightness_report.json`                                 |
| Teacher-forced control + along/perp | `probe_teacher_forced_velocity_diagnostics.py`          | `teacher_forced_velocity_report_*.json`                                 |
