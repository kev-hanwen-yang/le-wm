import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))

import matplotlib.pyplot as plt
import torch

from probe.embedding_cache import preprocess_pixels
from probe.rollout_ablation import safe_policy_name
from probe.rollout_probe_eval import load_rollout_prediction_artifact


# Component: geometric diagnostics for rollout latents.
#
# This module contains geometric diagnostics for rollout latents.
#
# 1. MSE to encoded ground truth. For each predicted rollout latent z^t, it
# encodes the actual future frame f_t with the same frozen LeWM encoder to get
# the true latent z_t, then computes:
# - squared L2 error: ||z^t - z_t||^2, averaged across episodes
# - per-dim latent MSE: mean((z^t - z_t)^2), averaged across episodes and dims
# - full latent-dimension MSE matrix: (H, 192), useful for later diagnostics
#
# 2. Min-distance to the empirical encoded manifold. For each predicted latent
# z^t, it computes min_m ||z^t - m||^2 against encoded train/val latents. It
# also computes the same quantity for true future latents z_t as a baseline.
# This separates "wrong future latent" from "off-manifold latent".
#
# 3. Systematic bias direction. For each horizon, it averages latent prediction
# errors across episodes:
#   mu_t = mean_i(z^t_i - z_t_i)
# and compares the bias energy ||mu_t||^2 to total error energy
# mean_i ||z^t_i - z_t_i||^2. This tells whether errors share a consistent
# direction or mostly cancel like random episode-specific noise.
#
# 4. Norm trajectory. For each rollout horizon, it checks whether predicted
# latent magnitudes ||z^t|| stay on the same scale as true future latents and
# encoded train/val observations. This catches shrinkage, growth, or collapse
# that may not be obvious from direction-based diagnostics alone.
#
# 5. Temporal straightness. For each sequence, it computes latent velocities
# v_t = z_{t+1} - z_t, then measures cosine similarity between consecutive
# velocities. This tells whether rollout paths move smoothly through latent
# space or zig-zag from one predicted step to the next.
#
# 6. Teacher-forced velocity. For each horizon h, it predicts z_h from real
# anchors [z_{h-15}, z_{h-10}, z_{h-5}] and real action windows, without feeding
# predictions back. This isolates the one-step predictor map from open-loop
# autoregressive compounding.
#
# Inputs:
# - predicted rollout artifact: contains episode_idx, predicted_latents, target_steps
# - raw Push-T HDF5: provides the actual future pixels f15, f20, ..., f100
#
# Outputs:
# - true future latent cache: true_future_latents aligned with predicted_latents
# - JSON reports and PNG curves under experiment-specific subfolders:
#   `.stable-wm/probes/latent_diagnostics/mse_to_encoded_ground_truth/`
#   `.stable-wm/probes/latent_diagnostics/min_distance_to_manifold/`
#   `.stable-wm/probes/latent_diagnostics/systematic_bias_direction/`
#   `.stable-wm/probes/latent_diagnostics/norm_trajectory/`
#   `.stable-wm/probes/latent_diagnostics/temporal_straightness/`
#   `.stable-wm/probes/latent_diagnostics/teacher_forced_velocity/`


def latent_diagnostics_root(cache_dir):
    """Return the common root for all latent-diagnostic experiment outputs."""
    return Path(cache_dir) / "probes" / "latent_diagnostics"


def latent_experiment_dir(cache_dir, experiment_name):
    """Return the output directory for one latent diagnostic experiment."""
    return latent_diagnostics_root(cache_dir) / experiment_name


def latent_diagnostic_paths(
    cache_dir,
    dataset_name,
    policy_name,
    split_seed,
    img_size,
    num_episodes,
    raw_horizon,
):
    """Return standard paths for true-latent cache and latent-MSE reports."""
    artifact_dir = latent_experiment_dir(cache_dir, "mse_to_encoded_ground_truth")
    base_name = (
        f"{dataset_name}_{safe_policy_name(policy_name)}_"
        f"split_seed{split_seed}_episodes{num_episodes}_"
        f"horizon{raw_horizon}_img{img_size}"
    )
    return {
        "true_latents": artifact_dir / f"{base_name}_true_future_latents.pt",
        "json": artifact_dir / f"{base_name}_latent_mse_report.json",
        "sq_l2_chart": artifact_dir / f"{base_name}_sq_l2_by_horizon.png",
        "per_dim_mse_chart": artifact_dir / f"{base_name}_per_dim_mse_by_horizon.png",
    }


def legacy_latent_diagnostic_paths(
    cache_dir,
    dataset_name,
    policy_name,
    split_seed,
    img_size,
    num_episodes,
    raw_horizon,
):
    """Return the old flat latent-MSE paths for backward-compatible cache lookup."""
    artifact_dir = latent_diagnostics_root(cache_dir)
    base_name = (
        f"{dataset_name}_{safe_policy_name(policy_name)}_"
        f"split_seed{split_seed}_episodes{num_episodes}_"
        f"horizon{raw_horizon}_img{img_size}"
    )
    return {
        "true_latents": artifact_dir / f"{base_name}_true_future_latents.pt",
        "json": artifact_dir / f"{base_name}_latent_mse_report.json",
        "sq_l2_chart": artifact_dir / f"{base_name}_sq_l2_by_horizon.png",
        "per_dim_mse_chart": artifact_dir / f"{base_name}_per_dim_mse_by_horizon.png",
    }


def systematic_bias_diagnostic_paths(
    cache_dir,
    dataset_name,
    policy_name,
    split_seed,
    img_size,
    num_episodes,
    raw_horizon,
):
    """Return output paths for systematic-bias-direction diagnostics."""
    artifact_dir = latent_experiment_dir(cache_dir, "systematic_bias_direction")
    base_name = (
        f"{dataset_name}_{safe_policy_name(policy_name)}_"
        f"split_seed{split_seed}_episodes{num_episodes}_"
        f"horizon{raw_horizon}_img{img_size}"
    )
    return {
        "json": artifact_dir / f"{base_name}_systematic_bias_report.json",
        "bias_norm_chart": artifact_dir / f"{base_name}_bias_norm_by_horizon.png",
        "per_dim_chart": artifact_dir / f"{base_name}_bias_vs_total_per_dim_error.png",
        "bias_fraction_chart": artifact_dir / f"{base_name}_bias_fraction_by_horizon.png",
    }


def norm_trajectory_diagnostic_paths(
    cache_dir,
    dataset_name,
    policy_name,
    split_seed,
    img_size,
    num_episodes,
    raw_horizon,
    reference_splits,
):
    """Return output paths for latent-norm trajectory diagnostics."""
    artifact_dir = latent_experiment_dir(cache_dir, "norm_trajectory")
    split_text = "_".join(reference_splits)
    base_name = (
        f"{dataset_name}_{safe_policy_name(policy_name)}_"
        f"split_seed{split_seed}_episodes{num_episodes}_"
        f"horizon{raw_horizon}_img{img_size}_ref{split_text}"
    )
    return {
        "json": artifact_dir / f"{base_name}_norm_trajectory_report.json",
        "pred_vs_true_chart": artifact_dir / f"{base_name}_pred_vs_true_norm_by_horizon.png",
        "reference_band_chart": artifact_dir / f"{base_name}_pred_norm_reference_band.png",
        "norm_ratio_chart": artifact_dir / f"{base_name}_norm_ratio_by_horizon.png",
        "norm_zscore_chart": artifact_dir / f"{base_name}_norm_zscore_by_horizon.png",
    }


def temporal_straightness_diagnostic_paths(
    cache_dir,
    dataset_name,
    policy_name,
    split_seed,
    img_size,
    num_episodes,
    raw_horizon,
):
    """Return output paths for temporal-straightness diagnostics."""
    artifact_dir = latent_experiment_dir(cache_dir, "temporal_straightness")
    base_name = (
        f"{dataset_name}_{safe_policy_name(policy_name)}_"
        f"split_seed{split_seed}_episodes{num_episodes}_"
        f"horizon{raw_horizon}_img{img_size}"
    )
    return {
        "json": artifact_dir / f"{base_name}_temporal_straightness_report.json",
        "straightness_chart": artifact_dir / f"{base_name}_pred_vs_true_straightness_by_horizon.png",
        "cosine_histogram": artifact_dir / f"{base_name}_straightness_cosine_histogram.png",
        "velocity_norm_chart": artifact_dir / f"{base_name}_velocity_norm_by_horizon.png",
    }


def teacher_forced_velocity_diagnostic_paths(
    cache_dir,
    dataset_name,
    policy_name,
    split_seed,
    img_size,
    num_episodes,
    raw_horizon,
):
    """Return output paths for teacher-forced velocity diagnostics."""
    artifact_dir = latent_experiment_dir(cache_dir, "teacher_forced_velocity")
    base_name = (
        f"{dataset_name}_{safe_policy_name(policy_name)}_"
        f"split_seed{split_seed}_episodes{num_episodes}_"
        f"horizon{raw_horizon}_img{img_size}"
    )
    return {
        "json": artifact_dir / f"teacher_forced_velocity_report_split_seed{split_seed}_episodes{num_episodes}_horizon{raw_horizon}_img{img_size}.json",
        "markdown": artifact_dir / f"teacher_forced_velocity_report_split_seed{split_seed}_episodes{num_episodes}_horizon{raw_horizon}_img{img_size}.md",
        "velocity_chart": artifact_dir / f"{base_name}_tf_velocity_by_horizon.png",
        "ratio_chart": artifact_dir / f"{base_name}_tf_velocity_ratio_by_horizon.png",
        "error_chart": artifact_dir / f"{base_name}_tf_one_step_error_by_horizon.png",
    }


def resolve_true_future_latents_path(
    cache_dir,
    dataset_name,
    policy_name,
    split_seed,
    img_size,
    num_episodes,
    raw_horizon,
):
    """Prefer the new subfolder cache, falling back to the legacy flat cache."""
    new_paths = latent_diagnostic_paths(
        cache_dir=cache_dir,
        dataset_name=dataset_name,
        policy_name=policy_name,
        split_seed=split_seed,
        img_size=img_size,
        num_episodes=num_episodes,
        raw_horizon=raw_horizon,
    )
    if new_paths["true_latents"].exists():
        return new_paths["true_latents"]

    legacy_paths = legacy_latent_diagnostic_paths(
        cache_dir=cache_dir,
        dataset_name=dataset_name,
        policy_name=policy_name,
        split_seed=split_seed,
        img_size=img_size,
        num_episodes=num_episodes,
        raw_horizon=raw_horizon,
    )
    if legacy_paths["true_latents"].exists():
        return legacy_paths["true_latents"]

    return new_paths["true_latents"]


def validate_predicted_latent_artifact(predicted):
    """Check that predicted rollout tensors have aligned episode/horizon axes."""
    required_keys = ("episode_idx", "predicted_latents", "target_steps")
    for key in required_keys:
        if key not in predicted:
            raise KeyError(f"Missing key {key!r} in predicted rollout artifact")

    predicted_latents = predicted["predicted_latents"]
    episode_idx = predicted["episode_idx"]
    target_steps = predicted["target_steps"]
    if predicted_latents.ndim != 3:
        raise ValueError(
            f"predicted_latents must be (B, H, D), got {tuple(predicted_latents.shape)}"
        )
    if episode_idx.numel() != predicted_latents.shape[0]:
        raise ValueError(
            f"episode_idx length {episode_idx.numel()} does not match B="
            f"{predicted_latents.shape[0]}"
        )
    if target_steps.numel() != predicted_latents.shape[1]:
        raise ValueError(
            f"target_steps length {target_steps.numel()} does not match H="
            f"{predicted_latents.shape[1]}"
        )


def load_future_pixels(raw_dataset, episode_ids, target_steps):
    """Load actual future frames for selected episodes and target steps.

    Example for the 1000-episode horizon-100 run:
    - episode_ids: (1000,)
    - target_steps: (18,) = [15, 20, ..., 100]
    - output pixels: (B, 18, 3, 224, 224), uint8
    """
    future_pixels = []
    for episode_id in episode_ids.reshape(-1).tolist():
        raw_episode = raw_dataset.load_episode(int(episode_id))
        if int(target_steps.max().item()) >= raw_episode["pixels"].size(0):
            raise ValueError(
                f"Episode {int(episode_id)} length={raw_episode['pixels'].size(0)} "
                f"cannot provide target step {int(target_steps.max().item())}"
            )
        future_pixels.append(raw_episode["pixels"][target_steps])
    return torch.stack(future_pixels, dim=0)


def encode_future_latents(
    predicted,
    raw_dataset,
    model,
    device,
    image_transform,
    episode_batch_size=16,
    log_every=10,
):
    """Encode actual future frames into LeWM latents aligned with predictions.

    The function does not keep all raw images in memory. It loads and encodes
    episode batches, then concatenates CPU latent tensors. Output shape matches
    the predicted rollout latents:
    - true_future_latents: (B, H, 192)
    """
    validate_predicted_latent_artifact(predicted)
    episode_ids = predicted["episode_idx"].long()
    target_steps = predicted["target_steps"].long()
    num_episodes = episode_ids.numel()

    true_latent_batches = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, num_episodes, episode_batch_size):
            end = min(start + episode_batch_size, num_episodes)
            batch_pixels = load_future_pixels(
                raw_dataset,
                episode_ids[start:end],
                target_steps,
            )
            batch_pixels = preprocess_pixels(batch_pixels, image_transform, device)
            batch_latents = model.encode({"pixels": batch_pixels})["emb"]
            true_latent_batches.append(batch_latents.cpu().float())

            batch_idx = start // episode_batch_size + 1
            if batch_idx % log_every == 0 or end == num_episodes:
                print(f"encoded true future latents: {end}/{num_episodes} episodes")

    true_future_latents = torch.cat(true_latent_batches, dim=0)
    if true_future_latents.shape != predicted["predicted_latents"].shape:
        raise ValueError(
            f"true_future_latents shape {tuple(true_future_latents.shape)} does not "
            f"match predicted_latents shape {tuple(predicted['predicted_latents'].shape)}"
        )
    return true_future_latents


def compute_latent_mse_metrics(predicted_latents, true_future_latents):
    """Compute squared-L2 and per-dimension latent MSE diagnostics.

    Shapes:
    - predicted_latents: (B, H, D)
    - true_future_latents: (B, H, D)
    - sq_l2_per_episode: (B, H)
    - sq_l2_mean_by_horizon: (H,)
    - per_dim_mse_by_horizon: (H,)
    - latent_dim_mse_by_horizon: (H, D)
    """
    if predicted_latents.shape != true_future_latents.shape:
        raise ValueError(
            f"predicted_latents shape {tuple(predicted_latents.shape)} does not match "
            f"true_future_latents shape {tuple(true_future_latents.shape)}"
        )
    diff_sq = (predicted_latents.float() - true_future_latents.float()).square()
    sq_l2_per_episode = diff_sq.sum(dim=-1)
    sq_l2_mean_by_horizon = sq_l2_per_episode.mean(dim=0)
    per_dim_mse_by_horizon = diff_sq.mean(dim=(0, 2))
    latent_dim_mse_by_horizon = diff_sq.mean(dim=0)
    return {
        "sq_l2_per_episode": sq_l2_per_episode,
        "sq_l2_mean_by_horizon": sq_l2_mean_by_horizon,
        "per_dim_mse_by_horizon": per_dim_mse_by_horizon,
        "latent_dim_mse_by_horizon": latent_dim_mse_by_horizon,
    }


def compute_systematic_bias_metrics(predicted_latents, true_future_latents):
    """Compute horizon-wise systematic latent bias metrics.

    Shapes for the standard 1000-episode rollout:
    - predicted_latents: (1000, 18, 192)
    - true_future_latents: (1000, 18, 192)
    - error: (1000, 18, 192)
    - mean_error_by_horizon: (18, 192), this is mu_t
    - bias_norm_by_horizon: (18,)

    `bias_fraction_by_horizon` compares directional bias energy to total error
    energy. A value near 0 means errors mostly cancel across episodes; a value
    near 1 means most error points in the same latent-space direction.
    """
    if predicted_latents.shape != true_future_latents.shape:
        raise ValueError(
            f"predicted_latents shape {tuple(predicted_latents.shape)} does not match "
            f"true_future_latents shape {tuple(true_future_latents.shape)}"
        )
    if predicted_latents.ndim != 3:
        raise ValueError(
            f"Expected predicted_latents to be (B, H, D), got {tuple(predicted_latents.shape)}"
        )

    error = predicted_latents.float() - true_future_latents.float()
    latent_dim = error.shape[-1]
    mean_error_by_horizon = error.mean(dim=0)
    bias_sq_l2_by_horizon = mean_error_by_horizon.square().sum(dim=-1)
    bias_norm_by_horizon = bias_sq_l2_by_horizon.sqrt()
    bias_per_dim_mse_by_horizon = bias_sq_l2_by_horizon / latent_dim

    sq_l2_per_episode = error.square().sum(dim=-1)
    total_sq_l2_mean_by_horizon = sq_l2_per_episode.mean(dim=0)
    total_per_dim_mse_by_horizon = total_sq_l2_mean_by_horizon / latent_dim
    bias_fraction_by_horizon = (
        bias_sq_l2_by_horizon / total_sq_l2_mean_by_horizon.clamp_min(1e-8)
    )

    return {
        "mean_error_by_horizon": mean_error_by_horizon,
        "bias_sq_l2_by_horizon": bias_sq_l2_by_horizon,
        "bias_norm_by_horizon": bias_norm_by_horizon,
        "bias_per_dim_mse_by_horizon": bias_per_dim_mse_by_horizon,
        "total_sq_l2_mean_by_horizon": total_sq_l2_mean_by_horizon,
        "total_per_dim_mse_by_horizon": total_per_dim_mse_by_horizon,
        "bias_fraction_by_horizon": bias_fraction_by_horizon,
    }


def compute_latent_norm_stats(latents):
    """Compute scalar summary statistics for a flat set of latent norms."""
    norms = latents.float().norm(dim=-1).reshape(-1)
    return {
        "count": int(norms.numel()),
        "mean": norms.mean().item(),
        "std": norms.std(unbiased=True).item() if norms.numel() > 1 else 0.0,
        "p05": torch.quantile(norms, 0.05).item(),
        "p50": torch.quantile(norms, 0.50).item(),
        "p95": torch.quantile(norms, 0.95).item(),
        "min": norms.min().item(),
        "max": norms.max().item(),
    }


def compute_reference_norm_stats(reference_split_paths, norm_chunk_size=100000):
    """Compute train/val encoded-latent norm statistics without pairwise distances.

    Each reference split is loaded once. Norms are computed in row chunks so the
    operation avoids creating a large temporary norm tensor for the full split.
    The final concatenated norm vector is small: one float per encoded frame.
    """
    all_norms = []
    split_summaries = []
    for reference_path in reference_split_paths:
        reference_path = Path(reference_path)
        payload = torch.load(reference_path, map_location="cpu")
        if "encoded" not in payload:
            raise KeyError(f"Expected key 'encoded' in {reference_path}")
        encoded = payload["encoded"]
        emb = encoded["emb"].float()

        split_norms = []
        for start in range(0, emb.size(0), norm_chunk_size):
            end = min(start + norm_chunk_size, emb.size(0))
            split_norms.append(emb[start:end].norm(dim=-1).cpu())
        split_norms = torch.cat(split_norms, dim=0)
        all_norms.append(split_norms)
        split_summaries.append(
            {
                "split_name": reference_path.stem,
                "path": str(reference_path),
                **compute_latent_norm_stats(split_norms.unsqueeze(-1)),
            }
        )
        print(f"processed reference norm split {reference_path.name}: {emb.size(0)} embeddings")
        del payload, encoded, emb

    norms = torch.cat(all_norms, dim=0)
    overall = compute_latent_norm_stats(norms.unsqueeze(-1))
    return {
        "overall": overall,
        "splits": split_summaries,
    }


def compute_norm_trajectory_metrics(predicted_latents, true_future_latents, reference_norm_stats):
    """Compute predicted/true norm trajectories and reference-normalized scores.

    Shapes for the standard 1000-episode rollout:
    - predicted_latents: (1000, 18, 192)
    - true_future_latents: (1000, 18, 192)
    - pred_norm_per_episode: (1000, 18)
    - pred_norm_mean_by_horizon: (18,)

    `norm_ratio_by_horizon` uses the episodewise ratio mean:
    mean_i(||z^t_i|| / ||z_t_i||), which is less sensitive to a few large norms
    than dividing horizon means after aggregation.
    """
    if predicted_latents.shape != true_future_latents.shape:
        raise ValueError(
            f"predicted_latents shape {tuple(predicted_latents.shape)} does not match "
            f"true_future_latents shape {tuple(true_future_latents.shape)}"
        )
    if predicted_latents.ndim != 3:
        raise ValueError(
            f"Expected predicted_latents to be (B, H, D), got {tuple(predicted_latents.shape)}"
        )

    pred_norm_per_episode = predicted_latents.float().norm(dim=-1)
    true_norm_per_episode = true_future_latents.float().norm(dim=-1)
    pred_norm_mean_by_horizon = pred_norm_per_episode.mean(dim=0)
    true_norm_mean_by_horizon = true_norm_per_episode.mean(dim=0)
    pred_norm_std_by_horizon = pred_norm_per_episode.std(dim=0, unbiased=True)
    true_norm_std_by_horizon = true_norm_per_episode.std(dim=0, unbiased=True)
    norm_ratio_by_horizon = (pred_norm_per_episode / true_norm_per_episode.clamp_min(1e-8)).mean(dim=0)
    mean_norm_ratio_by_horizon = pred_norm_mean_by_horizon / true_norm_mean_by_horizon.clamp_min(1e-8)

    reference_mean = reference_norm_stats["overall"]["mean"]
    reference_std = max(reference_norm_stats["overall"]["std"], 1e-8)
    pred_norm_zscore_by_horizon = (pred_norm_mean_by_horizon - reference_mean) / reference_std
    true_norm_zscore_by_horizon = (true_norm_mean_by_horizon - reference_mean) / reference_std

    return {
        "pred_norm_per_episode": pred_norm_per_episode,
        "true_norm_per_episode": true_norm_per_episode,
        "pred_norm_mean_by_horizon": pred_norm_mean_by_horizon,
        "true_norm_mean_by_horizon": true_norm_mean_by_horizon,
        "pred_norm_std_by_horizon": pred_norm_std_by_horizon,
        "true_norm_std_by_horizon": true_norm_std_by_horizon,
        "norm_ratio_by_horizon": norm_ratio_by_horizon,
        "mean_norm_ratio_by_horizon": mean_norm_ratio_by_horizon,
        "pred_norm_zscore_by_horizon": pred_norm_zscore_by_horizon,
        "true_norm_zscore_by_horizon": true_norm_zscore_by_horizon,
    }


def compute_velocity_straightness(latents, eps=1e-8):
    """Compute temporal straightness and velocity norms for one latent sequence.

    Input shape:
    - latents: (B, H, D), for example (1000, 18, 192)

    Derived shapes:
    - velocity: (B, H-1, D), where velocity[:, k] = z_{k+1} - z_k
    - cosine: (B, H-2), cosine between consecutive velocity vectors

    The cosine calculation follows the paper formula and uses `eps` only for
    numerical stability when a velocity vector has near-zero magnitude.
    """
    if latents.ndim != 3:
        raise ValueError(f"Expected latents to be (B, H, D), got {tuple(latents.shape)}")
    if latents.shape[1] < 3:
        raise ValueError("Need at least 3 latent timesteps to compute temporal straightness")

    velocity = latents.float()[:, 1:] - latents.float()[:, :-1]
    velocity_norm = velocity.norm(dim=-1)
    prev_velocity = velocity[:, :-1]
    next_velocity = velocity[:, 1:]
    dot = (prev_velocity * next_velocity).sum(dim=-1)
    denom = prev_velocity.norm(dim=-1) * next_velocity.norm(dim=-1)
    cosine = dot / denom.clamp_min(eps)
    return {
        "velocity": velocity,
        "velocity_norm": velocity_norm,
        "cosine": cosine.clamp(min=-1.0, max=1.0),
    }


def compute_temporal_straightness_metrics(predicted_latents, true_future_latents, eps=1e-8):
    """Compute predicted-vs-true temporal straightness metrics.

    The straightness x-axis uses `target_steps[1:-1]` in the script because each
    cosine compares two adjacent velocity vectors around the middle latent:
    cos((z20 - z15), (z25 - z20)) is labeled at raw step 20.
    """
    if predicted_latents.shape != true_future_latents.shape:
        raise ValueError(
            f"predicted_latents shape {tuple(predicted_latents.shape)} does not match "
            f"true_future_latents shape {tuple(true_future_latents.shape)}"
        )

    pred = compute_velocity_straightness(predicted_latents, eps=eps)
    true = compute_velocity_straightness(true_future_latents, eps=eps)
    pred_cosine = pred["cosine"]
    true_cosine = true["cosine"]
    pred_velocity_norm = pred["velocity_norm"]
    true_velocity_norm = true["velocity_norm"]

    return {
        "pred_cosine_per_episode": pred_cosine,
        "true_cosine_per_episode": true_cosine,
        "pred_straightness_mean_by_horizon": pred_cosine.mean(dim=0),
        "true_straightness_mean_by_horizon": true_cosine.mean(dim=0),
        "pred_straightness_std_by_horizon": pred_cosine.std(dim=0, unbiased=True),
        "true_straightness_std_by_horizon": true_cosine.std(dim=0, unbiased=True),
        "pred_global_straightness_mean": pred_cosine.mean(),
        "true_global_straightness_mean": true_cosine.mean(),
        "pred_global_straightness_std": pred_cosine.reshape(-1).std(unbiased=True),
        "true_global_straightness_std": true_cosine.reshape(-1).std(unbiased=True),
        "pred_velocity_norm_per_episode": pred_velocity_norm,
        "true_velocity_norm_per_episode": true_velocity_norm,
        "pred_velocity_norm_mean_by_horizon": pred_velocity_norm.mean(dim=0),
        "true_velocity_norm_mean_by_horizon": true_velocity_norm.mean(dim=0),
        "pred_velocity_norm_std_by_horizon": pred_velocity_norm.std(dim=0, unbiased=True),
        "true_velocity_norm_std_by_horizon": true_velocity_norm.std(dim=0, unbiased=True),
    }


def build_teacher_forced_real_latents(context_emb, true_future_latents):
    """Concatenate context and true future latents into model-step sequence.

    For the standard horizon-100 rollout:
    - context_emb: (B, 3, 192), corresponding to z0, z5, z10
    - true_future_latents: (B, 18, 192), corresponding to z15, ..., z100
    - output: (B, 21, 192), corresponding to z0, z5, ..., z100
    """
    if context_emb.ndim != 3 or true_future_latents.ndim != 3:
        raise ValueError(
            f"Expected both tensors to be (B, T, D), got "
            f"{tuple(context_emb.shape)} and {tuple(true_future_latents.shape)}"
        )
    if context_emb.shape[0] != true_future_latents.shape[0]:
        raise ValueError("context_emb and true_future_latents have different batch sizes")
    if context_emb.shape[-1] != true_future_latents.shape[-1]:
        raise ValueError("context_emb and true_future_latents have different latent dims")
    return torch.cat([context_emb.float(), true_future_latents.float()], dim=1)


def predict_teacher_forced_latents(
    real_latents,
    action_emb,
    model,
    device,
    batch_size=256,
    history_size=3,
):
    """Run one-step predictor on real latent/action windows, without feedback.

    At target index 0, the predictor sees:
    - latents: [z0, z5, z10]
    - actions: [a0-5, a5-10, a10-15]
    and predicts z15.

    At target index 1, it sees real latents [z5, z10, z15], not the previous
    prediction. This is what makes the diagnostic teacher-forced.
    """
    if real_latents.ndim != 3 or action_emb.ndim != 3:
        raise ValueError(
            f"Expected real_latents/action_emb to be (B, T, D), got "
            f"{tuple(real_latents.shape)} and {tuple(action_emb.shape)}"
        )
    if real_latents.shape[0] != action_emb.shape[0]:
        raise ValueError("real_latents and action_emb have different batch sizes")
    num_targets = real_latents.size(1) - history_size
    expected_targets = action_emb.size(1) - history_size + 1
    if num_targets != expected_targets:
        raise ValueError(
            f"real_latents imply {num_targets} targets but action_emb implies "
            f"{expected_targets} targets"
        )

    predicted_batches = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, real_latents.size(0), batch_size):
            end = min(start + batch_size, real_latents.size(0))
            batch_real = real_latents[start:end].to(device)
            batch_action = action_emb[start:end].to(device)
            batch_predictions = []
            for target_idx in range(num_targets):
                emb_window = batch_real[:, target_idx : target_idx + history_size]
                action_window = batch_action[:, target_idx : target_idx + history_size]
                pred_seq = model.predict(emb_window, action_window)
                batch_predictions.append(pred_seq[:, -1:].cpu())
            predicted_batches.append(torch.cat(batch_predictions, dim=1))
    return torch.cat(predicted_batches, dim=0).float()


def compute_teacher_forced_velocity_metrics(
    real_latents,
    teacher_forced_latents,
    open_loop_latents,
):
    """Compute teacher-forced and open-loop velocity/error metrics.

    Shapes for the standard run:
    - real_latents: (1000, 21, 192), z0, z5, ..., z100
    - teacher_forced_latents: (1000, 18, 192), zhat_tf15, ..., zhat_tf100
    - open_loop_latents: (1000, 18, 192), zhat_ol15, ..., zhat_ol100

    Teacher-forced velocity is anchored to the real previous latent:
    ||zhat_tf[h] - z[h-5]||.

    Open-loop velocity uses the open-loop path itself after the first step:
    ||zhat_ol[h] - zhat_ol[h-5]||, with h=15 anchored at real z10.
    """
    if teacher_forced_latents.shape != open_loop_latents.shape:
        raise ValueError("teacher_forced_latents and open_loop_latents must have the same shape")
    if real_latents.ndim != 3 or teacher_forced_latents.ndim != 3:
        raise ValueError("Expected real_latents and predictions to be 3D tensors")
    if real_latents.shape[0] != teacher_forced_latents.shape[0]:
        raise ValueError("real_latents and predictions have different batch sizes")
    if real_latents.shape[-1] != teacher_forced_latents.shape[-1]:
        raise ValueError("real_latents and predictions have different latent dims")
    if real_latents.shape[1] != teacher_forced_latents.shape[1] + 3:
        raise ValueError(
            f"Expected real_latents T to be prediction T + 3, got "
            f"{real_latents.shape[1]} and {teacher_forced_latents.shape[1]}"
        )

    anchor_real = real_latents[:, 2:-1]
    target_real = real_latents[:, 3:]
    v_true = (target_real - anchor_real).norm(dim=-1)
    v_pred_tf = (teacher_forced_latents.float() - anchor_real).norm(dim=-1)
    err_tf = (teacher_forced_latents.float() - target_real).norm(dim=-1)

    open_loop_prev = torch.cat(
        [anchor_real[:, :1], open_loop_latents.float()[:, :-1]],
        dim=1,
    )
    v_pred_openloop = (open_loop_latents.float() - open_loop_prev).norm(dim=-1)
    err_openloop = (open_loop_latents.float() - target_real).norm(dim=-1)
    latent_dim = real_latents.shape[-1]
    std_kwargs = {"unbiased": v_true.size(0) > 1}

    return {
        "v_true_per_episode": v_true,
        "v_pred_tf_per_episode": v_pred_tf,
        "err_tf_per_episode": err_tf,
        "v_pred_openloop_per_episode": v_pred_openloop,
        "err_openloop_per_episode": err_openloop,
        "v_true_mean": v_true.mean(dim=0),
        "v_true_std": v_true.std(dim=0, **std_kwargs),
        "v_pred_tf_mean": v_pred_tf.mean(dim=0),
        "v_pred_tf_std": v_pred_tf.std(dim=0, **std_kwargs),
        "err_tf_mean": err_tf.mean(dim=0),
        "err_tf_std": err_tf.std(dim=0, **std_kwargs),
        "v_pred_openloop_mean": v_pred_openloop.mean(dim=0),
        "v_pred_openloop_std": v_pred_openloop.std(dim=0, **std_kwargs),
        "err_openloop_mean": err_openloop.mean(dim=0),
        "err_openloop_std": err_openloop.std(dim=0, **std_kwargs),
        "ratio_mean": v_pred_tf.mean(dim=0) / v_true.mean(dim=0).clamp_min(1e-8),
        "openloop_ratio_mean": v_pred_openloop.mean(dim=0) / v_true.mean(dim=0).clamp_min(1e-8),
        "tf_per_dim_mse": err_tf.square().mean(dim=0) / latent_dim,
        "openloop_per_dim_mse": err_openloop.square().mean(dim=0) / latent_dim,
    }


def verify_teacher_forced_first_step(
    teacher_forced_latents,
    open_loop_latents,
    metrics,
    atol=1e-5,
    rtol=1e-5,
):
    """Verify h=15 teacher-forced and open-loop predictions are identical.

    At h=15 both paths use the same real context latents and action window, so
    prediction, predicted velocity, and prediction error should match up to the
    numerical tolerance of the device used for inference.
    """
    checks = {
        "prediction": torch.allclose(
            teacher_forced_latents[:, 0],
            open_loop_latents[:, 0],
            atol=atol,
            rtol=rtol,
        ),
        "v_pred": torch.allclose(
            metrics["v_pred_tf_per_episode"][:, 0],
            metrics["v_pred_openloop_per_episode"][:, 0],
            atol=atol,
            rtol=rtol,
        ),
        "err": torch.allclose(
            metrics["err_tf_per_episode"][:, 0],
            metrics["err_openloop_per_episode"][:, 0],
            atol=atol,
            rtol=rtol,
        ),
    }
    max_abs = {
        "prediction": (teacher_forced_latents[:, 0] - open_loop_latents[:, 0]).abs().max().item(),
        "v_pred": (
            metrics["v_pred_tf_per_episode"][:, 0]
            - metrics["v_pred_openloop_per_episode"][:, 0]
        ).abs().max().item(),
        "err": (
            metrics["err_tf_per_episode"][:, 0]
            - metrics["err_openloop_per_episode"][:, 0]
        ).abs().max().item(),
    }
    passed = all(checks.values())
    result = {
        "passed": bool(passed),
        "atol": float(atol),
        "rtol": float(rtol),
        "checks": checks,
        "max_abs_diff": max_abs,
    }
    if not passed:
        raise AssertionError(f"Teacher-forced first-step sanity check failed: {result}")
    return result


def save_true_future_latents(predicted, true_future_latents, metadata, save_path):
    """Save encoded ground-truth future latents for reuse by later diagnostics."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "true": {
            "episode_idx": predicted["episode_idx"].long().cpu(),
            "target_steps": predicted["target_steps"].long().cpu(),
            "true_future_latents": true_future_latents.cpu().float(),
        },
    }
    torch.save(payload, save_path)
    return save_path


def load_true_future_latents(true_latents_path):
    """Load a cached true-future-latent artifact."""
    payload = torch.load(true_latents_path, map_location="cpu")
    if "true" not in payload:
        raise KeyError(f"Expected key 'true' in {true_latents_path}")
    return payload["true"], payload.get("metadata", {})


def save_latent_mse_report(metrics, predicted, metadata, json_path):
    """Save machine-readable latent-MSE metrics as JSON."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    target_steps = predicted["target_steps"].long()
    for horizon_idx, raw_step in enumerate(target_steps.tolist()):
        rows.append(
            {
                "horizon_idx": int(horizon_idx),
                "raw_step": int(raw_step),
                "num_episodes": int(predicted["predicted_latents"].shape[0]),
                "latent_dim": int(predicted["predicted_latents"].shape[-1]),
                "sq_l2_mean": metrics["sq_l2_mean_by_horizon"][horizon_idx].item(),
                "per_dim_mse": metrics["per_dim_mse_by_horizon"][horizon_idx].item(),
            }
        )

    payload = {
        "metadata": metadata,
        "rows": rows,
        "latent_dim_mse_by_horizon": metrics["latent_dim_mse_by_horizon"].tolist(),
    }
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    return json_path


def plot_metric_by_horizon(target_steps, values, ylabel, title, save_path):
    """Plot one latent diagnostic metric as a function of raw rollout horizon."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(target_steps.tolist(), values.tolist(), marker="o")
    plt.xlabel("Raw rollout horizon")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def save_latent_mse_charts(metrics, target_steps, sq_l2_path, per_dim_mse_path):
    """Save squared-L2 and per-dim latent MSE horizon curves."""
    saved_sq_l2 = plot_metric_by_horizon(
        target_steps,
        metrics["sq_l2_mean_by_horizon"],
        ylabel="Mean squared L2 error",
        title="Predicted Latent vs Encoded Ground Truth",
        save_path=sq_l2_path,
    )
    saved_per_dim = plot_metric_by_horizon(
        target_steps,
        metrics["per_dim_mse_by_horizon"],
        ylabel="Per-dim latent MSE",
        title="Per-Dimension Latent MSE vs Rollout Horizon",
        save_path=per_dim_mse_path,
    )
    return {
        "sq_l2_chart": saved_sq_l2,
        "per_dim_mse_chart": saved_per_dim,
    }


def save_systematic_bias_report(metrics, predicted, metadata, json_path):
    """Save systematic-bias-direction metrics as JSON.

    The rows contain scalar summaries for plotting and quick reading. The full
    `mean_error_by_horizon` matrix is also saved because it is the actual bias
    direction vector mu_t in latent space.
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    target_steps = predicted["target_steps"].long()
    rows = []
    for horizon_idx, raw_step in enumerate(target_steps.tolist()):
        rows.append(
            {
                "horizon_idx": int(horizon_idx),
                "raw_step": int(raw_step),
                "num_episodes": int(predicted["predicted_latents"].shape[0]),
                "latent_dim": int(predicted["predicted_latents"].shape[-1]),
                "bias_norm": metrics["bias_norm_by_horizon"][horizon_idx].item(),
                "bias_sq_l2": metrics["bias_sq_l2_by_horizon"][horizon_idx].item(),
                "bias_per_dim_mse": metrics["bias_per_dim_mse_by_horizon"][horizon_idx].item(),
                "total_sq_l2_mean": metrics["total_sq_l2_mean_by_horizon"][horizon_idx].item(),
                "total_per_dim_mse": metrics["total_per_dim_mse_by_horizon"][horizon_idx].item(),
                "bias_fraction": metrics["bias_fraction_by_horizon"][horizon_idx].item(),
            }
        )

    payload = {
        "metadata": metadata,
        "rows": rows,
        "mean_error_by_horizon": metrics["mean_error_by_horizon"].tolist(),
    }
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    return json_path


def plot_two_metrics_by_horizon(
    target_steps,
    first_values,
    second_values,
    first_label,
    second_label,
    ylabel,
    title,
    save_path,
):
    """Plot two horizon curves together for direct scale comparison."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(target_steps.tolist(), first_values.tolist(), marker="o", label=first_label)
    plt.plot(target_steps.tolist(), second_values.tolist(), marker="o", label=second_label)
    plt.xlabel("Raw rollout horizon")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def save_systematic_bias_charts(metrics, target_steps, paths):
    """Save bias norm, per-dim bias-vs-total error, and bias-fraction charts."""
    bias_norm_chart = plot_metric_by_horizon(
        target_steps,
        metrics["bias_norm_by_horizon"],
        ylabel="||mean error vector||",
        title="Systematic Latent Bias Norm vs Rollout Horizon",
        save_path=paths["bias_norm_chart"],
    )
    per_dim_chart = plot_two_metrics_by_horizon(
        target_steps,
        first_values=metrics["bias_per_dim_mse_by_horizon"],
        second_values=metrics["total_per_dim_mse_by_horizon"],
        first_label="Bias per-dim energy",
        second_label="Total per-dim MSE",
        ylabel="Per-dim squared latent error",
        title="Systematic Bias Energy vs Total Error Energy",
        save_path=paths["per_dim_chart"],
    )
    bias_fraction_chart = plot_metric_by_horizon(
        target_steps,
        metrics["bias_fraction_by_horizon"],
        ylabel="||mean error||^2 / mean ||error||^2",
        title="Bias Fraction vs Rollout Horizon",
        save_path=paths["bias_fraction_chart"],
    )
    return {
        "bias_norm_chart": bias_norm_chart,
        "per_dim_chart": per_dim_chart,
        "bias_fraction_chart": bias_fraction_chart,
    }


def save_norm_trajectory_report(metrics, predicted, reference_norm_stats, metadata, json_path):
    """Save latent norm trajectory metrics as JSON."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    target_steps = predicted["target_steps"].long()
    rows = []
    for horizon_idx, raw_step in enumerate(target_steps.tolist()):
        rows.append(
            {
                "horizon_idx": int(horizon_idx),
                "raw_step": int(raw_step),
                "num_episodes": int(predicted["predicted_latents"].shape[0]),
                "latent_dim": int(predicted["predicted_latents"].shape[-1]),
                "pred_norm_mean": metrics["pred_norm_mean_by_horizon"][horizon_idx].item(),
                "pred_norm_std": metrics["pred_norm_std_by_horizon"][horizon_idx].item(),
                "true_norm_mean": metrics["true_norm_mean_by_horizon"][horizon_idx].item(),
                "true_norm_std": metrics["true_norm_std_by_horizon"][horizon_idx].item(),
                "norm_ratio": metrics["norm_ratio_by_horizon"][horizon_idx].item(),
                "mean_norm_ratio": metrics["mean_norm_ratio_by_horizon"][horizon_idx].item(),
                "pred_norm_zscore": metrics["pred_norm_zscore_by_horizon"][horizon_idx].item(),
                "true_norm_zscore": metrics["true_norm_zscore_by_horizon"][horizon_idx].item(),
            }
        )

    payload = {
        "metadata": metadata,
        "reference_norm_stats": reference_norm_stats,
        "rows": rows,
    }
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    return json_path


def plot_norm_with_reference_band(target_steps, pred_norm, true_norm, reference_norm_stats, save_path):
    """Plot predicted/true norms with train-val mean +/- std as the reference band."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    target_steps_list = target_steps.tolist()
    reference_mean = reference_norm_stats["overall"]["mean"]
    reference_std = reference_norm_stats["overall"]["std"]
    lower = reference_mean - reference_std
    upper = reference_mean + reference_std

    plt.figure(figsize=(8, 5))
    plt.fill_between(
        target_steps_list,
        [lower] * len(target_steps_list),
        [upper] * len(target_steps_list),
        alpha=0.2,
        label="Train/val norm mean +/- std",
    )
    plt.axhline(reference_mean, color="gray", linestyle="--", linewidth=1, label="Train/val mean")
    plt.plot(target_steps_list, pred_norm.tolist(), marker="o", label="Predicted latents")
    plt.plot(target_steps_list, true_norm.tolist(), marker="o", label="True future latents")
    plt.xlabel("Raw rollout horizon")
    plt.ylabel("Latent L2 norm")
    plt.title("Predicted Norm vs Train/Val Reference Norm Band")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def save_norm_trajectory_charts(metrics, target_steps, reference_norm_stats, paths):
    """Save norm trajectory, reference-band, norm-ratio, and z-score charts."""
    pred_vs_true_chart = plot_two_metrics_by_horizon(
        target_steps,
        first_values=metrics["pred_norm_mean_by_horizon"],
        second_values=metrics["true_norm_mean_by_horizon"],
        first_label="Predicted latents",
        second_label="True future latents",
        ylabel="Latent L2 norm",
        title="Predicted vs True Future Latent Norm",
        save_path=paths["pred_vs_true_chart"],
    )
    reference_band_chart = plot_norm_with_reference_band(
        target_steps,
        pred_norm=metrics["pred_norm_mean_by_horizon"],
        true_norm=metrics["true_norm_mean_by_horizon"],
        reference_norm_stats=reference_norm_stats,
        save_path=paths["reference_band_chart"],
    )
    norm_ratio_chart = plot_metric_by_horizon(
        target_steps,
        metrics["norm_ratio_by_horizon"],
        ylabel="mean(||pred|| / ||true||)",
        title="Predicted-to-True Latent Norm Ratio",
        save_path=paths["norm_ratio_chart"],
    )
    norm_zscore_chart = plot_two_metrics_by_horizon(
        target_steps,
        first_values=metrics["pred_norm_zscore_by_horizon"],
        second_values=metrics["true_norm_zscore_by_horizon"],
        first_label="Predicted latents",
        second_label="True future latents",
        ylabel="Norm z-score vs train/val",
        title="Latent Norm Z-Score vs Train/Val Reference",
        save_path=paths["norm_zscore_chart"],
    )
    return {
        "pred_vs_true_chart": pred_vs_true_chart,
        "reference_band_chart": reference_band_chart,
        "norm_ratio_chart": norm_ratio_chart,
        "norm_zscore_chart": norm_zscore_chart,
    }


def save_temporal_straightness_report(metrics, predicted, metadata, json_path):
    """Save temporal-straightness metrics as JSON.

    The report contains one row per turn horizon for cosine straightness and one
    row per velocity horizon for velocity norms. It also saves global scalar
    means/stds matching the paper-style average over episodes and timesteps.
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    target_steps = predicted["target_steps"].long()
    turn_steps = target_steps[1:-1]
    velocity_steps = target_steps[1:]

    straightness_rows = []
    for horizon_idx, raw_step in enumerate(turn_steps.tolist()):
        straightness_rows.append(
            {
                "horizon_idx": int(horizon_idx),
                "raw_step": int(raw_step),
                "num_episodes": int(predicted["predicted_latents"].shape[0]),
                "latent_dim": int(predicted["predicted_latents"].shape[-1]),
                "pred_straightness_mean": metrics["pred_straightness_mean_by_horizon"][horizon_idx].item(),
                "pred_straightness_std": metrics["pred_straightness_std_by_horizon"][horizon_idx].item(),
                "true_straightness_mean": metrics["true_straightness_mean_by_horizon"][horizon_idx].item(),
                "true_straightness_std": metrics["true_straightness_std_by_horizon"][horizon_idx].item(),
            }
        )

    velocity_rows = []
    for horizon_idx, raw_step in enumerate(velocity_steps.tolist()):
        velocity_rows.append(
            {
                "horizon_idx": int(horizon_idx),
                "raw_step": int(raw_step),
                "pred_velocity_norm_mean": metrics["pred_velocity_norm_mean_by_horizon"][horizon_idx].item(),
                "pred_velocity_norm_std": metrics["pred_velocity_norm_std_by_horizon"][horizon_idx].item(),
                "true_velocity_norm_mean": metrics["true_velocity_norm_mean_by_horizon"][horizon_idx].item(),
                "true_velocity_norm_std": metrics["true_velocity_norm_std_by_horizon"][horizon_idx].item(),
            }
        )

    payload = {
        "metadata": metadata,
        "global_summary": {
            "pred_global_straightness_mean": metrics["pred_global_straightness_mean"].item(),
            "pred_global_straightness_std": metrics["pred_global_straightness_std"].item(),
            "true_global_straightness_mean": metrics["true_global_straightness_mean"].item(),
            "true_global_straightness_std": metrics["true_global_straightness_std"].item(),
        },
        "straightness_rows": straightness_rows,
        "velocity_rows": velocity_rows,
    }
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    return json_path


def plot_temporal_straightness_histogram(metrics, save_path):
    """Plot distributions of predicted and true velocity-cosine values."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pred_values = metrics["pred_cosine_per_episode"].reshape(-1).tolist()
    true_values = metrics["true_cosine_per_episode"].reshape(-1).tolist()
    plt.figure(figsize=(8, 5))
    plt.hist(true_values, bins=50, alpha=0.55, density=True, label="True future latents")
    plt.hist(pred_values, bins=50, alpha=0.55, density=True, label="Predicted latents")
    plt.xlabel("Cosine between consecutive velocity vectors")
    plt.ylabel("Density")
    plt.title("Temporal Straightness Cosine Distribution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def save_temporal_straightness_charts(metrics, target_steps, paths):
    """Save straightness-by-horizon, cosine histogram, and velocity-norm charts."""
    turn_steps = target_steps[1:-1]
    velocity_steps = target_steps[1:]
    straightness_chart = plot_two_metrics_by_horizon(
        turn_steps,
        first_values=metrics["pred_straightness_mean_by_horizon"],
        second_values=metrics["true_straightness_mean_by_horizon"],
        first_label="Predicted latents",
        second_label="True future latents",
        ylabel="Mean cosine similarity",
        title="Temporal Straightness vs Rollout Horizon",
        save_path=paths["straightness_chart"],
    )
    cosine_histogram = plot_temporal_straightness_histogram(
        metrics,
        save_path=paths["cosine_histogram"],
    )
    velocity_norm_chart = plot_two_metrics_by_horizon(
        velocity_steps,
        first_values=metrics["pred_velocity_norm_mean_by_horizon"],
        second_values=metrics["true_velocity_norm_mean_by_horizon"],
        first_label="Predicted latents",
        second_label="True future latents",
        ylabel="Mean velocity L2 norm",
        title="Latent Velocity Norm vs Rollout Horizon",
        save_path=paths["velocity_norm_chart"],
    )
    return {
        "straightness_chart": straightness_chart,
        "cosine_histogram": cosine_histogram,
        "velocity_norm_chart": velocity_norm_chart,
    }


def teacher_forced_ratio_interpretation(ratio):
    """Return a short heuristic interpretation for ratio_mean at a chosen horizon."""
    if ratio < 0.8:
        return "under-shoot: the one-step map appears intrinsically smoothed or damped."
    if 0.9 <= ratio <= 1.1:
        return "faithful one-step velocity scale: open-loop collapse is more likely exposure bias."
    if 1.2 <= ratio <= 1.4:
        return "near the isotropic-noise context line: no obvious motion-relevant velocity-scale bias."
    if ratio > 1.5:
        return "jump-past-mean / over-shoot: the one-step map may move too far toward a wrong attractor."
    return "intermediate regime: compare the teacher-forced and open-loop ratio curves."


def save_teacher_forced_velocity_report(
    metrics,
    target_steps,
    metadata,
    sanity_check,
    json_path,
):
    """Save teacher-forced velocity diagnostics as JSON arrays plus rows."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    horizons = [int(step) for step in target_steps.tolist()]
    arrays = {
        "horizons": horizons,
        "v_true_mean": metrics["v_true_mean"].tolist(),
        "v_true_std": metrics["v_true_std"].tolist(),
        "v_pred_tf_mean": metrics["v_pred_tf_mean"].tolist(),
        "v_pred_tf_std": metrics["v_pred_tf_std"].tolist(),
        "err_tf_mean": metrics["err_tf_mean"].tolist(),
        "err_tf_std": metrics["err_tf_std"].tolist(),
        "ratio_mean": metrics["ratio_mean"].tolist(),
        "v_pred_openloop_mean": metrics["v_pred_openloop_mean"].tolist(),
        "v_pred_openloop_std": metrics["v_pred_openloop_std"].tolist(),
        "err_openloop_mean": metrics["err_openloop_mean"].tolist(),
        "err_openloop_std": metrics["err_openloop_std"].tolist(),
        "openloop_ratio_mean": metrics["openloop_ratio_mean"].tolist(),
        "tf_per_dim_mse": metrics["tf_per_dim_mse"].tolist(),
        "openloop_per_dim_mse": metrics["openloop_per_dim_mse"].tolist(),
    }
    rows = []
    for idx, raw_step in enumerate(horizons):
        rows.append(
            {
                "horizon_idx": int(idx),
                "raw_step": int(raw_step),
                "v_true_mean": arrays["v_true_mean"][idx],
                "v_true_std": arrays["v_true_std"][idx],
                "v_pred_tf_mean": arrays["v_pred_tf_mean"][idx],
                "v_pred_tf_std": arrays["v_pred_tf_std"][idx],
                "err_tf_mean": arrays["err_tf_mean"][idx],
                "err_tf_std": arrays["err_tf_std"][idx],
                "ratio_mean": arrays["ratio_mean"][idx],
                "v_pred_openloop_mean": arrays["v_pred_openloop_mean"][idx],
                "v_pred_openloop_std": arrays["v_pred_openloop_std"][idx],
                "err_openloop_mean": arrays["err_openloop_mean"][idx],
                "err_openloop_std": arrays["err_openloop_std"][idx],
                "openloop_ratio_mean": arrays["openloop_ratio_mean"][idx],
                "tf_per_dim_mse": arrays["tf_per_dim_mse"][idx],
                "openloop_per_dim_mse": arrays["openloop_per_dim_mse"][idx],
            }
        )

    payload = {
        "metadata": metadata,
        "sanity_check": sanity_check,
        **arrays,
        "rows": rows,
    }
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    return json_path


def plot_teacher_forced_velocity_chart(metrics, target_steps, save_path):
    """Plot true, teacher-forced predicted, and open-loop predicted velocity norms."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    steps = target_steps.tolist()
    plt.figure(figsize=(8, 5))
    plt.plot(steps, metrics["v_true_mean"].tolist(), marker="o", label="v_true")
    plt.plot(steps, metrics["v_pred_tf_mean"].tolist(), marker="o", label="v_pred_tf")
    plt.plot(steps, metrics["v_pred_openloop_mean"].tolist(), marker="o", label="v_pred_openloop")
    plt.xlabel("Raw rollout horizon")
    plt.ylabel("Mean latent velocity norm")
    plt.title("Teacher-Forced vs Open-Loop Latent Velocity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def plot_teacher_forced_ratio_chart(metrics, target_steps, save_path):
    """Plot teacher-forced and open-loop velocity ratios with reference lines."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    steps = target_steps.tolist()
    plt.figure(figsize=(8, 5))
    plt.plot(steps, metrics["ratio_mean"].tolist(), marker="o", label="Teacher-forced ratio")
    plt.plot(steps, metrics["openloop_ratio_mean"].tolist(), marker="o", label="Open-loop ratio")
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Faithful ratio = 1.0")
    plt.axhline(1.3, color="gray", linestyle=":", linewidth=1, label="Isotropic-error context = 1.3")
    plt.xlabel("Raw rollout horizon")
    plt.ylabel("Predicted velocity / true velocity")
    plt.title("Teacher-Forced vs Open-Loop Velocity Ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def plot_teacher_forced_error_chart(metrics, target_steps, save_path):
    """Plot teacher-forced and open-loop per-dim latent prediction MSE."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    steps = target_steps.tolist()
    plt.figure(figsize=(8, 5))
    plt.plot(steps, metrics["tf_per_dim_mse"].tolist(), marker="o", label="Teacher-forced per-dim MSE")
    plt.plot(steps, metrics["openloop_per_dim_mse"].tolist(), marker="o", label="Open-loop per-dim MSE")
    plt.xlabel("Raw rollout horizon")
    plt.ylabel("Per-dim latent MSE")
    plt.title("Teacher-Forced One-Step Error vs Open-Loop Error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def save_teacher_forced_velocity_charts(metrics, target_steps, paths):
    """Save all teacher-forced velocity diagnostic charts."""
    velocity_chart = plot_teacher_forced_velocity_chart(
        metrics,
        target_steps=target_steps,
        save_path=paths["velocity_chart"],
    )
    ratio_chart = plot_teacher_forced_ratio_chart(
        metrics,
        target_steps=target_steps,
        save_path=paths["ratio_chart"],
    )
    error_chart = plot_teacher_forced_error_chart(
        metrics,
        target_steps=target_steps,
        save_path=paths["error_chart"],
    )
    return {
        "velocity_chart": velocity_chart,
        "ratio_chart": ratio_chart,
        "error_chart": error_chart,
    }


def save_teacher_forced_velocity_markdown(report, chart_paths, markdown_path):
    """Save a compact markdown summary keyed off ratio_mean at h=50."""
    markdown_path = Path(markdown_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    horizons = report["horizons"]
    ratios = report["ratio_mean"]
    ratio_by_horizon = {int(h): float(r) for h, r in zip(horizons, ratios)}
    selected_horizons = [15, 30, 50, 100]
    ratio_50 = ratio_by_horizon[50]
    interpretation = teacher_forced_ratio_interpretation(ratio_50)
    lines = [
        "# Teacher-Forced Velocity Diagnostic",
        "",
        "| Horizon | Teacher-forced ratio |",
        "|---:|---:|",
    ]
    for horizon in selected_horizons:
        lines.append(f"| {horizon} | {ratio_by_horizon[horizon]:.4f} |")
    lines.extend(
        [
            "",
            f"At h=50, ratio_mean={ratio_50:.4f}: {interpretation}",
            "",
            "Charts:",
            f"- [Velocity by horizon]({Path(chart_paths['velocity_chart']).name})",
            f"- [Velocity ratio by horizon]({Path(chart_paths['ratio_chart']).name})",
            f"- [One-step error by horizon]({Path(chart_paths['error_chart']).name})",
            "",
            "Caveat: this test isolates the one-step map and does not by itself diagnose the open-loop collapse mechanism; that comes from comparing the teacher-forced and open-loop ratio curves in chart 2.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n")
    return markdown_path


def verify_context_alignment(
    encoded_rollout_path,
    raw_dataset,
    model,
    device,
    image_transform,
    max_episodes=4,
    atol=1e-5,
):
    """Re-encode context frames and compare with saved rollout context embeddings.

    This proves the raw HDF5 episode/frame indexing and image preprocessing used
    for true future latents match the existing rollout artifact. It checks
    frames f0, f5, f10 against saved `context_emb`.
    """
    payload = torch.load(encoded_rollout_path, map_location="cpu")
    if "encoded" not in payload:
        raise KeyError(f"Expected key 'encoded' in {encoded_rollout_path}")
    encoded = payload["encoded"]
    episode_ids = encoded["episode_idx"][:max_episodes].long()
    context_steps = encoded["context_steps"].long()
    reference_context_emb = encoded["context_emb"][:max_episodes].float()

    context_pixels = []
    for episode_id in episode_ids.tolist():
        raw_episode = raw_dataset.load_episode(int(episode_id))
        context_pixels.append(raw_episode["pixels"][context_steps])
    context_pixels = torch.stack(context_pixels, dim=0)

    model.eval()
    with torch.inference_mode():
        pixels = preprocess_pixels(context_pixels, image_transform, device)
        reencoded_context_emb = model.encode({"pixels": pixels})["emb"].cpu().float()

    diff = (reencoded_context_emb - reference_context_emb).abs()
    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()
    allclose = torch.allclose(
        reencoded_context_emb,
        reference_context_emb,
        atol=atol,
        rtol=atol,
    )
    return {
        "checked_episodes": episode_ids.tolist(),
        "context_steps": context_steps.tolist(),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "allclose": bool(allclose),
    }


def manifold_diagnostic_paths(
    cache_dir,
    dataset_name,
    policy_name,
    split_seed,
    img_size,
    num_episodes,
    raw_horizon,
    reference_splits,
    reference_max_latents_per_split=None,
):
    """Return output paths for min-distance-to-manifold diagnostics."""
    artifact_dir = latent_experiment_dir(cache_dir, "min_distance_to_manifold")
    split_text = "_".join(reference_splits)
    reference_text = (
        f"_refmax{reference_max_latents_per_split}"
        if reference_max_latents_per_split is not None
        else ""
    )
    base_name = (
        f"{dataset_name}_{safe_policy_name(policy_name)}_"
        f"split_seed{split_seed}_episodes{num_episodes}_"
        f"horizon{raw_horizon}_img{img_size}_ref{split_text}{reference_text}"
    )
    return {
        "json": artifact_dir / f"{base_name}_manifold_distance_report.json",
        "min_sq_l2_chart": artifact_dir / f"{base_name}_min_sq_l2_to_manifold.png",
        "min_per_dim_chart": artifact_dir / f"{base_name}_min_per_dim_mse_to_manifold.png",
        "ratio_chart": artifact_dir / f"{base_name}_manifold_drift_ratio.png",
    }


def validate_true_future_latent_artifact(predicted, true_artifact):
    """Check true future latents align with the predicted rollout artifact."""
    if "true_future_latents" not in true_artifact:
        raise KeyError("Missing key 'true_future_latents' in true latent artifact")
    if "episode_idx" not in true_artifact or "target_steps" not in true_artifact:
        raise KeyError("True latent artifact must include episode_idx and target_steps")

    if not torch.equal(predicted["episode_idx"].long(), true_artifact["episode_idx"].long()):
        raise ValueError("Predicted and true latent artifacts have different episode_idx")
    if not torch.equal(predicted["target_steps"].long(), true_artifact["target_steps"].long()):
        raise ValueError("Predicted and true latent artifacts have different target_steps")
    if predicted["predicted_latents"].shape != true_artifact["true_future_latents"].shape:
        raise ValueError(
            f"predicted_latents shape {tuple(predicted['predicted_latents'].shape)} "
            f"does not match true_future_latents shape "
            f"{tuple(true_artifact['true_future_latents'].shape)}"
        )


def flatten_rollout_latents(latents):
    """Flatten rollout latents from (B, H, D) into query matrix (B*H, D)."""
    if latents.ndim != 3:
        raise ValueError(f"Expected latents to be (B, H, D), got {tuple(latents.shape)}")
    return latents.reshape(latents.shape[0] * latents.shape[1], latents.shape[2]).float()


def squared_l2_distance_matrix(query, reference):
    """Compute squared L2 distances for one query/reference chunk.

    This avoids `torch.cdist(...).square()`, which computes a square root and
    then squares it again. The output shape is (Q, R).
    """
    query_norm = query.square().sum(dim=1, keepdim=True)
    reference_norm = reference.square().sum(dim=1).unsqueeze(0)
    dist_sq = query_norm + reference_norm - 2.0 * query @ reference.T
    return dist_sq.clamp_min(0.0)


def update_nearest_from_distance_chunk(
    best,
    dist_sq,
    query_start,
    ref_start,
    ref_episode_idx,
    ref_step_idx,
    split_index,
):
    """Update nearest-neighbor records from one distance chunk.

    `best` stores global best distances for all queries. `dist_sq` only covers
    a small block of query rows and reference rows, so this function maps local
    argmin offsets back to global query/reference indices and metadata.
    """
    chunk_best_dist, chunk_best_offset = dist_sq.min(dim=1)
    query_end = query_start + dist_sq.size(0)
    current_best = best["min_sq_l2"][query_start:query_end]
    improved = chunk_best_dist.cpu() < current_best
    if not improved.any():
        return

    improved_query_indices = torch.arange(query_start, query_end)[improved]
    improved_ref_offsets = chunk_best_offset.cpu()[improved]
    best["min_sq_l2"][improved_query_indices] = chunk_best_dist.cpu()[improved]
    best["nearest_split_index"][improved_query_indices] = int(split_index)
    best["nearest_ref_index"][improved_query_indices] = ref_start + improved_ref_offsets
    best["nearest_episode_idx"][improved_query_indices] = ref_episode_idx[improved_ref_offsets].long()
    best["nearest_step_idx"][improved_query_indices] = ref_step_idx[improved_ref_offsets].long()


def initialize_nearest_record(num_queries):
    """Allocate CPU tensors for nearest-neighbor distances and metadata."""
    return {
        "min_sq_l2": torch.full((num_queries,), float("inf"), dtype=torch.float32),
        "nearest_split_index": torch.full((num_queries,), -1, dtype=torch.long),
        "nearest_ref_index": torch.full((num_queries,), -1, dtype=torch.long),
        "nearest_episode_idx": torch.full((num_queries,), -1, dtype=torch.long),
        "nearest_step_idx": torch.full((num_queries,), -1, dtype=torch.long),
    }


def compute_min_distance_to_encoded_splits(
    query_latents,
    reference_split_paths,
    device,
    query_chunk_size=256,
    reference_chunk_size=20000,
    max_reference_latents_per_split=None,
    reference_sample_seed=0,
):
    """Compute exact min distance from queries to encoded train/val latents.

    This is memory-efficient exact search. It never constructs the full
    `(num_queries, num_reference_latents)` distance matrix. Instead, it:
    1. loads one encoded reference split at a time,
    2. iterates through reference chunks, e.g. 20k latents,
    3. iterates through query chunks, e.g. 256 rollout latents,
    4. keeps only the best distance/index seen so far.
    """
    query_latents = query_latents.float().cpu()
    best = initialize_nearest_record(query_latents.size(0))
    split_names = []
    reference_counts = []

    for split_index, reference_path in enumerate(reference_split_paths):
        reference_path = Path(reference_path)
        split_names.append(reference_path.stem)
        payload = torch.load(reference_path, map_location="cpu")
        if "encoded" not in payload:
            raise KeyError(f"Expected key 'encoded' in {reference_path}")
        encoded = payload["encoded"]
        reference_emb = encoded["emb"].float()
        reference_episode_idx = encoded["episode_idx"].long()
        reference_step_idx = encoded["step_idx"].long()
        if (
            max_reference_latents_per_split is not None
            and reference_emb.size(0) > max_reference_latents_per_split
        ):
            generator = torch.Generator().manual_seed(reference_sample_seed + split_index)
            keep = torch.randperm(reference_emb.size(0), generator=generator)[
                :max_reference_latents_per_split
            ].sort().values
            reference_emb = reference_emb[keep]
            reference_episode_idx = reference_episode_idx[keep]
            reference_step_idx = reference_step_idx[keep]
        reference_counts.append(int(reference_emb.size(0)))

        num_ref_chunks = (reference_emb.size(0) + reference_chunk_size - 1) // reference_chunk_size
        for ref_chunk_idx, ref_start in enumerate(range(0, reference_emb.size(0), reference_chunk_size)):
            ref_end = min(ref_start + reference_chunk_size, reference_emb.size(0))
            ref_chunk = reference_emb[ref_start:ref_end].to(device)
            ref_episode_chunk = reference_episode_idx[ref_start:ref_end]
            ref_step_chunk = reference_step_idx[ref_start:ref_end]

            for query_start in range(0, query_latents.size(0), query_chunk_size):
                query_end = min(query_start + query_chunk_size, query_latents.size(0))
                query_chunk = query_latents[query_start:query_end].to(device)
                dist_sq = squared_l2_distance_matrix(query_chunk, ref_chunk)
                update_nearest_from_distance_chunk(
                    best,
                    dist_sq=dist_sq,
                    query_start=query_start,
                    ref_start=ref_start,
                    ref_episode_idx=ref_episode_chunk,
                    ref_step_idx=ref_step_chunk,
                    split_index=split_index,
                )
                del dist_sq
            if (ref_chunk_idx + 1) % 10 == 0 or ref_chunk_idx + 1 == num_ref_chunks:
                print(
                    f"  {reference_path.name}: processed reference chunk "
                    f"{ref_chunk_idx + 1}/{num_ref_chunks}"
                )

        print(
            f"processed reference split {reference_path.name}: "
            f"{reference_emb.size(0)} embeddings"
        )
        del payload, encoded, reference_emb, reference_episode_idx, reference_step_idx

    best["split_names"] = split_names
    best["reference_counts"] = reference_counts
    return best


def reshape_nearest_record(nearest, batch_size, horizon_count):
    """Reshape flat nearest-neighbor records back to rollout shape (B, H)."""
    reshaped = {
        key: value.reshape(batch_size, horizon_count)
        for key, value in nearest.items()
        if torch.is_tensor(value)
    }
    reshaped["split_names"] = nearest["split_names"]
    reshaped["reference_counts"] = nearest["reference_counts"]
    return reshaped


def aggregate_manifold_distance_metrics(pred_nearest, true_nearest, target_steps, latent_dim):
    """Aggregate predicted and true min-distance metrics by horizon."""
    pred_min_sq = pred_nearest["min_sq_l2"].float()
    true_min_sq = true_nearest["min_sq_l2"].float()
    pred_mean = pred_min_sq.mean(dim=0)
    true_mean = true_min_sq.mean(dim=0)
    pred_per_dim = pred_mean / latent_dim
    true_per_dim = true_mean / latent_dim
    ratio = pred_mean / true_mean.clamp_min(1e-8)

    rows = []
    for horizon_idx, raw_step in enumerate(target_steps.tolist()):
        rows.append(
            {
                "horizon_idx": int(horizon_idx),
                "raw_step": int(raw_step),
                "pred_min_sq_l2_mean": pred_mean[horizon_idx].item(),
                "pred_min_per_dim_mse_mean": pred_per_dim[horizon_idx].item(),
                "true_min_sq_l2_mean": true_mean[horizon_idx].item(),
                "true_min_per_dim_mse_mean": true_per_dim[horizon_idx].item(),
                "manifold_drift_ratio": ratio[horizon_idx].item(),
            }
        )

    return {
        "rows": rows,
        "pred_min_sq_l2_mean": pred_mean,
        "true_min_sq_l2_mean": true_mean,
        "pred_min_per_dim_mse_mean": pred_per_dim,
        "true_min_per_dim_mse_mean": true_per_dim,
        "manifold_drift_ratio": ratio,
    }


def save_manifold_distance_report(
    metrics,
    pred_nearest,
    true_nearest,
    metadata,
    json_path,
):
    """Save min-distance-to-manifold diagnostics as JSON."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "rows": metrics["rows"],
        "reference": {
            "split_names": pred_nearest["split_names"],
            "reference_counts": pred_nearest["reference_counts"],
        },
        "nearest": {
            "pred_min_sq_l2": pred_nearest["min_sq_l2"].tolist(),
            "pred_nearest_split_index": pred_nearest["nearest_split_index"].tolist(),
            "pred_nearest_episode_idx": pred_nearest["nearest_episode_idx"].tolist(),
            "pred_nearest_step_idx": pred_nearest["nearest_step_idx"].tolist(),
            "true_min_sq_l2": true_nearest["min_sq_l2"].tolist(),
            "true_nearest_split_index": true_nearest["nearest_split_index"].tolist(),
            "true_nearest_episode_idx": true_nearest["nearest_episode_idx"].tolist(),
            "true_nearest_step_idx": true_nearest["nearest_step_idx"].tolist(),
        },
    }
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    return json_path


def plot_pred_true_metric(target_steps, pred_values, true_values, ylabel, title, save_path):
    """Plot predicted-vs-true nearest-manifold distance curves."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(target_steps.tolist(), pred_values.tolist(), marker="o", label="Predicted latents")
    plt.plot(target_steps.tolist(), true_values.tolist(), marker="o", label="True encoded latents")
    plt.xlabel("Raw rollout horizon")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def plot_ratio_metric(target_steps, ratio_values, save_path):
    """Plot predicted/true manifold-distance ratio by horizon."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(target_steps.tolist(), ratio_values.tolist(), marker="o")
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Raw rollout horizon")
    plt.ylabel("Predicted min-distance / true min-distance")
    plt.title("Manifold Drift Ratio vs Rollout Horizon")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def save_manifold_distance_charts(metrics, target_steps, paths):
    """Save min squared-L2, per-dim MSE, and ratio charts."""
    min_sq_l2_chart = plot_pred_true_metric(
        target_steps,
        pred_values=metrics["pred_min_sq_l2_mean"],
        true_values=metrics["true_min_sq_l2_mean"],
        ylabel="Mean min squared L2",
        title="Min Distance to Encoded Manifold",
        save_path=paths["min_sq_l2_chart"],
    )
    min_per_dim_chart = plot_pred_true_metric(
        target_steps,
        pred_values=metrics["pred_min_per_dim_mse_mean"],
        true_values=metrics["true_min_per_dim_mse_mean"],
        ylabel="Mean min per-dim MSE",
        title="Min Per-Dim Distance to Encoded Manifold",
        save_path=paths["min_per_dim_chart"],
    )
    ratio_chart = plot_ratio_metric(
        target_steps,
        ratio_values=metrics["manifold_drift_ratio"],
        save_path=paths["ratio_chart"],
    )
    return {
        "min_sq_l2_chart": min_sq_l2_chart,
        "min_per_dim_chart": min_per_dim_chart,
        "ratio_chart": ratio_chart,
    }
