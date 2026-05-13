import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))

import matplotlib.pyplot as plt
import torch

from probe.evaluate import load_encoded_split, load_probe_checkpoint, probe_checkpoint_path
from probe.rollout_probe_eval import load_rollout_prediction_artifact
from probe.rollout_windows import load_raw_pusht_dataset


# Component: qualitative visualization for predicted rollout latents.
#
# LeWM is a JEPA-style latent predictor and this checkpoint does not include a
# pixel decoder. Therefore this module cannot turn a predicted latent directly
# into an image. Instead, it uses a nearest-neighbor proxy: for each predicted
# latent z^t, find the real test-set frame whose encoded latent is closest in
# L2 distance, then display that real frame as the "nearest-neighbor latent
# proxy." This makes the proxy explicit and avoids claiming a true decode.
#
# The same selected examples are also evaluated with trained location probes:
# - agent_location probe predicts (agent_x, agent_y)
# - block_location probe predicts (block_x, block_y)
# These predictions are plotted against the ground-truth physical state at the
# same raw horizon.


def horizon_index_from_step(target_steps, raw_step):
    """Return the index in target_steps matching a requested raw horizon."""
    matches = (target_steps.long() == int(raw_step)).nonzero(as_tuple=False).reshape(-1)
    if matches.numel() != 1:
        raise ValueError(
            f"Expected exactly one target step {raw_step}, found {matches.numel()} "
            f"in {target_steps.tolist()}"
        )
    return int(matches.item())


def sample_example_indices(num_examples, num_samples, seed):
    """Select random rollout examples reproducibly from the saved artifact."""
    if num_samples > num_examples:
        raise ValueError(f"num_samples={num_samples} exceeds num_examples={num_examples}")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randperm(num_examples, generator=generator)[:num_samples].long()


def chw_uint8_to_hwc_numpy(pixel):
    """Convert a Push-T frame from torch CHW uint8 to numpy HWC for Matplotlib."""
    if pixel.ndim != 3:
        raise ValueError(f"Expected pixel shape (C, H, W), got {tuple(pixel.shape)}")
    return pixel.permute(1, 2, 0).cpu().numpy()


def load_real_frames_for_examples(raw_dataset, episode_ids, raw_step):
    """Load real Push-T frames at `raw_step` for selected episode IDs."""
    frames = []
    for episode_id in episode_ids.tolist():
        raw_episode = raw_dataset.load_episode(int(episode_id))
        if raw_step >= raw_episode["pixels"].size(0):
            raise ValueError(
                f"Episode {int(episode_id)} has length {raw_episode['pixels'].size(0)}, "
                f"cannot load raw_step={raw_step}"
            )
        frames.append(raw_episode["pixels"][raw_step])
    return torch.stack(frames, dim=0)


def find_nearest_encoded_frames(query_latents, encoded_test, chunk_size=65536):
    """Find nearest real encoded test frame for each predicted latent.

    Args:
        query_latents: predicted latents, shape (N, 192).
        encoded_test: Table 1 encoded test cache with `emb`, `episode_idx`,
            and `step_idx`.

    Returns:
        A dict with nearest encoded-cache row index, episode ID, step ID, and
        squared L2 distance for each query.
    """
    reference_emb = encoded_test["emb"].float()
    query_latents = query_latents.float()
    best_dist = torch.full((query_latents.size(0),), float("inf"))
    best_index = torch.full((query_latents.size(0),), -1, dtype=torch.long)

    for start in range(0, reference_emb.size(0), chunk_size):
        end = min(start + chunk_size, reference_emb.size(0))
        chunk = reference_emb[start:end]
        distances = torch.cdist(query_latents, chunk).square()
        chunk_best_dist, chunk_best_offset = distances.min(dim=1)
        improved = chunk_best_dist < best_dist
        best_dist[improved] = chunk_best_dist[improved]
        best_index[improved] = chunk_best_offset[improved] + start

    return {
        "nearest_index": best_index,
        "nearest_episode_idx": encoded_test["episode_idx"][best_index].long(),
        "nearest_step_idx": encoded_test["step_idx"][best_index].long(),
        "nearest_l2_sq": best_dist,
    }


def load_nearest_neighbor_frames(raw_dataset, nearest_episode_ids, nearest_step_ids):
    """Load real frames corresponding to nearest-neighbor latent matches."""
    frames = []
    for episode_id, step_id in zip(
        nearest_episode_ids.tolist(),
        nearest_step_ids.tolist(),
        strict=True,
    ):
        raw_episode = raw_dataset.load_episode(int(episode_id))
        frames.append(raw_episode["pixels"][int(step_id)])
    return torch.stack(frames, dim=0)


def decode_probe_raw_values(latents, cache_dir, target_name, probe_type, device):
    """Apply one trained probe and convert normalized outputs to raw units."""
    checkpoint_path = probe_checkpoint_path(
        cache_dir,
        target_name=target_name,
        probe_type=probe_type,
    )
    loaded = load_probe_checkpoint(checkpoint_path, device=device)
    probe = loaded["probe"]
    target_mean = loaded["target_mean"].to(device)
    target_std = loaded["target_std"].to(device)

    probe.eval()
    with torch.inference_mode():
        pred_norm = probe(latents.to(device))
        pred_raw = pred_norm * target_std + target_mean
    return pred_raw.cpu(), str(checkpoint_path)


def build_horizon_visualization_data(
    predicted_path,
    encoded_test_path,
    cache_dir,
    raw_step,
    num_samples,
    seed,
    probe_type,
    device,
    dataset_name="pusht_expert_train",
):
    """Assemble frames, nearest-neighbor proxy frames, and probe predictions.

    This is the main data-building function for the qualitative figure. For a
    requested horizon, e.g. raw_step=35, it extracts:
    - selected predicted latents z^35: (20, 192)
    - real frames f35 from the same episodes
    - nearest-neighbor proxy frames from the encoded test cache
    - probe-predicted and true agent/block locations
    """
    predicted, metadata = load_rollout_prediction_artifact(predicted_path)
    horizon_idx = horizon_index_from_step(predicted["target_steps"], raw_step)
    selected_indices = sample_example_indices(
        num_examples=predicted["predicted_latents"].size(0),
        num_samples=num_samples,
        seed=seed,
    )
    selected_episode_ids = predicted["episode_idx"][selected_indices].long()
    selected_latents = predicted["predicted_latents"][selected_indices, horizon_idx].float()
    selected_states = predicted["target_states"][selected_indices, horizon_idx].float()

    raw_dataset = load_raw_pusht_dataset(dataset_name=dataset_name, cache_dir=cache_dir)
    real_frames = load_real_frames_for_examples(raw_dataset, selected_episode_ids, raw_step)

    encoded_test = load_encoded_split(encoded_test_path)
    nearest = find_nearest_encoded_frames(selected_latents, encoded_test)
    nearest_frames = load_nearest_neighbor_frames(
        raw_dataset,
        nearest["nearest_episode_idx"],
        nearest["nearest_step_idx"],
    )

    pred_agent, agent_probe_path = decode_probe_raw_values(
        selected_latents,
        cache_dir=cache_dir,
        target_name="agent_location",
        probe_type=probe_type,
        device=device,
    )
    pred_block, block_probe_path = decode_probe_raw_values(
        selected_latents,
        cache_dir=cache_dir,
        target_name="block_location",
        probe_type=probe_type,
        device=device,
    )

    return {
        "metadata": metadata,
        "raw_step": int(raw_step),
        "horizon_idx": int(horizon_idx),
        "selected_indices": selected_indices,
        "selected_episode_ids": selected_episode_ids,
        "real_frames": real_frames,
        "nearest_frames": nearest_frames,
        "nearest": nearest,
        "true_agent": selected_states[:, 0:2],
        "pred_agent": pred_agent,
        "true_block": selected_states[:, 2:4],
        "pred_block": pred_block,
        "agent_probe_path": agent_probe_path,
        "block_probe_path": block_probe_path,
    }


def plot_real_vs_nearest_proxy_grid(visualization_data, save_path):
    """Plot real frames beside nearest-neighbor latent proxy frames."""
    real_frames = visualization_data["real_frames"]
    nearest_frames = visualization_data["nearest_frames"]
    episode_ids = visualization_data["selected_episode_ids"]
    nearest_episode_ids = visualization_data["nearest"]["nearest_episode_idx"]
    nearest_step_ids = visualization_data["nearest"]["nearest_step_idx"]
    nearest_l2_sq = visualization_data["nearest"]["nearest_l2_sq"]
    raw_step = visualization_data["raw_step"]

    n = real_frames.size(0)
    fig, axes = plt.subplots(n, 2, figsize=(6, 2.1 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for row in range(n):
        axes[row, 0].imshow(chw_uint8_to_hwc_numpy(real_frames[row]))
        axes[row, 0].set_title(f"Real ep {int(episode_ids[row])}, step {raw_step}", fontsize=8)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(chw_uint8_to_hwc_numpy(nearest_frames[row]))
        axes[row, 1].set_title(
            "NN latent proxy "
            f"ep {int(nearest_episode_ids[row])}, step {int(nearest_step_ids[row])}\n"
            f"L2^2={nearest_l2_sq[row].item():.3f}",
            fontsize=8,
        )
        axes[row, 1].axis("off")

    fig.suptitle(
        "Real Frame vs Nearest-Neighbor Predicted-Latent Proxy\n"
        "Note: LeWM has no pixel decoder; right column is retrieval from encoded test frames.",
        fontsize=11,
    )
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def plot_probe_location_scatters(visualization_data, save_path):
    """Plot predicted-vs-true XY locations for agent and block probes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pairs = [
        ("Agent Location", visualization_data["true_agent"], visualization_data["pred_agent"]),
        ("Block Location", visualization_data["true_block"], visualization_data["pred_block"]),
    ]

    for ax, (title, true_xy, pred_xy) in zip(axes, pairs, strict=True):
        ax.scatter(true_xy[:, 0], true_xy[:, 1], c="tab:blue", label="True", alpha=0.85)
        ax.scatter(pred_xy[:, 0], pred_xy[:, 1], c="tab:red", marker="x", label="Predicted", alpha=0.9)
        for i in range(true_xy.size(0)):
            ax.plot(
                [true_xy[i, 0], pred_xy[i, 0]],
                [true_xy[i, 1], pred_xy[i, 1]],
                c="gray",
                alpha=0.35,
                linewidth=0.8,
            )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(0, 512)
        ax.set_ylim(512, 0)
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.suptitle(
        f"Probe-Predicted vs True Locations at Raw Horizon {visualization_data['raw_step']}"
    )
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path
