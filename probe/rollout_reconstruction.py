import contextlib
import io
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import stable_worldmodel.envs  # noqa: F401 - registers swm/PushT-v1
import torch

from probe.evaluate import load_probe_checkpoint, probe_checkpoint_path
from probe.rollout_probe_eval import load_rollout_prediction_artifact
from probe.rollout_visualization import horizon_index_from_step, load_real_frames_for_examples
from probe.rollout_windows import load_raw_pusht_dataset


# Component: physical-quantity reconstruction from predicted rollout latents.
#
# This is not an image decoder. It applies trained probes to a predicted latent
# z^t to decode physical quantities:
# - agent_location: (agent_x, agent_y), in Push-T world coordinates
# - block_location: (block_x, block_y), in Push-T world coordinates
# - block_angle: radians
#
# Then it renders a Push-T frame from those decoded physical quantities using
# the simulator renderer. We disable the target drawing because the HDF5 dataset
# does not store the goal/target state needed to reconstruct the green target
# exactly. The real frame is shown next to the reconstructed physical state.


def circular_angle_error(pred_angle, true_angle):
    """Return absolute circular angle error in radians."""
    diff = torch.remainder(pred_angle - true_angle + torch.pi, 2 * torch.pi) - torch.pi
    return diff.abs()


def decode_physical_quantities(latents, cache_dir, probe_type, device):
    """Decode agent xy, block xy, and block angle from predicted latents.

    The probes were trained on normalized targets, so each checkpoint stores
    `target_mean` and `target_std`. This function converts probe outputs back
    to raw Push-T units:
    - xy values are in the same 0-512 world coordinate system as the dataset
    - angle is in radians
    """
    decoded = {}
    checkpoint_paths = {}
    for target_name in ("agent_location", "block_location", "block_angle"):
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
        decoded[target_name] = pred_raw.cpu()
        checkpoint_paths[target_name] = str(checkpoint_path)

    return decoded, checkpoint_paths


def compute_physical_errors(decoded, true_states):
    """Compute per-example physical quantity errors in interpretable units."""
    true_agent = true_states[:, 0:2].float()
    true_block = true_states[:, 2:4].float()
    true_angle = true_states[:, 4:5].float()

    agent_l2 = torch.linalg.vector_norm(decoded["agent_location"] - true_agent, dim=1)
    block_l2 = torch.linalg.vector_norm(decoded["block_location"] - true_block, dim=1)
    angle_rad = circular_angle_error(decoded["block_angle"], true_angle).reshape(-1)
    angle_deg = torch.rad2deg(angle_rad)

    # Normalized score for ranking: coordinate errors are divided by the 512x512
    # Push-T world size; angle error is divided by pi so a 180-degree miss is 1.
    score = agent_l2 / 512.0 + block_l2 / 512.0 + angle_rad / torch.pi
    return {
        "agent_l2": agent_l2,
        "block_l2": block_l2,
        "angle_rad": angle_rad,
        "angle_deg": angle_deg,
        "score": score,
    }


def select_best_reconstructions(errors, num_examples):
    """Pick the examples with the smallest combined normalized error score."""
    if num_examples > errors["score"].numel():
        raise ValueError(
            f"num_examples={num_examples} exceeds available examples={errors['score'].numel()}"
        )
    return torch.argsort(errors["score"])[:num_examples].long()


def make_reconstructed_state(decoded, selected_indices):
    """Build 7D Push-T state vectors from decoded physical quantities."""
    agent_xy = decoded["agent_location"][selected_indices]
    block_xy = decoded["block_location"][selected_indices]
    block_angle = decoded["block_angle"][selected_indices]
    velocity = torch.zeros((selected_indices.numel(), 2), dtype=agent_xy.dtype)
    return torch.cat([agent_xy, block_xy, block_angle, velocity], dim=1)


def render_reconstructed_states(reconstructed_states, resolution=224):
    """Render predicted physical states with the Push-T simulator.

    The renderer is used only as a drawing tool. It does not run dynamics here;
    each frame is created by setting the decoded state directly with `_set_state`.
    `with_target=False` avoids drawing a wrong green target because the dataset
    does not store target pose.
    """
    with contextlib.redirect_stderr(io.StringIO()):
        env = gym.make("swm/PushT-v1", resolution=resolution, with_target=False)
        env.reset(seed=0)

    frames = []
    try:
        for state in reconstructed_states:
            env.unwrapped._set_state(state.numpy())
            frames.append(torch.from_numpy(env.render()).permute(2, 0, 1).contiguous())
    finally:
        env.close()
    return torch.stack(frames, dim=0).to(torch.uint8)


def chw_uint8_to_hwc_numpy(pixel):
    """Convert a CHW uint8 frame to HWC numpy for Matplotlib."""
    return pixel.permute(1, 2, 0).cpu().numpy()


def build_best_reconstruction_data(
    predicted_path,
    cache_dir,
    raw_step,
    num_examples,
    probe_type,
    device,
    dataset_name="pusht_expert_train",
):
    """Decode and select the best predicted-latent physical reconstructions.

    For a requested horizon, e.g. raw_step=35, this function:
    1. loads predicted latents z^35 and true states s35,
    2. decodes agent/block/angle with trained probes,
    3. computes errors in raw units,
    4. selects the lowest-error examples,
    5. loads real frames and renders reconstructed physical states.
    """
    predicted, metadata = load_rollout_prediction_artifact(predicted_path)
    horizon_idx = horizon_index_from_step(predicted["target_steps"], raw_step)
    latents = predicted["predicted_latents"][:, horizon_idx].float()
    true_states = predicted["target_states"][:, horizon_idx].float()

    decoded, checkpoint_paths = decode_physical_quantities(
        latents,
        cache_dir=cache_dir,
        probe_type=probe_type,
        device=device,
    )
    errors = compute_physical_errors(decoded, true_states)
    selected_indices = select_best_reconstructions(errors, num_examples=num_examples)

    raw_dataset = load_raw_pusht_dataset(dataset_name=dataset_name, cache_dir=cache_dir)
    selected_episode_ids = predicted["episode_idx"][selected_indices].long()
    real_frames = load_real_frames_for_examples(raw_dataset, selected_episode_ids, raw_step)

    reconstructed_states = make_reconstructed_state(decoded, selected_indices)
    reconstructed_frames = render_reconstructed_states(
        reconstructed_states,
        resolution=real_frames.shape[-1],
    )

    return {
        "metadata": metadata,
        "raw_step": int(raw_step),
        "horizon_idx": int(horizon_idx),
        "probe_type": probe_type,
        "checkpoint_paths": checkpoint_paths,
        "selected_indices": selected_indices,
        "selected_episode_ids": selected_episode_ids,
        "true_states": true_states[selected_indices],
        "decoded_states": reconstructed_states,
        "real_frames": real_frames,
        "reconstructed_frames": reconstructed_frames,
        "errors": {key: value[selected_indices] for key, value in errors.items()},
    }


def format_error_summary(data):
    """Create human-readable per-example error rows for command output."""
    rows = []
    for i, episode_id in enumerate(data["selected_episode_ids"].tolist()):
        rows.append(
            {
                "rank": i + 1,
                "episode_id": int(episode_id),
                "agent_l2": data["errors"]["agent_l2"][i].item(),
                "block_l2": data["errors"]["block_l2"][i].item(),
                "angle_rad": data["errors"]["angle_rad"][i].item(),
                "angle_deg": data["errors"]["angle_deg"][i].item(),
                "score": data["errors"]["score"][i].item(),
            }
        )
    return rows


def plot_reconstruction_grid(data, save_path):
    """Plot real frame and reconstructed physical-state frame side by side."""
    real_frames = data["real_frames"]
    reconstructed_frames = data["reconstructed_frames"]
    rows = format_error_summary(data)
    n = real_frames.size(0)

    fig, axes = plt.subplots(n, 2, figsize=(7, 2.2 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for row_idx, row in enumerate(rows):
        axes[row_idx, 0].imshow(chw_uint8_to_hwc_numpy(real_frames[row_idx]))
        axes[row_idx, 0].set_title(
            f"Real ep {row['episode_id']}, step {data['raw_step']}",
            fontsize=8,
        )
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(chw_uint8_to_hwc_numpy(reconstructed_frames[row_idx]))
        axes[row_idx, 1].set_title(
            "Reconstructed from probes\n"
            f"agent {row['agent_l2']:.1f}px, "
            f"block {row['block_l2']:.1f}px, "
            f"angle {row['angle_rad']:.2f}rad/{row['angle_deg']:.1f}deg",
            fontsize=8,
        )
        axes[row_idx, 1].axis("off")

    fig.suptitle(
        f"Best Predicted-Latent Physical Reconstructions at Raw Horizon {data['raw_step']}\n"
        "Right column is simulator rendering from decoded agent/block/angle; target is omitted.",
        fontsize=11,
    )
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path
