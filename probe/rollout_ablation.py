from pathlib import Path

import stable_worldmodel as swm
import torch

from probe.embedding_cache import preprocess_pixels


# Component: long-horizon open-loop rollout ablation.
#
# This module owns the model-side part of the horizon-degradation ablation. It
# uses the same released LeWM checkpoint as Table 1 probing, but keeps the full
# JEPA model because this ablation needs both observed-frame encoding and latent
# dynamics prediction:
# - encoder/projector: pixels -> latent embeddings z_t
# - action_encoder: raw action block (10,) -> action embedding (192,)
# - predictor/pred_proj: latent/action history -> next latent embedding
#
# Data contract used here after rollout-window construction:
# - context_pixels: (B, 3, 3, 224, 224), uint8 for f0, f5, f10
# - action_tokens: (B, 20, 10), float32 for a0-5, ..., a95-100
# - target_states: (B, 18, 7), float32 for s15, s20, ..., s100
#
# The model is frozen and put in eval mode because this ablation measures the
# released model's open-loop behavior; it should not update any weights.


REQUIRED_WORLD_MODEL_COMPONENTS = (
    "encoder",
    "projector",
    "action_encoder",
    "predictor",
    "pred_proj",
)


def eval_device() -> str:
    """Return the best available inference device for this machine."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_cache_dir() -> Path:
    """Prefer the project-local `.stable-wm` cache when it exists."""
    project_cache_dir = Path.cwd() / ".stable-wm"
    if project_cache_dir.exists():
        return project_cache_dir
    return Path(swm.data.utils.get_cache_dir())


def validate_world_model_components(model):
    """Fail early if the loaded checkpoint is not a full JEPA world model."""
    missing_components = [
        name for name in REQUIRED_WORLD_MODEL_COMPONENTS if not hasattr(model, name)
    ]
    if missing_components:
        raise AttributeError(
            "Loaded model is missing required JEPA components: "
            + ", ".join(missing_components)
        )


def load_released_lewm_world_model(
    policy_name: str = "pusht/lewm",
    device: str | None = None,
    cache_dir: str | Path | None = None,
):
    """Load the released LeWM checkpoint as a frozen inference-only model."""
    device = device or eval_device()
    cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()

    model = swm.policy.AutoCostModel(policy_name, cache_dir=cache_dir)
    validate_world_model_components(model)

    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    return model


def safe_policy_name(policy_name):
    """Make policy names such as `pusht/lewm` safe for filenames."""
    return policy_name.replace("/", "_")


def rollout_artifact_paths(
    cache_dir,
    dataset_name,
    policy_name,
    split_seed,
    img_size,
    num_episodes,
    raw_horizon,
):
    """Return standard save paths for encoded rollout data and predictions."""
    artifact_dir = Path(cache_dir) / "probes" / "rollout_ablation"
    base_name = (
        f"{dataset_name}_{safe_policy_name(policy_name)}_"
        f"split_seed{split_seed}_episodes{num_episodes}_"
        f"horizon{raw_horizon}_img{img_size}"
    )
    return {
        "encoded": artifact_dir / f"{base_name}_encoded.pt",
        "predicted": artifact_dir / f"{base_name}_predicted_latents.pt",
    }


def validate_rollout_windows(windows, history_size, frameskip, raw_horizon):
    """Check rollout-window tensor shapes before expensive model inference."""
    context_pixels = windows["context_pixels"]
    action_tokens = windows["action_tokens"]
    target_states = windows["target_states"]

    expected_action_steps = raw_horizon // frameskip
    expected_target_steps = expected_action_steps - history_size + 1

    if context_pixels.ndim != 5:
        raise ValueError(f"context_pixels must be 5D, got {tuple(context_pixels.shape)}")
    if context_pixels.shape[1] != history_size:
        raise ValueError(
            f"context_pixels T must be history_size={history_size}, got "
            f"{context_pixels.shape[1]}"
        )
    if action_tokens.shape[1:] != (expected_action_steps, frameskip * 2):
        raise ValueError(
            "action_tokens must have shape "
            f"(B, {expected_action_steps}, {frameskip * 2}), got "
            f"{tuple(action_tokens.shape)}"
        )
    if target_states.shape[1:] != (expected_target_steps, 7):
        raise ValueError(
            "target_states must have shape "
            f"(B, {expected_target_steps}, 7), got {tuple(target_states.shape)}"
        )


def encode_rollout_windows(
    windows,
    model,
    device,
    image_transform,
    batch_size=16,
    history_size=3,
    frameskip=5,
    raw_horizon=100,
):
    """Encode context frames and raw action tokens for rollout.

    Input example for Push-T with 100 episodes:
    - context_pixels: (100, 3, 3, 224, 224), uint8
    - action_tokens: (100, 20, 10), float32

    Output:
    - context_emb: (100, 3, 192), z0/z5/z10 for each episode
    - action_emb: (100, 20, 192), one encoded action block per model step

    We save action embeddings once because the open-loop rollout repeatedly uses
    sliding action windows, and the action encoder is frozen for this ablation.
    """
    validate_rollout_windows(
        windows,
        history_size=history_size,
        frameskip=frameskip,
        raw_horizon=raw_horizon,
    )

    context_pixels = windows["context_pixels"]
    action_tokens = windows["action_tokens"].float()
    num_episodes = context_pixels.size(0)

    context_embs = []
    action_embs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, num_episodes, batch_size):
            end = min(start + batch_size, num_episodes)
            batch_pixels = preprocess_pixels(
                context_pixels[start:end],
                image_transform,
                device,
            )
            batch_actions = action_tokens[start:end].to(device)

            context_emb = model.encode({"pixels": batch_pixels})["emb"]
            action_emb = model.action_encoder(batch_actions)

            context_embs.append(context_emb.cpu())
            action_embs.append(action_emb.cpu())

    encoded = {
        "episode_idx": windows["episode_idx"].long().cpu(),
        "context_emb": torch.cat(context_embs, dim=0).float(),
        "action_emb": torch.cat(action_embs, dim=0).float(),
        "action_tokens": action_tokens.cpu(),
        "target_states": windows["target_states"].float().cpu(),
        "context_steps": windows["context_steps"].long().cpu(),
        "target_steps": windows["target_steps"].long().cpu(),
        "action_starts": windows["action_starts"].long().cpu(),
        "action_ends": windows["action_ends"].long().cpu(),
    }
    validate_encoded_rollout_dataset(
        encoded,
        history_size=history_size,
        frameskip=frameskip,
        raw_horizon=raw_horizon,
    )
    return encoded


def validate_encoded_rollout_dataset(encoded, history_size, frameskip, raw_horizon):
    """Check encoded rollout tensors before predictor rollout."""
    context_emb = encoded["context_emb"]
    action_emb = encoded["action_emb"]
    target_states = encoded["target_states"]

    expected_action_steps = raw_horizon // frameskip
    expected_target_steps = expected_action_steps - history_size + 1

    if context_emb.ndim != 3 or context_emb.shape[1] != history_size:
        raise ValueError(
            f"context_emb must be (B, {history_size}, D), got "
            f"{tuple(context_emb.shape)}"
        )
    if action_emb.ndim != 3 or action_emb.shape[:2] != (
        context_emb.shape[0],
        expected_action_steps,
    ):
        raise ValueError(
            f"action_emb must be (B, {expected_action_steps}, D), got "
            f"{tuple(action_emb.shape)}"
        )
    if target_states.shape[:2] != (context_emb.shape[0], expected_target_steps):
        raise ValueError(
            f"target_states must be (B, {expected_target_steps}, 7), got "
            f"{tuple(target_states.shape)}"
        )


def open_loop_rollout_from_embeddings(
    encoded,
    model,
    device,
    batch_size=256,
    history_size=3,
):
    """Autoregressively roll LeWM forward using precomputed embeddings.

    At target index 0, the predictor sees:
    - embeddings: [z0,   z5,    z10]
    - actions:    [a0-5, a5-10, a10-15]
    and returns z^15 from the last output position.

    At target index 1, the earliest embedding is effectively dropped by taking
    the last `history_size` items:
    - embeddings: [z5,    z10,    z^15]
    - actions:    [a5-10, a10-15, a15-20]
    and returns z^20. This continues until z^100.
    """
    validate_encoded_rollout_dataset(
        encoded,
        history_size=history_size,
        frameskip=int(encoded["action_tokens"].shape[-1] // 2),
        raw_horizon=int(encoded["action_ends"][-1].item()),
    )

    context_emb_all = encoded["context_emb"].float()
    action_emb_all = encoded["action_emb"].float()
    target_states = encoded["target_states"].float()
    num_episodes = context_emb_all.size(0)
    num_targets = target_states.size(1)
    expected_targets = action_emb_all.size(1) - history_size + 1
    if num_targets != expected_targets:
        raise ValueError(
            f"target_states T={num_targets} does not match action_emb T="
            f"{action_emb_all.size(1)} and history_size={history_size}"
        )

    predicted_batches = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, num_episodes, batch_size):
            end = min(start + batch_size, num_episodes)
            emb_history = context_emb_all[start:end].to(device)
            action_emb = action_emb_all[start:end].to(device)
            predicted_latents = []

            for target_idx in range(num_targets):
                emb_window = emb_history[:, -history_size:]
                action_window = action_emb[:, target_idx : target_idx + history_size]
                pred_seq = model.predict(emb_window, action_window)
                next_latent = pred_seq[:, -1:]
                predicted_latents.append(next_latent.cpu())
                emb_history = torch.cat([emb_history, next_latent], dim=1)

            predicted_batches.append(torch.cat(predicted_latents, dim=1))

    predicted_latents = torch.cat(predicted_batches, dim=0).float()
    if predicted_latents.shape[:2] != target_states.shape[:2]:
        raise ValueError(
            f"predicted_latents shape {tuple(predicted_latents.shape)} does not "
            f"align with target_states shape {tuple(target_states.shape)}"
        )
    return predicted_latents


def save_rollout_encoded_dataset(encoded, metadata, save_path):
    """Save encoded context/action tensors for reusable rollout experiments."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "encoded": encoded}, save_path)
    return save_path


def save_rollout_predictions(encoded, predicted_latents, metadata, save_path):
    """Save predicted latents with matching ground-truth states and horizons."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "predicted": {
            "episode_idx": encoded["episode_idx"],
            "predicted_latents": predicted_latents.cpu().float(),
            "target_states": encoded["target_states"],
            "target_steps": encoded["target_steps"],
            "context_steps": encoded["context_steps"],
            "action_starts": encoded["action_starts"],
            "action_ends": encoded["action_ends"],
        },
    }
    torch.save(payload, save_path)
    return save_path
