"""Frozen LeWM inference helpers used by the Phase I experiment."""

from pathlib import Path

import h5py
import numpy as np
import torch

from counterfactual.contracts import ActionNormalizer, TimeActionContract
from probe.embedding_cache import get_image_transform, preprocess_pixels
from probe.rollout_ablation import default_cache_dir, eval_device, load_released_lewm_world_model


IMAGE_PREPROCESSING_CONTRACT = {
    "input_layout": "raw Push-T RGB uint8 frames (T, H, W, C)",
    "model_layout": "(1, T, C, H, W)",
    "transform": [
        "torchvision.transforms.v2.ToImage",
        "torchvision.transforms.v2.ToDtype(torch.float32, scale=True)",
        "torchvision.transforms.v2.Normalize(ImageNet mean/std from stable_pretraining)",
        "torchvision.transforms.v2.Resize(224)",
    ],
}


def resolve_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Resolve the project-local Stable World Model cache without guessing."""
    return Path(cache_dir) if cache_dir is not None else default_cache_dir()


def checkpoint_path(cache_dir: str | Path, policy_name: str) -> Path:
    """Return the serialized model-object checkpoint path for a policy name."""
    return Path(cache_dir) / f"{policy_name}_object.ckpt"


def load_frozen_lewm(policy_name: str, cache_dir: str | Path, device: str | None = None):
    """Load the released JEPA checkpoint in deterministic inference mode."""
    return load_released_lewm_world_model(
        policy_name=policy_name,
        cache_dir=cache_dir,
        device=device or eval_device(),
    )


def action_normalizer_from_hdf5(dataset_path: str | Path) -> ActionNormalizer:
    """Reproduce ``utils.get_column_normalizer`` for the training action column."""
    with h5py.File(dataset_path, "r") as handle:
        raw_actions = np.array(handle["action"][:])
    actions = torch.from_numpy(raw_actions)
    actions = actions[~torch.isnan(actions).any(dim=1)]
    if not len(actions):
        raise ValueError("No finite actions were found in the dataset")
    return ActionNormalizer(mean=actions.mean(dim=0), std=actions.std(dim=0))


def pixels_hwc_to_btc(pixels_hwc: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert raw Push-T images from (T,H,W,C) into LeWM's (1,T,C,H,W)."""
    pixels = torch.as_tensor(pixels_hwc)
    if pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise ValueError(
            "Expected raw RGB frames with shape (T, H, W, 3), got "
            f"{tuple(pixels.shape)}"
        )
    return pixels.permute(0, 3, 1, 2).unsqueeze(0).contiguous()


def run_frozen_window(
    model,
    pixels_hwc: np.ndarray | torch.Tensor,
    raw_actions: np.ndarray | torch.Tensor,
    normalizer: ActionNormalizer,
    contract: TimeActionContract,
    device: str,
    image_size: int,
) -> dict[str, torch.Tensor]:
    """Encode a three-frame window and predict the aligned next latent.

    ``raw_actions`` must include exactly three five-step blocks: actions for
    0→5, 5→10, and 10→15.  The final predictor output is therefore the latent
    at absolute raw step 15.
    """
    pixels = pixels_hwc_to_btc(pixels_hwc)
    if pixels.shape[1] != contract.history_size:
        raise ValueError(
            f"Expected {contract.history_size} context frames, got {pixels.shape[1]}"
        )

    raw_actions = torch.as_tensor(raw_actions)
    required_actions = contract.history_size * contract.frameskip
    if raw_actions.shape != (required_actions, contract.action_dim):
        raise ValueError(
            "Expected raw actions with shape "
            f"({required_actions}, {contract.action_dim}), got {tuple(raw_actions.shape)}"
        )

    image_transform = get_image_transform(image_size)
    model_pixels = preprocess_pixels(pixels, image_transform, device)
    action_tokens = normalizer.tokenize(raw_actions, contract.frameskip).unsqueeze(0)

    model.eval()
    with torch.inference_mode():
        context_emb = model.encode({"pixels": model_pixels})["emb"]
        action_emb = model.action_encoder(action_tokens.to(device))
        prediction = model.predict(context_emb, action_emb)

    return {
        "context_emb": context_emb.detach().cpu(),
        "action_tokens": action_tokens.detach().cpu(),
        "action_emb": action_emb.detach().cpu(),
        "prediction": prediction.detach().cpu(),
    }
