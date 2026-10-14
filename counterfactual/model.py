"""Frozen LeWM inference helpers used by the Phase I experiment."""

from pathlib import Path

import h5py
import numpy as np
import torch

from counterfactual.contracts import ActionNormalizer, TimeActionContract
from probe.embedding_cache import get_image_transform, preprocess_pixels
from probe.rollout_ablation import default_cache_dir, eval_device, load_released_lewm_world_model
from probe.rollout_ablation import open_loop_rollout_from_embeddings


IMAGE_PREPROCESSING_CONTRACT = {
    "input_layout": "raw Push-T RGB uint8 frames (T, H, W, C)",
    "model_layout": "(1, T, C, H, W)",
    "transform": [
        "torchvision.transforms.v2.ToImage",
        "torchvision.transforms.v2.ToDtype(torch.float32, scale=True)",
        "torchvision.transforms.v2.Normalize(ImageNet mean/std from stable_pretraining)",
        "torchvision.transforms.v2.Resize(image_size, bilinear, antialias=True)",
    ],
}


def resolve_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Resolve the project-local Stable World Model cache without guessing."""
    return Path(cache_dir) if cache_dir is not None else default_cache_dir()


def checkpoint_path(cache_dir: str | Path, policy_name: str) -> Path:
    """Return the serialized model-object checkpoint path for a policy name."""
    policy_path = Path(policy_name)
    if policy_path.is_absolute() or ".." in policy_path.parts or policy_path.suffix:
        raise ValueError("policy_name must be a cache-relative checkpoint stem")
    return (Path(cache_dir) / f"{policy_name}_object.ckpt").resolve()


def load_frozen_lewm(policy_name: str, cache_dir: str | Path, device: str | None = None):
    """Load the released JEPA checkpoint in deterministic inference mode."""
    return load_released_lewm_world_model(
        policy_name=str(checkpoint_path(cache_dir, policy_name)).removesuffix("_object.ckpt"),
        cache_dir=cache_dir,
        device=device or eval_device(),
    )


def action_normalizer_from_hdf5(dataset_path: str | Path) -> ActionNormalizer:
    """Reproduce ``utils.get_column_normalizer`` for the training action column."""
    with h5py.File(dataset_path, "r") as handle:
        raw_actions = np.array(handle["action"][:])
    actions = torch.from_numpy(raw_actions)
    actions = actions[~torch.isnan(actions).any(dim=1)]
    if len(actions) < 2 or not torch.isfinite(actions).all():
        raise ValueError("Action statistics require at least two finite non-NaN rows")
    return ActionNormalizer(mean=actions.mean(dim=0), std=actions.std(dim=0))


def pixels_hwc_to_btc(pixels_hwc: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert raw Push-T images from (T,H,W,C) into LeWM's (1,T,C,H,W)."""
    pixels = torch.as_tensor(pixels_hwc)
    if pixels.dtype != torch.uint8:
        raise ValueError("Raw RGB frames must be uint8 so image scaling matches training")
    if pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise ValueError(
            "Expected raw RGB frames with shape (T, H, W, 3), got "
            f"{tuple(pixels.shape)}"
        )
    return pixels.permute(0, 3, 1, 2).unsqueeze(0).contiguous()


def frames_hwc_to_b1chw(pixels_hwc: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert independent raw frames from (B,H,W,C) to (B,1,C,H,W).

    Cache validation requires one output row per independent frame. LeWM flattens
    B*T for encoding; the distinction controls output indexing, not ViT positions.
    """
    pixels = torch.as_tensor(pixels_hwc)
    if pixels.dtype != torch.uint8:
        raise ValueError("Raw RGB frames must be uint8 so image scaling matches training")
    if pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise ValueError(
            "Expected independent raw RGB frames with shape (B, H, W, 3), got "
            f"{tuple(pixels.shape)}"
        )
    return pixels.permute(0, 3, 1, 2).unsqueeze(1).contiguous()


def require_tensor(tensor, shape: tuple, name: str) -> None:
    """Reject malformed or non-finite model inputs and outputs before comparisons."""
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(tensor.shape)}; expected {shape}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")


def validate_model_contract(model, contract: TimeActionContract) -> dict:
    """Check the loaded architecture against the declared released-model contract."""
    observed = {
        "history_size": int(model.predictor.pos_embedding.shape[1]),
        "latent_dim": int(model.encoder.config.hidden_size),
        "token_dim": int(model.action_encoder.patch_embed.in_channels),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "frozen": all(not parameter.requires_grad for parameter in model.parameters()),
        "all_modules_eval": all(not module.training for module in model.modules()),
    }
    for name, expected in {
        "history_size": contract.history_size,
        "latent_dim": contract.latent_dim,
        "token_dim": contract.token_dim,
    }.items():
        if observed[name] != expected:
            raise ValueError(f"Checkpoint {name}={observed[name]} disagrees with contract {expected}")
    if not observed["frozen"] or not observed["all_modules_eval"]:
        raise ValueError("Checkpoint must be frozen and every module in eval mode")
    return observed


def run_frozen_window(
    model,
    pixels_hwc: np.ndarray | torch.Tensor,
    raw_actions: np.ndarray | torch.Tensor,
    normalizer: ActionNormalizer,
    contract: TimeActionContract,
    device: str,
    image_size: int,
    lookahead_steps: int = 1,
) -> dict[str, torch.Tensor]:
    """Encode a three-frame window and predict the aligned next latent.

    The first three five-step blocks cover 0→5, 5→10, and 10→15. Additional
    blocks are encoded together for the requested rollout. The final one-step
    predictor output corresponds to absolute raw step 15.
    """
    pixels = pixels_hwc_to_btc(pixels_hwc)
    if pixels.shape[1] != contract.history_size:
        raise ValueError(
            f"Expected {contract.history_size} context frames, got {pixels.shape[1]}"
        )

    raw_actions = torch.as_tensor(raw_actions)
    required_actions = int(contract.rollout_indices(lookahead_steps)["action_ends"][-1])
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
        prediction = model.predict(context_emb, action_emb[:, :contract.history_size])

    expected_shape = (1, contract.history_size, contract.latent_dim)
    for name, tensor in {"context_emb": context_emb, "prediction": prediction}.items():
        require_tensor(tensor, expected_shape, name)
    require_tensor(action_emb, (1, required_actions // contract.frameskip, contract.latent_dim), "action_emb")

    return {
        "context_emb": context_emb.detach().cpu(),
        "action_tokens": action_tokens.detach().cpu(),
        "action_emb": action_emb.detach().cpu(),
        "prediction": prediction.detach().cpu(),
    }


def run_frozen_rollout(
    model, pixels_hwc, raw_actions, normalizer, contract, device, image_size,
    lookahead_steps: int = 5,
) -> dict[str, torch.Tensor]:
    """Reuse the existing sliding-history rollout with training-standardized actions."""
    indices = contract.rollout_indices(lookahead_steps)
    required_actions = int(indices["action_ends"][-1])
    raw_actions = torch.as_tensor(raw_actions)
    require_tensor(raw_actions, (required_actions, contract.action_dim), "raw_actions")
    first = run_frozen_window(
        model, pixels_hwc, raw_actions,
        normalizer, contract, device, image_size, lookahead_steps,
    )
    encoded = {
        "context_emb": first["context_emb"],
        "action_emb": first["action_emb"],
        "action_tokens": first["action_tokens"],
        "target_states": torch.zeros((1, lookahead_steps, 7)),
        "action_ends": indices["action_ends"],
    }
    predicted = open_loop_rollout_from_embeddings(
        encoded, model, device, batch_size=1, history_size=contract.history_size,
    )
    require_tensor(predicted, (1, lookahead_steps, contract.latent_dim), "rollout")
    if not torch.equal(predicted[:, :1], first["prediction"][:, -1:]):
        raise ValueError("First rollout output disagrees with one-step prediction")
    return {**first, "rollout": predicted}
