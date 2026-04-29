import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
 
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf, open_dict
from torchvision.transforms import v2 as transforms

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack
import stable_worldmodel as swm



def eval_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Matches eval image preprocessing: image conversion, float scaling, ImageNet normalize, resize
def get_image_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def load_frozen_encoder(policy_name: str, device: str, cache_dir=None):
    model = swm.policy.AutoCostModel(policy_name, cache_dir=cache_dir) # get the checkpoint model
    model = model.to(device)
    model.eval() # switches the model into evaluation mode, it should be fixed
    model.requires_grad_(False) # disable gradients for all params
    model.interpolate_pos_encoding = True
    return model


def preprocess_pixels(pixels, image_transform, device):
    # HDF5Dataset returns pixels as (B, T, C, H, W) e.g. (64, 1, 3, 224, 224). The image transform expects
    # image tensors as (N, C, H, W), so flatten B and T before preprocessing.
    b, t = pixels.shape[:2]
    pixels = pixels.reshape(b * t, *pixels.shape[2:])
    pixels = image_transform(pixels)
    pixels = pixels.reshape(b, t, *pixels.shape[1:])
    return pixels.to(device)


# extract emb in batches
def extract_embeddings(dataset, indices, model, device, image_transform, batch_size=256):
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)

    embeddings = []
    states = []
    proprios = []
    episode_ids = []
    step_ids = []

    with torch.inference_mode(): # disable gradient tracking for efficiency
        for batch in loader:
            pixels = preprocess_pixels(batch["pixels"], image_transform, device)
            emb = model.encode({"pixels": pixels})["emb"]
            embeddings.append(emb[:, 0].cpu()) # B T E -> B E:(64, 1, 192) -> (64, 192)
            states.append(batch["state"][:, 0].float())
            proprios.append(batch["proprio"][:, 0].float())
            episode_ids.append(batch["episode_idx"][:, 0])
            step_ids.append(batch["step_idx"][:, 0])

    return {
        "emb": torch.cat(embeddings, dim=0),
        "state": torch.cat(states, dim=0),
        "proprio": torch.cat(proprios, dim=0),
        "episode_idx": torch.cat(episode_ids, dim=0),
        "step_idx": torch.cat(step_ids, dim=0),
    }


def build_agent_location_pairs(encoded_data):
    embeddings = encoded_data["emb"].float()
    agent_location = encoded_data["state"][:, 0:2].float() # state[0] = agent_x, state[1] = agent_y
    return {
        "x": embeddings,
        "y": agent_location,
        "target_name": "agent_location",
        "episode_idx": encoded_data["episode_idx"], # metadata
        "step_idx": encoded_data["step_idx"],       # metadata
    }


def precompute_encoded_split(
    dataset,
    indices,
    model,
    device,
    image_transform,
    split_name,
    batch_size,
    log_every=100,
):
    loader = make_probe_loader(dataset, indices, batch_size=batch_size, shuffle=False)
    n_samples = len(indices) # count how many frames are in this split (e.g. train, val)
    encoded = {
        "emb": torch.empty((n_samples, 192), dtype=torch.float32),
        "state": torch.empty((n_samples, 7), dtype=torch.float32),
        "proprio": torch.empty((n_samples, 4), dtype=torch.float32),
        "episode_idx": torch.empty((n_samples,), dtype=torch.long),
        "step_idx": torch.empty((n_samples,), dtype=torch.long),
    }

    write_pos = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, start=1):
            emb = encode_batch(batch, model, device, image_transform)
            batch_size_actual = emb.size(0)
            target_slice = slice(write_pos, write_pos + batch_size_actual)

            encoded["emb"][target_slice] = emb.cpu()
            encoded["state"][target_slice] = batch["state"][:, 0].float()
            encoded["proprio"][target_slice] = batch["proprio"][:, 0].float()
            encoded["episode_idx"][target_slice] = batch["episode_idx"][:, 0].long()
            encoded["step_idx"][target_slice] = batch["step_idx"][:, 0].long()

            write_pos += batch_size_actual
            if batch_idx % log_every == 0 or write_pos == n_samples:
                print(
                    f"precompute {split_name}: {write_pos}/{n_samples} "
                    f"frames encoded"
                )

    return encoded


def encoded_cache_path(cache_dir, dataset_name, policy_name, split_name, seed, img_size):
    safe_policy_name = policy_name.replace("/", "_")
    filename = (
        f"{dataset_name}_{safe_policy_name}_{split_name}_"
        f"seed{seed}_img{img_size}_encoded.pt"
    )
    return cache_dir / "probes" / "encoded" / filename


def load_or_precompute_encoded_split(
    dataset,
    indices,
    model,
    device,
    image_transform,
    split_name,
    cache_path,
    metadata,
    batch_size,
):
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        print(f"loaded cached {split_name} embeddings from {cache_path}")
        return payload["encoded"]

    encoded = precompute_encoded_split(
        dataset,
        indices,
        model,
        device,
        image_transform,
        split_name=split_name,
        batch_size=batch_size,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "encoded": encoded}, cache_path)
    print(f"saved cached {split_name} embeddings to {cache_path}")
    return encoded


def move_pairs_to_device(pairs, device):
    return {
        "x": pairs["x"].to(device),
        "y": pairs["y"].to(device),
        "target_name": pairs["target_name"],
        "episode_idx": pairs["episode_idx"],
        "step_idx": pairs["step_idx"],
    }


def fit_agent_location_normalizer(train_pairs, device):
    target = train_pairs["y"]
    mean = target.mean(dim=0, keepdim=True).to(device)
    std = target.std(dim=0, keepdim=True).clamp_min(1e-6).to(device)
    return mean, std


def make_probe_loader(dataset, indices, batch_size, shuffle):
    subset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def make_pair_loader(pairs, batch_size, shuffle):
    tensor_dataset = torch.utils.data.TensorDataset(pairs["x"], pairs["y"])
    return torch.utils.data.DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def encode_batch(batch, model, device, image_transform):
    pixels = preprocess_pixels(batch["pixels"], image_transform, device)
    with torch.no_grad():
        emb = model.encode({"pixels": pixels})["emb"][:, 0]
    return emb.detach()


def pearson_r(pred, target):
    pred_centered = pred - pred.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    numerator = (pred_centered * target_centered).sum(dim=0)
    denominator = torch.sqrt(
        pred_centered.square().sum(dim=0) * target_centered.square().sum(dim=0)
    ).clamp_min(1e-8)
    r_per_dim = numerator / denominator
    return r_per_dim.mean(), r_per_dim


def evaluate_linear_probe(
    pairs,
    probe,
    target_mean,
    target_std,
    batch_size,
):
    loader = make_pair_loader(pairs, batch_size=batch_size, shuffle=False)
    probe.eval()

    preds = []
    targets = []
    with torch.no_grad():
        for emb, target in loader:
            pred_norm = probe(emb)
            pred = pred_norm * target_std + target_mean
            preds.append(pred.cpu())
            targets.append(target.cpu())

    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    mse = F.mse_loss(pred, target)
    r_mean, r_per_dim = pearson_r(pred, target)
    return {
        "mse": mse.item(),
        "rmse": torch.sqrt(mse).item(),
        "r_mean": r_mean.item(),
        "r_x": r_per_dim[0].item(),
        "r_y": r_per_dim[1].item(),
    }


def train_agent_location_linear_probe(
    train_pairs,
    val_pairs,
    device,
    save_path,
    batch_size=64,
    max_epochs=20,
    patience=3,
    lr=1e-3,
    weight_decay=1e-4,
):
    train_pairs = move_pairs_to_device(train_pairs, device)
    val_pairs = move_pairs_to_device(val_pairs, device)
    target_mean, target_std = fit_agent_location_normalizer(train_pairs, device)
    train_loader = make_pair_loader(train_pairs, batch_size=batch_size, shuffle=True)

    probe = torch.nn.Linear(192, 2).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_mse = float("inf")
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0

    print(
        "Training linear probe for agent_location "
        f"(input_dim=192, output_dim=2, batch_size={batch_size}, "
        f"lr={lr}, weight_decay={weight_decay}, max_epochs={max_epochs}, "
        f"patience={patience})"
    )
    print(f"target_mean={target_mean.cpu().numpy().round(4).tolist()}")
    print(f"target_std={target_std.cpu().numpy().round(4).tolist()}")

    for epoch in range(1, max_epochs + 1):
        probe.train()
        train_loss_sum = 0.0
        train_count = 0

        for emb, target in train_loader:
            target_norm = (target - target_mean) / target_std

            pred_norm = probe(emb)
            loss = F.mse_loss(pred_norm, target_norm)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size_actual = target.size(0)
            train_loss_sum += loss.item() * batch_size_actual
            train_count += batch_size_actual

        train_loss = train_loss_sum / train_count
        val_stats = evaluate_linear_probe(
            val_pairs,
            probe,
            target_mean,
            target_std,
            batch_size,
        )

        improved = val_stats["mse"] < best_val_mse
        if improved:
            best_val_mse = val_stats["mse"]
            best_epoch = epoch
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in probe.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        marker = "*" if improved else ""
        print(
            f"epoch={epoch:03d} train_norm_mse={train_loss:.6f} "
            f"val_mse={val_stats['mse']:.6f} val_rmse={val_stats['rmse']:.6f} "
            f"val_r={val_stats['r_mean']:.4f} "
            f"(r_x={val_stats['r_x']:.4f}, r_y={val_stats['r_y']:.4f}) {marker}"
        )

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch was {best_epoch} with val_mse={best_val_mse:.6f}"
            )
            break

    if best_state_dict is not None:
        probe.load_state_dict(best_state_dict)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "target_name": "agent_location",
            "probe_type": "linear",
            "probe_state_dict": probe.state_dict(),
            "target_mean": target_mean.cpu(),
            "target_std": target_std.cpu(),
            "best_epoch": best_epoch,
            "best_val_mse": best_val_mse,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "max_epochs": max_epochs,
            "patience": patience,
        },
        save_path,
    )
    print(f"Saved best linear probe to {save_path}")
    return probe


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.get("cache_dir") or swm.data.utils.get_cache_dir())
    keys_to_load = [
        "pixels",
        "state",
        "proprio",
        "episode_idx",
        "step_idx",
    ]
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        frameskip=1, # each dataset item contains one raw frame
        num_steps=1, # and one timestep
        keys_to_load=keys_to_load,
        keys_to_cache=["state", "proprio", "episode_idx", "step_idx"],
        cache_dir=dataset_path,
    )
    return dataset


def split_by_episode(dataset, train_ratio=0.8, val_ratio=0.1, seed=42): # shuffles unique episode_idx values and creates train/val/test sets without mixing frames from the same episode
    episode_ids = dataset.get_col_data("episode_idx")
    unique_episodes = np.unique(episode_ids)

    rng = np.random.default_rng(seed)
    shuffled_episodes = unique_episodes.copy()
    rng.shuffle(shuffled_episodes)

    n_train = int(len(shuffled_episodes) * train_ratio)
    n_val = int(len(shuffled_episodes) * val_ratio)

    train_episodes = shuffled_episodes[:n_train]
    val_episodes = shuffled_episodes[n_train : n_train + n_val]
    test_episodes = shuffled_episodes[n_train + n_val :]

    train_mask = np.isin(episode_ids, train_episodes)
    val_mask = np.isin(episode_ids, val_episodes)
    test_mask = np.isin(episode_ids, test_episodes)

    splits = {
        "train": np.flatnonzero(train_mask),
        "val": np.flatnonzero(val_mask),
        "test": np.flatnonzero(test_mask),
    }
    return splits, {
        "train": train_episodes,
        "val": val_episodes,
        "test": test_episodes,
    }


def visualize_pixels(sample):
    pixels = sample["pixels"]
    print(f'pixels: expected shape=(1, 3, 224, 224), actual shape={tuple(pixels.shape)}, dtype={pixels.dtype}')

    timestep_0_pixels = pixels[0]
    image = timestep_0_pixels.permute(1, 2, 0)

    if image.dtype == torch.uint8:
        image_for_display = image.cpu().numpy()
    else:
        image_for_display = image.detach().cpu().float().clamp(0.0, 1.0).numpy()

    state_t0 = sample["state"][0]
    image_height, image_width = image_for_display.shape[:2]
    world_size = 512
    world_to_image_scale = torch.tensor(
        [image_width / world_size, image_height / world_size],
        dtype=state_t0.dtype,
        device=state_t0.device,
    )

    agent_xy_world = state_t0[0:2]
    block_xy_world = state_t0[2:4]
    agent_xy_image = agent_xy_world * world_to_image_scale
    block_xy_image = block_xy_world * world_to_image_scale

    plt.imshow(image_for_display)
    plt.scatter(
        agent_xy_image[0].item(),
        agent_xy_image[1].item(),
        c="red",
        marker="x",
        s=80,
        label="state[0:2] agent location",
    )
    plt.scatter(
        block_xy_image[0].item(),
        block_xy_image[1].item(),
        c="orange",
        marker="+",
        s=100,
        label="state[2:4] block location",
    )
    plt.title("sample['pixels'][0]")
    plt.axis("off")
    plt.legend()
    plt.show()

    return agent_xy_image, block_xy_image


def print_physical_labels(sample, agent_xy_image, block_xy_image):
    state_t0 = sample["state"][0]
    proprio_t0 = sample["proprio"][0]

    agent_xy_world = state_t0[0:2]
    block_xy_world = state_t0[2:4]
    block_angle_rad = state_t0[4]
    block_angle_deg = torch.rad2deg(block_angle_rad)
    agent_velocity_world = state_t0[5:7]

    print(f"state[0]: {state_t0}")
    print(f"proprio[0]: {proprio_t0}")
    print(f"state[0:2] agent location in world coordinates: {agent_xy_world}")
    print(f"state[0:2] agent location in image coordinates: {agent_xy_image}")
    print(f"state[2:4] block location in world coordinates: {block_xy_world}")
    print(f"state[2:4] block location in image coordinates: {block_xy_image}")
    print(f"state[4] block angle in radians: {block_angle_rad}")
    print(f"state[4] block angle in degrees: {block_angle_deg}")
    print(f"state[5:7] agent velocity in world coordinates per step: {agent_velocity_world}")
    print(f"state[0:2] agent location: {state_t0[0:2]}")
    print(f"state[2:4] block location: {state_t0[2:4]}")
    print(f"state[4] block angle: {state_t0[4]}")
    print(f"state[5:7] agent velocity: {state_t0[5:7]}")


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    splits, split_episodes = split_by_episode(dataset, seed=cfg.seed)
    device = eval_device()
    cache_dir = Path(cfg.get("cache_dir") or swm.data.utils.get_cache_dir())
    policy_name = "pusht/lewm"

    model = load_frozen_encoder(policy_name, device, cache_dir=cache_dir)
    image_transform = get_image_transform(cfg.eval.img_size)

    print("Dataset columns:", dataset.column_names)
    print("Dataset length:", len(dataset))
    print("Device:", device)
    for split_name, indices in splits.items():
        print(
            f"{split_name}: {len(split_episodes[split_name])} episodes, "
            f"{len(indices)} frames"
        )

    # Create train, val, test splits with encoded frames
    encoded_splits = {}
    for split_name in ("train", "val", "test"):
        metadata = {
            "dataset_name": cfg.eval.dataset_name,
            "policy_name": policy_name,
            "split_name": split_name,
            "seed": int(cfg.seed),
            "img_size": int(cfg.eval.img_size),
            "num_frames": int(len(splits[split_name])),
            "episode_count": int(len(split_episodes[split_name])),
            "keys": ["emb", "state", "proprio", "episode_idx", "step_idx"],
            "state_layout": [
                "agent_x",
                "agent_y",
                "block_x",
                "block_y",
                "block_angle",
                "agent_vx",
                "agent_vy",
            ],
        }
        encoded_splits[split_name] = load_or_precompute_encoded_split(
            dataset,
            splits[split_name],
            model,
            device,
            image_transform,
            split_name=split_name,
            cache_path=encoded_cache_path(
                cache_dir,
                cfg.eval.dataset_name,
                policy_name,
                split_name,
                cfg.seed,
                cfg.eval.img_size,
            ),
            metadata=metadata,
            batch_size=2048,
        )

    train_encoded = encoded_splits["train"]
    val_encoded = encoded_splits["val"]
    train_pairs = build_agent_location_pairs(train_encoded)
    val_pairs = build_agent_location_pairs(val_encoded)

    print(
        "precomputed train pairs: "
        f"x shape={tuple(train_pairs['x'].shape)}, "
        f"y shape={tuple(train_pairs['y'].shape)}"
    )
    print(
        "precomputed val pairs: "
        f"x shape={tuple(val_pairs['x'].shape)}, "
        f"y shape={tuple(val_pairs['y'].shape)}"
    )

    probe_save_path = cache_dir / "probes" / "pusht_agent_location_linear.pt"
    train_agent_location_linear_probe(
        train_pairs,
        val_pairs,
        device,
        save_path=probe_save_path,
        batch_size=1024,
        max_epochs=20,
        patience=3,
        lr=1e-3,
        weight_decay=1e-4,
    )

    ## Inspect a sample
    # sample = dataset[30]

    # for key, value in sample.items():
    #     if torch.is_tensor(value):
    #         print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")

    #     else:
    #         print(f"{key}: type={type(value)}, value={value}")

    # agent_xy_image, block_xy_image = visualize_pixels(sample)
    # print_physical_labels(sample, agent_xy_image, block_xy_image)



    

if __name__ == "__main__":
    run()