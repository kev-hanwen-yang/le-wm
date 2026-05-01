from pathlib import Path
import hydra
 
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig
import stable_worldmodel as swm
from probe.data import get_dataset, split_by_episode
from probe.embedding_cache import (
    encoded_cache_path,
    eval_device,
    get_image_transform,
    load_frozen_encoder,
    load_or_precompute_encoded_split,
)
from probe.targets import TARGET_BUILDERS
from probe.train import train_probe


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
    target_name = cfg.get("target_name", "agent_location")
    probe_type = cfg.get("probe_type", "linear")
    mlp_hidden_dim = int(cfg.get("mlp_hidden_dim", 256))
    mlp_num_hidden_layers = int(cfg.get("mlp_num_hidden_layers", 1))
    mlp_dropout = float(cfg.get("mlp_dropout", 0.1))
    probe_seed = cfg.get("probe_seed", None)
    if probe_seed is not None:
        probe_seed = int(probe_seed)
    if target_name not in TARGET_BUILDERS:
        valid_targets = ", ".join(TARGET_BUILDERS)
        raise ValueError(f"Unknown target_name={target_name!r}. Valid targets: {valid_targets}")
    build_target_pairs = TARGET_BUILDERS[target_name]

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
    train_pairs = build_target_pairs(train_encoded)
    val_pairs = build_target_pairs(val_encoded)

    print(
        "precomputed train pairs: "
        f"embeddings shape={tuple(train_pairs['embeddings'].shape)}, "
        f"target_name={train_pairs['target_name']}, "
        f"target shape={tuple(train_pairs['target'].shape)}"
    )
    print(
        "precomputed val pairs: "
        f"embeddings shape={tuple(val_pairs['embeddings'].shape)}, "
        f"target_name={val_pairs['target_name']}, "
        f"target shape={tuple(val_pairs['target'].shape)}"
    )

    probe_dir = cache_dir / "probes"
    if probe_seed is not None:
        # Keep probe-training seeds in separate folders so repeated seeds do not
        # overwrite the original single-run checkpoints or each other.
        probe_dir = probe_dir / f"seed{probe_seed}"
    probe_save_path = probe_dir / f"pusht_{target_name}_{probe_type}.pt"
    train_probe(
        train_pairs,
        val_pairs,
        device,
        save_path=probe_save_path,
        batch_size=1024,
        max_epochs=20,
        patience=3,
        lr=1e-3,
        weight_decay=1e-4,
        probe_type=probe_type,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_num_hidden_layers=mlp_num_hidden_layers,
        mlp_dropout=mlp_dropout,
        probe_seed=probe_seed,
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