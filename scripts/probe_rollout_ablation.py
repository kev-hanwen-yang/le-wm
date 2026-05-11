#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import torch


# Script: build Push-T long-horizon rollout ablation artifacts.
#
# This script is the runnable entrypoint for the open-loop ablation. It first
# reconstructs raw rollout windows from the original HDF5 dataset and the Table
# 1 encoded test split, then saves:
# - encoded rollout dataset: context_emb/action_emb/target_states
# - predicted latent dataset: zhat15, zhat20, ..., zhat100 with target states


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probe.embedding_cache import encoded_cache_path, get_image_transform  # noqa: E402
from probe.rollout_ablation import (  # noqa: E402
    encode_rollout_windows,
    eval_device,
    load_released_lewm_world_model,
    open_loop_rollout_from_embeddings,
    rollout_artifact_paths,
    save_rollout_encoded_dataset,
    save_rollout_predictions,
)
from probe.rollout_windows import (  # noqa: E402
    build_rollout_windows,
    load_encoded_test_episode_ids,
    load_raw_pusht_dataset,
    select_rollout_episode_ids,
)


def parse_args():
    """Parse CLI options for the rollout-ablation artifact builder."""
    parser = argparse.ArgumentParser(
        description="Build encoded and predicted-latent datasets for Push-T rollout ablation."
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--dataset-name", default="pusht_expert_train")
    parser.add_argument("--policy-name", default="pusht/lewm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--raw-horizon", type=int, default=100)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--encode-batch-size", type=int, default=16)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def default_cache_dir():
    """Prefer the project-local `.stable-wm` directory for dataset artifacts."""
    project_cache_dir = REPO_ROOT / ".stable-wm"
    if project_cache_dir.exists():
        return project_cache_dir

    import stable_worldmodel as swm

    return Path(swm.data.utils.get_cache_dir())


def build_metadata(args, cache_dir, device, selected_episode_ids):
    """Create metadata saved beside encoded and predicted rollout tensors."""
    return {
        "dataset_name": args.dataset_name,
        "policy_name": args.policy_name,
        "split_seed": args.seed,
        "img_size": args.img_size,
        "num_episodes": int(selected_episode_ids.numel()),
        "requested_num_episodes": args.num_episodes,
        "raw_horizon": args.raw_horizon,
        "history_size": args.history_size,
        "frameskip": args.frameskip,
        "device": device,
        "cache_dir": str(cache_dir),
        "selected_episode_ids": selected_episode_ids.tolist(),
    }


def print_tensor_summary(name, tensor):
    """Print a compact tensor summary for verification."""
    print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")


def main():
    """Run the full artifact-building pipeline for the rollout ablation."""
    args = parse_args()
    cache_dir = args.cache_dir or default_cache_dir()
    device = eval_device() if args.device == "auto" else args.device
    min_raw_length = args.raw_horizon + 1

    test_cache_path = encoded_cache_path(
        cache_dir,
        args.dataset_name,
        args.policy_name,
        "test",
        args.seed,
        args.img_size,
    )
    if not test_cache_path.exists():
        raise FileNotFoundError(
            f"Missing encoded test cache: {test_cache_path}. "
            "Run probe.py first to create the Table 1 test split cache."
        )

    paths = rollout_artifact_paths(
        cache_dir=cache_dir,
        dataset_name=args.dataset_name,
        policy_name=args.policy_name,
        split_seed=args.seed,
        img_size=args.img_size,
        num_episodes=args.num_episodes,
        raw_horizon=args.raw_horizon,
    )
    if not args.overwrite and paths["encoded"].exists() and paths["predicted"].exists():
        print(f"encoded rollout dataset already exists: {paths['encoded']}")
        print(f"predicted rollout dataset already exists: {paths['predicted']}")
        return

    print(f"device: {device}")
    print(f"test encoded cache: {test_cache_path}")
    test_episode_ids = load_encoded_test_episode_ids(test_cache_path)
    raw_dataset = load_raw_pusht_dataset(
        dataset_name=args.dataset_name,
        cache_dir=cache_dir,
    )
    selected_episode_ids = select_rollout_episode_ids(
        test_episode_ids,
        raw_dataset,
        max_episodes=args.num_episodes,
        min_raw_length=min_raw_length,
    )
    if selected_episode_ids.numel() == 0:
        raise RuntimeError(
            f"No test episodes have raw length >= {min_raw_length}. "
            "Decrease --raw-horizon or inspect the dataset."
        )

    print(f"test episode count: {test_episode_ids.numel()}")
    print(f"selected episode count: {selected_episode_ids.numel()}")
    print(f"first selected episodes: {selected_episode_ids[:10].tolist()}")
    print(
        "selected lengths:",
        [int(raw_dataset.lengths[int(ep)]) for ep in selected_episode_ids[:10].tolist()],
    )

    windows = build_rollout_windows(
        raw_dataset,
        selected_episode_ids,
        history_size=args.history_size,
        frameskip=args.frameskip,
        raw_horizon=args.raw_horizon,
    )
    print_tensor_summary("context_pixels", windows["context_pixels"])
    print_tensor_summary("action_tokens", windows["action_tokens"])
    print_tensor_summary("target_states", windows["target_states"])
    print(f"context_steps: {windows['context_steps'].tolist()}")
    print(f"target_steps: {windows['target_steps'].tolist()}")
    print(f"action_starts: {windows['action_starts'].tolist()}")
    print(f"action_ends: {windows['action_ends'].tolist()}")

    model = load_released_lewm_world_model(
        policy_name=args.policy_name,
        device=device,
        cache_dir=cache_dir,
    )
    image_transform = get_image_transform(args.img_size)
    encoded = encode_rollout_windows(
        windows,
        model=model,
        device=device,
        image_transform=image_transform,
        batch_size=args.encode_batch_size,
        history_size=args.history_size,
        frameskip=args.frameskip,
        raw_horizon=args.raw_horizon,
    )
    print_tensor_summary("context_emb", encoded["context_emb"])
    print_tensor_summary("action_emb", encoded["action_emb"])

    metadata = build_metadata(args, cache_dir, device, selected_episode_ids)
    encoded_path = save_rollout_encoded_dataset(encoded, metadata, paths["encoded"])
    print(f"saved encoded rollout dataset: {encoded_path}")

    predicted_latents = open_loop_rollout_from_embeddings(
        encoded,
        model=model,
        device=device,
        batch_size=args.rollout_batch_size,
        history_size=args.history_size,
    )
    print_tensor_summary("predicted_latents", predicted_latents)
    predicted_path = save_rollout_predictions(
        encoded,
        predicted_latents,
        metadata,
        paths["predicted"],
    )
    print(f"saved predicted rollout dataset: {predicted_path}")

    reloaded = torch.load(predicted_path, map_location="cpu")
    predicted = reloaded["predicted"]
    print("reloaded predicted artifact verification:")
    print_tensor_summary("reloaded predicted_latents", predicted["predicted_latents"])
    print_tensor_summary("reloaded target_states", predicted["target_states"])
    print(f"reloaded target_steps: {predicted['target_steps'].tolist()}")


if __name__ == "__main__":
    main()
