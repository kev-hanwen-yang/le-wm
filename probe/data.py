from pathlib import Path

import numpy as np
import stable_worldmodel as swm


# Component: dataset loading and episode splitting.
#
# This module owns the raw Push-T HDF5Dataset setup used by probing. It loads
# one frame per item (`num_steps=1`, `frameskip=1`) so each encoded observation
# can be paired with the physical state from the exact same timestep. It also
# creates train/val/test splits by whole `episode_idx` values, not by individual
# frames, to avoid leakage from near-identical neighboring frames in the same
# trajectory.


PROBE_KEYS_TO_LOAD = [
    "pixels",
    "state",
    "proprio",
    "episode_idx",
    "step_idx",
]

PROBE_KEYS_TO_CACHE = [
    "state",
    "proprio",
    "episode_idx",
    "step_idx",
]


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.get("cache_dir") or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name,
        frameskip=1,  # Keep raw frame spacing for single-frame probing labels.
        num_steps=1,  # Return one timestep: pixels/state/proprio all have T=1.
        keys_to_load=PROBE_KEYS_TO_LOAD,
        keys_to_cache=PROBE_KEYS_TO_CACHE,
        cache_dir=dataset_path,
    )


def split_by_episode(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    episode_ids = dataset.get_col_data("episode_idx")
    unique_episodes = np.unique(episode_ids)

    # Shuffle episode IDs, not frame indices, so no episode appears in multiple splits.
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
    split_episodes = {
        "train": train_episodes,
        "val": val_episodes,
        "test": test_episodes,
    }
    return splits, split_episodes
