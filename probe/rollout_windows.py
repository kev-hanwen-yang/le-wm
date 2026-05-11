from pathlib import Path

import stable_worldmodel as swm
import torch


# Component: raw rollout-window construction for the horizon ablation.
#
# The Table 1 encoded cache stores one row per frame with `emb`, `state`,
# `proprio`, `episode_idx`, and `step_idx`, but it intentionally does not store
# raw `pixels` or raw 2D `action`. Long-horizon rollout needs those raw fields:
# - context pixels: observed frames f0, f5, f10 -> z0, z5, z10
# - action tokens: 5 raw 2D actions flattened into one LeWM action token (10,)
# - target states: s15, s20, ..., s100 for horizon-wise probe evaluation
#
# This module therefore loads full raw Push-T episodes with `frameskip=1` and
# manually builds LeWM model-step tensors. Manual grouping makes the temporal
# alignment explicit and avoids asking HDF5Dataset(frameskip=5, num_steps=21) to
# load an extra unused action token after the raw horizon.


ROLLOUT_KEYS_TO_LOAD = [
    "pixels",
    "action",
    "state",
    "episode_idx",
    "step_idx",
]

ROLLOUT_KEYS_TO_CACHE = [
    "action",
    "state",
    "episode_idx",
    "step_idx",
]


def load_encoded_test_episode_ids(encoded_cache_path):
    payload = torch.load(encoded_cache_path, map_location="cpu")
    if "encoded" not in payload:
        raise KeyError(f"Expected key 'encoded' in {encoded_cache_path}")

    encoded = payload["encoded"]
    if "episode_idx" not in encoded:
        raise KeyError(f"Expected key 'episode_idx' in encoded cache {encoded_cache_path}")

    return unique_preserve_order(encoded["episode_idx"].long().cpu())


def unique_preserve_order(values):
    unique_values = []
    seen = set()
    for value in values.reshape(-1).tolist():
        value = int(value)
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return torch.tensor(unique_values, dtype=torch.long)


def assert_selected_episodes_from_test_cache(selected_episode_ids, test_episode_ids):
    test_ids = {int(episode_id) for episode_id in test_episode_ids.reshape(-1).tolist()}
    selected_ids = [int(episode_id) for episode_id in selected_episode_ids]
    missing_ids = [episode_id for episode_id in selected_ids if episode_id not in test_ids]
    if missing_ids:
        raise AssertionError(
            "Selected rollout episodes are not all from the encoded test cache. "
            f"Missing episode IDs: {missing_ids[:10]}"
        )


def load_raw_pusht_dataset(
    dataset_name="pusht_expert_train",
    cache_dir=None,
):
    dataset_path = Path(cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name,
        frameskip=1,  # Load raw consecutive timesteps; grouping is done below.
        num_steps=1,
        keys_to_load=ROLLOUT_KEYS_TO_LOAD,
        keys_to_cache=ROLLOUT_KEYS_TO_CACHE,
        cache_dir=dataset_path,
    )


def select_rollout_episode_ids(
    test_episode_ids,
    raw_dataset,
    max_episodes, # e.g. sample 100 episodes
    min_raw_length, # e.g. at least 101 raw timesteps
):
    selected_episode_ids = []
    for episode_id in test_episode_ids.reshape(-1).tolist():
        episode_id = int(episode_id)
        if int(raw_dataset.lengths[episode_id]) < min_raw_length:
            continue
        selected_episode_ids.append(episode_id)
        if len(selected_episode_ids) == max_episodes:
            break

    selected_episode_ids = torch.tensor(selected_episode_ids, dtype=torch.long)
    assert_selected_episodes_from_test_cache(selected_episode_ids, test_episode_ids)
    return selected_episode_ids


def model_step_indices(history_size=3, frameskip=5, raw_horizon=100):
    if raw_horizon % frameskip != 0:
        raise ValueError(
            f"raw_horizon={raw_horizon} must be divisible by frameskip={frameskip}"
        )

    context_steps = torch.arange(0, history_size * frameskip, frameskip)
    target_steps = torch.arange(history_size * frameskip, raw_horizon + 1, frameskip)
    action_starts = torch.arange(0, raw_horizon, frameskip)
    return {
        "context_steps": context_steps.long(),
        "target_steps": target_steps.long(),
        "action_starts": action_starts.long(),
    }


def build_action_tokens(raw_action, action_starts, frameskip=5):
    action_tokens = []
    for start in action_starts.tolist():
        action_block = raw_action[start : start + frameskip]
        if action_block.size(0) != frameskip:
            raise ValueError(
                f"Action block starting at {start} has length {action_block.size(0)}, "
                f"expected frameskip={frameskip}"
            )
        action_tokens.append(action_block.reshape(-1))
    return torch.stack(action_tokens, dim=0).float()


def build_rollout_window(
    raw_episode,
    episode_id,
    history_size=3,
    frameskip=5,
    raw_horizon=100,
):
    indices = model_step_indices(
        history_size=history_size,
        frameskip=frameskip,
        raw_horizon=raw_horizon,
    )
    context_steps = indices["context_steps"]
    target_steps = indices["target_steps"]
    action_starts = indices["action_starts"]

    episode_length = raw_episode["state"].size(0)
    required_length = raw_horizon + 1
    if episode_length < required_length:
        raise ValueError(
            f"Episode {episode_id} length={episode_length}, expected at least "
            f"{required_length} raw timesteps for raw_horizon={raw_horizon}"
        )

    context_pixels = raw_episode["pixels"][context_steps].contiguous()
    action_tokens = build_action_tokens(
        raw_episode["action"].float(),
        action_starts,
        frameskip=frameskip,
    )
    target_states = raw_episode["state"][target_steps].float().contiguous()

    step_idx = raw_episode.get("step_idx", torch.arange(episode_length))
    alignment = {
        "context_steps": step_idx[context_steps].long().clone(),
        "target_steps": step_idx[target_steps].long().clone(),
        "action_starts": action_starts.clone(),
        "action_ends": (action_starts + frameskip).long(),
    }

    assert_rollout_window_alignment(
        raw_episode=raw_episode,
        window={
            "context_pixels": context_pixels,
            "action_tokens": action_tokens,
            "target_states": target_states,
            "alignment": alignment,
        },
        history_size=history_size,
        frameskip=frameskip,
        raw_horizon=raw_horizon,
    )

    return {
        "episode_id": int(episode_id),
        "context_pixels": context_pixels,
        "action_tokens": action_tokens,
        "target_states": target_states,
        "alignment": alignment,
    }


def assert_rollout_window_alignment(
    raw_episode,
    window,
    history_size=3,
    frameskip=5,
    raw_horizon=100,
):
    indices = model_step_indices(
        history_size=history_size,
        frameskip=frameskip,
        raw_horizon=raw_horizon,
    )
    context_steps = indices["context_steps"]
    target_steps = indices["target_steps"]
    action_starts = indices["action_starts"]

    expected_context_steps = context_steps
    expected_target_steps = target_steps
    if not torch.equal(window["alignment"]["context_steps"].cpu(), expected_context_steps):
        raise AssertionError(
            f"Expected context steps {expected_context_steps.tolist()}, got "
            f"{window['alignment']['context_steps'].tolist()}"
        )
    if not torch.equal(window["alignment"]["target_steps"].cpu(), expected_target_steps):
        raise AssertionError(
            f"Expected target steps {expected_target_steps.tolist()}, got "
            f"{window['alignment']['target_steps'].tolist()}"
        )

    if not torch.equal(window["context_pixels"], raw_episode["pixels"][context_steps]):
        raise AssertionError("context_pixels do not match raw episode frames")
    if not torch.equal(window["target_states"], raw_episode["state"][target_steps].float()):
        raise AssertionError("target_states do not match raw episode states")

    for token_idx, start in enumerate(action_starts.tolist()):
        expected_action_token = raw_episode["action"][start : start + frameskip].float().reshape(-1)
        if not torch.equal(window["action_tokens"][token_idx], expected_action_token):
            raise AssertionError(
                f"action_tokens[{token_idx}] does not match raw_action[{start}:{start + frameskip}]"
            )


def build_rollout_windows(
    raw_dataset,
    selected_episode_ids,
    history_size=3,
    frameskip=5,
    raw_horizon=100,
):
    windows = [
        build_rollout_window(
            raw_dataset.load_episode(int(episode_id)),
            episode_id=int(episode_id),
            history_size=history_size,
            frameskip=frameskip,
            raw_horizon=raw_horizon,
        )
        for episode_id in selected_episode_ids.reshape(-1).tolist()
    ]

    return {
        "episode_idx": torch.tensor([window["episode_id"] for window in windows], dtype=torch.long),
        "context_pixels": torch.stack([window["context_pixels"] for window in windows], dim=0),
        "action_tokens": torch.stack([window["action_tokens"] for window in windows], dim=0),
        "target_states": torch.stack([window["target_states"] for window in windows], dim=0),
        "context_steps": windows[0]["alignment"]["context_steps"],
        "target_steps": windows[0]["alignment"]["target_steps"],
        "action_starts": windows[0]["alignment"]["action_starts"],
        "action_ends": windows[0]["alignment"]["action_ends"],
    }
