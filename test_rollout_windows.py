from pathlib import Path
import unittest

import torch

from probe.rollout_windows import (
    assert_selected_episodes_from_test_cache,
    build_rollout_window,
    build_rollout_windows,
    load_encoded_test_episode_ids,
    load_raw_pusht_dataset,
    model_step_indices,
    select_rollout_episode_ids,
)


def make_fake_episode(length=101):
    return {
        "pixels": torch.arange(length * 3 * 2 * 2, dtype=torch.uint8).reshape(
            length, 3, 2, 2
        ),
        "action": torch.arange(length * 2, dtype=torch.float32).reshape(length, 2),
        "state": torch.arange(length * 7, dtype=torch.float32).reshape(length, 7),
        "episode_idx": torch.zeros(length, dtype=torch.long),
        "step_idx": torch.arange(length, dtype=torch.long),
    }


def test_model_step_indices_for_raw_horizon_100():
    indices = model_step_indices(history_size=3, frameskip=5, raw_horizon=100)

    assert indices["context_steps"].tolist() == [0, 5, 10]
    assert indices["target_steps"].tolist() == list(range(15, 101, 5))
    assert indices["action_starts"].tolist() == list(range(0, 100, 5))


def test_build_rollout_window_aligns_context_actions_and_targets():
    raw_episode = make_fake_episode(length=101)

    window = build_rollout_window(
        raw_episode,
        episode_id=7,
        history_size=3,
        frameskip=5,
        raw_horizon=100,
    )

    assert window["episode_id"] == 7
    assert window["context_pixels"].shape == (3, 3, 2, 2)
    assert window["action_tokens"].shape == (20, 10)
    assert window["target_states"].shape == (18, 7)
    assert window["alignment"]["context_steps"].tolist() == [0, 5, 10]
    assert window["alignment"]["target_steps"].tolist() == list(range(15, 101, 5))

    assert torch.equal(window["context_pixels"][0], raw_episode["pixels"][0])
    assert torch.equal(window["context_pixels"][1], raw_episode["pixels"][5])
    assert torch.equal(window["target_states"][0], raw_episode["state"][15])
    assert torch.equal(window["target_states"][-1], raw_episode["state"][100])

    assert torch.equal(window["action_tokens"][0], raw_episode["action"][0:5].reshape(10))
    assert torch.equal(window["action_tokens"][2], raw_episode["action"][10:15].reshape(10))
    assert torch.equal(window["action_tokens"][19], raw_episode["action"][95:100].reshape(10))


def test_select_rollout_episode_ids_requires_test_cache_subset():
    test_episode_ids = torch.tensor([4, 8, 12])

    assert_selected_episodes_from_test_cache(torch.tensor([4, 12]), test_episode_ids)
    with unittest.TestCase().assertRaisesRegex(
        AssertionError,
        "not all from the encoded test cache",
    ):
        assert_selected_episodes_from_test_cache(torch.tensor([4, 99]), test_episode_ids)


def test_real_encoded_cache_and_raw_dataset_smoke():
    cache_dir = Path(".stable-wm")
    encoded_cache_path = (
        cache_dir
        / "probes"
        / "encoded"
        / "pusht_expert_train_pusht_lewm_test_seed42_img224_encoded.pt"
    )
    dataset_path = cache_dir / "pusht_expert_train.h5"
    if not encoded_cache_path.exists() or not dataset_path.exists():
        raise unittest.SkipTest("PushT data/cache files are not available locally")

    test_episode_ids = load_encoded_test_episode_ids(encoded_cache_path)
    raw_dataset = load_raw_pusht_dataset(cache_dir=cache_dir)
    selected_episode_ids = select_rollout_episode_ids(
        test_episode_ids,
        raw_dataset,
        max_episodes=3,
        min_raw_length=101,
    )
    windows = build_rollout_windows(
        raw_dataset,
        selected_episode_ids,
        history_size=3,
        frameskip=5,
        raw_horizon=100,
    )

    assert set(selected_episode_ids.tolist()).issubset(set(test_episode_ids.tolist()))
    assert windows["context_pixels"].shape == (3, 3, 3, 224, 224)
    assert windows["action_tokens"].shape == (3, 20, 10)
    assert windows["target_states"].shape == (3, 18, 7)
    assert windows["context_steps"].tolist() == [0, 5, 10]
    assert windows["target_steps"].tolist() == list(range(15, 101, 5))
    assert windows["action_starts"].tolist() == list(range(0, 100, 5))


if __name__ == "__main__":
    test_model_step_indices_for_raw_horizon_100()
    test_build_rollout_window_aligns_context_actions_and_targets()
    test_select_rollout_episode_ids_requires_test_cache_subset()
    try:
        test_real_encoded_cache_and_raw_dataset_smoke()
    except unittest.SkipTest as error:
        print(f"skipped real-data smoke test: {error}")
    print("rollout window tests passed")
