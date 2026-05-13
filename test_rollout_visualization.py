import torch

from probe.rollout_visualization import (
    find_nearest_encoded_frames,
    horizon_index_from_step,
    sample_example_indices,
)


def test_horizon_index_from_step_finds_requested_horizon():
    target_steps = torch.tensor([15, 20, 25, 30, 35, 40])

    assert horizon_index_from_step(target_steps, 35) == 4


def test_sample_example_indices_is_reproducible():
    first = sample_example_indices(num_examples=100, num_samples=20, seed=123)
    second = sample_example_indices(num_examples=100, num_samples=20, seed=123)

    assert first.numel() == 20
    assert torch.equal(first, second)
    assert len(set(first.tolist())) == 20


def test_find_nearest_encoded_frames_returns_expected_rows():
    encoded_test = {
        "emb": torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [10.0, 10.0],
            ],
            dtype=torch.float32,
        ),
        "episode_idx": torch.tensor([5, 6, 7], dtype=torch.long),
        "step_idx": torch.tensor([15, 20, 35], dtype=torch.long),
    }
    query_latents = torch.tensor([[0.1, 0.2], [9.0, 9.0]], dtype=torch.float32)

    nearest = find_nearest_encoded_frames(query_latents, encoded_test, chunk_size=2)

    assert nearest["nearest_index"].tolist() == [0, 2]
    assert nearest["nearest_episode_idx"].tolist() == [5, 7]
    assert nearest["nearest_step_idx"].tolist() == [15, 35]


if __name__ == "__main__":
    test_horizon_index_from_step_finds_requested_horizon()
    test_sample_example_indices_is_reproducible()
    test_find_nearest_encoded_frames_returns_expected_rows()
    print("rollout visualization tests passed")
