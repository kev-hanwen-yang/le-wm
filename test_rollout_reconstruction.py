import torch

from probe.rollout_reconstruction import (
    circular_angle_error,
    compute_physical_errors,
    make_reconstructed_state,
    select_best_reconstructions,
)


def test_circular_angle_error_wraps_around_two_pi():
    pred = torch.tensor([[0.1]])
    true = torch.tensor([[2 * torch.pi - 0.1]])

    error = circular_angle_error(pred, true)

    assert torch.allclose(error, torch.tensor([[0.2]]), atol=1e-6)


def test_select_best_reconstructions_uses_lowest_score():
    errors = {"score": torch.tensor([0.5, 0.1, 0.3])}

    selected = select_best_reconstructions(errors, num_examples=2)

    assert selected.tolist() == [1, 2]


def test_compute_physical_errors_returns_expected_units():
    decoded = {
        "agent_location": torch.tensor([[3.0, 4.0]]),
        "block_location": torch.tensor([[10.0, 10.0]]),
        "block_angle": torch.tensor([[0.0]]),
    }
    true_states = torch.tensor([[0.0, 0.0, 13.0, 14.0, torch.pi, 0.0, 0.0]])

    errors = compute_physical_errors(decoded, true_states)

    assert torch.allclose(errors["agent_l2"], torch.tensor([5.0]))
    assert torch.allclose(errors["block_l2"], torch.tensor([5.0]))
    assert torch.allclose(errors["angle_rad"], torch.tensor([torch.pi]))
    assert torch.allclose(errors["angle_deg"], torch.tensor([180.0]))


def test_make_reconstructed_state_assembles_7d_state():
    decoded = {
        "agent_location": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "block_location": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        "block_angle": torch.tensor([[0.5], [1.5]]),
    }

    state = make_reconstructed_state(decoded, torch.tensor([1]))

    assert state.shape == (1, 7)
    assert torch.equal(state[0], torch.tensor([3.0, 4.0, 7.0, 8.0, 1.5, 0.0, 0.0]))


if __name__ == "__main__":
    test_circular_angle_error_wraps_around_two_pi()
    test_select_best_reconstructions_uses_lowest_score()
    test_compute_physical_errors_returns_expected_units()
    test_make_reconstructed_state_assembles_7d_state()
    print("rollout reconstruction tests passed")
