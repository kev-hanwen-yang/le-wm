import torch

from probe.rollout_probe_eval import (
    evaluate_probe_at_horizons,
    select_target_from_states,
    validate_rollout_prediction_shapes,
)


class IdentityProbe(torch.nn.Module):
    """Probe used in tests where normalized predictions equal input latents."""

    def forward(self, x):
        """Return the first two latent dimensions as normalized predictions."""
        return x[:, :2]


def test_select_target_from_states_uses_table1_state_slices():
    target_states = torch.arange(2 * 3 * 7, dtype=torch.float32).reshape(2, 3, 7)

    assert torch.equal(select_target_from_states(target_states, "agent_location"), target_states[..., 0:2])
    assert torch.equal(select_target_from_states(target_states, "block_location"), target_states[..., 2:4])
    assert torch.equal(select_target_from_states(target_states, "block_angle"), target_states[..., 4:5])


def test_validate_rollout_prediction_shapes_rejects_misaligned_horizon_count():
    predicted = {
        "predicted_latents": torch.zeros((4, 18, 192)),
        "target_states": torch.zeros((4, 17, 7)),
        "target_steps": torch.arange(15, 101, 5),
    }

    try:
        validate_rollout_prediction_shapes(predicted)
    except ValueError as error:
        assert "do not share" in str(error)
    else:
        raise AssertionError("Expected ValueError for mismatched horizon count")


def test_evaluate_probe_at_horizons_computes_zero_error_for_perfect_predictions():
    target_states = torch.zeros((3, 2, 7), dtype=torch.float32)
    target_states[:, 0, 0:2] = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    target_states[:, 1, 0:2] = torch.tensor([[2.0, 1.0], [4.0, 2.0], [6.0, 3.0]])
    predicted = {
        "predicted_latents": torch.zeros((3, 2, 192), dtype=torch.float32),
        "target_states": target_states,
        "target_steps": torch.tensor([15, 20], dtype=torch.long),
    }
    predicted["predicted_latents"][:, :, 0:2] = target_states[:, :, 0:2]
    loaded_probe = {
        "probe": IdentityProbe(),
        "target_mean": torch.zeros((1, 2), dtype=torch.float32),
        "target_std": torch.ones((1, 2), dtype=torch.float32),
        "probe_type": "linear",
    }

    rows = evaluate_probe_at_horizons(
        predicted,
        loaded_probe=loaded_probe,
        target_name="agent_location",
        device="cpu",
        batch_size=2,
    )

    assert len(rows) == 2
    assert [row["raw_step"] for row in rows] == [15, 20]
    assert all(row["norm_mse"] == 0.0 for row in rows)
    assert all(row["raw_mse"] == 0.0 for row in rows)
    assert all(abs(row["r_mean"] - 1.0) < 1e-6 for row in rows)


if __name__ == "__main__":
    test_select_target_from_states_uses_table1_state_slices()
    test_validate_rollout_prediction_shapes_rejects_misaligned_horizon_count()
    test_evaluate_probe_at_horizons_computes_zero_error_for_perfect_predictions()
    print("rollout probe eval tests passed")
