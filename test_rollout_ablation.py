import torch

from probe.rollout_ablation import open_loop_rollout_from_embeddings


class FakePredictorModel:
    """Small fake model used to verify autoregressive indexing logic."""

    def eval(self):
        """Match the tiny part of the PyTorch model API used by rollout."""
        return self

    def predict(self, emb_window, action_window):
        """Return action embeddings so the expected next latent is obvious."""
        return action_window


def test_open_loop_rollout_uses_sliding_action_windows():
    context_emb = torch.zeros((2, 3, 4), dtype=torch.float32)
    action_emb = torch.arange(2 * 20 * 4, dtype=torch.float32).reshape(2, 20, 4)
    encoded = {
        "context_emb": context_emb,
        "action_emb": action_emb,
        "action_tokens": torch.zeros((2, 20, 10), dtype=torch.float32),
        "target_states": torch.zeros((2, 18, 7), dtype=torch.float32),
        "action_ends": torch.arange(5, 101, 5, dtype=torch.long),
    }

    predicted_latents = open_loop_rollout_from_embeddings(
        encoded,
        model=FakePredictorModel(),
        device="cpu",
        batch_size=2,
        history_size=3,
    )

    assert predicted_latents.shape == (2, 18, 4)
    assert torch.equal(predicted_latents[:, 0], action_emb[:, 2])
    assert torch.equal(predicted_latents[:, 1], action_emb[:, 3])
    assert torch.equal(predicted_latents[:, -1], action_emb[:, 19])


if __name__ == "__main__":
    test_open_loop_rollout_uses_sliding_action_windows()
    print("rollout ablation tests passed")
