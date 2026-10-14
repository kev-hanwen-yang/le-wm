"""Direct-execution unit tests for Phase I time/action contracts."""

import unittest

import torch

from counterfactual.contracts import ActionNormalizer, TimeActionContract


def test_phase1_time_contract_matches_specification():
    contract = TimeActionContract()
    indices = contract.rollout_indices(lookahead_steps=5)
    assert indices["context_steps"].tolist() == [0, 5, 10]
    assert indices["target_steps"].tolist() == [15, 20, 25, 30, 35]
    assert indices["action_starts"].tolist() == [0, 5, 10, 15, 20, 25, 30]
    assert indices["action_ends"].tolist() == [5, 10, 15, 20, 25, 30, 35]
    assert contract.token_dim == 10
    assert contract.as_dict()["phase1_rollout_coordinates"][0] == {
        "k": 1,
        "target_h": 15,
        "preceding_action_token": 2,
        "preceding_action_raw_steps": [10, 15],
    }


def test_actions_are_standardized_before_flattening():
    normalizer = ActionNormalizer(mean=torch.tensor([1.0, 10.0]), std=torch.tensor([2.0, 5.0]))
    raw = torch.tensor(
        [[1.0, 10.0], [3.0, 15.0], [5.0, 20.0], [7.0, 25.0], [9.0, 30.0]],
    )
    token = normalizer.tokenize(raw, frameskip=5)
    assert token.shape == (1, 10)
    assert torch.equal(token[0], torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]))


def test_batched_action_tokenization_preserves_each_example():
    normalizer = ActionNormalizer(mean=torch.zeros(2), std=torch.ones(2))
    raw = torch.arange(2 * 10 * 2, dtype=torch.float32).reshape(2, 10, 2)
    tokens = normalizer.tokenize(raw, frameskip=5)
    assert tokens.shape == (2, 2, 10)
    assert torch.equal(tokens[0, 0], raw[0, :5].reshape(-1))
    assert torch.equal(tokens[1, 1], raw[1, 5:].reshape(-1))


def test_invalid_action_length_is_rejected():
    normalizer = ActionNormalizer(mean=torch.zeros(2), std=torch.ones(2))
    with unittest.TestCase().assertRaisesRegex(ValueError, "not divisible"):
        normalizer.tokenize(torch.zeros(6, 2), frameskip=5)


if __name__ == "__main__":
    test_phase1_time_contract_matches_specification()
    test_actions_are_standardized_before_flattening()
    test_batched_action_tokenization_preserves_each_example()
    test_invalid_action_length_is_rejected()
    print("counterfactual contract tests passed")
