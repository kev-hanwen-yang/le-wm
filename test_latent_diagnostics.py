from pathlib import Path
import tempfile

import torch

from probe.latent_diagnostics import (
    aggregate_manifold_distance_metrics,
    build_teacher_forced_real_latents,
    compute_latent_mse_metrics,
    compute_min_distance_to_encoded_splits,
    compute_norm_trajectory_metrics,
    compute_systematic_bias_metrics,
    compute_temporal_straightness_metrics,
    compute_teacher_forced_velocity_metrics,
    latent_diagnostic_paths,
    legacy_latent_diagnostic_paths,
    load_future_pixels,
    manifold_diagnostic_paths,
    norm_trajectory_diagnostic_paths,
    predict_teacher_forced_latents,
    resolve_true_future_latents_path,
    squared_l2_distance_matrix,
    systematic_bias_diagnostic_paths,
    teacher_forced_velocity_diagnostic_paths,
    temporal_straightness_diagnostic_paths,
    validate_predicted_latent_artifact,
    verify_teacher_forced_first_step,
)


class FakeRawDataset:
    """Tiny raw dataset fixture with `load_episode`, matching HDF5Dataset API."""

    def __init__(self):
        """Create two fake episodes with pixels shaped like (T, C, H, W)."""
        self.episodes = {
            3: {"pixels": torch.arange(6 * 1 * 2 * 2, dtype=torch.uint8).reshape(6, 1, 2, 2)},
            4: {"pixels": torch.arange(100, 100 + 6 * 1 * 2 * 2, dtype=torch.uint8).reshape(6, 1, 2, 2)},
        }

    def load_episode(self, episode_id):
        """Return a fake raw episode by integer ID."""
        return self.episodes[int(episode_id)]


class FakeTeacherForcedModel:
    """Tiny predictor fixture that makes action/latent windowing observable."""

    def eval(self):
        """Match the PyTorch module API used by predict_teacher_forced_latents."""
        return self

    def predict(self, emb, act_emb):
        """Return a sequence whose last item is last latent plus last action."""
        return emb + act_emb


def test_validate_predicted_latent_artifact_accepts_aligned_shapes():
    predicted = {
        "episode_idx": torch.tensor([3, 4]),
        "predicted_latents": torch.zeros((2, 3, 5)),
        "target_steps": torch.tensor([1, 2, 3]),
    }

    validate_predicted_latent_artifact(predicted)


def test_load_future_pixels_uses_episode_ids_and_target_steps():
    raw_dataset = FakeRawDataset()

    pixels = load_future_pixels(
        raw_dataset,
        episode_ids=torch.tensor([3, 4]),
        target_steps=torch.tensor([1, 3, 5]),
    )

    assert pixels.shape == (2, 3, 1, 2, 2)
    assert torch.equal(pixels[0, 0], raw_dataset.load_episode(3)["pixels"][1])
    assert torch.equal(pixels[1, 2], raw_dataset.load_episode(4)["pixels"][5])


def test_compute_latent_mse_metrics_matches_manual_values():
    predicted = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    true = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])

    metrics = compute_latent_mse_metrics(predicted, true)

    assert torch.allclose(metrics["sq_l2_per_episode"], torch.tensor([[5.0, 13.0]]))
    assert torch.allclose(metrics["sq_l2_mean_by_horizon"], torch.tensor([5.0, 13.0]))
    assert torch.allclose(metrics["per_dim_mse_by_horizon"], torch.tensor([2.5, 6.5]))
    assert torch.allclose(
        metrics["latent_dim_mse_by_horizon"],
        torch.tensor([[1.0, 4.0], [4.0, 9.0]]),
    )


def test_compute_systematic_bias_metrics_detects_consistent_direction():
    predicted = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 0.0]],
            [[1.0, 1.0], [0.0, 2.0]],
        ]
    )
    true = torch.zeros_like(predicted)

    metrics = compute_systematic_bias_metrics(predicted, true)

    assert torch.allclose(
        metrics["mean_error_by_horizon"],
        torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
    )
    assert torch.allclose(metrics["bias_sq_l2_by_horizon"], torch.tensor([2.0, 2.0]))
    assert torch.allclose(metrics["bias_per_dim_mse_by_horizon"], torch.tensor([1.0, 1.0]))
    assert torch.allclose(metrics["total_sq_l2_mean_by_horizon"], torch.tensor([2.0, 4.0]))
    assert torch.allclose(metrics["total_per_dim_mse_by_horizon"], torch.tensor([1.0, 2.0]))
    assert torch.allclose(metrics["bias_fraction_by_horizon"], torch.tensor([1.0, 0.5]))


def test_compute_norm_trajectory_metrics_matches_manual_values():
    predicted = torch.tensor(
        [
            [[3.0, 4.0], [6.0, 8.0]],
            [[0.0, 5.0], [0.0, 10.0]],
        ]
    )
    true = torch.tensor(
        [
            [[0.0, 5.0], [0.0, 5.0]],
            [[3.0, 4.0], [3.0, 4.0]],
        ]
    )
    reference_stats = {"overall": {"mean": 5.0, "std": 2.0}}

    metrics = compute_norm_trajectory_metrics(predicted, true, reference_stats)

    assert torch.allclose(metrics["pred_norm_per_episode"], torch.tensor([[5.0, 10.0], [5.0, 10.0]]))
    assert torch.allclose(metrics["true_norm_per_episode"], torch.tensor([[5.0, 5.0], [5.0, 5.0]]))
    assert torch.allclose(metrics["pred_norm_mean_by_horizon"], torch.tensor([5.0, 10.0]))
    assert torch.allclose(metrics["true_norm_mean_by_horizon"], torch.tensor([5.0, 5.0]))
    assert torch.allclose(metrics["norm_ratio_by_horizon"], torch.tensor([1.0, 2.0]))
    assert torch.allclose(metrics["mean_norm_ratio_by_horizon"], torch.tensor([1.0, 2.0]))
    assert torch.allclose(metrics["pred_norm_zscore_by_horizon"], torch.tensor([0.0, 2.5]))
    assert torch.allclose(metrics["true_norm_zscore_by_horizon"], torch.tensor([0.0, 0.0]))


def test_compute_temporal_straightness_metrics_matches_manual_values():
    predicted = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]],
        ]
    )
    true = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0]],
            [[0.0, 1.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]],
        ]
    )

    metrics = compute_temporal_straightness_metrics(predicted, true)

    assert torch.allclose(metrics["pred_cosine_per_episode"], torch.ones((2, 2)))
    assert torch.allclose(metrics["true_cosine_per_episode"], torch.tensor([[0.0, 1.0], [0.0, 1.0]]))
    assert torch.allclose(metrics["pred_straightness_mean_by_horizon"], torch.tensor([1.0, 1.0]))
    assert torch.allclose(metrics["true_straightness_mean_by_horizon"], torch.tensor([0.0, 1.0]))
    assert torch.allclose(metrics["pred_global_straightness_mean"], torch.tensor(1.0))
    assert torch.allclose(metrics["true_global_straightness_mean"], torch.tensor(0.5))
    assert torch.allclose(metrics["pred_velocity_norm_mean_by_horizon"], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(metrics["true_velocity_norm_mean_by_horizon"], torch.tensor([1.0, 1.0, 1.0]))


def test_build_teacher_forced_real_latents_concatenates_context_and_future():
    context = torch.tensor([[[0.0], [5.0], [10.0]]])
    future = torch.tensor([[[15.0], [20.0]]])

    real_latents = build_teacher_forced_real_latents(context, future)

    assert torch.equal(real_latents, torch.tensor([[[0.0], [5.0], [10.0], [15.0], [20.0]]]))


def test_predict_teacher_forced_latents_uses_sliding_real_windows():
    real_latents = torch.tensor([[[0.0], [5.0], [10.0], [15.0], [20.0]]])
    action_emb = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])

    pred = predict_teacher_forced_latents(
        real_latents=real_latents,
        action_emb=action_emb,
        model=FakeTeacherForcedModel(),
        device="cpu",
        batch_size=1,
        history_size=3,
    )

    assert torch.equal(pred, torch.tensor([[[13.0], [19.0]]]))


def test_compute_teacher_forced_velocity_metrics_and_first_step_sanity():
    real_latents = torch.tensor([[[0.0], [5.0], [10.0], [15.0], [20.0]]])
    teacher_forced = torch.tensor([[[13.0], [19.0]]])
    open_loop = torch.tensor([[[13.0], [18.0]]])

    metrics = compute_teacher_forced_velocity_metrics(
        real_latents=real_latents,
        teacher_forced_latents=teacher_forced,
        open_loop_latents=open_loop,
    )
    sanity = verify_teacher_forced_first_step(
        teacher_forced_latents=teacher_forced,
        open_loop_latents=open_loop,
        metrics=metrics,
    )

    assert sanity["passed"]
    assert torch.allclose(metrics["v_true_mean"], torch.tensor([5.0, 5.0]))
    assert torch.allclose(metrics["v_pred_tf_mean"], torch.tensor([3.0, 4.0]))
    assert torch.allclose(metrics["err_tf_mean"], torch.tensor([2.0, 1.0]))
    assert torch.allclose(metrics["v_pred_openloop_mean"], torch.tensor([3.0, 5.0]))
    assert torch.allclose(metrics["err_openloop_mean"], torch.tensor([2.0, 2.0]))
    assert torch.allclose(metrics["ratio_mean"], torch.tensor([0.6, 0.8]))
    assert torch.allclose(metrics["openloop_ratio_mean"], torch.tensor([0.6, 1.0]))
    assert torch.allclose(metrics["tf_per_dim_mse"], torch.tensor([4.0, 1.0]))


def test_latent_diagnostic_paths_include_num_episodes_and_horizon():
    paths = latent_diagnostic_paths(
        cache_dir=Path(".stable-wm"),
        dataset_name="pusht_expert_train",
        policy_name="pusht/lewm",
        split_seed=42,
        img_size=224,
        num_episodes=1000,
        raw_horizon=100,
    )

    assert "episodes1000_horizon100" in str(paths["json"])
    assert "mse_to_encoded_ground_truth" in str(paths["json"])
    assert paths["true_latents"].name.endswith("_true_future_latents.pt")


def test_manifold_paths_use_separate_experiment_subfolder():
    paths = manifold_diagnostic_paths(
        cache_dir=Path(".stable-wm"),
        dataset_name="pusht_expert_train",
        policy_name="pusht/lewm",
        split_seed=42,
        img_size=224,
        num_episodes=1000,
        raw_horizon=100,
        reference_splits=["train", "val"],
    )

    assert "min_distance_to_manifold" in str(paths["json"])
    assert paths["ratio_chart"].name.endswith("_manifold_drift_ratio.png")


def test_systematic_bias_paths_use_separate_experiment_subfolder():
    paths = systematic_bias_diagnostic_paths(
        cache_dir=Path(".stable-wm"),
        dataset_name="pusht_expert_train",
        policy_name="pusht/lewm",
        split_seed=42,
        img_size=224,
        num_episodes=1000,
        raw_horizon=100,
    )

    assert "systematic_bias_direction" in str(paths["json"])
    assert paths["bias_fraction_chart"].name.endswith("_bias_fraction_by_horizon.png")


def test_norm_trajectory_paths_use_separate_experiment_subfolder():
    paths = norm_trajectory_diagnostic_paths(
        cache_dir=Path(".stable-wm"),
        dataset_name="pusht_expert_train",
        policy_name="pusht/lewm",
        split_seed=42,
        img_size=224,
        num_episodes=1000,
        raw_horizon=100,
        reference_splits=["train", "val"],
    )

    assert "norm_trajectory" in str(paths["json"])
    assert paths["norm_zscore_chart"].name.endswith("_norm_zscore_by_horizon.png")


def test_temporal_straightness_paths_use_separate_experiment_subfolder():
    paths = temporal_straightness_diagnostic_paths(
        cache_dir=Path(".stable-wm"),
        dataset_name="pusht_expert_train",
        policy_name="pusht/lewm",
        split_seed=42,
        img_size=224,
        num_episodes=1000,
        raw_horizon=100,
    )

    assert "temporal_straightness" in str(paths["json"])
    assert paths["velocity_norm_chart"].name.endswith("_velocity_norm_by_horizon.png")


def test_teacher_forced_velocity_paths_use_separate_experiment_subfolder():
    paths = teacher_forced_velocity_diagnostic_paths(
        cache_dir=Path(".stable-wm"),
        dataset_name="pusht_expert_train",
        policy_name="pusht/lewm",
        split_seed=42,
        img_size=224,
        num_episodes=1000,
        raw_horizon=100,
    )

    assert "teacher_forced_velocity" in str(paths["json"])
    assert paths["json"].name == "teacher_forced_velocity_report_split_seed42_episodes1000_horizon100_img224.json"
    assert paths["ratio_chart"].name.endswith("_tf_velocity_ratio_by_horizon.png")


def test_resolve_true_future_latents_path_prefers_new_then_legacy():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        new_path = latent_diagnostic_paths(
            cache_dir=temp_dir,
            dataset_name="pusht_expert_train",
            policy_name="pusht/lewm",
            split_seed=42,
            img_size=224,
            num_episodes=1000,
            raw_horizon=100,
        )["true_latents"]
        legacy_path = legacy_latent_diagnostic_paths(
            cache_dir=temp_dir,
            dataset_name="pusht_expert_train",
            policy_name="pusht/lewm",
            split_seed=42,
            img_size=224,
            num_episodes=1000,
            raw_horizon=100,
        )["true_latents"]

        legacy_path.parent.mkdir(parents=True)
        legacy_path.write_text("legacy")
        assert resolve_true_future_latents_path(
            cache_dir=temp_dir,
            dataset_name="pusht_expert_train",
            policy_name="pusht/lewm",
            split_seed=42,
            img_size=224,
            num_episodes=1000,
            raw_horizon=100,
        ) == legacy_path

        new_path.parent.mkdir(parents=True)
        new_path.write_text("new")
        assert resolve_true_future_latents_path(
            cache_dir=temp_dir,
            dataset_name="pusht_expert_train",
            policy_name="pusht/lewm",
            split_seed=42,
            img_size=224,
            num_episodes=1000,
            raw_horizon=100,
        ) == new_path


def test_squared_l2_distance_matrix_matches_manual_values():
    query = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    reference = torch.tensor([[1.0, 0.0], [2.0, 2.0]])

    dist_sq = squared_l2_distance_matrix(query, reference)

    expected = torch.tensor([[1.0, 8.0], [1.0, 4.0]])
    assert torch.allclose(dist_sq, expected)


def test_compute_min_distance_to_encoded_splits_tracks_nearest_metadata():
    query = torch.tensor([[0.0, 0.0], [2.1, 0.0], [5.0, 5.0]])

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        split_a = temp_dir / "train_encoded.pt"
        split_b = temp_dir / "val_encoded.pt"
        torch.save(
            {
                "encoded": {
                    "emb": torch.tensor([[1.0, 0.0], [10.0, 10.0]]),
                    "episode_idx": torch.tensor([11, 12]),
                    "step_idx": torch.tensor([3, 4]),
                }
            },
            split_a,
        )
        torch.save(
            {
                "encoded": {
                    "emb": torch.tensor([[2.0, 0.0], [5.0, 4.0]]),
                    "episode_idx": torch.tensor([21, 22]),
                    "step_idx": torch.tensor([5, 6]),
                }
            },
            split_b,
        )

        nearest = compute_min_distance_to_encoded_splits(
            query_latents=query,
            reference_split_paths=[split_a, split_b],
            device="cpu",
            query_chunk_size=2,
            reference_chunk_size=1,
        )

    assert torch.allclose(nearest["min_sq_l2"], torch.tensor([1.0, 0.01, 1.0]), atol=1e-6)
    assert torch.equal(nearest["nearest_split_index"], torch.tensor([0, 1, 1]))
    assert torch.equal(nearest["nearest_episode_idx"], torch.tensor([11, 21, 22]))
    assert torch.equal(nearest["nearest_step_idx"], torch.tensor([3, 5, 6]))


def test_aggregate_manifold_distance_metrics_uses_true_baseline_ratio():
    pred_nearest = {
        "min_sq_l2": torch.tensor([[2.0, 8.0], [4.0, 16.0]]),
        "split_names": ["train"],
        "reference_counts": [10],
    }
    true_nearest = {
        "min_sq_l2": torch.tensor([[1.0, 4.0], [1.0, 4.0]]),
        "split_names": ["train"],
        "reference_counts": [10],
    }

    metrics = aggregate_manifold_distance_metrics(
        pred_nearest=pred_nearest,
        true_nearest=true_nearest,
        target_steps=torch.tensor([15, 20]),
        latent_dim=2,
    )

    assert torch.allclose(metrics["pred_min_sq_l2_mean"], torch.tensor([3.0, 12.0]))
    assert torch.allclose(metrics["true_min_sq_l2_mean"], torch.tensor([1.0, 4.0]))
    assert torch.allclose(metrics["pred_min_per_dim_mse_mean"], torch.tensor([1.5, 6.0]))
    assert torch.allclose(metrics["manifold_drift_ratio"], torch.tensor([3.0, 3.0]))
    assert metrics["rows"][0]["raw_step"] == 15


if __name__ == "__main__":
    test_validate_predicted_latent_artifact_accepts_aligned_shapes()
    test_load_future_pixels_uses_episode_ids_and_target_steps()
    test_compute_latent_mse_metrics_matches_manual_values()
    test_compute_systematic_bias_metrics_detects_consistent_direction()
    test_compute_norm_trajectory_metrics_matches_manual_values()
    test_compute_temporal_straightness_metrics_matches_manual_values()
    test_build_teacher_forced_real_latents_concatenates_context_and_future()
    test_predict_teacher_forced_latents_uses_sliding_real_windows()
    test_compute_teacher_forced_velocity_metrics_and_first_step_sanity()
    test_latent_diagnostic_paths_include_num_episodes_and_horizon()
    test_manifold_paths_use_separate_experiment_subfolder()
    test_systematic_bias_paths_use_separate_experiment_subfolder()
    test_norm_trajectory_paths_use_separate_experiment_subfolder()
    test_temporal_straightness_paths_use_separate_experiment_subfolder()
    test_teacher_forced_velocity_paths_use_separate_experiment_subfolder()
    test_resolve_true_future_latents_path_prefers_new_then_legacy()
    test_squared_l2_distance_matrix_matches_manual_values()
    test_compute_min_distance_to_encoded_splits_tracks_nearest_metadata()
    test_aggregate_manifold_distance_metrics_uses_true_baseline_ratio()
    print("latent diagnostics tests passed")
