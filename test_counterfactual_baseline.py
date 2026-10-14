"""Direct-execution tests for Phase I baseline bookkeeping helpers."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch, MagicMock

import h5py
import numpy as np
import torch

from counterfactual.baseline import (
    build_cem_smoke_command, cache_samples, check_model_determinism,
    choose_smoke_window, deterministic_sample_indices, encoded_cache_path,
    rebuild_mismatched_cache, run_cem_smoke, sha256_file, validate_encoded_cache,
)
from counterfactual.contracts import ActionNormalizer, TimeActionContract
from counterfactual.model import (
    action_normalizer_from_hdf5, frames_hwc_to_b1chw, pixels_hwc_to_btc, run_frozen_rollout,
)
from probe.embedding_cache import get_image_transform, preprocess_pixels


class TinyModel(torch.nn.Module):
    """Synthetic encoder/predictor with observable recurrent action dependence."""

    def __init__(self, latent_dim=192):
        super().__init__()
        self.latent_dim = latent_dim

    def encode(self, info):
        pixels = info["pixels"]
        values = pixels.mean(dim=(-3, -2, -1)).unsqueeze(-1)
        return {"emb": values.expand(*values.shape[:-1], self.latent_dim).contiguous()}

    def action_encoder(self, tokens):
        values = tokens.sum(dim=-1, keepdim=True)
        return values.expand(*values.shape[:-1], self.latent_dim)

    def predict(self, embeddings, actions):
        return embeddings + actions


def make_fixture(root):
    dataset_path = root / "tiny.h5"
    pixels = np.arange(40 * 2 * 2 * 3, dtype=np.uint8).reshape(40, 2, 2, 3)
    states = np.arange(40 * 7, dtype=np.float32).reshape(40, 7)
    proprios = np.arange(40 * 4, dtype=np.float32).reshape(40, 4)
    with h5py.File(dataset_path, "w") as handle:
        for key, value in {
            "ep_offset": np.arange(0, 40, 4), "ep_len": np.full(10, 4),
            "episode_idx": np.repeat(np.arange(10), 4), "step_idx": np.tile(np.arange(4), 10),
            "pixels": pixels, "state": states, "proprio": proprios,
        }.items():
            handle.create_dataset(key, data=value)
    shuffled = np.arange(10)
    np.random.default_rng(42).shuffle(shuffled)
    episode = int(shuffled[-1])
    raw_indices = np.arange(episode * 4, episode * 4 + 4)
    inputs = preprocess_pixels(frames_hwc_to_b1chw(pixels[raw_indices]), get_image_transform(224), "cpu")
    encoded = {
        "emb": TinyModel().encode({"pixels": inputs})["emb"][:, 0],
        "episode_idx": torch.full((4,), episode, dtype=torch.int64),
        "step_idx": torch.arange(4), "state": torch.from_numpy(states[raw_indices]),
        "proprio": torch.from_numpy(proprios[raw_indices]),
    }
    payload = {"metadata": {"split_name": "test", "seed": 42}, "encoded": encoded}
    cache_path = root / "encoded.pt"
    torch.save(payload, cache_path)
    return dataset_path, cache_path, payload


def validate_fixture(dataset_path, cache_path):
    return validate_encoded_cache(
        cache_path=cache_path, dataset_path=dataset_path, model=TinyModel(), device="cpu",
        image_size=224, sample_count=4, atol=1e-5, rtol=1e-4,
    )


def test_deterministic_sample_indices_cover_cache_endpoints():
    assert deterministic_sample_indices(10, 3) == [0, 4, 9]
    assert deterministic_sample_indices(2, 10) == [0, 1]


def test_encoded_cache_path_matches_existing_naming_contract():
    path = encoded_cache_path(".stable-wm", "pusht_expert_train", "pusht/lewm", "test", 42, 224)
    assert path.as_posix().endswith(
        "probes/encoded/pusht_expert_train_pusht_lewm_test_seed42_img224_encoded.pt"
    )


def test_choose_smoke_window_uses_h_0_5_10_and_actions_0_to_15():
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "tiny.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("ep_len", data=np.array([20], dtype=np.int64))
            handle.create_dataset("ep_offset", data=np.array([0], dtype=np.int64))
            handle.create_dataset("pixels", data=np.arange(20 * 2 * 2 * 3, dtype=np.uint8).reshape(20, 2, 2, 3))
            handle.create_dataset("action", data=np.arange(20 * 2, dtype=np.float32).reshape(20, 2))
        window = choose_smoke_window(path, TimeActionContract(), episode_id=0)
    assert window["episode_id"] == 0
    assert window["pixels"].shape == (3, 2, 2, 3)
    assert window["raw_actions"].shape == (15, 2)
    assert window["pixels"][1, 0, 0, 0] == 5 * 2 * 2 * 3
    assert torch.equal(torch.as_tensor(window["raw_actions"][-1]), torch.tensor([28.0, 29.0]))


def test_sequence_and_independent_frame_layouts_are_not_conflated():
    frames = np.zeros((3, 2, 2, 3), dtype=np.uint8)
    assert pixels_hwc_to_btc(frames).shape == (1, 3, 3, 2, 2)
    assert frames_hwc_to_b1chw(frames).shape == (3, 1, 3, 2, 2)


def test_real_cache_validation_and_repair_paths():
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        dataset_path, cache_path, payload = make_fixture(root)
        assert validate_fixture(dataset_path, cache_path)["status"] == "match"
        payload["encoded"]["emb"] += 1
        torch.save(payload, cache_path)
        mismatch = validate_fixture(dataset_path, cache_path)
        assert mismatch["status"] == "mismatch"
        original_hash = sha256_file(cache_path)
        repaired = rebuild_mismatched_cache(
            validation=mismatch, destination=root / "repaired.h5", dataset_path=dataset_path,
            model=TinyModel(), device="cpu", image_size=224, checkpoint_sha256="synthetic-test", batch_size=2,
        )
        assert validate_fixture(dataset_path, repaired)["status"] == "match"
        assert sha256_file(cache_path) == original_hash
        with unittest.TestCase().assertRaises(ValueError):
            rebuild_mismatched_cache(
                validation={"status": "match"}, destination=root / "unnecessary.h5",
                dataset_path=dataset_path, model=TinyModel(), device="cpu", image_size=224,
                checkpoint_sha256="synthetic-test",
            )
        assert not (root / "unnecessary.h5").exists()


def test_invalid_labels_splits_and_shapes_do_not_pass_or_rebuild():
    for corruption in ("state", "split", "shape", "nan", "duplicate"):
        with TemporaryDirectory() as temp_dir:
            dataset_path, cache_path, payload = make_fixture(Path(temp_dir))
            if corruption == "state":
                payload["encoded"]["state"][0, 0] += 1
            elif corruption == "split":
                payload["metadata"]["split_name"] = "val"
            elif corruption == "shape":
                payload["encoded"]["emb"] = payload["encoded"]["emb"][:, :1]
            elif corruption == "nan":
                payload["encoded"]["emb"][0, 0] = float("nan")
            else:
                payload["encoded"]["step_idx"][1] = 0
            torch.save(payload, cache_path)
            result = validate_fixture(dataset_path, cache_path)
            assert result["status"] == "invalid", (corruption, result)
            assert result["reason"]


def test_five_step_rollout_uses_the_correct_action_windows_and_feedback():
    contract = TimeActionContract(latent_dim=192)
    normalizer = ActionNormalizer(mean=torch.zeros(2), std=torch.ones(2))
    actions = torch.arange(70, dtype=torch.float32).reshape(35, 2)
    window = {"episode_id": 0, "pixels": np.zeros((3, 2, 2, 3), dtype=np.uint8), "raw_actions": actions.numpy()}
    output = run_frozen_rollout(TinyModel(), window["pixels"], actions, normalizer, contract, "cpu", 224)
    expected = output["context_emb"][:, -1].clone()
    tokens = normalizer.tokenize(actions, 5)
    for index in range(5):
        expected += tokens[index + 2].sum()
        assert torch.equal(output["rollout"][:, index], expected)
    with TemporaryDirectory() as temp_dir:
        report = check_model_determinism(
            model=TinyModel(), window=window, normalizer=normalizer, contract=contract,
            device="cpu", image_size=224, artifact_path=Path(temp_dir) / "repeat.h5",
        )
        assert report["deterministic"] and report["rollout_repeat_max_abs"] == 0
        assert report["rollout_target_steps"] == [15, 20, 25, 30, 35]


def test_cem_command_roundtrips_spaces_and_configured_inputs():
    from hydra.core.override_parser.overrides_parser import OverridesParser

    command, result_path = build_cem_smoke_command(
        repo_root="/tmp/repo with spaces", cache_dir="/tmp/cache with spaces",
        policy_name="pusht/lewm", seed=42, num_eval=10, run_id="test", device="cuda",
        dataset_name="pusht_expert_train", image_size=224,
    )
    overrides = OverridesParser.create().parse_overrides(command[4:])
    values = {entry.key_or_group: entry.value() for entry in overrides}
    assert values["cache_dir"] == str(Path("/tmp/cache with spaces").resolve())
    assert values["output.filename"] == str(result_path)
    assert values["policy"] == str(Path("/tmp/cache with spaces/pusht/lewm").resolve())
    assert "'cuda'" in command[2]


def test_zero_exit_without_valid_cem_metrics_is_failure():
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        process = MagicMock()
        process.wait.return_value = 0
        with patch("counterfactual.baseline.subprocess.Popen", return_value=process):
            result = run_cem_smoke(
                repo_root=root, cache_dir=root, policy_name="pusht/lewm", seed=42,
                num_eval=10, run_id="test", device="cpu", results_dir=root,
            )
        assert result["status"] == "invalid_metrics"
        assert not result["metrics_valid"]


def test_action_normalizer_matches_original_training_transform():
    from utils import get_column_normalizer
    from types import SimpleNamespace

    values = np.arange(12, dtype=np.float32).reshape(6, 2)
    values[-1] = float("nan")
    original = get_column_normalizer(SimpleNamespace(get_col_data=lambda key: values), "action", "action")
    raw = torch.from_numpy(values[:5])
    expected = original({"action": raw.clone()})["action"]
    with TemporaryDirectory() as temp_dir:
        dataset_path = Path(temp_dir) / "actions.h5"
        with h5py.File(dataset_path, "w") as handle:
            handle.create_dataset("action", data=values)
        normalizer = action_normalizer_from_hdf5(dataset_path)
    assert torch.equal(normalizer.normalize_rows(raw), expected)


def test_cli_records_early_failure_and_rejects_run_reuse():
    from scripts import run_phase1
    from omegaconf import OmegaConf

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        config = OmegaConf.load(Path(__file__).parent / "config/phase1/pusht.yaml")
        OmegaConf.save(config, root / "config.yaml")
        args = run_phase1.parse_args(["--config", "config.yaml", "--run-id", "failure", "--skip-cem"])
        with patch.object(run_phase1, "REPO_ROOT", root), patch.object(run_phase1, "git_metadata", return_value={"revision": "synthetic"}), patch.object(run_phase1, "source_metadata", return_value={}):
            assert run_phase1.run_baseline(args) == 1
            manifest_path = root / "diagnostic_results/phase1/failure/baseline_manifest.json"
            report = json.loads(manifest_path.read_text())
            assert report["failure"]["type"] == "FileNotFoundError"
            assert not report["step1_gate_passed"]
            original = manifest_path.read_bytes()
            with unittest.TestCase().assertRaises(FileExistsError):
                run_phase1.run_baseline(args)
            assert manifest_path.read_bytes() == original
            args.run_id = "../invalid"
            with unittest.TestCase().assertRaises(ValueError):
                run_phase1.run_baseline(args)


if __name__ == "__main__":
    test_deterministic_sample_indices_cover_cache_endpoints()
    test_encoded_cache_path_matches_existing_naming_contract()
    test_choose_smoke_window_uses_h_0_5_10_and_actions_0_to_15()
    test_sequence_and_independent_frame_layouts_are_not_conflated()
    test_real_cache_validation_and_repair_paths()
    test_invalid_labels_splits_and_shapes_do_not_pass_or_rebuild()
    test_five_step_rollout_uses_the_correct_action_windows_and_feedback()
    test_cem_command_roundtrips_spaces_and_configured_inputs()
    test_zero_exit_without_valid_cem_metrics_is_failure()
    test_action_normalizer_matches_original_training_transform()
    test_cli_records_early_failure_and_rejects_run_reuse()
    print("counterfactual baseline tests passed")
