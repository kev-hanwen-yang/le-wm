"""Direct-execution tests for Phase I baseline bookkeeping helpers."""

from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import torch

from counterfactual.baseline import choose_smoke_window, deterministic_sample_indices, encoded_cache_path
from counterfactual.contracts import TimeActionContract
from counterfactual.model import frames_hwc_to_b1chw, pixels_hwc_to_btc


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


if __name__ == "__main__":
    test_deterministic_sample_indices_cover_cache_endpoints()
    test_encoded_cache_path_matches_existing_naming_contract()
    test_choose_smoke_window_uses_h_0_5_10_and_actions_0_to_15()
    test_sequence_and_independent_frame_layouts_are_not_conflated()
    print("counterfactual baseline tests passed")
