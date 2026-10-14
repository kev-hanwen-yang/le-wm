"""Offline direct-execution tests; synthetic results are not scientific evidence."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import h5py
import numpy as np
from omegaconf import OmegaConf
import torch

from counterfactual.observation_compatibility import (
    codec_roundtrip, family_partition, family_rms_bounds, initial_match_eligible,
    natural_dose_rows, pair_metrics, readout_latents, safe_ratio, spatial_regions,
    spatial_residual, summarize,
)
from scripts.run_phase1 import parse_args


def test_partition_is_disjoint_reproducible_and_matches_plan_counts():
    families = [dict(family_id=f"family_{index:03d}") for index in range(185)]
    partition = family_partition(families)
    assert [len(partition[name]) for name in partition] == [148, 18, 19]
    assert partition == family_partition(list(reversed(families)))
    assert len(set(sum(partition.values(), []))) == 185


def test_matching_rejects_dynamic_pose_illustrations_and_incompatible_readbacks():
    record = dict(sample_kind="initial_first", source_state=[120., 120., 256., 230., 0., 0., 0.], metrics={},
                  fresh_validation=dict(quiescent=True, reset_physics_steps=2, intended_state_max_abs=0.))
    assert initial_match_eligible(record)
    for corruption in ("middle", "velocity", "drift", "extra_step", "nonquiescent"):
        bad = deepcopy(record)
        if corruption == "middle": bad["sample_kind"] = "middle"
        elif corruption == "velocity": bad["source_state"][5] = 1.
        elif corruption == "drift": bad["fresh_validation"]["intended_state_max_abs"] = .01
        elif corruption == "extra_step": bad["fresh_validation"]["reset_physics_steps"] = 3
        else: bad["fresh_validation"]["quiescent"] = False
        assert not initial_match_eligible(bad)


def test_disjoint_spatial_regions_and_exact_zero_residual_have_no_epsilon():
    render = np.full((224, 224, 3), 255, dtype=np.uint8)
    render[20:30, 20:30] = [65, 105, 225]
    render[50:70, 50:70] = [119, 136, 153]
    render[100:120, 100:120] = [144, 238, 144]
    masks = spatial_regions(render)
    assert np.all(sum(masks[name].astype(int) for name in ("agent", "block", "goal", "background")) == 1)
    zero = spatial_residual(render, render, masks)
    assert all(value["energy_share"] is None for value in zero.values())
    shifted = render.copy(); shifted[20:30, 20:30, 0] += 1
    measured = spatial_residual(shifted, render, masks)
    assert measured["agent"]["energy_share"] == 1.
    assert measured["block"]["energy_share"] == 0.


def test_lossless_and_lossy_codec_controls_preserve_input_and_record_commands():
    image = np.random.default_rng(42).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    original = image.copy(); commands = []
    with TemporaryDirectory() as directory:
        root = Path(directory)
        exact = codec_roundtrip(image, root / "exact.mkv", "ffv1", "bgr0", None, commands)
        lossy = codec_roundtrip(image, root / "lossy.mkv", "libx264", "yuv420p", 23, commands)
        assert np.array_equal(exact, image)
        assert not np.array_equal(lossy, image)
        assert np.array_equal(image, original)
        assert len(commands) == 4


def test_state_only_natural_dose_selection_preserves_block_and_logs_missing_doses():
    with TemporaryDirectory() as directory:
        with h5py.File(Path(directory) / "synthetic.h5", "x") as source:
            source.create_dataset("ep_offset", data=[0])
            source.create_dataset("ep_len", data=[4])
            states = np.array([[100, 100, 300, 300, 0, 0, 0], [110, 100, 300, 300, 0, 1, 0],
                               [120, 100, 301, 300, 0, 1, 0], [130, 100, 300, 300, 0, 1, 0]], dtype=np.float32)
            source.create_dataset("state", data=states)
            rows = natural_dose_rows(source, {"episodes": [0]}, [10., 20., 30.], 1.)
            assert [item["status"] for item in rows] == ["matched", "missing", "matched"]
            assert rows[0]["row"] == 1 and rows[2]["row"] == 3


def test_readout_unnormalization_and_wrapped_differences_are_in_physical_units():
    probe = torch.nn.Linear(192, 2)
    with torch.no_grad(): probe.weight.zero_(); probe.bias.fill_(1.)
    loaded = {"agent_location": dict(probe=probe, target_mean=torch.tensor([[100., 200.]]), target_std=torch.tensor([[2., 3.]]))}
    out = readout_latents(np.zeros((2, 192), dtype=np.float32), loaded)
    assert np.array_equal(out["agent_location"], [[102., 203.], [102., 203.]])
    z = np.zeros((2, 192), dtype=np.float32); z[1, 0] = 3.; z[1, 1] = 4.
    reads = dict(agent_location=np.array([[0., 0.], [3., 4.]]), block_location=np.zeros((2, 2)),
                 block_angle=np.array([[2 * np.pi - .01], [.01]]))
    result = pair_metrics(0, 1, z, reads)
    assert result["latent_l2"] == result["agent_drift_world_px"] == 5.
    assert abs(result["angle_drift_rad"] - .02) < 1e-10


def test_family_bootstrap_uses_families_not_frames_and_handles_undefined_ratios():
    result = family_rms_bounds([1., 4., 9.], 42, 10000)
    assert result["n_families"] == 3 and abs(result["rms"] - np.sqrt(14 / 3)) < 1e-12
    assert result == family_rms_bounds([1., 4., 9.], 42, 10000)
    assert family_rms_bounds([], 42, 10)["status"] == "undefined"
    assert safe_ratio(1., 0.) is None and safe_ratio(1., None) is None and safe_ratio(1., 2.) == .5


def test_perfect_synthetic_diagnostics_still_do_not_freeze_confirmatory_contract():
    cfg = OmegaConf.load(Path(__file__).parent / "config/phase1/pusht.yaml").simulator.compatibility
    zero = dict(latent_l2=0., agent_drift_world_px=0., block_drift_world_px=0., angle_drift_rad=0.)
    records = [dict(split=split, raw_reconstruction=zero, natural_repeat_metrics=[zero],
                    dose_metrics=[dict(status="matched", dose=10., latent_l2=1.)])
               for split in ("development", "validation")]
    with TemporaryDirectory() as directory:
        result = summarize(records, cfg, Path(directory))
        assert result["development_only_pilot_recommended"]
        assert not result["confirmatory_contract_freeze_supported"]
        assert (Path(directory) / "development_calibration.json").exists()


def test_cli_assessment_does_not_authorize_collection():
    args = parse_args(["--stage", "observation-compatibility", "--run-id", "synthetic"])
    assert args.stage == "observation-compatibility"


if __name__ == "__main__":
    tests = [function for name, function in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test(); print(f"PASS {test.__name__}", flush=True)
    print(f"counterfactual observation compatibility tests passed ({len(tests)} tests)")
