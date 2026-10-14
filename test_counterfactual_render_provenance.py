"""Direct-execution, offline Step 2 render/provenance regression tests."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from counterfactual.render_provenance import (
    RENDER_CANDIDATE, SOURCE_URLS, image_metrics, initial_families,
    preprocessing_parity, render_display_only, require_encoder_dataset_ready,
    select_families, selected_frames, snapshot_public_sources, validate_candidate,
    validate_render_boundary, validate_release_archive,
)
from counterfactual.baseline import sha256_file
from counterfactual.simulator import (
    SceneSpec, SimulatorSettings, SimulatorValidationError, body_vector,
    create_fresh_scene, physical_readback, canonical_json,
)
from scripts.run_phase1 import parse_args


def fixture():
    return SceneSpec("render_audit_fixture", 20261014, (120., 120.), (256., 230.), 0.)


def test_candidate_is_explicit_and_incompatible_settings_fail_closed():
    validate_candidate(RENDER_CANDIDATE)
    for key, value in (("with_target", False), ("with_target", 1), ("resolution", 128),
                       ("goal_pose_world_px_rad", [256., 256., 0.]), ("render_action", True),
                       ("render_goal", False), ("candidate_only", False)):
        bad = deepcopy(RENDER_CANDIDATE); bad[key] = value
        with unittest.TestCase().assertRaises(SimulatorValidationError):
            validate_candidate(bad)
    env, _ = create_fresh_scene(fixture(), SimulatorSettings())
    try:
        validate_render_boundary(env)
        env.render_action = True
        with unittest.TestCase().assertRaises(SimulatorValidationError):
            validate_render_boundary(env)
    finally:
        env.close()


def test_goal_visibility_pose_determinism_and_original_goal_image_are_preserved():
    env, _ = create_fresh_scene(fixture(), SimulatorSettings())
    try:
        before = canonical_json(physical_readback(env))
        goal_image = env._goal.copy()
        pose = RENDER_CANDIDATE["goal_pose_world_px_rad"]
        shown = render_display_only(env, visible=True, goal_pose=pose)
        repeat = render_display_only(env, visible=True, goal_pose=pose)
        hidden = render_display_only(env, visible=False, goal_pose=pose)
        shifted = render_display_only(env, visible=True, goal_pose=[304., 224., pose[2]])
        rotated = render_display_only(env, visible=True, goal_pose=[256., 256., np.pi / 2])
        assert shown.shape == (224, 224, 3) and shown.dtype == np.uint8
        assert np.array_equal(shown, repeat)
        assert all(not np.array_equal(shown, image) for image in (hidden, shifted, rotated))
        assert canonical_json(physical_readback(env)) == before
        assert np.array_equal(env._goal, goal_image)
        assert env.physics_step_index == 2
    finally:
        env.close()


def test_render_controls_restore_even_when_renderer_raises():
    env, _ = create_fresh_scene(fixture(), SimulatorSettings())
    try:
        before = canonical_json(physical_readback(env))
        with patch.object(env, "render", side_effect=RuntimeError("synthetic render failure")):
            with unittest.TestCase().assertRaises(RuntimeError):
                render_display_only(env, visible=False, goal_pose=[256., 256., 0.])
        assert canonical_json(physical_readback(env)) == before
        assert env.with_target is True
    finally:
        env.close()


def test_render_audit_does_not_change_future_physics_rewards_or_official_outputs():
    audited, _ = create_fresh_scene(fixture(), SimulatorSettings())
    reference, _ = create_fresh_scene(fixture(), SimulatorSettings())
    try:
        render_display_only(audited, visible=False, goal_pose=[256., 256., 0.])
        render_display_only(audited, visible=True, goal_pose=RENDER_CANDIDATE["goal_pose_world_px_rad"])
        for command in np.tile(np.array([.2, .1], dtype=np.float32), (35, 1)):
            result_a, result_b = audited.step(command.copy()), reference.step(command.copy())
            assert np.array_equal(body_vector(audited), body_vector(reference))
            assert result_a[1:4] == result_b[1:4]
            assert canonical_json(result_a[0]) == canonical_json(result_b[0])
            assert canonical_json(result_a[4]) == canonical_json(result_b[4])
        assert np.array_equal(audited.physics_states, reference.physics_states)
    finally:
        audited.close(); reference.close()


def test_preprocessing_matches_training_eval_step1_and_independent_imagenet_formula():
    pixels = np.random.default_rng(42).integers(0, 256, (2, 224, 224, 3), dtype=np.uint8)
    report = preprocessing_parity(pixels)
    assert report["status"] == "PASS" and report["shape"] == [1, 2, 3, 224, 224]
    assert all(value == 0 for value in report["max_abs_deltas"].values())
    for bad in (pixels.astype(float), np.zeros((2, 128, 128, 3), dtype=np.uint8)):
        with unittest.TestCase().assertRaises((ValueError, SimulatorValidationError)):
            preprocessing_parity(bad)


def test_image_metrics_use_raw_channels_and_explicit_empty_roi_exclusions():
    source = np.zeros((2, 2, 3), dtype=np.uint8)
    target = np.full_like(source, 10)
    report = image_metrics(source, target, np.zeros((2, 2), dtype=bool))
    assert report["pixel_rmse"] == report["pixel_mae"] == 10.
    assert report["goal_roi_rmse"] is None
    assert report["nongoal_roi_rmse"] == 10.
    assert report["goal_mask_iou"] is None  # no silent epsilon / pseudo-overlap


def test_sampling_groups_repeated_initial_poses_without_image_based_selection():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "synthetic.h5"
        with h5py.File(path, "x") as handle:
            handle.create_dataset("ep_offset", data=[0, 3, 6, 9])
            handle.create_dataset("ep_len", data=[3, 3, 3, 3])
            states = np.zeros((12, 7), dtype=np.float32)
            states[6:, 4] = 1.
            handle.create_dataset("state", data=states)
            handle.create_dataset("pixels", data=np.zeros((12, 224, 224, 3), dtype=np.uint8))
            handle.create_dataset("episode_idx", data=np.repeat(np.arange(4), 3))
            handle.create_dataset("step_idx", data=np.tile(np.arange(3), 4))
            families = initial_families(handle)
            assert len(families) == 2 and families[0]["episodes"] == [0, 1]
            assert select_families(families, 2) == select_families(families, 2)
            frames = list(selected_frames(handle, families))
            assert len(frames) == 8 and len({item["raw_row"] for item in frames}) == 8
            handle["step_idx"][0] = 1
            with unittest.TestCase().assertRaises(SimulatorValidationError):
                list(selected_frames(handle, families))


def test_reused_public_source_evidence_is_hash_checked_and_never_executed():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"; source.mkdir()
        index = {}
        for name, url in SOURCE_URLS.items():
            path = source / f"{name}.txt"
            path.write_text("# synthetic source snapshot; not provenance evidence\n")
            index[name] = dict(url=url, sha256=sha256_file(path))
        (source / "index.json").write_text(json.dumps(index))
        records = snapshot_public_sources(root / "copy", source / "index.json")
        assert set(records) == set(SOURCE_URLS)
        (source / "early_converter.txt").write_text("corrupted")
        with unittest.TestCase().assertRaises(SimulatorValidationError):
            snapshot_public_sources(root / "rejected", source / "index.json")


def test_unresolved_audit_never_authorizes_step3_and_cli_is_audit_only():
    for record in ({}, {"encoder_dataset_ready": False},
                   {"encoder_dataset_ready": True, "verified_generation_contract": False}):
        with unittest.TestCase().assertRaises(SimulatorValidationError):
            require_encoder_dataset_ready(record)
    assert parse_args(["--stage", "render-provenance", "--run-id", "fixture"]).stage == "render-provenance"


def test_full_archive_identity_checks_compressed_release_and_decompressed_bytes():
    import zstandard
    with TemporaryDirectory() as directory:
        root = Path(directory)
        original = root / "source.h5"; archive = root / "source.h5.zst"
        original.write_bytes(b"synthetic byte identity fixture" * 100)
        archive.write_bytes(zstandard.ZstdCompressor().compress(original.read_bytes()))
        tree = root / "tree.json"
        tree.write_text(json.dumps([dict(path="pusht_expert_train.h5.zst", size=archive.stat().st_size,
                                        lfs={"oid": sha256_file(archive)})]))
        result = validate_release_archive(original, archive, tree)
        assert result["status"] == "PASS" and result["complete_byte_equality"]
        assert result["dataset_sha256"] == sha256_file(original)
        original.write_bytes(b"incompatible local artifact")
        with unittest.TestCase().assertRaises(SimulatorValidationError):
            validate_release_archive(original, archive, tree)
        archive.write_bytes(b"not the pinned release")
        with unittest.TestCase().assertRaises(SimulatorValidationError):
            validate_release_archive(original, archive, tree)
        assert validate_release_archive(original, root / "absent.zst", tree)["status"] == "UNAVAILABLE"


if __name__ == "__main__":
    tests = [function for name, function in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}", flush=True)
    print(f"counterfactual render provenance tests passed ({len(tests)} tests)")
