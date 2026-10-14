"""Direct-execution tests for Step 2 gate/provenance/artifact bookkeeping."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from omegaconf import OmegaConf

from counterfactual.baseline import file_metadata, package_versions, sha256_file
from counterfactual.contracts import ActionNormalizer, TimeActionContract
from counterfactual.simulator import SimulatorValidationError
from counterfactual.simulator_validation import (
    audit_dataset_rendering, run_simulator_stage, validate_baseline_parent,
    validation_cases, write_replay,
)
from scripts.run_phase1 import parse_args


def config():
    return OmegaConf.load(Path(__file__).parent / "config/phase1/pusht.yaml")


def parent_fixture(root):
    cfg = config()
    cache = root / ".stable-wm"
    checkpoint = cache / "pusht/lewm_object.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic-checkpoint-identity-fixture")
    dataset = cache / "pusht_expert_train.h5"
    dataset.write_bytes(b"synthetic-dataset-metadata-fixture")
    frozen = root / "frozen_source.py"
    frozen.write_text("# synthetic identity fixture\n")
    cfg.baseline.checkpoint_sha256 = sha256_file(checkpoint)
    cfg_path = root / "config.yaml"
    OmegaConf.save(cfg, cfg_path)
    parent = dict(stage="phase1_step1_frozen_baseline", run_status="completed", step1_gate_passed=True,
                  contract=TimeActionContract().as_dict(), checkpoint={"sha256": sha256_file(checkpoint)},
                  config=OmegaConf.to_container(cfg), config_file={"path": str(cfg_path)},
                  source_sha256={str(frozen): sha256_file(frozen)}, versions=package_versions(),
                  dataset=file_metadata(dataset), run_id="synthetic-parent",
                  action_normalizer={"mean": [0., 0.], "std": [1., 1.]})
    path = root / "parent.json"
    path.write_text(json.dumps(parent))
    return cfg, path, parent, frozen


def test_parent_gate_rejects_partial_baseline_checkpoint_source_and_version_changes():
    for corruption in ("partial", "checkpoint", "source", "version", "contract", "dataset", "config"):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, path, parent, frozen = parent_fixture(root)
            validate_baseline_parent(path, root, cfg)
            if corruption == "partial":
                parent["step1_gate_passed"] = False
            elif corruption == "checkpoint":
                parent["checkpoint"]["sha256"] = "different"
            elif corruption == "source":
                frozen.write_text("# modified synthetic fixture\n")
            elif corruption == "version":
                parent["versions"]["pymunk"] = "different"
            elif corruption == "contract":
                parent["contract"]["history_size"] = 1
            elif corruption == "dataset":
                parent["dataset"]["size_bytes"] += 1
            else:
                cfg.experiment.seed += 1
            path.write_text(json.dumps(parent))
            with unittest.TestCase().assertRaises(SimulatorValidationError):
                validate_baseline_parent(path, root, cfg)


def test_missing_render_metadata_stays_unresolved_and_samples_are_saved():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        dataset = root / "source.h5"
        with h5py.File(dataset, "x") as handle:
            handle.create_dataset("ep_offset", data=[0, 2])
            handle.create_dataset("pixels", data=np.full((4, 2, 2, 3), 144, dtype=np.uint8))
            handle.create_dataset("state", data=np.zeros((4, 7)))
        report = audit_dataset_rendering(dataset, root / "audit.h5")
        assert report["status"] == "unresolved"
        assert not report["verified_training_renderer"]
        assert "goal_state" in report["missing_metadata"]
        assert len(report["samples"]) == 2
        with h5py.File(root / "audit.h5", "r") as handle:
            assert handle["episode_0/pixels"].shape == (2, 2, 3)


def test_hdf5_serialization_preserves_arrays_and_declares_incomplete_snapshot():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "trace.h5"
        with h5py.File(path, "x") as handle:
            write_replay(handle.create_group("fixture"), dict(
                body=np.zeros((3, 18)), contact_events=[{"pair": "agent-block"}],
                rng_hash=np.array(["a", "b", "c"]), scene={"seed": 42},
            ))
        with h5py.File(path, "r") as handle:
            assert handle["fixture/body"].shape == (3, 18)
            assert not handle["fixture"].attrs["readback_is_complete_snapshot"]
            assert json.loads(handle["fixture/contact_events"].asstr()[()])[0]["pair"] == "agent-block"
            assert json.loads(handle["fixture/rng_hash"].asstr()[()]) == ["a", "b", "c"]


def test_cli_preserves_early_failure_and_refuses_run_reuse_or_bad_flags():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        cfg = config()
        cfg.simulator.baseline_manifest = "missing.json"
        OmegaConf.save(cfg, root / "config.yaml")
        args = parse_args(["--stage", "simulator", "--config", "config.yaml", "--run-id", "failed"])
        with patch("counterfactual.simulator_validation.git_metadata", return_value={"revision": "synthetic"}):
            assert run_simulator_stage(args, root) == 1
            path = root / "diagnostic_results/phase1/failed/simulator_manifest.json"
            report = json.loads(path.read_text())
            assert report["failure"]["type"] == "FileNotFoundError"
            assert not report["step2_gate_passed"] and not report["encoder_dataset_ready"]
            previous = path.read_bytes()
            with unittest.TestCase().assertRaises(FileExistsError):
                run_simulator_stage(args, root)
            assert path.read_bytes() == previous
            args.run_id = "../bad"
            with unittest.TestCase().assertRaises(ValueError):
                run_simulator_stage(args, root)
            args.run_id = "flags"
            args.skip_cem = True
            with unittest.TestCase().assertRaises(ValueError):
                run_simulator_stage(args, root)
            assert not (root / "diagnostic_results/phase1/flags").exists()


def test_two_case_stage_integration_with_explicitly_synthetic_parent():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        cfg = config()
        cfg.simulator.seeds = [42]
        OmegaConf.save(cfg, root / "config.yaml")
        # Source fixtures and parent are bookkeeping stubs; actual Push-T physics
        # runs unchanged. This is not a scientific baseline or power experiment.
        for name in ("test_counterfactual_simulator.py", "test_counterfactual_simulator_stage.py"):
            (root / name).write_text("# synthetic source-inventory fixture\n")
        dataset = root / ".stable-wm/pusht_expert_train.h5"
        dataset.parent.mkdir()
        with h5py.File(dataset, "x") as handle:
            handle.create_dataset("ep_offset", data=[0])
            handle.create_dataset("pixels", data=np.zeros((1, 2, 2, 3), dtype=np.uint8))
            handle.create_dataset("state", data=np.zeros((1, 7)))
        cases = [case for case in validation_cases(42) if case[2] in ("free_motion", "agent_block")]
        checkpoint = root / ".stable-wm/pusht/lewm_object.ckpt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"synthetic-checkpoint")
        parent = {"path": str(root / "synthetic_parent.json"), "sha256": None,
                  "checkpoint_sha256": sha256_file(checkpoint), "dataset_metadata": file_metadata(dataset)}
        Path(parent["path"]).write_text("{}")
        parent["sha256"] = sha256_file(parent["path"])
        normalizer = ActionNormalizer(mean=np.zeros(2), std=np.ones(2))
        args = parse_args(["--stage", "simulator", "--config", "config.yaml", "--run-id", "integration"])
        with patch("counterfactual.simulator_validation.git_metadata", return_value={"revision": "synthetic"}), \
             patch("counterfactual.simulator_validation.validate_baseline_parent", return_value=(parent, normalizer)), \
             patch("counterfactual.simulator_validation.source_metadata", return_value={}), \
             patch("counterfactual.simulator_validation.validation_cases", return_value=cases):
            assert run_simulator_stage(args, root) == 0
        report = json.loads((root / "diagnostic_results/phase1/integration/simulator_manifest.json").read_text())
        assert report["step2_gate_passed"] and not report["encoder_dataset_ready"]
        assert len(report["cases"]) == 2 and len(report["unsafe_restore_audit"]) == 2
        with h5py.File(root / ".stable-wm/phase1/integration/simulator_replay.h5", "r") as handle:
            assert handle[f"{cases[0][0].scene_id}/first/standardized_tokens"].shape == (7, 10)


if __name__ == "__main__":
    tests = [function for name, function in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}", flush=True)
    print(f"counterfactual simulator stage tests passed ({len(tests)} tests)")
