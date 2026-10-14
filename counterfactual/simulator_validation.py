"""Step 2 audit suite, immutable artifacts, parent gate and failure records."""

from __future__ import annotations

from dataclasses import asdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import re
import sys
import time
import traceback

import h5py
import hdf5plugin  # Registers the source dataset's compression filters.
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import pymunk
from stable_worldmodel import spaces as swm_spaces

from counterfactual.baseline import (
    file_metadata, git_metadata, package_versions, sha256_file, source_metadata,
    write_json, write_source_snapshot,
)
from counterfactual.contracts import ActionNormalizer, TimeActionContract
from counterfactual.model import checkpoint_path
from counterfactual.simulator import (
    ANGLE_COLUMNS, BODY_FIELDS, PushT, SceneSpec, SimulatorSettings,
    SimulatorValidationError, body_vector, canonical_json, compare_replays,
    create_fresh_scene, jsonable, replay_fixed_actions, wrap_angle,
)


def validation_cases(seed: int):
    """Predetermined mechanics fixtures, not selected counterfactual data."""
    def actions(value=(0., 0.), duration=25):
        result = np.zeros((35, 2), dtype=np.float32)
        result[10:10 + duration] = value
        return result

    release = actions((.2, 0.), 10)
    release[20:25, 0] = -.2
    definitions = [
        ("stationary", (120., 120.), (256., 230.), 0., actions(), "stationary"),
        ("free_motion", (100., 100.), (350., 300.), 0., actions((.1, .05)), "free_motion"),
        ("near_contact_hold", (179., 245.), (256., 230.), 0., actions(), "stationary"),
        ("contact_push", (175., 242.), (256., 230.), 0., actions((.2, 0.), 15), "agent_block"),
        ("contact_release", (175., 242.), (256., 230.), 0., release, "post_contact"),
        ("wall_contact", (300., 245.), (385., 230.), 0., actions((.3, 0.)), "wall"),
        ("rotated_hold", (120., 120.), (256., 230.), np.pi / 4, actions(), "stationary"),
        ("angle_branch_hold", (120., 120.), (256., 230.), 2 * np.pi - .01, actions(), "stationary"),
    ]
    for name, agent, block, angle, commands, expected in definitions:
        yield SceneSpec(f"{name}_seed{seed}", seed, agent, block, angle), commands, expected


def validate_case_behavior(trace: dict, expected: str, atol: float) -> dict:
    body = trace["body"]
    history_delta = body[:11] - body[0]
    history_delta[:, list(ANGLE_COLUMNS)] = wrap_angle(history_delta[:, list(ANGLE_COLUMNS)])
    history_ok = bool(np.all(np.abs(history_delta) <= atol))
    contacts = trace["typed_contact_points"].sum(axis=0)
    agent_shift = float(np.linalg.norm(body[-1, :2] - body[0, :2]))
    block_shift = float(np.linalg.norm(body[-1, 9:11] - body[0, 9:11]))
    moving_after_push = bool(np.linalg.norm(body[21:, 12:14], axis=1).max() > atol)
    if expected == "stationary":
        expected_ok = bool(np.all(np.abs(body - body[0]) <= atol) and contacts.sum() == 0)
    elif expected == "free_motion":
        expected_ok = bool(agent_shift > 1 and block_shift <= atol and contacts.sum() == 0)
    elif expected == "agent_block":
        expected_ok = bool(contacts[0] > 0 and block_shift > 1)
    elif expected == "post_contact":
        expected_ok = bool(contacts[0] > 0 and block_shift > 1
                           and np.all(trace["typed_contact_points"][26:, 0] == 0))
    elif expected == "wall":
        expected_ok = bool(contacts[1] > 0 and block_shift > 1)
    else:
        raise SimulatorValidationError("Unknown validation fixture requirement")
    return dict(passed=history_ok and expected_ok, history_stationary=history_ok,
                expected_behavior=expected, expected_behavior_observed=expected_ok,
                agent_displacement_world_px=agent_shift, block_displacement_world_px=block_shift,
                contact_points=dict(zip(("agent_block", "block_wall", "agent_wall"), contacts.tolist())),
                moving_after_push=moving_after_push,
                history_max_abs_by_field=dict(zip(BODY_FIELDS, np.abs(history_delta).max(axis=0).tolist())))


def write_replay(group, trace, normalizer=None):
    """Persist numeric trajectories/images and copied event/RNG metadata."""
    for key, value in trace.items():
        if isinstance(value, np.ndarray) and value.dtype.kind != "U":
            options = dict(compression="gzip", compression_opts=1, shuffle=True) if value.size else {}
            group.create_dataset(key, data=value, **options)
        else:
            group.create_dataset(key, data=canonical_json(value), dtype=h5py.string_dtype("utf-8"))
    group.attrs["body_fields"] = canonical_json(BODY_FIELDS)
    group.attrs["readback_is_complete_snapshot"] = False
    if normalizer is not None:
        group.create_dataset("standardized_tokens", data=normalizer.tokenize(trace["actions"], 5).numpy())


def audit_unsafe_public_restore(trace, scene, settings, at_raw_step=20):
    """Negative diagnostic only: show why the public observation is not restore."""
    env, _ = create_fresh_scene(scene, settings)
    try:
        env.audit_phase = "unsafe_public_restore"
        public = trace["public_state"][at_raw_step].copy()
        steps_before = env.physics_step_index
        env._set_state(public)
        restored = [body_vector(env)]
        for command in trace["actions"][at_raw_step:]:
            env.step(command.copy())
            restored.append(body_vector(env))
        restored = np.stack(restored)
        reference = trace["body"][at_raw_step:]
        delta = restored - reference
        delta[:, list(ANGLE_COLUMNS)] = wrap_angle(delta[:, list(ANGLE_COLUMNS)])
        return dict(
            supported=False, source_raw_step=at_raw_step,
            set_state_extra_physics_steps=env.physics_step_index - steps_before - 10 * (len(restored) - 1),
            matches_reference=bool(np.all(np.abs(delta) <= settings.atol)),
            max_abs_by_field=dict(zip(BODY_FIELDS, np.abs(delta).max(axis=0).tolist())),
            omitted_source_block_velocity=reference[0, 12:14].tolist(),
            omitted_source_block_angular_velocity=float(reference[0, 14]),
            source_public_state=public, reference_body=reference, restored_body=restored,
            reason="7D omits block velocity/angular velocity and hidden caches; _set_state adds a physics step",
        )
    finally:
        env.close()


def audit_dataset_rendering(path: Path, artifact_path: Path, sample_count=16) -> dict:
    """Save evidence; no heuristic color match is promoted to render identity."""
    samples = []
    with h5py.File(path, "r") as source, h5py.File(artifact_path, "x") as output:
        count = len(source["ep_offset"])
        episodes = np.unique(np.linspace(0, count - 1, min(sample_count, count), dtype=np.int64))
        rows = source["ep_offset"][episodes]
        props = source["pixels"].id.get_create_plist()
        filters = [jsonable(props.get_filter(i)) for i in range(props.get_nfilters())]
        for episode, row in zip(episodes, rows):
            image = source["pixels"][int(row)]
            state = source["state"][int(row)]
            group = output.create_group(f"episode_{int(episode)}")
            group.create_dataset("pixels", data=image, compression="gzip", compression_opts=1)
            group.create_dataset("public_state", data=state)
            goal_color_pixels = int(np.all(image == [144, 238, 144], axis=-1).sum())
            samples.append(dict(episode_id=int(episode), raw_row=int(row),
                                image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                                exact_default_goal_color_pixels=goal_color_pixels))
        if len(episodes):
            Image.fromarray(source["pixels"][int(rows[0])]).save(artifact_path.parent / "training_render_sample.png")
        keys = list(source.keys())
        missing = [key for key in ("goal_state", "goal_pose", "seed", "variation_values") if key not in source]
        return dict(status="unresolved", verified_training_renderer=False, columns=keys,
                    root_attrs=jsonable(dict(source.attrs)), missing_metadata=missing,
                    pixels_shape=list(source["pixels"].shape), pixels_filters=filters, samples=samples,
                    inference="Pixel-color counts are descriptive, not proof of goal visibility or renderer identity",
                    required_before_encoder_dataset="Recover source render/goal convention or obtain explicit protocol approval; do not silently infer with_target=False")


def validate_baseline_parent(path, repo_root, cfg):
    """Require a real full baseline and unchanged frozen inference/physics sources."""
    path = Path(path).resolve()
    parent = json.loads(path.read_text())
    if parent.get("stage") != "phase1_step1_frozen_baseline" or parent.get("run_status") != "completed" or parent.get("step1_gate_passed") is not True:
        raise SimulatorValidationError("Simulator stage requires a completed, passing Step 1 manifest")
    if parent.get("contract") != TimeActionContract().as_dict():
        raise SimulatorValidationError("Parent time/action contract is incompatible")
    checkpoint = checkpoint_path(repo_root / cfg.experiment.cache_dir, cfg.experiment.policy_name)
    expected = str(cfg.baseline.checkpoint_sha256)
    if parent.get("checkpoint", {}).get("sha256") != expected or sha256_file(checkpoint) != expected:
        raise SimulatorValidationError("Parent/current checkpoint identity is incompatible")
    parent_exp = parent["config"]["experiment"]
    for name in ("policy_name", "dataset_name", "image_size"):
        if parent_exp[name] != cfg.experiment[name]:
            raise SimulatorValidationError(f"Parent experiment {name} changed")
    original_config = {key: value for key, value in parent["config"].items() if key != "simulator"}
    current_config = {key: value for key, value in OmegaConf.to_container(cfg, resolve=True).items() if key != "simulator"}
    if current_config != original_config:
        raise SimulatorValidationError("Frozen Step 1 config changed beyond the new simulator section")
    sources = parent.get("source_sha256", {})
    if not sources:
        raise SimulatorValidationError("Parent lacks source identity")
    # CLI gains one stage and config gains one section; all existing frozen
    # inference/physics code must still match. New modules are snapshotted here.
    permitted_extensions = {str((repo_root / "scripts/run_phase1.py").resolve()),
                            str(Path(parent["config_file"]["path"]).resolve())}
    checked = []
    for filename, expected_hash in sources.items():
        if filename not in permitted_extensions:
            if sha256_file(filename) != expected_hash:
                raise SimulatorValidationError(f"Frozen baseline source changed: {filename}")
            checked.append(filename)
    current_versions = package_versions()
    for name in ("torch", "stable-worldmodel", "stable-pretraining", "pymunk", "pygame", "opencv-python", "numpy", "gymnasium"):
        if parent.get("versions", {}).get(name) != current_versions[name]:
            raise SimulatorValidationError(f"Parent runtime version changed: {name}")
    dataset = repo_root / cfg.experiment.cache_dir / f"{cfg.experiment.dataset_name}.h5"
    old = parent["dataset"]
    current = file_metadata(dataset)
    if current["size_bytes"] != old["size_bytes"] or current["modified_utc"] != old["modified_utc"]:
        raise SimulatorValidationError("Source dataset file identity changed since baseline")
    statistics = parent.get("action_normalizer", {})
    normalizer = ActionNormalizer(mean=statistics["mean"], std=statistics["std"])
    return dict(path=str(path), sha256=sha256_file(path), run_id=parent["run_id"],
                checkpoint_sha256=expected, frozen_sources_checked=len(checked), dataset_metadata=current,
                allowed_extensions=["CLI dispatch", "simulator configuration section"],
                dataset_identity_check="baseline file size/mtime plus Step 1 sampled identity, not full dataset hash"), normalizer


def write_simulator_summary(path, manifest):
    lines = ["# Phase I Step 2 — Fresh-start simulator replay", "",
             f"- Status: `{manifest['run_status']}`",
             f"- Fresh-start Step 2 gate: `{manifest['step2_gate_passed']}`",
             f"- Encoder-facing dataset ready: `{manifest['encoder_dataset_ready']}`",
             f"- Completed cases: `{len(manifest['cases'])}`",
             f"- Tolerance: `{manifest['settings'].get('atol')}`, `rtol=0`",
             "", "| Case | Repeat | Observer-neutral | History stationary | Agent–block points | Block–wall points |",
             "|---|---|---|---|---|---|"]
    for case in manifest["cases"]:
        behavior = case["behavior"]
        lines.append(f"| {case['scene']['scene_id']} | {case['repeat']['passed']} | {case['observer_neutral']['passed']} | {behavior['history_stationary']} | {behavior['contact_points']['agent_block']} | {behavior['contact_points']['block_wall']} |")
    lines.extend(["", "## Scope and unresolved assumptions", "",
                  "Only fresh quiescent starts are supported. No complete Chipmunk snapshot/restore is claimed.",
                  "Dynamic public-state restoration remains explicitly unsupported; saved negative diagnostics explain why.",
                  "Rendering is reproducible under the stated settings; exact compatibility with training rendering remains unresolved because source goal/render metadata is absent.",
                  "Do not collect encoder-facing ABC/H2 data until that convention is resolved. No H1/H2/H3 experiment ran here."])
    if "failure" in manifest:
        lines.extend(["", "## Failure", "", f"{manifest['failure']['type']}: {manifest['failure']['message']}"])
    path.write_text("\n".join(lines) + "\n")


def write_case_table(path, cases):
    fields = ["scene_id", "seed", "repeat_passed", "observer_neutral", "behavior_passed",
              "history_stationary", "raw_state_count", "physics_state_count",
              "agent_displacement_world_px", "block_displacement_world_px", "agent_block_points", "block_wall_points"]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            behavior = case["behavior"]
            writer.writerow(dict(scene_id=case["scene"]["scene_id"], seed=case["scene"]["seed"],
                                 repeat_passed=case["repeat"]["passed"], observer_neutral=case["observer_neutral"]["passed"],
                                 behavior_passed=behavior["passed"], history_stationary=behavior["history_stationary"],
                                 raw_state_count=case["repeat"]["raw_state_count"], physics_state_count=case["repeat"]["physics_state_count"],
                                 agent_displacement_world_px=behavior["agent_displacement_world_px"],
                                 block_displacement_world_px=behavior["block_displacement_world_px"],
                                 agent_block_points=behavior["contact_points"]["agent_block"], block_wall_points=behavior["contact_points"]["block_wall"]))


def run_simulator_stage(args, repo_root: Path) -> int:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", args.run_id):
        raise ValueError("run-id must be a single safe 1–96 character identifier")
    if args.skip_cem or args.rebuild_mismatched_caches or args.device not in (None, "cpu"):
        raise ValueError("Simulator stage is CPU physics only; baseline-only flags are not applicable")
    config_path = (repo_root / args.config).resolve()
    cfg = OmegaConf.load(config_path)
    section = cfg.simulator
    settings = SimulatorSettings(resolution=int(cfg.experiment.image_size), with_target=section.with_target,
                                 atol=float(section.atol))
    if TimeActionContract(**OmegaConf.to_container(cfg.contract)) != TimeActionContract() or int(section.raw_steps) != 35 or float(section.rtol) != 0:
        raise ValueError("Step 2 requires the frozen time contract, 35 raw actions and rtol=0")
    seeds = OmegaConf.to_container(section.seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Simulator suite seeds must be unique explicit nonnegative integers")
    timeout = float(section.timeout_seconds)
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("Simulator timeout must be positive/finite")
    results = (repo_root / cfg.experiment.results_dir / "phase1" / args.run_id).resolve()
    artifacts = (repo_root / cfg.experiment.cache_dir / "phase1" / args.run_id).resolve()
    for directory in (results, artifacts):
        if directory.exists():
            raise FileExistsError(f"Refusing to reuse {directory}; choose a new --run-id")
    git = git_metadata(repo_root)
    results.mkdir(parents=True); artifacts.mkdir(parents=True)
    manifest_path, summary_path = results / "simulator_manifest.json", results / "simulator_summary.md"
    manifest = dict(schema_version=1, stage="phase1_step2_fresh_start_replay", run_id=args.run_id,
                    run_status="running", step2_gate_passed=False, encoder_dataset_ready=False,
                    device="cpu", body_precision="float64", raw_action_precision="float32",
                    created_utc=datetime.now(timezone.utc).isoformat(), git=git,
                    config=OmegaConf.to_container(cfg, resolve=True), settings=asdict(settings),
                    config_file=file_metadata(config_path, include_hash=True),
                    command=[sys.executable, *sys.argv], cases=[], unsafe_restore_audit=[],
                    output_locations=dict(manifest=str(manifest_path), summary=str(summary_path),
                                          artifact_root=str(artifacts), replay_hdf5=str(artifacts / "simulator_replay.h5")))

    def save():
        write_json(manifest_path, jsonable(manifest))
        write_simulator_summary(summary_path, manifest)
        write_case_table(results / "simulator_cases.csv", manifest["cases"])

    save()
    deadline = time.monotonic() + timeout
    try:
        parent_path = args.baseline_manifest or str(section.baseline_manifest)
        manifest["baseline_parent"], normalizer = validate_baseline_parent(repo_root / parent_path, repo_root, cfg)
        manifest["action_normalizer"] = normalizer.as_dict()
        manifest["versions"] = package_versions()
        hashes = source_metadata(repo_root)
        for component in (swm_spaces, pymunk.Space, pymunk.Body, pymunk.Shape, pymunk.Arbiter):
            path = Path(inspect.getfile(component)).resolve()
            hashes[str(path)] = sha256_file(path)
        for path in (config_path, repo_root / "test_counterfactual_simulator.py", repo_root / "test_counterfactual_simulator_stage.py"):
            hashes[str(path)] = sha256_file(path)
        for name in ("AGENTS.md", "CODEX_EXPERIMENT_PLAN.md", "CODEX_REAL_PLAN.md", "reproduce_phase1.md"):
            path = repo_root / name
            if path.exists():
                hashes[str(path)] = sha256_file(path)
        manifest["source_sha256"] = hashes
        binary = Path(importlib.util.find_spec("pymunk._chipmunk").origin)
        manifest["chipmunk_binary"] = file_metadata(binary, include_hash=True)
        write_source_snapshot(artifacts / "source_snapshot.h5", hashes)
        manifest["reproducibility_scope"] = "same audited CPU/Pymunk/renderer backend; not cross-version/device"
        manifest["scope"] = dict(fresh_quiescent_start=True, arbitrary_dynamic_restore=False,
                                 model_inference=False, physics_changed=False, closed_loop=False)
        save()
        with h5py.File(artifacts / "simulator_replay.h5", "x") as handle:
            handle.attrs["schema_version"] = 1
            handle.attrs["contract"] = canonical_json(TimeActionContract().as_dict())
            handle.attrs["initial_time_convention"] = "h=0 after unchanged two-step reset; rollout h=1..35"
            for seed in seeds:
                for scene, actions, expected in validation_cases(seed):
                    first = replay_fixed_actions(scene, actions, settings, deadline=deadline)
                    group = handle.create_group(scene.scene_id)
                    write_replay(group.create_group("first"), first, normalizer)
                    handle.flush()
                    second = replay_fixed_actions(scene, actions, settings, deadline=deadline)
                    write_replay(group.create_group("second"), second, normalizer)
                    plain = replay_fixed_actions(scene, actions, settings, audited=False, deadline=deadline)
                    write_replay(group.create_group("unmodified"), plain, normalizer)
                    comparison = compare_replays(first, second, settings.atol)
                    neutral = compare_replays(first, plain, settings.atol, require_audit=False)
                    behavior = validate_case_behavior(first, expected, settings.atol)
                    case = dict(scene=asdict(scene), repeat=comparison, observer_neutral=neutral,
                                behavior=behavior, initial_validation=first["initial_validation"],
                                action_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
                                trajectory_body_sha256=hashlib.sha256(first["body"].tobytes()).hexdigest(),
                                image_sequence_sha256=hashlib.sha256(first["images"].tobytes()).hexdigest())
                    manifest["cases"].append(case)
                    handle.flush(); save()
                    print(f"Simulator {scene.scene_id}: repeat={comparison['passed']}, observer={neutral['passed']}, behavior={behavior['passed']}", flush=True)
                    if not comparison["passed"] or not neutral["passed"] or not behavior["passed"]:
                        raise SimulatorValidationError(f"Replay/observer/excitation gate failed: {scene.scene_id}")
                    if seed == seeds[0] and expected in ("free_motion", "agent_block"):
                        audit = audit_unsafe_public_restore(first, scene, settings)
                        write_replay(group.create_group("unsafe_public_restore"), audit)
                        small = {key: value for key, value in audit.items() if key not in ("reference_body", "restored_body", "source_public_state")}
                        manifest["unsafe_restore_audit"].append(dict(scene_id=scene.scene_id, **small))
                    if scene.scene_id.startswith("stationary_") and seed == seeds[0]:
                        Image.fromarray(first["images"][0]).save(artifacts / "fresh_render_sample.png")
        dataset = repo_root / cfg.experiment.cache_dir / f"{cfg.experiment.dataset_name}.h5"
        manifest["training_render_audit"] = audit_dataset_rendering(dataset, artifacts / "dataset_render_audit.h5")
        if time.monotonic() > deadline:
            raise TimeoutError("Simulator validation exceeded configured hard timeout")
        for path, expected_hash in hashes.items():
            if sha256_file(path) != expected_hash:
                raise SimulatorValidationError(f"Source changed during replay: {path}")
        if sha256_file(binary) != manifest["chipmunk_binary"]["sha256"]:
            raise SimulatorValidationError("Chipmunk binary changed during replay")
        if sha256_file(manifest["baseline_parent"]["path"]) != manifest["baseline_parent"]["sha256"]:
            raise SimulatorValidationError("Parent manifest changed during replay")
        if sha256_file(checkpoint_path(repo_root / cfg.experiment.cache_dir, cfg.experiment.policy_name)) != manifest["baseline_parent"]["checkpoint_sha256"]:
            raise SimulatorValidationError("Frozen checkpoint changed during replay")
        if file_metadata(dataset) != manifest["baseline_parent"]["dataset_metadata"]:
            raise SimulatorValidationError("Source dataset metadata changed during replay")
        manifest["artifacts"] = {path.name: file_metadata(path, include_hash=True) for path in artifacts.iterdir() if path.is_file()}
        manifest["run_status"] = "completed"
        manifest["step2_gate_passed"] = True
        manifest["next_gate"] = "Resolve source training goal/render convention before encoder-facing Step 3 data; no automatic progression"
    except Exception as error:
        manifest["run_status"] = "failed"
        manifest["failure"] = dict(type=type(error).__name__, message=str(error))
        (results / "failure_traceback.txt").write_text(traceback.format_exc())
        print(f"Simulator failed: {type(error).__name__}: {error}", file=sys.stderr)
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    print(f"Simulator {manifest['run_status']}; manifest: {manifest_path}")
    return 0 if manifest["step2_gate_passed"] else 1
