"""Step 2 observation-provenance audit; never collects counterfactual data.

Source defaults and approximate pose-matched images cannot identify an unknown
dataset generation run. This audit therefore cannot promote Step 3 readiness.
All renderer controls are temporary display-only changes on validated fresh
scenes, with restoration and mechanics/RNG checks. No body state is restored.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback
import urllib.request

import h5py
import hdf5plugin  # noqa: F401 — register the source file's lossless Blosc filter
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
import torch

from counterfactual.baseline import (
    encoded_cache_path, file_metadata, git_metadata, package_versions, sha256_file, source_metadata,
    write_json, write_source_snapshot,
)
from counterfactual.model import IMAGE_PREPROCESSING_CONTRACT, pixels_hwc_to_btc
from counterfactual.simulator import (
    SceneSpec, SimulatorSettings, SimulatorValidationError, canonical_json,
    create_fresh_scene, jsonable, physical_readback, rng_snapshot, simulator_contract,
)
from counterfactual.simulator_validation import validate_baseline_parent
from probe.embedding_cache import get_image_transform, preprocess_pixels


SWM_EARLY = "7124d26abef3b38afdd738e7f9f676900b46d8b8"
SWM_MARCH = "a00659833e13b2148663b5bfaf5e549146c6d22b"
DINO_REVISION = "0a9492fa12044b852ae9e001cc74604b79c8bb0c"
HF_REVISION = "655cd446b9929369d7d406001da85c15d1457850"


def github_source(repo, revision, path):
    return f"https://raw.githubusercontent.com/{repo}/{revision}/{path}"


# Public source snapshots only. These are evidence of available implementations,
# not proof that this exact code/config produced the local training artifact.
SOURCE_URLS = {
    "early_converter": github_source("galilai-group/stable-worldmodel", SWM_EARLY, "scripts/data/pusht_converter.py"),
    "early_video_loader": github_source("galilai-group/stable-worldmodel", SWM_EARLY, "scripts/data/pusht_dataset.py"),
    "dino_renderer": github_source("gaoyuezhou/dino_wm", DINO_REVISION, "env/pusht/pusht_env.py"),
    "dino_video_loader": github_source("gaoyuezhou/dino_wm", DINO_REVISION, "datasets/pusht_dset.py"),
    "march_renderer": github_source("galilai-group/stable-worldmodel", SWM_MARCH, "stable_worldmodel/envs/pusht/env.py"),
    "march_weak_collector": github_source("galilai-group/stable-worldmodel", SWM_MARCH, "scripts/data/collect_weak_pusht.py"),
    "march_collection_config": github_source("galilai-group/stable-worldmodel", SWM_MARCH, "scripts/data/config/default.yaml"),
    "dataset_card": f"https://huggingface.co/datasets/quentinll/lewm-pusht/raw/{HF_REVISION}/README.md",
    "dataset_tree": f"https://huggingface.co/api/datasets/quentinll/lewm-pusht/tree/{HF_REVISION}",
}

RENDER_CANDIDATE = dict(
    resolution=224, with_target=True, render_goal=True,
    goal_pose_world_px_rad=[256., 256., float(np.pi / 4)],
    render_action=False, world_size=512, candidate_only=True,
)

REGRESSION_FILES = (
    "test_counterfactual_contracts.py", "test_counterfactual_baseline.py",
    "test_counterfactual_simulator.py", "test_counterfactual_simulator_stage.py",
    "test_counterfactual_render_provenance.py", "test_rollout_ablation.py",
    "test_rollout_windows.py", "test_rollout_probe_eval.py",
    "test_rollout_visualization.py", "test_rollout_reconstruction.py", "test_latent_diagnostics.py",
)


def validate_candidate(candidate):
    # Strict schema/types: integer 1 must not masquerade as an explicit boolean.
    if canonical_json(candidate) != canonical_json(RENDER_CANDIDATE):
        raise SimulatorValidationError("Render comparison candidate changed; not an approved generation contract")


def require_encoder_dataset_ready(manifest):
    """Fail closed: this descriptive audit never authorizes dataset collection."""
    if manifest.get("encoder_dataset_ready") is not True or manifest.get("verified_generation_contract") is not True:
        raise SimulatorValidationError("Step 3 blocked: source observation-generation contract is not verified")


def snapshot_public_sources(directory: Path, reuse_index=None):
    directory.mkdir()
    previous = json.loads(Path(reuse_index).read_text()) if reuse_index else None
    records = {}
    for name, url in SOURCE_URLS.items():
        if previous is not None:
            record = previous[name]
            if record["url"] != url:
                raise SimulatorValidationError(f"Unpinned source evidence: {name}")
            # Restrict reused snapshots to the fixed filename beside the index.
            content = (Path(reuse_index).parent / f"{name}.txt").read_bytes()
            if hashlib.sha256(content).hexdigest() != record["sha256"]:
                raise SimulatorValidationError(f"Source snapshot changed: {name}")
        else:
            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read(2_000_001)
            if len(content) > 2_000_000:
                raise SimulatorValidationError("Refusing an unexpectedly large source response")
        path = directory / f"{name}.txt"
        path.write_bytes(content)
        lines = content.decode("utf-8").splitlines()
        patterns = ("goal_pose", "with_target", "render_size", "cv2.resize", "VideoReader",
                    "get_batch", "astype(np.uint8)", "normalize_action", "record_dataset", "pusht_noise")
        records[name] = dict(url=url, path=str(path), sha256=sha256_file(path),
                             size_bytes=len(content), line_count=len(lines),
                             relevant_lines=[dict(line=i + 1, text=line) for i, line in enumerate(lines)
                                             if any(pattern in line for pattern in patterns)])
    write_json(directory / "index.json", records)
    return records


def initial_families(source):
    offsets, lengths = source["ep_offset"][:], source["ep_len"][:]
    if len(offsets) == 0 or len(offsets) != len(lengths) or np.any(lengths <= 0) or not np.array_equal(offsets, np.r_[0, np.cumsum(lengths[:-1])]):
        raise SimulatorValidationError("Non-contiguous or empty source episodes")
    if int(lengths.sum()) != len(source["pixels"]):
        raise SimulatorValidationError("Episode lengths disagree with source row count")
    starts = source["state"][offsets]
    if starts.shape != (len(offsets), 7) or not np.isfinite(starts).all():
        raise SimulatorValidationError("Invalid source initial states")
    groups = {}
    for episode, state in enumerate(starts):
        # Exact stored float32 public-observation identity, not a full physical
        # snapshot, original demo identity, or evidence of appearance variation.
        key = hashlib.sha256(state.tobytes()).hexdigest()
        group = groups.setdefault(key, dict(family_id=key, initial_state=state.tolist(), episodes=[]))
        group["episodes"].append(episode)
    return list(groups.values())


def select_families(families, count):
    if type(count) is not int or count < 1:
        raise SimulatorValidationError("Family sample count must be a positive integer")
    states = np.array([item["initial_state"] for item in families])
    selected = []
    # Include spatial extrema, then evenly spaced orientation order. No image
    # score, reconstruction success, or encoder output is used for selection.
    for column in range(5):
        for index in (int(states[:, column].argmin()), int(states[:, column].argmax())):
            if index not in selected:
                selected.append(index)
    order = np.argsort(states[:, 4] % (2 * np.pi), kind="stable")
    for index in order[np.linspace(0, len(order) - 1, min(count, len(order)), dtype=int)]:
        if int(index) not in selected:
            selected.append(int(index))
    for index in order:
        if int(index) not in selected:
            selected.append(int(index))
    return [families[index] for index in selected[:min(count, len(families))]]


def selected_frames(source, families):
    offsets, lengths = source["ep_offset"][:], source["ep_len"][:]
    for family in families:
        first, last = family["episodes"][0], family["episodes"][-1]
        candidates = [(first, 0, "initial_first"), (last, 0, "initial_last"),
                      (first, int(lengths[first]) // 2, "middle"),
                      (first, int(lengths[first]) - 1, "final")]
        seen = set()
        for episode, step, label in candidates:
            row = int(offsets[episode]) + step
            if row in seen:
                continue
            seen.add(row)
            if int(source["episode_idx"][row]) != episode or int(source["step_idx"][row]) != step:
                raise SimulatorValidationError("Source frame mapping is inconsistent")
            yield dict(family_id=family["family_id"], episode_id=episode, step_idx=step,
                       raw_row=row, sample_kind=label)


def render_display_only(env, *, visible, goal_pose):
    """Temporary marker controls only. No physics/reset/body corrections."""
    if type(visible) is not bool:
        raise SimulatorValidationError("Goal visibility must be an explicit boolean")
    pose = np.asarray(goal_pose, dtype=np.float64)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise SimulatorValidationError("Render marker requires a finite x/y/radian pose")
    before = canonical_json(physical_readback(env))
    rng_before = canonical_json(rng_snapshot(env))
    steps_before = env.physics_step_index
    goal_image_before = np.array(env._goal, copy=True)
    saved = env.with_target, env.goal_pose
    try:
        env.with_target, env.goal_pose = visible, pose.copy()
        image = np.array(env.render(), copy=True)
    finally:
        env.with_target, env.goal_pose = saved
    if (canonical_json(physical_readback(env)) != before or canonical_json(rng_snapshot(env)) != rng_before
            or env.physics_step_index != steps_before or not np.array_equal(env._goal, goal_image_before)):
        raise SimulatorValidationError("Display-only rendering changed mechanics, goal state, or RNG")
    if image.dtype != np.uint8 or image.shape != (224, 224, 3):
        raise SimulatorValidationError("Raw render shape/dtype changed")
    return image


def validate_render_boundary(env):
    """Detect incompatible installed styles/settings before comparisons.

The initial physical fixture's marker pose is deliberately not the candidate:
only render_display_only supplies the comparison pose, then restores the fixture.
"""
    actual = simulator_contract(env)
    expected = dict(resolution=224, world_size=512, with_target=True,
                    render_action=False, render_goal=True, mode="rgb_array")
    if canonical_json(actual["rendering"]) != canonical_json(expected):
        raise SimulatorValidationError("Installed render settings differ from the comparison candidate")
    for group, rgb in (("agent", [65, 105, 225]), ("block", [119, 136, 153]),
                       ("goal", [144, 238, 144]), ("background", [255, 255, 255])):
        if actual["variation"][group]["color"] != rgb:
            raise SimulatorValidationError(f"Installed {group} color differs from source default")
    shapes = actual["bodies"]
    if (len(shapes["agent"]["shapes"]) != 1 or shapes["agent"]["shapes"][0]["kind"] != "Circle"
            or shapes["agent"]["shapes"][0]["radius"] != 15.
            or len(shapes["block"]["shapes"]) != 2
            or any(item["kind"] != "Poly" for item in shapes["block"]["shapes"])):
        raise SimulatorValidationError("Installed object geometry differs from default circle/T")
    return actual


def color_masks(image):
    """Fixed, descriptive RGB segmentation; not pose estimation or exact labels."""
    red, green, blue = np.moveaxis(image.astype(np.int16), -1, 0)
    return dict(goal=(green - red > 30) & (green - blue > 30) & (green > 130),
                agent=(blue - red > 40) & (blue - green > 30),
                block=(blue - red > 10) & (green - red > 5) & (blue - green < 35) & (green < 200))


def masked_rmse(delta, mask):
    return float(np.sqrt(np.mean(delta[mask] ** 2))) if mask.any() else None


def image_metrics(source, reconstructed, goal_mask):
    if source.dtype != np.uint8 or reconstructed.dtype != np.uint8 or source.shape != reconstructed.shape:
        raise SimulatorValidationError("Compare matching raw uint8 RGB shapes only")
    delta = reconstructed.astype(np.float64) - source.astype(np.float64)
    original_masks, new_masks = color_masks(source), color_masks(reconstructed)
    metrics = dict(pixel_rmse=float(np.sqrt(np.mean(delta ** 2))), pixel_mae=float(np.mean(np.abs(delta))),
                   pixel_max_abs=float(np.max(np.abs(delta))),
                   exact_rgb_pixel_fraction=float(np.all(delta == 0, axis=-1).mean()),
                   goal_roi_rmse=masked_rmse(delta, goal_mask))
    metrics["nongoal_roi_rmse"] = masked_rmse(delta, ~goal_mask)
    for name in original_masks:
        old, new = original_masks[name], new_masks[name]
        union = old | new
        metrics[f"{name}_mask_iou"] = float((old & new).sum() / union.sum()) if union.any() else None
        metrics[f"{name}_source_pixels"] = int(old.sum())
        for label, mask in (("source", old), ("render", new)):
            # Coordinates are x/y in the saved 224-pixel grid, not world px.
            coords = np.argwhere(mask)
            metrics[f"{name}_{label}_centroid_image_xy"] = coords.mean(axis=0)[::-1].tolist() if len(coords) else None
            metrics[f"{name}_{label}_bbox_image_xyxy"] = [*coords.min(axis=0)[::-1].tolist(), *coords.max(axis=0)[::-1].tolist()] if len(coords) else None
            pixels = source[mask] if label == "source" else reconstructed[mask]
            metrics[f"{name}_{label}_median_rgb"] = np.median(pixels, axis=0).tolist() if len(pixels) else None
    background = np.all(reconstructed == 255, axis=-1)
    metrics["background_rmse"] = masked_rmse(delta, background)
    metrics["source_background_median_rgb"] = np.median(source[background], axis=0).tolist() if background.any() else None
    return metrics


def preprocessing_parity(images):
    """Exercise real training, official evaluation, and Step 1 transforms."""
    from utils import get_img_preprocessor
    from eval import img_transform
    if images.ndim != 4 or images.shape[1:] != (224, 224, 3):
        raise SimulatorValidationError("Parity audit expects saved 224x224 raw RGB frames")
    raw = pixels_hwc_to_btc(images)
    actual = preprocess_pixels(raw, get_image_transform(224), "cpu")
    train = get_img_preprocessor("pixels", "pixels", 224)({"pixels": raw[0].clone()})["pixels"]
    official = img_transform(OmegaConf.create({"eval": {"img_size": 224}}))(raw[0])
    # Independent formula checks uint8 scaling and normalization, without reusing
    # the preprocessing implementation. At 224 there is no spatial resize.
    mean = torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1)
    # torchvision multiplies by the reciprocal in float32, rather than dividing
    # by 255. These have slightly different rounding; match the frozen operation.
    manual = (raw[0].float() * (1. / 255.) - mean) / std
    deltas = dict(training=float((actual[0] - train).abs().max()),
                  official_evaluation=float((actual[0] - official).abs().max()),
                  independent_formula=float((actual[0] - manual).abs().max()))
    if not all(value == 0. for value in deltas.values()):
        raise SimulatorValidationError(f"Frozen preprocessing mismatch: {deltas}")
    return dict(status="PASS", max_abs_deltas=deltas, shape=list(actual.shape), dtype=str(actual.dtype),
                contract=IMAGE_PREPROCESSING_CONTRACT, imagenet_mean=mean.flatten().tolist(),
                imagenet_std=std.flatten().tolist(), raw_scale="0..255 RGB uint8", tensor_scale="ImageNet-normalized float32")


def comparison_montage(path, source, renders):
    labels = ["source", *renders, "absolute error x4"]
    images = [source, *renders.values(), np.clip(np.abs(source.astype(float) - renders["visible"].astype(float)) * 4, 0, 255).astype(np.uint8)]
    canvas = Image.new("RGB", (224 * len(images), 246), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (label, image) in enumerate(zip(labels, images)):
        draw.text((224 * column + 4, 3), label, fill="black")
        canvas.paste(Image.fromarray(image), (224 * column, 22))
    canvas.save(path)


def audit_source_images(dataset, artifacts, results, count, seed):
    records, parity_images = [], []
    with h5py.File(dataset, "r") as source, h5py.File(artifacts / "render_comparisons.h5", "x") as output:
        families = initial_families(source)
        selected = select_families(families, count)
        write_json(results / "initial_families.json", families)
        filters = source["pixels"].id.get_create_plist()
        structure = dict(columns={key: dict(shape=list(source[key].shape), dtype=str(source[key].dtype)) for key in source},
                         root_attrs=jsonable(dict(source.attrs)),
                         column_attrs={key: jsonable(dict(source[key].attrs)) for key in source},
                         pixel_filters=[jsonable(filters.get_filter(i)) for i in range(filters.get_nfilters())],
                         family_count=len(families), episode_count=len(source["ep_offset"]),
                         episodes_per_family=sorted(set(len(item["episodes"]) for item in families)),
                         selected_family_ids=[item["family_id"] for item in selected],
                         initial_state_min=np.min([item["initial_state"] for item in selected], axis=0).tolist(),
                         initial_state_max=np.max([item["initial_state"] for item in selected], axis=0).tolist(),
                         family_definition="exact stored initial 7D observation; not full physics/demo/appearance identity")
        for identity in selected_frames(source, selected):
            row = identity["raw_row"]
            image, state = source["pixels"][row], source["state"][row].astype(np.float64)
            group = output.create_group(str(row))
            group.create_dataset("source", data=image, compression="gzip")
            group.create_dataset("source_state", data=state)
            group.attrs["identity"] = canonical_json(identity)
            record = dict(**identity, source_image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                          source_state=state.tolist(), comparison_status="not_attempted", exact_reconstruction=False,
                          source_goal_pixels=int(color_masks(image)["goal"].sum()))
            records.append(record)
            if len(parity_images) < 8:
                parity_images.append(image)
            # This is a new quiescent illustration of stored position/angle,
            # NOT restoration of this source frame's dynamics or hidden state.
            scene = SceneSpec(f"source_pose_{row}", seed, tuple(state[:2]), tuple(state[2:4]), float(state[4]))
            try:
                env, validation = create_fresh_scene(scene, SimulatorSettings())
            except SimulatorValidationError as error:
                record.update(comparison_status="rejected_fresh_pose", reason=str(error))
                continue  # never replace a rejected observation with an easier one
            try:
                boundary = validate_render_boundary(env)
                pose = np.array(RENDER_CANDIDATE["goal_pose_world_px_rad"])
                renders = dict(
                    visible=render_display_only(env, visible=True, goal_pose=pose),
                    hidden=render_display_only(env, visible=False, goal_pose=pose),
                    shifted=render_display_only(env, visible=True, goal_pose=pose + [48., -32., 0.]),
                    rotated=render_display_only(env, visible=True, goal_pose=pose + [0., 0., np.pi / 4]),
                )
                repeated = render_display_only(env, visible=True, goal_pose=pose)
                if not np.array_equal(renders["visible"], repeated):
                    raise SimulatorValidationError("Rendering is not deterministic")
                goal_roi = np.any(renders["visible"] != renders["hidden"], axis=-1)
                metrics = {name: image_metrics(image, candidate, goal_roi) for name, candidate in renders.items()}
                for name, candidate in renders.items():
                    group.create_dataset(name, data=candidate, compression="gzip")
                group.create_dataset("goal_roi", data=goal_roi, compression="gzip")
                group.create_dataset("visible_error_rgb", data=renders["visible"].astype(np.int16) - image.astype(np.int16), compression="gzip")
                group.attrs["metrics"] = canonical_json(metrics)
                record.update(comparison_status="approximate_pose_matched", fresh_validation=validation,
                              deterministic_render=True, mechanics_unchanged_by_render=True, metrics=metrics,
                              fresh_body_contract=boundary if not any("fresh_body_contract" in item for item in records) else None,
                              limitation="Stored float32 pose only; source block dynamics/codec/config unavailable. Source agent velocity intentionally NOT restored.")
                if identity["sample_kind"] == "initial_first":
                    comparison_montage(results / f"comparison_{row}.png", image, renders)
            finally:
                env.close()
        output.attrs["purpose"] = "Step 2 descriptive provenance audit only; no AB/ABC/H1/H2 data"
    write_json(results / "sample_records.json", records)
    comparisons = [item for item in records if "metrics" in item]
    initial = [item for item in comparisons if item["sample_kind"] == "initial_first"]
    # Aggregate initial_first only: no pseudo-replication from same-pose repeats.
    aggregates = {}
    for condition in ("visible", "hidden", "shifted", "rotated"):
        values = [item["metrics"][condition]["pixel_rmse"] for item in initial]
        aggregates[condition] = dict(n_initial_families=len(values),
                                     mean_rmse=float(np.mean(values)) if values else None,
                                     median_rmse=float(np.median(values)) if values else None,
                                     min_rmse=min(values) if values else None, max_rmse=max(values) if values else None)
    with (results / "image_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["family_id", "episode_id", "step_idx", "raw_row", "sample_kind", "condition", "pixel_rmse", "pixel_mae", "goal_roi_rmse", "nongoal_roi_rmse", "agent_mask_iou", "block_mask_iou", "goal_mask_iou", "background_rmse"])
        writer.writeheader()
        for item in comparisons:
            for condition, metrics in item["metrics"].items():
                writer.writerow({key: item.get(key, metrics.get(key, condition if key == "condition" else None)) for key in writer.fieldnames})
    return dict(structure=structure, sample_count=len(records), compared_count=len(comparisons),
                rejected_count=len(records) - len(comparisons), initial_family_aggregates=aggregates,
                exact_image_matches=sum(item["metrics"]["visible"]["pixel_rmse"] == 0 for item in comparisons),
                preprocessing=preprocessing_parity(np.stack(parity_images)))


def audit_cache_mappings(dataset, cache_root, results):
    """Read-only, memory-mapped source-row lookup and sampled label identity.

This does not re-encode images or strengthen Step 1's sampled latent guarantee.
"""
    records = json.loads((results / "sample_records.json").read_text())
    requested = np.array(sorted(item["raw_row"] for item in records), dtype=np.int64)
    mapping, summaries = {}, {}
    with h5py.File(dataset, "r") as source:
        offsets, lengths = source["ep_offset"][:], source["ep_len"][:]
        for split in ("train", "val", "test"):
            path = encoded_cache_path(cache_root, "pusht_expert_train", "pusht/lewm", split, 42, 224)
            before = file_metadata(path)
            payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            encoded = payload["encoded"]
            episodes, steps = encoded["episode_idx"].numpy(), encoded["step_idx"].numpy()
            if np.any(episodes < 0) or np.any(episodes >= len(offsets)) or np.any(steps < 0) or np.any(steps >= lengths[episodes]):
                raise SimulatorValidationError("Invalid existing cache episode/step mapping")
            raw_rows = offsets[episodes] + steps
            matches = np.flatnonzero(np.isin(raw_rows, requested))
            for index in matches:
                row = int(raw_rows[index])
                if row in mapping:
                    raise SimulatorValidationError("Audited source row belongs to multiple cached rows/splits")
                for key in ("state", "proprio"):
                    if not np.array_equal(encoded[key][index].numpy(), source[key][row]):
                        raise SimulatorValidationError("Cache label does not match audited source image row")
                mapping[row] = dict(split=split, cache_row=int(index), episode_id=int(episodes[index]), step_idx=int(steps[index]))
            if before != file_metadata(path):
                raise SimulatorValidationError("Existing cache metadata changed during audit")
            summaries[split] = dict(file=before, metadata=payload["metadata"], sampled_rows_matched=len(matches))
            del payload, encoded, episodes, steps, raw_rows
    if set(mapping) != set(requested.tolist()):
        raise SimulatorValidationError("Audited source images missing from existing caches")
    report = dict(status="PASS", rows={str(key): value for key, value in sorted(mapping.items())},
                  splits=summaries, scope="Audited frame mappings/labels only; embeddings reuse passing Step 1 sampled guarantee")
    write_json(results / "cache_frame_mappings.json", report)
    return dict(status="PASS", audited_rows=len(mapping), splits=summaries)


def validate_release_archive(dataset, archive, official_tree):
    """Prove file identity without downloading or writing a second 46-GB HDF5.

Lossless container identity is deliberately separate from renderer provenance.
The optional retained archive is checked against the pinned HF LFS SHA-256;
its decompression is byte-compared against the complete local HDF5 stream.
"""
    if not archive.exists():
        return dict(status="UNAVAILABLE", reason="Retained release archive not present; no download attempted")
    import zstandard  # already installed; only needed for this optional check
    entries = json.loads(Path(official_tree).read_text())
    entry = next(item for item in entries if item["path"] == "pusht_expert_train.h5.zst")
    before, archive_before = file_metadata(dataset), file_metadata(archive)
    print("Checking complete retained archive against pinned HF LFS SHA-256", flush=True)
    compressed_hash = sha256_file(archive)
    if compressed_hash != entry["lfs"]["oid"] or archive_before["size_bytes"] != entry["size"]:
        raise SimulatorValidationError("Retained archive differs from pinned official dataset release")
    digest, size = hashlib.sha256(), 0
    print("Byte-comparing complete decompressed release against local HDF5 (no output copy)", flush=True)
    with archive.open("rb") as compressed, dataset.open("rb") as original:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as decoded:
            while chunk := decoded.read(4 * 1024 * 1024):
                if original.read(len(chunk)) != chunk:
                    raise SimulatorValidationError(f"Decompressed release differs from local HDF5 at byte block {size}")
                digest.update(chunk)
                size += len(chunk)
                if size % (8 * 1024 ** 3) == 0:
                    print(f"Release identity compared: {size} bytes", flush=True)
        if original.read(1):
            raise SimulatorValidationError("Local HDF5 has trailing bytes not present in the release")
    if before != file_metadata(dataset) or archive_before != file_metadata(archive):
        raise SimulatorValidationError("Source/archive metadata changed during full identity check")
    return dict(status="PASS", hf_revision=HF_REVISION, archive={**archive_before, "sha256": compressed_hash},
                dataset_sha256=digest.hexdigest(), byte_count=size, complete_byte_equality=True,
                zstandard_version=zstandard.__version__,
                scope="Exact official release/file identity; does NOT recover pre-release renderer/config/codec")


def run_regressions(repo_root, artifacts):
    records = []
    directory = artifacts / "regression_logs"
    directory.mkdir()
    for name in REGRESSION_FILES:
        command = [sys.executable, str(repo_root / name)]
        print(f"Regression: {name}", flush=True)
        completed = subprocess.run(command, cwd=repo_root, capture_output=True, timeout=300)
        stdout, stderr = directory / f"{name}.stdout.txt", directory / f"{name}.stderr.txt"
        stdout.write_bytes(completed.stdout); stderr.write_bytes(completed.stderr)
        records.append(dict(command=command, return_code=completed.returncode,
                            stdout=file_metadata(stdout, include_hash=True), stderr=file_metadata(stderr, include_hash=True)))
        write_json(directory / "regressions.json", records)
        if completed.returncode != 0:
            raise SimulatorValidationError(f"Regression failed: {name}; see {directory}")
    return records


def validate_simulator_parent(path, repo_root):
    parent = json.loads(Path(path).read_text())
    if parent.get("stage") != "phase1_step2_fresh_start_replay" or parent.get("step2_gate_passed") is not True or parent.get("run_status") != "completed":
        raise SimulatorValidationError("Provenance requires a completed passing Step 2 parent")
    # Preserve all recorded Python inference/physics/test implementations except
    # the CLI that adds dispatch. Documentation/config extensions are separate.
    for filename, expected in parent["source_sha256"].items():
        if filename.endswith(".py") and Path(filename).resolve() != (repo_root / "scripts/run_phase1.py").resolve():
            if sha256_file(filename) != expected:
                raise SimulatorValidationError(f"Step 2 implementation changed: {filename}")
    if sha256_file(parent["chipmunk_binary"]["path"]) != parent["chipmunk_binary"]["sha256"]:
        raise SimulatorValidationError("Step 2 physics binary changed")
    return dict(path=str(Path(path).resolve()), sha256=sha256_file(path), run_id=parent["run_id"],
                step2_gate_passed=True, mechanics_scope="same backend, unchanged Step 2 implementations; rerun tests separately")


def run_render_provenance_stage(args, repo_root):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", args.run_id):
        raise ValueError("run-id must be a single safe identifier")
    if args.skip_cem or args.rebuild_mismatched_caches or args.device not in (None, "cpu"):
        raise ValueError("Render audit is CPU only and does not accept baseline mutation flags")
    config_path = (repo_root / args.config).resolve()
    cfg = OmegaConf.load(config_path)
    section = cfg.simulator.provenance
    validate_candidate(OmegaConf.to_container(section.candidate))
    results = repo_root / cfg.experiment.results_dir / "phase1" / args.run_id
    artifacts = repo_root / cfg.experiment.cache_dir / "phase1" / args.run_id
    for path in (results, artifacts):
        if path.exists():
            raise FileExistsError(f"Refusing to reuse {path}")
    results.mkdir(parents=True); artifacts.mkdir(parents=True)
    manifest_path = results / "render_provenance_manifest.json"
    manifest = dict(schema_version=1, stage="phase1_step2_render_provenance", run_id=args.run_id,
                    run_status="running", encoder_dataset_ready=False, verified_generation_contract=False,
                    created_utc=datetime.now(timezone.utc).isoformat(), git=git_metadata(repo_root),
                    command=[sys.executable, *sys.argv], config=OmegaConf.to_container(cfg),
                    comparison_candidate=RENDER_CANDIDATE, versions=package_versions(),
                    scope="Source/image audit only; no dataset generation, intervention, or planning")
    write_json(manifest_path, manifest)
    try:
        manifest["baseline_parent"], _ = validate_baseline_parent(repo_root / (args.baseline_manifest or cfg.simulator.baseline_manifest), repo_root, cfg)
        manifest["simulator_parent"] = validate_simulator_parent(repo_root / (args.simulator_manifest or section.simulator_manifest), repo_root)
        hashes = source_metadata(repo_root)
        import stable_worldmodel.envs.utils as draw_utils
        import stable_pretraining.data.transforms as training_transforms
        for filename in (inspect.getfile(draw_utils), inspect.getfile(training_transforms), config_path,
                         repo_root / "train.py", *(repo_root / name for name in REGRESSION_FILES)):
            hashes[str(Path(filename).resolve())] = sha256_file(filename)
        manifest["source_sha256"] = hashes
        write_source_snapshot(artifacts / "source_snapshot.h5", hashes)
        manifest["public_sources"] = snapshot_public_sources(artifacts / "sources", args.source_evidence)
        dataset = repo_root / cfg.experiment.cache_dir / f"{cfg.experiment.dataset_name}.h5"
        before = file_metadata(dataset)
        manifest["dataset"] = before
        manifest["release_archive_identity"] = validate_release_archive(
            dataset, dataset.with_suffix(".h5.zst"), artifacts / "sources/dataset_tree.txt")
        write_json(manifest_path, jsonable(manifest))
        manifest["image_audit"] = audit_source_images(dataset, artifacts, results, int(section.family_sample_count), int(cfg.simulator.seeds[0]))
        manifest["cache_frame_mapping"] = audit_cache_mappings(dataset, repo_root / cfg.experiment.cache_dir, results)
        write_json(manifest_path, jsonable(manifest))
        manifest["regressions"] = run_regressions(repo_root, artifacts)
        if before != file_metadata(dataset):
            raise SimulatorValidationError("Source dataset metadata changed during audit")
        for filename, expected in hashes.items():
            if sha256_file(filename) != expected:
                raise SimulatorValidationError(f"Source changed during audit: {filename}")
        for parent in (manifest["baseline_parent"], manifest["simulator_parent"]):
            if sha256_file(parent["path"]) != parent["sha256"]:
                raise SimulatorValidationError("Parent evidence changed during audit")
        manifest["gate_status"] = dict(
            fresh_start_mechanics="PASS (unchanged passing parent; fresh mechanics regression suite passed)",
            goal_render_visibility="source-default confirmed; local dataset empirically assessed, not run-provenance confirmed",
            goal_pose_and_variation="source-default fixed (256,256,pi/4); actual dataset generation config unresolved",
            raw_image_generation_compatibility="INCONCLUSIVE",
            frozen_encoder_preprocessing_compatibility="PASS",
            encoder_facing_step3_dataset_readiness=False,
        )
        manifest["blockers"] = [
            "No generation/conversion run manifest connects this HDF5 to an exact renderer/config/library revision.",
            "HDF5 omits source goal/style/seed/variation metadata and original MP4 encoding settings.",
            "Available historical converter copies decoded MP4 pixels but targets Minari, not this HDF5; complete conversion chain unknown.",
            "Raw fresh renders are not byte-identical to source images; codec/backend/pose residuals cannot be uniquely attributed.",
            "Rendered marker is not the installed physical task goal; do not infer goal_state from green pixels.",
        ]
        manifest["run_status"] = "completed"
        manifest["artifacts"] = {str(path.relative_to(artifacts)): file_metadata(path, include_hash=True)
                                 for path in artifacts.rglob("*") if path.is_file()}
    except Exception as error:
        manifest["run_status"] = "failed"
        manifest["failure"] = dict(type=type(error).__name__, message=str(error))
        (results / "failure_traceback.txt").write_text(traceback.format_exc())
        print(f"Render provenance failed: {error}", file=sys.stderr)
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, jsonable(manifest))
    print(f"Render provenance {manifest['run_status']}; Step 3 ready=False; {manifest_path}")
    return 0 if manifest["run_status"] == "completed" else 1
