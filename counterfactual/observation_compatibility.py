"""Development-only encoder observation assessment, separate from provenance.

Read saved observations only. No environment construction, body assignments,
goal changes, probe fitting, latent patching, predictor or Step 3 collection.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback

import cv2
import h5py
import hdf5plugin  # noqa: F401
import imageio_ffmpeg
import numpy as np
from omegaconf import OmegaConf
import torch

from counterfactual.baseline import (
    configure_determinism, encoded_cache_path, file_metadata, git_metadata,
    package_versions, sha256_file, source_metadata, write_json, write_source_snapshot,
)
from counterfactual.model import frames_hwc_to_b1chw, load_frozen_lewm, validate_model_contract
from counterfactual.contracts import TimeActionContract
from counterfactual.render_provenance import color_masks, run_regressions, validate_simulator_parent
from counterfactual.simulator import SimulatorValidationError, jsonable, wrap_angle
from counterfactual.simulator_validation import validate_baseline_parent
from probe.embedding_cache import get_image_transform, preprocess_pixels
from probe.evaluate import load_probe_checkpoint, probe_checkpoint_path

METRICS = ("latent_l2", "agent_drift_world_px", "block_drift_world_px", "angle_drift_rad")


def family_partition(families, seed=42):
    ids = sorted(item["family_id"] for item in families)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate family identity")
    shuffled = np.random.default_rng(seed).permutation(ids).tolist()
    ndev, nval = int(.8 * len(ids)), int(.1 * len(ids))
    return dict(development=shuffled[:ndev], validation=shuffled[ndev:ndev + nval],
                protected_test=shuffled[ndev + nval:])


def initial_match_eligible(record):
    """Render/public-state match only; never a full historical-state claim."""
    state = np.asarray(record["source_state"])
    check = record.get("fresh_validation", {})
    return bool(record.get("sample_kind") == "initial_first" and "metrics" in record
                and state.shape == (7,) and np.isfinite(state).all() and np.all(state[5:] == 0)
                and check.get("quiescent") is True and check.get("reset_physics_steps") == 2
                and check.get("intended_state_max_abs", float("inf")) <= 1e-10)


def spatial_regions(render):
    masks = color_masks(render)
    # Disjoint masks in the SAVED 224-pixel grid. Priority avoids double counting.
    masks["block"] &= ~masks["agent"]
    masks["goal"] &= ~(masks["agent"] | masks["block"])
    masks["background"] = ~(masks["agent"] | masks["block"] | masks["goal"])
    kernel = np.ones((5, 5), np.uint8)
    regions = {}
    for name, mask in masks.items():
        interior = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
        regions[name] = mask
        regions[f"{name}_interior"] = interior
        regions[f"{name}_edge"] = mask & ~interior
    return regions


def spatial_residual(source, render, regions):
    delta = source.astype(np.float64) - render.astype(np.float64)
    total_energy = float(np.sum(delta ** 2))
    rows = {}
    for name, mask in regions.items():
        values = delta[mask]
        energy = float(np.sum(values ** 2))
        rows[name] = dict(pixel_count=int(mask.sum()),
                          rgb_rmse=float(np.sqrt(np.mean(values ** 2))) if len(values) else None,
                          signed_mean_rgb=values.mean(axis=0).tolist() if len(values) else None,
                          energy_share=energy / total_energy if total_energy > 0 else None)
    return rows


def codec_roundtrip(image, output, codec, pixel_format, crf, commands):
    if image.dtype != np.uint8 or image.shape != (224, 224, 3):
        raise ValueError("Codec input must be raw 224 RGB uint8")
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    encode = [executable, "-v", "error", "-nostdin", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
              "-s", "224x224", "-r", "10", "-i", "pipe:0", "-frames:v", "1", "-an",
              "-c:v", codec, "-pix_fmt", pixel_format, "-threads", "1"]
    if codec == "libx264":
        encode += ["-preset", "medium", "-crf", str(crf), "-g", "1", "-bf", "0"]
    encode += [str(output)]
    commands.append(encode)
    completed = subprocess.run(encode, input=image.tobytes(), capture_output=True, timeout=30)
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode(errors="replace"))
    decode = [executable, "-v", "error", "-nostdin", "-i", str(output), "-frames:v", "1",
              "-f", "rawvideo", "-pix_fmt", "rgb24", "-threads", "1", "pipe:1"]
    commands.append(decode)
    completed = subprocess.run(decode, capture_output=True, timeout=30)
    if completed.returncode or len(completed.stdout) != image.nbytes:
        raise RuntimeError("Codec decode failed or returned an incompatible frame")
    return np.frombuffer(completed.stdout, np.uint8).reshape(image.shape).copy()


def natural_dose_rows(source, family, doses, tolerance):
    episode = family["episodes"][0]
    offset, length = int(source["ep_offset"][episode]), int(source["ep_len"][episode])
    states = source["state"][offset:offset + length]
    fixed_block = np.all(states[:, 2:5] == states[0, 2:5], axis=1)
    distance = np.linalg.norm(states[:, :2].astype(float) - states[0, :2], axis=1)
    candidates = np.flatnonzero(fixed_block & (np.arange(length) > 0))
    result = []
    for dose in doses:
        index = int(candidates[np.argmin(np.abs(distance[candidates] - dose))]) if len(candidates) else None
        if index is None or abs(distance[index] - dose) > tolerance:
            result.append(dict(dose=float(dose), status="missing", reason="No fixed-block source frame within state-only dose tolerance"))
        else:
            result.append(dict(dose=float(dose), status="matched", row=offset + index, step=index,
                               actual_agent_displacement_world_px=float(distance[index])))
    return result


def read_cached_bank(cache_root, source, families, partition, frames_per_family):
    requests, owners = [], []
    for family in families:
        if family["family_id"] not in partition["development"]:
            continue
        episode = family["episodes"][0]
        offset, length = int(source["ep_offset"][episode]), int(source["ep_len"][episode])
        for step in np.unique(np.linspace(0, length - 1, min(length, frames_per_family), dtype=int)):
            requests.append(offset + int(step)); owners.append(family["family_id"])
    requested = np.array(requests)
    mapping = {}
    files = {}
    offsets = source["ep_offset"][:]
    for split in ("train", "val", "test"):
        path = encoded_cache_path(cache_root, "pusht_expert_train", "pusht/lewm", split, 42, 224)
        files[split] = file_metadata(path)
        data = torch.load(path, map_location="cpu", mmap=True, weights_only=False)["encoded"]
        raw = offsets[data["episode_idx"].numpy()] + data["step_idx"].numpy()
        for index in np.flatnonzero(np.isin(raw, requested)):
            row = int(raw[index])
            if row in mapping:
                raise SimulatorValidationError("Duplicate natural-bank cache mapping")
            if not np.array_equal(data["state"][index].numpy(), source["state"][row]):
                raise SimulatorValidationError("Natural-bank cache label identity failed")
            mapping[row] = data["emb"][index].numpy().copy()
        if file_metadata(path) != files[split]:
            raise SimulatorValidationError("Cache metadata changed")
    if set(mapping) != set(requests):
        raise SimulatorValidationError("Incomplete natural-bank mapping")
    return np.stack([mapping[row] for row in requests]), np.array(owners), requests, files


def encode_images(model, images, batch_size=16):
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = frames_hwc_to_b1chw(np.stack(images[start:start + batch_size]))
            pixels = preprocess_pixels(batch, get_image_transform(224), "cpu")
            outputs.append(model.encode({"pixels": pixels})["emb"][:, 0].cpu().numpy())
    result = np.concatenate(outputs)
    if result.shape != (len(images), 192) or not np.isfinite(result).all():
        raise SimulatorValidationError("Nonfinite/incompatible frozen encoder output")
    return result


def readout_latents(latents, loaded):
    with torch.inference_mode():
        values = torch.from_numpy(latents).float()
        return {name: (probe["probe"](values) * probe["target_std"] + probe["target_mean"]).numpy()
                for name, probe in loaded.items()}


def pair_metrics(left, right, z, readouts):
    delta = z[right].astype(float) - z[left]
    return dict(latent_l2=float(np.linalg.norm(delta)),
                latent_norm_change=float(np.linalg.norm(z[right].astype(float)) - np.linalg.norm(z[left].astype(float))),
                agent_drift_world_px=float(np.linalg.norm(readouts["agent_location"][right] - readouts["agent_location"][left])),
                block_drift_world_px=float(np.linalg.norm(readouts["block_location"][right] - readouts["block_location"][left])),
                angle_drift_rad=float(np.abs(wrap_angle(float(readouts["block_angle"][right, 0]) - float(readouts["block_angle"][left, 0])))))


def family_rms_bounds(family_squared_values, seed, draws):
    values = np.asarray(family_squared_values, dtype=float)
    if len(values) == 0 or np.any(values < 0) or not np.isfinite(values).all():
        return dict(status="undefined", n_families=len(values))
    samples = np.random.default_rng(seed).integers(0, len(values), size=(draws, len(values)))
    statistic = np.sqrt(values[samples].mean(axis=1))
    return dict(status="descriptive", n_families=len(values), rms=float(np.sqrt(values.mean())),
                lower_one_sided_95=float(np.quantile(statistic, .05)),
                upper_one_sided_95=float(np.quantile(statistic, .95)))


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator is not None and denominator > 0 else None


def summarize(records, cfg, results):
    calibration, validation = {}, {}
    for metric in METRICS:
        squared = [np.mean([row[metric] ** 2 for row in item["natural_repeat_metrics"]])
                   for item in records if item["split"] == "development"]
        calibration[metric] = family_rms_bounds(squared, cfg.bootstrap_seed, cfg.bootstrap_draws)
    # Persist development margins BEFORE evaluating validation aggregates.
    write_json(results / "development_calibration.json", calibration)
    for metric in METRICS:
        squared = [item["raw_reconstruction"][metric] ** 2 for item in records if item["split"] == "validation"]
        bound = family_rms_bounds(squared, cfg.bootstrap_seed, cfg.bootstrap_draws)
        margin = calibration[metric].get("upper_one_sided_95")
        bound["development_natural_nuisance_margin"] = margin
        bound["within_natural_nuisance_margin"] = bool(margin is not None and bound.get("upper_one_sided_95", float("inf")) <= margin)
        validation[metric] = bound
    caps = dict(agent_drift_world_px=float(cfg.drift_caps.agent_world_px),
                block_drift_world_px=float(cfg.drift_caps.block_world_px), angle_drift_rad=float(cfg.drift_caps.angle_rad))
    practical = {name: validation[name].get("upper_one_sided_95", float("inf")) <= cap for name, cap in caps.items()}
    dose10 = [pair["latent_l2"] for item in records if item["split"] == "development"
              for pair in item["dose_metrics"] if pair["status"] == "matched" and pair["dose"] == 10.]
    signal = float(np.median(dose10)) if dose10 else None
    nuisance = validation["latent_l2"].get("upper_one_sided_95")
    pilot = all(practical.values()) and signal is not None and nuisance is not None and nuisance < signal
    return dict(development_calibration=calibration, validation=validation, proposed_drift_caps=caps,
                practical_checks=practical, development_median_10px_latent_signal=signal,
                confirmatory_block_effect_drift_cap_world_px=float(cfg.confirmatory_block_effect_drift_cap_world_px),
                confirmatory_block_effect_resolution_check=validation["block_drift_world_px"].get("upper_one_sided_95", float("inf")) <= cfg.confirmatory_block_effect_drift_cap_world_px,
                validation_nuisance_to_10px_signal=safe_ratio(nuisance, signal),
                budget_sensitivity={str(scale): {name: validation[name].get("upper_one_sided_95", float("inf")) <= scale * cap for name, cap in caps.items()} for scale in (.5, 1., 2.)},
                development_only_pilot_recommended=pilot, confirmatory_contract_freeze_supported=False,
                freeze_limitations=["Only five matched validation families, previously exposed to pixel audit.",
                                    "Full historical dynamics/pre-quantization state cannot be certified; source comparisons are render/public-state matched.",
                                    "Legacy MLPs lack family-held-out provenance; absolute/differential readout resolution requires further validation.",
                                    "No coherent-history or near-contact domain calibration; actual patch-scale validation belongs to the later intervention gate, not an observation-freeze prerequisite.",
                                    "Prospective display/task-goal contract requires explicit review; no production image transformation is approved."])


def assess_saved_images(parent, section, cache_root, artifacts, results, model, loaded):
    # Paths below are taken from the actual parent artifact, not guessed runs.
    comparison_path = Path(parent["artifacts"]["render_comparisons.h5"]["path"])
    results_parent = Path(parent["baseline_parent"]["path"]).parents[1] / parent["run_id"]
    families = json.loads((results_parent / "initial_families.json").read_text())
    prior = json.loads((results_parent / "sample_records.json").read_text())
    partition = family_partition(families, section.split_seed)
    write_json(results / "family_partition.json", partition)
    lookup = {family["family_id"]: family for family in families}
    images, identities, pending, commands, spatial = [], [], [], [], []

    def add(image, identity):
        images.append(np.array(image, copy=True)); identities.append(identity)
        return len(images) - 1

    with h5py.File(cache_root / "pusht_expert_train.h5", "r") as source, h5py.File(comparison_path, "r") as saved:
        bank, bank_families, bank_rows, cache_files = read_cached_bank(cache_root, source, families, partition, section.natural_reference_frames_per_family)
        for old in prior:
            family_id = old["family_id"]
            if not initial_match_eligible(old) or family_id in partition["protected_test"]:
                continue
            split = "development" if family_id in partition["development"] else "validation"
            raw_row = old["raw_row"]
            original, render = saved[str(raw_row)]["source"][:], saved[str(raw_row)]["visible"][:]
            if not np.array_equal(original, source["pixels"][raw_row]) or not np.array_equal(saved[str(raw_row)]["source_state"][:], source["state"][raw_row]):
                raise SimulatorValidationError("Saved comparison/source row identity changed")
            family = lookup[family_id]
            base = add(original, dict(family_id=family_id, condition="hdf5_source", raw_row=raw_row))
            reconstructed = add(render, dict(family_id=family_id, condition="raw_reconstruction", raw_row=raw_row))
            repeat_indices = []
            for episode in family["episodes"][1:section.reference_episodes_per_family]:
                row = int(source["ep_offset"][episode])
                if not np.array_equal(source["state"][row], source["state"][raw_row]):
                    raise SimulatorValidationError("Natural repeat does not match source public state")
                repeat_indices.append(add(source["pixels"][row], dict(family_id=family_id, condition="natural_same_pose", raw_row=row)))
            region = spatial_regions(render)
            spatial.append(dict(family_id=family_id, split=split, condition="hdf5_minus_raw", regions=spatial_residual(original, render, region)))
            codec_indices = {}
            directory = artifacts / "codec" / family_id
            directory.mkdir(parents=True)
            conditions = [("ffv1_lossless", "ffv1", "bgr0", None)] + [(f"x264_{fmt}_crf{crf}", "libx264", fmt, crf) for fmt in section.codec_pixel_formats for crf in section.codec_crfs]
            for name, codec, fmt, crf in conditions:
                decoded = codec_roundtrip(render, directory / f"{name}.mkv", codec, fmt, crf, commands)
                if name == "ffv1_lossless" and not np.array_equal(render, decoded):
                    raise SimulatorValidationError("Lossless codec control changed RGB bytes")
                codec_indices[name] = add(decoded, dict(family_id=family_id, condition=name, raw_row=raw_row))
                spatial.append(dict(family_id=family_id, split=split, condition=name + "_minus_raw", regions=spatial_residual(decoded, render, region)))
                spatial.append(dict(family_id=family_id, split=split, condition="hdf5_minus_" + name, regions=spatial_residual(original, decoded, region)))
            doses = natural_dose_rows(source, family, section.dose_world_px, section.dose_match_tolerance_world_px)
            for dose in doses:
                if dose["status"] == "matched":
                    dose["image_index"] = add(source["pixels"][dose["row"]], dict(family_id=family_id, condition=f"natural_dose{dose['dose']}", raw_row=dose["row"]))
            pending.append(dict(family_id=family_id, split=split, raw_row=raw_row, state=source["state"][raw_row].tolist(),
                                match_scope="render-relevant/public state only; historical hidden dynamics/pre-quantization pose not certified",
                                base=base, reconstructed=reconstructed, repeats=repeat_indices, codecs=codec_indices, doses=doses))
    if not pending:
        raise SimulatorValidationError("No eligible non-test initial observations")
    print(f"Encoding {len(images)} saved/diagnostic images from {len(pending)} families", flush=True)
    z = encode_images(model, images)
    repeat = encode_images(model, images)
    if not np.array_equal(z, repeat):
        raise SimulatorValidationError("Frozen encoding repeat mismatch")
    # Independent identity check against the original sampled-cache mappings:
    # a large source/render drift must not be an accidental checkpoint/pipeline
    # mismatch. Never load protected-test query latents for this check.
    mappings = json.loads((results_parent / "cache_frame_mappings.json").read_text())["rows"]
    maxima, matches = [], []
    for split in ("train", "val", "test"):
        path = encoded_cache_path(cache_root, "pusht_expert_train", "pusht/lewm", split, 42, 224)
        cache = torch.load(path, map_location="cpu", mmap=True, weights_only=False)["encoded"]["emb"]
        for item in pending:
            mapping = mappings[str(item["raw_row"])]
            if mapping["split"] == split:
                existing = cache[mapping["cache_row"]].numpy()
                actual = z[item["base"]]
                maxima.append(float(np.max(np.abs(actual - existing))))
                matches.append(bool(np.allclose(actual, existing, atol=1e-5, rtol=1e-4)))
    if len(matches) != len(pending) or not all(matches):
        raise SimulatorValidationError("Original source encodings disagree with frozen cache identity")
    readouts = readout_latents(z, loaded)
    records = []
    for item in pending:
        left, right = item["base"], item["reconstructed"]
        record = {key: item[key] for key in ("family_id", "split", "raw_row", "state", "match_scope")}
        record["raw_reconstruction"] = pair_metrics(left, right, z, readouts)
        raw_tensor = preprocess_pixels(frames_hwc_to_b1chw(np.stack([images[left], images[right]])), get_image_transform(224), "cpu")
        record["raw_reconstruction"]["preprocessed_tensor_rmse"] = float(torch.sqrt(torch.mean((raw_tensor[1] - raw_tensor[0]) ** 2)))
        record["natural_repeat_metrics"] = [pair_metrics(left, index, z, readouts) for index in item["repeats"]]
        record["codec_same_state_metrics"] = {name: pair_metrics(right, index, z, readouts) for name, index in item["codecs"].items()}
        record["hdf5_vs_codec_metrics"] = {name: pair_metrics(left, index, z, readouts) for name, index in item["codecs"].items()}
        record["dose_metrics"] = [{**dose, **(pair_metrics(left, dose["image_index"], z, readouts) if dose["status"] == "matched" else {})} for dose in item["doses"]]
        record["absolute_readout_errors"] = {}
        for name, selection in (("agent_location", slice(0, 2)), ("block_location", slice(2, 4)), ("block_angle", slice(4, 5))):
            truth = np.array(item["state"])[selection]
            errors = {}
            for label, index in (("source", left), ("raw_reconstruction", right)):
                difference = readouts[name][index].astype(float) - truth
                errors[label] = float(np.abs(wrap_angle(difference[0]))) if name == "block_angle" else float(np.linalg.norm(difference))
            record["absolute_readout_errors"][name] = errors
        references = bank[bank_families != item["family_id"]].astype(float)
        record["natural_bank_distances"] = {label: float(np.min(np.linalg.norm(references - z[index], axis=1))) for label, index in (("source", left), ("raw_reconstruction", right))}
        records.append(record)
    write_json(results / "case_metrics.json", records)
    write_json(results / "spatial_metrics.json", spatial)
    write_json(artifacts / "codec_commands.json", commands)
    with h5py.File(artifacts / "compatibility_arrays.h5", "x") as output:
        output.create_dataset("images", data=np.stack(images), compression="gzip")
        output.create_dataset("embeddings", data=z)
        for name, values in readouts.items():
            output.create_dataset(f"readouts/{name}", data=values)
        output.create_dataset("natural_reference_latents", data=bank)
        output.create_dataset("natural_reference_rows", data=bank_rows)
        output.attrs["image_identities"] = json.dumps(identities)
        output.attrs["natural_reference_families"] = json.dumps(bank_families.tolist())
    with (results / "case_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["family_id", "split", "raw_row", *METRICS])
        writer.writeheader()
        for record in records:
            writer.writerow({**{key: record[key] for key in ("family_id", "split", "raw_row")}, **{key: record["raw_reconstruction"][key] for key in METRICS}})
    return records, dict(n_families=len(records), image_count=len(images), repeated_encoding_max_abs=0.,
                         source_cache_identity=dict(sample_count=len(matches), max_abs_embedding_difference=max(maxima),
                                                    all_match=True, atol=1e-5, rtol=1e-4),
                         reference_bank_rows=len(bank), cache_files=cache_files,
                         match_scope="See per-case scope; no full historical-state claim")


def run_compatibility_stage(args, repo_root):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", args.run_id):
        raise ValueError("Invalid immutable run identifier")
    if args.device not in (None, "cpu") or args.skip_cem or args.rebuild_mismatched_caches or args.source_evidence:
        raise ValueError("Compatibility assessment is read-only CPU inference, not baseline/cache mutation")
    cfg = OmegaConf.load(repo_root / args.config)
    section = cfg.simulator.compatibility
    results = repo_root / cfg.experiment.results_dir / "phase1" / args.run_id
    artifacts = repo_root / cfg.experiment.cache_dir / "phase1" / args.run_id
    for directory in (results, artifacts):
        if directory.exists():
            raise FileExistsError(f"Refusing to reuse {directory}")
    results.mkdir(parents=True); artifacts.mkdir(parents=True)
    manifest_path = results / "compatibility_manifest.json"
    manifest = dict(stage="phase1_step2_observation_compatibility", schema_version=1, run_id=args.run_id,
                    run_status="running", created_utc=datetime.now(timezone.utc).isoformat(),
                    git=git_metadata(repo_root), command=[sys.executable, *sys.argv], config=OmegaConf.to_container(cfg),
                    exact_historical_provenance="UNRESOLVED", encoder_dataset_ready=False,
                    confirmatory_contract_frozen=False, pilot_collection_performed=False,
                    scope="Saved-image development assessment only; no environments, goals, probes or interventions modified")
    write_json(manifest_path, manifest)
    try:
        manifest["baseline_parent"], _ = validate_baseline_parent(repo_root / (args.baseline_manifest or cfg.simulator.baseline_manifest), repo_root, cfg)
        parent_path = repo_root / (args.provenance_manifest or section.provenance_manifest)
        parent = json.loads(parent_path.read_text())
        if parent.get("run_status") != "completed" or parent.get("stage") != "phase1_step2_render_provenance":
            raise SimulatorValidationError("Completed immutable provenance assessment required")
        manifest["provenance_parent"] = file_metadata(parent_path, include_hash=True)
        manifest["simulator_parent"] = validate_simulator_parent(Path(parent["simulator_parent"]["path"]), repo_root)
        for key in ("render_comparisons.h5",):
            if sha256_file(parent["artifacts"][key]["path"]) != parent["artifacts"][key]["sha256"]:
                raise SimulatorValidationError("Parent comparison artifact changed")
        manifest["versions"] = package_versions()
        manifest["deterministic_settings"] = configure_determinism(int(cfg.experiment.seed))
        hashes = source_metadata(repo_root)
        for relative in (args.config, "observation_compatibility_protocol.md", "test_counterfactual_observation_compatibility.py", "probe/evaluate.py", "probe/models.py"):
            path = (repo_root / relative).resolve(); hashes[str(path)] = sha256_file(path)
        manifest["source_sha256"] = hashes
        write_source_snapshot(artifacts / "source_snapshot.h5", hashes)
        binary = Path(imageio_ffmpeg.get_ffmpeg_exe())
        manifest["ffmpeg"] = dict(binary=file_metadata(binary, include_hash=True),
                                  version=subprocess.run([str(binary), "-version"], capture_output=True, check=True).stdout.decode().splitlines()[0])
        cache_root = repo_root / cfg.experiment.cache_dir
        model = load_frozen_lewm(cfg.experiment.policy_name, cache_root, "cpu")
        manifest["model_contract"] = validate_model_contract(model, TimeActionContract())
        loaded = {}
        manifest["readouts"] = {}
        for target in ("agent_location", "block_location", "block_angle"):
            path = probe_checkpoint_path(cache_root, target, "mlp")
            loaded[target] = load_probe_checkpoint(path, "cpu")
            loaded[target]["probe"].requires_grad_(False)
            manifest["readouts"][target] = file_metadata(path, include_hash=True)
        records, manifest["measurements"] = assess_saved_images(parent, section, cache_root, artifacts, results, model, loaded)
        manifest["assessment"] = summarize(records, section, results)
        write_json(manifest_path, jsonable(manifest))
        manifest["regressions"] = run_regressions(repo_root, artifacts)
        command = [sys.executable, str(repo_root / "test_counterfactual_observation_compatibility.py")]
        completed = subprocess.run(command, cwd=repo_root, capture_output=True, timeout=300)
        (artifacts / "regression_logs/compatibility.stdout.txt").write_bytes(completed.stdout)
        (artifacts / "regression_logs/compatibility.stderr.txt").write_bytes(completed.stderr)
        manifest["compatibility_tests"] = dict(command=command, return_code=completed.returncode)
        if completed.returncode:
            raise SimulatorValidationError("Compatibility regression failed")
        for filename, expected in hashes.items():
            if sha256_file(filename) != expected:
                raise SimulatorValidationError(f"Source changed during assessment: {filename}")
        for record in manifest["readouts"].values():
            if sha256_file(record["path"]) != record["sha256"]:
                raise SimulatorValidationError("Readout checkpoint changed")
        from counterfactual.model import checkpoint_path
        if sha256_file(checkpoint_path(cache_root, cfg.experiment.policy_name)) != manifest["baseline_parent"]["checkpoint_sha256"]:
            raise SimulatorValidationError("Frozen model checkpoint changed during assessment")
        if file_metadata(cache_root / "pusht_expert_train.h5") != manifest["baseline_parent"]["dataset_metadata"]:
            raise SimulatorValidationError("Original dataset metadata changed during assessment")
        for record in manifest["measurements"]["cache_files"].values():
            if file_metadata(record["path"]) != record:
                raise SimulatorValidationError("Existing cache metadata changed during assessment")
        from counterfactual.compatibility_report import write_compatibility_plots
        write_compatibility_plots(results)
        manifest["run_status"] = "completed"
        manifest["artifacts"] = {str(path.relative_to(artifacts)): file_metadata(path, include_hash=True) for path in artifacts.rglob("*") if path.is_file()}
    except Exception as error:
        manifest["run_status"] = "failed"
        manifest["failure"] = dict(type=type(error).__name__, message=str(error))
        (results / "failure_traceback.txt").write_text(traceback.format_exc())
        print(f"Compatibility assessment failed: {error}", file=sys.stderr)
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, jsonable(manifest))
    print(f"Compatibility {manifest['run_status']}; confirmatory freeze=False; {manifest_path}", flush=True)
    return 0 if manifest["run_status"] == "completed" else 1
