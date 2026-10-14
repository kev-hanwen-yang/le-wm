"""Step 1 baseline checks and provenance reporting for Phase I."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import inspect
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import h5py
import numpy as np
import torch

from counterfactual.contracts import TimeActionContract
from counterfactual.model import checkpoint_path, frames_hwc_to_b1chw, run_frozen_rollout, require_tensor
from probe.embedding_cache import get_image_transform, preprocess_pixels


def configure_determinism(seed: int) -> dict:
    """Seed the inference process and require deterministic PyTorch operations."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "seed": seed,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def invoke_official_cem(seed: int, device: str, record_path: str) -> None:
    """Observe official evaluation inputs/metrics while retaining its algorithm."""
    settings = configure_determinism(seed)
    import eval as official

    original_evaluate = official.swm.World.evaluate_from_dataset

    def observed_evaluate(world, dataset, **kwargs):
        solver = world.policy.solver
        record = {
            "determinism": settings,
            "device": str(solver.device),
            "model_device": str(next(solver.model.parameters()).device),
            "evaluation_inputs": {
                key: value for key, value in kwargs.items() if key != "video_path"
            },
            "solver": {
                key: getattr(solver, key)
                for key in ("batch_size", "num_samples", "n_steps", "topk", "var_scale")
            },
            "plan_config": vars(world.policy.cfg),
            "normalizers": {
                key: {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
                for key, scaler in world.policy.process.items()
            },
            "simulator_reset_seeds": None,
            "simulator_reset_note": "Official evaluator uses dataset seed column; this dataset has none.",
        }
        write_json(record_path, record)
        started = time.monotonic()
        metrics = original_evaluate(world, dataset, **kwargs)
        record["metrics"] = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in metrics.items()
        }
        record["evaluation_seconds"] = time.monotonic() - started
        write_json(record_path, record)
        return metrics

    official.eval_device = lambda: device
    official.swm.World.evaluate_from_dataset = observed_evaluate
    official.run.__wrapped__.__module__ = "__main__"
    sys.argv[0] = "eval.py"
    official.run()


class CacheIdentityError(RuntimeError):
    """Raised when sampled cached embeddings do not reproduce from the checkpoint."""


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 hash without loading a checkpoint into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_metadata(path: str | Path, include_hash: bool = False) -> dict:
    path = Path(path)
    stat = path.stat()
    result = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }
    if include_hash:
        result["sha256"] = sha256_file(path)
    return result


def git_metadata(repo_root: str | Path) -> dict:
    """Capture immutable revision plus any uncommitted-state warning."""
    repo_root = Path(repo_root)

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True
        )
        return completed.stdout.strip()

    return {
        "revision": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "status_porcelain": git("status", "--porcelain"),
    }


def source_metadata(repo_root: str | Path) -> dict:
    """Fingerprint edited source and installed inference/physics implementations."""
    from jepa import JEPA
    from stable_worldmodel.solver import CEMSolver
    import importlib.util

    repo_root = Path(repo_root)
    paths = list((repo_root / "counterfactual").glob("*.py")) + [
        repo_root / relative for relative in (
            "scripts/run_phase1.py", "jepa.py", "module.py", "utils.py", "eval.py",
            "probe/embedding_cache.py", "probe/rollout_ablation.py", "probe/data.py",
            "config/eval/pusht.yaml", "config/eval/solver/cem.yaml", "config/train/lewm.yaml",
        )
    ]
    paths.extend(Path(inspect.getfile(component)) for component in (JEPA, CEMSolver))
    paths.append(Path(importlib.util.find_spec("stable_worldmodel").origin).parent / "envs/pusht/env.py")
    return {str(path.resolve()): sha256_file(path) for path in sorted(set(paths))}


def write_source_snapshot(path: Path, hashes: dict) -> None:
    """Persist exact local/installed source bytes for a run with uncommitted edits."""
    with h5py.File(path, "x") as handle:
        for index, (source_path, expected_hash) in enumerate(sorted(hashes.items())):
            content = Path(source_path).read_bytes()
            if hashlib.sha256(content).hexdigest() != expected_hash:
                raise RuntimeError(f"Source changed before snapshot: {source_path}")
            dataset = handle.create_dataset(str(index), data=np.frombuffer(content, dtype=np.uint8))
            dataset.attrs["path"] = source_path
            dataset.attrs["sha256"] = expected_hash


def package_versions() -> dict:
    packages = [
        "stable-worldmodel",
        "stable-pretraining",
        "torch",
        "torchvision",
        "transformers",
        "hdf5plugin",
        "lightning",
        "hydra-core",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "gymnasium",
        "h5py",
        "einops",
        "pymunk",
        "pygame",
        "opencv-python",
    ]
    result = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "MISSING"
    result["python"] = sys.version.replace("\n", " ")
    result["platform"] = platform.platform()
    result["torch_cuda_available"] = torch.cuda.is_available()
    result["torch_mps_available"] = bool(
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    )
    return result


def model_precision(model) -> dict:
    """Report the actual inference precision instead of assuming float32."""
    parameter = next(model.parameters())
    return {
        "parameter_dtype": str(parameter.dtype),
        "default_dtype": str(torch.get_default_dtype()),
        "inference_mode": True,
        "autocast": False,
    }


def encoded_cache_path(cache_dir: str | Path, dataset_name: str, policy_name: str, split: str, split_seed: int, image_size: int) -> Path:
    safe_policy = policy_name.replace("/", "_")
    return (
        Path(cache_dir)
        / "probes"
        / "encoded"
        / f"{dataset_name}_{safe_policy}_{split}_seed{split_seed}_img{image_size}_encoded.pt"
    )


def deterministic_sample_indices(length: int, sample_count: int) -> list[int]:
    if length < 1:
        raise ValueError("Cannot sample an empty encoded cache")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    return sorted({int(index) for index in np.linspace(0, length - 1, min(length, sample_count))})


def cache_samples(length: int, sample_count: int, seed: int) -> list[int]:
    """Sample across cache strata with fixed-seed interior rows and endpoints."""
    base = deterministic_sample_indices(length, sample_count)
    if len(base) <= 2:
        return base
    edges = np.linspace(0, length, len(base) + 1, dtype=np.int64)
    generator = np.random.default_rng(seed)
    return [0] + [
        int(generator.integers(edges[index], edges[index + 1]))
        for index in range(1, len(base) - 1)
    ] + [length - 1]


def load_cache_payload(path: Path):
    """Use memory mapping for existing large Torch caches; HDF5 repairs stream rows."""
    if path.suffix == ".h5":
        handle = h5py.File(path, "r")
        return handle, handle["encoded"], json.loads(handle.attrs["metadata"])
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return None, payload.get("encoded"), payload.get("metadata", {})


def read_cache_rows(column, indices) -> np.ndarray:
    rows = column[indices]
    return rows.numpy() if torch.is_tensor(rows) else np.asarray(rows)


def validate_encoded_cache(
    *,
    cache_path: str | Path,
    dataset_path: str | Path,
    model,
    device: str,
    image_size: int,
    sample_count: int,
    atol: float,
    rtol: float,
    contract: TimeActionContract | None = None,
    expected_metadata: dict | None = None,
    sample_seed: int = 42,
) -> dict:
    """Verify schema, episode membership, sampled labels, and sampled encodings."""
    cache_path = Path(cache_path)
    dataset_path = Path(dataset_path)
    contract = contract or TimeActionContract()
    result = {"status": "invalid", "cache": {"path": str(cache_path)}}
    if not cache_path.exists():
        return {**result, "status": "missing"}
    cache_handle = None
    try:
        cache_handle, encoded, metadata = load_cache_payload(cache_path)
        required = {"emb", "episode_idx", "step_idx", "state", "proprio"}
        if encoded is None or not required.issubset(encoded):
            raise ValueError(f"Cache requires columns {sorted(required)}")
        row_count = len(encoded["emb"])
        for key, shape in {
            "emb": (row_count, contract.latent_dim),
            "episode_idx": (row_count,), "step_idx": (row_count,),
            "state": (row_count, 7), "proprio": (row_count, 4),
        }.items():
            if tuple(encoded[key].shape) != shape:
                raise ValueError(f"Cache {key} shape disagrees with {shape}")
        for key, value in (expected_metadata or {}).items():
            if metadata.get(key) != value:
                raise ValueError(f"Cache metadata {key}={metadata.get(key)!r}; expected {value!r}")
        all_episode_ids = read_cache_rows(encoded["episode_idx"], slice(None))
        all_step_ids = read_cache_rows(encoded["step_idx"], slice(None))
        if all_episode_ids.dtype.kind not in "iu" or all_step_ids.dtype.kind not in "iu":
            raise ValueError("Cache episode/step IDs must be integers")
        indices = cache_samples(row_count, sample_count, sample_seed)
        with h5py.File(dataset_path, "r") as handle:
            offsets, lengths = handle["ep_offset"][:], handle["ep_len"][:]
            if np.any(all_episode_ids < 0) or np.any(all_episode_ids >= len(offsets)):
                raise ValueError("Cache episode ID is outside the dataset")
            if np.any(all_step_ids < 0) or np.any(all_step_ids >= lengths[all_episode_ids]):
                raise ValueError("Cache step ID is outside its episode")
            all_raw = offsets[all_episode_ids] + all_step_ids
            if np.any(np.diff(all_raw) <= 0):
                raise ValueError("Cache rows must be unique and ordered by raw dataset row")
            split = metadata.get("split_name")
            if split in ("train", "val", "test"):
                shuffled = np.arange(len(offsets))
                np.random.default_rng(int(metadata["seed"])).shuffle(shuffled)
                train_end = int(len(shuffled) * 0.8)
                val_end = train_end + int(len(shuffled) * 0.1)
                expected_episodes = {
                    "train": shuffled[:train_end], "val": shuffled[train_end:val_end],
                    "test": shuffled[val_end:],
                }[split]
                if not np.array_equal(np.unique(all_episode_ids), np.sort(expected_episodes)):
                    raise ValueError("Cache episode membership disagrees with the existing 80/10/10 split")
                if row_count != int(lengths[expected_episodes].sum()):
                    raise ValueError("Cache does not contain every frame of its episode split")
            raw_indices = all_raw[indices]
            episode_ids, step_ids = all_episode_ids[indices], all_step_ids[indices]
            if not np.array_equal(handle["episode_idx"][raw_indices], episode_ids):
                raise ValueError("Sampled dataset episode IDs disagree with cached IDs")
            if not np.array_equal(handle["step_idx"][raw_indices], step_ids):
                raise ValueError("Sampled dataset step IDs disagree with cached IDs")
            for key in ("state", "proprio"):
                if not np.array_equal(handle[key][raw_indices], read_cache_rows(encoded[key], indices)):
                    raise ValueError(f"Sampled {key} labels disagree with raw dataset")
            pixels = handle["pixels"][raw_indices]
        expected = torch.from_numpy(read_cache_rows(encoded["emb"], indices)).float()
        require_tensor(expected, (len(indices), contract.latent_dim), "cached sample")
        model_pixels = preprocess_pixels(
            frames_hwc_to_b1chw(pixels), get_image_transform(image_size), device,
        )
        with torch.inference_mode():
            actual = model.encode({"pixels": model_pixels})["emb"][:, 0].cpu().float()
        require_tensor(actual, tuple(expected.shape), "fresh cache sample")
        diff = actual - expected
        matches = bool(torch.allclose(actual, expected, atol=atol, rtol=rtol))
        return {
            "status": "match" if matches else "mismatch",
            "cache": file_metadata(cache_path), "cache_metadata": metadata,
            "sample_indices": indices, "episode_ids": episode_ids.tolist(),
            "step_ids": step_ids.tolist(), "raw_dataset_indices": raw_indices.tolist(),
            "sample_seed": sample_seed, "atol": atol, "rtol": rtol,
            "max_abs_difference": float(diff.abs().max()),
            "rmse": float(torch.sqrt(torch.mean(diff.square()))),
            "embedding_shape": list(encoded["emb"].shape),
            "sampled_pixels_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
            "scope": "all-row schema/index/split checks; sampled labels and embeddings",
        }
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        return {**result, "reason": f"{type(error).__name__}: {error}"}
    finally:
        if cache_handle is not None:
            cache_handle.close()


def rebuild_mismatched_cache(
    *, validation: dict, destination: Path, dataset_path: Path, model,
    device: str, image_size: int, checkpoint_sha256: str, batch_size: int = 64,
) -> Path:
    """Repair only a validated embedding mismatch into a new HDF5 artifact."""
    if validation["status"] != "mismatch":
        raise ValueError("Only an embedding mismatch with valid labels/indices may be rebuilt")
    source = Path(validation["cache"]["path"])
    cache_handle, encoded, metadata = load_cache_payload(source)
    metadata = {**metadata, "checkpoint_sha256": checkpoint_sha256, "rebuilt_from": str(source)}
    try:
        with h5py.File(dataset_path, "r") as raw, h5py.File(destination, "x") as output:
            output.attrs["metadata"] = json.dumps(metadata)
            group = output.create_group("encoded")
            for key, column in encoded.items():
                dtype = read_cache_rows(column, slice(0, 1)).dtype
                group.create_dataset(key, shape=column.shape, dtype=dtype, chunks=True)
            offsets = raw["ep_offset"][:]
            transform = get_image_transform(image_size)
            for start in range(0, len(encoded["emb"]), batch_size):
                rows = slice(start, min(start + batch_size, len(encoded["emb"])))
                episodes = read_cache_rows(encoded["episode_idx"], rows)
                steps = read_cache_rows(encoded["step_idx"], rows)
                pixels = raw["pixels"][offsets[episodes] + steps]
                prepared = preprocess_pixels(frames_hwc_to_b1chw(pixels), transform, device)
                with torch.inference_mode():
                    embeddings = model.encode({"pixels": prepared})["emb"][:, 0].cpu()
                require_tensor(embeddings, (len(episodes), encoded["emb"].shape[1]), "rebuilt embeddings")
                group["emb"][rows] = embeddings.numpy()
                for key in encoded:
                    if key != "emb":
                        group[key][rows] = read_cache_rows(encoded[key], rows)
                if start == 0 or rows.stop == len(encoded["emb"]):
                    print(f"Rebuild {destination.name}: {rows.stop}/{len(encoded['emb'])} frames", flush=True)
    finally:
        if cache_handle is not None:
            cache_handle.close()
    return destination


def choose_smoke_window(dataset_path: str | Path, contract: TimeActionContract, episode_id: int | None = None, lookahead_steps: int = 1) -> dict:
    """Load a valid h=0,5,10 / action 0:15 window directly from HDF5."""
    dataset_path = Path(dataset_path)
    required_steps = int(contract.rollout_indices(lookahead_steps)["target_steps"][-1]) + 1
    with h5py.File(dataset_path, "r") as handle:
        lengths = handle["ep_len"][:]
        if episode_id is None:
            candidates = np.flatnonzero(lengths >= required_steps)
            if not len(candidates):
                raise ValueError(f"No episode has the required {required_steps} raw steps")
            episode_id = int(candidates[0])
        if not 0 <= episode_id < len(lengths):
            raise ValueError(f"episode_id={episode_id} is outside [0, {len(lengths)})")
        if lengths[episode_id] < required_steps:
            raise ValueError(
                f"Episode {episode_id} has {lengths[episode_id]} steps; need {required_steps}"
            )
        offset = int(handle["ep_offset"][episode_id])
        context_steps = contract.context_steps.numpy()
        pixels = handle["pixels"][offset + context_steps]
        actions = handle["action"][offset : offset + required_steps - 1]
    return {"episode_id": episode_id, "pixels": pixels, "raw_actions": actions}


def check_model_determinism(
    *,
    model,
    window: dict,
    normalizer,
    contract: TimeActionContract,
    device: str,
    image_size: int,
    lookahead_steps: int = 5,
    artifact_path: Path | None = None,
) -> dict:
    """Run identical model inputs twice and require finite, identical outputs."""
    first = run_frozen_rollout(
        model, window["pixels"], window["raw_actions"], normalizer, contract, device, image_size, lookahead_steps,
    )
    second = run_frozen_rollout(
        model, window["pixels"], window["raw_actions"], normalizer, contract, device, image_size, lookahead_steps,
    )
    context_delta = first["context_emb"] - second["context_emb"]
    prediction_delta = first["prediction"] - second["prediction"]
    rollout_delta = first["rollout"] - second["rollout"]
    if artifact_path is not None:
        with h5py.File(artifact_path, "x") as handle:
            handle.attrs["episode_id"] = window["episode_id"]
            handle.create_dataset("pixels", data=window["pixels"])
            handle.create_dataset("raw_actions", data=window["raw_actions"])
            handle.create_dataset("target_steps", data=contract.rollout_indices(lookahead_steps)["target_steps"].numpy())
            for branch, payload in (("first", first), ("repeat", second)):
                group = handle.create_group(branch)
                for key, tensor in payload.items():
                    group.create_dataset(key, data=tensor.numpy())
    return {
        "episode_id": window["episode_id"],
        "context_embedding_shape": list(first["context_emb"].shape),
        "action_token_shape": list(first["action_tokens"].shape),
        "action_embedding_shape": list(first["action_emb"].shape),
        "prediction_shape": list(first["prediction"].shape),
        "rollout_shape": list(first["rollout"].shape),
        "rollout_target_steps": contract.rollout_indices(lookahead_steps)["target_steps"].tolist(),
        "context_repeat_max_abs": float(context_delta.abs().max()),
        "prediction_repeat_max_abs": float(prediction_delta.abs().max()),
        "rollout_repeat_max_abs": float(rollout_delta.abs().max()),
        "all_finite": bool(
            all(torch.isfinite(tensor).all() for payload in (first, second) for tensor in payload.values())
        ),
        "deterministic": all(torch.equal(first[key], second[key]) for key in first),
        "first_prediction_absolute_raw_step": contract.first_prediction_step,
    }


def build_cem_smoke_command(
    *,
    repo_root: str | Path,
    cache_dir: str | Path,
    policy_name: str,
    seed: int,
    num_eval: int,
    run_id: str,
    device: str = "cpu",
    dataset_name: str = "pusht_expert_train",
    image_size: int = 224,
    record_path: str | Path | None = None,
) -> tuple[list[str], Path]:
    """Build the unmodified official evaluator invocation for a small smoke test."""
    cache_dir = Path(cache_dir).resolve()
    result_path = cache_dir / "phase1" / run_id / "cem_smoke.txt"
    policy_stem = str(checkpoint_path(cache_dir, policy_name)).removesuffix("_object.ckpt")
    record_path = Path(record_path or result_path.with_name("cem_metrics.json")).resolve()
    wrapper = (
        "from counterfactual.baseline import invoke_official_cem; "
        f"invoke_official_cem({seed!r}, {device!r}, {str(record_path)!r})"
    )
    def override(key, value):
        return f"{key}={json.dumps(str(value), ensure_ascii=False)}"

    command = [
        sys.executable,
        "-c",
        wrapper,
        "--config-name=pusht.yaml",
        override("policy", policy_stem),
        override("cache_dir", cache_dir),
        f"seed={seed}",
        f"eval.num_eval={num_eval}",
        override("eval.dataset_name", dataset_name),
        f"eval.img_size={image_size}",
        override("output.filename", result_path),
        override("hydra.run.dir", result_path.parent / "hydra"),
    ]
    return command, result_path


def run_cem_smoke(*, results_dir: Path, timeout_seconds: float = 1800, **kwargs) -> dict:
    """Execute the official evaluator once; failures are recorded, never retried."""
    record_path = results_dir / "cem_metrics.json"
    command, result_path = build_cem_smoke_command(**kwargs, record_path=record_path)
    if result_path.exists() or record_path.exists():
        raise FileExistsError("Refusing to append CEM results to an existing run")
    repo_root = Path(kwargs["repo_root"])
    cache_dir = Path(kwargs["cache_dir"]).resolve()
    environment = os.environ.copy()
    environment["STABLEWM_HOME"] = str(cache_dir)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    started = time.monotonic()
    status = "failed"
    with (results_dir / "cem_smoke.stdout.txt").open("w") as stdout, (results_dir / "cem_smoke.stderr.txt").open("w") as stderr:
        process = subprocess.Popen(command, cwd=repo_root, env=environment, stdout=stdout, stderr=stderr)
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = process.wait()
            status = "timed_out"
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    metrics = record.get("metrics", {})
    successes = metrics.get("episode_successes", [])
    success_rate = metrics.get("success_rate")
    expected_count = kwargs["num_eval"]
    metrics_valid = (
        type(success_rate) in (int, float)
        and np.isfinite(success_rate) and 0 <= success_rate <= 100
        and len(successes) == expected_count
        and all(type(value) is bool for value in successes)
        and np.isclose(success_rate, 100 * sum(successes) / expected_count)
        and len(record.get("evaluation_inputs", {}).get("episodes_idx", [])) == expected_count
        and len(record.get("evaluation_inputs", {}).get("start_steps", [])) == expected_count
        and record.get("device") == kwargs["device"]
        and record.get("model_device") == kwargs["device"]
        and result_path.exists()
    )
    if return_code == 0:
        status = "completed" if metrics_valid else "invalid_metrics"
    return {
        "command": command,
        "result_path": str(result_path),
        "return_code": return_code,
        "success_rate": success_rate,
        "episode_successes": successes,
        "metrics_valid": bool(metrics_valid),
        "status": status,
        "elapsed_seconds": time.monotonic() - started,
        "evaluation_seconds": record.get("evaluation_seconds"),
        "timeout_seconds": timeout_seconds,
        "record_path": str(record_path),
        "stdout_path": str(results_dir / "cem_smoke.stdout.txt"),
        "stderr_path": str(results_dir / "cem_smoke.stderr.txt"),
        "seed_protocol": "Seeded Python/NumPy/PyTorch and CEM; official simulator reset seeds are recorded separately",
        "execution_device": kwargs["device"],
    }


def write_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
    return path
