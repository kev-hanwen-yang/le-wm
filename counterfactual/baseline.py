"""Step 1 baseline checks and provenance reporting for Phase I."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import h5py
import numpy as np
import torch

from counterfactual.contracts import TimeActionContract
from counterfactual.model import frames_hwc_to_b1chw, run_frozen_window
from probe.embedding_cache import get_image_transform, preprocess_pixels


def _seeded_eval_wrapper(seed: int) -> str:
    """Seed process-level RNGs before executing the unchanged official evaluator."""
    return (
        "import random, runpy, sys; import numpy as np; import torch; "
        f"random.seed({seed}); np.random.seed({seed}); torch.manual_seed({seed}); "
        f"torch.cuda.manual_seed_all({seed}); sys.argv[0] = 'eval.py'; "
        "runpy.run_path('eval.py', run_name='__main__')"
    )


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


def package_versions() -> dict:
    packages = [
        "stable-worldmodel",
        "stable-pretraining",
        "torch",
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
) -> dict:
    """Re-encode deterministic cached rows and compare them to cached embeddings."""
    cache_path = Path(cache_path)
    dataset_path = Path(dataset_path)
    if not cache_path.exists():
        return {"status": "missing", "cache": file_metadata(cache_path) if cache_path.exists() else {"path": str(cache_path)}}

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    encoded = payload.get("encoded")
    if not isinstance(encoded, dict):
        raise CacheIdentityError(f"Missing 'encoded' mapping in {cache_path}")
    required = {"emb", "episode_idx", "step_idx"}
    missing = required.difference(encoded)
    if missing:
        raise CacheIdentityError(f"Cache {cache_path} is missing keys: {sorted(missing)}")

    indices = deterministic_sample_indices(len(encoded["emb"]), sample_count)
    episode_ids = encoded["episode_idx"][indices].long().cpu().numpy()
    step_ids = encoded["step_idx"][indices].long().cpu().numpy()
    expected = encoded["emb"][indices].float().cpu()

    with h5py.File(dataset_path, "r") as handle:
        offsets = handle["ep_offset"][:]
        raw_indices = offsets[episode_ids] + step_ids
        pixels = handle["pixels"][raw_indices]

    image_transform = get_image_transform(image_size)
    model_pixels = preprocess_pixels(frames_hwc_to_b1chw(pixels), image_transform, device)
    model.eval()
    with torch.inference_mode():
        actual = model.encode({"pixels": model_pixels})["emb"][:, 0].cpu().float()

    diff = actual - expected
    max_abs = float(diff.abs().max())
    rmse = float(torch.sqrt(torch.mean(diff.square())))
    matches = bool(torch.allclose(actual, expected, atol=atol, rtol=rtol))
    return {
        "status": "match" if matches else "mismatch",
        "cache": file_metadata(cache_path),
        "cache_metadata": payload.get("metadata", {}),
        "sample_indices": indices,
        "episode_ids": episode_ids.tolist(),
        "step_ids": step_ids.tolist(),
        "raw_dataset_indices": raw_indices.tolist(),
        "atol": atol,
        "rtol": rtol,
        "max_abs_difference": max_abs,
        "rmse": rmse,
        "embedding_shape": list(encoded["emb"].shape),
    }


def choose_smoke_window(dataset_path: str | Path, contract: TimeActionContract, episode_id: int | None = None) -> dict:
    """Load a valid h=0,5,10 / action 0:15 window directly from HDF5."""
    dataset_path = Path(dataset_path)
    required_steps = contract.first_prediction_step + 1
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
        actions = handle["action"][offset : offset + contract.first_prediction_step]
    return {"episode_id": episode_id, "pixels": pixels, "raw_actions": actions}


def check_model_determinism(
    *,
    model,
    window: dict,
    normalizer,
    contract: TimeActionContract,
    device: str,
    image_size: int,
) -> dict:
    """Run identical model inputs twice and require finite, identical outputs."""
    first = run_frozen_window(
        model, window["pixels"], window["raw_actions"], normalizer, contract, device, image_size
    )
    second = run_frozen_window(
        model, window["pixels"], window["raw_actions"], normalizer, contract, device, image_size
    )
    context_delta = first["context_emb"] - second["context_emb"]
    prediction_delta = first["prediction"] - second["prediction"]
    return {
        "episode_id": window["episode_id"],
        "context_embedding_shape": list(first["context_emb"].shape),
        "action_token_shape": list(first["action_tokens"].shape),
        "action_embedding_shape": list(first["action_emb"].shape),
        "prediction_shape": list(first["prediction"].shape),
        "context_repeat_max_abs": float(context_delta.abs().max()),
        "prediction_repeat_max_abs": float(prediction_delta.abs().max()),
        "all_finite": bool(
            torch.isfinite(first["context_emb"]).all()
            and torch.isfinite(first["prediction"]).all()
        ),
        "deterministic": bool(torch.equal(first["context_emb"], second["context_emb"]) and torch.equal(first["prediction"], second["prediction"])),
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
) -> tuple[list[str], Path]:
    """Build the unmodified official evaluator invocation for a small smoke test."""
    cache_dir = Path(cache_dir).resolve()
    result_path = cache_dir / "phase1" / run_id / "cem_smoke.txt"
    output_filename = Path("..") / "phase1" / run_id / "cem_smoke.txt"
    command = [
        sys.executable,
        "-c",
        _seeded_eval_wrapper(seed),
        "--config-name=pusht.yaml",
        f"policy={policy_name}",
        # The repository's launcher config contains a machine-specific absolute
        # path.  A relative override is Hydra-safe even when this checkout path
        # has spaces, and resolves to the same project-local cache as STABLEWM_HOME.
        "cache_dir=.stable-wm",
        # The released evaluation config defaults to CUDA.  This checkout has
        # no CUDA backend, so this changes execution placement only.
        "solver.device=cpu",
        f"seed={seed}",
        f"eval.num_eval={num_eval}",
        f"output.filename={output_filename.as_posix()}",
        f"hydra.run.dir=.stable-wm/phase1/{run_id}/hydra",
    ]
    return command, result_path


def run_cem_smoke(**kwargs) -> dict:
    """Execute the official evaluator once; failures are recorded, never retried."""
    command, result_path = build_cem_smoke_command(**kwargs)
    repo_root = Path(kwargs["repo_root"])
    cache_dir = Path(kwargs["cache_dir"]).resolve()
    environment = os.environ.copy()
    environment["STABLEWM_HOME"] = str(cache_dir)
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    result_text = result_path.read_text() if result_path.exists() else ""
    success_matches = re.findall(r"'success_rate': ([0-9.]+)", result_text)
    return {
        "command": command,
        "result_path": str(result_path),
        "return_code": completed.returncode,
        "success_rate": float(success_matches[-1]) if success_matches else None,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "status": "completed" if completed.returncode == 0 else "failed",
        "seed_protocol": "random, NumPy, and PyTorch seeded in a wrapper; eval.py source unchanged",
        "execution_device_override": "cpu (required because CUDA is unavailable; cost and CEM settings unchanged)",
    }


def write_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
