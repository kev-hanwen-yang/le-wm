#!/usr/bin/env python3
"""Run explicit Phase I baseline and fresh-start simulator validation gates."""

from __future__ import annotations

import argparse
import math
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from counterfactual.baseline import (
    CacheIdentityError, check_model_determinism, choose_smoke_window,
    configure_determinism, encoded_cache_path, file_metadata, git_metadata,
    model_precision, package_versions, rebuild_mismatched_cache, run_cem_smoke,
    source_metadata, validate_encoded_cache, write_json, write_source_snapshot,
)
from counterfactual.contracts import TimeActionContract
from counterfactual.model import (
    IMAGE_PREPROCESSING_CONTRACT, action_normalizer_from_hdf5, checkpoint_path,
    load_frozen_lewm, resolve_cache_dir, validate_model_contract,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/phase1/pusht.yaml")
    parser.add_argument("--stage", choices=["baseline", "simulator"], default="baseline")
    parser.add_argument("--run-id", required=True, help="New immutable output identifier")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--skip-cem", action="store_true", help="Partial verification; cannot pass the Step 1 gate")
    parser.add_argument("--baseline-manifest", default=None,
                        help="Passing Step 1 parent manifest for the simulator stage")
    parser.add_argument(
        "--rebuild-mismatched-caches", action="store_true",
        help="Re-encode only embedding mismatches into this run's HDF5 artifacts",
    )
    return parser.parse_args(argv)


def write_summary(path: Path, manifest: dict) -> None:
    determinism = manifest.get("determinism", {})
    cem = manifest.get("cem_smoke", {})
    lines = [
        "# Phase I Step 1 — Frozen baseline", "",
        f"- Run status: `{manifest['run_status']}`",
        f"- Step 1 gate passed: `{manifest['step1_gate_passed']}`",
        f"- Git revision: `{manifest['git']['revision']}`",
        f"- Checkpoint SHA-256: `{manifest.get('checkpoint', {}).get('sha256', 'not loaded')}`",
        f"- Device: `{manifest.get('device', 'not loaded')}`",
        f"- Repeated encoding max delta: `{determinism.get('context_repeat_max_abs', 'not run')}`",
        f"- Repeated prediction max delta: `{determinism.get('prediction_repeat_max_abs', 'not run')}`",
        f"- Repeated five-step rollout max delta: `{determinism.get('rollout_repeat_max_abs', 'not run')}`",
        "", "## Cached-embedding identity", "",
        "| Split | Status | Max abs. difference | RMSE |",
        "|---|---|---|---|",
    ]
    for split, result in manifest.get("cache_validation", {}).items():
        lines.append(
            f"| {split} | {result['status']} | {result.get('max_abs_difference', 'n/a')} | {result.get('rmse', 'n/a')} |"
        )
    lines.extend([
        "", "## Official CEM smoke", "",
        f"- Status: `{cem.get('status', 'not run')}`",
        f"- Return code: `{cem.get('return_code', 'n/a')}`",
        f"- Requested episodes: `{manifest['config']['baseline']['cem_num_eval']}`",
        f"- Reported success rate (%): `{cem.get('success_rate', 'n/a')}`",
        "", "This is a functionality gate. Simulator replay is the separate Step 2 gate.",
        "The official evaluator resets without dataset seeds; process/CEM seeding alone does not prove simulator determinism.",
    ])
    if "failure" in manifest:
        lines.extend(["", "## Failure", "", f"`{manifest['failure']['type']}`: {manifest['failure']['message']}"])
    path.write_text("\n".join(lines) + "\n")


def run_baseline(args) -> int:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", args.run_id):
        raise ValueError("run-id must be a single 1–96 character identifier using letters, digits, underscores, or hyphens")
    config_path = (REPO_ROOT / args.config).resolve()
    cfg = OmegaConf.load(config_path)
    experiment, baseline = cfg.experiment, cfg.baseline
    contract = TimeActionContract(**OmegaConf.to_container(cfg.contract, resolve=True))
    if contract != TimeActionContract():
        raise ValueError("Step 1 freezes the released Push-T contract (3 frames, block 5, action 2, latent 192)")
    if int(baseline.rollout_steps) != 5:
        raise ValueError("Step 1 requires all five lookahead steps h=15..35")
    if int(baseline.cem_num_eval) != 10:
        raise ValueError("Step 1 smoke requires 10 episodes")
    if list(baseline.cache_splits) != ["train", "val", "test"]:
        raise ValueError("Step 1 requires train, val, and test cache validation")
    if int(baseline.cache_sample_count) < 1 or float(baseline.cem_timeout_seconds) <= 0:
        raise ValueError("Cache sample count and CEM timeout must be positive")
    for name in ("cache_atol", "cache_rtol", "cem_timeout_seconds"):
        if not math.isfinite(float(baseline[name])) or float(baseline[name]) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    cache_dir = resolve_cache_dir(REPO_ROOT / experiment.cache_dir).resolve()
    results_dir = (REPO_ROOT / experiment.results_dir / "phase1" / args.run_id).resolve()
    artifact_dir = cache_dir / "phase1" / args.run_id
    for directory in (results_dir, artifact_dir):
        if directory.exists():
            raise FileExistsError(f"Refusing to reuse {directory}; choose a new --run-id")
    source_state = git_metadata(REPO_ROOT)
    results_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    manifest_path = results_dir / "baseline_manifest.json"
    summary_path = results_dir / "baseline_summary.md"
    manifest = {
        "schema_version": 2,
        "stage": "phase1_step1_frozen_baseline",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id, "run_status": "running", "step1_gate_passed": False,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "config_file": file_metadata(config_path, include_hash=True),
        "command": [sys.executable, *sys.argv], "git": source_state,
        "cache_validation": {}, "cem_smoke": {"status": "not_run"},
        "output_locations": {
            "manifest": str(manifest_path), "summary": str(summary_path),
            "large_artifact_root": str(artifact_dir),
        },
    }

    def save():
        write_json(manifest_path, manifest)
        write_summary(summary_path, manifest)

    save()
    try:
        manifest["versions"] = package_versions()
        manifest["source_sha256"] = source_metadata(REPO_ROOT)
        manifest["source_sha256"][str(config_path)] = manifest["config_file"]["sha256"]
        write_source_snapshot(artifact_dir / "source_snapshot.h5", manifest["source_sha256"])
        manifest["output_locations"]["source_snapshot"] = str(artifact_dir / "source_snapshot.h5")
        manifest["deterministic_settings"] = configure_determinism(int(experiment.seed))
        dataset_path = cache_dir / f"{experiment.dataset_name}.h5"
        model_path = checkpoint_path(cache_dir, experiment.policy_name)
        manifest["dataset"] = file_metadata(dataset_path)
        manifest["checkpoint"] = file_metadata(model_path, include_hash=True)
        if manifest["checkpoint"]["sha256"] != baseline.checkpoint_sha256:
            raise ValueError("Checkpoint SHA-256 disagrees with frozen configuration")
        normalizer = action_normalizer_from_hdf5(dataset_path)
        manifest["action_normalizer"] = normalizer.as_dict()
        model = load_frozen_lewm(experiment.policy_name, cache_dir, args.device)
        device = str(next(model.parameters()).device)
        manifest["device"] = device
        manifest["precision"] = model_precision(model)
        manifest["model_contract"] = validate_model_contract(model, contract)
        manifest["contract"] = contract.as_dict()
        manifest["image_preprocessing"] = {**IMAGE_PREPROCESSING_CONTRACT, "resize_size": int(experiment.image_size)}
        window = choose_smoke_window(dataset_path, contract, baseline.smoke_episode_id, lookahead_steps=5)
        manifest["determinism"] = check_model_determinism(
            model=model, window=window, normalizer=normalizer, contract=contract,
            device=device, image_size=int(experiment.image_size), lookahead_steps=5,
            artifact_path=artifact_dir / "frozen_rollout.h5",
        )
        if not manifest["determinism"]["all_finite"] or not manifest["determinism"]["deterministic"]:
            raise RuntimeError("Repeated encoding/prediction/rollout gate failed")
        save()
        for split in baseline.cache_splits:
            path = encoded_cache_path(
                cache_dir, experiment.dataset_name, experiment.policy_name, split,
                int(baseline.cache_split_seed), int(experiment.image_size),
            )
            parameters = dict(
                dataset_path=dataset_path, model=model, device=device,
                image_size=int(experiment.image_size), sample_count=int(baseline.cache_sample_count),
                atol=float(baseline.cache_atol), rtol=float(baseline.cache_rtol),
                contract=contract, sample_seed=int(baseline.cache_split_seed),
                expected_metadata={
                    "dataset_name": experiment.dataset_name, "policy_name": experiment.policy_name,
                    "split_name": split, "seed": int(baseline.cache_split_seed),
                    "img_size": int(experiment.image_size),
                },
            )
            result = validate_encoded_cache(cache_path=path, **parameters)
            manifest["cache_validation"][split] = result
            save()
            if result["status"] == "mismatch" and args.rebuild_mismatched_caches:
                repaired = rebuild_mismatched_cache(
                    validation=result, destination=artifact_dir / f"{split}_encoded.h5",
                    dataset_path=dataset_path, model=model, device=device,
                    image_size=int(experiment.image_size),
                    checkpoint_sha256=manifest["checkpoint"]["sha256"],
                )
                repaired_result = validate_encoded_cache(cache_path=repaired, **parameters)
                repaired_result["original_validation"] = result
                manifest["cache_validation"][split] = repaired_result
                save()
        failures = [name for name, result in manifest["cache_validation"].items() if result["status"] != "match"]
        if failures:
            raise CacheIdentityError("Cache identity failed: " + ", ".join(failures))
        if args.skip_cem or not bool(baseline.cem_enabled):
            manifest["cem_smoke"] = {"status": "skipped"}
            manifest["run_status"] = "verification_only"
        else:
            print(f"Running official CEM: seed={experiment.seed}, episodes=10, device={device}; logs: {results_dir}", flush=True)
            manifest["cem_smoke"] = run_cem_smoke(
                repo_root=REPO_ROOT, cache_dir=cache_dir, policy_name=str(experiment.policy_name),
                seed=int(experiment.seed), num_eval=10, run_id=args.run_id, device=device,
                dataset_name=str(experiment.dataset_name), image_size=int(experiment.image_size),
                results_dir=results_dir, timeout_seconds=float(baseline.cem_timeout_seconds),
            )
            if manifest["cem_smoke"]["status"] != "completed":
                raise RuntimeError(f"CEM gate failed: {manifest['cem_smoke']['status']}")
            manifest["run_status"] = "completed"
            manifest["step1_gate_passed"] = True
        if file_metadata(model_path, include_hash=True)["sha256"] != manifest["checkpoint"]["sha256"]:
            raise RuntimeError("Checkpoint changed during the run")
    except Exception as error:
        manifest["run_status"] = "failed"
        manifest["step1_gate_passed"] = False
        manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
        (results_dir / "failure_traceback.txt").write_text(traceback.format_exc())
        print(f"Baseline failed: {type(error).__name__}: {error}", file=sys.stderr)
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    print(f"Baseline {manifest['run_status']}; manifest: {manifest_path}")
    return 1 if manifest["run_status"] == "failed" else 0


def main() -> int:
    args = parse_args()
    if args.stage == "simulator":
        from counterfactual.simulator_validation import run_simulator_stage
        return run_simulator_stage(args, REPO_ROOT)
    return run_baseline(args)


if __name__ == "__main__":
    raise SystemExit(main())
