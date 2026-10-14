#!/usr/bin/env python3
"""Staged command-line entry point for the LeWM Push-T Phase I experiment.

Only the frozen-baseline stage is implemented here.  Future stages must remain
explicit additions rather than silently running counterfactual or planning work.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from counterfactual.baseline import (
    CacheIdentityError,
    check_model_determinism,
    choose_smoke_window,
    encoded_cache_path,
    file_metadata,
    git_metadata,
    model_precision,
    package_versions,
    run_cem_smoke,
    validate_encoded_cache,
    write_json,
)
from counterfactual.contracts import TimeActionContract
from counterfactual.model import (
    action_normalizer_from_hdf5,
    checkpoint_path,
    IMAGE_PREPROCESSING_CONTRACT,
    load_frozen_lewm,
    resolve_cache_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/phase1/pusht.yaml")
    parser.add_argument("--stage", choices=["baseline"], default="baseline")
    parser.add_argument("--run-id", required=True, help="Immutable output identifier")
    parser.add_argument("--device", default=None, help="cuda, mps, or cpu; default auto-detect")
    parser.add_argument("--skip-cem", action="store_true", help="Run contracts/cache checks only")
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Append a new baseline record only when the run directory already exists.",
    )
    return parser.parse_args()


def write_summary(path: Path, manifest: dict) -> None:
    determinism = manifest["determinism"]
    cache_rows = manifest["cache_validation"]
    cem = manifest["cem_smoke"]
    lines = [
        "# Phase I Step 1 — Frozen baseline",
        "",
        f"- Git revision: `{manifest['git']['revision']}`",
        f"- Checkpoint SHA-256: `{manifest['checkpoint']['sha256']}`",
        f"- Device: `{manifest['device']}`",
        f"- Repeated encoding max absolute delta: `{determinism['context_repeat_max_abs']}`",
        f"- Repeated prediction max absolute delta: `{determinism['prediction_repeat_max_abs']}`",
        f"- Deterministic / finite: `{determinism['deterministic']}` / `{determinism['all_finite']}`",
        "",
        "## Cached-embedding identity",
        "",
        "| Split | Status | Max abs. difference | RMSE |",
        "|---|---:|---:|---:|",
    ]
    for split, result in cache_rows.items():
        lines.append(
            f"| {split} | {result['status']} | {result.get('max_abs_difference', 'n/a')} | {result.get('rmse', 'n/a')} |"
        )
    lines.extend(
        [
            "",
            "## Official CEM smoke",
            "",
            f"- Status: `{cem['status']}`",
            f"- Return code: `{cem['return_code']}`",
            f"- Fixed-seed episodes: `{manifest['cem_num_eval']}`",
            f"- Reported success rate: `{cem['success_rate']}`",
            "",
            "This smoke uses the existing evaluator and is not a physical counterfactual result.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def run_baseline(args) -> int:
    config_path = (REPO_ROOT / args.config).resolve()
    cfg = OmegaConf.load(config_path)
    experiment = cfg.experiment
    baseline = cfg.baseline
    contract = TimeActionContract(**OmegaConf.to_container(cfg.contract, resolve=True))
    cache_dir = resolve_cache_dir(REPO_ROOT / experiment.cache_dir)
    results_dir = (REPO_ROOT / experiment.results_dir / "phase1" / args.run_id).resolve()
    artifact_dir = (cache_dir / "phase1" / args.run_id).resolve()

    if results_dir.exists() and not args.allow_existing_output:
        raise FileExistsError(
            f"Refusing to overwrite {results_dir}. Choose a new --run-id or pass "
            "--allow-existing-output after reviewing the existing manifest."
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = cache_dir / f"{experiment.dataset_name}.h5"
    model_path = checkpoint_path(cache_dir, experiment.policy_name)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset is missing: {dataset_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint is missing: {model_path}")

    normalizer = action_normalizer_from_hdf5(dataset_path)
    model = load_frozen_lewm(experiment.policy_name, cache_dir, args.device)
    device = str(next(model.parameters()).device)
    window = choose_smoke_window(dataset_path, contract, baseline.smoke_episode_id)
    determinism = check_model_determinism(
        model=model,
        window=window,
        normalizer=normalizer,
        contract=contract,
        device=device,
        image_size=int(experiment.image_size),
    )
    if not determinism["all_finite"] or not determinism["deterministic"]:
        raise RuntimeError(f"Frozen model determinism gate failed: {determinism}")

    cache_validation = {}
    for split in baseline.cache_splits:
        path = encoded_cache_path(
            cache_dir,
            experiment.dataset_name,
            experiment.policy_name,
            split,
            int(baseline.cache_split_seed),
            int(experiment.image_size),
        )
        cache_validation[split] = validate_encoded_cache(
            cache_path=path,
            dataset_path=dataset_path,
            model=model,
            device=device,
            image_size=int(experiment.image_size),
            sample_count=int(baseline.cache_sample_count),
            atol=float(baseline.cache_atol),
            rtol=float(baseline.cache_rtol),
        )

    def make_manifest(cem_smoke: dict, cem_enabled: bool, run_status: str) -> dict:
        return {
            "stage": "phase1_step1_frozen_baseline",
            "run_status": run_status,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": args.run_id,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "command": sys.argv,
            "git": git_metadata(REPO_ROOT),
            "device": device,
            "precision": model_precision(model),
            "versions": package_versions(),
            "checkpoint": file_metadata(model_path, include_hash=True),
            "dataset": file_metadata(dataset_path),
            "image_preprocessing": {
                **IMAGE_PREPROCESSING_CONTRACT,
                "resize_size": int(experiment.image_size),
            },
            "contract": contract.as_dict(),
            "action_normalizer": normalizer.as_dict(),
            "determinism": determinism,
            "cache_validation": cache_validation,
            "cem_num_eval": int(baseline.cem_num_eval) if cem_enabled else 0,
            "cem_smoke": cem_smoke,
            "cem_seed_protocol": "random, NumPy, and PyTorch seeded before the official evaluator",
            "output_locations": {
                "manifest": str(results_dir / "baseline_manifest.json"),
                "summary": str(results_dir / "baseline_summary.md"),
                "large_artifact_root": str(artifact_dir),
            },
        }

    mismatches = [split for split, result in cache_validation.items() if result["status"] != "match"]
    if mismatches:
        manifest = make_manifest(
            {
                "status": "not_run_cache_identity_failed",
                "return_code": None,
                "success_rate": None,
                "command": None,
                "result_path": None,
            },
            cem_enabled=False,
            run_status="cache_identity_failed",
        )
        write_json(results_dir / "baseline_manifest.json", manifest)
        write_summary(results_dir / "baseline_summary.md", manifest)
        raise CacheIdentityError(
            "Encoded cache identity failed for " + ", ".join(mismatches) + ". "
            "The failure manifest was written. Do not use these caches for Phase I; "
            "rebuild only the failing cache after reviewing that manifest."
        )

    cem_enabled = bool(baseline.cem_enabled) and not args.skip_cem
    if cem_enabled:
        cem_smoke = run_cem_smoke(
            repo_root=REPO_ROOT,
            cache_dir=cache_dir,
            policy_name=str(experiment.policy_name),
            seed=int(experiment.seed),
            num_eval=int(baseline.cem_num_eval),
            run_id=args.run_id,
        )
        (results_dir / "cem_smoke.stdout.txt").write_text(cem_smoke.pop("stdout"))
        (results_dir / "cem_smoke.stderr.txt").write_text(cem_smoke.pop("stderr"))
    else:
        cem_smoke = {
            "status": "skipped",
            "return_code": None,
            "success_rate": None,
            "command": None,
            "result_path": None,
        }

    manifest = make_manifest(
        cem_smoke,
        cem_enabled=cem_enabled,
        run_status="completed" if cem_smoke["status"] != "failed" else "cem_smoke_failed",
    )
    manifest_path = write_json(results_dir / "baseline_manifest.json", manifest)
    write_summary(results_dir / "baseline_summary.md", manifest)
    print(f"Wrote baseline manifest: {manifest_path}")
    print(f"Wrote baseline summary: {results_dir / 'baseline_summary.md'}")

    if cem_enabled and cem_smoke["return_code"] != 0:
        return cem_smoke["return_code"] or 1
    return 0


def main() -> int:
    args = parse_args()
    if args.stage != "baseline":
        raise ValueError(f"Step 1 implementation only supports baseline, got {args.stage}")
    return run_baseline(args)


if __name__ == "__main__":
    raise SystemExit(main())
