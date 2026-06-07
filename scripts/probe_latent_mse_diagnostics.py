#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path


# Script: MSE-to-encoded-ground-truth latent diagnostic.
#
# This script measures whether predicted rollout latents drift away from the
# encoder outputs of the actual future frames. It reuses an existing predicted
# rollout artifact, encodes the corresponding real future frames, and reports:
# - mean ||z^t - z_t||^2 by horizon
# - per-dim latent MSE by horizon


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib-cache"))

from probe.embedding_cache import get_image_transform  # noqa: E402
from probe.latent_diagnostics import (  # noqa: E402
    compute_latent_mse_metrics,
    encode_future_latents,
    latent_diagnostic_paths,
    load_true_future_latents,
    resolve_true_future_latents_path,
    save_latent_mse_charts,
    save_latent_mse_report,
    save_true_future_latents,
    validate_predicted_latent_artifact,
    verify_context_alignment,
)
from probe.rollout_ablation import (  # noqa: E402
    eval_device,
    load_released_lewm_world_model,
    rollout_artifact_paths,
)
from probe.rollout_probe_eval import load_rollout_prediction_artifact  # noqa: E402
from probe.rollout_windows import load_raw_pusht_dataset  # noqa: E402


def parse_args():
    """Parse CLI options for latent MSE diagnostics."""
    parser = argparse.ArgumentParser(
        description="Compute MSE between predicted rollout latents and encoded future frames."
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--dataset-name", default="pusht_expert_train")
    parser.add_argument("--policy-name", default="pusht/lewm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--raw-horizon", type=int, default=100)
    parser.add_argument("--episode-batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--overwrite-true-latents", action="store_true")
    return parser.parse_args()


def default_cache_dir():
    """Prefer the project-local `.stable-wm` cache directory."""
    project_cache_dir = REPO_ROOT / ".stable-wm"
    if project_cache_dir.exists():
        return project_cache_dir

    import stable_worldmodel as swm

    return Path(swm.data.utils.get_cache_dir())


def build_metadata(args, prediction_metadata, context_check):
    """Collect metadata saved into the true-latent cache and JSON report."""
    return {
        "dataset_name": args.dataset_name,
        "policy_name": args.policy_name,
        "split_seed": args.seed,
        "img_size": args.img_size,
        "num_episodes": args.num_episodes,
        "raw_horizon": args.raw_horizon,
        "episode_batch_size": args.episode_batch_size,
        "prediction_metadata": prediction_metadata,
        "context_alignment_check": context_check,
    }


def print_metric_summary(rows):
    """Print first and last horizon metrics for quick verification."""
    first = rows[0]
    last = rows[-1]
    print(
        "first horizon: "
        f"step={first['raw_step']} "
        f"sq_l2_mean={first['sq_l2_mean']:.6f} "
        f"per_dim_mse={first['per_dim_mse']:.6f}"
    )
    print(
        "last horizon: "
        f"step={last['raw_step']} "
        f"sq_l2_mean={last['sq_l2_mean']:.6f} "
        f"per_dim_mse={last['per_dim_mse']:.6f}"
    )


def main():
    """Run MSE-to-encoded-ground-truth diagnostics and save artifacts."""
    args = parse_args()
    cache_dir = args.cache_dir or default_cache_dir()
    device = eval_device() if args.device == "auto" else args.device

    rollout_paths = rollout_artifact_paths(
        cache_dir=cache_dir,
        dataset_name=args.dataset_name,
        policy_name=args.policy_name,
        split_seed=args.seed,
        img_size=args.img_size,
        num_episodes=args.num_episodes,
        raw_horizon=args.raw_horizon,
    )
    predicted_path = rollout_paths["predicted"]
    encoded_rollout_path = rollout_paths["encoded"]
    if not predicted_path.exists():
        raise FileNotFoundError(
            f"Missing predicted rollout artifact: {predicted_path}. "
            "Run scripts/probe_rollout_ablation.py first."
        )
    if not encoded_rollout_path.exists():
        raise FileNotFoundError(
            f"Missing encoded rollout artifact: {encoded_rollout_path}. "
            "Run scripts/probe_rollout_ablation.py first."
        )

    diagnostic_paths = latent_diagnostic_paths(
        cache_dir=cache_dir,
        dataset_name=args.dataset_name,
        policy_name=args.policy_name,
        split_seed=args.seed,
        img_size=args.img_size,
        num_episodes=args.num_episodes,
        raw_horizon=args.raw_horizon,
    )

    print(f"device: {device}")
    print(f"predicted rollout artifact: {predicted_path}")
    print(f"encoded rollout artifact: {encoded_rollout_path}")
    existing_true_latents_path = resolve_true_future_latents_path(
        cache_dir=cache_dir,
        dataset_name=args.dataset_name,
        policy_name=args.policy_name,
        split_seed=args.seed,
        img_size=args.img_size,
        num_episodes=args.num_episodes,
        raw_horizon=args.raw_horizon,
    )
    print(f"true future latent cache: {diagnostic_paths['true_latents']}")

    predicted, prediction_metadata = load_rollout_prediction_artifact(predicted_path)
    validate_predicted_latent_artifact(predicted)
    print(f"predicted_latents: {tuple(predicted['predicted_latents'].shape)}")
    print(f"target_steps: {predicted['target_steps'].tolist()}")

    raw_dataset = load_raw_pusht_dataset(
        dataset_name=args.dataset_name,
        cache_dir=cache_dir,
    )
    model = load_released_lewm_world_model(
        policy_name=args.policy_name,
        device=device,
        cache_dir=cache_dir,
    )
    image_transform = get_image_transform(args.img_size)

    context_check = verify_context_alignment(
        encoded_rollout_path=encoded_rollout_path,
        raw_dataset=raw_dataset,
        model=model,
        device=device,
        image_transform=image_transform,
    )
    print("context alignment check:", context_check)
    if not context_check["allclose"]:
        raise RuntimeError("Context alignment check failed; refusing to compute diagnostics.")

    metadata = build_metadata(args, prediction_metadata, context_check)
    if existing_true_latents_path.exists() and not args.overwrite_true_latents:
        true_payload, _ = load_true_future_latents(existing_true_latents_path)
        true_future_latents = true_payload["true_future_latents"]
        print(f"loaded cached true future latents: {existing_true_latents_path}")
        if existing_true_latents_path != diagnostic_paths["true_latents"]:
            save_true_future_latents(
                predicted,
                true_future_latents=true_future_latents,
                metadata=metadata,
                save_path=diagnostic_paths["true_latents"],
            )
            print(f"copied true future latents to new experiment folder: {diagnostic_paths['true_latents']}")
    else:
        true_future_latents = encode_future_latents(
            predicted,
            raw_dataset=raw_dataset,
            model=model,
            device=device,
            image_transform=image_transform,
            episode_batch_size=args.episode_batch_size,
        )
        save_true_future_latents(
            predicted,
            true_future_latents=true_future_latents,
            metadata=metadata,
            save_path=diagnostic_paths["true_latents"],
        )
        print(f"saved true future latents: {diagnostic_paths['true_latents']}")

    print(f"true_future_latents: {tuple(true_future_latents.shape)}")
    metrics = compute_latent_mse_metrics(
        predicted["predicted_latents"],
        true_future_latents,
    )
    json_path = save_latent_mse_report(
        metrics,
        predicted=predicted,
        metadata=metadata,
        json_path=diagnostic_paths["json"],
    )
    chart_paths = save_latent_mse_charts(
        metrics,
        target_steps=predicted["target_steps"].long(),
        sq_l2_path=diagnostic_paths["sq_l2_chart"],
        per_dim_mse_path=diagnostic_paths["per_dim_mse_chart"],
    )

    import json

    with Path(json_path).open() as f:
        report = json.load(f)
    print_metric_summary(report["rows"])
    print(f"saved JSON report: {json_path}")
    print(f"saved squared-L2 chart: {chart_paths['sq_l2_chart']}")
    print(f"saved per-dim MSE chart: {chart_paths['per_dim_mse_chart']}")


if __name__ == "__main__":
    main()
