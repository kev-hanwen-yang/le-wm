#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path


# Script: systematic-bias-direction latent diagnostic.
#
# This experiment asks whether rollout errors point in a consistent latent-space
# direction. MSE-to-ground-truth measures error magnitude:
#   mean_i ||z^t_i - z_t_i||^2
# This script additionally measures directional bias:
#   mu_t = mean_i(z^t_i - z_t_i)
#   ||mu_t||^2 / mean_i ||z^t_i - z_t_i||^2
#
# If the bias fraction is small, errors are mostly episode-specific and cancel
# across episodes. If it is large, a meaningful part of the error is systematic.


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib-cache"))

from probe.latent_diagnostics import (  # noqa: E402
    compute_systematic_bias_metrics,
    load_true_future_latents,
    resolve_true_future_latents_path,
    save_systematic_bias_charts,
    save_systematic_bias_report,
    systematic_bias_diagnostic_paths,
    validate_predicted_latent_artifact,
    validate_true_future_latent_artifact,
)
from probe.rollout_ablation import rollout_artifact_paths  # noqa: E402
from probe.rollout_probe_eval import load_rollout_prediction_artifact  # noqa: E402


def parse_args():
    """Parse CLI options for systematic-bias diagnostics."""
    parser = argparse.ArgumentParser(
        description="Compute systematic latent bias direction across rollout episodes."
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--dataset-name", default="pusht_expert_train")
    parser.add_argument("--policy-name", default="pusht/lewm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--raw-horizon", type=int, default=100)
    return parser.parse_args()


def default_cache_dir():
    """Prefer the project-local `.stable-wm` cache directory."""
    project_cache_dir = REPO_ROOT / ".stable-wm"
    if project_cache_dir.exists():
        return project_cache_dir

    import stable_worldmodel as swm

    return Path(swm.data.utils.get_cache_dir())


def build_metadata(args, prediction_metadata, true_latent_metadata):
    """Collect metadata saved into the systematic-bias JSON report."""
    return {
        "dataset_name": args.dataset_name,
        "policy_name": args.policy_name,
        "split_seed": args.seed,
        "img_size": args.img_size,
        "num_episodes": args.num_episodes,
        "raw_horizon": args.raw_horizon,
        "prediction_metadata": prediction_metadata,
        "true_latent_metadata": true_latent_metadata,
    }


def print_metric_summary(rows):
    """Print first and last horizon metrics for quick verification."""
    first = rows[0]
    last = rows[-1]
    print(
        "first horizon: "
        f"step={first['raw_step']} "
        f"bias_norm={first['bias_norm']:.6f} "
        f"bias_per_dim_mse={first['bias_per_dim_mse']:.6f} "
        f"total_per_dim_mse={first['total_per_dim_mse']:.6f} "
        f"bias_fraction={first['bias_fraction']:.6f}"
    )
    print(
        "last horizon: "
        f"step={last['raw_step']} "
        f"bias_norm={last['bias_norm']:.6f} "
        f"bias_per_dim_mse={last['bias_per_dim_mse']:.6f} "
        f"total_per_dim_mse={last['total_per_dim_mse']:.6f} "
        f"bias_fraction={last['bias_fraction']:.6f}"
    )


def main():
    """Run systematic-bias-direction diagnostics and save report/charts."""
    args = parse_args()
    cache_dir = args.cache_dir or default_cache_dir()

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
    if not predicted_path.exists():
        raise FileNotFoundError(
            f"Missing predicted rollout artifact: {predicted_path}. "
            "Run scripts/probe_rollout_ablation.py first."
        )

    true_latents_path = resolve_true_future_latents_path(
        cache_dir=cache_dir,
        dataset_name=args.dataset_name,
        policy_name=args.policy_name,
        split_seed=args.seed,
        img_size=args.img_size,
        num_episodes=args.num_episodes,
        raw_horizon=args.raw_horizon,
    )
    if not true_latents_path.exists():
        raise FileNotFoundError(
            f"Missing true future latent cache: {true_latents_path}. "
            "Run scripts/probe_latent_mse_diagnostics.py first."
        )

    output_paths = systematic_bias_diagnostic_paths(
        cache_dir=cache_dir,
        dataset_name=args.dataset_name,
        policy_name=args.policy_name,
        split_seed=args.seed,
        img_size=args.img_size,
        num_episodes=args.num_episodes,
        raw_horizon=args.raw_horizon,
    )

    print(f"predicted rollout artifact: {predicted_path}")
    print(f"true future latent cache: {true_latents_path}")

    predicted, prediction_metadata = load_rollout_prediction_artifact(predicted_path)
    validate_predicted_latent_artifact(predicted)
    true_artifact, true_latent_metadata = load_true_future_latents(true_latents_path)
    validate_true_future_latent_artifact(predicted, true_artifact)

    predicted_latents = predicted["predicted_latents"].float()
    true_future_latents = true_artifact["true_future_latents"].float()
    target_steps = predicted["target_steps"].long()
    print(f"predicted_latents: {tuple(predicted_latents.shape)}")
    print(f"true_future_latents: {tuple(true_future_latents.shape)}")
    print(f"target_steps: {target_steps.tolist()}")

    metrics = compute_systematic_bias_metrics(
        predicted_latents=predicted_latents,
        true_future_latents=true_future_latents,
    )
    metadata = build_metadata(args, prediction_metadata, true_latent_metadata)
    json_path = save_systematic_bias_report(
        metrics=metrics,
        predicted=predicted,
        metadata=metadata,
        json_path=output_paths["json"],
    )
    chart_paths = save_systematic_bias_charts(
        metrics=metrics,
        target_steps=target_steps,
        paths=output_paths,
    )

    import json

    with Path(json_path).open() as f:
        report = json.load(f)
    print_metric_summary(report["rows"])
    print(f"saved JSON report: {json_path}")
    print(f"saved bias norm chart: {chart_paths['bias_norm_chart']}")
    print(f"saved per-dim bias-vs-total chart: {chart_paths['per_dim_chart']}")
    print(f"saved bias fraction chart: {chart_paths['bias_fraction_chart']}")


if __name__ == "__main__":
    main()
