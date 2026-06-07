#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path


# Script: temporal straightness latent diagnostic.
#
# This experiment follows the LeWM paper's velocity-based straightness formula:
#   v_t = z_{t+1} - z_t
#   straightness = mean cos(v_t, v_{t+1})
#
# It compares predicted rollout latents against true future encoded latents.
# The cosine horizon is labeled by the middle latent in each turn, e.g.
# cos(z20-z15, z25-z20) is plotted at raw step 20.


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib-cache"))

from probe.latent_diagnostics import (  # noqa: E402
    compute_temporal_straightness_metrics,
    load_true_future_latents,
    resolve_true_future_latents_path,
    save_temporal_straightness_charts,
    save_temporal_straightness_report,
    temporal_straightness_diagnostic_paths,
    validate_predicted_latent_artifact,
    validate_true_future_latent_artifact,
)
from probe.rollout_ablation import rollout_artifact_paths  # noqa: E402
from probe.rollout_probe_eval import load_rollout_prediction_artifact  # noqa: E402


def parse_args():
    """Parse CLI options for temporal-straightness diagnostics."""
    parser = argparse.ArgumentParser(
        description="Compute velocity-based temporal straightness for rollout latents."
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--dataset-name", default="pusht_expert_train")
    parser.add_argument("--policy-name", default="pusht/lewm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--raw-horizon", type=int, default=100)
    parser.add_argument("--eps", type=float, default=1e-8)
    return parser.parse_args()


def default_cache_dir():
    """Prefer the project-local `.stable-wm` cache directory."""
    project_cache_dir = REPO_ROOT / ".stable-wm"
    if project_cache_dir.exists():
        return project_cache_dir

    import stable_worldmodel as swm

    return Path(swm.data.utils.get_cache_dir())


def build_metadata(args, prediction_metadata, true_latent_metadata):
    """Collect metadata saved into the temporal-straightness JSON report."""
    return {
        "dataset_name": args.dataset_name,
        "policy_name": args.policy_name,
        "split_seed": args.seed,
        "img_size": args.img_size,
        "num_episodes": args.num_episodes,
        "raw_horizon": args.raw_horizon,
        "eps": args.eps,
        "prediction_metadata": prediction_metadata,
        "true_latent_metadata": true_latent_metadata,
    }


def print_metric_summary(report):
    """Print global and endpoint metrics for quick verification."""
    summary = report["global_summary"]
    first = report["straightness_rows"][0]
    last = report["straightness_rows"][-1]
    first_velocity = report["velocity_rows"][0]
    last_velocity = report["velocity_rows"][-1]
    print(
        "global straightness: "
        f"pred={summary['pred_global_straightness_mean']:.6f} "
        f"true={summary['true_global_straightness_mean']:.6f}"
    )
    print(
        "first turn: "
        f"step={first['raw_step']} "
        f"pred={first['pred_straightness_mean']:.6f} "
        f"true={first['true_straightness_mean']:.6f}"
    )
    print(
        "last turn: "
        f"step={last['raw_step']} "
        f"pred={last['pred_straightness_mean']:.6f} "
        f"true={last['true_straightness_mean']:.6f}"
    )
    print(
        "velocity norm endpoints: "
        f"step={first_velocity['raw_step']} pred={first_velocity['pred_velocity_norm_mean']:.6f} "
        f"true={first_velocity['true_velocity_norm_mean']:.6f}; "
        f"step={last_velocity['raw_step']} pred={last_velocity['pred_velocity_norm_mean']:.6f} "
        f"true={last_velocity['true_velocity_norm_mean']:.6f}"
    )


def main():
    """Run temporal-straightness diagnostics and save report/charts."""
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

    output_paths = temporal_straightness_diagnostic_paths(
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
    print(f"turn_steps: {target_steps[1:-1].tolist()}")
    print(f"velocity_steps: {target_steps[1:].tolist()}")

    metrics = compute_temporal_straightness_metrics(
        predicted_latents=predicted_latents,
        true_future_latents=true_future_latents,
        eps=args.eps,
    )
    metadata = build_metadata(args, prediction_metadata, true_latent_metadata)
    json_path = save_temporal_straightness_report(
        metrics=metrics,
        predicted=predicted,
        metadata=metadata,
        json_path=output_paths["json"],
    )
    chart_paths = save_temporal_straightness_charts(
        metrics=metrics,
        target_steps=target_steps,
        paths=output_paths,
    )

    import json

    with Path(json_path).open() as f:
        report = json.load(f)
    print_metric_summary(report)
    print(f"saved JSON report: {json_path}")
    print(f"saved straightness chart: {chart_paths['straightness_chart']}")
    print(f"saved cosine histogram: {chart_paths['cosine_histogram']}")
    print(f"saved velocity norm chart: {chart_paths['velocity_norm_chart']}")


if __name__ == "__main__":
    main()
