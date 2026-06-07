#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path


# Script: latent norm trajectory diagnostic.
#
# This experiment checks whether predicted rollout latents stay at a normal
# magnitude as horizon increases. It compares:
# - predicted latent norm ||z^t||
# - true future encoded latent norm ||z_t||
# - full train/val encoded-latent norm distribution
#
# Norm trajectory is a scale diagnostic. It can reveal shrinkage, growth, or
# collapse even when a predicted latent still points in a plausible direction.


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib-cache"))

from probe.embedding_cache import encoded_cache_path  # noqa: E402
from probe.latent_diagnostics import (  # noqa: E402
    compute_norm_trajectory_metrics,
    compute_reference_norm_stats,
    load_true_future_latents,
    norm_trajectory_diagnostic_paths,
    resolve_true_future_latents_path,
    save_norm_trajectory_charts,
    save_norm_trajectory_report,
    validate_predicted_latent_artifact,
    validate_true_future_latent_artifact,
)
from probe.rollout_ablation import rollout_artifact_paths  # noqa: E402
from probe.rollout_probe_eval import load_rollout_prediction_artifact  # noqa: E402


def parse_args():
    """Parse CLI options for latent norm trajectory diagnostics."""
    parser = argparse.ArgumentParser(
        description="Compute predicted latent norm trajectory against true and train/val norms."
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--dataset-name", default="pusht_expert_train")
    parser.add_argument("--policy-name", default="pusht/lewm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--raw-horizon", type=int, default=100)
    parser.add_argument("--reference-splits", nargs="+", default=["train", "val"])
    parser.add_argument("--norm-chunk-size", type=int, default=100000)
    return parser.parse_args()


def default_cache_dir():
    """Prefer the project-local `.stable-wm` cache directory."""
    project_cache_dir = REPO_ROOT / ".stable-wm"
    if project_cache_dir.exists():
        return project_cache_dir

    import stable_worldmodel as swm

    return Path(swm.data.utils.get_cache_dir())


def build_reference_paths(args, cache_dir):
    """Resolve encoded train/val split paths used for reference norm statistics."""
    paths = []
    for split in args.reference_splits:
        path = encoded_cache_path(
            cache_dir,
            args.dataset_name,
            args.policy_name,
            split,
            args.seed,
            args.img_size,
        )
        if not path.exists():
            raise FileNotFoundError(
                f"Missing encoded reference split: {path}. "
                "Run the probe embedding-cache extraction/training pipeline first."
            )
        paths.append(path)
    return paths


def build_metadata(args, prediction_metadata, true_latent_metadata, reference_paths):
    """Collect metadata saved into the norm trajectory JSON report."""
    return {
        "dataset_name": args.dataset_name,
        "policy_name": args.policy_name,
        "split_seed": args.seed,
        "img_size": args.img_size,
        "num_episodes": args.num_episodes,
        "raw_horizon": args.raw_horizon,
        "reference_splits": args.reference_splits,
        "reference_paths": [str(path) for path in reference_paths],
        "norm_chunk_size": args.norm_chunk_size,
        "prediction_metadata": prediction_metadata,
        "true_latent_metadata": true_latent_metadata,
    }


def print_metric_summary(rows, reference_norm_stats):
    """Print first/last horizon metrics and reference stats for quick verification."""
    first = rows[0]
    last = rows[-1]
    reference = reference_norm_stats["overall"]
    print(
        "reference norm stats: "
        f"count={reference['count']} "
        f"mean={reference['mean']:.6f} "
        f"std={reference['std']:.6f} "
        f"p05={reference['p05']:.6f} "
        f"p95={reference['p95']:.6f}"
    )
    print(
        "first horizon: "
        f"step={first['raw_step']} "
        f"pred_norm={first['pred_norm_mean']:.6f} "
        f"true_norm={first['true_norm_mean']:.6f} "
        f"ratio={first['norm_ratio']:.6f} "
        f"pred_z={first['pred_norm_zscore']:.6f}"
    )
    print(
        "last horizon: "
        f"step={last['raw_step']} "
        f"pred_norm={last['pred_norm_mean']:.6f} "
        f"true_norm={last['true_norm_mean']:.6f} "
        f"ratio={last['norm_ratio']:.6f} "
        f"pred_z={last['pred_norm_zscore']:.6f}"
    )


def main():
    """Run latent norm trajectory diagnostics and save report/charts."""
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

    reference_paths = build_reference_paths(args, cache_dir)
    output_paths = norm_trajectory_diagnostic_paths(
        cache_dir=cache_dir,
        dataset_name=args.dataset_name,
        policy_name=args.policy_name,
        split_seed=args.seed,
        img_size=args.img_size,
        num_episodes=args.num_episodes,
        raw_horizon=args.raw_horizon,
        reference_splits=args.reference_splits,
    )

    print(f"predicted rollout artifact: {predicted_path}")
    print(f"true future latent cache: {true_latents_path}")
    print("reference norm splits:")
    for path in reference_paths:
        print(f"  - {path}")

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

    reference_norm_stats = compute_reference_norm_stats(
        reference_split_paths=reference_paths,
        norm_chunk_size=args.norm_chunk_size,
    )
    metrics = compute_norm_trajectory_metrics(
        predicted_latents=predicted_latents,
        true_future_latents=true_future_latents,
        reference_norm_stats=reference_norm_stats,
    )
    metadata = build_metadata(args, prediction_metadata, true_latent_metadata, reference_paths)
    json_path = save_norm_trajectory_report(
        metrics=metrics,
        predicted=predicted,
        reference_norm_stats=reference_norm_stats,
        metadata=metadata,
        json_path=output_paths["json"],
    )
    chart_paths = save_norm_trajectory_charts(
        metrics=metrics,
        target_steps=target_steps,
        reference_norm_stats=reference_norm_stats,
        paths=output_paths,
    )

    import json

    with Path(json_path).open() as f:
        report = json.load(f)
    print_metric_summary(report["rows"], report["reference_norm_stats"])
    print(f"saved JSON report: {json_path}")
    print(f"saved predicted-vs-true norm chart: {chart_paths['pred_vs_true_chart']}")
    print(f"saved reference band chart: {chart_paths['reference_band_chart']}")
    print(f"saved norm ratio chart: {chart_paths['norm_ratio_chart']}")
    print(f"saved norm z-score chart: {chart_paths['norm_zscore_chart']}")


if __name__ == "__main__":
    main()
