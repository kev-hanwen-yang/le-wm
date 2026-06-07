#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path


# Script: min-distance-to-manifold latent diagnostic.
#
# This script measures whether predicted rollout latents stay near the empirical
# encoded-latent manifold. For each predicted latent z^t, it computes:
#   min_m ||z^t - m||^2
# where m ranges over encoded train/val latents from the frozen LeWM encoder.
#
# It also computes the same quantity for the true future latents z_t. That
# baseline matters because real encoded frames are not necessarily identical to
# any reference latent unless the exact same frame is in the reference set.
#
# The computation is exact but memory-efficient: it scans train/val encoded
# latents in chunks and keeps only the best distance/index for each query.


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib-cache"))

from probe.embedding_cache import encoded_cache_path  # noqa: E402
from probe.latent_diagnostics import (  # noqa: E402
    aggregate_manifold_distance_metrics,
    compute_min_distance_to_encoded_splits,
    flatten_rollout_latents,
    load_true_future_latents,
    manifold_diagnostic_paths,
    reshape_nearest_record,
    resolve_true_future_latents_path,
    save_manifold_distance_charts,
    save_manifold_distance_report,
    validate_predicted_latent_artifact,
    validate_true_future_latent_artifact,
)
from probe.rollout_ablation import eval_device, rollout_artifact_paths  # noqa: E402
from probe.rollout_probe_eval import load_rollout_prediction_artifact  # noqa: E402


def parse_args():
    """Parse CLI options for manifold-distance diagnostics."""
    parser = argparse.ArgumentParser(
        description="Compute min distance from rollout latents to encoded train/val manifold."
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--dataset-name", default="pusht_expert_train")
    parser.add_argument("--policy-name", default="pusht/lewm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--raw-horizon", type=int, default=100)
    parser.add_argument("--reference-splits", nargs="+", default=["train", "val"])
    parser.add_argument("--query-chunk-size", type=int, default=256)
    parser.add_argument("--reference-chunk-size", type=int, default=20000)
    parser.add_argument(
        "--reference-max-latents-per-split",
        type=int,
        default=None,
        help=(
            "Optional deterministic reference subset size for smoke tests. "
            "Leave unset for exact full-manifold results."
        ),
    )
    parser.add_argument("--reference-sample-seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def default_cache_dir():
    """Prefer the project-local `.stable-wm` cache directory."""
    project_cache_dir = REPO_ROOT / ".stable-wm"
    if project_cache_dir.exists():
        return project_cache_dir

    import stable_worldmodel as swm

    return Path(swm.data.utils.get_cache_dir())


def true_future_latent_path(cache_dir, dataset_name, policy_name, seed, img_size, num_episodes, raw_horizon):
    """Return the true-future-latent cache produced by the latent MSE script."""
    return resolve_true_future_latents_path(
        cache_dir=cache_dir,
        dataset_name=dataset_name,
        policy_name=policy_name,
        split_seed=seed,
        img_size=img_size,
        num_episodes=num_episodes,
        raw_horizon=raw_horizon,
    )


def build_reference_paths(args, cache_dir):
    """Resolve encoded-cache paths for train/val reference manifold splits."""
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


def build_metadata(args, prediction_metadata, reference_paths):
    """Collect metadata saved into the manifold-distance JSON report."""
    return {
        "dataset_name": args.dataset_name,
        "policy_name": args.policy_name,
        "split_seed": args.seed,
        "img_size": args.img_size,
        "num_episodes": args.num_episodes,
        "raw_horizon": args.raw_horizon,
        "reference_splits": args.reference_splits,
        "reference_paths": [str(path) for path in reference_paths],
        "reference_max_latents_per_split": args.reference_max_latents_per_split,
        "reference_sample_seed": args.reference_sample_seed,
        "query_chunk_size": args.query_chunk_size,
        "reference_chunk_size": args.reference_chunk_size,
        "prediction_metadata": prediction_metadata,
    }


def print_metric_summary(rows):
    """Print first and last horizon metrics for quick verification."""
    first = rows[0]
    last = rows[-1]
    print(
        "first horizon: "
        f"step={first['raw_step']} "
        f"pred_min_sq_l2={first['pred_min_sq_l2_mean']:.6f} "
        f"true_min_sq_l2={first['true_min_sq_l2_mean']:.6f} "
        f"ratio={first['manifold_drift_ratio']:.3f}"
    )
    print(
        "last horizon: "
        f"step={last['raw_step']} "
        f"pred_min_sq_l2={last['pred_min_sq_l2_mean']:.6f} "
        f"true_min_sq_l2={last['true_min_sq_l2_mean']:.6f} "
        f"ratio={last['manifold_drift_ratio']:.3f}"
    )


def main():
    """Run min-distance-to-manifold diagnostics and save report/charts."""
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
    if not predicted_path.exists():
        raise FileNotFoundError(
            f"Missing predicted rollout artifact: {predicted_path}. "
            "Run scripts/probe_rollout_ablation.py first."
        )

    true_latents_path = true_future_latent_path(
        cache_dir=cache_dir,
        dataset_name=args.dataset_name,
        policy_name=args.policy_name,
        seed=args.seed,
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
    output_paths = manifold_diagnostic_paths(
        cache_dir=cache_dir,
        dataset_name=args.dataset_name,
        policy_name=args.policy_name,
        split_seed=args.seed,
        img_size=args.img_size,
        num_episodes=args.num_episodes,
        raw_horizon=args.raw_horizon,
        reference_splits=args.reference_splits,
        reference_max_latents_per_split=args.reference_max_latents_per_split,
    )

    print(f"device: {device}")
    print(f"predicted rollout artifact: {predicted_path}")
    print(f"true future latent cache: {true_latents_path}")
    print("reference manifold splits:")
    for path in reference_paths:
        print(f"  - {path}")

    predicted, prediction_metadata = load_rollout_prediction_artifact(predicted_path)
    validate_predicted_latent_artifact(predicted)
    true_artifact, true_metadata = load_true_future_latents(true_latents_path)
    validate_true_future_latent_artifact(predicted, true_artifact)

    pred_latents = predicted["predicted_latents"].float()
    true_latents = true_artifact["true_future_latents"].float()
    batch_size, horizon_count, latent_dim = pred_latents.shape
    target_steps = predicted["target_steps"].long()
    print(f"predicted_latents: {tuple(pred_latents.shape)}")
    print(f"true_future_latents: {tuple(true_latents.shape)}")
    print(f"target_steps: {target_steps.tolist()}")

    pred_queries = flatten_rollout_latents(pred_latents)
    true_queries = flatten_rollout_latents(true_latents)
    print(f"predicted query matrix: {tuple(pred_queries.shape)}")
    print(f"true query matrix: {tuple(true_queries.shape)}")

    print("computing predicted latent min distances...")
    pred_nearest_flat = compute_min_distance_to_encoded_splits(
        query_latents=pred_queries,
        reference_split_paths=reference_paths,
        device=device,
        query_chunk_size=args.query_chunk_size,
        reference_chunk_size=args.reference_chunk_size,
        max_reference_latents_per_split=args.reference_max_latents_per_split,
        reference_sample_seed=args.reference_sample_seed,
    )
    print("computing true latent baseline min distances...")
    true_nearest_flat = compute_min_distance_to_encoded_splits(
        query_latents=true_queries,
        reference_split_paths=reference_paths,
        device=device,
        query_chunk_size=args.query_chunk_size,
        reference_chunk_size=args.reference_chunk_size,
        max_reference_latents_per_split=args.reference_max_latents_per_split,
        reference_sample_seed=args.reference_sample_seed,
    )

    pred_nearest = reshape_nearest_record(pred_nearest_flat, batch_size, horizon_count)
    true_nearest = reshape_nearest_record(true_nearest_flat, batch_size, horizon_count)
    metrics = aggregate_manifold_distance_metrics(
        pred_nearest=pred_nearest,
        true_nearest=true_nearest,
        target_steps=target_steps,
        latent_dim=latent_dim,
    )

    metadata = build_metadata(args, prediction_metadata, reference_paths)
    metadata["true_latent_metadata"] = true_metadata
    json_path = save_manifold_distance_report(
        metrics=metrics,
        pred_nearest=pred_nearest,
        true_nearest=true_nearest,
        metadata=metadata,
        json_path=output_paths["json"],
    )
    chart_paths = save_manifold_distance_charts(
        metrics=metrics,
        target_steps=target_steps,
        paths=output_paths,
    )

    print_metric_summary(metrics["rows"])
    print(f"saved JSON report: {json_path}")
    print(f"saved min squared-L2 chart: {chart_paths['min_sq_l2_chart']}")
    print(f"saved min per-dim MSE chart: {chart_paths['min_per_dim_chart']}")
    print(f"saved ratio chart: {chart_paths['ratio_chart']}")


if __name__ == "__main__":
    main()
