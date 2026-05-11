#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path

import torch


# Script: apply trained probes to predicted rollout latents.
#
# This script consumes the predicted-latent artifact from
# `scripts/probe_rollout_ablation.py`, applies one linear and one MLP probe for
# each physical quantity, and saves horizon-wise metrics plus charts.


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep Matplotlib from trying to write to ~/.matplotlib on machines where the
# home directory is not writable in the current environment.
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib-cache"))

from probe.rollout_ablation import eval_device, rollout_artifact_paths  # noqa: E402
from probe.rollout_probe_eval import (  # noqa: E402
    DEFAULT_PROBE_TYPES,
    DEFAULT_TARGET_NAMES,
    evaluate_rollout_probe_grid,
    load_rollout_prediction_artifact,
    save_rollout_probe_report,
)


def parse_args():
    """Parse CLI options for horizon-wise rollout probe evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained probes on open-loop predicted rollout latents."
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--dataset-name", default="pusht_expert_train")
    parser.add_argument("--policy-name", default="pusht/lewm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--raw-horizon", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGET_NAMES))
    parser.add_argument("--probe-types", nargs="+", default=list(DEFAULT_PROBE_TYPES))
    parser.add_argument("--report-name", default=None)
    return parser.parse_args()


def default_cache_dir():
    """Prefer the project-local `.stable-wm` directory for report inputs/outputs."""
    project_cache_dir = REPO_ROOT / ".stable-wm"
    if project_cache_dir.exists():
        return project_cache_dir

    import stable_worldmodel as swm

    return Path(swm.data.utils.get_cache_dir())


def build_report_metadata(args, prediction_metadata, predicted):
    """Collect compact metadata saved into the JSON and Markdown reports."""
    return {
        "dataset_name": args.dataset_name,
        "policy_name": args.policy_name,
        "split_seed": args.seed,
        "img_size": args.img_size,
        "num_episodes": int(predicted["predicted_latents"].shape[0]),
        "raw_horizon": args.raw_horizon,
        "target_steps": predicted["target_steps"].tolist(),
        "target_names": args.targets,
        "probe_types": args.probe_types,
        "prediction_metadata": prediction_metadata,
    }


def print_final_horizon_summary(rows):
    """Print the final-horizon metrics so command output is immediately useful."""
    final_step = max(row["raw_step"] for row in rows)
    print(f"Final horizon step: {final_step}")
    for row in rows:
        if row["raw_step"] != final_step:
            continue
        print(
            f"{row['probe_type']:6s} {row['target_name']:15s} "
            f"norm_mse={row['norm_mse']:.6f} "
            f"raw_rmse={row['raw_rmse']:.6f} "
            f"r={row['r_mean']:.6f}"
        )


def main():
    """Run the rollout probe report pipeline."""
    args = parse_args()
    cache_dir = args.cache_dir or default_cache_dir()
    device = eval_device() if args.device == "auto" else args.device

    paths = rollout_artifact_paths(
        cache_dir=cache_dir,
        dataset_name=args.dataset_name,
        policy_name=args.policy_name,
        split_seed=args.seed,
        img_size=args.img_size,
        num_episodes=args.num_episodes,
        raw_horizon=args.raw_horizon,
    )
    predicted_path = paths["predicted"]
    if not predicted_path.exists():
        raise FileNotFoundError(
            f"Missing predicted rollout artifact: {predicted_path}. "
            "Run scripts/probe_rollout_ablation.py first."
        )

    print(f"device: {device}")
    print(f"predicted rollout artifact: {predicted_path}")
    predicted, prediction_metadata = load_rollout_prediction_artifact(predicted_path)
    print(
        "predicted_latents:",
        tuple(predicted["predicted_latents"].shape),
        predicted["predicted_latents"].dtype,
    )
    print("target_states:", tuple(predicted["target_states"].shape), predicted["target_states"].dtype)
    print("target_steps:", predicted["target_steps"].tolist())

    rows = evaluate_rollout_probe_grid(
        predicted,
        cache_dir=cache_dir,
        device=device,
        batch_size=args.batch_size,
        target_names=tuple(args.targets),
        probe_types=tuple(args.probe_types),  # default order: linear first, then MLP.
    )
    metadata = build_report_metadata(args, prediction_metadata, predicted)

    report_name = args.report_name
    if report_name is None:
        report_name = (
            f"rollout_probe_report_split_seed{args.seed}_"
            f"episodes{args.num_episodes}_horizon{args.raw_horizon}_img{args.img_size}"
        )
    report_paths = save_rollout_probe_report(
        rows,
        metadata=metadata,
        output_dir=cache_dir / "probes" / "reports",
        report_name=report_name,
    )

    print_final_horizon_summary(rows)
    print(f"Saved Markdown report to {report_paths['markdown']}")
    print(f"Saved JSON report to {report_paths['json']}")
    print(f"Saved normalized MSE chart to {report_paths['norm_mse_chart']}")
    print(f"Saved Pearson r chart to {report_paths['r_chart']}")
    print(f"Saved raw RMSE chart to {report_paths['raw_rmse_chart']}")


if __name__ == "__main__":
    main()
