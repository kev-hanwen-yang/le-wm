#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path


# Script: reconstruct best decoded physical quantities from rollout latents.
#
# This creates a qualitative figure, not a learned pixel decoder output. It
# decodes agent location, block location, and block angle from predicted latents
# with trained probes, picks the examples with the smallest combined physical
# error, then renders the decoded physical state beside the real frame.


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib-cache"))

from probe.rollout_ablation import eval_device, rollout_artifact_paths  # noqa: E402
from probe.rollout_reconstruction import (  # noqa: E402
    build_best_reconstruction_data,
    format_error_summary,
    plot_reconstruction_grid,
)


def parse_args():
    """Parse CLI options for physical-state reconstruction visualization."""
    parser = argparse.ArgumentParser(
        description="Render best decoded physical quantities from predicted rollout latents."
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--dataset-name", default="pusht_expert_train")
    parser.add_argument("--policy-name", default="pusht/lewm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--raw-horizon", type=int, default=100)
    parser.add_argument("--horizon-step", type=int, default=35)
    parser.add_argument("--num-examples", type=int, default=12)
    parser.add_argument("--probe-type", default="mlp", choices=["linear", "mlp"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def default_cache_dir():
    """Prefer the project-local `.stable-wm` directory for inputs and outputs."""
    project_cache_dir = REPO_ROOT / ".stable-wm"
    if project_cache_dir.exists():
        return project_cache_dir

    import stable_worldmodel as swm

    return Path(swm.data.utils.get_cache_dir())


def output_path(output_dir, horizon_step, num_examples, probe_type):
    """Return the PNG output path for one reconstruction run."""
    return (
        Path(output_dir)
        / f"rollout_horizon{horizon_step}_best{num_examples}_{probe_type}_physical_reconstruction.png"
    )


def main():
    """Run best-example physical reconstruction and save the figure."""
    args = parse_args()
    cache_dir = args.cache_dir or default_cache_dir()
    device = eval_device() if args.device == "auto" else args.device

    predicted_path = rollout_artifact_paths(
        cache_dir=cache_dir,
        dataset_name=args.dataset_name,
        policy_name=args.policy_name,
        split_seed=args.seed,
        img_size=args.img_size,
        num_episodes=args.num_episodes,
        raw_horizon=args.raw_horizon,
    )["predicted"]
    if not predicted_path.exists():
        raise FileNotFoundError(
            f"Missing predicted rollout artifact: {predicted_path}. "
            "Run scripts/probe_rollout_ablation.py first."
        )

    output_dir = args.output_dir or (
        cache_dir
        / "probes"
        / "reports"
        / "rollout_visualizations"
        / f"episodes{args.num_episodes}_horizon{args.raw_horizon}"
    )
    save_path = output_path(
        output_dir,
        horizon_step=args.horizon_step,
        num_examples=args.num_examples,
        probe_type=args.probe_type,
    )

    print(f"device: {device}")
    print(f"predicted rollout artifact: {predicted_path}")
    print(f"horizon step: {args.horizon_step}")
    print(f"num examples: {args.num_examples}")
    print(f"probe type: {args.probe_type}")

    data = build_best_reconstruction_data(
        predicted_path=predicted_path,
        cache_dir=cache_dir,
        raw_step=args.horizon_step,
        num_examples=args.num_examples,
        probe_type=args.probe_type,
        device=device,
        dataset_name=args.dataset_name,
    )
    figure_path = plot_reconstruction_grid(data, save_path=save_path)

    print("probe checkpoints:")
    for target_name, checkpoint_path in data["checkpoint_paths"].items():
        print(f"  {target_name}: {checkpoint_path}")
    print("selected best examples and errors:")
    for row in format_error_summary(data):
        print(
            f"rank={row['rank']:02d} ep={row['episode_id']} "
            f"agent_l2={row['agent_l2']:.3f} coord-units "
            f"block_l2={row['block_l2']:.3f} coord-units "
            f"angle={row['angle_rad']:.4f} rad/{row['angle_deg']:.2f} deg "
            f"score={row['score']:.6f}"
        )
    print(f"Saved reconstruction figure to {figure_path}")


if __name__ == "__main__":
    main()
