#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path


# Script: qualitative visualization for predicted rollout latents.
#
# It selects random rollout examples at a requested horizon, displays each real
# frame next to a nearest-neighbor latent proxy frame, and plots probe-predicted
# vs true agent/block locations. The proxy is needed because this LeWM checkpoint
# has no latent-to-pixel decoder.


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib-cache"))

from probe.embedding_cache import encoded_cache_path  # noqa: E402
from probe.rollout_ablation import eval_device, rollout_artifact_paths  # noqa: E402
from probe.rollout_visualization import (  # noqa: E402
    build_horizon_visualization_data,
    plot_probe_location_scatters,
    plot_real_vs_nearest_proxy_grid,
)


def parse_args():
    """Parse CLI options for qualitative rollout visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize real frames, nearest-neighbor latent proxies, and probe locations."
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--dataset-name", default="pusht_expert_train")
    parser.add_argument("--policy-name", default="pusht/lewm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--raw-horizon", type=int, default=100)
    parser.add_argument("--horizon-step", type=int, default=35)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=123)
    parser.add_argument("--probe-type", default="linear", choices=["linear", "mlp"])
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


def visualization_output_paths(output_dir, num_samples, horizon_step, probe_type):
    """Return the PNG paths for one qualitative visualization run."""
    output_dir = Path(output_dir)
    base_name = f"rollout_horizon{horizon_step}_samples{num_samples}_{probe_type}"
    return {
        "frame_grid": output_dir / f"{base_name}_real_vs_nn_proxy.png",
        "scatter": output_dir / f"{base_name}_probe_location_scatters.png",
    }


def main():
    """Build and save qualitative rollout visualization figures."""
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

    test_encoded_path = encoded_cache_path(
        cache_dir,
        args.dataset_name,
        args.policy_name,
        "test",
        args.seed,
        args.img_size,
    )
    if not test_encoded_path.exists():
        raise FileNotFoundError(f"Missing encoded test cache: {test_encoded_path}")

    output_dir = args.output_dir or (
        cache_dir
        / "probes"
        / "reports"
        / "rollout_visualizations"
        / f"episodes{args.num_episodes}_horizon{args.raw_horizon}"
    )
    paths = visualization_output_paths(
        output_dir,
        num_samples=args.num_samples,
        horizon_step=args.horizon_step,
        probe_type=args.probe_type,
    )

    print(f"device: {device}")
    print(f"predicted rollout artifact: {predicted_path}")
    print(f"encoded test cache: {test_encoded_path}")
    print(f"horizon step: {args.horizon_step}")
    print(f"num samples: {args.num_samples}")
    print(f"sample seed: {args.sample_seed}")
    print(f"probe type: {args.probe_type}")

    visualization_data = build_horizon_visualization_data(
        predicted_path=predicted_path,
        encoded_test_path=test_encoded_path,
        cache_dir=cache_dir,
        raw_step=args.horizon_step,
        num_samples=args.num_samples,
        seed=args.sample_seed,
        probe_type=args.probe_type,
        device=device,
        dataset_name=args.dataset_name,
    )

    frame_grid_path = plot_real_vs_nearest_proxy_grid(
        visualization_data,
        save_path=paths["frame_grid"],
    )
    scatter_path = plot_probe_location_scatters(
        visualization_data,
        save_path=paths["scatter"],
    )

    print("selected rollout episode IDs:", visualization_data["selected_episode_ids"].tolist())
    print("nearest proxy episode IDs:", visualization_data["nearest"]["nearest_episode_idx"].tolist())
    print("nearest proxy step IDs:", visualization_data["nearest"]["nearest_step_idx"].tolist())
    print("nearest L2^2:", [round(v, 6) for v in visualization_data["nearest"]["nearest_l2_sq"].tolist()])
    print(f"agent probe: {visualization_data['agent_probe_path']}")
    print(f"block probe: {visualization_data['block_probe_path']}")
    print(f"Saved frame grid to {frame_grid_path}")
    print(f"Saved scatter plots to {scatter_path}")


if __name__ == "__main__":
    main()
