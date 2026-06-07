#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path


# Script: teacher-forced predictor velocity diagnostic.
#
# This experiment isolates the one-step LeWM predictor map from open-loop
# autoregressive compounding. For each horizon h, it feeds real latent anchors:
#   [z[h-15], z[h-10], z[h-5]]
# and real action embeddings:
#   [a[h-15:h-10], a[h-10:h-5], a[h-5:h]]
# into the predictor exactly once, then compares the predicted velocity
# ||zhat_tf[h] - z[h-5]|| against the true velocity ||z[h] - z[h-5]||.


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib-cache"))

import torch  # noqa: E402

from probe.latent_diagnostics import (  # noqa: E402
    build_teacher_forced_real_latents,
    compute_teacher_forced_velocity_metrics,
    load_true_future_latents,
    predict_teacher_forced_latents,
    resolve_true_future_latents_path,
    save_teacher_forced_velocity_charts,
    save_teacher_forced_velocity_markdown,
    save_teacher_forced_velocity_report,
    teacher_forced_velocity_diagnostic_paths,
    validate_predicted_latent_artifact,
    validate_true_future_latent_artifact,
    verify_teacher_forced_first_step,
)
from probe.rollout_ablation import (  # noqa: E402
    eval_device,
    load_released_lewm_world_model,
    rollout_artifact_paths,
)
from probe.rollout_probe_eval import load_rollout_prediction_artifact  # noqa: E402


def parse_args():
    """Parse CLI options for teacher-forced velocity diagnostics."""
    parser = argparse.ArgumentParser(
        description="Compute teacher-forced LeWM predictor velocity diagnostics."
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--dataset-name", default="pusht_expert_train")
    parser.add_argument("--policy-name", default="pusht/lewm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--raw-horizon", type=int, default=100)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--sanity-atol", type=float, default=1e-5)
    parser.add_argument("--sanity-rtol", type=float, default=1e-5)
    return parser.parse_args()


def default_cache_dir():
    """Prefer the project-local `.stable-wm` cache directory."""
    project_cache_dir = REPO_ROOT / ".stable-wm"
    if project_cache_dir.exists():
        return project_cache_dir

    import stable_worldmodel as swm

    return Path(swm.data.utils.get_cache_dir())


def load_encoded_rollout_artifact(encoded_path):
    """Load saved context/action embeddings from the rollout ablation artifact."""
    payload = torch.load(encoded_path, map_location="cpu")
    if "encoded" not in payload:
        raise KeyError(f"Expected key 'encoded' in {encoded_path}")
    return payload["encoded"], payload.get("metadata", {})


def validate_teacher_forced_inputs(encoded, predicted, true_artifact, history_size):
    """Check saved rollout, true-latent, and action tensors are mutually aligned."""
    validate_predicted_latent_artifact(predicted)
    validate_true_future_latent_artifact(predicted, true_artifact)
    if not torch.equal(encoded["episode_idx"].long(), predicted["episode_idx"].long()):
        raise ValueError("Encoded rollout and predicted rollout episode_idx do not match")
    if not torch.equal(encoded["target_steps"].long(), predicted["target_steps"].long()):
        raise ValueError("Encoded rollout and predicted rollout target_steps do not match")
    if encoded["context_emb"].shape[1] != history_size:
        raise ValueError(
            f"context_emb history length {encoded['context_emb'].shape[1]} "
            f"does not match history_size={history_size}"
        )
    expected_action_steps = predicted["predicted_latents"].shape[1] + history_size - 1
    if encoded["action_emb"].shape[1] != expected_action_steps:
        raise ValueError(
            f"action_emb length {encoded['action_emb'].shape[1]} does not match "
            f"expected {expected_action_steps}"
        )


def build_metadata(args, prediction_metadata, encoded_metadata, true_latent_metadata, sanity_check):
    """Collect metadata saved into the teacher-forced velocity JSON report."""
    return {
        "dataset_name": args.dataset_name,
        "policy_name": args.policy_name,
        "split_seed": args.seed,
        "img_size": args.img_size,
        "num_episodes": args.num_episodes,
        "raw_horizon": args.raw_horizon,
        "history_size": args.history_size,
        "batch_size": args.batch_size,
        "device": args.device,
        "sanity_atol": args.sanity_atol,
        "sanity_rtol": args.sanity_rtol,
        "prediction_metadata": prediction_metadata,
        "encoded_metadata": encoded_metadata,
        "true_latent_metadata": true_latent_metadata,
        "sanity_check": sanity_check,
    }


def print_summary(report):
    """Print key ratios and sanity status for quick verification."""
    ratio_by_horizon = {
        int(horizon): float(ratio)
        for horizon, ratio in zip(report["horizons"], report["ratio_mean"])
    }
    openloop_ratio_by_horizon = {
        int(horizon): float(ratio)
        for horizon, ratio in zip(report["horizons"], report["openloop_ratio_mean"])
    }
    print("first-step sanity:", report["sanity_check"])
    for horizon in [15, 30, 50, 100]:
        print(
            f"h={horizon}: "
            f"tf_ratio={ratio_by_horizon[horizon]:.6f} "
            f"openloop_ratio={openloop_ratio_by_horizon[horizon]:.6f}"
        )


def main():
    """Run teacher-forced velocity diagnostics and save report/charts."""
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
    encoded_path = rollout_paths["encoded"]
    if not predicted_path.exists():
        raise FileNotFoundError(
            f"Missing predicted rollout artifact: {predicted_path}. "
            "Run scripts/probe_rollout_ablation.py first."
        )
    if not encoded_path.exists():
        raise FileNotFoundError(
            f"Missing encoded rollout artifact: {encoded_path}. "
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

    output_paths = teacher_forced_velocity_diagnostic_paths(
        cache_dir=cache_dir,
        dataset_name=args.dataset_name,
        policy_name=args.policy_name,
        split_seed=args.seed,
        img_size=args.img_size,
        num_episodes=args.num_episodes,
        raw_horizon=args.raw_horizon,
    )

    print(f"device: {device}")
    print(f"encoded rollout artifact: {encoded_path}")
    print(f"predicted rollout artifact: {predicted_path}")
    print(f"true future latent cache: {true_latents_path}")

    encoded, encoded_metadata = load_encoded_rollout_artifact(encoded_path)
    predicted, prediction_metadata = load_rollout_prediction_artifact(predicted_path)
    true_artifact, true_latent_metadata = load_true_future_latents(true_latents_path)
    validate_teacher_forced_inputs(
        encoded=encoded,
        predicted=predicted,
        true_artifact=true_artifact,
        history_size=args.history_size,
    )

    real_latents = build_teacher_forced_real_latents(
        context_emb=encoded["context_emb"].float(),
        true_future_latents=true_artifact["true_future_latents"].float(),
    )
    print(f"real_latents: {tuple(real_latents.shape)}")
    print(f"action_emb: {tuple(encoded['action_emb'].shape)}")
    print(f"target_steps: {predicted['target_steps'].long().tolist()}")

    model = load_released_lewm_world_model(
        policy_name=args.policy_name,
        device=device,
        cache_dir=cache_dir,
    )
    teacher_forced_latents = predict_teacher_forced_latents(
        real_latents=real_latents,
        action_emb=encoded["action_emb"].float(),
        model=model,
        device=device,
        batch_size=args.batch_size,
        history_size=args.history_size,
    )
    open_loop_latents = predicted["predicted_latents"].float()
    metrics = compute_teacher_forced_velocity_metrics(
        real_latents=real_latents,
        teacher_forced_latents=teacher_forced_latents,
        open_loop_latents=open_loop_latents,
    )
    sanity_check = verify_teacher_forced_first_step(
        teacher_forced_latents=teacher_forced_latents,
        open_loop_latents=open_loop_latents,
        metrics=metrics,
        atol=args.sanity_atol,
        rtol=args.sanity_rtol,
    )
    metadata = build_metadata(
        args=args,
        prediction_metadata=prediction_metadata,
        encoded_metadata=encoded_metadata,
        true_latent_metadata=true_latent_metadata,
        sanity_check=sanity_check,
    )
    json_path = save_teacher_forced_velocity_report(
        metrics=metrics,
        target_steps=predicted["target_steps"].long(),
        metadata=metadata,
        sanity_check=sanity_check,
        json_path=output_paths["json"],
    )
    chart_paths = save_teacher_forced_velocity_charts(
        metrics=metrics,
        target_steps=predicted["target_steps"].long(),
        paths=output_paths,
    )
    with Path(json_path).open() as f:
        report = json.load(f)
    markdown_path = save_teacher_forced_velocity_markdown(
        report=report,
        chart_paths=chart_paths,
        markdown_path=output_paths["markdown"],
    )

    print_summary(report)
    print(f"saved JSON report: {json_path}")
    print(f"saved markdown report: {markdown_path}")
    print(f"saved velocity chart: {chart_paths['velocity_chart']}")
    print(f"saved ratio chart: {chart_paths['ratio_chart']}")
    print(f"saved error chart: {chart_paths['error_chart']}")


if __name__ == "__main__":
    main()
