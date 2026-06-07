import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from probe.evaluate import load_probe_checkpoint, probe_checkpoint_path
from probe.metrics import pearson_r


# Component: horizon-wise probe evaluation for predicted rollout latents.
#
# This module applies already-trained Table 1 probes to the predicted latents
# from the open-loop rollout ablation. The probes output physical quantities,
# so the ground truth used here is `target_states`, not true future latent
# embeddings:
# - predicted_latents: (B, H, 192), e.g. (100, 18, 192)
# - target_states: (B, H, 7),       e.g. (100, 18, 7)
# - target_steps: (H,),             e.g. [15, 20, ..., 100]
#
# For each horizon and each target, we report normalized MSE, raw MSE/RMSE, and
# Pearson r. We intentionally do not compute std across probe seeds because this
# ablation uses one linear and one MLP checkpoint per physical quantity.


DEFAULT_TARGET_NAMES = ("agent_location", "block_location", "block_angle")
DEFAULT_PROBE_TYPES = ("linear", "mlp")
TARGET_STATE_SLICES = {
    "agent_location": slice(0, 2),
    "block_location": slice(2, 4),
    "block_angle": slice(4, 5),
}
TARGET_DISPLAY_NAMES = {
    "agent_location": "Agent Location",
    "block_location": "Block Location",
    "block_angle": "Block Angle",
}


def load_rollout_prediction_artifact(predicted_path):
    """Load the saved rollout prediction artifact and return its `predicted` dict."""
    payload = torch.load(predicted_path, map_location="cpu")
    if "predicted" not in payload:
        raise KeyError(f"Expected key 'predicted' in rollout artifact: {predicted_path}")
    return payload["predicted"], payload.get("metadata", {})


def select_target_from_states(target_states, target_name):
    """Select one physical quantity from Push-T state vectors.

    Push-T state layout used by Table 1 probing:
    - state[0:2]: agent_location, shape (..., 2)
    - state[2:4]: block_location, shape (..., 2)
    - state[4:5]: block_angle, shape (..., 1)
    """
    if target_name not in TARGET_STATE_SLICES:
        valid_targets = ", ".join(TARGET_STATE_SLICES)
        raise ValueError(f"Unknown target_name={target_name!r}. Valid targets: {valid_targets}")
    return target_states[..., TARGET_STATE_SLICES[target_name]].float()


def validate_rollout_prediction_shapes(predicted):
    """Check the rollout artifact has aligned latent/state/horizon tensors."""
    predicted_latents = predicted["predicted_latents"]
    target_states = predicted["target_states"]
    target_steps = predicted["target_steps"]

    if predicted_latents.ndim != 3:
        raise ValueError(
            f"predicted_latents must be (B, H, D), got {tuple(predicted_latents.shape)}"
        )
    if target_states.ndim != 3:
        raise ValueError(f"target_states must be (B, H, 7), got {tuple(target_states.shape)}")
    if predicted_latents.shape[:2] != target_states.shape[:2]:
        raise ValueError(
            f"predicted_latents {tuple(predicted_latents.shape)} and target_states "
            f"{tuple(target_states.shape)} do not share (B, H)"
        )
    if target_steps.numel() != predicted_latents.shape[1]:
        raise ValueError(
            f"target_steps length {target_steps.numel()} does not match rollout horizon "
            f"count {predicted_latents.shape[1]}"
        )


def evaluate_probe_at_horizons(
    predicted,
    loaded_probe,
    target_name,
    device,
    batch_size=4096,
):
    """Apply one trained probe to every rollout horizon.

    The probe predicts normalized target values. For each horizon we compute:
    - `norm_mse`: MSE between normalized prediction and normalized target
    - `raw_mse`/`raw_rmse`: error after unnormalizing back to physical units
    - `r_mean`: Pearson correlation between raw prediction and raw target
    """
    validate_rollout_prediction_shapes(predicted)
    predicted_latents = predicted["predicted_latents"].float()
    target = select_target_from_states(predicted["target_states"], target_name)
    target_steps = predicted["target_steps"].long()

    probe = loaded_probe["probe"]
    target_mean = loaded_probe["target_mean"].to(device)
    target_std = loaded_probe["target_std"].to(device)

    rows = []
    probe.eval()
    with torch.inference_mode():
        for horizon_idx, raw_step in enumerate(target_steps.tolist()):
            latents_at_horizon = predicted_latents[:, horizon_idx]
            target_raw = target[:, horizon_idx]

            pred_norm_batches = []
            for start in range(0, latents_at_horizon.size(0), batch_size):
                end = min(start + batch_size, latents_at_horizon.size(0))
                pred_norm = probe(latents_at_horizon[start:end].to(device))
                pred_norm_batches.append(pred_norm.cpu())

            pred_norm = torch.cat(pred_norm_batches, dim=0)
            target_norm = ((target_raw.to(device) - target_mean) / target_std).cpu()
            pred_raw = (pred_norm.to(device) * target_std + target_mean).cpu()

            norm_mse = F.mse_loss(pred_norm, target_norm)
            raw_mse = F.mse_loss(pred_raw, target_raw)
            r_mean, r_per_dim = pearson_r(pred_raw, target_raw)

            rows.append(
                {
                    "target_name": target_name,
                    "probe_type": loaded_probe["probe_type"],
                    "horizon_idx": horizon_idx,
                    "raw_step": int(raw_step),
                    "num_episodes": int(latents_at_horizon.size(0)),
                    "norm_mse": norm_mse.item(),
                    "raw_mse": raw_mse.item(),
                    "raw_rmse": torch.sqrt(raw_mse).item(),
                    "r_mean": r_mean.item(),
                    "r_per_dim": r_per_dim.tolist(),
                }
            )
    return rows


def evaluate_rollout_probe_grid(
    predicted,
    cache_dir,
    device,
    batch_size=4096,
    target_names=DEFAULT_TARGET_NAMES,
    probe_types=DEFAULT_PROBE_TYPES,
):
    """Evaluate one linear and one MLP probe per target over all horizons."""
    rows = []
    for probe_type in probe_types:
        for target_name in target_names:
            checkpoint_path = probe_checkpoint_path(
                cache_dir,
                target_name=target_name,
                probe_type=probe_type,
            )
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing probe checkpoint: {checkpoint_path}")

            loaded_probe = load_probe_checkpoint(checkpoint_path, device=device)
            if loaded_probe["target_name"] != target_name:
                raise ValueError(
                    f"Checkpoint {checkpoint_path} has target "
                    f"{loaded_probe['target_name']!r}, expected {target_name!r}"
                )
            probe_rows = evaluate_probe_at_horizons(
                predicted,
                loaded_probe=loaded_probe,
                target_name=target_name,
                device=device,
                batch_size=batch_size,
            )
            for row in probe_rows:
                row["checkpoint_path"] = str(checkpoint_path)
            rows.extend(probe_rows)
    return rows


def report_paths(output_dir, report_name):
    """Return JSON, Markdown, and chart paths for one rollout probe report."""
    output_dir = Path(output_dir)
    charts_dir = output_dir / f"{report_name}_charts"
    return {
        "json": output_dir / f"{report_name}.json",
        "markdown": output_dir / f"{report_name}.md",
        "charts_dir": charts_dir,
        "norm_mse_chart": charts_dir / "norm_mse_by_horizon.png",
        "r_chart": charts_dir / "pearson_r_by_horizon.png",
        "raw_rmse_chart": charts_dir / "raw_rmse_by_horizon.png",
    }


def save_json_report(rows, metadata, json_path):
    """Save machine-readable horizon-wise metrics."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump({"metadata": metadata, "rows": rows}, f, indent=2)
    return json_path


def format_markdown_report(rows, metadata, chart_paths):
    """Create a compact Markdown report linking charts and final-horizon metrics."""
    lines = [
        "# Push-T Rollout Probe Report",
        "",
        "## Setup",
        "",
        f"- Episodes: {metadata.get('num_episodes')}",
        f"- Target steps: {metadata.get('target_steps')}",
        f"- Probe types: {metadata.get('probe_types')}",
        f"- Targets: {metadata.get('target_names')}",
        "",
        "## Charts",
        "",
        f"- Normalized MSE: `{chart_paths['norm_mse_chart']}`",
        f"- Pearson r: `{chart_paths['r_chart']}`",
        f"- Raw RMSE: `{chart_paths['raw_rmse_chart']}`",
        "",
        "## Final Horizon",
        "",
        "| Probe | Physical Quantity | Step | Norm MSE | Raw RMSE | Pearson r |",
        "|---|---|---:|---:|---:|---:|",
    ]

    final_rows = [
        row for row in rows if row["raw_step"] == max(item["raw_step"] for item in rows)
    ]
    for row in final_rows:
        lines.append(
            "| "
            f"{row['probe_type']} | "
            f"{TARGET_DISPLAY_NAMES[row['target_name']]} | "
            f"{row['raw_step']} | "
            f"{row['norm_mse']:.6f} | "
            f"{row['raw_rmse']:.6f} | "
            f"{row['r_mean']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def save_markdown_report(rows, metadata, chart_paths, markdown_path):
    """Save a human-readable Markdown summary for the rollout probe ablation."""
    markdown_path = Path(markdown_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(format_markdown_report(rows, metadata, chart_paths))
    return markdown_path


def plot_metric_by_horizon(rows, metric_key, ylabel, title, save_path):
    """Plot one metric over rollout horizon for all target/probe combinations."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    for probe_type in DEFAULT_PROBE_TYPES:
        for target_name in DEFAULT_TARGET_NAMES:
            series = [
                row
                for row in rows
                if row["probe_type"] == probe_type and row["target_name"] == target_name
            ]
            if not series:
                continue
            series = sorted(series, key=lambda row: row["raw_step"])
            line_style = "-" if probe_type == "linear" else "--"
            label = f"{probe_type} {TARGET_DISPLAY_NAMES[target_name]}"
            plt.plot(
                [row["raw_step"] for row in series],
                [row[metric_key] for row in series],
                linestyle=line_style,
                marker="o",
                label=label,
            )

    plt.xlabel("Raw rollout horizon")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def save_charts(rows, chart_paths):
    """Save normalized MSE, raw RMSE, and Pearson r horizon curves."""
    saved = {
        "norm_mse_chart": plot_metric_by_horizon(
            rows,
            metric_key="norm_mse",
            ylabel="Normalized MSE",
            title="Probe Normalized MSE vs Rollout Horizon",
            save_path=chart_paths["norm_mse_chart"],
        ),
        "r_chart": plot_metric_by_horizon(
            rows,
            metric_key="r_mean",
            ylabel="Pearson r",
            title="Probe Pearson r vs Rollout Horizon",
            save_path=chart_paths["r_chart"],
        ),
        "raw_rmse_chart": plot_metric_by_horizon(
            rows,
            metric_key="raw_rmse",
            ylabel="Raw RMSE",
            title="Probe Raw RMSE vs Rollout Horizon",
            save_path=chart_paths["raw_rmse_chart"],
        ),
    }
    return saved


def save_rollout_probe_report(rows, metadata, output_dir, report_name):
    """Save all rollout probe report artifacts and return their paths."""
    paths = report_paths(output_dir=output_dir, report_name=report_name)
    chart_paths = save_charts(rows, paths)
    json_path = save_json_report(rows, metadata, paths["json"])
    markdown_path = save_markdown_report(rows, metadata, chart_paths, paths["markdown"])
    return {
        "json": json_path,
        "markdown": markdown_path,
        **chart_paths,
    }
