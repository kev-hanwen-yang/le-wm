"""Regenerate Step 2 diagnostic figures/tables from saved numeric outputs only."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_compatibility_plots(results):
    results = Path(results)
    records = json.loads((results / "case_metrics.json").read_text())
    spatial = json.loads((results / "spatial_metrics.json").read_text())
    calibration = json.loads((results / "development_calibration.json").read_text())
    manifest = json.loads((results / "compatibility_manifest.json").read_text())
    caps = manifest["assessment"]["proposed_drift_caps"]
    regions = ["background", "agent", "block", "goal"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    rows = []
    for offset, split, color in ((-.18, "development", "#507aa3"), (.18, "validation", "#c57a42")):
        selected = [item for item in spatial if item["split"] == split and item["condition"] == "hdf5_minus_raw"]
        rmse = [np.mean([item["regions"][region]["rgb_rmse"] for item in selected]) for region in regions]
        energy = [100 * np.mean([item["regions"][region]["energy_share"] for item in selected]) for region in regions]
        axes[0].bar(np.arange(4) + offset, rmse, .36, color=color, label=f"{split}, n={len(selected)}")
        axes[1].bar(np.arange(4) + offset, energy, .36, color=color)
        for region, value, share in zip(regions, rmse, energy):
            rows.append(dict(split=split, region=region, n_families=len(selected), mean_region_rgb_rmse=value,
                             mean_squared_residual_energy_percent=share))
    for axis in axes:
        axis.set_xticks(np.arange(4), regions); axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean region RGB RMSE (0–255 channel units)")
    axes[1].set_ylabel("Mean share of squared RGB residual (%)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Source vs raw reconstruction: descriptive spatial residuals")
    fig.tight_layout(); fig.savefig(results / "spatial_residuals.png", dpi=160); plt.close(fig)
    with (results / "spatial_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    metrics = [("latent_l2", "Latent L2", None),
               ("agent_drift_world_px", "Agent readout drift (world px)", caps["agent_drift_world_px"]),
               ("block_drift_world_px", "Block readout drift (world px)", caps["block_drift_world_px"]),
               ("angle_drift_rad", "Angle readout drift (rad)", caps["angle_drift_rad"])]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    for axis, (key, label, cap) in zip(axes, metrics):
        validation = [item["raw_reconstruction"][key] for item in records if item["split"] == "validation"]
        axis.scatter(np.arange(len(validation)), validation, s=25, color="#c57a42", label="validation cases")
        axis.axhline(calibration[key]["upper_one_sided_95"], color="#507aa3", label="development natural margin")
        if cap is not None: axis.axhline(cap, color="#8b3c4b", linestyle="--", label="proposed pilot budget")
        axis.set_title(label, fontsize=9); axis.set_xlabel("Independent family"); axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Observation nuisance is not small against natural reference / proposed budgets")
    fig.tight_layout(); fig.savefig(results / "encoder_readout_drift.png", dpi=160); plt.close(fig)

    rows = []
    for split in ("development", "validation"):
        selected = [item for item in records if item["split"] == split]
        for condition in selected[0]["codec_same_state_metrics"]:
            row = dict(split=split, condition=condition, n_families=len(selected))
            for metric, _, _ in metrics:
                row["same_state_" + metric + "_rms"] = float(np.sqrt(np.mean([item["codec_same_state_metrics"][condition][metric] ** 2 for item in selected])))
                row["hdf5_comparison_" + metric + "_rms"] = float(np.sqrt(np.mean([item["hdf5_vs_codec_metrics"][condition][metric] ** 2 for item in selected])))
            rows.append(row)
    with (results / "codec_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
