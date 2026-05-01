import json
from pathlib import Path

import numpy as np
import torch

from probe.metrics import evaluate_linear_probe
from probe.targets import TARGET_BUILDERS
from probe.train import SUPPORTED_PROBE_TYPES, build_probe, move_pairs_to_device


# Component: held-out test evaluation and reporting.
#
# This module evaluates saved probe checkpoints on cached test embeddings. It
# does not train probes and does not run the LeWM encoder. Each checkpoint stores
# the target normalizer and probe weights selected by validation early stopping;
# this module reloads those artifacts and computes clean held-out test metrics.
#
# Multi-seed reporting policy:
# - When `probe_seeds` is provided, checkpoints are loaded from
#   `probes/seed{probe_seed}/...` and aggregated per target/probe pair.
# - Existing seed-level fields (`test_norm_mse_values`,
#   `test_norm_mse_mean/std`, `test_r_values`, `test_r_mean/std`) are kept to
#   describe variation across probe-training seeds.
# - Paper-style fields (`test_sample_norm_mse_mean/std`) describe the
#   per-sample MSE distribution over held-out test examples. The Markdown table
#   uses these paper-style fields for MSE ± std.
# - r is still summarized across probe seeds because Pearson r is a global
#   test-set statistic, not a per-sample statistic.
# - The dataset split seed remains separate and is encoded in the test cache path.


DEFAULT_TARGET_NAMES = ("agent_location", "block_location", "block_angle")
DEFAULT_PROBE_TYPES = ("linear", "mlp")
PROPERTY_DISPLAY_NAMES = {
    "agent_location": "Agent Location",
    "block_location": "Block Location",
    "block_angle": "Block Angle",
}


def probe_checkpoint_path(cache_dir, target_name, probe_type, probe_seed=None):
    probe_dir = Path(cache_dir) / "probes"
    if probe_seed is not None:
        probe_dir = probe_dir / f"seed{probe_seed}"
    return probe_dir / f"pusht_{target_name}_{probe_type}.pt"


def load_encoded_split(cache_path):
    payload = torch.load(cache_path, map_location="cpu")
    if "encoded" not in payload:
        raise KeyError(f"Expected key 'encoded' in cached split: {cache_path}")
    return payload["encoded"]


def load_probe_checkpoint(checkpoint_path, device):
    payload = torch.load(checkpoint_path, map_location="cpu")
    target_name = payload["target_name"]
    probe_type = payload["probe_type"]
    target_dim = payload["target_mean"].shape[1]

    probe = build_probe(
        probe_type=probe_type,
        input_dim=192,
        output_dim=target_dim,
        mlp_hidden_dim=payload.get("mlp_hidden_dim", 256),
        mlp_num_hidden_layers=payload.get("mlp_num_hidden_layers", 1),
        mlp_dropout=payload.get("mlp_dropout", 0.1),
    ).to(device)
    probe.load_state_dict(payload["probe_state_dict"])
    probe.eval()

    return {
        "target_name": target_name,
        "probe_type": probe_type,
        "probe": probe,
        "target_mean": payload["target_mean"].to(device),
        "target_std": payload["target_std"].to(device),
        "payload": payload,
    }


def evaluate_checkpoint_on_pairs(pairs, checkpoint_path, device, batch_size, probe_seed=None):
    loaded = load_probe_checkpoint(checkpoint_path, device)
    if pairs["target_name"] != loaded["target_name"]:
        raise ValueError(
            f"Checkpoint target_name={loaded['target_name']!r} does not match "
            f"pairs target_name={pairs['target_name']!r}"
        )

    pairs = move_pairs_to_device(pairs, device)
    stats = evaluate_linear_probe(
        pairs,
        loaded["probe"],
        loaded["target_mean"],
        loaded["target_std"],
        batch_size=batch_size,
    )
    return {
        "target_name": loaded["target_name"],
        "probe_type": loaded["probe_type"],
        "probe_seed": loaded["payload"].get("probe_seed", probe_seed),
        "checkpoint_path": str(checkpoint_path),
        "test_norm_mse_mean": stats["norm_mse"],
        "test_norm_mse_std": 0.0,
        "test_sample_norm_mse_mean": stats["sample_norm_mse_mean"],
        "test_sample_norm_mse_std": stats["sample_norm_mse_std"],
        "test_sample_norm_mse_count": stats["sample_norm_mse_count"],
        "test_r_mean": stats["r_mean"],
        "test_r_std": 0.0,
        "test_r_per_dim": stats["r_per_dim"],
        "test_raw_mse": stats["raw_mse"],
        "test_raw_rmse": stats["raw_rmse"],
        "test_sample_raw_mse_mean": stats["sample_raw_mse_mean"],
        "test_sample_raw_mse_std": stats["sample_raw_mse_std"],
        "best_epoch": loaded["payload"].get("best_epoch"),
        "best_val_norm_mse": loaded["payload"].get("best_val_norm_mse"),
        "best_val_raw_mse": loaded["payload"].get("best_val_raw_mse"),
        "status": "ok",
    }


def evaluate_probe_grid(
    test_encoded,
    cache_dir,
    device,
    batch_size,
    target_names=DEFAULT_TARGET_NAMES,
    probe_types=DEFAULT_PROBE_TYPES,
    probe_seeds=None,
):
    rows = []
    for target_name in target_names:
        if target_name not in TARGET_BUILDERS:
            raise ValueError(f"Unknown target_name={target_name!r}")
        test_pairs = TARGET_BUILDERS[target_name](test_encoded)

        for probe_type in probe_types:
            if probe_type not in SUPPORTED_PROBE_TYPES:
                raise ValueError(f"Unknown probe_type={probe_type!r}")

            if probe_seeds is not None:
                rows.append(
                    evaluate_seeded_probe_group(
                        test_pairs,
                        cache_dir=cache_dir,
                        target_name=target_name,
                        probe_type=probe_type,
                        probe_seeds=probe_seeds,
                        device=device,
                        batch_size=batch_size,
                    )
                )
                continue

            checkpoint_path = probe_checkpoint_path(cache_dir, target_name, probe_type)
            if not checkpoint_path.exists():
                rows.append(
                    {
                        "target_name": target_name,
                        "probe_type": probe_type,
                        "checkpoint_path": str(checkpoint_path),
                        "status": "missing",
                    }
                )
                continue

            rows.append(
                evaluate_checkpoint_on_pairs(
                    test_pairs,
                    checkpoint_path,
                    device=device,
                    batch_size=batch_size,
                )
            )
    return rows


def evaluate_seeded_probe_group(
    test_pairs,
    cache_dir,
    target_name,
    probe_type,
    probe_seeds,
    device,
    batch_size,
):
    seed_rows = []
    missing_probe_seeds = []
    for probe_seed in probe_seeds:
        checkpoint_path = probe_checkpoint_path(
            cache_dir,
            target_name,
            probe_type,
            probe_seed=probe_seed,
        )
        if not checkpoint_path.exists():
            missing_probe_seeds.append(probe_seed)
            continue
        seed_rows.append(
            evaluate_checkpoint_on_pairs(
                test_pairs,
                checkpoint_path,
                device=device,
                batch_size=batch_size,
                probe_seed=probe_seed,
            )
        )

    if not seed_rows:
        return {
            "target_name": target_name,
            "probe_type": probe_type,
            "probe_seeds": list(probe_seeds),
            "completed_probe_seeds": [],
            "missing_probe_seeds": missing_probe_seeds,
            "status": "missing",
        }

    mse_values = [row["test_norm_mse_mean"] for row in seed_rows]
    r_values = [row["test_r_mean"] for row in seed_rows]
    r_per_dim_values = [row["test_r_per_dim"] for row in seed_rows]
    mse_mean, mse_std = mean_and_std(mse_values)
    r_mean, r_std = mean_and_std(r_values)
    r_per_dim_mean, r_per_dim_std = mean_and_std_per_dim(r_per_dim_values)
    sample_mse_mean, sample_mse_std, sample_mse_count = pooled_sample_mean_and_std(
        [
            (
                row["test_sample_norm_mse_mean"],
                row["test_sample_norm_mse_std"],
                row["test_sample_norm_mse_count"],
            )
            for row in seed_rows
        ]
    )
    sample_raw_mse_mean, sample_raw_mse_std, _ = pooled_sample_mean_and_std(
        [
            (
                row["test_sample_raw_mse_mean"],
                row["test_sample_raw_mse_std"],
                row["test_sample_norm_mse_count"],
            )
            for row in seed_rows
        ]
    )

    return {
        "target_name": target_name,
        "probe_type": probe_type,
        "probe_seeds": list(probe_seeds),
        "completed_probe_seeds": [row["probe_seed"] for row in seed_rows],
        "missing_probe_seeds": missing_probe_seeds,
        "num_completed_probe_seeds": len(seed_rows),
        "test_norm_mse_values": mse_values,
        "test_norm_mse_mean": mse_mean,
        "test_norm_mse_std": mse_std,
        "test_sample_norm_mse_mean": sample_mse_mean,
        "test_sample_norm_mse_std": sample_mse_std,
        "test_sample_norm_mse_count": sample_mse_count,
        "test_r_values": r_values,
        "test_r_mean": r_mean,
        "test_r_std": r_std,
        "test_r_per_dim_mean": r_per_dim_mean,
        "test_r_per_dim_std": r_per_dim_std,
        "test_sample_raw_mse_mean": sample_raw_mse_mean,
        "test_sample_raw_mse_std": sample_raw_mse_std,
        "seed_results": seed_rows,
        "status": "ok" if not missing_probe_seeds else "partial",
    }


def mean_and_std(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None, None
    if values.size == 1:
        return float(values.mean()), 0.0
    return float(values.mean()), float(values.std(ddof=1))


def mean_and_std_per_dim(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return [], []
    if values.shape[0] == 1:
        return values[0].tolist(), [0.0] * values.shape[1]
    return values.mean(axis=0).tolist(), values.std(axis=0, ddof=1).tolist()


def pooled_sample_mean_and_std(mean_std_count_tuples):
    # Combine per-seed sample-MSE distributions without storing every sample MSE
    # value in the JSON report. This treats all test sample errors from all
    # completed probe seeds as one paper-like distribution.
    total_count = sum(count for _, _, count in mean_std_count_tuples)
    if total_count == 0:
        return None, None, 0

    pooled_mean = (
        sum(mean * count for mean, _, count in mean_std_count_tuples) / total_count
    )
    if total_count == 1:
        return float(pooled_mean), 0.0, int(total_count)

    sum_squares = 0.0
    for mean, std, count in mean_std_count_tuples:
        if count <= 0:
            continue
        within = (count - 1) * (std**2)
        between = count * ((mean - pooled_mean) ** 2)
        sum_squares += within + between

    pooled_std = np.sqrt(sum_squares / (total_count - 1))
    return float(pooled_mean), float(pooled_std), int(total_count)


def format_markdown_table(rows):
    rows_by_target_probe = {
        (row["target_name"], row["probe_type"]): row for row in rows
    }
    target_names_in_rows = {row["target_name"] for row in rows}
    target_order = [
        target_name
        for target_name in DEFAULT_TARGET_NAMES
        if target_name in target_names_in_rows
    ]
    target_order.extend(
        sorted(target_names_in_rows.difference(DEFAULT_TARGET_NAMES))
    )
    table = [
        "<table>",
        "  <thead>",
        "    <tr>",
        "      <th rowspan=\"2\">Property</th>",
        "      <th colspan=\"2\">Linear</th>",
        "      <th colspan=\"2\">MLP</th>",
        "    </tr>",
        "    <tr>",
        "      <th>MSE ↓</th>",
        "      <th>r ↑</th>",
        "      <th>MSE ↓</th>",
        "      <th>r ↑</th>",
        "    </tr>",
        "  </thead>",
        "  <tbody>",
    ]

    for target_name in target_order:
        linear_row = rows_by_target_probe.get((target_name, "linear"))
        mlp_row = rows_by_target_probe.get((target_name, "mlp"))
        linear_mse, linear_r = format_probe_cells(linear_row)
        mlp_mse, mlp_r = format_probe_cells(mlp_row)
        property_name = PROPERTY_DISPLAY_NAMES.get(target_name, target_name)
        table.extend(
            [
                "    <tr>",
                f"      <td>{property_name}</td>",
                f"      <td>{linear_mse}</td>",
                f"      <td>{linear_r}</td>",
                f"      <td>{mlp_mse}</td>",
                f"      <td>{mlp_r}</td>",
                "    </tr>",
            ]
        )
    table.extend(["  </tbody>", "</table>"])
    return "\n".join(table)


def format_probe_cells(row):
    if row is None or row["status"] == "missing":
        return "missing", "missing"
    # MSE uses the paper-like per-sample distribution over the held-out test set.
    # Seed-level MSE variability is still kept separately in JSON as
    # `test_norm_mse_mean/std` and `test_norm_mse_values`.
    mse_mean = row.get("test_sample_norm_mse_mean", row["test_norm_mse_mean"])
    mse_std = row.get("test_sample_norm_mse_std", row["test_norm_mse_std"])
    mse = f"{mse_mean:.3f} ± {mse_std:.3f}"
    r_std = row.get("test_r_std", 0.0)
    r = f"{row['test_r_mean']:.3f} ± {r_std:.6f}"
    return mse, r


def format_seed_level_markdown_table(rows):
    # This table preserves the original multi-seed reporting view. It is
    # intentionally separate from `format_markdown_table`: the paper-style table
    # uses per-sample test MSE std, while this table uses across-probe-seed std.
    headers = [
        "target",
        "probe",
        "status",
        "completed_probe_seeds",
        "missing_probe_seeds",
        "seed_level_mse_mean +- std",
        "seed_level_r_mean +- std",
        "mse_by_seed",
        "r_by_seed",
    ]
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        if row["status"] == "missing":
            values = [
                row["target_name"],
                row["probe_type"],
                row["status"],
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        else:
            norm_mse = (
                f"{row['test_norm_mse_mean']:.6f} +- "
                f"{row['test_norm_mse_std']:.6f}"
            )
            r_std = row.get("test_r_std", 0.0)
            r = f"{row['test_r_mean']:.6f} +- {r_std:.6f}"
            completed_probe_seeds = row.get("completed_probe_seeds")
            if completed_probe_seeds is None:
                probe_seed = row.get("probe_seed")
                completed_probe_seeds = [] if probe_seed is None else [probe_seed]
            missing_probe_seeds = row.get("missing_probe_seeds", [])
            mse_values = row.get("test_norm_mse_values", [row["test_norm_mse_mean"]])
            r_values = row.get("test_r_values", [row["test_r_mean"]])
            mse_by_seed = format_seed_values(completed_probe_seeds, mse_values)
            r_by_seed = format_seed_values(completed_probe_seeds, r_values)
            values = [
                row["target_name"],
                row["probe_type"],
                row["status"],
                format_seed_list(completed_probe_seeds),
                format_seed_list(missing_probe_seeds),
                norm_mse,
                r,
                mse_by_seed,
                r_by_seed,
            ]
        table.append("| " + " | ".join(values) + " |")
    return "\n".join(table)


def format_seed_list(seeds):
    if not seeds:
        return ""
    return ", ".join(str(seed) for seed in seeds)


def format_seed_values(seeds, values):
    if not values:
        return ""
    if len(seeds) == len(values):
        return ", ".join(
            f"{seed}: {value:.6f}" for seed, value in zip(seeds, values)
        )
    return ", ".join(f"{value:.6f}" for value in values)


def format_detailed_markdown_table(rows):
    headers = [
        "target",
        "probe",
        "status",
        "paper_sample_mse_mean +- std",
        "seed_level_mse_mean +- std",
        "seed_level_r_mean +- std",
        "test_r_per_dim",
        "best_val_norm_mse",
    ]
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        if row["status"] == "missing":
            values = [
                row["target_name"],
                row["probe_type"],
                row["status"],
                "",
                "",
                "",
                "",
                "",
            ]
        else:
            sample_mse = (
                f"{row['test_sample_norm_mse_mean']:.6f} +- "
                f"{row['test_sample_norm_mse_std']:.6f}"
            )
            seed_mse = (
                f"{row['test_norm_mse_mean']:.6f} +- "
                f"{row['test_norm_mse_std']:.6f}"
            )
            r_std = row.get("test_r_std", 0.0)
            seed_r = f"{row['test_r_mean']:.6f} +- {r_std:.6f}"
            r_per_dim_values = row.get(
                "test_r_per_dim_mean",
                row.get("test_r_per_dim", []),
            )
            r_per_dim = "[" + ", ".join(f"{v:.4f}" for v in r_per_dim_values) + "]"
            best_val_norm_mse = row.get("best_val_norm_mse")
            best_val_text = (
                f"{best_val_norm_mse:.6f}" if best_val_norm_mse is not None else ""
            )
            values = [
                row["target_name"],
                row["probe_type"],
                row["status"],
                sample_mse,
                seed_mse,
                seed_r,
                r_per_dim,
                best_val_text,
            ]
        table.append("| " + " | ".join(values) + " |")
    return "\n".join(table)


def save_report(rows, output_dir, report_name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = output_dir / f"{report_name}.md"
    seed_markdown_path = output_dir / f"{report_name}_seed_summary.md"
    json_path = output_dir / f"{report_name}.json"

    markdown_path.write_text(format_markdown_table(rows) + "\n", encoding="utf-8")
    seed_markdown_path.write_text(
        format_seed_level_markdown_table(rows) + "\n",
        encoding="utf-8",
    )
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return markdown_path, seed_markdown_path, json_path
