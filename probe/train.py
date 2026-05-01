import random

import numpy as np
import torch
import torch.nn.functional as F

from probe.metrics import evaluate_linear_probe
from probe.models import LinearRegressionProbe, MLPRegressionProbe


# Component: supervised probe training.
#
# This module owns the optimization loop for probes trained on cached LeWM
# embeddings. It keeps the encoder out of the training path: inputs are already
# precomputed embeddings, targets are physical quantities, and only the probe
# model parameters receive gradients.
#
# Current trainer:
# - train_probe: trains either a linear probe or a small MLP probe to predict
#   any target returned by `probe.targets`, e.g. agent_location(N, 2),
#   block_location(N, 2), or block_angle(N, 1), from emb(N, 192).
#
# Seed policy:
# - `cfg.seed` in probe.py controls the episode split and encoded-cache name.
# - `probe_seed` here controls only probe training randomness: initialization,
#   shuffled batch order, and dropout masks. Keeping these separate lets the
#   report measure training variability without changing the held-out episodes.


SUPPORTED_PROBE_TYPES = {"linear", "mlp"}


def seed_probe_training(probe_seed):
    if probe_seed is None:
        return
    random.seed(probe_seed)
    np.random.seed(probe_seed)
    torch.manual_seed(probe_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(probe_seed)


def move_pairs_to_device(pairs, device):
    return {
        "embeddings": pairs["embeddings"].to(device),
        "target": pairs["target"].to(device),
        "target_name": pairs["target_name"],
        "episode_idx": pairs["episode_idx"],
        "step_idx": pairs["step_idx"],
    }


def fit_target_normalizer(train_pairs, device):
    target = train_pairs["target"]
    mean = target.mean(dim=0, keepdim=True).to(device)
    std = target.std(dim=0, keepdim=True).clamp_min(1e-6).to(device)
    return mean, std


def make_pair_loader(pairs, batch_size, shuffle, probe_seed=None):
    tensor_dataset = torch.utils.data.TensorDataset(
        pairs["embeddings"],
        pairs["target"],
    )
    generator = None
    if shuffle and probe_seed is not None:
        # DataLoader shuffling is the second source of probe-training randomness
        # after model initialization, so seed it explicitly for reproducible runs.
        generator = torch.Generator()
        generator.manual_seed(probe_seed)
    return torch.utils.data.DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def build_probe(
    probe_type,
    input_dim,
    output_dim,
    mlp_hidden_dim=256,
    mlp_num_hidden_layers=1,
    mlp_dropout=0.1,
):
    if probe_type == "linear":
        return LinearRegressionProbe(input_dim=input_dim, output_dim=output_dim)
    if probe_type == "mlp":
        return MLPRegressionProbe(
            input_dim=input_dim,
            hidden_dim=mlp_hidden_dim,
            output_dim=output_dim,
            num_hidden_layers=mlp_num_hidden_layers,
            dropout=mlp_dropout,
        )
    valid_probe_types = ", ".join(sorted(SUPPORTED_PROBE_TYPES))
    raise ValueError(f"Unknown probe_type={probe_type!r}. Valid probe types: {valid_probe_types}")


def train_probe(
    train_pairs,
    val_pairs,
    device,
    save_path,
    batch_size,
    max_epochs,
    patience,
    lr,
    weight_decay,
    probe_type="linear",
    mlp_hidden_dim=256,
    mlp_num_hidden_layers=1,
    mlp_dropout=0.1,
    probe_seed=None,
):
    seed_probe_training(probe_seed) #  this makes repeated runs with the same probe_seed reproducible, and different probe_seeds measure probe-training variability.

    # Move train/val sets from CPU to the selected accelerator before training.
    train_pairs = move_pairs_to_device(train_pairs, device)
    val_pairs = move_pairs_to_device(val_pairs, device)
    target_name = train_pairs["target_name"]
    target_dim = train_pairs["target"].shape[1]
    target_mean, target_std = fit_target_normalizer(train_pairs, device)
    train_loader = make_pair_loader(
        train_pairs,
        batch_size=batch_size,
        shuffle=True,
        probe_seed=probe_seed,
    )

    probe = build_probe(
        probe_type=probe_type,
        input_dim=192,
        output_dim=target_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_num_hidden_layers=mlp_num_hidden_layers,
        mlp_dropout=mlp_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    # Early stopping tracks normalized validation MSE because that is the metric
    # intended for Table 1-style comparisons.
    best_val_norm_mse = float("inf")
    best_val_raw_mse = float("inf")
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0

    print(
        f"Training {probe_type} probe for {target_name} "
        f"(input_dim=192, output_dim={target_dim}, batch_size={batch_size}, "
        f"lr={lr}, weight_decay={weight_decay}, max_epochs={max_epochs}, "
        f"patience={patience}, mlp_hidden_dim={mlp_hidden_dim}, "
        f"mlp_num_hidden_layers={mlp_num_hidden_layers}, mlp_dropout={mlp_dropout}, "
        f"probe_seed={probe_seed})"
    )
    print(f"target_mean={target_mean.cpu().numpy().round(4).tolist()}")
    print(f"target_std={target_std.cpu().numpy().round(4).tolist()}")

    for epoch in range(1, max_epochs + 1):
        probe.train()
        train_loss_sum = 0.0
        train_count = 0

        for emb, target in train_loader:
            target_norm = (target - target_mean) / target_std

            pred_norm = probe(emb)
            loss = F.mse_loss(pred_norm, target_norm)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size_actual = target.size(0)
            train_loss_sum += loss.item() * batch_size_actual
            train_count += batch_size_actual

        # Last batch may be smaller than the batch size, so this computes the
        # correct epoch-average normalized training MSE.
        train_loss = train_loss_sum / train_count

        val_stats = evaluate_linear_probe(
            val_pairs,
            probe,
            target_mean,
            target_std,
            batch_size,
        )

        improved = val_stats["norm_mse"] < best_val_norm_mse
        if improved:
            best_val_norm_mse = val_stats["norm_mse"]
            best_val_raw_mse = val_stats["raw_mse"]
            best_epoch = epoch
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in probe.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        marker = "*" if improved else ""
        print(
            f"epoch={epoch:03d} train_norm_mse={train_loss:.6f} "
            f"val_norm_mse={val_stats['norm_mse']:.6f} "
            f"val_raw_mse={val_stats['raw_mse']:.6f} "
            f"val_raw_rmse={val_stats['raw_rmse']:.6f} "
            f"val_r={val_stats['r_mean']:.4f} "
            f"r_per_dim={[round(r, 4) for r in val_stats['r_per_dim']]} {marker}"
        )

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch was {best_epoch} "
                f"with val_norm_mse={best_val_norm_mse:.6f}"
            )
            break

    if best_state_dict is not None:
        probe.load_state_dict(best_state_dict)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "target_name": target_name,
            "probe_type": probe_type,
            "probe_state_dict": probe.state_dict(),
            "target_mean": target_mean.cpu(),
            "target_std": target_std.cpu(),
            "best_epoch": best_epoch,
            "best_val_norm_mse": best_val_norm_mse,
            "best_val_raw_mse": best_val_raw_mse,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "max_epochs": max_epochs,
            "patience": patience,
            "mlp_hidden_dim": mlp_hidden_dim,
            "mlp_num_hidden_layers": mlp_num_hidden_layers,
            "mlp_dropout": mlp_dropout,
            "probe_seed": probe_seed,
        },
        save_path,
    )
    print(f"Saved best {probe_type} probe to {save_path}")
    return probe


def train_linear_probe(*args, **kwargs):
    return train_probe(*args, probe_type="linear", **kwargs)
