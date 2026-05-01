import torch
import torch.nn.functional as F


# Component: probe evaluation metrics.
#
# This module owns metrics used to judge how well a probe decodes physical
# quantities from frozen LeWM embeddings. During training the probe predicts
# normalized targets for stable optimization, but for interpretation we also
# unnormalize predictions and report raw MSE/RMSE plus Pearson correlation.
#
# Current metrics:
# - norm_mse: MSE on normalized targets, useful for Table 1-style comparison.
# - raw_mse/raw_rmse: error in the original target units, useful for intuition.
# - Pearson r: per-dimension correlation between predicted and true quantities.


def _make_pair_loader(pairs, batch_size, shuffle):
    tensor_dataset = torch.utils.data.TensorDataset(
        pairs["embeddings"],
        pairs["target"],
    )
    return torch.utils.data.DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def pearson_r(pred, target):
    pred_centered = pred - pred.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    numerator = (pred_centered * target_centered).sum(dim=0)
    denominator = torch.sqrt(
        pred_centered.square().sum(dim=0) * target_centered.square().sum(dim=0)
    ).clamp_min(1e-8)
    r_per_dim = numerator / denominator
    return r_per_dim.mean(), r_per_dim


def evaluate_linear_probe(
    pairs,
    probe,
    target_mean,
    target_std,
    batch_size,
):
    loader = _make_pair_loader(pairs, batch_size=batch_size, shuffle=False)
    probe.eval()

    pred_norms = []
    target_norms = []
    preds = []
    targets = []
    with torch.no_grad():
        for emb, target in loader:
            pred_norm = probe(emb)  # Probe predicts normalized target values.
            target_norm = (target - target_mean) / target_std
            pred = pred_norm * target_std + target_mean  # Convert back to raw units.
            pred_norms.append(pred_norm.cpu())
            target_norms.append(target_norm.cpu())
            preds.append(pred.cpu())
            targets.append(target.cpu())

    pred_norm = torch.cat(pred_norms, dim=0)
    target_norm = torch.cat(target_norms, dim=0)
    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    norm_mse = F.mse_loss(pred_norm, target_norm)
    raw_mse = F.mse_loss(pred, target)
    r_mean, r_per_dim = pearson_r(pred, target)
    return {
        "norm_mse": norm_mse.item(),
        "raw_mse": raw_mse.item(),
        "raw_rmse": torch.sqrt(raw_mse).item(),
        "r_mean": r_mean.item(),
        "r_per_dim": r_per_dim.tolist(),
    }
