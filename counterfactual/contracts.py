"""Explicit, testable contracts shared by all Phase I stages.

The world model was trained on three image frames and five *standardized* raw
actions per model step.  Keeping that convention in one module prevents the
counterfactual experiment from accidentally inheriting the raw-action rollout
path used by some earlier diagnostics.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TimeActionContract:
    """Mapping between Push-T raw steps and LeWM predictor steps."""

    history_size: int = 3
    frameskip: int = 5
    action_dim: int = 2
    latent_dim: int = 192

    def __post_init__(self):
        if self.history_size < 1:
            raise ValueError("history_size must be positive")
        if self.frameskip < 1:
            raise ValueError("frameskip must be positive")
        if self.action_dim < 1:
            raise ValueError("action_dim must be positive")

    @property
    def context_steps(self) -> torch.Tensor:
        return torch.arange(self.history_size, dtype=torch.long) * self.frameskip

    @property
    def first_prediction_step(self) -> int:
        return self.history_size * self.frameskip

    @property
    def token_dim(self) -> int:
        return self.frameskip * self.action_dim

    def rollout_indices(self, lookahead_steps: int) -> dict[str, torch.Tensor]:
        """Return h=0,5,10 context and h=15,... targets for a requested k."""
        if lookahead_steps < 1:
            raise ValueError("lookahead_steps must be at least one")

        # h=15 is predicted after the third action token (10→15).  Therefore
        # k predictions require the three context tokens plus only k-1 future
        # tokens; the final k=5 target h=35 follows the 30→35 token.
        action_steps = self.history_size + lookahead_steps - 1
        return {
            "context_steps": self.context_steps,
            "target_steps": torch.arange(
                self.first_prediction_step,
                self.first_prediction_step + lookahead_steps * self.frameskip,
                self.frameskip,
                dtype=torch.long,
            ),
            "action_starts": torch.arange(
                0, action_steps * self.frameskip, self.frameskip, dtype=torch.long
            ),
            "action_ends": torch.arange(
                self.frameskip,
                (action_steps + 1) * self.frameskip,
                self.frameskip,
                dtype=torch.long,
            ),
        }

    def as_dict(self) -> dict:
        indices = self.rollout_indices(lookahead_steps=5)
        rollout_coordinates = [
            {
                "k": index + 1,
                "target_h": int(step),
                "preceding_action_token": index + self.history_size - 1,
                "preceding_action_raw_steps": [
                    int(indices["action_starts"][index + self.history_size - 1]),
                    int(indices["action_ends"][index + self.history_size - 1]),
                ],
            }
            for index, step in enumerate(indices["target_steps"])
        ]
        return {
            "history_size": self.history_size,
            "frameskip": self.frameskip,
            "action_dim": self.action_dim,
            "token_dim": self.token_dim,
            "latent_dim": self.latent_dim,
            "context_steps": indices["context_steps"].tolist(),
            "first_prediction_step": self.first_prediction_step,
            "phase1_target_steps": indices["target_steps"].tolist(),
            "phase1_action_starts": indices["action_starts"].tolist(),
            "phase1_action_ends": indices["action_ends"].tolist(),
            "phase1_rollout_coordinates": rollout_coordinates,
        }


@dataclass(frozen=True)
class ActionNormalizer:
    """Training-time Push-T action standardization with safe tokenization."""

    mean: torch.Tensor
    std: torch.Tensor

    def __post_init__(self):
        mean = torch.as_tensor(self.mean, dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(self.std, dtype=torch.float32).reshape(-1)
        if mean.numel() != std.numel():
            raise ValueError("action mean and std must have the same dimension")
        if mean.numel() != 2:
            raise ValueError("Push-T action contract requires two action coordinates")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("action statistics must be finite")
        if torch.any(std <= 0):
            raise ValueError("action standard deviations must be positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    def normalize_rows(self, raw_actions: torch.Tensor) -> torch.Tensor:
        """Standardize (..., 2) raw simulator actions in model-training order."""
        raw_actions = torch.as_tensor(raw_actions)
        if raw_actions.shape[-1] != self.mean.numel():
            raise ValueError(
                f"Expected raw actions ending in {self.mean.numel()} coordinates, got "
                f"{tuple(raw_actions.shape)}"
            )
        mean = self.mean.to(device=raw_actions.device, dtype=torch.float32)
        std = self.std.to(device=raw_actions.device, dtype=torch.float32)
        return (raw_actions.float() - mean) / std

    def tokenize(self, raw_actions: torch.Tensor, frameskip: int) -> torch.Tensor:
        """Normalize raw actions before flattening blocks into LeWM tokens.

        Accepted shapes are ``(T, 2)`` and ``(B, T, 2)``.  The output is
        ``(T/frameskip, frameskip*2)`` or ``(B, T/frameskip, frameskip*2)``.
        """
        if frameskip < 1:
            raise ValueError("frameskip must be positive")
        actions = torch.as_tensor(raw_actions)
        if actions.ndim not in (2, 3) or actions.shape[-1] != self.mean.numel():
            raise ValueError(
                "raw_actions must have shape (T, 2) or (B, T, 2), got "
                f"{tuple(actions.shape)}"
            )
        if actions.shape[-2] % frameskip:
            raise ValueError(
                f"Raw action length {actions.shape[-2]} is not divisible by "
                f"frameskip={frameskip}"
            )

        standardized = self.normalize_rows(actions)
        token_shape = (*standardized.shape[:-2], -1, frameskip * self.mean.numel())
        return standardized.reshape(token_shape).contiguous()

    def as_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "estimator": "torch.mean / torch.std(unbiased=True) over non-NaN raw actions",
            "order": "standardize each raw (x, y) action before grouping five rows",
        }
