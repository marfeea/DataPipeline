"""Reward-prediction loss extracted from the reference Mamba world model."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class RewardLossConfig:
    """Configuration matching ``MambaWorldModel.compute_loss`` reward weighting."""

    positive_threshold: float = 0.5
    positive_weight: float = 1.0
    continuous_weight: float = 0.0

    def validate(self) -> None:
        """Reject non-finite weighting parameters before loss computation."""

        values = (self.positive_threshold, self.positive_weight, self.continuous_weight)
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("reward loss configuration must contain finite values")


def _as_finite_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def smooth_l1_reward_error(predicted: Any, target: Any, *, beta: float = 1.0) -> np.ndarray:
    """Return elementwise PyTorch-default Smooth L1 error for reward targets."""

    if not isfinite(float(beta)) or beta <= 0.0:
        raise ValueError("beta must be finite and greater than zero")
    prediction = _as_finite_array(predicted, name="predicted reward")
    target_array = _as_finite_array(target, name="target reward")
    try:
        difference = np.abs(prediction - target_array)
    except ValueError as exc:
        raise ValueError("predicted and target rewards are not broadcast-compatible") from exc
    return np.where(
        difference < beta,
        0.5 * difference * difference / beta,
        difference - 0.5 * beta,
    )


def compute_reward_prediction_loss(
    predicted: Any,
    target: Any,
    config: RewardLossConfig | None = None,
    *,
    reward_valid: Any | None = None,
) -> float:
    """Compute the reference reward-head loss as a pure data-pipeline function.

    The reference model predicts a scalar reward and compares it with the stored
    ``batch["reward"]`` using elementwise Smooth L1.  When ``config`` is given,
    targets in ``[0, 1]`` receive an additional continuous weight and targets at
    or above ``positive_threshold`` receive ``positive_weight``.
    Pass the batch ``reward_valid`` mask for v4 data; invalid targets contribute
    neither numerator nor denominator, and an all-invalid batch returns zero.
    """

    error = smooth_l1_reward_error(predicted, target)
    target_array = _as_finite_array(target, name="target reward")
    mask = np.ones(error.shape, dtype=bool)
    if reward_valid is not None:
        supplied = np.asarray(reward_valid)
        if supplied.dtype.kind != "b":
            raise ValueError("reward_valid must be boolean")
        try:
            mask = np.broadcast_to(supplied, error.shape)
        except ValueError as exc:
            raise ValueError("reward_valid shape must broadcast to the reward error shape") from exc
    if config is None:
        return float(np.mean(error[mask])) if np.any(mask) else 0.0

    config.validate()
    weights = 1.0 + config.continuous_weight * np.clip(target_array, 0.0, 1.0)
    weights = weights + (config.positive_weight - 1.0) * (target_array >= config.positive_threshold).astype(np.float64)
    try:
        weighted_error = error * weights
    except ValueError as exc:
        raise ValueError("reward weighting shape is not broadcast-compatible") from exc
    weights = np.broadcast_to(weights, error.shape)
    denominator = max(float(np.sum(weights[mask])), 1.0)
    return float(np.sum(weighted_error[mask]) / denominator)


__all__ = [
    "RewardLossConfig",
    "compute_reward_prediction_loss",
    "smooth_l1_reward_error",
]
