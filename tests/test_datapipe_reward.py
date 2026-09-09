from __future__ import annotations

import numpy as np
import pytest

from DataPipe import RewardLossConfig, compute_reward_prediction_loss, smooth_l1_reward_error


def test_smooth_l1_matches_reference_piecewise_formula() -> None:
    result = smooth_l1_reward_error(np.array([0.0, 0.0]), np.array([0.5, 2.0]))
    np.testing.assert_allclose(result, np.array([0.125, 1.5]))


def test_unweighted_reward_prediction_loss_is_mean_smooth_l1() -> None:
    assert compute_reward_prediction_loss([0.0, 0.0], [0.5, 2.0]) == pytest.approx(0.8125)


def test_weighted_reward_prediction_loss_matches_mainlogic_weighting() -> None:
    config = RewardLossConfig(positive_threshold=0.5, positive_weight=3.0, continuous_weight=1.0)
    # Target 1.0 gets weight 1 + 1*1 + (3-1) = 4; target 0 gets weight 1.
    assert compute_reward_prediction_loss([0.0, 0.0], [1.0, 0.0], config) == pytest.approx(0.4)


def test_reward_prediction_loss_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        compute_reward_prediction_loss([np.nan], [0.0])
