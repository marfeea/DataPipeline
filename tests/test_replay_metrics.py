import numpy as np
import pytest

from DataPipe.replay_metrics import replay_error, summarize_replay


def test_replay_pose_errors_and_quaternion_sign():
    reference = {"q": [0] * 6, "gripper_position": [0, 0], "tcp_pose_wxyz": [0, 0, 0, 1, 0, 0, 0]}
    actual = {"q": [0.1] * 6, "gripper_position": [0.002, 0], "tcp_pose_wxyz": [0.003, 0.004, 0, -1, 0, 0, 0]}
    error = replay_error(reference, actual)
    assert error["tcp_position_error_m"] == pytest.approx(0.005)
    assert error["tcp_rotation_error_rad"] == 0
    assert error["joint_max_abs_rad"] == pytest.approx(0.1)
    assert error["gripper_max_abs_m"] == pytest.approx(0.002)
    actual["tcp_pose_wxyz"] = [0, 0, 0, 0, 0, 0, 1]
    assert replay_error(reference, actual)["tcp_rotation_error_rad"] == pytest.approx(np.pi)
    assert summarize_replay([error, error])["metrics"]["tcp_position_error_m"]["rms"] == pytest.approx(0.005)
    assert summarize_replay([])["compared_frames"] == 0


def test_invalid_replay_quaternion_is_not_reported_as_zero_error():
    reference = {"q": [0] * 6, "gripper_position": [0, 0], "tcp_pose_wxyz": [0] * 7}
    with pytest.raises(ValueError, match="unit norm"):
        replay_error(reference, reference)
