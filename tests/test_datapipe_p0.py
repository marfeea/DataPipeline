from __future__ import annotations

import sys
from math import pi
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from DataPipe import (  # noqa: E402
    ACTION_DIMENSION,
    ARM_ACTION_DIMENSION,
    ActionRecord,
    EpisodeBuilder,
    TerminationReason,
    p0_field_specs,
)


def make_action(value: float = 0.25) -> ActionRecord:
    return ActionRecord(
        raw_action=(value,) * ACTION_DIMENSION,
        action=(value,) * ACTION_DIMENSION,
        tcp_displacement=(0.001,) * 3 + (0.01,) * 3,
        ik_output=(0.1,) * ARM_ACTION_DIMENSION,
        joint_position=(0.2,) * ARM_ACTION_DIMENSION,
        joint_velocity=(0.3,) * ARM_ACTION_DIMENSION,
        tcp_heading_target_rad=value * pi,
    )


def make_episode(reason: TerminationReason):
    builder = EpisodeBuilder(f"episode-{reason.value}")
    builder.start("o0")
    builder.append(make_action(), 1.0, "o1")
    builder.append(make_action(0.5), 2.0, "o2")
    builder.append(make_action(-0.5), 3.0, "o3")
    return builder.close(reason, "o3-terminal")


def test_p0_1_uses_t_plus_one_observations_and_fixed_timeline() -> None:
    episode = make_episode(TerminationReason.SUCCESS)

    assert len(episode.observations) == 4
    assert len(episode.steps) == 3
    assert episode.timeline() == ("o0", "a0", "r0", "o1", "a1", "r1", "o2", "a2", "r2", "o3")
    assert episode.terminal_observation == "o3-terminal"


@pytest.mark.parametrize("reason", list(TerminationReason))
def test_p0_2_encodes_success_timeout_and_safety_boundaries(reason: TerminationReason) -> None:
    episode = make_episode(reason)

    assert episode.observations[0].boundary.is_first
    assert all(not frame.boundary.is_first for frame in episode.observations[1:])
    assert episode.observations[-1].boundary.is_last
    assert episode.observations[-1].boundary.is_terminal is reason.is_terminal
    assert all(not frame.boundary.is_last for frame in episode.observations[:-1])


def test_reset_is_rejected_until_terminal_observation_is_committed() -> None:
    builder = EpisodeBuilder("episode-reset-order")
    builder.start("o0")
    with pytest.raises(RuntimeError, match="terminal observation"):
        builder.reset()

    builder.append(make_action(), 0.0, "o1")
    episode = builder.close(TerminationReason.TIMEOUT, "o1-terminal")
    builder.reset()
    assert episode.terminal_observation == "o1-terminal"


def test_p0_3_keeps_action_mapping_and_six_joint_order() -> None:
    action = make_action()
    assert action.action == action.normalized_action
    assert len(action.raw_action) == ACTION_DIMENSION
    assert len(action.tcp_displacement) == ARM_ACTION_DIMENSION
    assert action.ik_output == (0.1,) * 6
    assert action.joint_position == (0.2,) * 6
    assert action.joint_velocity == (0.3,) * 6
    assert tuple(field.name for field in p0_field_specs()) == (
        "observation",
        "action",
        "reward",
        "raw_action",
        "normalized_action",
        "tcp_displacement",
        "ik_output",
        "joint_position",
        "joint_velocity",
        "gripper_position",
        "gripper_velocity",
        "is_first",
        "is_last",
        "is_terminal",
    )


def test_normalized_action_must_be_in_range_and_finite() -> None:
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        ActionRecord((0.0,) * 8, (1.01,) * 8, (0.0,) * 6, (0.0,) * 6, (0.0,) * 6, (0.0,) * 6)
    with pytest.raises(ValueError, match="NaN"):
        ActionRecord((0.0,) * 7 + (float("nan"),), (0.0,) * 8, (0.0,) * 6, (0.0,) * 6, (0.0,) * 6, (0.0,) * 6)
