"""Pure geometry and measured acceptance for vertical placement experiments."""

from dataclasses import asdict, dataclass
from itertools import product

import numpy as np

from scripts.pick_place_reward_candidate import RewardConfig, rotation_error
from scripts.pipelines.curobo_bootstrap.physical_failure import relative_pose
from scripts.pipelines.curobo_bootstrap.replay import matrix_pose_wxyz
from scripts.pipelines.curobo_bootstrap.robot_adapter import pose_matrix


@dataclass(frozen=True)
class PlacementCandidate:
    destination: int = 2
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    descent_speed_m_s: float = 0.06
    final_speed_m_s: float = 0.01
    hold_s: float = 0.5
    open_s: float = 0.3
    ik_seed_index: int = 0
    high_clearance_m: float = 0.0
    tracking_compensation: bool = False

    def __post_init__(self):
        if self.destination not in range(5):
            raise ValueError("destination must be 0..4")
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("candidate values must be finite")
        if min(self.descent_speed_m_s, self.final_speed_m_s, self.hold_s, self.open_s) <= 0:
            raise ValueError("speeds and durations must be positive")
        if type(self.ik_seed_index) is not int or self.ik_seed_index < 0 or self.high_clearance_m < 0:
            raise ValueError("seed index must be a nonnegative integer and clearance nonnegative")

    def payload(self):
        return asdict(self)


def tcp_release_pose(goal_object_pose_w, tcp_from_object, candidate):
    """Offsets are world XYZ; rotations are around the object's local axes."""
    goal = np.asarray(goal_object_pose_w, dtype=float)
    desired = pose_matrix(goal[:3], goal[3:] / np.linalg.norm(goal[3:]))
    r, p, y = np.deg2rad([candidate.roll_deg, candidate.pitch_deg, candidate.yaw_deg])
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    desired[:3, :3] = desired[:3, :3] @ rz @ ry @ rx
    desired[:3, 3] += [candidate.x_m, candidate.y_m, candidate.z_m]
    return desired @ np.linalg.inv(np.asarray(tcp_from_object)), desired


def coarse_candidates(destination):
    # First isolate release height; then probe horizontal sensitivity.
    for z in (0, 0.003, -0.003, 0.006, -0.006, 0.01, -0.01):
        yield PlacementCandidate(destination=destination, z_m=z)
    for x, y in product((0, -0.004, 0.004, -0.008, 0.008), repeat=2):
        if x or y:
            yield PlacementCandidate(destination=destination, x_m=x, y_m=y)


def placement_metrics(sample, goal, *, grasp_verified, config=None):
    """Reuse the existing success thresholds without treating TCP arrival as success."""
    c = config or RewardConfig()
    pose = np.asarray(sample["object_pose_w"])
    velocity = np.asarray(sample["object_velocity_w"])
    position = float(np.linalg.norm(pose[:3] - np.asarray(goal[:3])))
    angle = float(rotation_error(pose[3:], goal[3:]))
    linear, angular = float(np.linalg.norm(velocity[:3])), float(np.linalg.norm(velocity[3:]))
    fingers = np.asarray(sample["finger_position"])
    forces = np.asarray(sample["contact_force_n"])
    checks = dict(
        grasp_verified=bool(grasp_verified),
        contact_available=bool(sample["contact_available"]),
        gripper_open=bool(np.min(np.abs(fingers)) >= c.open_finger_m),
        support=bool(forces[2] > 0.01),
        fingers_detached=bool(np.max(forces[:2]) <= 0.01),
        position=bool(position <= c.position_tolerance_m),
        orientation=bool(angle <= c.rotation_tolerance_rad),
        linear_speed=bool(linear <= c.max_object_speed_m_s),
        angular_speed=bool(angular <= c.max_object_angular_speed_rad_s),
        joint_limits=not sample.get("joint_limit", False),
        illegal_contact=not sample.get("illegal_contact", False),
    )
    return dict(
        position_error_m=position,
        rotation_error_rad=angle,
        linear_speed_m_s=linear,
        angular_speed_rad_s=angular,
        checks=checks,
        stable=all(checks.values()),
        score=position / c.position_tolerance_m
        + angle / c.rotation_tolerance_rad
        + min(linear / c.max_object_speed_m_s, 20)
        + min(angular / c.max_object_angular_speed_rad_s, 20),
    )


def continuous_stable_s(samples):
    """Duration of the final uninterrupted passing window (irregular clocks supported)."""
    start = None
    previous = None
    for sample in samples:
        time = sample["time_s"]
        if previous is not None and time <= previous:
            raise ValueError("sample times must increase")
        previous = time
        start = (time if start is None else start) if sample["metrics"]["stable"] else None
    return 0.0 if start is None else previous - start


def carrying_metrics(sample, goal, reference_relative):
    """Distinguish intended touchdown from premature collision or grasp loss."""
    relative = relative_pose(sample["tcp_pose_w"], sample["object_pose_w"])
    translation = float(np.linalg.norm(relative[:3, 3] - reference_relative[:3, 3]))
    angle = rotation_error(pose_list(relative)[3:], pose_list(reference_relative)[3:])
    forces = np.asarray(sample["contact_force_n"])
    supported = bool(sample["contact_available"] and forces[2] > 0.01)
    held = bool(
        sample["contact_available"] and min(forces[:2]) > 0.01 and translation <= 0.005 and angle <= np.deg2rad(5)
    )
    distance = float(np.linalg.norm(np.asarray(sample["object_pose_w"][:3]) - goal[:3]))
    return dict(
        held=held,
        supported=supported,
        relative_translation_drift_m=translation,
        relative_rotation_drift_rad=angle,
        unsupported_loss=not held and not supported,
        premature_support=supported and distance > RewardConfig().position_tolerance_m,
    )


def pose_list(matrix):
    return list(matrix_pose_wxyz(matrix))
