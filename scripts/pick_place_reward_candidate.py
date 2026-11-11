"""Opt-in pick/place reward proposal; never imported by a collector or writer.

All poses are measured in one world frame, quaternions are wxyz. Offline
logical-transfer traces may be scored, but must not become valid RL labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

VERSION = "pick-place-candidate/1"


def finite_vector(value, size):
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"expected finite vector of size {size}")
    return vector


def rotation_error(first, second):
    """Shortest SO(3) angle, invariant to quaternion sign and normalization."""
    a, b = finite_vector(first, 4), finite_vector(second, 4)
    if np.linalg.norm(a) < 1e-12 or np.linalg.norm(b) < 1e-12:
        raise ValueError("zero quaternion")
    return float(2 * np.arccos(np.clip(abs(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)), 0, 1)))


@dataclass(frozen=True)
class RewardConfig:
    reach_weight: float = 2.0
    lift_weight: float = 2.0
    position_weight: float = 4.0
    orientation_weight: float = 2.0
    grasp_bonus: float = 2.0
    success_bonus: float = 10.0
    time_cost_per_s: float = 0.1
    speed_cost: float = 0.01
    geometry_overlap_cost: float = 5.0
    collision_cost: float = 25.0
    limit_cost: float = 10.0
    drop_cost: float = 10.0
    timeout_cost: float = 15.0
    orientation_gate_m: float = 0.1
    position_tolerance_m: float = 0.02
    rotation_tolerance_rad: float = np.pi / 18
    settle_s: float = 0.25
    max_object_speed_m_s: float = 0.03
    max_object_angular_speed_rad_s: float = np.pi / 18
    closed_finger_m: float = 0.005
    open_finger_m: float = 0.00805

    def __post_init__(self):
        if not all(np.isfinite(x) and x > 0 for x in asdict(self).values()):
            raise ValueError("reward parameters must be finite and positive")


@dataclass(frozen=True)
class RewardTask:
    grasp_position_w: tuple
    source_position_w: tuple
    goal_pose_w: tuple
    lift_target_m: float = 0.05

    def __post_init__(self):
        finite_vector(self.grasp_position_w, 3)
        finite_vector(self.source_position_w, 3)
        goal = finite_vector(self.goal_pose_w, 7)
        rotation_error(goal[3:], goal[3:])
        if not np.isfinite(self.lift_target_m) or self.lift_target_m <= 0:
            raise ValueError("lift target must be positive")
        if np.linalg.norm(goal[:3] - self.source_position_w) < 1e-9:
            raise ValueError("source and goal must differ")


@dataclass(frozen=True)
class RewardObservation:
    time_s: float
    tcp_position_w: tuple
    object_pose_w: tuple
    joint_velocity: tuple
    gripper_position: tuple
    phase: str
    # A live adapter must establish these from measured evidence, not plan flags.
    grasp_verified: bool = False
    support_verified: bool = False
    physical_dynamics: bool = False
    evidence_valid: bool = False
    object_linear_speed: float | None = None
    object_angular_speed: float | None = None
    geometry_overlap: bool = False
    illegal_contact: bool = False
    joint_limit: bool = False
    dropped: bool = False
    timeout: bool = False

    def __post_init__(self):
        if not np.isfinite(self.time_s):
            raise ValueError("time must be finite")
        finite_vector(self.tcp_position_w, 3)
        pose = finite_vector(self.object_pose_w, 7)
        rotation_error(pose[3:], pose[3:])
        finite_vector(self.joint_velocity, 6)
        finite_vector(self.gripper_position, 2)
        for value in (self.object_linear_speed, self.object_angular_speed):
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError("speed must be finite and nonnegative")
        for name in (
            "grasp_verified",
            "support_verified",
            "physical_dynamics",
            "evidence_valid",
            "geometry_overlap",
            "illegal_contact",
            "joint_limit",
            "dropped",
            "timeout",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.phase not in {"reach", "close", "lift", "transport", "place", "release"}:
            raise ValueError("unknown reward phase")


class PickPlaceReward:
    """One instance per episode. step(next_obs) scores (obs, action, next_obs).

    Reach/lift frontiers pay at most once. Placement potentials are signed,
    so leaving the goal takes back credit. Time and speed are rates integrated
    in seconds, not per-frame constants. This is not policy-invariant PBRS.
    """

    def __init__(self, task: RewardTask, initial: RewardObservation, config: RewardConfig | None = None):
        self.task, self.config = task, config or RewardConfig()
        self.previous = initial
        self.initial_reach_distance = max(
            float(np.linalg.norm(np.asarray(initial.tcp_position_w) - task.grasp_position_w)), 1e-9
        )
        self.initial_goal_distance = float(
            np.linalg.norm(np.asarray(task.source_position_w) - np.asarray(task.goal_pose_w[:3]))
        )
        self.reach_frontier = self.lift_frontier = 0.0
        self.position_potential = self.orientation_potential = 0.0
        self.lifted = self.grasp_paid = self.overlap_paid = self.done = False
        self.settled_s = 0.0
        self.previous_stable = False

    def _stable(self, obs, position_error, angle):
        c = self.config
        return bool(
            self.grasp_paid
            and obs.phase == "release"
            and obs.support_verified
            and obs.physical_dynamics
            and obs.evidence_valid
            and min(abs(x) for x in obs.gripper_position) >= c.open_finger_m
            and position_error <= c.position_tolerance_m
            and angle <= c.rotation_tolerance_rad
            and obs.object_linear_speed is not None
            and obs.object_linear_speed <= c.max_object_speed_m_s
            and obs.object_angular_speed is not None
            and obs.object_angular_speed <= c.max_object_angular_speed_rad_s
        )

    def step(self, obs: RewardObservation):
        if self.done:
            raise ValueError("episode already terminated")
        dt = obs.time_s - self.previous.time_s
        if dt <= 0:
            raise ValueError("observation time must strictly increase")
        c, task = self.config, self.task
        terms = dict.fromkeys(
            ("reach", "lift", "position", "orientation", "grasp", "success", "time", "speed", "overlap", "failure"),
            0.0,
        )
        d = float(np.linalg.norm(np.asarray(obs.object_pose_w[:3]) - task.goal_pose_w[:3]))
        angle = rotation_error(obs.object_pose_w[3:], task.goal_pose_w[3:])
        height = obs.object_pose_w[2] - task.source_position_w[2]
        if obs.phase in {"reach", "close"}:
            distance = np.linalg.norm(np.asarray(obs.tcp_position_w) - task.grasp_position_w)
            frontier = max(self.reach_frontier, float(np.clip(1 - distance / self.initial_reach_distance, 0, 1)))
            terms["reach"] = c.reach_weight * (frontier - self.reach_frontier)
            self.reach_frontier = frontier
        if obs.grasp_verified and obs.physical_dynamics and obs.evidence_valid and not self.grasp_paid:
            terms["grasp"] = c.grasp_bonus
            self.grasp_paid = True
        # Closed jaws plus object rise is an explicitly provisional offline proxy.
        lift_evidence = self.grasp_paid or not obs.evidence_valid
        if obs.phase == "lift" and lift_evidence and max(abs(x) for x in obs.gripper_position) <= c.closed_finger_m:
            frontier = max(self.lift_frontier, float(np.clip(height / task.lift_target_m, 0, 1)))
            terms["lift"] = c.lift_weight * (frontier - self.lift_frontier)
            self.lift_frontier = frontier
            self.lifted = self.lifted or frontier >= 0.5
        if self.lifted and obs.phase in {"transport", "place", "release"}:
            position = float(np.clip(1 - d / self.initial_goal_distance, 0, 1))
            orientation = float(np.clip(1 - d / c.orientation_gate_m, 0, 1) * (1 - angle / np.pi))
            terms["position"] = c.position_weight * (position - self.position_potential)
            terms["orientation"] = c.orientation_weight * (orientation - self.orientation_potential)
            self.position_potential, self.orientation_potential = position, orientation
        terms["time"] = -c.time_cost_per_s * dt
        # Trapezoid integral of measured squared joint speed; not electrical energy.
        terms["speed"] = (
            -c.speed_cost
            * dt
            * 0.5
            * (
                np.dot(obs.joint_velocity, obs.joint_velocity)
                + np.dot(self.previous.joint_velocity, self.previous.joint_velocity)
            )
        )
        if obs.geometry_overlap and not self.overlap_paid:
            terms["overlap"] = -c.geometry_overlap_cost
            self.overlap_paid = True
        stable = self._stable(obs, d, angle)
        self.settled_s = self.settled_s + dt if stable and self.previous_stable else 0.0
        self.previous_stable = stable
        failure = next((name for name in ("illegal_contact", "joint_limit", "dropped") if getattr(obs, name)), None)
        if failure is not None:
            cost = {"illegal_contact": c.collision_cost, "joint_limit": c.limit_cost, "dropped": c.drop_cost}[failure]
            terms["failure"] = -cost
            # Safety failure cancels every positive contribution on this transition.
            terms = {key: min(0.0, value) for key, value in terms.items()}
            self.done = True
        elif self.settled_s + 1e-12 >= c.settle_s:
            terms["success"] = c.success_bonus
            self.done = True
        elif obs.timeout:
            terms["failure"] = -c.timeout_cost
            self.done = True
            failure = "timeout"
        valid = bool(
            self.previous.evidence_valid
            and obs.evidence_valid
            and self.previous.physical_dynamics
            and obs.physical_dynamics
        )
        self.previous = obs
        return {
            "reward": float(sum(terms.values())),
            "reward_valid": valid,
            "terms": {key: float(value) for key, value in terms.items()},
            "position_error_m": d,
            "rotation_error_rad": angle,
            "vertical_lift_m": float(height),
            "settled_s": self.settled_s,
            "terminated": self.done,
            "failure": failure,
        }
