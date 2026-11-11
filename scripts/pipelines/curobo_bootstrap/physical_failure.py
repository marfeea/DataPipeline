"""Evidence and storage for the isolated, attachment-free physics experiment.

The original hybrid NPZ fields are retained; observations and transition rewards
are sidecars. Planned PICK/PLACE and object_attached are never outcome evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from scripts.pick_place_reward_candidate import RewardObservation
from scripts.pick_place_reward_candidate_v2 import VERSION, PickPlaceRewardV2

from .pick_place_artifact import FIELDS, save_hybrid_trajectory, save_json
from .robot_adapter import pose_matrix


@dataclass(frozen=True)
class EvidenceConfig:
    grasp_hold_s: float = 0.25
    loss_hold_s: float = 0.05
    translation_tolerance_m: float = 0.005
    rotation_tolerance_rad: float = np.deg2rad(5)
    lift_clearance_m: float = 0.02
    drop_height_m: float = 0.01
    contact_force_threshold_n: float = 0.01


def relative_pose(tcp_pose, object_pose):
    def matrix(pose):
        q = np.asarray(pose[3:])
        return pose_matrix(pose[:3], q / np.linalg.norm(q))

    return np.linalg.inv(matrix(tcp_pose)) @ matrix(object_pose)


class GraspEvidence:
    """Require two measured finger contacts, lift and a stable relative pose.

    Contact tensors must also have observed positive support and both finger
    controls before their absence can certify failure. Missing data is unknown.
    """

    def __init__(self, initial, config=None):
        self.config = config or EvidenceConfig()
        self.initial = initial
        self.candidate = None
        self.candidate_time = None
        self.ever_grasped = False
        self.verified_relative = None
        self.last_held_z = None
        self.loss_time = None
        self.seen_channels = np.zeros(3, dtype=bool)
        self.dropped = False
        self.supported_release = False
        self.last_time = None

    def step(self, sample):
        c = self.config
        time = sample["time_s"]
        if self.last_time is not None and time <= self.last_time:
            raise ValueError("evidence timestamps must increase")
        self.last_time = time
        available = sample["contact_available"]
        forces = np.asarray(sample["contact_force_n"])
        if available:
            if forces.shape != (3,) or not np.isfinite(forces).all():
                raise ValueError("expected three finite contact force magnitudes")
            self.seen_channels |= forces > c.contact_force_threshold_n
        both = bool(available and np.all(forces[:2] > c.contact_force_threshold_n))
        supported = bool(available and forces[2] > c.contact_force_threshold_n)
        relative = relative_pose(sample["tcp_pose_w"], sample["target_pose_w"])
        # Once intended release has established goal support, settling/contact
        # flicker must not turn it into a post-grasp drop on a later frame.
        self.supported_release = self.supported_release or sample.get("normal_supported_release", False)
        height = sample["target_pose_w"][2] - self.initial["target_pose_w"][2]

        def close_to(reference):
            delta = reference[:3, :3].T @ relative[:3, :3]
            angle = float(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1)))
            return bool(
                np.linalg.norm(relative[:3, 3] - reference[:3, 3]) <= c.translation_tolerance_m
                and angle <= c.rotation_tolerance_rad
            )

        candidate = both and not supported and height >= c.lift_clearance_m
        if candidate:
            if self.candidate is None or not close_to(self.candidate):
                self.candidate, self.candidate_time = relative, time
            if time - self.candidate_time + 1e-12 >= c.grasp_hold_s:
                if not self.ever_grasped:
                    self.verified_relative = self.candidate.copy()
                self.ever_grasped = True
        else:
            self.candidate = self.candidate_time = None
        held = bool(self.ever_grasped and both and close_to(self.verified_relative))
        if held:
            self.last_held_z = sample["target_pose_w"][2]
            self.loss_time = None
        elif self.ever_grasped and available:
            if self.loss_time is None:
                self.loss_time = time
            downward = sample["target_velocity_w"][2] < -0.01
            fallen = self.last_held_z is not None and self.last_held_z - sample["target_pose_w"][2] >= c.drop_height_m
            # Support already established at a stationary placement is not a drop.
            normal_release = self.supported_release
            if (
                time - self.loss_time + 1e-12 >= c.loss_hold_s
                and fallen
                and downward
                and not close_to(self.verified_relative)
                and not normal_release
            ):
                self.dropped = True
        else:
            self.loss_time = None
        return {
            "grasp_verified": held,
            "ever_grasp_verified": self.ever_grasped,
            "support_verified": supported,
            "dropped": self.dropped,
            "evidence_valid": bool(available and self.seen_channels.all()),
            "contact_positive_controls": self.seen_channels.tolist(),
            "relative_object_transform_tcp": relative.tolist(),
            "vertical_lift_m": float(height),
        }


def reward_observation(sample, evidence, *, timeout=False):
    return RewardObservation(
        time_s=sample["time_s"],
        tcp_position_w=tuple(sample["tcp_pose_w"][:3]),
        object_pose_w=tuple(sample["target_pose_w"]),
        joint_velocity=tuple(sample["dq"]),
        gripper_position=tuple(sample["gripper_position"]),
        phase=sample["phase"],
        physical_dynamics=True,
        evidence_valid=evidence["evidence_valid"],
        grasp_verified=evidence["grasp_verified"],
        support_verified=evidence["support_verified"],
        dropped=evidence["dropped"],
        timeout=timeout,
        joint_limit=bool(sample.get("joint_limit", False)),
        object_linear_speed=float(np.linalg.norm(sample["target_velocity_w"][:3])),
        object_angular_speed=float(np.linalg.norm(sample["target_velocity_w"][3:])),
    )


def write_artifacts(output, records, rewards, task, reward_task, report):
    """N observations and N-1 transitions, including an explicit terminal/tail mask."""
    output = Path(output)
    if not records:
        return
    n = len(records)
    if len(rewards) != n - 1:
        raise ValueError("reward/observation alignment must be T/T+1")
    dq = np.asarray([r["dq"] for r in records])
    times = np.asarray([r["time_s"] for r in records])
    ddq = np.zeros_like(dq)
    if n > 1:
        ddq[1:] = np.diff(dq, axis=0) / np.diff(times)[:, None]
    arrays = {
        "time_s": times,
        "joint_position": np.asarray([r["q"] for r in records]),
        "joint_velocity": dq,
        "joint_acceleration": ddq,
        "joint_names": np.asarray(report["joint_names"]),
        "stage_id": np.asarray([r["stage_id"] for r in records]),
        "segment_source": np.full(n, "physical_target"),
        "event_code": np.asarray([r["event_code"] for r in records]),
        "gripper_target": np.asarray([r["executed_gripper_target"] for r in records]),
        "object_attached": np.zeros(n, dtype=bool),
    }
    assert set(arrays) == FIELDS
    save_hybrid_trajectory(output / "trajectory.npz", arrays)
    save_json(output / "trace.json", records)
    save_json(output / "rewards.json", rewards)
    np.savez_compressed(
        output / "rewards.npz",
        reward=np.asarray([r["reward"] for r in rewards]),
        reward_valid=np.asarray([r["reward_valid"] for r in rewards], dtype=bool),
        terminated=np.asarray([r["terminated"] for r in rewards], dtype=bool),
        transition_valid=np.asarray([r["transition_valid"] for r in rewards], dtype=bool),
        action_joint_target=np.asarray([r["commanded_joint_target"] for r in records[1:]]).reshape(-1, 6),
        action_gripper_target=np.asarray([r["executed_gripper_target"] for r in records[1:]]).reshape(-1, 2),
        from_observation=np.arange(n - 1),
        to_observation=np.arange(1, n),
    )
    np.savez_compressed(
        output / "observations.npz",
        time_s=times,
        target_pose_w=np.asarray([r["target_pose_w"] for r in records]),
        target_velocity_w=np.asarray([r["target_velocity_w"] for r in records]),
        tcp_pose_w=np.asarray([r["tcp_pose_w"] for r in records]),
        gripper_position=np.asarray([r["gripper_position"] for r in records]),
        gripper_velocity=np.asarray([r["gripper_velocity"] for r in records]),
        contact_force_n=np.asarray([r["contact_force_n"] for r in records]),
        contact_available=np.asarray([r["contact_available"] for r in records], dtype=bool),
        grasp_verified=np.asarray([r["evidence"]["grasp_verified"] for r in records], dtype=bool),
    )
    save_json(
        output / "scene_snapshot.json",
        {
            "schema": "physical_failure_scene_snapshot/1",
            "task": task,
            "reference_snapshot": report["reference_snapshot"],
            "no_logical_attachment": True,
            "placement_profile": report.get("placement_profile"),
        },
    )
    report.update(
        reward_version=VERSION,
        reward_task=asdict(reward_task),
        samples=n,
        transitions=n - 1,
        reward_return=float(sum(r["reward"] for r in rewards)),
        valid_reward_transitions=sum(r["reward_valid"] for r in rewards),
        evidence_config=asdict(EvidenceConfig()),
        data_contract="original hybrid NPZ fields; measured observations and T rewards in sidecars",
    )


def make_reward_model(task, initial):
    return PickPlaceRewardV2(task, initial)
