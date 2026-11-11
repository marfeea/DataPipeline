"""Strict opt-in v4 extension; legacy v3 exports remain readable and writable."""

from __future__ import annotations

import numpy as np

from .alignment import pose_displacement, pose_matrix, rigid_transform

POSE_FIELDS = (
    "tcp_pose_b",
    "tcp_pose_w",
    "grasp_pose_o",
    "grasp_pose_b",
    "grasp_pose_w",
    "grasp_in_ee_pose_b",
    "grasp_in_ee_pose_w",
    "goal_object_pose_b",
    "goal_object_pose_w",
)
REQUIRED_FIELDS = (*POSE_FIELDS, "ee_twist_b", "ee_twist_w", "gripper_force", "T_w_b")
FILE_METADATA = {
    "position_unit": "meter",
    "angle_unit": "radian",
    "linear_velocity_unit": "meter_per_second",
    "angular_velocity_unit": "radian_per_second",
    "force_unit": "Newton",
    "torque_unit": "Newton_meter",
    "quaternion_order": "xyzw",
    "default_pose_frame": "robot_base",
    "transform_convention": "target_from_source",
    "reward_status": "not_defined",
    "reward_target_field": "rewards",
    "reward_valid": False,
    "action_schema": "tcp_delta_xyz_plus_tcp_angle_plus_gripper",
    "legacy_action_dimension": 8,
    "action_components": "dx,dy,dz,unused_rx,unused_ry,unused_rz,tcp_angle,gripper",
    "action_angle_semantics": "absolute_heading_normalized_by_pi",
    "action_rotation_scale_rad": float(np.pi),
}


def validate_observations(episode):
    """Fail before opening the partial shard; never synthesize missing sensors."""
    for frame in episode.observations:
        value = frame.value
        missing = set(REQUIRED_FIELDS).difference(value)
        if missing:
            raise ValueError(f"aligned v4 observation missing {sorted(missing)}")
        for name in POSE_FIELDS:
            pose_matrix(value[name])
        transform = rigid_transform(value["T_w_b"])
        for stem in ("tcp_pose", "grasp_pose", "goal_object_pose"):
            if not np.allclose(transform @ pose_matrix(value[stem + "_b"]), pose_matrix(value[stem + "_w"]), atol=2e-5):
                raise ValueError(f"{stem} base/world transform mismatch")
        for name in ("ee_twist_b", "ee_twist_w"):
            array = np.asarray(value[name])
            if array.shape != (6,) or not np.isfinite(array).all():
                raise ValueError(f"{name} must be a finite 6-vector")
        expected = np.r_[transform[:3, :3] @ value["ee_twist_b"][:3], transform[:3, :3] @ value["ee_twist_b"][3:]]
        if not np.allclose(expected, value["ee_twist_w"], atol=2e-5):
            raise ValueError("ee_twist base/world rotation mismatch")
        for suffix in ("b", "w"):
            from .collection_alignment import relative_pose_in_axes

            expected_relative = relative_pose_in_axes(value[f"tcp_pose_{suffix}"], value[f"grasp_pose_{suffix}"])
            if not np.allclose(
                pose_matrix(expected_relative), pose_matrix(value[f"grasp_in_ee_pose_{suffix}"]), atol=2e-5
            ):
                raise ValueError(f"grasp_in_ee_pose_{suffix} must retain the full relative SE(3)")
        force = np.asarray(value["gripper_force"])
        if force.shape not in ((1,), (2,)) or not np.isfinite(force).all():
            raise ValueError("gripper_force must be finite [1] or [2], in Newton")
    for step in episode.steps:
        if step.action.action_applied is None:
            raise ValueError("aligned v4 requires explicit action_applied")
        if any(abs(x) > 1e-7 for x in step.action.action[3:6]):
            raise ValueError("aligned v4 requires zero legacy rotation increments")
        if any(abs(x) > 1e-7 for x in step.action.action_applied[3:6]):
            raise ValueError("aligned v4 action_applied requires zero legacy rotation increments")


def write_alignment(group, episode):
    """Add v4 fields to the existing eight-component compatibility interface."""
    obs, diagnostics = group["obs"], group["diagnostics"]
    transforms = group.create_group("transforms")
    values = np.asarray(obs["T_w_b"])
    transforms.create_dataset(
        "T_w_b", data=values[0] if np.array_equal(values, np.broadcast_to(values[0], values.shape)) else values
    )
    del obs["T_w_b"]
    transforms["T_w_b"].attrs.update(
        {
            "axis": "episode" if transforms["T_w_b"].ndim == 2 else "T+1",
            "convention": "target_from_source",
            "source_frame": "robot_base",
            "target_frame": "world",
        }
    )
    if "camera_calibration" in group:
        for camera_name, camera in group["camera_calibration"].items():
            for convention in ("ros", "opengl"):
                legacy_name = f"extrinsic_quaternion_w_{convention}_wxyz"
                dataset = camera[legacy_name]
                dataset[...] = np.asarray(dataset)[:, [1, 2, 3, 0]]
                dataset.attrs.update(
                    {
                        "quaternion_order": "xyzw",
                        "axis": "T+1",
                        "frame": "world",
                        "axis_order": "qx,qy,qz,qw",
                        "source": "measured",
                        "unit": "unit_quaternion",
                    }
                )
                camera[f"extrinsic_quaternion_w_{convention}_xyzw"] = dataset
                del camera[legacy_name]
        obs["camera_calibration"] = group["camera_calibration"]
    group["action_commanded"] = group["actions"]
    group.create_dataset(
        "action_applied", data=np.asarray([s.action.action_applied for s in episode.steps], dtype=np.float32)
    )
    group.create_dataset("reward_valid", data=np.zeros(episode.length, dtype=np.bool_))
    for name in ("action_commanded", "action_applied"):
        diagnostics[name] = group[name]
        group[name].attrs.update(
            {"axis": "T", "frame": "robot_base", "unit": "normalized", "axis_order": FILE_METADATA["action_components"]}
        )
    diagnostics["joint_target_commanded"] = diagnostics["ik_solution_joint_position"]
    diagnostics["joint_target_applied"] = diagnostics["commanded_joint_target"]
    diagnostics.create_dataset(
        "safety_limit_flags",
        data=np.asarray(
            [(s.action.delta_limited, s.action.velocity_limited, s.action.position_limited) for s in episode.steps],
            dtype=np.bool_,
        ),
    )
    diagnostics["safety_limit_flags"].attrs["axis_order"] = "delta_limited,velocity_limited,position_limited"
    # The old field contains processed commands. Preserve it with an honest name.
    diagnostics["processed_tcp_command"] = diagnostics["tcp_displacement"]
    del diagnostics["tcp_displacement"]
    diagnostics.create_dataset(
        "tcp_displacement",
        data=np.asarray(
            [
                pose_displacement(a.value["tcp_pose_b"], b.value["tcp_pose_b"])
                for a, b in zip(episode.observations[:-1], episode.observations[1:], strict=True)
            ],
            dtype=np.float32,
        ),
    )
    diagnostics["tcp_displacement"].attrs.update(
        {
            "axis": "T",
            "frame": "robot_base",
            "source": "adjacent_observations",
            "unit": "meter,radian",
            "axis_order": "dx,dy,dz,rx,ry,rz",
        }
    )
    for name in POSE_FIELDS:
        frame = "world" if name.endswith("_w") else "object" if name.endswith("_o") else "robot_base"
        obs[name].attrs.update(
            {
                "axis": "T+1",
                "frame": frame,
                "unit": "meter,unit_quaternion",
                "axis_order": "x,y,z,qx,qy,qz,qw",
                "quaternion_order": "xyzw",
                "source": "measured" if name == "tcp_pose_b" else "derived",
            }
        )
        if name.startswith("grasp_in_ee"):
            obs[name].attrs["definition"] = "translation=grasp-ee; rotation=R_grasp @ R_ee.T; axes specified by frame"
    for name in ("ee_twist_b", "ee_twist_w"):
        obs[name].attrs.update(
            {
                "axis": "T+1",
                "frame": "robot_base" if name.endswith("_b") else "world",
                "unit": "meter_per_second,radian_per_second",
                "axis_order": "vx,vy,vz,wx,wy,wz",
                "reference_point": "tcp_origin",
                "source": "rigid_body_velocity_shifted_to_tcp",
            }
        )
    obs["gripper_force"].attrs.update(
        {
            "axis": "T+1",
            "frame": "gripper_joint_space",
            "unit": "Newton",
            "axis_order": "UpperFinger,DownFinger",
            "source": "physx_dof_projected_joint_force",
        }
    )
    for name in ("gripper_position", "gripper_velocity"):
        obs[name].attrs.update(
            {
                "axis": "T+1",
                "frame": "gripper_joint_space",
                "source": "measured",
                "unit": "meter" if name == "gripper_position" else "meter_per_second",
                "axis_order": "UpperFinger,DownFinger",
            }
        )
    for name in ("tcp_pose", "target_pose"):
        obs[name].attrs.update({
            "frame": "robot_base", "quaternion_order": "xyzw", "axis": "T+1",
            "unit": "meter,unit_quaternion", "axis_order": "x,y,z,qx,qy,qz,qw",
            "source": "measured" if name == "tcp_pose" else "controller_target",
        })
    obs["object_position_b"].attrs.update({"source": "target_config_proxy", "is_dynamic": False})
    group.attrs.update(
        {
            **FILE_METADATA,
            "object_state_source": "target_config_proxy",
            "object_state_is_dynamic": False,
            "goal_error_status": "unavailable_without_measured_object_state",
        }
    )


def validate_alignment(group):
    """Validate persisted v4 data, including transformed poses and measured deltas."""
    errors = []
    t = group["actions"].shape[0]
    for path, shape in (
        ("action_commanded", (t, 8)),
        ("action_applied", (t, 8)),
        ("diagnostics/action_commanded", (t, 8)),
        ("diagnostics/action_applied", (t, 8)),
        ("reward_valid", (t,)),
        ("diagnostics/joint_target_commanded", (t, 6)),
        ("diagnostics/joint_target_applied", (t, 6)),
        ("diagnostics/safety_limit_flags", (t, 3)),
    ):
        if path not in group or group[path].shape != shape:
            errors.append(f"{path} must have shape {shape}")
        elif not np.isfinite(group[path][...]).all():
            errors.append(f"{path} must be finite")
    diagnostics = group["diagnostics"]
    for name in ("action_commanded", "action_applied"):
        if name in group and name in diagnostics and not np.array_equal(group[name], diagnostics[name]):
            errors.append(f"diagnostics/{name} disagrees with {name}")
    if "safety_limit_flags" in diagnostics and diagnostics["safety_limit_flags"].shape == (t, 3):
        flags = np.asarray(diagnostics["safety_limit_flags"])
        if not np.isin(flags, (0, 1)).all():
            errors.append("safety_limit_flags must be boolean")
        for column, name in enumerate(("delta_limited", "velocity_limited", "position_limited")):
            if name not in diagnostics or not np.array_equal(flags[:, column], diagnostics[name]):
                errors.append(f"safety_limit_flags disagree with {name}")
    for canonical, legacy in (
        ("joint_target_commanded", "ik_solution_joint_position"),
        ("joint_target_applied", "commanded_joint_target"),
    ):
        if (
            canonical in diagnostics
            and legacy in diagnostics
            and not np.array_equal(diagnostics[canonical], diagnostics[legacy])
        ):
            errors.append(f"{canonical} disagrees with {legacy}")
    if "reward_valid" in group and np.any(group["reward_valid"]):
        errors.append("not_defined rewards must all have reward_valid=false")
    for key, value in FILE_METADATA.items():
        if group.attrs.get(key) != value:
            errors.append(f"aligned metadata {key} must equal {value!r}")
        if group.parent.attrs.get(key) != value:
            errors.append(f"aligned file metadata {key} must equal {value!r}")
    if "transforms/T_w_b" not in group:
        return (*errors, "missing transforms/T_w_b")
    transform = group["transforms/T_w_b"]
    if transform.shape not in ((4, 4), (t + 1, 4, 4)):
        return (*errors, "T_w_b must have shape [4,4] or [T+1,4,4]")
    for name in REQUIRED_FIELDS[:-1]:
        if name not in group["obs"]:
            errors.append(f"missing obs/{name}")
    for name in POSE_FIELDS:
        if name in group["obs"]:
            dataset = group["obs"][name]
            expected_frame = "world" if name.endswith("_w") else "object" if name.endswith("_o") else "robot_base"
            if dataset.shape != (t + 1, 7):
                errors.append(f"obs/{name} must have shape [T+1,7]")
            if dataset.attrs.get("quaternion_order") != "xyzw" or dataset.attrs.get("frame") != expected_frame:
                errors.append(f"obs/{name} quaternion/frame metadata mismatch")
    if "gripper_force" in group["obs"] and group["obs/gripper_force"].attrs.get("unit") != "Newton":
        errors.append("gripper_force unit must be Newton")
    try:
        validate_reward_mask(group)
    except ValueError as exc:
        errors.append(str(exc))
    errors.extend(_validate_field_metadata(group))
    errors.extend(_validate_camera_calibration(group, t))
    if errors:
        return tuple(errors)
    from types import SimpleNamespace

    try:
        frames = [
            SimpleNamespace(
                value={
                    **{name: group["obs"][name][i] for name in REQUIRED_FIELDS[:-1]},
                    "T_w_b": transform[...] if transform.ndim == 2 else transform[i],
                }
            )
            for i in range(t + 1)
        ]
        steps = [
            SimpleNamespace(
                action=SimpleNamespace(action=group["actions"][i], action_applied=group["action_applied"][i])
            )
            for i in range(t)
        ]
        validate_observations(SimpleNamespace(observations=frames, steps=steps))
        expected = np.asarray(
            [
                pose_displacement(a.value["tcp_pose_b"], b.value["tcp_pose_b"])
                for a, b in zip(frames[:-1], frames[1:], strict=True)
            ]
        )
        if not np.allclose(expected, group["diagnostics/tcp_displacement"], atol=2e-5):
            errors.append("tcp_displacement does not match adjacent observations")
        if not np.array_equal(group["actions"], group["action_commanded"]):
            errors.append("actions must equal action_commanded")
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        errors.append(str(exc))
    return tuple(errors)


def _validate_camera_calibration(group, t):
    errors = []
    if "camera_calibration" in group:
        for camera_name, camera in group["camera_calibration"].items():
            for convention in ("ros", "opengl"):
                key = f"extrinsic_quaternion_w_{convention}_xyzw"
                if key not in camera or camera[key].shape != (t + 1, 4):
                    errors.append(f"camera {camera_name} requires {key} [T+1,4]")
                elif camera[key].attrs.get("quaternion_order") != "xyzw" or not np.allclose(
                    np.linalg.norm(camera[key], axis=1), 1.0, atol=1e-5
                ):
                    errors.append(f"camera {camera_name} {key} must contain unit XYZW quaternions")
    return tuple(errors)


def _validate_field_metadata(group):
    """Verify the semantic attributes needed to interpret aligned sensor fields."""
    errors = []
    expected = {}
    for name in POSE_FIELDS:
        expected[f"obs/{name}"] = {
            "axis": "T+1", "unit": "meter,unit_quaternion",
            "axis_order": "x,y,z,qx,qy,qz,qw",
            "source": "measured" if name == "tcp_pose_b" else "derived",
        }
    for suffix, frame in (("b", "robot_base"), ("w", "world")):
        expected[f"obs/ee_twist_{suffix}"] = {
            "axis": "T+1", "frame": frame,
            "unit": "meter_per_second,radian_per_second",
            "axis_order": "vx,vy,vz,wx,wy,wz", "reference_point": "tcp_origin",
            "source": "rigid_body_velocity_shifted_to_tcp",
        }
    expected["obs/gripper_force"] = {
        "axis": "T+1", "frame": "gripper_joint_space", "unit": "Newton",
        "source": "physx_dof_projected_joint_force",
    }
    expected["obs/object_position_b"] = {"source": "target_config_proxy", "is_dynamic": False}
    expected["diagnostics/tcp_displacement"] = {
        "axis": "T", "frame": "robot_base", "source": "adjacent_observations",
        "unit": "meter,radian", "axis_order": "dx,dy,dz,rx,ry,rz",
    }
    for name in ("action_commanded", "action_applied"):
        expected[name] = {
            "axis": "T", "frame": "robot_base", "unit": "normalized",
            "axis_order": FILE_METADATA["action_components"],
        }
    transform = group.get("transforms/T_w_b")
    if transform is not None:
        expected["transforms/T_w_b"] = {
            "axis": "episode" if transform.ndim == 2 else "T+1",
            "convention": "target_from_source", "source_frame": "robot_base", "target_frame": "world",
        }
    for path, attributes in expected.items():
        if path not in group:
            errors.append(f"missing {path}")
            continue
        for key, value in attributes.items():
            if group[path].attrs.get(key) != value:
                errors.append(f"{path} metadata {key} must equal {value!r}")
    for key, value in {
        "object_state_source": "target_config_proxy", "object_state_is_dynamic": False,
        "goal_error_status": "unavailable_without_measured_object_state",
    }.items():
        if group.attrs.get(key) != value:
            errors.append(f"aligned metadata {key} must equal {value!r}")
    return tuple(errors)


def validate_reward_mask(group):
    """Reject missing, malformed or enabled placeholder targets before training."""
    valid = group.get("reward_valid")
    if (
        valid is None
        or getattr(valid, "shape", None) != (group["actions"].shape[0],)
        or getattr(getattr(valid, "dtype", None), "kind", None) != "b"
        or group.attrs.get("reward_status") != "not_defined"
        or np.any(valid)
    ):
        raise ValueError("v4 placeholder rewards require an all-false boolean reward_valid mask with shape [T]")
