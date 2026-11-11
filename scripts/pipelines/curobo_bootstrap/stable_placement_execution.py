"""Execute direct pre-place TCP targets, retaining historical object-goal audits.

The frozen route supplies transport. This controller replaces its placement and
release segment with joint targets, never robot/object state writes.
"""

from dataclasses import asdict

import numpy as np

from configs.task import TARGET_STATES
from scripts.pick_place_reward_candidate import rotation_error

from .contracts import config_hash
from .physical_failure import relative_pose
from .placement_search import carrying_metrics, continuous_stable_s, placement_metrics
from .replay import matrix_pose_wxyz
from .robot_adapter import pose_matrix
from .task_targets import legacy_tcp_poses_b


def placement_profile(task):
    """Freeze the current recipe even when transport uses a historical reference."""
    destination = task["destination"]
    state = destination if (destination.get("pre_place_tcp_position_e") is not None
                            or destination.get("place_object_position_e") is not None) else asdict(
        TARGET_STATES[task["destination_index"]]
    )
    direct_tcp = state.get("pre_place_tcp_position_e") is not None
    profile = dict(
        schema="direct_tcp_placement/2" if direct_tcp else "searched_stable_placement/1",
        destination_index=task["destination_index"],
        object_rotation_wxyz=list(destination["rotation_wxyz"]),
        tracking_compensation=state["place_tracking_compensation"],
        entry_lift_m=state["place_entry_lift_m"],
        standoff_m=state["place_standoff_m"],
        approach_direction_e=(np.asarray(destination["preposition_e"]) - (
            state["pre_place_tcp_position_e"] if direct_tcp else destination["grasp_tcp_position_e"]
        )).tolist(),
        descent_speed_m_s=0.06,
        final_speed_m_s=0.01,
        hold_s=0.5,
        open_s=0.3,
        settle_s=4.0,
        withdraw_m=0.06,
        high_clearance_m=0.4,
    )
    position_key = "pre_place_tcp_position_e" if direct_tcp else "object_position_e"
    profile[position_key] = list(state["pre_place_tcp_position_e"] if direct_tcp else state["place_object_position_e"])
    profile["profile_hash"] = config_hash(profile)
    return profile


def placement_geometry(task, profile, sample):
    """New profiles use the specified TCP directly; old profiles keep their geometry."""
    grip = relative_pose(sample["tcp_pose_w"], sample["target_pose_w"])
    if "pre_place_tcp_position_e" in profile:
        tcp_b = legacy_tcp_poses_b(task)["place"]
        release = np.asarray(task["frames"]["T_WB"]) @ pose_matrix(tcp_b[:3], tcp_b[3:])
        release[:3, 3] = (np.asarray(task["frames"]["T_WE"]) @ np.r_[profile["pre_place_tcp_position_e"], 1])[:3]
        object_release = release @ grip
    else:
        object_release = np.asarray(task["frames"]["T_WE"]) @ pose_matrix(
            profile["object_position_e"], profile["object_rotation_wxyz"]
        )
        release = object_release @ np.linalg.inv(grip)
    high = release.copy()
    clearance = task["lift_clearance_m"] if profile["standoff_m"] else profile["high_clearance_m"]
    high[2, 3] = max(sample["tcp_pose_w"][2], release[2, 3] + clearance)
    points = {"place_high": high}
    if profile["standoff_m"]:
        away = np.asarray(task["frames"]["T_WE"])[:3, :3] @ profile["approach_direction_e"]
        away[2] = 0
        norm = np.linalg.norm(away)
        if norm <= 1e-9:
            raise ValueError("shelf placement requires a horizontal entry direction")
        high[:3, 3] += away / norm * profile["standoff_m"]
        outside = release.copy()
        outside[:2, 3] = high[:2, 3]
        outside[2, 3] += profile["entry_lift_m"]
        entry = release.copy()
        entry[2, 3] += profile["entry_lift_m"]
        points.update(place_descent=outside, place_entry=entry)
    points["place"] = release
    return grip, object_release, points


def execute_stable_placement(task, profile, initial, step, should_abort, *, solver=None, continue_on_failure=False):
    """Callbacks perform one real 60 Hz step and return its measured observation."""
    from scipy.spatial.transform import Rotation, Slerp

    if solver is None:
        from scripts.native.aubo_ik import solve_tcp_ik

        solver = solve_tcp_ik
    if not initial["evidence"]["grasp_verified"] and not continue_on_failure:
        return dict(passed=False, failure_reason="placement requires a currently verified physical grasp")
    grip, object_release, points = placement_geometry(task, profile, initial)
    base_from_world = np.linalg.inv(np.asarray(task["frames"]["T_WB"]))
    goal = matrix_pose_wxyz(np.asarray(task["frames"]["T_WE"]) @ pose_matrix(
        task["destination"]["position_e"], task["destination"]["rotation_wxyz"]
    ))
    q = np.asarray(initial["q"])
    current = pose_matrix(initial["tcp_pose_w"][:3], initial["tcp_pose_w"][3:])
    bias = np.zeros(3)
    last = initial
    rows = []
    loss_s = 0.0
    report = dict(
        profile=profile, passed=False, tcp_from_object=grip.tolist(),
        execution_completed=False, continue_on_task_failure=continue_on_failure,
        initial_grasp_verified=initial["evidence"]["grasp_verified"], validation_failures=[],
        calibration_observation=round(initial["time_s"] * 60),
        expected_release_object_pose_w=list(matrix_pose_wxyz(object_release)),
        tcp_waypoint_pose_w={k: list(matrix_pose_wxyz(v)) for k, v in points.items()},
    )

    def require(condition, reason, stage):
        if condition:
            return
        if not continue_on_failure:
            raise RuntimeError(reason)
        if not any(f["reason"] == reason and f["stage"] == stage for f in report["validation_failures"]):
            report["validation_failures"].append(dict(time_s=last["time_s"], stage=stage, reason=reason))
            report.setdefault("failure_reason", reason)

    require(initial["evidence"]["grasp_verified"],
            "placement requires a currently verified physical grasp", "placement_entry")

    def command(pose, fingers, stage, phase):
        nonlocal q, last, loss_s
        if should_abort():
            raise RuntimeError("physical task terminated before stable placement completed")
        commanded = pose.copy()
        commanded[:3, 3] += bias
        target_b = matrix_pose_wxyz(base_from_world @ commanded)
        result = solver(target_b[:3], target_b[3:], seed=q.copy())
        if not result.success or result.q is None:
            raise RuntimeError(f"stable placement IK failed in {stage}: {result.reason}")
        q = np.asarray(result.q)
        last = step(q, np.asarray(fingers), stage, phase, list(matrix_pose_wxyz(pose)))
        row = dict(
            time_s=last["time_s"], object_pose_w=last["target_pose_w"],
            object_velocity_w=last["target_velocity_w"], contact_available=last["contact_available"],
            contact_force_n=last["contact_force_n"], illegal_contact=last.get("illegal_contact", False),
            joint_limit=last["joint_limit"], stage=stage, finger_position=last["gripper_position"],
            tcp_pose_w=last["tcp_pose_w"],
        )
        row["metrics"] = placement_metrics(row, goal, grasp_verified=True)
        rows.append(row)
        if phase == "place":
            carrying = carrying_metrics(row, goal, grip)
            loss_s = loss_s + 1 / 60 if carrying["unsupported_loss"] else 0.0
            require(loss_s + 1e-12 < 0.05, "physical grasp lost before intended release", stage)
            require(not (last["contact_available"] and last["contact_force_n"][2] > 0.01
                         and last["target_pose_w"][2] > goal[2] + 0.02),
                    "support contact above the intended placement region", stage)
        return last

    def hold(pose, fingers, stage, phase, seconds):
        for _ in range(round(seconds * 60)):
            command(pose, fingers, stage, phase)

    def move(start, finish, stage, phase, fingers):
        displacement = finish[:3, 3] - start[:3, 3]
        distance = np.linalg.norm(displacement)
        angle = rotation_error(matrix_pose_wxyz(start)[3:], matrix_pose_wxyz(finish)[3:])
        interpolation = Slerp([0, 1], Rotation.from_matrix(np.stack([start[:3, :3], finish[:3, :3]])))
        progress, count = 0.0, 0
        while progress < 1 - 1e-12:
            speed = profile["final_speed_m_s"] if distance * (1 - progress) < 0.025 else profile["descent_speed_m_s"]
            rate = min(speed / max(distance, 1e-12), 0.5 / max(angle, 1e-12))
            progress = min(1.0, progress + rate / 60 * min(1.0, (count + 1) / 18))
            pose = start.copy()
            pose[:3, 3] += displacement * progress
            pose[:3, :3] = interpolation([progress]).as_matrix()[0]
            command(pose, fingers, stage, phase)
            count += 1

    try:
        move(current, points["place_high"], "stable_place_align", "place", [0, 0])
        current = points["place_high"]
        hold(current, [0, 0], "stable_place_high_hold", "place", 0.75)
        if profile["tracking_compensation"] and not profile["standoff_m"]:
            bias = current[:3, 3] - np.asarray(last["tcp_pose_w"][:3])
            require(np.linalg.norm(bias) <= 0.01, "placement tracking bias exceeds 10 mm", "stable_place_high_hold")
            hold(current, [0, 0], "stable_place_high_hold", "place", 0.75)
        report["tracking_bias_w_m"] = bias.tolist()
        require(last["evidence"]["grasp_verified"],
                "physical grasp lost before placement descent", "stable_place_high_hold")
        for name in ("place_descent", "place_entry", "place"):
            if name in points:
                move(current, points[name], f"stable_{name}", "place", [0, 0])
                current = points[name]
                if name == "place_entry" and profile["tracking_compensation"]:
                    hold(current, [0, 0], "stable_entry_hold", "place", 0.75)
                    bias = current[:3, 3] - np.asarray(last["tcp_pose_w"][:3])
                    require(np.linalg.norm(bias) <= 0.01, "entry tracking bias exceeds 10 mm", "stable_entry_hold")
                    report["tracking_bias_w_m"] = bias.tolist()
                    hold(current, [0, 0], "stable_entry_hold", "place", 0.75)
        hold(current, [0, 0], "stable_pre_release_hold", "place", profile["hold_s"])
        report["endpoint_tcp_error_m"] = float(np.linalg.norm(np.asarray(last["tcp_pose_w"][:3]) - current[:3, 3]))
        require(report["endpoint_tcp_error_m"] <= 0.003,
                "release TCP missed searched position by more than 3 mm", "stable_pre_release_hold")
        for i in range(round(profile["open_s"] * 60)):
            fraction = (i + 1) / round(profile["open_s"] * 60)
            command(current, [0.0115 * fraction, -0.0115 * fraction], "stable_open_gripper", "release")
        hold(current, [0.0115, -0.0115], "stable_settle", "release", profile["settle_s"])
        report["stable_before_withdraw_s"] = continuous_stable_s(rows)
        withdrawn = current.copy()
        away = current[:3, 3] - object_release[:3, 3]
        away[2] = 0
        if np.linalg.norm(away) <= 1e-9:
            raise RuntimeError("cannot determine lateral withdrawal direction")
        withdrawn[:3, 3] += away / np.linalg.norm(away) * profile["withdraw_m"]
        move(current, withdrawn, "stable_withdraw", "release", [0.0115, -0.0115])
        hold(withdrawn, [0.0115, -0.0115], "stable_after_withdraw", "release", profile["settle_s"])
        after_withdraw = [r for r in rows if r["stage"] == "stable_after_withdraw"]
        report["stable_after_withdraw_s"] = continuous_stable_s(after_withdraw)
        report["final_metrics"] = rows[-1]["metrics"]
        report["execution_completed"] = True
        report["passed"] = (not report["validation_failures"] and report["stable_before_withdraw_s"] >= 0.25
                            and report["stable_after_withdraw_s"] >= 0.5)
        require(report["stable_before_withdraw_s"] >= 0.25 and report["stable_after_withdraw_s"] >= 0.5,
                "placement did not remain stable for the required observation windows", "stable_after_withdraw")
    except RuntimeError as error:
        report["failure_reason"] = str(error)
    return report
