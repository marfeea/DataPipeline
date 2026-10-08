"""P3 read-only alignment audit. Missing CuRobo produces a blocked gate report."""

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from DataPipe.runtime import ensure_curriculumrl_available

ensure_curriculumrl_available()
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--urdf", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--tcp-link", default="tool_center_point")
parser.add_argument("--with-curobo", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(f"Refusing to overwrite audit: {args.output}")
launcher = AppLauncher(args)

import numpy as np
import torch
from CurriculumRL.logic.aubo_ik import forward_kinematics
from CurriculumRL.tasks.tcp_docking.trajectory_env_cfg import TcpDockingTrajectoryEnv, TcpDockingTrajectoryEnvCfg

from configs.coordinate_frames import FLANGE_TO_TCP_ROTATION_F_WXYZ, FLANGE_TO_TCP_TRANSLATION_F
from DataPipe.pipelines.curobo_bootstrap.robot_adapter import (
    UrdfChain,
    audit_fk,
    compare_joint_contract,
    pose_error,
    pose_matrix,
)


def main():
    cfg = TcpDockingTrajectoryEnvCfg()
    cfg.sim.device = args.device
    cfg.seed = 7
    cfg.actions.joint_trajectory.open_position = (0.0115, -0.0115)
    cfg.actions.joint_trajectory.closed_position = (0, 0)
    env = TcpDockingTrajectoryEnv(cfg)
    try:
        env.reset()
        controller = env.action_manager.get_term("joint_trajectory").controller
        robot = controller.robot
        flange_ids, names = robot.find_bodies(["Flange"], preserve_order=True)
        if names != ["Flange"]:
            raise ValueError("flange body resolution failed")
        arm_names = robot.find_joints(
            ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Flange"], preserve_order=True
        )[1]

        def array(tensor):
            return tensor.detach().cpu().numpy().astype(float)

        offset = pose_matrix(FLANGE_TO_TCP_TRANSLATION_F, FLANGE_TO_TCP_ROTATION_F_WXYZ)
        chain = UrdfChain(args.urdf, "AUBO_E5", "AUBO_E5_Flange")
        native_tcp = UrdfChain(args.urdf, "AUBO_E5_Flange", args.tcp_link).fk([])
        limits = array(robot.data.joint_pos_limits[0, controller.arm_ids])
        velocity = array(robot.data.joint_vel_limits[0, controller.arm_ids])
        actual_contract = {
            "joint_names": arm_names,
            "lower": limits[:, 0].tolist(),
            "upper": limits[:, 1].tolist(),
            "velocity": velocity.tolist(),
        }
        issues = compare_joint_contract(
            chain.joint_names, chain.limits["lower"], chain.limits["upper"], chain.limits["velocity"], actual_contract
        )
        q0 = array(robot.data.joint_pos[0, controller.arm_ids])
        samples = np.vstack((np.zeros(6), q0, q0 + np.random.default_rng(7).uniform(-0.3, 0.3, (12, 6))))
        records = []
        quaternion_issues = []

        def measured_pose(position, quaternion, label):
            p, q = array(position), array(quaternion)
            norm = float(np.linalg.norm(q))
            if not np.isfinite(norm) or norm == 0:
                raise ValueError("invalid measured quaternion")
            if abs(norm * norm - 1) > 1e-6:
                quaternion_issues.append({"frame": label, "norm": norm, "raw_wxyz": q.tolist()})
                if "measured_quaternion_not_unit" not in issues:
                    issues.append("measured_quaternion_not_unit")
            # Explicit diagnostic conversion only; any norm violation still blocks the gate.
            return pose_matrix(p, q / norm)

        def isaac_fk(q):
            target = torch.as_tensor(q, dtype=robot.data.joint_pos.dtype, device=env.device).unsqueeze(0)
            robot.write_joint_state_to_sim(target, torch.zeros_like(target), joint_ids=controller.arm_ids)
            env.sim.forward()
            env.scene.update(env.physics_dt)
            base_w = measured_pose(robot.data.root_pos_w[0], robot.data.root_quat_w[0], "base_w")
            flange_w = measured_pose(
                robot.data.body_pos_w[0, flange_ids[0]], robot.data.body_quat_w[0, flange_ids[0]], "flange_w"
            )
            tcp_b = np.linalg.inv(base_w) @ flange_w @ offset
            records.append(
                {
                    "requested_q": q.tolist(),
                    "actual_q": array(robot.data.joint_pos[0, controller.arm_ids]).tolist(),
                    "base_w": base_w.tolist(),
                    "tcp_b": tcp_b.tolist(),
                }
            )
            return tcp_b

        backends = {
            "current": forward_kinematics,
            "urdf_project_tcp": lambda q: chain.fk(q) @ offset,
            "isaac": isaac_fk,
        }
        curobo_info = {"requested": args.with_curobo}
        if args.with_curobo:
            from DataPipe.pipelines.curobo_bootstrap.curobo_fk import CuroboFk

            try:
                curobo = CuroboFk(args.urdf, joint_names=arm_names, ee_link=args.tcp_link, device=args.device)
                backends["curobo"] = curobo
                curobo_info.update(version=curobo.version, contract=curobo.contract)
                issues.extend(
                    "curobo_" + item
                    for item in compare_joint_contract(arm_names, limits[:, 0], limits[:, 1], velocity, curobo.contract)
                )
            except Exception as error:
                curobo_info["error"] = f"{type(error).__name__}: {error}"
                issues.append("curobo_initialization_failed")
        fk = audit_fk(
            samples,
            backends,
            position_tolerance_m=1e-5,
            rotation_tolerance_rad=1e-3,
        )
        tcp_position, tcp_rotation = pose_error(offset, native_tcp)
        if tcp_position > 1e-5 or tcp_rotation > 1e-3:
            issues.append("native_urdf_tcp_offset_mismatch")
        report = {
            "passed": fk["passed"] and not issues,
            "status": "blocked" if issues or not fk["passed"] else "passed",
            "python": sys.executable,
            "urdf": str(chain.path),
            "tcp_link": args.tcp_link,
            "urdf_sha256": chain.sha256,
            "urdf_contract": {"joint_names": chain.joint_names, **chain.limits},
            "isaac_contract": actual_contract,
            "contract_issues": issues,
            "fk": fk,
            "curobo": curobo_info,
            "configured_flange_to_tcp": offset.tolist(),
            "native_urdf_flange_to_tcp": native_tcp.tolist(),
            "tcp_offset_error": {"position_m": tcp_position, "rotation_rad": tcp_rotation},
            "samples": records,
            "quaternion_issues": quaternion_issues,
            "measured_pose_conversion": "explicit_normalization_for_diagnostics_only; violations_block_gate",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "samples"}), flush=True)
    finally:
        env.close()


try:
    main()
except Exception:
    traceback.print_exc()
    raise
finally:
    launcher.app.close()
