"""Search real, gravity-driven vertical releases from a measured held-object state.

Initial arm/object state writes are allowed only at the start of each trial.
All subsequent movement and release use joint drive targets and PhysX steps.
Search results are placement evidence, not a claim of full-route reward validity.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from time import monotonic

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.task import ILLEGAL_CONTACT_FORCE_N, TARGET_STATES
from scripts.pick_place_reward_candidate import RewardConfig, rotation_error
from scripts.pipelines.curobo_bootstrap.contact_timing import configure_contact_timing
from scripts.pipelines.curobo_bootstrap.physical_failure import relative_pose
from scripts.pipelines.curobo_bootstrap.pick_place_artifact import save_json, sha256_file
from scripts.pipelines.curobo_bootstrap.placement_search import (
    PlacementCandidate,
    carrying_metrics,
    coarse_candidates,
    continuous_stable_s,
    placement_metrics,
    pose_list,
    tcp_release_pose,
)
from scripts.pipelines.curobo_bootstrap.robot_adapter import pose_matrix


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--destinations", type=int, nargs="+", default=[2])
    parser.add_argument("--candidates", type=Path, help="JSON array of PlacementCandidate fields")
    parser.add_argument("--max-trials", type=int, default=20)
    parser.add_argument("--high-clearance-m", type=float, default=0.4)
    parser.add_argument("--tail-s", type=float, default=2.0)
    parser.add_argument("--withdraw-m", type=float, default=0.06)
    parser.add_argument("--withdraw-mode", choices=("vertical", "lateral", "none"), default="lateral")
    parser.add_argument("--physics-hz", type=int, default=120)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--kit-cache-dir", type=Path, default=ROOT / "data/isaacsim_kit_cache")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.backend = "isaacsim"
    if any(d not in range(5) for d in args.destinations):
        parser.error("destinations must be 0..4")
    if args.physics_hz < 60 or args.physics_hz % 60 or args.max_trials <= 0:
        parser.error("physics-hz must be a multiple of 60; max-trials must be positive")
    if args.high_clearance_m <= 0 or args.tail_s < 0.5 or args.withdraw_m <= 0:
        parser.error("positive clearance/withdrawal and tail >= 0.5 s required")
    return args


def array(value):
    return value.detach().cpu().numpy().astype(float) if hasattr(value, "detach") else np.asarray(value, dtype=float)


class SearchRuntime:
    def __init__(self, args, grasp, seeds):
        import torch

        from isaacsim.core.api.sensors import RigidContactView
        from pxr import Usd, UsdPhysics

        from configs.assets import ROBOT_PRIM_CONTRACT, SCENE_ENTITY_AUBO, SCENE_ENTITY_TARGET
        from scripts.native.aubo_ik import forward_kinematics, solve_tcp_ik
        from scripts.native.ik_controller import AuboIKAction
        from scripts.native.scene import ENV_ROOT, NativeTrajectoryEnv, NativeTrajectoryEnvCfg
        from scripts.pipelines.curobo_bootstrap.physics_contact_reports import PhysicsContactReports

        self.args, self.grasp, self.torch = args, grasp, torch
        self.initial_seeds = seeds
        self.solve, self.fk = solve_tcp_ik, forward_kinematics
        cfg = NativeTrajectoryEnvCfg()
        cfg.sim.device = args.device
        self.contract = configure_contact_timing(cfg, physics_hz=args.physics_hz, velocity_iterations=4)
        self.cfg = cfg
        self.env = NativeTrajectoryEnv(cfg)
        self.env.reset()
        self.robot, self.target = self.env.scene[SCENE_ENTITY_AUBO], self.env.scene[SCENE_ENTITY_TARGET]
        self.arm_ids, _ = self.robot.find_joints(ROBOT_PRIM_CONTRACT.arm_joints)
        self.finger_ids, _ = self.robot.find_joints(ROBOT_PRIM_CONTRACT.gripper_joints)
        self.ik = AuboIKAction(self.robot, cfg.sim.dt, cfg.decimation)
        self.world_from_base = pose_matrix(array(self.robot.data.root_pos_w[0]), array(self.robot.data.root_quat_w[0]))
        self.base_from_world = np.linalg.inv(self.world_from_base)
        self.tcp_from_object = relative_pose(grasp["tcp_pose_w"], grasp["target_pose_w"])
        stage = self.env.scene.stage
        finger_paths = []
        for name in ROBOT_PRIM_CONTRACT.gripper_joints:
            joints = [
                p
                for p in Usd.PrimRange(stage.GetPrimAtPath(ENV_ROOT + "/AUBObot/AUBO_E5"))
                if p.GetName() == name and p.IsA(UsdPhysics.Joint)
            ]
            if len(joints) != 1:
                raise ValueError(f"finger joint must be unique: {name}")
            finger = stage.GetPrimAtPath(UsdPhysics.Joint(joints[0]).GetBody1Rel().GetTargets()[0])
            while not finger.HasAPI(UsdPhysics.RigidBodyAPI):
                finger = finger.GetParent()
            finger_paths.append(str(finger.GetPath()))
        self.support_paths = [
            str(p.GetPath())
            for p in Usd.PrimRange(stage.GetPrimAtPath(ENV_ROOT + "/station/static"), Usd.TraverseInstanceProxies())
            if p.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False
        ]
        self.sensor = RigidContactView(
            self.target.view.prim_paths[0],
            [*finger_paths, *self.support_paths],
            name="placement_contacts",
            prepare_contact_sensors=False,
            disable_stablization=False,
            max_contact_count=1024,
        )
        self.sensor.initialize(self.env.world.physics_sim_view)
        if self.sensor.num_filters != 2 + len(self.support_paths):
            raise ValueError("contact filter count differs")
        self.contacts = PhysicsContactReports()
        self.original_arm_write = self.robot.write_joint_state_to_sim
        self.original_pose_write = self.target.write_root_pose_to_sim
        self.original_velocity_write = self.target.write_root_velocity_to_sim
        self.initializing = True

        def guarded_arm(*a, **kw):
            if not self.initializing:
                raise RuntimeError("arm state writes forbidden during trial")
            return self.original_arm_write(*a, **kw)

        def guarded_pose(*a, **kw):
            if not self.initializing:
                raise RuntimeError("object pose writes forbidden during trial")
            return self.original_pose_write(*a, **kw)

        def guarded_velocity(*a, **kw):
            if not self.initializing:
                raise RuntimeError("object velocity writes forbidden during trial")
            return self.original_velocity_write(*a, **kw)

        self.robot.write_joint_state_to_sim = guarded_arm
        self.target.write_root_pose_to_sim = guarded_pose
        self.target.write_root_velocity_to_sim = guarded_velocity
        self.capture = None
        if args.capture:
            from scripts.pipelines.curobo_bootstrap.replay_visualization import ReplayFrameRecorder

            self.env._create_cameras((640, 480))
            self.recorder_class = ReplayFrameRecorder

    def tensor(self, value):
        return self.torch.as_tensor(
            np.asarray(value), dtype=self.robot.data.joint_pos.dtype, device=self.args.device
        ).view(1, -1)

    def tcp_w(self):
        p, q, _ = self.ik.current_tcp_pose_b()
        quaternion = array(q[0])
        return self.world_from_base @ pose_matrix(array(p[0]), quaternion / np.linalg.norm(quaternion))

    def solution(self, tcp_w, seed):
        command = tcp_w.copy()
        command[:3, 3] += getattr(self, "tracking_bias_w", np.zeros(3))
        pose = pose_list(self.base_from_world @ command)
        result = self.solve(
            pose[:3], pose[3:], seed=seed, max_iterations=120, position_tolerance_m=2e-5, orientation_tolerance_rad=2e-4
        )
        if not result.success:
            raise RuntimeError(f"IK unreachable: {result.reason}; {result.position_error_m:.5f} m")
        return result.q

    def initialize(self, tcp_high_w, seed_index):
        self.initializing = True
        self.tracking_bias_w = np.zeros(3)
        if seed_index >= len(self.initial_seeds):
            raise ValueError("seed index outside archived seed library")
        seeds = [self.initial_seeds[seed_index]]
        for seed in seeds:
            try:
                q = self.solution(tcp_high_w, seed)
                break
            except RuntimeError:
                pass
        else:
            raise RuntimeError("high pose unreachable with selected initial seed")
        positions = self.robot.data.joint_pos.clone()
        positions[:, self.arm_ids] = self.tensor(q)
        positions[:, self.finger_ids] = self.tensor(self.grasp["gripper_position"])
        self.robot.write_joint_state_to_sim(positions, self.torch.zeros_like(positions))
        self.robot.set_joint_position_target(self.tensor(q), joint_ids=self.arm_ids)
        self.robot.set_joint_position_target(self.tensor([0, 0]), joint_ids=self.finger_ids)
        self.env.sim.forward()
        self.env.scene.update(self.cfg.sim.dt)
        actual_high = self.tcp_w()
        obj = pose_list(actual_high @ self.tcp_from_object)
        self.target.write_root_pose_to_sim(self.tensor(obj))
        self.target.write_root_velocity_to_sim(self.tensor([0] * 6))
        self.env.sim.forward()
        self.env.scene.update(self.cfg.sim.dt)
        self.ik.reset()
        self.initializing = False
        self.time = 0.0
        self.rows = []
        self.grasp_verified = False
        self.carry_reference = None
        self.carry_loss_s = 0.0
        self.seed = q
        self.high_w = actual_high
        self.high_relative = self.tcp_from_object.copy()
        return q

    def step(self, q, fingers, stage, requested_tcp_w):
        self.robot.set_joint_position_target(self.tensor(q), joint_ids=self.arm_ids)
        self.robot.set_joint_position_target(self.tensor(fingers), joint_ids=self.finger_ids)
        self.contacts.begin_control_step()
        force_sum = np.zeros(2 + len(self.support_paths))
        available = True
        for _ in range(self.cfg.decimation):
            self.env.sim.step(render=False)
            self.env.scene.update(self.cfg.sim.dt)
            matrix = array(self.sensor.get_contact_force_matrix(dt=self.cfg.sim.dt))
            if not np.isfinite(matrix).all():
                available = False
            force_sum += np.linalg.norm(matrix.reshape(-1, 3), axis=1)
        forces = force_sum / self.cfg.decimation
        self.time += 1 / 60
        obj = [*array(self.target.data.root_pos_w[0]), *array(self.target.data.root_quat_w[0])]
        tcp = pose_list(self.tcp_w())
        joint = array(self.robot.data.joint_pos[0, self.arm_ids])
        limits = array(self.robot.data.joint_pos_limits[0, self.arm_ids])
        pairs = self.contacts.sample()
        # Contact reports are diagnostic; report their coverage separately.
        illegal = any(
            p["impulse_sum_Ns"] * 60 > ILLEGAL_CONTACT_FORCE_N
            and any("AUBObot" in a for a in p["actors"])
            and any("/station/static/" in a for a in p["actors"])
            for p in pairs
        )
        row = dict(
            time_s=self.time,
            stage=stage,
            q=joint.tolist(),
            dq=array(self.robot.data.joint_vel[0, self.arm_ids]).tolist(),
            object_pose_w=obj,
            object_velocity_w=array(self.target.view.get_velocities()[0]).tolist(),
            tcp_pose_w=tcp,
            requested_tcp_pose_w=pose_list(requested_tcp_w),
            finger_position=array(self.robot.data.joint_pos[0, self.finger_ids]).tolist(),
            commanded_fingers=list(fingers),
            commanded_arm_q=np.asarray(q).tolist(),
            contact_force_n=[*forces[:2], float(sum(forces[2:]))],
            support_forces_n=forces[2:].tolist(),
            contact_available=available,
            physics_contacts=pairs,
            illegal_contact=illegal,
            joint_limit=bool(np.any(joint < limits[:, 0]) or np.any(joint > limits[:, 1])),
        )
        row["metrics"] = placement_metrics(row, self.goal, grasp_verified=self.grasp_verified)
        carry_failure = None
        if self.carry_reference is not None and stage in {"descent", "pre_release_hold"}:
            row["carrying"] = carrying_metrics(row, self.goal, self.carry_reference)
            self.carry_loss_s = self.carry_loss_s + 1 / 60 if row["carrying"]["unsupported_loss"] else 0.0
            row["carry_loss_s"] = self.carry_loss_s
            if row["carrying"]["premature_support"]:
                carry_failure = "support contact before object reached placement region"
            elif self.carry_loss_s + 1e-12 >= 0.05:
                carry_failure = "grasp lost before intended release"
        self.rows.append(row)
        if self.capture:
            self.env.sim.render()
            self.capture.capture(
                self.env.cameras,
                render_frame=len(self.rows),
                trajectory_index=len(self.rows) - 1,
                time_s=self.time,
                event=stage,
            )
        if carry_failure:
            raise RuntimeError(carry_failure)
        return row

    def hold(self, q, fingers, stage, requested, seconds):
        for _ in range(round(seconds * 60)):
            self.step(q, fingers, stage, requested)

    def move(self, start, finish, candidate, stage, fingers, *, vertical=True):
        if vertical and not np.allclose(start[:2, 3], finish[:2, 3], atol=1e-12):
            raise ValueError("only vertical movement is allowed")
        displacement = finish[:3, 3] - start[:3, 3]
        distance = np.linalg.norm(displacement)
        direction = displacement / distance if distance > 0 else np.zeros(3)
        travel = 0.0
        count = 0
        while travel < distance - 1e-12:
            remaining = distance - travel
            speed = candidate.final_speed_m_s if remaining < 0.025 else candidate.descent_speed_m_s
            speed *= min(1.0, (count + 1) / 18)
            travel = min(distance, travel + speed / 60)
            pose = start.copy()
            pose[:3, 3] = start[:3, 3] + direction * travel
            self.seed = self.solution(pose, self.seed)
            self.step(self.seed, fingers, stage, pose)
            count += 1
        return self.seed

    def trial(self, candidate, output):
        output.mkdir(parents=True)
        started = monotonic()
        report = dict(
            schema="stable_placement_trial/1",
            candidate=candidate.payload(),
            passed=False,
            stage="initialization",
            execution_contract=self.contract,
            logical_attachment=False,
            state_writes_during_trial=0,
            scope="isolated held-state vertical placement; full transport not tested",
        )
        self.rows = []
        self.goal = [
            *TARGET_STATES[candidate.destination].position_e,
            *TARGET_STATES[candidate.destination].rotation_wxyz,
        ]
        release, expected = tcp_release_pose(self.goal, self.tcp_from_object, candidate)
        high = release.copy()
        high[2, 3] += candidate.high_clearance_m or self.args.high_clearance_m
        report.update(
            goal_object_pose_w=self.goal,
            release_tcp_pose_w=pose_list(release),
            high_tcp_pose_w=pose_list(high),
            expected_release_object_pose_w=pose_list(expected),
            tcp_from_object=self.tcp_from_object.tolist(),
            support_paths=self.support_paths,
            thresholds=asdict(RewardConfig()),
        )
        save_json(output / "report.json", report)
        try:
            self.seed = self.initialize(high, candidate.ik_seed_index)
            if self.args.capture:
                self.capture = self.recorder_class(output / "execution_visualization", self.env.camera_specs)
            report["stage"] = "verify_initial_grasp"
            self.hold(self.seed, [0, 0], "initial_grasp", high, 0.75)
            if candidate.tracking_compensation:
                self.tracking_bias_w = high[:3, 3] - np.asarray(self.rows[-1]["tcp_pose_w"][:3])
                if np.linalg.norm(self.tracking_bias_w) > 0.01:
                    raise RuntimeError("high pose tracking bias exceeds 10 mm")
                self.seed = self.solution(high, self.seed)
                self.hold(self.seed, [0, 0], "initial_grasp", high, 0.75)
            report["tracking_bias_w_m"] = self.tracking_bias_w.tolist()
            window = self.rows[-16:]
            relatives = [relative_pose(r["tcp_pose_w"], r["object_pose_w"]) for r in window]
            translation_drift = max(np.linalg.norm(r[:3, 3] - relatives[0][:3, 3]) for r in relatives)
            angle_drift = max(rotation_error(pose_list(r)[3:], pose_list(relatives[0])[3:]) for r in relatives)
            self.grasp_verified = bool(
                all(
                    r["contact_available"] and min(r["contact_force_n"][:2]) > 0.01 and r["contact_force_n"][2] <= 0.01
                    for r in window
                )
                and translation_drift < 0.005
                and angle_drift < np.deg2rad(5)
            )
            report["initial_grasp"] = dict(
                verified=self.grasp_verified,
                translation_drift_m=translation_drift,
                rotation_drift_rad=angle_drift,
                final_contact_force_n=window[-1]["contact_force_n"],
            )
            if not self.grasp_verified:
                raise RuntimeError("initialized grasp failed measured contact/stability gate")
            self.carry_reference = relatives[-1]
            report["stage"] = "descent"
            self.move(high, release, candidate, "descent", [0, 0])
            self.hold(self.seed, [0, 0], "pre_release_hold", release, candidate.hold_s)
            endpoint = self.rows[-1]
            report["endpoint_tcp_error_m"] = float(
                np.linalg.norm(np.asarray(endpoint["tcp_pose_w"][:3]) - release[:3, 3])
            )
            report["stage"] = "release"
            for i in range(round(candidate.open_s * 60)):
                fraction = min(1, (i + 1) / round(candidate.open_s * 60))
                self.step(self.seed, [0.0115 * fraction, -0.0115 * fraction], "release", release)
            self.hold(self.seed, [0.0115, -0.0115], "settle", release, self.args.tail_s)
            report["stable_before_withdraw_s"] = continuous_stable_s(self.rows)
            report["metrics_before_withdraw"] = self.rows[-1]["metrics"]
            report["placement_passed"] = report["stable_before_withdraw_s"] >= RewardConfig().settle_s
            report["stage"] = "withdraw"
            withdrawn = release.copy()
            if self.args.withdraw_mode == "vertical":
                withdrawn[2, 3] += self.args.withdraw_m
            elif self.args.withdraw_mode == "lateral":
                away = release[:3, 3] - expected[:3, 3]
                away[2] = 0
                away /= np.linalg.norm(away)
                withdrawn[:3, 3] += away * self.args.withdraw_m
            report["withdraw_mode"] = self.args.withdraw_mode
            report["withdraw_tcp_pose_w"] = pose_list(withdrawn)
            self.move(
                release,
                withdrawn,
                candidate,
                "withdraw",
                [0.0115, -0.0115],
                vertical=self.args.withdraw_mode != "lateral",
            )
            self.hold(self.seed, [0.0115, -0.0115], "after_withdraw", withdrawn, self.args.tail_s)
            report["withdraw_tcp_error_m"] = float(
                np.linalg.norm(np.asarray(self.rows[-1]["tcp_pose_w"][:3]) - withdrawn[:3, 3])
            )
            report["stable_after_withdraw_s"] = continuous_stable_s(
                [r for r in self.rows if r["stage"] == "after_withdraw"]
            )
            report["final_metrics"] = self.rows[-1]["metrics"]
            descent = [r for r in self.rows if r["stage"] in {"descent", "pre_release_hold", "release"}]
            report["max_descent_xy_error_m"] = max(
                float(np.linalg.norm(np.asarray(r["tcp_pose_w"][:2]) - release[:2, 3])) for r in descent
            )
            report["max_descent_orientation_error_rad"] = max(
                rotation_error(r["tcp_pose_w"][3:], r["requested_tcp_pose_w"][3:]) for r in descent
            )
            report["passed"] = bool(
                report["stable_before_withdraw_s"] >= RewardConfig().settle_s
                and report["stable_after_withdraw_s"] >= 0.5
                and report["max_descent_xy_error_m"] <= 0.003
                and report["endpoint_tcp_error_m"] <= 0.003
                and report["withdraw_tcp_error_m"] <= 0.003
                and report["max_descent_orientation_error_rad"] <= np.deg2rad(2)
                and not any(r["illegal_contact"] or r["joint_limit"] for r in self.rows)
            )
            report["stage"] = "complete"
            if not report["passed"]:
                report["failure_reason"] = "placement stability or vertical tracking gate failed"
        except Exception as error:
            report["failure_reason"] = str(error)
            report["traceback"] = traceback.format_exc()
        finally:
            report["elapsed_s"] = monotonic() - started
            report["sample_count"] = len(self.rows)
            report["contact_report_events"] = sum(len(r["physics_contacts"]) for r in self.rows)
            report["collision_evidence_complete"] = False
            if self.rows:
                report.setdefault("final_metrics", self.rows[-1]["metrics"])
            save_json(output / "trace.json", self.rows)
            if self.capture:
                report["visualization"] = self.capture.finish(source_frames=len(self.rows), render_calls=len(self.rows))
                self.capture = None
            save_json(output / "report.json", report)
        return report

    def close(self):
        self.contacts.finish()
        self.env.close()


def main():
    args = parse_args()
    runtime_path = Path(sys.executable).parent
    os.environ["PATH"] = os.pathsep.join(
        [
            str(runtime_path),
            str(runtime_path / "Scripts"),
            str(runtime_path / "Library/bin"),
            os.environ.get("PATH", ""),
        ]
    )
    if args.output.exists() and not args.resume:
        raise FileExistsError("Use a new output directory or --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    trace = json.loads(args.grasp_trace.read_text(encoding="utf-8"))
    held = [r for r in trace if r.get("evidence", {}).get("grasp_verified") and r.get("stage") == "free_space_transfer"]
    if not held:
        raise ValueError("grasp trace lacks a measured verified high transport state")
    grasp = held[-1]
    source_high = [
        r for r in trace if r.get("stage") == "physical_lift_hold" and r.get("evidence", {}).get("grasp_verified")
    ]
    rng = np.random.default_rng(7)
    seeds = [grasp["q"], source_high[-1]["q"] if source_high else grasp["q"]]
    seeds += [rng.uniform(-3, 3, 6).tolist() for _ in range(16)]
    config = dict(
        schema="stable_placement_search/1",
        grasp_trace=str(args.grasp_trace.resolve()),
        grasp_trace_sha256=sha256_file(args.grasp_trace),
        held_state=grasp,
        initial_seed_library=seeds,
        high_clearance_m=args.high_clearance_m,
        tail_s=args.tail_s,
        withdraw_m=args.withdraw_m,
        withdraw_mode=args.withdraw_mode,
        physics_hz=args.physics_hz,
        capture=args.capture,
        device=args.device,
        backend=args.backend,
        thresholds=asdict(RewardConfig()),
        script_sha256=sha256_file(Path(__file__)),
        core_sha256=sha256_file(ROOT / "scripts/pipelines/curobo_bootstrap/placement_search.py"),
    )
    manifest = args.output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != config:
        raise ValueError("resume manifest differs; use a new directory")
    save_json(manifest, config)
    candidates = (
        [PlacementCandidate(**c) for c in json.loads(args.candidates.read_text(encoding="utf-8"))]
        if args.candidates
        else [c for d in args.destinations for c in coarse_candidates(d)]
    )
    from scripts.sim_backend import launch

    launcher = launch(args)
    runtime = None
    reports = []
    try:
        runtime = SearchRuntime(args, grasp, seeds)
        for index, candidate in enumerate(candidates[: args.max_trials]):
            output = args.output / f"trial_{index:04d}"
            if (output / "report.json").exists() and args.resume:
                report = json.loads((output / "report.json").read_text(encoding="utf-8"))
                if report["candidate"] != candidate.payload():
                    raise ValueError("resume candidates differ")
            else:
                report = runtime.trial(candidate, output)
            reports.append(dict(trial=index, report_path=str(output / "report.json"), **report))
            best = sorted(reports, key=lambda r: (not r["passed"], r.get("final_metrics", {}).get("score", 1e6)))
            save_json(
                args.output / "summary.json",
                dict(trials=len(reports), successes=sum(r["passed"] for r in reports), best=best[:5], results=reports),
            )
            print(
                "PLACEMENT_RESULT "
                + json.dumps(
                    dict(
                        trial=index,
                        candidate=candidate.payload(),
                        passed=report["passed"],
                        reason=report.get("failure_reason"),
                        metrics=report.get("final_metrics"),
                        elapsed_s=report["elapsed_s"],
                    )
                ),
                flush=True,
            )
    finally:
        if runtime:
            runtime.close()
        launcher.app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
