"""Read back physical experiment data, reward chronology and 256px evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.pick_place_reward_candidate import RewardTask, rotation_error
from scripts.pipelines.curobo_bootstrap.contracts import config_hash
from scripts.pipelines.curobo_bootstrap.physical_failure import GraspEvidence, make_reward_model, reward_observation
from scripts.pipelines.curobo_bootstrap.pick_place_artifact import load_hybrid_trajectory, save_json, sha256_file
from scripts.pipelines.curobo_bootstrap.placement_search import continuous_stable_s, placement_metrics
from scripts.pipelines.curobo_bootstrap.replay import matrix_pose_wxyz
from scripts.pipelines.curobo_bootstrap.stable_placement_execution import placement_geometry


def audit(run):
    run = Path(run)

    def read(name):
        return json.loads((run / name).read_text(encoding="utf-8"))

    report, records, rewards = read("report.json"), read("trace.json"), read("rewards.json")
    arrays = load_hybrid_trajectory(run / "trajectory.npz")
    n = len(records)
    assert n >= 2 and len(rewards) == n - 1
    assert report["passed"] and report["backend"] == "isaacsim"
    assert report["execution_contract"]["camera_resolution"] == [256, 256]
    assert report["execution_contract"]["target_state_writes_after_reset"] == 0
    assert report["execution_contract"]["arm_state_writes_after_reset"] == 0
    assert report["execution_contract"]["logical_attachment"] is False
    assert report["target_physics"]["kinematic_enabled"] is False
    assert report["target_physics"]["gravity_disabled"] is False
    assert not report["isaaclab_modules_loaded"]
    assert all(sha256_file(path) == value for path, value in report["input_sha256"].items())
    for path, value in {
        **report["protected_source_sha256"], **report.get("experiment_source_sha256", {}),
    }.items():
        archived = report.get("execution_source_snapshot_paths", {}).get(path, path)
        assert sha256_file(archived) == value
    assert not arrays["object_attached"].any()
    np.testing.assert_allclose(arrays["time_s"], np.arange(n) / 60, atol=1e-12, rtol=0)
    np.testing.assert_equal(arrays["joint_position"], [r["q"] for r in records])
    np.testing.assert_equal(arrays["joint_velocity"], [r["dq"] for r in records])
    np.testing.assert_equal(arrays["gripper_target"], [r["executed_gripper_target"] for r in records])
    with np.load(run / "observations.npz", allow_pickle=False) as observations:
        for name in ("target_pose_w", "target_velocity_w", "tcp_pose_w", "gripper_position", "gripper_velocity"):
            np.testing.assert_equal(observations[name], [r[name] for r in records])
            assert np.isfinite(observations[name]).all()
    with np.load(run / "rewards.npz", allow_pickle=False) as transitions:
        for name in ("reward", "reward_valid", "terminated", "transition_valid"):
            np.testing.assert_equal(transitions[name], [r[name] for r in rewards])
        for name in ("reward_valid", "terminated", "transition_valid"):
            assert transitions[name].dtype == np.bool_
        if "candidate_reward_valid" in transitions.files:
            np.testing.assert_equal(
                transitions["candidate_reward_valid"], [r["candidate_reward_valid"] for r in rewards]
            )
        if not all(report.get("reward_evidence_coverage", {}).values()):
            assert not transitions["reward_valid"].any()
        np.testing.assert_equal(transitions["action_joint_target"], [r["commanded_joint_target"] for r in records[1:]])
        np.testing.assert_equal(
            transitions["action_gripper_target"], [r["executed_gripper_target"] for r in records[1:]]
        )
        np.testing.assert_equal(transitions["from_observation"], np.arange(n - 1))
        np.testing.assert_equal(transitions["to_observation"], np.arange(1, n))
    detector, model = GraspEvidence(records[0]), None
    goal = np.asarray(report["reward_task"]["goal_pose_w"])
    for i, record in enumerate(records):
        assert record["state_write_after_reset"] is False
        prior_support = i > 0 and records[i - 1]["evidence"]["support_verified"]
        current_support = record["contact_available"] and record["contact_force_n"][2] > 0.01
        normal_release = bool(
            i > 0
            and record["phase"] == "release"
            and not record["fault_injection_active"]
            and (prior_support or current_support)
            and np.linalg.norm(np.asarray(record["target_pose_w"][:3]) - goal[:3]) < 0.02
            and rotation_error(record["target_pose_w"][3:], goal[3:]) < np.deg2rad(10)
        )
        assert record.get("normal_supported_release", False) == normal_release
        if record["fault_injection_active"]:
            assert report["scenario"] != "natural"
            np.testing.assert_equal(record["executed_gripper_target"], [0.0115, -0.0115])
        else:
            np.testing.assert_equal(record["executed_gripper_target"], record["requested_gripper_target"])
        evidence = detector.step(record)
        assert evidence == record["evidence"]
        obs = reward_observation(record, evidence, timeout=(i == n - 1 and not (model and model.done)))
        if model is None:
            model = make_reward_model(RewardTask(**report["reward_task"]), obs)
        elif not model.done:
            result = model.step(obs)
            actual = rewards[i - 1]
            assert actual["transition_valid"] and actual["from_observation"] == i - 1
            assert actual["to_observation"] == i
            assert actual.get("candidate_reward_valid", actual["reward_valid"]) == result["reward_valid"]
            for name in ("terminated", "failure"):
                assert actual[name] == result[name]
            np.testing.assert_allclose(actual["reward"], result["reward"], atol=1e-12, rtol=0)
            assert actual["terms"] == result["terms"]
        else:
            assert not rewards[i - 1]["transition_valid"] and not rewards[i - 1]["reward_valid"]
            assert rewards[i - 1]["reward"] == 0
    assert model.done
    if "placement_profile" in report:
        profile = report["placement_profile"]
        assert profile["profile_hash"] == config_hash({k: v for k, v in profile.items() if k != "profile_hash"})
        assert read("scene_snapshot.json")["placement_profile"] == profile
        placement = report.get("stable_placement")
        if placement and "calibration_observation" in placement:
            task = read("scene_snapshot.json")["task"]
            grip, desired, points = placement_geometry(task, profile, records[placement["calibration_observation"]])
            np.testing.assert_allclose(placement["tcp_from_object"], grip, rtol=0, atol=1e-12)
            np.testing.assert_allclose(
                placement["expected_release_object_pose_w"], matrix_pose_wxyz(desired), atol=1e-12
            )
            for name, pose in points.items():
                np.testing.assert_allclose(placement["tcp_waypoint_pose_w"][name], matrix_pose_wxyz(pose), atol=1e-12)
            stable = []
            for record in records:
                if not record["stage"].startswith("stable_"):
                    continue
                row = dict(record, object_pose_w=record["target_pose_w"],
                           object_velocity_w=record["target_velocity_w"], finger_position=record["gripper_position"])
                row["metrics"] = placement_metrics(row, goal, grasp_verified=True)
                stable.append(row)
            if "final_metrics" in placement:
                assert placement["final_metrics"] == stable[-1]["metrics"]
                before = [r for r in stable if r["stage"] not in {"stable_withdraw", "stable_after_withdraw"}]
                after = [r for r in stable if r["stage"] == "stable_after_withdraw"]
                assert placement["stable_before_withdraw_s"] == continuous_stable_s(before)
                assert placement["stable_after_withdraw_s"] == continuous_stable_s(after)
                expected_pass = continuous_stable_s(before) >= 0.25 and continuous_stable_s(after) >= 0.5
                expected_pass = expected_pass and not placement.get("validation_failures", [])
                assert placement["passed"] == expected_pass
            if "initial_grasp_verified" in placement:
                assert placement["initial_grasp_verified"] == records[placement["calibration_observation"]][
                    "evidence"
                ]["grasp_verified"]
                if not placement["initial_grasp_verified"]:
                    assert not placement["passed"]
                    assert any(f["reason"] == "placement requires a currently verified physical grasp"
                               for f in placement["validation_failures"])
            if placement.get("execution_completed"):
                executed = [r["stage"] for r in records]
                assert executed.index("stable_place") < executed.index("stable_open_gripper")
                assert executed.index("stable_open_gripper") < executed.index("stable_withdraw")
                assert executed.index("stable_withdraw") < executed.index("stable_after_withdraw")
    visual_dir = Path(report.get("visualization", {}).get("directory", run / "execution_visualization"))
    visual = json.loads((visual_dir / "frames.json").read_text(encoding="utf-8"))
    images = visual["frames"]
    forced = visual.get("forced_render_frames", [])
    assert len(forced) == len(set(forced)) and all(1 <= f <= n and f % 12 for f in forced)
    expected_render_frames = set(range(12, n + 1, 12)) | set(forced)
    if report["execution_contract"].get("camera_capture_policy") == "periodic_plus_stage_and_gripper_transitions":
        assert report["mode"] == "gui"
        required_forced = {i + 1 for i, r in enumerate(records) if (i + 1) % 12 and (
            i == 0 or records[i - 1]["stage"] != r["stage"] or r["phase"] == "close"
            or r["stage"] in {"open_gripper", "stable_open_gripper"}
        )}
        assert set(forced) == required_forced
    assert len(images) == len(expected_render_frames) * 3 and images
    assert {f["render_frame"] for f in images} == expected_render_frames
    assert {f["camera"] for f in images} == {"HandCamera", "CameraSensor_2", "CameraSensor_3"}
    for frame in images:
        index = frame["trajectory_index"]
        assert frame["render_frame"] == index + 1
        assert abs(frame["time_s"] - records[index]["time_s"]) < 1e-12
        with Image.open(visual_dir / frame["rgb"]) as image:
            assert image.size == (256, 256) and image.mode == "RGB"
            assert np.std(np.asarray(image)) > 1
    return dict(
        schema="physical_failure_audit/1",
        passed=True,
        samples=n,
        transitions=n - 1,
        image_count=len(images),
        outcome=report["outcome"],
        reward_version=report["reward_version"],
        valid_reward_transitions=report["valid_reward_transitions"],
        candidate_valid_reward_transitions=report.get("candidate_valid_reward_transitions"),
        reward_evidence_coverage=report.get("reward_evidence_coverage"),
        original_collection_unchanged=True,
        input_artifacts_unchanged=True,
        reward_recomputed=True,
        no_logical_attachment=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    try:
        result = audit(args.run)
    except Exception as error:
        result = dict(schema="physical_failure_audit/1", passed=False, error=f"{type(error).__name__}: {error}")
    save_json(args.report, result)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
