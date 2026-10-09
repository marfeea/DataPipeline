"""Independently inspect native run003 execution, trajectory and every PNG."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DataPipe.pipelines.curobo_bootstrap.pick_place_artifact import sha256_file, verify_hybrid_artifact


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reference_calibration(directory):
    frames = read(directory / "frames.json")["frames"]
    calibration = {}
    for frame in frames:
        if frame["camera"] in calibration:
            continue
        with Image.open(directory / frame["rgb"]) as image:
            calibration[frame["camera"]] = {**frame, "reference_resolution_px": image.size}
    return list(calibration.values())


def inspect_frames(directory, *, source_count, trace=None, reference_frames=None):
    manifest = read(directory / "frames.json")
    frames = manifest["frames"]
    assert manifest["source_frames"] == manifest["render_calls"] == source_count
    assert manifest["frame_interval_render_calls"] == 12
    assert len(frames) == source_count // 12 * 3
    by_frame = {}
    render_references = {}
    poses = []
    calibration = {frame["camera"]: frame for frame in (reference_frames or [])}
    for frame in frames:
        render = frame["render_frame"]
        index = frame["trajectory_index"]
        assert render % 12 == 0 and index == render - 1 and index < source_count
        assert abs(frame["time_s"] - index / 60) < 1e-8
        by_frame.setdefault(render, set()).add(frame["camera"])
        reference = frame["sensor_render_reference"]
        token = reference["referenceTimeNumerator"] / reference["referenceTimeDenominator"]
        render_references.setdefault(render, set()).add(token)
        path = (directory / frame["rgb"]).resolve()
        assert path.is_relative_to(directory.resolve())
        with Image.open(path) as image:
            image.load()
            pixels = np.asarray(image)
            assert image.size == (640, 480) and image.mode == "RGB"
            assert np.std(pixels) > 1, f"blank camera frame: {path}"
        for key in (
            "camera_position_w",
            "camera_quaternion_w_ros_wxyz",
            "camera_quaternion_w_opengl_wxyz",
            "intrinsic_matrix",
        ):
            assert np.isfinite(frame[key]).all(), f"invalid camera metadata: {key}"
        for key in ("camera_quaternion_w_ros_wxyz", "camera_quaternion_w_opengl_wxyz"):
            assert abs(np.linalg.norm(frame[key]) - 1) < 1e-5
        k = np.asarray(frame["intrinsic_matrix"])
        assert k.shape == (3, 3) and k[0, 0] > 0 and k[1, 1] > 0
        if calibration:
            reference_frame = calibration[frame["camera"]]
            expected_k = np.asarray(reference_frame["intrinsic_matrix"], dtype=float).copy()
            width, height = reference_frame.get("reference_resolution_px", (2 * expected_k[0, 2], 2 * expected_k[1, 2]))
            # These cameras keep square pixels and horizontal aperture when
            # changing aspect ratio. Both focal lengths scale with width;
            # the centered principal point scales in its own image dimension.
            expected_k[0, 0] *= 640 / width
            expected_k[1, 1] *= 640 / width
            expected_k[0, 2] *= 640 / width
            expected_k[1, 2] *= 480 / height
            assert np.allclose(k, expected_k, atol=1e-3, rtol=0), "camera intrinsics changed"
            if frame["camera"] != "HandCamera":
                assert np.allclose(
                    frame["camera_position_w"], reference_frame["camera_position_w"], atol=1e-6, rtol=0
                ), "external camera position changed"
                left = np.asarray(frame["camera_quaternion_w_opengl_wxyz"])
                right = np.asarray(reference_frame["camera_quaternion_w_opengl_wxyz"])
                assert abs(abs(np.dot(left, right)) - 1) < 1e-6, "external camera orientation changed"
        if trace is not None:
            record = trace[index]
            assert frame["execution_stage"] == record["stage"]
            assert np.array_equal(frame["joint_position"], record["q"])
            assert np.array_equal(frame["joint_velocity"], record["dq"])
            assert np.array_equal(frame["observed_target_pose_w"], record["target_pose_w"])
        if frame["camera"] == "HandCamera":
            poses.append(frame["camera_position_w"])
    assert all(names == {"HandCamera", "CameraSensor_2", "CameraSensor_3"} for names in by_frame.values())
    assert all(len(tokens) == 1 for tokens in render_references.values()), "three cameras rendered different frames"
    tokens = [next(iter(render_references[k])) for k in sorted(render_references)]
    assert np.all(np.diff(tokens) > 0), "camera buffers did not refresh between captures"
    assert len(list(directory.rglob("*.png"))) == len(frames)
    if poses:
        assert np.max(np.ptp(poses, axis=0)) > 0.01, "hand camera did not follow robot"
    return {
        "passed": True,
        "image_count": len(frames),
        "resolution": [640, 480],
        "decoded_every_image": True,
        "reference_calibration_verified": bool(calibration),
        "manifest_sha256": sha256_file(directory / "frames.json"),
    }


def audit(output):
    collection = read(output / "collection_report.json")
    accepted = [r for r in collection["routes"] if r.get("passed")]
    results = []
    features = []
    assert len(accepted) >= collection["requested_count"] >= 6
    assert len({r["route"] for r in accepted}) == len(accepted)
    for route in accepted:
        artifact_dir = Path(route["artifact"])
        artifact = verify_hybrid_artifact(artifact_dir)
        q = artifact["trajectory"]["joint_position"]
        features.append(
            np.stack([np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(q)), q[:, j]) for j in range(6)], axis=-1)
        )
        integrated = Path(route["integrated_run"])
        contact = read(integrated / "report.json")
        replay = read(artifact_dir / "replay_report.json")
        for runtime in (contact, replay):
            assert runtime["passed"] and runtime["backend"] == "isaacsim"
            assert runtime["isaaclab_modules_loaded"] == []
            assert runtime["native_world_geometry"]["passed"]
            assert runtime["native_world_geometry"]["max_vertex_error_m"] <= 2e-7
        assert contact["execution_contract"]["physics_hz"] == 120
        assert contact["execution_contract"]["control_hz"] == 60
        trace = read(integrated / "trace.json")
        for name in ("report.json", "trace.json"):
            assert sha256_file(artifact_dir / "contact_execution" / name) == sha256_file(integrated / name)
        assert (artifact_dir / "asset_path_map.json").is_file()
        assert (artifact_dir / "source_run_report.json").is_file()
        assert len(trace) == contact["record_count"] and trace
        assert contact["target_lift_distance_m"] >= 0.1
        assert artifact["report"]["measured"]["final_target_position_error_m"] <= 0.02
        assert contact["close_gripper"]["position_error"] <= 0.008
        assert contact["open_gripper"]["position_error"] <= 0.008
        execution_frames = read(integrated / "execution_visualization/frames.json")
        assert execution_frames["execution_passed"]
        assert execution_frames["source_trace_sha256"] == sha256_file(integrated / "trace.json")
        samples = len(artifact["trajectory"]["time_s"])
        reference_frames = reference_calibration(Path(route["reference"]) / "replay_visualization")
        live = inspect_frames(
            artifact_dir / "contact_execution/execution_visualization",
            source_count=len(trace),
            trace=trace,
            reference_frames=reference_frames,
        )
        visual = inspect_frames(
            artifact_dir / "replay_visualization", source_count=samples, reference_frames=reference_frames
        )
        results.append(
            {
                "route": route["route"],
                "passed": True,
                "samples": samples,
                "execution": live,
                "replay": visual,
                "final_target_position_error_m": artifact["report"]["measured"]["final_target_position_error_m"],
                "trajectory_sha256": sha256_file(artifact_dir / "trajectory.npz"),
            }
        )
    reference = Path(collection["archive"])
    reference_manifest = read(reference / "archive_manifest.json")
    checked = 0
    for entry in reference_manifest["records"]:
        for name, expected in entry["files_sha256"].items():
            assert sha256_file(reference / entry["archive_directory"] / name) == expected
            checked += 1
    features = np.stack(features)
    distances = np.sqrt(np.mean((features[:, None] - features[None, :]) ** 2, axis=(2, 3)))
    minimum_distance = float(distances[np.triu_indices(len(results), 1)].min())
    return {
        "schema": "run003_isaacsim_independent_audit/1",
        "passed": True,
        "successful_directions": len(results),
        "reference_files_verified": checked,
        "routes": results,
        "native_pairwise_joint_rms_rad": distances.tolist(),
        "minimum_native_joint_rms_rad": minimum_distance,
        "scope": "run003 hybrid free-space state writing, new physical contact execution and state replay",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.output_root.resolve())
    except Exception as error:
        result = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    (args.output_root / "independent_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
