"""Regression gates for the native port, independent of Kit startup."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from configs.curobo.world import MeshConfig, WorldConfig
from DataPipe.native.world_identity import reconcile_world
from DataPipe.pipelines.curobo_bootstrap.pick_place_artifact import sha256_file
from DataPipe.pipelines.curobo_bootstrap.world_adapter import WorldSnapshot


def snapshot(path):
    return WorldSnapshot(
        WorldConfig(
            (),
            "target",
            (MeshConfig("target", (0, 0, 0, 1, 0, 0, 0), str(path.resolve()), sha256_file(path), "/World/target"),),
        )
    )


def config(world):
    from dataclasses import asdict

    return asdict(world.config)


def write_mesh(path, offset=0, reverse=False):
    vertices = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]]) + offset
    trimesh.Trimesh(vertices=vertices, faces=[[0, 2, 1] if reverse else [0, 1, 2]], process=False).export(path)


def test_world_rounding_retains_both_actual_and_planning_identity(tmp_path):
    old, new = tmp_path / "old.obj", tmp_path / "new.obj"
    write_mesh(old)
    write_mesh(new, 1e-7)
    expected, actual = snapshot(old), snapshot(new)
    world, audit = reconcile_world(actual, config(expected))
    assert world.world_hash == expected.world_hash != actual.world_hash
    assert audit["actual_world_hash"] == actual.world_hash
    assert audit["max_vertex_error_m"] <= 2e-7


@pytest.mark.parametrize("offset,reverse", [(1e-5, False), (0, True)])
def test_world_drift_and_topology_changes_are_rejected(tmp_path, offset, reverse):
    old, new = tmp_path / "old.obj", tmp_path / "new.obj"
    write_mesh(old)
    write_mesh(new, offset, reverse)
    with pytest.raises(ValueError, match="geometry drift|topology"):
        reconcile_world(snapshot(new), config(snapshot(old)))


def test_frozen_mesh_hash_is_verified(tmp_path):
    old, new = tmp_path / "old.obj", tmp_path / "new.obj"
    write_mesh(old)
    write_mesh(new)
    frozen = config(snapshot(old))
    write_mesh(old, 1e-4)
    with pytest.raises(ValueError, match="hash differs"):
        reconcile_world(snapshot(new), frozen)


def test_native_control_definitions_preserve_original_algorithms():
    root = Path(__file__).resolve().parents[1] / "DataPipe/native"
    provenance = json.loads((root / "control_provenance.json").read_text())
    for name, info in provenance.items():
        source = Path(info["source"])
        if not source.exists():
            pytest.skip("original CurriculumRL checkout not present")
        assert sha256_file(source) == info["sha256"]
        original = ast.parse(source.read_text(encoding="utf-8"))
        native = ast.parse((root / (name + ".py")).read_text(encoding="utf-8"))

        def definitions(tree):
            return {node.name: ast.dump(node) for node in tree.body if isinstance(node, ast.FunctionDef | ast.ClassDef)}

        left, right = definitions(original), definitions(native)
        assert right
        for key, value in right.items():
            assert left[key] == value, f"control algorithm changed: {name}.{key}"


@pytest.mark.parametrize("stale_pose,stale_buffer", [(True, False), (False, True)])
def test_visual_audit_rejects_stale_wrist_pose_and_reused_buffers(tmp_path, stale_pose, stale_buffer):
    import importlib.util

    from PIL import Image

    script = Path(__file__).resolve().parents[1] / "scripts/audit_run003_isaacsim.py"
    spec = importlib.util.spec_from_file_location("native_auditor", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frames = []
    pixels = np.tile(np.arange(640, dtype=np.uint8)[None, :, None], (480, 1, 3))
    for render in (12, 24):
        for name in ("HandCamera", "CameraSensor_2", "CameraSensor_3"):
            relative = f"{name}_{render}.png"
            Image.fromarray(pixels).save(tmp_path / relative)
            frames.append(
                {
                    "camera": name,
                    "render_frame": render,
                    "trajectory_index": render - 1,
                    "time_s": (render - 1) / 60,
                    "rgb": relative,
                    "camera_position_w": [0 if stale_pose else render / 100, 0, 1],
                    "camera_quaternion_w_ros_wxyz": [0, 1, 0, 0],
                    "camera_quaternion_w_opengl_wxyz": [1, 0, 0, 0],
                    "intrinsic_matrix": [[554, 0, 320], [0, 554, 240], [0, 0, 1]],
                    "sensor_render_reference": {
                        "referenceTimeNumerator": 12 if stale_buffer else render,
                        "referenceTimeDenominator": 30,
                    },
                }
            )
    (tmp_path / "frames.json").write_text(
        json.dumps(
            {
                "source_frames": 24,
                "render_calls": 24,
                "frame_interval_render_calls": 12,
                "frames": frames,
            }
        )
    )
    with pytest.raises(AssertionError, match="hand camera did not follow|buffers did not refresh"):
        module.inspect_frames(tmp_path, source_count=24)


@pytest.mark.parametrize("changed_focal", [False, True])
def test_calibration_preserves_horizontal_fov_across_reference_resolutions(tmp_path, changed_focal):
    import importlib.util

    from PIL import Image

    script = Path(__file__).resolve().parents[1] / "scripts/audit_run003_isaacsim.py"
    spec = importlib.util.spec_from_file_location("native_calibration_auditor", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frames, reference = [], []
    pixels = np.tile(np.arange(640, dtype=np.uint8)[None, :, None], (480, 1, 3))
    for render in (12, 24):
        for name in ("HandCamera", "CameraSensor_2", "CameraSensor_3"):
            relative = f"{name}_{render}.png"
            Image.fromarray(pixels).save(tmp_path / relative)
            frame = {
                "camera": name,
                "render_frame": render,
                "trajectory_index": render - 1,
                "time_s": (render - 1) / 60,
                "rgb": relative,
                "camera_position_w": [render / 100 if name == "HandCamera" else 0, 0, 1],
                "camera_quaternion_w_ros_wxyz": [0, 1, 0, 0],
                "camera_quaternion_w_opengl_wxyz": [1, 0, 0, 0],
                "intrinsic_matrix": [[554, 0, 320], [0, 554, 240], [0, 0, 1]],
                "sensor_render_reference": {"referenceTimeNumerator": render, "referenceTimeDenominator": 30},
            }
            frames.append(frame)
            if render == 12:
                focal = 110.8 * (1.01 if changed_focal else 1)
                reference.append(
                    {
                        **frame,
                        "intrinsic_matrix": [[focal, 0, 64], [0, focal, 64], [0, 0, 1]],
                        "reference_resolution_px": [128, 128],
                    }
                )
    (tmp_path / "frames.json").write_text(
        json.dumps(
            {
                "source_frames": 24,
                "render_calls": 24,
                "frame_interval_render_calls": 12,
                "frames": frames,
            }
        )
    )
    if changed_focal:
        with pytest.raises(AssertionError, match="camera intrinsics changed"):
            module.inspect_frames(tmp_path, source_count=24, reference_frames=reference)
    else:
        assert module.inspect_frames(tmp_path, source_count=24, reference_frames=reference)[
            "reference_calibration_verified"
        ]
