"""Contract corruption and reproducibility checks without a planning runtime."""

import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from configs.curobo.acceptance import AcceptanceConfig
from configs.curobo.bootstrap import BootstrapConfig
from configs.curobo.planner import PlannerConfig
from configs.curobo.robot import RobotConfig
from configs.curobo.world import CuboidConfig, WorldConfig
from DataPipe.pipelines.curobo_bootstrap.contracts import (
    ConfigHashes,
    DiscreteEvent,
    EpisodeSpec,
    EventFlag,
    FailureDiagnostic,
    MotionSegment,
    PlannedTrajectory,
    Pose,
    Stage,
    TrajectoryProgram,
    config_hash,
)


def episode():
    pose = Pose((0.1, 0.2, 0.3), (1, 0, 0, 0), "B")
    return EpisodeSpec(
        "trial_0001",
        42,
        (0,) * 6,
        pose,
        pose,
        pose,
        pose,
        ConfigHashes(*(config_hash({"name": name}) for name in range(5))),
    )


def trajectory():
    return PlannedTrajectory(
        tuple(f"joint{i}" for i in range(6)),
        10,
        (0.1, 0.2),
        ((0,) * 6,) * 2,
        ((0,) * 6,) * 2,
        ((0,) * 6,) * 2,
        ((0, 0, 0, 1, 0, 0, 0),) * 2,
        (1, 1),
        (0, 0),
        (0, 1),
        (0, 0),
        (0, 0),
    )


def robot():
    return RobotConfig(
        tuple(f"joint{i}" for i in range(6)),
        (-1,) * 6,
        (1,) * 6,
        (1,) * 6,
        (2,) * 6,
        (3,) * 6,
        "base",
        "flange",
        "tcp",
        (0, 0, 0, 1, 0, 0, 0),
    )


def test_episode_and_trajectory_roundtrip():
    for item in (episode(), trajectory()):
        assert type(item).from_json(item.to_json()) == item
        assert config_hash(asdict(item)) == config_hash(json.loads(item.to_json()))


def test_hash_order_and_mutation():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})
    with pytest.raises(ValueError):
        config_hash({"bad": float("nan")})


@pytest.mark.parametrize(
    "changes",
    [
        {"episode_id": ""},
        {"seed": -1},
        {"seed": True},
        {"seed": 2**63},
        {"initial_joint_position": (0,) * 5},
        {"initial_joint_position": (float("nan"),) * 6},
        {"lift_m": 0},
        {"pregrasp_m": -1},
        {"gripper_dwell_s": float("inf")},
        {"contract_version": 2},
        {"grasp_tcp_pose_b": Pose((0, 0, 0), (1, 0, 0, 0), "W")},
    ],
)
def test_episode_rejects_corruption(changes):
    with pytest.raises(ValueError):
        replace(episode(), **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"time_s": ()},
        {"time_s": (0, 0.1)},
        {"time_s": (0.1, 0.3)},
        {"joint_position": ((0,) * 6,)},
        {"joint_velocity": ((0,) * 5,) * 2},
        {"joint_acceleration": ((float("nan"),) * 6,) * 2},
        {"planned_tcp_pose_b": ((0,) * 7,) * 2},
        {"gripper_command": (0, 1)},
        {"segment_index": (1, 0)},
        {"segment_index": (0, 99)},
        {"segment_sample_index": (0, 2)},
        {"segment_sample_index": (1, 2)},
        {"event_flags": (0, 64)},
        {"event_flags": (0, True)},
        {"collision_distance": (float("inf"), 1)},
        {"control_hz": 0},
    ],
)
def test_trajectory_rejects_corruption(changes):
    with pytest.raises(ValueError):
        replace(trajectory(), **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"position_lower": (2,) * 6},
        {"velocity_limit": (0,) * 6},
        {"joint_names": ("same",) * 6},
        {"quaternion_order": "xyzw"},
        {"joint_unit": "deg"},
        {"flange_to_tcp": (0,) * 7},
    ],
)
def test_robot_rejects_implicit_conversion(changes):
    with pytest.raises(ValueError):
        replace(robot(), **changes)


def test_configuration_is_immutable_and_world_hash_is_order_independent():
    values = [1] * 6
    cfg = replace(robot(), velocity_limit=values)
    values[0] = 100
    assert cfg.velocity_limit[0] == 1
    a = CuboidConfig("target", (0, 0, 0, 1, 0, 0, 0), (1, 1, 1))
    b = CuboidConfig("table", (0, 0, -1, 1, 0, 0, 0), (2, 2, 1))
    assert config_hash(asdict(WorldConfig((a, b), "target"))) == config_hash(asdict(WorldConfig((b, a), "target")))
    with pytest.raises(ValueError):
        WorldConfig((a, a), "target")


@pytest.mark.parametrize("version", ["", "latest"])
def test_unpinned_planner_rejected(version):
    with pytest.raises(ValueError):
        PlannerConfig(version)


def test_offline_only_and_output_isolation():
    with pytest.raises(ValueError):
        PlannerConfig("test-only", online_replanning=True)
    with pytest.raises(ValueError):
        PlannerConfig("test-only", fallback_used=True)
    for run in ("../legacy", "a/b", "a\\b", ""):
        with pytest.raises(ValueError):
            BootstrapConfig(run)
    with pytest.raises(ValueError):
        BootstrapConfig("ok", output_root="data/legacy")
    assert BootstrapConfig("run1").run_directory == Path("data/curobo_bootstrap/run1")
    with pytest.raises(ValueError):
        AcceptanceConfig(*(float("nan"),) * 7)


def test_program_and_diagnostics():
    events = {
        Stage.CLOSE_GRIPPER: EventFlag.CLOSE_GRIPPER,
        Stage.VERIFY_GRASP: EventFlag.VERIFY_GRASP | EventFlag.ATTACH,
        Stage.OPEN_GRIPPER: EventFlag.OPEN_GRIPPER,
        Stage.VERIFY_PLACE: EventFlag.VERIFY_PLACE | EventFlag.DETACH,
    }
    steps = tuple(
        DiscreteEvent(stage, events[stage]) if stage in events else MotionSegment(stage, episode().grasp_tcp_pose_b)
        for stage in Stage
    )
    program = TrajectoryProgram("episode", steps)
    assert len(program.steps) == 11
    assert TrajectoryProgram.from_json(program.to_json()) == program
    corrupted = json.loads(program.to_json())
    corrupted["steps"][0]["kind"] = "unknown"
    with pytest.raises(ValueError):
        TrajectoryProgram.from_json(json.dumps(corrupted))
    with pytest.raises(ValueError):
        TrajectoryProgram("episode", steps[::-1])
    with pytest.raises(ValueError):
        DiscreteEvent(Stage.VERIFY_GRASP, EventFlag.NONE)
    failure = FailureDiagnostic("episode", "planning", "NO_SOLUTION", "No valid motion", 2)
    assert json.loads(failure.to_json())["segment_index"] == 2


def test_contract_source_loads_with_only_standard_library():
    # DataPipe.__init__ eagerly imports legacy modules. Load this module directly
    # to verify its own dependency boundary without changing that public API.
    path = Path(importlib.util.find_spec("DataPipe.pipelines.curobo_bootstrap.contracts").origin)
    script = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location('neutral_contracts', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert module.config_hash({'a': 1})
assert not any(k.split('.')[0] in {'numpy', 'h5py', 'torch', 'curobo', 'isaacsim', 'CurriculumRL'} for k in sys.modules)
"""
    result = subprocess.run([sys.executable, "-S", "-c", script, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
