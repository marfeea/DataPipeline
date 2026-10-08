from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from configs.curobo.planner import PlannerConfig
from configs.curobo.world import WorldConfig
from DataPipe.pipelines.curobo_bootstrap.planner import (
    EXPERIMENTAL_ACCELERATION_LIMIT,
    GOAL_Q,
    JOINT_NAMES,
    SAFE_Q,
    CuroboMvpPlanner,
    TrajectoryData,
    load_trajectory,
    mvp_world,
    planning_robot_dict,
    save_trajectory,
    validate_trajectory,
)
from DataPipe.pipelines.curobo_bootstrap.world_adapter import AttachedSphere, WorldSnapshot

ASSET = Path("configs/curobo/assets/aubo_e5_project/collision_model.json")
LOWER = np.full(6, -3.0)
UPPER = np.full(6, 3.0)
VELOCITY = np.full(6, 2.0)


def test_cuda_graph_and_graph_search_flags_are_typed_and_default_to_v1_behavior():
    default = PlannerConfig(version="0.7.8")
    assert default.cuda_graph_enabled is True
    assert default.graph_first is False
    assert PlannerConfig(version="0.7.8", cuda_graph_enabled=False, graph_first=True).graph_first
    with pytest.raises(ValueError, match="cuda_graph_enabled"):
        PlannerConfig(version="0.7.8", cuda_graph_enabled=0)
    with pytest.raises(ValueError, match="graph_first"):
        PlannerConfig(version="0.7.8", graph_first=1)


def test_planner_attachment_rejects_duplicate_or_changed_world_before_runtime_mutation():
    planner = object.__new__(CuroboMvpPlanner)
    planner.world = mvp_world()
    planner.collision = SimpleNamespace(attached=False)
    attached = planner.world.attach((AttachedSphere((0, 0, 0), 0.01),))
    changed = WorldSnapshot(WorldConfig(attached.config.cuboids, "probe")).attach(attached.attached_spheres)
    with pytest.raises(ValueError, match="preserve"):
        planner.attach(changed)
    planner.world = attached
    planner.collision.attached = True
    with pytest.raises(RuntimeError, match="already attached"):
        planner.attach(attached)


def valid_trajectory():
    time_s = np.arange(7, dtype=np.float64) / 60.0
    q = np.linspace(SAFE_Q, GOAL_Q, len(time_s))
    dq = np.gradient(q, time_s, axis=0)
    ddq = np.gradient(dq, time_s, axis=0)
    return TrajectoryData(time_s, q, dq, ddq, JOINT_NAMES)


def validate(trajectory, collision_query=None):
    if collision_query is None:

        def collision_query(q):
            return {"world": np.zeros(len(q)), "self": np.zeros(len(q))}

    return validate_trajectory(
        trajectory,
        start_q=SAFE_Q,
        goal_q=GOAL_Q,
        position_lower=LOWER,
        position_upper=UPPER,
        velocity_limit=VELOCITY,
        acceleration_limit=EXPERIMENTAL_ACCELERATION_LIMIT,
        collision_query=collision_query,
    )


def test_planning_robot_dict_adds_explicit_cspace_without_changing_asset():
    payload = planning_robot_dict(ASSET)
    cspace = payload["kinematics"]["cspace"]
    assert cspace["joint_names"] == list(JOINT_NAMES)
    assert cspace["retract_config"] == SAFE_Q.tolist()
    assert cspace["max_acceleration"] == [5.0] * 6
    assert cspace["max_jerk"] == [50.0] * 6


def test_npz_round_trip_uses_non_object_joint_names(tmp_path):
    path = tmp_path / "trajectory.npz"
    save_trajectory(path, valid_trajectory())
    with np.load(path, allow_pickle=False) as raw:
        assert raw["joint_names"].dtype.kind == "U"
    loaded = load_trajectory(path)
    assert loaded.joint_names == JOINT_NAMES
    assert validate(loaded)["passed"]


def test_wrong_time_step_is_rejected():
    trajectory = valid_trajectory()
    broken = TrajectoryData(
        trajectory.time_s * 2,
        trajectory.joint_position,
        trajectory.joint_velocity,
        trajectory.joint_acceleration,
        trajectory.joint_names,
    )
    result = validate(broken)
    assert not result["passed"]
    assert not result["checks"]["time_step"]


def test_nonfinite_value_is_rejected():
    trajectory = valid_trajectory()
    q = trajectory.joint_position.copy()
    q[2, 3] = np.nan
    broken = TrajectoryData(
        trajectory.time_s, q, trajectory.joint_velocity, trajectory.joint_acceleration, trajectory.joint_names
    )
    result = validate(broken)
    assert not result["passed"]
    assert not result["checks"]["finite"]


def test_goal_error_is_rejected():
    trajectory = valid_trajectory()
    q = trajectory.joint_position.copy()
    q[-1, 0] -= 0.02
    result = validate(
        TrajectoryData(
            trajectory.time_s, q, trajectory.joint_velocity, trajectory.joint_acceleration, trajectory.joint_names
        )
    )
    assert not result["passed"]
    assert not result["checks"]["goal"]


def test_collision_score_is_rejected():
    def collision_query(q):
        return {"world": np.array([0.0, 0.02]), "self": np.zeros(2)}

    result = validate(valid_trajectory(), collision_query=collision_query)
    assert not result["passed"]
    assert not result["checks"]["world_collision_free"]


def test_missing_collision_backend_is_rejected():
    trajectory = valid_trajectory()
    result = validate_trajectory(
        trajectory,
        start_q=SAFE_Q,
        goal_q=GOAL_Q,
        position_lower=LOWER,
        position_upper=UPPER,
        velocity_limit=VELOCITY,
        acceleration_limit=EXPERIMENTAL_ACCELERATION_LIMIT,
    )
    assert not result["passed"]
    assert not result["checks"]["world_collision_free"]
    assert not result["checks"]["self_collision_free"]


def test_scalar_collision_result_cannot_claim_all_samples_checked():
    def collision_query(q):
        return {"world": np.array(0.0), "self": np.array(0.0)}

    result = validate(valid_trajectory(), collision_query=collision_query)
    assert not result["passed"]
    assert not result["checks"]["collision_result_shapes"]


def test_extra_npz_field_is_rejected(tmp_path):
    trajectory = valid_trajectory()
    path = tmp_path / "trajectory.npz"
    save_trajectory(path, trajectory)
    np.savez(
        path,
        time_s=trajectory.time_s,
        joint_position=trajectory.joint_position,
        joint_velocity=trajectory.joint_velocity,
        joint_acceleration=trajectory.joint_acceleration,
        joint_names=np.asarray(JOINT_NAMES),
        extra=np.ones(1),
    )
    try:
        load_trajectory(path)
    except ValueError as error:
        assert "fields differ" in str(error)
    else:
        raise AssertionError("extra trajectory fields must be rejected")
