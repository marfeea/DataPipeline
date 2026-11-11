import numpy as np
import pytest

from configs.coordinate_frames import (
    FLANGE_TO_TCP_TRANSLATION_F,
    LEGACY_FLANGE_TO_TCP_TRANSLATION_F,
    LEGACY_TCP_TO_GRASP_CENTER_TRANSLATION_T,
    TCP_CALIBRATION_VERSION,
)
from configs.curobo.robot import DEFAULT_COLLISION_MODEL
from scripts.native.aubo_ik import forward_kinematics
from scripts.pipelines.curobo_bootstrap.collision_model import load_collision_model
from scripts.pipelines.curobo_bootstrap.robot_adapter import UrdfChain
from scripts.pipelines.curobo_bootstrap.task_targets import (
    PickPlaceV2Config,
    require_current_tcp_targets,
    task_snapshot,
)


def test_measured_grasp_frame_offset_is_applied_in_flange_coordinates():
    rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    np.testing.assert_allclose(np.array(LEGACY_FLANGE_TO_TCP_TRANSLATION_F)
                               + rotation @ LEGACY_TCP_TO_GRASP_CENTER_TRANSLATION_T,
                               FLANGE_TO_TCP_TRANSLATION_F, atol=1e-12, rtol=0)


def test_control_fk_and_planning_urdf_share_the_grasp_center_tcp():
    model = load_collision_model(DEFAULT_COLLISION_MODEL)
    chain = UrdfChain(model["urdf"], "AUBO_E5", "project_tcp")
    legacy = UrdfChain(DEFAULT_COLLISION_MODEL.with_name("aubo_e5_project.urdf"), "AUBO_E5", "project_tcp")
    offset = np.eye(4)
    offset[:3, 3] = LEGACY_TCP_TO_GRASP_CENTER_TRANSLATION_T
    for q in np.random.default_rng(7).uniform(-2, 2, (20, 6)):
        # URDF and the rounded DH chain have sub-micrometre fitting residuals.
        np.testing.assert_allclose(chain.fk(q), forward_kinematics(q), atol=1e-6, rtol=0)
        np.testing.assert_allclose(chain.fk(q), legacy.fk(q) @ offset, atol=1e-12, rtol=0)


def test_old_tcp_calibration_cannot_be_replayed_as_the_new_frame():
    task = task_snapshot(config=PickPlaceV2Config(), environment_origin_w=(0, 0, 0),
                         base_pose_w=(1.287, -0.034, 0.816, 1, 0, 0, 0))
    assert task["tcp_geometry"]["version"] == TCP_CALIBRATION_VERSION
    require_current_tcp_targets(task)
    task["tcp_geometry"]["translation_f"] = list(LEGACY_FLANGE_TO_TCP_TRANSLATION_F)
    with pytest.raises(ValueError, match="old TCP calibration"):
        require_current_tcp_targets(task)
