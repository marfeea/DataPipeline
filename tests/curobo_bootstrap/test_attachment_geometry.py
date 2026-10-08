import numpy as np
import pytest

from DataPipe.pipelines.curobo_bootstrap.attachment_geometry import attached_sphere_from_measured_target, root_vertices_in_base


def test_measured_attachment_tracks_object_motion_into_tcp():
    vertices = np.array([[-0.02, -0.01, -0.03], [0.02, 0.01, 0.03]])
    sphere = attached_sphere_from_measured_target(
        vertices,
        base_pose_w=[1, 2, 3, 1, 0, 0, 0],
        object_pose_w_at_reset=[1, 2, 3, 1, 0, 0, 0],
        object_pose_w_current=[1, 2, 3.4, 1, 0, 0, 0],
        tcp_pose_b_current=[0, 0, 0.4, 1, 0, 0, 0],
    )
    assert sphere.center_tcp == pytest.approx((0, 0, 0))
    assert sphere.radius_m == pytest.approx(np.linalg.norm([0.02, 0.01, 0.03]))


def test_measured_attachment_rejects_bad_quaternion():
    with pytest.raises(ValueError, match="approximately unit"):
        attached_sphere_from_measured_target(
            [[0, 0, 0], [1, 1, 1]],
            base_pose_w=[0, 0, 0, 2, 0, 0, 0],
            object_pose_w_at_reset=[0, 0, 0, 1, 0, 0, 0],
            object_pose_w_current=[0, 0, 0, 1, 0, 0, 0],
            tcp_pose_b_current=[0, 0, 0, 1, 0, 0, 0],
        )


@pytest.mark.parametrize("source", [[1.5, .2, .94, 1, 0, 0, 0], [.9, -.2, 1., 2**-.5, 0, 0, 2**-.5]])
def test_source_relocation_does_not_displace_carried_geometry(source):
    from DataPipe.pipelines.curobo_bootstrap.robot_adapter import pose_matrix
    from DataPipe.pipelines.curobo_bootstrap.replay import matrix_pose_wxyz
    local = np.array([[-.02, -.01, -.03], [.02, .01, .03]])
    base = [1.28, -.03, .816, 1, 0, 0, 0]
    vertices = root_vertices_in_base(local, root_pose_w=source, base_pose_w=base)
    lifted = np.array(source, dtype=float)
    lifted[2] += .4
    tcp = matrix_pose_wxyz(np.linalg.inv(pose_matrix(base[:3], base[3:])) @ pose_matrix(lifted[:3], lifted[3:]))
    sphere = attached_sphere_from_measured_target(vertices, base_pose_w=base, object_pose_w_at_reset=source, object_pose_w_current=lifted, tcp_pose_b_current=tcp)
    assert sphere.center_tcp == pytest.approx([0, 0, 0], abs=1e-12)
    assert sphere.radius_m == pytest.approx(np.linalg.norm([.02, .01, .03]))
