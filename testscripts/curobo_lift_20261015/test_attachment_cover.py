"""Geometric coverage check for the less conservative attachment proxy."""
import itertools
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from lift_experiment import attached_cover, line_axis_rotation


def test_four_spheres_cover_box_interior_and_all_cell_corners():
    identity = [0,0,0,1,0,0,0]
    vertices = np.array(list(itertools.product([-.01,.01],[-.015,.015],[-.08,.08])))
    spheres = attached_cover(vertices,base_pose_w=identity,object_pose_reset=identity,
                             object_pose_current=identity,tcp_pose_b=identity)
    # Corners of all four cells include surfaces between them, not just mesh vertices.
    points = np.array(list(itertools.product([-.01,.01],[-.015,.015],np.linspace(-.08,.08,9))))
    distance = np.min([np.linalg.norm(points-s.center_tcp,axis=1)-s.radius_m for s in spheres],axis=0)
    assert np.max(distance) <= 1e-12
    assert max(s.radius_m for s in spheres) < .03


def test_flat_object_uses_two_dimensional_grid():
    identity = [0,0,0,1,0,0,0]
    vertices = np.array(list(itertools.product([-.06,.06],[-.07,.07],[-.01,.01])))
    spheres = attached_cover(vertices,base_pose_w=identity,object_pose_reset=identity,
                             object_pose_current=identity,tcp_pose_b=identity)
    assert len({s.center_tcp[0] for s in spheres}) == 2
    assert len({s.center_tcp[1] for s in spheres}) == 2
    assert max(s.radius_m for s in spheres) < .048


def test_virtual_tool_axes_lock_the_original_diagonal_lift_line():
    start = np.array([-.2306,.2488,.1213])
    goal = np.array([-.237,.25,.5549])
    rotation = line_axis_rotation(start,goal,np.eye(3))
    np.testing.assert_allclose(rotation.T @ rotation,np.eye(3),atol=1e-12)
    assert np.isclose(np.linalg.det(rotation),1)
    projected = rotation.T @ (goal-start)
    np.testing.assert_allclose(projected[:2],[0,0],atol=1e-12)
    assert projected[2] > 0
