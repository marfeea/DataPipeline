import numpy as np
import pytest

from DataPipe.pipelines.curobo_bootstrap.robot_adapter import (
    UrdfChain,
    audit_fk,
    compare_joint_contract,
    pose_error,
    pose_matrix,
    rigid_transform,
)


def test_quaternion_sign_is_equivalent_but_bad_rotation_rejected():
    assert pose_error(pose_matrix((0, 0, 0), (1, 0, 0, 0)), pose_matrix((0, 0, 0), (-1, 0, 0, 0))) == (0, 0)
    with pytest.raises(ValueError):
        pose_matrix((0, 0, 0), (2, 0, 0, 0))
    with pytest.raises(ValueError):
        rigid_transform(np.diag([-1, 1, 1, 1]))


def test_missing_curobo_never_passes_even_with_identical_other_fk():
    result = audit_fk(
        [[0] * 6],
        {"current": lambda q: np.eye(4), "isaac": lambda q: np.eye(4)},
        position_tolerance_m=1e-5,
        rotation_tolerance_rad=1e-5,
    )
    assert not result["passed"]
    assert result["missing_backends"] == ["curobo"]


def test_three_backends_and_mismatch_gate():
    callbacks = dict.fromkeys(("current", "isaac", "curobo"), lambda q: np.eye(4))
    assert audit_fk([[0] * 6], callbacks, position_tolerance_m=1e-5, rotation_tolerance_rad=1e-5)["passed"]
    callbacks["curobo"] = lambda q: pose_matrix((0.01, 0, 0), (1, 0, 0, 0))
    result = audit_fk([[0] * 6], callbacks, position_tolerance_m=1e-5, rotation_tolerance_rad=1e-5)
    assert not result["passed"]
    assert result["comparisons"]["isaac:curobo"]["position_max_m"] == 0.01


def test_urdf_origin_rotation_and_joint_axis(tmp_path):
    path = tmp_path / "robot.urdf"
    path.write_text("""<robot name="test"><link name="base"/><link name="tip"/>
      <joint name="j" type="revolute"><parent link="base"/><child link="tip"/>
      <origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/><axis xyz="0 0 1"/>
      <limit lower="-3" upper="3" velocity="2"/></joint></robot>""")
    chain = UrdfChain(path, "base", "tip")
    result = chain.fk([np.pi / 2])
    np.testing.assert_allclose(result[:3, :3], np.diag([-1, -1, 1]), atol=1e-10)
    np.testing.assert_equal(result[:3, 3], [1, 2, 3])
    assert chain.joint_names == ("j",)
    with pytest.raises(ValueError):
        chain.fk([0, 0])
    with pytest.raises(ValueError):
        UrdfChain(path, "tip", "base")


@pytest.mark.parametrize("field", ["joint_names", "lower", "upper", "velocity"])
def test_joint_contract_differences_are_not_silently_corrected(field):
    names = tuple(f"j{i}" for i in range(6))
    other = {"joint_names": names, "lower": [-1] * 6, "upper": [1] * 6, "velocity": [2] * 6}
    other[field] = names[::-1] if field == "joint_names" else [0] * 6
    assert compare_joint_contract(names, [-1] * 6, [1] * 6, [2] * 6, other)
